# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from hagisilta_i18n import tr

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import cv2
from torch import nn


TEMPORAL_SCHEMA = "hagisilta-temporal-refiner-v1"
VISUAL_TEMPORAL_SCHEMA = "hagisilta-visual-temporal-refiner-v2"
DEFAULT_OFFSETS_SECONDS = (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3)
DEFAULT_VISUAL_SIZE = 96
DEFAULT_MAX_RESIDUAL_RATIO = 0.035
FINE_NOISE_FLOOR_PX = 2.0
STRUCTURAL_ERROR_THRESHOLD_PX = 8.0


def temporal_refiner_root(project: Path, shuffle: int) -> Path:
    return project / ".hagisilta" / "temporal-refiner" / f"shuffle_{int(shuffle):02d}"


def temporal_checkpoint_path(project: Path, shuffle: int) -> Path:
    return temporal_refiner_root(project, shuffle) / "temporal-refiner.pt"


def read_pose_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".h5", ".hdf5"}:
        frame = pd.read_hdf(path)
    else:
        frame = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    if not isinstance(frame.columns, pd.MultiIndex) or frame.columns.nlevels < 3:
        raise ValueError(tr('DLC 좌표 열 형식이 아닙니다: {v0}', v0=path))
    return frame


def _pose_column(frame: pd.DataFrame, bodypart: str, coord: str):
    bodypart_level = frame.columns.nlevels - 2
    coord_level = frame.columns.nlevels - 1
    return next(
        (
            column
            for column in frame.columns
            if str(column[bodypart_level]) == bodypart
            and str(column[coord_level]) == coord
        ),
        None,
    )


def pose_bodyparts(frame: pd.DataFrame) -> list[str]:
    level = frame.columns.nlevels - 2
    return list(dict.fromkeys(map(str, frame.columns.get_level_values(level))))


def pose_array(frame: pd.DataFrame, bodyparts: list[str]) -> np.ndarray:
    result = np.full((len(frame), len(bodyparts), 3), np.nan, dtype=np.float32)
    for keypoint, bodypart in enumerate(bodyparts):
        for channel, coord in enumerate(("x", "y", "likelihood")):
            column = _pose_column(frame, bodypart, coord)
            if column is not None:
                result[:, keypoint, channel] = pd.to_numeric(
                    frame[column], errors="coerce"
                ).to_numpy(dtype=np.float32)
    result[:, :, 2] = np.nan_to_num(result[:, :, 2], nan=0.0, posinf=0.0, neginf=0.0)
    return result


def normalize_pose(values: np.ndarray, width: int, height: int) -> np.ndarray:
    normalized = values.astype(np.float32, copy=True)
    normalized[:, :, 0] /= max(1, int(width) - 1)
    normalized[:, :, 1] /= max(1, int(height) - 1)
    normalized[:, :, :2] = np.nan_to_num(
        normalized[:, :, :2], nan=0.0, posinf=1.0, neginf=0.0
    )
    normalized[:, :, 2] = np.clip(normalized[:, :, 2], 0.0, 1.0)
    return normalized


def context_indices(
    center_frames: np.ndarray,
    frame_count: int,
    fps: float,
    offsets_seconds: list[float] | tuple[float, ...],
) -> np.ndarray:
    centers = np.asarray(center_frames, dtype=np.int64).reshape(-1, 1)
    offsets = np.asarray(
        [round(float(offset) * float(fps)) for offset in offsets_seconds],
        dtype=np.int64,
    ).reshape(1, -1)
    return np.clip(centers + offsets, 0, max(0, int(frame_count) - 1))


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1),
        )
        self.activation = nn.GELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(value + self.block(value))


class TemporalPoseRefiner(nn.Module):
    """Small time-normalized TCN that predicts a residual for the center pose."""

    def __init__(
        self,
        keypoints: int,
        hidden_channels: int = 96,
        max_residual_ratio: float = DEFAULT_MAX_RESIDUAL_RATIO,
    ) -> None:
        super().__init__()
        self.keypoints = int(keypoints)
        self.hidden_channels = int(hidden_channels)
        self.max_residual_ratio = float(max_residual_ratio)
        input_channels = self.keypoints * 3
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, 1),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(
            TemporalResidualBlock(hidden_channels, 1),
            TemporalResidualBlock(hidden_channels, 2),
            TemporalResidualBlock(hidden_channels, 4),
        )
        self.output = nn.Linear(hidden_channels + input_channels, self.keypoints * 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        # sequence: batch, time, keypoint, (x, y, likelihood)
        batch, time, keypoints, channels = sequence.shape
        if keypoints != self.keypoints or channels != 3:
            raise ValueError(tr('시간 문맥 입력의 신체점 또는 채널 수가 모델과 다릅니다.'))
        flattened = sequence.reshape(batch, time, keypoints * channels).transpose(1, 2)
        encoded = self.temporal(self.input_projection(flattened))
        center = time // 2
        fused = torch.cat((encoded[:, :, center], flattened[:, :, center]), dim=1)
        return torch.tanh(self.output(fused)).reshape(batch, keypoints, 2) * self.max_residual_ratio


class VisualTemporalPoseRefiner(nn.Module):
    """TCN using DLC coordinates plus learned video feature/pose heatmaps."""

    def __init__(
        self,
        keypoints: int,
        hidden_channels: int = 96,
        max_residual_ratio: float = DEFAULT_MAX_RESIDUAL_RATIO,
        visual_size: int = DEFAULT_VISUAL_SIZE,
    ) -> None:
        super().__init__()
        self.keypoints = int(keypoints)
        self.hidden_channels = int(hidden_channels)
        self.max_residual_ratio = float(max_residual_ratio)
        self.visual_size = int(visual_size)
        input_channels = self.keypoints * 3
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 8, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.evidence_encoder = nn.Sequential(
            nn.Conv2d(16 + self.keypoints, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((6, 6)),
        )
        self.evidence_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 6 * 6, hidden_channels),
            nn.GELU(),
        )
        self.pose_projection = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, 1),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(
            TemporalResidualBlock(hidden_channels * 2, 1),
            TemporalResidualBlock(hidden_channels * 2, 2),
            TemporalResidualBlock(hidden_channels * 2, 4),
        )
        self.output = nn.Linear(hidden_channels * 2 + input_channels, self.keypoints * 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def pose_heatmaps(self, pose: torch.Tensor, height: int, width: int) -> torch.Tensor:
        y_grid = torch.linspace(0.0, 1.0, height, device=pose.device, dtype=pose.dtype)
        x_grid = torch.linspace(0.0, 1.0, width, device=pose.device, dtype=pose.dtype)
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")
        x = pose[:, :, 0, None, None]
        y = pose[:, :, 1, None, None]
        likelihood = pose[:, :, 2, None, None]
        sigma = 0.06
        return torch.exp(-((xx - x).square() + (yy - y).square()) / (2 * sigma * sigma)) * likelihood

    def forward(self, sequence: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        # sequence: B,T,K,(x,y,p), video: B,T,1,H,W (uint8 or float 0..1)
        batch, time, keypoints, channels = sequence.shape
        if keypoints != self.keypoints or channels != 3:
            raise ValueError(tr('시간 문맥 좌표 입력이 모델 구성과 다릅니다.'))
        if video.ndim != 5 or video.shape[:2] != (batch, time):
            raise ValueError(tr('시간 문맥 영상 입력이 좌표 시퀀스와 다릅니다.'))
        flattened = sequence.reshape(batch, time, keypoints * channels).transpose(1, 2)
        frames = video.reshape(batch * time, 1, video.shape[-2], video.shape[-1]).float()
        if video.dtype == torch.uint8:
            frames = frames / 255.0
        image_features = self.image_encoder(frames)
        poses = sequence.reshape(batch * time, keypoints, channels)
        heatmaps = self.pose_heatmaps(poses, image_features.shape[-2], image_features.shape[-1])
        evidence = self.evidence_projection(
            self.evidence_encoder(torch.cat((image_features, heatmaps), dim=1))
        ).reshape(batch, time, -1).transpose(1, 2)
        encoded = self.temporal(torch.cat((self.pose_projection(flattened), evidence), dim=1))
        center = time // 2
        fused = torch.cat((encoded[:, :, center], flattened[:, :, center]), dim=1)
        return torch.tanh(self.output(fused)).reshape(batch, keypoints, 2) * self.max_residual_ratio


@dataclass
class TemporalTrainingArrays:
    features: np.ndarray
    visuals: np.ndarray | None
    residuals: np.ndarray
    masks: np.ndarray
    point_weights: np.ndarray
    structural_masks: np.ndarray
    scales: np.ndarray
    roles: np.ndarray
    ids: list[str]


def read_video_evidence(
    path: Path,
    requested_indices: np.ndarray | list[int],
    visual_size: int = DEFAULT_VISUAL_SIZE,
) -> dict[int, np.ndarray]:
    requested = sorted(set(map(int, np.asarray(requested_indices).reshape(-1))))
    if not requested:
        return {}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(tr('시간 문맥 영상 파일을 열 수 없습니다: {v0}', v0=path))
    targets = set(requested)
    last = requested[-1]
    found: dict[int, np.ndarray] = {}
    index = 0
    while index <= last:
        ok, frame = capture.read()
        if not ok:
            break
        if index in targets:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(
                gray, (int(visual_size), int(visual_size)), interpolation=cv2.INTER_AREA
            )
            found[index] = resized[None, :, :].astype(np.uint8, copy=False)
        index += 1
    capture.release()
    return found


def labels_target(
    labels: dict[str, dict[str, float | None]],
    bodyparts: list[str],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.zeros((len(bodyparts), 2), dtype=np.float32)
    mask = np.zeros((len(bodyparts), 2), dtype=np.float32)
    for index, bodypart in enumerate(bodyparts):
        values = labels.get(bodypart) or {}
        for channel, (coord, scale) in enumerate(
            (("x", max(1, width - 1)), ("y", max(1, height - 1)))
        ):
            value = values.get(coord)
            if value is not None and math.isfinite(float(value)):
                target[index, channel] = float(value) / scale
                mask[index, channel] = 1.0
    return target, mask


def build_training_arrays(
    manifest: dict[str, Any],
    pose_by_video: dict[str, Path],
    video_geometry: dict[str, tuple[int, int]],
    bodyparts: list[str],
    visual_size: int = DEFAULT_VISUAL_SIZE,
) -> tuple[TemporalTrainingArrays, list[str]]:
    pose_cache: dict[str, np.ndarray] = {}
    features: list[np.ndarray] = []
    visuals: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    point_weights: list[np.ndarray] = []
    structural_masks: list[np.ndarray] = []
    scales: list[tuple[float, float]] = []
    roles: list[str] = []
    ids: list[str] = []
    warnings: list[str] = []
    offsets = list(map(float, manifest.get("requested_offsets_seconds") or DEFAULT_OFFSETS_SECONDS))

    prepared: list[tuple[dict[str, Any], str, np.ndarray, np.ndarray, tuple[int, int]]] = []
    indices_by_video: dict[str, list[int]] = {}
    for entry in manifest.get("entries", []):
        video = str(entry.get("inference_video") or "")
        pose_path = pose_by_video.get(video)
        geometry = video_geometry.get(video)
        if pose_path is None or geometry is None:
            continue
        if video not in pose_cache:
            table = read_pose_table(pose_path)
            missing = sorted(set(bodyparts) - set(pose_bodyparts(table)))
            if missing:
                warnings.append(tr('{v0}: 예측에 없는 신체점 {v1}', v0=Path(video).name, v1=', '.join(missing)))
                pose_cache[video] = np.empty((0, len(bodyparts), 3), dtype=np.float32)
            else:
                pose_cache[video] = pose_array(table, bodyparts)
        pose = pose_cache[video]
        if len(pose) == 0:
            continue
        center = int(entry.get("center_frame", -1))
        if not 0 <= center < len(pose):
            warnings.append(tr('{v0}: 예측 범위를 벗어난 GT 프레임', v0=entry.get('id')))
            continue
        width, height = geometry
        normalized = normalize_pose(pose, width, height)
        indices = context_indices(
            np.asarray([center]), len(normalized), float(entry.get("fps") or 0.0), offsets
        )[0]
        target, mask = labels_target(entry.get("labels") or {}, bodyparts, width, height)
        if not mask.any():
            continue
        center_prediction = normalized[center, :, :2]
        prepared.append((entry, video, normalized[indices], indices, (width, height)))
        indices_by_video.setdefault(video, []).extend(map(int, indices))
        residual = target - center_prediction
        point_mask = mask.all(axis=1)
        pixel_scale = np.asarray([max(1, width - 1), max(1, height - 1)], dtype=np.float32)
        distance_px = np.linalg.norm(residual * pixel_scale, axis=1)
        prediction_px = center_prediction * pixel_scale
        outside = (
            (prediction_px[:, 0] < 0.0)
            | (prediction_px[:, 0] > width - 1)
            | (prediction_px[:, 1] < 0.0)
            | (prediction_px[:, 1] > height - 1)
        )
        structural = point_mask & (outside | (distance_px > STRUCTURAL_ERROR_THRESHOLD_PX))
        fine_mask = mask.copy()
        fine_mask[structural, :] = 0.0
        weights = np.where(distance_px < FINE_NOISE_FLOOR_PX, 0.05, 1.0).astype(np.float32)
        weights[~point_mask | structural] = 0.0
        residuals.append(residual)
        masks.append(fine_mask)
        point_weights.append(np.repeat(weights[:, None], 2, axis=1))
        structural_masks.append(structural)
        scales.append((max(1, width - 1), max(1, height - 1)))
        roles.append(str(entry.get("role") or "train"))
        ids.append(str(entry.get("id") or len(ids)))

    evidence_by_video: dict[str, dict[int, np.ndarray]] = {}
    evidence_items = list(indices_by_video.items())
    for evidence_index, (video, requested) in enumerate(evidence_items, start=1):
        try:
            print(
                f"TEMPORAL_VIDEO_EVIDENCE={evidence_index}/{len(evidence_items)}:{Path(video).name}",
                flush=True,
            )
            evidence_by_video[video] = read_video_evidence(Path(video), requested, visual_size)
        except RuntimeError as error:
            warnings.append(str(error))
            evidence_by_video[video] = {}
    kept: list[int] = []
    for sample_index, (_entry, video, feature, indices, _geometry) in enumerate(prepared):
        evidence = evidence_by_video.get(video, {})
        if any(int(index) not in evidence for index in indices):
            warnings.append(tr('{v0}: 영상 문맥 프레임을 읽지 못해 제외됨', v0=ids[sample_index]))
            continue
        features.append(feature)
        visuals.append(np.stack([evidence[int(index)] for index in indices]))
        kept.append(sample_index)

    if not features:
        raise RuntimeError(tr('시간 문맥 학습에 연결된 DLC 예측과 GT가 없습니다.'))
    return (
        TemporalTrainingArrays(
            features=np.stack(features),
            visuals=np.stack(visuals),
            residuals=np.stack([residuals[index] for index in kept]),
            masks=np.stack([masks[index] for index in kept]),
            point_weights=np.stack([point_weights[index] for index in kept]),
            structural_masks=np.stack([structural_masks[index] for index in kept]),
            scales=np.asarray([scales[index] for index in kept], dtype=np.float32),
            roles=np.asarray([roles[index] for index in kept]),
            ids=[ids[index] for index in kept],
        ),
        sorted(set(warnings)),
    )


def masked_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    point_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = nn.functional.smooth_l1_loss(prediction, target, reduction="none", beta=0.02)
    effective_mask = mask
    if point_weights is not None:
        effective_mask = effective_mask * point_weights
    return (loss * effective_mask).sum() / effective_mask.sum().clamp_min(1.0)


def evaluate_arrays(
    model: TemporalPoseRefiner | VisualTemporalPoseRefiner,
    arrays: TemporalTrainingArrays,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, float | int]:
    if len(indices) == 0:
        return {"samples": 0, "baseline_error_px": 0.0, "refined_error_px": 0.0}
    baseline_errors: list[np.ndarray] = []
    refined_errors: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            feature = torch.from_numpy(arrays.features[selected]).to(device)
            if isinstance(model, VisualTemporalPoseRefiner):
                if arrays.visuals is None:
                    raise RuntimeError(tr('영상 근거 학습 배열이 없습니다.'))
                video = torch.from_numpy(arrays.visuals[selected]).to(device)
                predicted_residual = model(feature, video).cpu().numpy()
            else:
                predicted_residual = model(feature).cpu().numpy()
            target_residual = arrays.residuals[selected]
            reachable = (
                np.max(np.abs(target_residual), axis=2)
                <= float(model.max_residual_ratio)
            )
            point_mask = arrays.masks[selected].all(axis=2) & reachable
            scale = arrays.scales[selected, None, :]
            baseline_distance = np.linalg.norm(target_residual * scale, axis=2)
            refined_distance = np.linalg.norm(
                (target_residual - predicted_residual) * scale, axis=2
            )
            baseline_errors.append(baseline_distance[point_mask])
            refined_errors.append(refined_distance[point_mask])
    baseline = np.concatenate(baseline_errors) if baseline_errors else np.asarray([])
    refined = np.concatenate(refined_errors) if refined_errors else np.asarray([])
    hard = baseline > 2.0
    severe = baseline > 5.0
    stable = ~hard
    return {
        "samples": int(len(indices)),
        "labeled_points": int(len(baseline)),
        "baseline_error_px": float(baseline.mean()) if len(baseline) else 0.0,
        "refined_error_px": float(refined.mean()) if len(refined) else 0.0,
        "hard_labeled_points": int(np.count_nonzero(hard)),
        "baseline_hard_error_px": float(baseline[hard].mean()) if hard.any() else 0.0,
        "refined_hard_error_px": float(refined[hard].mean()) if hard.any() else 0.0,
        "severe_labeled_points": int(np.count_nonzero(severe)),
        "baseline_severe_error_px": float(baseline[severe].mean()) if severe.any() else 0.0,
        "refined_severe_error_px": float(refined[severe].mean()) if severe.any() else 0.0,
        "stable_labeled_points": int(np.count_nonzero(stable)),
        "baseline_stable_error_px": float(baseline[stable].mean()) if stable.any() else 0.0,
        "refined_stable_error_px": float(refined[stable].mean()) if stable.any() else 0.0,
        "structural_positive_points": int(np.count_nonzero(arrays.structural_masks[indices])),
        "unreachable_fine_points": int(
            np.count_nonzero(
                arrays.masks[indices].all(axis=2)
                & (np.max(np.abs(arrays.residuals[indices]), axis=2) > float(model.max_residual_ratio))
            )
        ),
    }


def save_checkpoint(
    path: Path,
    model: TemporalPoseRefiner | VisualTemporalPoseRefiner,
    bodyparts: list[str],
    offsets_seconds: list[float],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": VISUAL_TEMPORAL_SCHEMA if isinstance(model, VisualTemporalPoseRefiner) else TEMPORAL_SCHEMA,
            "bodyparts": bodyparts,
            "offsets_seconds": offsets_seconds,
            "hidden_channels": model.hidden_channels,
            "max_residual_ratio": model.max_residual_ratio,
            "visual_size": getattr(model, "visual_size", None),
            "state_dict": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(
    path: Path, device: str | torch.device = "cpu"
) -> tuple[TemporalPoseRefiner | VisualTemporalPoseRefiner, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") not in {TEMPORAL_SCHEMA, VISUAL_TEMPORAL_SCHEMA}:
        raise ValueError(tr('지원하지 않는 시간 문맥 보정 모델입니다.'))
    bodyparts = list(map(str, payload.get("bodyparts") or []))
    model_kwargs: dict[str, Any] = {
        "hidden_channels": int(payload.get("hidden_channels") or 96),
        "max_residual_ratio": float(payload.get("max_residual_ratio") or 0.25),
    }
    if payload.get("schema") == VISUAL_TEMPORAL_SCHEMA:
        model_kwargs["visual_size"] = int(payload.get("visual_size") or DEFAULT_VISUAL_SIZE)
        model = VisualTemporalPoseRefiner(len(bodyparts), **model_kwargs)
    else:
        model = TemporalPoseRefiner(len(bodyparts), **model_kwargs)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


def _robust_threshold(values: np.ndarray, floor: float, multiplier: float = 8.0) -> float:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float(floor)
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median))) * 1.4826
    return max(float(floor), median + multiplier * max(mad, 0.25))


def roi_valid_bounds(
    width: int,
    height: int,
    metadata_path: Path | None = None,
) -> tuple[float, float, float, float]:
    """Return inclusive x/y bounds for real image pixels (padding is invalid)."""
    if metadata_path is None or not Path(metadata_path).is_file():
        return 0.0, float(width - 1), 0.0, float(height - 1)
    try:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        padding = metadata.get("padding_px") or {}
        left = int(padding.get("left") or 0)
        top = int(padding.get("top") or 0)
        source_size = metadata.get("source_crop_size_px") or [width, height]
        source_width, source_height = map(int, source_size[:2])
        return (
            float(left),
            float(min(width - 1, left + source_width - 1)),
            float(top),
            float(min(height - 1, top + source_height - 1)),
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0.0, float(width - 1), 0.0, float(height - 1)


def recover_structural_pose(
    values: np.ndarray,
    fps: float,
    valid_bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect name-independent structural failures and recover only with temporal evidence."""
    recovered = values.astype(np.float32, copy=True)
    positions = recovered[:, :, :2]
    likelihood = recovered[:, :, 2]
    frame_count, keypoints, _ = positions.shape
    x0, x1, y0, y1 = valid_bounds
    finite = np.isfinite(positions).all(axis=2)
    inside = (
        finite
        & (positions[:, :, 0] >= x0)
        & (positions[:, :, 0] <= x1)
        & (positions[:, :, 1] >= y0)
        & (positions[:, :, 1] <= y1)
    )
    reliable = inside & (likelihood >= 0.60)
    anomaly = ~inside
    reason_invalid = ~inside
    reason_motion = np.zeros_like(anomaly)
    reason_geometry = np.zeros_like(anomaly)
    reason_spike = np.zeros_like(anomaly)

    if frame_count >= 2:
        delta = positions[1:] - positions[:-1]
        delta_valid = reliable[1:] & reliable[:-1]
        masked_delta = np.where(delta_valid[:, :, None], delta, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            common = np.nanmedian(masked_delta, axis=1)
        deviation = np.linalg.norm(delta - common[:, None, :], axis=2)
        motion_threshold = _robust_threshold(deviation[delta_valid], STRUCTURAL_ERROR_THRESHOLD_PX)
        motion_bad = delta_valid & (deviation > motion_threshold)
        reason_motion[1:] |= motion_bad

    if keypoints >= 3 and frame_count:
        pair_votes = np.zeros((frame_count, keypoints), dtype=np.int16)
        for first in range(keypoints - 1):
            for second in range(first + 1, keypoints):
                pair_valid = reliable[:, first] & reliable[:, second]
                distance = np.linalg.norm(positions[:, first] - positions[:, second], axis=1)
                threshold = _robust_threshold(
                    np.abs(distance[pair_valid] - np.median(distance[pair_valid]))
                    if pair_valid.any() else np.asarray([]),
                    STRUCTURAL_ERROR_THRESHOLD_PX,
                )
                if pair_valid.any():
                    center = float(np.median(distance[pair_valid]))
                    bad = pair_valid & (np.abs(distance - center) > threshold)
                    pair_votes[bad, first] += 1
                    pair_votes[bad, second] += 1
        reason_geometry |= pair_votes >= max(2, keypoints - 2)

    if frame_count >= 3:
        neighbor_valid = reliable[:-2] & reliable[2:]
        midpoint = (positions[:-2] + positions[2:]) * 0.5
        spike_distance = np.linalg.norm(positions[1:-1] - midpoint, axis=2)
        neighbor_motion = np.linalg.norm(positions[2:] - positions[:-2], axis=2)
        spike_valid = neighbor_valid & reliable[1:-1]
        spike_threshold = _robust_threshold(
            spike_distance[spike_valid], STRUCTURAL_ERROR_THRESHOLD_PX
        )
        spike_bad = (
            spike_valid
            & (spike_distance > spike_threshold)
            & (spike_distance > 2.0 * neighbor_motion + 2.0)
        )
        reason_spike[1:-1] |= spike_bad
        coordinated_spike = np.count_nonzero(reason_spike, axis=1) >= 2
        reason_spike &= coordinated_spike[:, None]

    maximum_gap = max(2, int(round(max(float(fps), 1.0) * 0.5)))
    # Pair-distance and common-motion signals are retained as review/training
    # candidates. Coordinates are changed automatically only for invalid points
    # or an unambiguous one-frame leave/return spike.
    anomaly |= reason_spike
    resolved = np.zeros_like(anomaly)
    unresolved = np.zeros_like(anomaly)
    for keypoint in range(keypoints):
        frame = 0
        while frame < frame_count:
            if not anomaly[frame, keypoint]:
                frame += 1
                continue
            start = frame
            while frame + 1 < frame_count and anomaly[frame + 1, keypoint]:
                frame += 1
            end = frame
            before = start - 1
            after = end + 1
            before_ok = before >= 0 and reliable[before, keypoint] and not anomaly[before, keypoint]
            after_ok = after < frame_count and reliable[after, keypoint] and not anomaly[after, keypoint]
            outside_depth = max(
                float(np.nanmax(x0 - positions[start : end + 1, keypoint, 0])),
                float(np.nanmax(positions[start : end + 1, keypoint, 0] - x1)),
                float(np.nanmax(y0 - positions[start : end + 1, keypoint, 1])),
                float(np.nanmax(positions[start : end + 1, keypoint, 1] - y1)),
                0.0,
            )
            strong_invalid = (
                not reason_invalid[start : end + 1, keypoint].any()
                or outside_depth > 2.0
            )
            endpoints_close = (
                before_ok
                and after_ok
                and float(np.linalg.norm(positions[after, keypoint] - positions[before, keypoint]))
                <= 4.0 * STRUCTURAL_ERROR_THRESHOLD_PX
            )
            if (
                before_ok
                and after_ok
                and strong_invalid
                and endpoints_close
                and start - before <= maximum_gap
                and after - end <= maximum_gap
            ):
                span = after - before
                for current in range(start, end + 1):
                    alpha = (current - before) / span
                    positions[current, keypoint] = (
                        (1.0 - alpha) * positions[before, keypoint]
                        + alpha * positions[after, keypoint]
                    )
                    likelihood[current, keypoint] = min(
                        likelihood[before, keypoint], likelihood[after, keypoint]
                    )
                    resolved[current, keypoint] = True
            else:
                positions[start : end + 1, keypoint] = np.nan
                likelihood[start : end + 1, keypoint] = 0.0
                unresolved[start : end + 1, keypoint] = True
            frame += 1
    return recovered, {
        "structural_candidates": int(np.count_nonzero(anomaly)),
        "outside_candidates": int(np.count_nonzero(reason_invalid)),
        "motion_candidates": int(np.count_nonzero(reason_motion)),
        "geometry_candidates": int(np.count_nonzero(reason_geometry)),
        "leave_return_candidates": int(np.count_nonzero(reason_spike)),
        "recovered_points": int(np.count_nonzero(resolved)),
        "unresolved_points": int(np.count_nonzero(unresolved)),
    }


def refine_pose_table(
    frame: pd.DataFrame,
    checkpoint: Path,
    fps: float,
    width: int,
    height: int,
    device: str = "cpu",
    batch_size: int = 512,
    video_path: Path | None = None,
    roi_metadata_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    torch_device = torch.device(device)
    model, payload = load_checkpoint(checkpoint, torch_device)
    bodyparts = list(map(str, payload["bodyparts"]))
    missing = sorted(set(bodyparts) - set(pose_bodyparts(frame)))
    if missing:
        raise ValueError(tr('시간 문맥 모델과 DLC 결과의 신체점이 다릅니다: ') + ", ".join(missing))
    values = pose_array(frame, bodyparts)
    recovered_values, structural_report = recover_structural_pose(
        values,
        fps,
        roi_valid_bounds(width, height, roi_metadata_path),
    )
    normalized = normalize_pose(recovered_values, width, height)
    offsets = list(map(float, payload.get("offsets_seconds") or DEFAULT_OFFSETS_SECONDS))
    centers = np.arange(len(frame), dtype=np.int64)
    indices = context_indices(centers, len(frame), fps, offsets)
    visuals: np.ndarray | None = None
    if isinstance(model, VisualTemporalPoseRefiner):
        if video_path is None:
            raise ValueError(tr('영상 근거 시간 보정에는 추론 ROI 영상이 필요합니다.'))
        evidence = read_video_evidence(Path(video_path), centers, model.visual_size)
        if len(evidence) < len(frame):
            raise RuntimeError(
                tr('영상 근거 프레임이 부족합니다: 좌표 {v0}개 · 영상 {v1}개', v0=len(frame), v1=len(evidence))
            )
        visuals = np.stack([evidence[index] for index in range(len(frame))])
    corrected = normalized[:, :, :2].copy()
    with torch.no_grad():
        for start in range(0, len(frame), max(1, int(batch_size))):
            selected = indices[start : start + batch_size]
            feature = torch.from_numpy(normalized[selected]).to(torch_device)
            if isinstance(model, VisualTemporalPoseRefiner):
                assert visuals is not None
                video = torch.from_numpy(visuals[selected]).to(torch_device)
                residual = model(feature, video).cpu().numpy()
            else:
                residual = model(feature).cpu().numpy()
            corrected[start : start + len(residual)] += residual
    x0, x1, y0, y1 = roi_valid_bounds(width, height, roi_metadata_path)
    outside_after_fine = (
        (corrected[:, :, 0] < x0 / max(1, width - 1))
        | (corrected[:, :, 0] > x1 / max(1, width - 1))
        | (corrected[:, :, 1] < y0 / max(1, height - 1))
        | (corrected[:, :, 1] > y1 / max(1, height - 1))
    )
    if outside_after_fine.any():
        recovered_values[:, :, 2][outside_after_fine] = 0.0
    unresolved = ~np.isfinite(recovered_values[:, :, :2]).all(axis=2)
    unresolved |= outside_after_fine
    corrected[unresolved] = np.nan
    structural_report["fine_outside_unresolved"] = int(np.count_nonzero(outside_after_fine))
    structural_report["unresolved_points"] += int(np.count_nonzero(outside_after_fine))
    result = frame.copy()
    for keypoint, bodypart in enumerate(bodyparts):
        x_column = _pose_column(result, bodypart, "x")
        y_column = _pose_column(result, bodypart, "y")
        if x_column is not None:
            result[x_column] = corrected[:, keypoint, 0] * max(1, width - 1)
        if y_column is not None:
            result[y_column] = corrected[:, keypoint, 1] * max(1, height - 1)
        likelihood_column = _pose_column(result, bodypart, "likelihood")
        if likelihood_column is not None:
            result[likelihood_column] = recovered_values[:, keypoint, 2]
    report = {
        "schema": str(payload.get("schema") or TEMPORAL_SCHEMA),
        "input_evidence": "video-feature-pose-heatmap" if isinstance(model, VisualTemporalPoseRefiner) else "coordinates-only",
        "checkpoint": str(checkpoint),
        "frames": len(frame),
        "fps": float(fps),
        "offsets_seconds": offsets,
        "bodyparts": bodyparts,
        "structural_recovery": structural_report,
        "mean_shift_px": float(
            np.nanmean(np.linalg.norm(
                (corrected - normalized[:, :, :2])
                * np.asarray([max(1, width - 1), max(1, height - 1)]),
                axis=2,
            ))
        ),
    }
    return result, report


def write_pose_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        frame.to_hdf(path, key="df_with_missing", mode="w")
    else:
        frame.to_csv(path)


def write_refinement_report(report: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
