# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from hagisilta_i18n import tr

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


RESULT_RENDER_DPI = 240
RESULT_RENDER_WIDTH_IN = 16.0


def unlocked_version_path(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
    counter = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{stamp}_{counter}{path.suffix}")
        counter += 1
    return candidate


def write_csv_resilient(frame: pd.DataFrame, path: Path, **kwargs) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_csv(path, **kwargs)
        return path
    except PermissionError:
        alternate = unlocked_version_path(path)
        frame.to_csv(alternate, **kwargs)
        print(f"OUTPUT_LOCKED={path.resolve()}", flush=True)
        print(f"OUTPUT_VERSIONED={alternate.resolve()}", flush=True)
        return alternate


def write_text_resilient(path: Path, text: str, **kwargs) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(text, **kwargs)
        return path
    except PermissionError:
        alternate = unlocked_version_path(path)
        alternate.write_text(text, **kwargs)
        print(f"OUTPUT_LOCKED={path.resolve()}", flush=True)
        print(f"OUTPUT_VERSIONED={alternate.resolve()}", flush=True)
        return alternate


def save_figure_resilient(figure, path: Path, **kwargs) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(path, **kwargs)
        return path
    except PermissionError:
        alternate = unlocked_version_path(path)
        figure.savefig(alternate, **kwargs)
        print(f"OUTPUT_LOCKED={path.resolve()}", flush=True)
        print(f"OUTPUT_VERSIONED={alternate.resolve()}", flush=True)
        return alternate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate NORT exploration from DLC coordinates")
    parser.add_argument("--subject-root", type=Path)
    parser.add_argument("--setup-root", type=Path)
    parser.add_argument("--experiment-info", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-plan", type=Path)
    parser.add_argument("--pcutoff", type=float, default=0.6)
    parser.add_argument("--orientation", action="store_true")
    parser.add_argument("--minimum-exploration", type=float, default=0.2)
    parser.add_argument("--bridge-gap", type=float, default=0.1)
    parser.add_argument("--nose-bodypart", default="nose")
    parser.add_argument("--head-bodypart", default="head_center")
    parser.add_argument("--body-bodypart", default="body_center")
    parser.add_argument("--rear-bodypart", default="rear_center")
    return parser.parse_args()


def arena_id(path: Path) -> int | None:
    suffix = path.name.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else None


def load_pose(arena_dir: Path) -> tuple[pd.DataFrame, Path]:
    from hagisilta_pose_postprocess import load_analysis_pose
    selected = load_analysis_pose(arena_dir)
    if selected is not None:
        return selected
    h5_files = sorted(arena_dir.glob("*.h5"))
    if h5_files:
        return pd.read_hdf(h5_files[0]), h5_files[0]
    csv_files = [path for path in sorted(arena_dir.glob("*.csv")) if "qc" not in path.name.lower()]
    if not csv_files:
        raise FileNotFoundError(tr('DLC H5/CSV가 없습니다: {v0}', v0=arena_dir))
    return pd.read_csv(csv_files[0], header=[0, 1, 2]), csv_files[0]


def coordinate(frame: pd.DataFrame, bodypart: str, coord: str) -> np.ndarray:
    if not isinstance(frame.columns, pd.MultiIndex):
        raise ValueError(tr('DLC MultiIndex 헤더가 아닙니다.'))
    names = list(frame.columns.names)
    body_level = names.index("bodyparts") if "bodyparts" in names else 1
    coord_level = names.index("coords") if "coords" in names else frame.columns.nlevels - 1
    selected = frame.xs(bodypart, level=body_level, axis=1).xs(coord, level=coord_level - 1, axis=1)
    if isinstance(selected, pd.Series):
        return selected.to_numpy(dtype=float)
    return selected.iloc[:, 0].to_numpy(dtype=float)


def optional_coordinate(frame: pd.DataFrame, bodypart: str, coord: str) -> np.ndarray:
    """Return a DLC coordinate or NaNs when an optional bodypart is unavailable."""
    if not bodypart:
        return np.full(len(frame), np.nan, dtype=float)
    try:
        return coordinate(frame, bodypart, coord)
    except (KeyError, ValueError):
        matching = next(
            (part for part in available_bodyparts(frame) if part.lower() == bodypart.lower()), ""
        )
        if matching:
            try:
                return coordinate(frame, matching, coord)
            except (KeyError, ValueError):
                pass
        return np.full(len(frame), np.nan, dtype=float)


def available_bodyparts(frame: pd.DataFrame) -> list[str]:
    if not isinstance(frame.columns, pd.MultiIndex):
        return []
    names = list(frame.columns.names)
    body_level = names.index("bodyparts") if "bodyparts" in names else 1
    return list(dict.fromkeys(str(value) for value in frame.columns.get_level_values(body_level)))


def required_coordinate(
    frame: pd.DataFrame, bodypart: str, coord: str, role_label: str,
) -> np.ndarray:
    if not bodypart:
        raise ValueError(tr('NORT 판정에 사용할 {v0} 신체점을 지정하세요.', v0=role_label))
    try:
        return coordinate(frame, bodypart, coord)
    except (KeyError, ValueError) as exc:
        matching = next(
            (part for part in available_bodyparts(frame) if part.lower() == bodypart.lower()), ""
        )
        if matching:
            return coordinate(frame, matching, coord)
        available = ", ".join(available_bodyparts(frame)) or tr('없음')
        raise ValueError(
            tr("NORT {v0} 역할로 지정한 신체점 '{v1}'을(를) 좌표 파일에서 찾을 수 없습니다. 사용 가능한 신체점: {v2}", v0=role_label, v1=bodypart, v2=available)
        ) from exc


def signed_distance(polygon: np.ndarray, x: float, y: float) -> float:
    return float(cv2.pointPolygonTest(polygon, (float(x), float(y)), True))


def distance_to_region(polygon: np.ndarray, x: float, y: float) -> float:
    """Shortest distance to a filled polygon; points inside it have distance zero."""
    return max(0.0, -signed_distance(polygon, x, y))


def nearest_polygon_point(polygon: np.ndarray, x: float, y: float) -> np.ndarray:
    point = np.asarray([x, y], dtype=float)
    vertices = np.asarray(polygon, dtype=float).reshape(-1, 2)
    nearest = vertices[0].copy()
    best = math.inf
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
        edge = end - start
        denominator = float(np.dot(edge, edge))
        ratio = 0.0 if denominator <= 0 else float(np.clip(np.dot(point - start, edge) / denominator, 0.0, 1.0))
        candidate = start + ratio * edge
        distance = float(np.linalg.norm(candidate - point))
        if distance < best:
            best = distance
            nearest = candidate
    return nearest


def head_direction_angle(
    polygon: np.ndarray, nose_x: float, nose_y: float, head_x: float, head_y: float,
) -> float:
    """Angle in degrees between head->nose and head->nearest-object directions."""
    forward = np.asarray([nose_x - head_x, nose_y - head_y], dtype=float)
    target = nearest_polygon_point(polygon, head_x, head_y) - np.asarray([head_x, head_y], dtype=float)
    denominator = float(np.linalg.norm(forward) * np.linalg.norm(target))
    if denominator <= 1e-9:
        return math.nan
    cosine = float(np.clip(np.dot(forward, target) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open [start, end) runs from a boolean mask."""
    values = np.asarray(mask, dtype=bool)
    changes = np.diff(np.r_[False, values, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def event_metrics(mask: np.ndarray, fps: float, analysis_start_frame: int) -> dict[str, float | int]:
    runs = true_runs(mask)
    durations = np.asarray([(end - start) / fps for start, end in runs], dtype=float)
    return {
        "bout_count": len(runs),
        "mean_bout_seconds": float(durations.mean()) if durations.size else 0.0,
        "median_bout_seconds": float(np.median(durations)) if durations.size else 0.0,
        "latency_seconds": float((runs[0][0] - analysis_start_frame) / fps) if runs else math.nan,
    }


def bridge_short_gaps(mask: np.ndarray, maximum_frames: int) -> np.ndarray:
    """Join short false runs only when they are enclosed by true samples."""
    result = np.asarray(mask, dtype=bool).copy()
    if maximum_frames <= 0 or result.size < 3:
        return result
    index = 0
    while index < result.size:
        if result[index]:
            index += 1
            continue
        start = index
        while index < result.size and not result[index]:
            index += 1
        if start > 0 and index < result.size and index - start <= maximum_frames:
            result[start:index] = True
    return result


def remove_short_events(mask: np.ndarray, minimum_frames: int) -> np.ndarray:
    """Remove exploration runs shorter than the configured event duration."""
    result = np.asarray(mask, dtype=bool).copy()
    if minimum_frames <= 1:
        return result
    index = 0
    while index < result.size:
        if not result[index]:
            index += 1
            continue
        start = index
        while index < result.size and result[index]:
            index += 1
        if index - start < minimum_frames:
            result[start:index] = False
    return result


def analyze_arena(
    arena_dir: Path,
    setup_path: Path,
    record: dict,
    output_dir: Path,
    pcutoff: float,
    require_orientation: bool,
    minimum_exploration_seconds: float = 0.2,
    bridge_gap_seconds: float = 0.1,
    analysis_start_seconds: float = 0.0,
    analysis_end_seconds: float | None = None,
    bodypart_roles: dict[str, str] | None = None,
) -> dict:
    pose, pose_path = load_pose(arena_dir)
    setup = json.loads(setup_path.read_text(encoding="utf-8"))
    arena = (setup.get("arenas") or [{}])[0]
    objects = arena.get("objects") or {}
    if not all(label in objects for label in ("A", "B")):
        raise ValueError(tr('Object A/B 설정이 없습니다: {v0}', v0=setup_path))

    roles = {
        "nose": "nose",
        "head_center": "head_center",
        "body_center": "body_center",
        "rear_center": "rear_center",
        **(bodypart_roles or {}),
    }
    nose_part = str(roles.get("nose", "") or "")
    head_part = str(roles.get("head_center", "") or "")
    body_part = str(roles.get("body_center", "") or "")
    rear_part = str(roles.get("rear_center", "") or "")
    nose_x = required_coordinate(pose, nose_part, "x", tr('코'))
    nose_y = required_coordinate(pose, nose_part, "y", tr('코'))
    nose_likelihood = required_coordinate(pose, nose_part, "likelihood", tr('코'))
    if require_orientation:
        head_x = required_coordinate(pose, head_part, "x", tr('머리 중심'))
        head_y = required_coordinate(pose, head_part, "y", tr('머리 중심'))
        head_likelihood = required_coordinate(pose, head_part, "likelihood", tr('머리 중심'))
    else:
        head_x = optional_coordinate(pose, head_part, "x")
        head_y = optional_coordinate(pose, head_part, "y")
        head_likelihood = optional_coordinate(pose, head_part, "likelihood")
    body_x = optional_coordinate(pose, body_part, "x")
    body_y = optional_coordinate(pose, body_part, "y")
    body_likelihood = optional_coordinate(pose, body_part, "likelihood")
    rear_x = optional_coordinate(pose, rear_part, "x")
    rear_y = optional_coordinate(pose, rear_part, "y")
    rear_likelihood = optional_coordinate(pose, rear_part, "likelihood")

    metadata_path = arena_dir / "roi-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    fps = float(metadata.get("fps", (setup.get("video") or {}).get("fps", 0.0)) or 0.0)
    if fps <= 0:
        raise ValueError(tr('FPS를 확인할 수 없습니다: {v0}', v0=arena_dir))
    start_frame = min(max(int(math.floor(max(analysis_start_seconds, 0.0) * fps)), 0), len(pose))
    requested_end = len(pose) / fps if analysis_end_seconds is None else max(analysis_end_seconds, 0.0)
    end_frame = min(max(int(math.ceil(requested_end * fps)), start_frame), len(pose))
    if end_frame <= start_frame:
        raise ValueError(tr('분석 구간이 비어 있습니다: Arena {v0}', v0=record['arena_id']))
    in_analysis_range = np.zeros(len(pose), dtype=bool)
    in_analysis_range[start_frame:end_frame] = True
    nose_valid = np.isfinite(nose_x) & np.isfinite(nose_y) & (nose_likelihood >= pcutoff)
    head_valid = np.isfinite(head_x) & np.isfinite(head_y) & (head_likelihood >= pcutoff)
    valid = in_analysis_range & nose_valid & (head_valid if require_orientation else True)
    body_valid = (
        in_analysis_range
        & np.isfinite(body_x)
        & np.isfinite(body_y)
        & (body_likelihood >= pcutoff)
    )
    rear_valid = (
        in_analysis_range
        & np.isfinite(rear_x)
        & np.isfinite(rear_y)
        & (rear_likelihood >= pcutoff)
    )

    boundaries: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label in ("A", "B"):
        object_polygon = np.asarray(objects[label]["polygon_source_px"], dtype=np.float32)
        exploration_polygon = np.asarray(
            objects[label].get("exploration_boundary_source_px", objects[label]["polygon_source_px"]),
            dtype=np.float32,
        )
        boundaries[label] = (object_polygon, exploration_polygon)

    geometry = {
        label: {
            "inside_exploration": np.zeros(len(pose), dtype=bool),
            "nose_distance_px": np.full(len(pose), np.nan, dtype=float),
            "head_distance_px": np.full(len(pose), np.nan, dtype=float),
            "head_angle_degrees": np.full(len(pose), np.nan, dtype=float),
            "orientation_pass": np.zeros(len(pose), dtype=bool),
            "body_on_object": np.zeros(len(pose), dtype=bool),
            "rear_on_object": np.zeros(len(pose), dtype=bool),
            "climbing_qc_candidate": np.zeros(len(pose), dtype=bool),
            "climbing_excluded": np.zeros(len(pose), dtype=bool),
        }
        for label in ("A", "B")
    }
    for index in np.flatnonzero(in_analysis_range):
        if not (np.isfinite(nose_x[index]) and np.isfinite(nose_y[index])):
            continue
        for label, (object_polygon, exploration_polygon) in boundaries.items():
            values = geometry[label]
            values["inside_exploration"][index] = signed_distance(
                exploration_polygon, nose_x[index], nose_y[index]
            ) >= 0
            values["nose_distance_px"][index] = distance_to_region(
                object_polygon, nose_x[index], nose_y[index]
            )
            if np.isfinite(head_x[index]) and np.isfinite(head_y[index]):
                values["head_distance_px"][index] = distance_to_region(
                    object_polygon, head_x[index], head_y[index]
                )
                values["head_angle_degrees"][index] = head_direction_angle(
                    object_polygon, nose_x[index], nose_y[index], head_x[index], head_y[index]
                )
                values["orientation_pass"][index] = (
                    values["nose_distance_px"][index] <= values["head_distance_px"][index]
                )
            if body_valid[index]:
                values["body_on_object"][index] = signed_distance(
                    object_polygon, body_x[index], body_y[index]
                ) >= 0
            if rear_valid[index]:
                values["rear_on_object"][index] = signed_distance(
                    object_polygon, rear_x[index], rear_y[index]
                ) >= 0
            overlap = values["body_on_object"][index] or values["rear_on_object"][index]
            values["climbing_qc_candidate"][index] = overlap
            angle = values["head_angle_degrees"][index]
            # Only a strong 2-D climbing signature is automatically excluded. A single
            # overlapping body point stays visible as a QC candidate for human review.
            values["climbing_excluded"][index] = bool(
                values["body_on_object"][index]
                and values["rear_on_object"][index]
                and np.isfinite(angle)
                and angle > 90.0
            )

    raw_decisions = {"A": np.zeros(len(pose), dtype=bool), "B": np.zeros(len(pose), dtype=bool)}
    reason_code = np.full(len(pose), "outside_analysis_range", dtype=object)
    for index in np.flatnonzero(in_analysis_range):
        if not nose_valid[index]:
            reason_code[index] = "invalid_nose_pose"
            continue
        if require_orientation and not head_valid[index]:
            reason_code[index] = "invalid_head_pose"
            continue
        candidates: list[tuple[float, str]] = []
        saw_exploration_zone = False
        saw_orientation_rejection = False
        saw_climbing_exclusion = False
        for label in ("A", "B"):
            values = geometry[label]
            if not values["inside_exploration"][index]:
                continue
            saw_exploration_zone = True
            if values["climbing_excluded"][index]:
                saw_climbing_exclusion = True
                continue
            if require_orientation and not values["orientation_pass"][index]:
                saw_orientation_rejection = True
                continue
            candidates.append((float(values["nose_distance_px"][index]), label))
        if candidates:
            _distance, selected = min(candidates)
            raw_decisions[selected][index] = True
            reason_code[index] = f"candidate_{selected}"
        elif saw_climbing_exclusion:
            reason_code[index] = "climbing_excluded"
        elif saw_orientation_rejection:
            reason_code[index] = "orientation_rejected"
        elif saw_exploration_zone:
            reason_code[index] = "candidate_rejected"
        else:
            reason_code[index] = "outside_exploration_boundary"

    bridge_frames = max(0, int(round(max(bridge_gap_seconds, 0.0) * fps)))
    minimum_frames = max(1, int(math.ceil(max(minimum_exploration_seconds, 0.0) * fps)))
    decisions = {}
    for label in ("A", "B"):
        bridged = bridge_short_gaps(raw_decisions[label], bridge_frames)
        decisions[label] = remove_short_events(bridged, minimum_frames)
        reason_code[decisions[label]] = f"exploring_{label}"
        reason_code[decisions[label] & ~raw_decisions[label]] = f"bridged_gap_{label}"
        reason_code[raw_decisions[label] & ~decisions[label]] = f"short_event_removed_{label}"

    per_frame = pd.DataFrame({
        "frame": np.arange(len(pose), dtype=int),
        "time_seconds": np.arange(len(pose), dtype=float) / fps,
        "nose_x": nose_x,
        "nose_y": nose_y,
        "nose_likelihood": nose_likelihood,
        "head_center_x": head_x,
        "head_center_y": head_y,
        "head_center_likelihood": head_likelihood,
        "body_center_x": body_x,
        "body_center_y": body_y,
        "body_center_likelihood": body_likelihood,
        "rear_center_x": rear_x,
        "rear_center_y": rear_y,
        "rear_center_likelihood": rear_likelihood,
        "body_pose_valid": body_valid,
        "in_analysis_range": in_analysis_range,
        "valid_pose": valid,
        "decision_reason": reason_code,
        "raw_candidate_A": raw_decisions["A"],
        "raw_candidate_B": raw_decisions["B"],
        "exploring_A": decisions["A"],
        "exploring_B": decisions["B"],
    })
    for label in ("A", "B"):
        for field, values in geometry[label].items():
            per_frame[f"{field}_{label}"] = values
    arena_number = int(record["arena_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    per_frame_path = output_dir / f"arena_{arena_number:02d}_nort_frames.csv"
    per_frame_path = write_csv_resilient(per_frame, per_frame_path, index=False, encoding="utf-8-sig")

    seconds_a = float(decisions["A"].sum() / fps)
    seconds_b = float(decisions["B"].sum() / fps)
    role_a = record["object_a_role"]
    role_b = record["object_b_role"]
    novel = seconds_a if role_a == "Novel" else seconds_b
    familiar = seconds_a if role_a == "Familiar" else seconds_b
    denominator = novel + familiar
    discrimination_index = (novel - familiar) / denominator if denominator > 0 else math.nan
    preference_ratio = novel / denominator if denominator > 0 else math.nan
    metrics_a = event_metrics(decisions["A"], fps, start_frame)
    metrics_b = event_metrics(decisions["B"], fps, start_frame)
    novel_label = "A" if role_a == "Novel" else "B"
    familiar_label = "A" if role_a == "Familiar" else "B"
    novel_metrics = metrics_a if novel_label == "A" else metrics_b
    familiar_metrics = metrics_a if familiar_label == "A" else metrics_b

    def selected_stat(label: str, field: str, reducer) -> float:
        values = np.asarray(geometry[label][field], dtype=float)[decisions[label]]
        values = values[np.isfinite(values)]
        return float(reducer(values)) if values.size else math.nan

    climbing_qc = geometry["A"]["climbing_qc_candidate"] | geometry["B"]["climbing_qc_candidate"]
    climbing_excluded = geometry["A"]["climbing_excluded"] | geometry["B"]["climbing_excluded"]

    def sensitivity_decisions(cutoff: float, orientation_required: bool, minimum_seconds: float) -> dict[str, np.ndarray]:
        eligible = (
            in_analysis_range
            & np.isfinite(nose_x)
            & np.isfinite(nose_y)
            & (nose_likelihood >= cutoff)
        )
        if orientation_required:
            eligible &= np.isfinite(head_x) & np.isfinite(head_y) & (head_likelihood >= cutoff)
        raw = {"A": np.zeros(len(pose), dtype=bool), "B": np.zeros(len(pose), dtype=bool)}
        for index in np.flatnonzero(eligible):
            candidates = []
            for label in ("A", "B"):
                values = geometry[label]
                if not values["inside_exploration"][index] or values["climbing_excluded"][index]:
                    continue
                if orientation_required and not values["orientation_pass"][index]:
                    continue
                candidates.append((float(values["nose_distance_px"][index]), label))
            if candidates:
                raw[min(candidates)[1]][index] = True
        required_frames = max(1, int(math.ceil(max(minimum_seconds, 0.0) * fps)))
        return {
            label: remove_short_events(bridge_short_gaps(raw[label], bridge_frames), required_frames)
            for label in ("A", "B")
        }

    sensitivity_rows = []
    cutoff_values = sorted({round(max(0.0, pcutoff - 0.1), 2), round(pcutoff, 2), round(min(1.0, pcutoff + 0.1), 2)})
    duration_values = sorted({0.1, round(max(minimum_exploration_seconds, 0.0), 2), 0.5})
    for cutoff in cutoff_values:
        for orientation_required in (False, True):
            for minimum_seconds in duration_values:
                masks = sensitivity_decisions(cutoff, orientation_required, minimum_seconds)
                seconds = {label: float(masks[label].sum() / fps) for label in ("A", "B")}
                sensitivity_novel = seconds[novel_label]
                sensitivity_familiar = seconds[familiar_label]
                sensitivity_total = sensitivity_novel + sensitivity_familiar
                sensitivity_rows.append({
                    "is_current_setting": bool(
                        math.isclose(cutoff, round(pcutoff, 2), abs_tol=1e-9)
                        and orientation_required == require_orientation
                        and math.isclose(
                            minimum_seconds,
                            round(max(minimum_exploration_seconds, 0.0), 2),
                            abs_tol=1e-9,
                        )
                    ),
                    "pcutoff": cutoff,
                    "orientation_required": orientation_required,
                    "minimum_exploration_seconds": minimum_seconds,
                    "bridge_gap_seconds": bridge_gap_seconds,
                    "object_a_seconds": seconds["A"],
                    "object_b_seconds": seconds["B"],
                    "novel_seconds": sensitivity_novel,
                    "familiar_seconds": sensitivity_familiar,
                    "preference_ratio": sensitivity_novel / sensitivity_total if sensitivity_total > 0 else math.nan,
                    "discrimination_index": (
                        (sensitivity_novel - sensitivity_familiar) / sensitivity_total
                        if sensitivity_total > 0 else math.nan
                    ),
                    "novel_bout_count": int(event_metrics(masks[novel_label], fps, start_frame)["bout_count"]),
                    "familiar_bout_count": int(event_metrics(masks[familiar_label], fps, start_frame)["bout_count"]),
                })
    sensitivity_path = output_dir / f"arena_{arena_number:02d}_threshold_sensitivity.csv"
    sensitivity_path = write_csv_resilient(
        pd.DataFrame(sensitivity_rows), sensitivity_path, index=False, encoding="utf-8-sig"
    )
    return {
        "video_bundle": record.get("video_bundle", arena_dir.parent.name),
        "arena_id": arena_number,
        "subject_id": record["subject_id"],
        "object_a_role": role_a,
        "object_b_role": role_b,
        "object_a_seconds": seconds_a,
        "object_b_seconds": seconds_b,
        "novel_seconds": novel,
        "familiar_seconds": familiar,
        "preference_ratio": preference_ratio,
        "discrimination_index": discrimination_index,
        "object_a_bout_count": metrics_a["bout_count"],
        "object_b_bout_count": metrics_b["bout_count"],
        "novel_bout_count": novel_metrics["bout_count"],
        "familiar_bout_count": familiar_metrics["bout_count"],
        "novel_mean_bout_seconds": novel_metrics["mean_bout_seconds"],
        "familiar_mean_bout_seconds": familiar_metrics["mean_bout_seconds"],
        "novel_median_bout_seconds": novel_metrics["median_bout_seconds"],
        "familiar_median_bout_seconds": familiar_metrics["median_bout_seconds"],
        "novel_first_exploration_latency_seconds": novel_metrics["latency_seconds"],
        "familiar_first_exploration_latency_seconds": familiar_metrics["latency_seconds"],
        "novel_minimum_nose_object_distance_px": selected_stat(novel_label, "nose_distance_px", np.min),
        "familiar_minimum_nose_object_distance_px": selected_stat(familiar_label, "nose_distance_px", np.min),
        "novel_mean_head_direction_angle_degrees": selected_stat(novel_label, "head_angle_degrees", np.mean),
        "familiar_mean_head_direction_angle_degrees": selected_stat(familiar_label, "head_angle_degrees", np.mean),
        "climbing_qc_candidate_frames": int((climbing_qc & in_analysis_range).sum()),
        "climbing_excluded_frames": int((climbing_excluded & in_analysis_range).sum()),
        "valid_pose_percent": float(valid[in_analysis_range].mean() * 100.0),
        "fps": fps,
        "frames": int(end_frame - start_frame),
        "total_frames": int(len(pose)),
        "analysis_start_seconds": float(start_frame / fps),
        "analysis_end_seconds": float(end_frame / fps),
        "analysis_duration_seconds": float((end_frame - start_frame) / fps),
        "pose_file": str(pose_path.resolve()),
        "pose_postprocessing": json.dumps(pose.attrs.get("hagisilta_analysis_pose", {}), ensure_ascii=False),
        "nose_bodypart": nose_part,
        "head_bodypart": head_part,
        "body_bodypart": body_part,
        "rear_bodypart": rear_part,
        "arena_dir": str(arena_dir.resolve()),
        "setup_file": str(setup_path.resolve()),
        "per_frame_file": str(per_frame_path.resolve()),
        "threshold_sensitivity_file": str(sensitivity_path.resolve()),
    }


def arena_video(arena_dir: Path) -> Path | None:
    from hagisilta_pose_postprocess import selected_analysis_video
    selected = selected_analysis_video(arena_dir)
    if selected is not None:
        return selected
    from hagisilta_labeled_video import source_video_candidates
    videos = source_video_candidates(arena_dir)
    return videos[0] if videos else None


def background_frame(arena_dir: Path, frame_index: int, width: int, height: int) -> np.ndarray:
    video = arena_video(arena_dir)
    if video:
        capture = cv2.VideoCapture(str(video))
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
        ok, frame = capture.read()
        capture.release()
        if ok:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return np.full((max(height, 1), max(width, 1), 3), 35, dtype=np.uint8)


def seconds_heatmap(
    x: np.ndarray, y: np.ndarray, mask: np.ndarray,
    width: int, height: int, fps: float,
    grid_long_side: int = 96,
) -> np.ndarray:
    aspect = width / max(height, 1)
    grid_width = grid_long_side if aspect >= 1 else max(24, round(grid_long_side * aspect))
    grid_height = grid_long_side if aspect < 1 else max(24, round(grid_long_side / aspect))
    valid = mask & np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    heat, _y_edges, _x_edges = np.histogram2d(
        y[valid], x[valid], bins=(grid_height, grid_width),
        range=((0, height), (0, width)), weights=np.full(int(valid.sum()), 1.0 / fps),
    )
    total = float(heat.sum())
    heat = cv2.GaussianBlur(heat.astype(np.float32), (0, 0), sigmaX=1.35, sigmaY=1.35)
    if total > 0 and heat.sum() > 0:
        heat *= total / float(heat.sum())
    return heat


def draw_object_boundaries(ax, setup: dict) -> None:
    arena = (setup.get("arenas") or [{}])[0]
    for label, color in (("A", "#ffd166"), ("B", "#ff6b8a")):
        obj = (arena.get("objects") or {}).get(label) or {}
        polygon = np.asarray(obj.get("polygon_source_px") or [], dtype=float)
        exploration = np.asarray(obj.get("exploration_boundary_source_px") or [], dtype=float)
        for points, linestyle, alpha in ((exploration, "--", 0.8), (polygon, "-", 1.0)):
            if len(points) >= 3:
                closed = np.vstack([points, points[0]])
                ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.2, linestyle=linestyle, alpha=alpha)
        if len(polygon) >= 3:
            center = polygon.mean(axis=0)
            ax.text(center[0], center[1], label, color="white", fontsize=9, fontweight="bold", ha="center", va="center")


def save_heat_grid(path: Path, heat: np.ndarray) -> Path:
    columns = [f"x_{index:03d}" for index in range(heat.shape[1])]
    return write_csv_resilient(
        pd.DataFrame(heat, columns=columns), path, index_label="y_bin", encoding="utf-8-sig"
    )


def render_arena_visuals(
    result: dict,
    common_occupancy_max: float,
    common_exploration_max: float,
) -> dict:
    arena_number = int(result["arena_id"])
    output_dir = Path(result["per_frame_file"]).parent
    arena_dir = Path(result["arena_dir"])
    setup = json.loads(Path(result["setup_file"]).read_text(encoding="utf-8"))
    frame = pd.read_csv(result["per_frame_file"])
    video_meta = setup.get("video") or {}
    width = int(video_meta.get("width", 0) or 0)
    height = int(video_meta.get("height", 0) or 0)
    video = arena_video(arena_dir)
    if video and (width <= 0 or height <= 0):
        capture = cv2.VideoCapture(str(video))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        capture.release()
    width, height = max(width, 1), max(height, 1)
    start_frame = round(float(result["analysis_start_seconds"]) * float(result["fps"]))
    background = background_frame(arena_dir, start_frame, width, height)
    in_range = frame["in_analysis_range"].astype(bool).to_numpy()
    body_valid = frame["body_pose_valid"].astype(bool).to_numpy()
    explore_mask = in_range & (
        frame["exploring_A"].astype(bool).to_numpy() | frame["exploring_B"].astype(bool).to_numpy()
    )
    body_x = frame["body_center_x"].to_numpy(dtype=float)
    body_y = frame["body_center_y"].to_numpy(dtype=float)
    nose_x = frame["nose_x"].to_numpy(dtype=float)
    nose_y = frame["nose_y"].to_numpy(dtype=float)
    occupancy = seconds_heatmap(body_x, body_y, body_valid, width, height, float(result["fps"]))
    exploration = seconds_heatmap(nose_x, nose_y, explore_mask, width, height, float(result["fps"]))
    occupancy_csv = output_dir / f"arena_{arena_number:02d}_occupancy_heatmap_seconds.csv"
    exploration_csv = output_dir / f"arena_{arena_number:02d}_exploration_heatmap_seconds.csv"
    occupancy_csv = save_heat_grid(occupancy_csv, occupancy)
    exploration_csv = save_heat_grid(exploration_csv, exploration)

    trajectory_path = output_dir / f"arena_{arena_number:02d}_trajectory.png"
    fig, ax = plt.subplots(
        figsize=(RESULT_RENDER_WIDTH_IN, 12.0),
        dpi=RESULT_RENDER_DPI,
        constrained_layout=True,
    )
    ax.imshow(background, origin="upper")
    line_x = body_x.copy(); line_y = body_y.copy()
    line_x[~body_valid] = np.nan; line_y[~body_valid] = np.nan
    ax.plot(line_x, line_y, color="#00e5e5", linewidth=0.55, alpha=0.62)
    draw_object_boundaries(ax, setup)
    ax.set_title(f"{result['video_bundle']} · Arena {arena_number:02d} · Body-center trajectory")
    ax.set_xlim(0, width); ax.set_ylim(height, 0); ax.axis("off")
    trajectory_path = save_figure_resilient(fig, trajectory_path, facecolor="white"); plt.close(fig)

    visual_paths = {"trajectory_image": str(trajectory_path.resolve())}
    for key, heat, vmax, title, csv_path in (
        ("occupancy", occupancy, common_occupancy_max, "Body-center occupancy time", occupancy_csv),
        ("exploration", exploration, common_exploration_max, "Nose exploration time", exploration_csv),
    ):
        image_path = output_dir / f"arena_{arena_number:02d}_{key}_heatmap.png"
        fig, ax = plt.subplots(
            figsize=(RESULT_RENDER_WIDTH_IN, 12.0),
            dpi=RESULT_RENDER_DPI,
            constrained_layout=True,
        )
        ax.imshow(background, origin="upper")
        masked = np.ma.masked_less_equal(heat, 0)
        image = ax.imshow(
            masked, origin="upper", extent=(0, width, height, 0), cmap="turbo",
            norm=Normalize(vmin=0, vmax=max(float(vmax), 1e-9)), alpha=0.78,
            interpolation="bilinear",
        )
        draw_object_boundaries(ax, setup)
        ax.set_title(f"{result['video_bundle']} · Arena {arena_number:02d} · {title}")
        ax.set_xlim(0, width); ax.set_ylim(height, 0); ax.axis("off")
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        colorbar.set_label("Time per spatial bin (s)")
        image_path = save_figure_resilient(fig, image_path, facecolor="white"); plt.close(fig)
        visual_paths[f"{key}_heatmap_image"] = str(image_path.resolve())
        visual_paths[f"{key}_heatmap_csv"] = str(csv_path.resolve())
    return visual_paths


def generate_subject_visualizations(results: list[dict], output: Path) -> Path:
    cached: list[tuple[np.ndarray, np.ndarray]] = []
    for result in results:
        frame = pd.read_csv(result["per_frame_file"])
        setup = json.loads(Path(result["setup_file"]).read_text(encoding="utf-8"))
        video_meta = setup.get("video") or {}
        width, height = int(video_meta.get("width", 1) or 1), int(video_meta.get("height", 1) or 1)
        in_range = frame["in_analysis_range"].astype(bool).to_numpy()
        body_mask = frame["body_pose_valid"].astype(bool).to_numpy()
        explore_mask = in_range & (frame["exploring_A"].astype(bool).to_numpy() | frame["exploring_B"].astype(bool).to_numpy())
        occupancy = seconds_heatmap(
            frame["body_center_x"].to_numpy(dtype=float), frame["body_center_y"].to_numpy(dtype=float),
            body_mask, width, height, float(result["fps"]),
        )
        exploration = seconds_heatmap(
            frame["nose_x"].to_numpy(dtype=float), frame["nose_y"].to_numpy(dtype=float),
            explore_mask, width, height, float(result["fps"]),
        )
        cached.append((occupancy, exploration))
    occupancy_max = max((float(heat[0].max()) for heat in cached), default=1.0)
    exploration_max = max((float(heat[1].max()) for heat in cached), default=1.0)
    for result in results:
        result.update(render_arena_visuals(result, occupancy_max, exploration_max))

    figure, axes = plt.subplots(
        len(results), 3,
        figsize=(RESULT_RENDER_WIDTH_IN, max(4.2 * len(results), 4.5)),
        dpi=RESULT_RENDER_DPI,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for row, result in enumerate(results):
        paths = (
            result["trajectory_image"], result["occupancy_heatmap_image"], result["exploration_heatmap_image"]
        )
        for column, path in enumerate(paths):
            axes[row, column].imshow(plt.imread(path))
            axes[row, column].axis("off")
        axes[row, 0].set_ylabel(f"Arena {int(result['arena_id']):02d}", fontsize=10)
    for column, title in enumerate(("Trajectory", "Occupancy heatmap", "Exploration heatmap")):
        axes[0, column].set_title(title, fontsize=12, fontweight="bold")
    figure.suptitle(str(results[0].get("video_bundle", "NORT visualization")), fontsize=14, fontweight="bold")
    summary_path = output / "nort-visual-summary.png"
    summary_path = save_figure_resilient(figure, summary_path, facecolor="white")
    plt.close(figure)
    return summary_path


def analyze_subject(
    subject_root: Path,
    setup_root: Path,
    experiment_info: Path,
    output: Path,
    pcutoff: float,
    orientation: bool,
    minimum_exploration: float = 0.2,
    bridge_gap: float = 0.1,
    bodypart_roles: dict[str, str] | None = None,
) -> Path:
    base_output = output
    base_output.mkdir(parents=True, exist_ok=True)
    base_run_id = "run_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = base_run_id; counter = 2
    output = base_output / "runs" / run_id
    while output.exists():
        run_id = f"{base_run_id}_{counter}"; counter += 1
        output = base_output / "runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    print(f"NORT_RUN={output.resolve()}", flush=True)
    experiment = json.loads(experiment_info.read_text(encoding="utf-8"))
    legacy_analysis_range = experiment.get("analysis_range") or {}
    records = {int(row["arena_id"]): row for row in experiment.get("arenas", [])}
    results = []
    arena_dirs = sorted(path for path in subject_root.glob("arena_*") if path.is_dir())
    for sequence, arena_dir in enumerate(arena_dirs, start=1):
        number = arena_id(arena_dir)
        if number is None or number not in records:
            continue
        setup_path = setup_root / f"arena_{number:02d}_nort_setup.json"
        if not setup_path.is_file():
            raise FileNotFoundError(tr('NORT 공간 설정이 없습니다: {v0}', v0=setup_path))
        record = dict(records[number])
        record["video_bundle"] = experiment.get("video_bundle", subject_root.name)
        analysis_range = record.get("analysis_range") or legacy_analysis_range
        analysis_start = float(analysis_range.get("start_seconds", 0.0) or 0.0)
        raw_end = analysis_range.get("end_seconds")
        analysis_end = float(raw_end) if raw_end is not None else None
        print(f"NORT_ARENA={sequence}/{len(arena_dirs)}:{number}", flush=True)
        results.append(
            analyze_arena(
                arena_dir, setup_path, record, output,
                min(max(float(pcutoff), 0.0), 1.0), bool(orientation),
                max(float(minimum_exploration), 0.0), max(float(bridge_gap), 0.0),
                analysis_start, analysis_end, bodypart_roles,
            )
        )
    if not results:
        raise ValueError(tr('분석할 Arena가 없습니다.'))
    output.mkdir(parents=True, exist_ok=True)
    visual_summary = generate_subject_visualizations(results, output)
    for result in results:
        result["visual_summary_image"] = str(visual_summary.resolve())
    summary = pd.DataFrame(results)
    summary_csv = output / "nort-summary.csv"
    summary_json = output / "nort-summary.json"
    summary_csv = write_csv_resilient(summary, summary_csv, index=False, encoding="utf-8-sig")
    summary_json = write_text_resilient(
        summary_json, json.dumps(results, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )
    write_text_resilient(
        output / "nort-summary.latest.json",
        json.dumps({"csv": str(summary_csv.resolve()), "json": str(summary_json.resolve())}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_text_resilient(
        base_output / "latest-run.json",
        json.dumps({
            "run_id": run_id,
            "run_directory": str(output.resolve()),
            "summary_csv": str(summary_csv.resolve()),
            "summary_json": str(summary_json.resolve()),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"NORT_VISUAL_SUMMARY={visual_summary.resolve()}", flush=True)
    print(f"NORT_SUMMARY={summary_csv.resolve()}", flush=True)
    return summary_csv


def main() -> int:
    args = parse_args()
    if args.batch_plan:
        plan = json.loads(args.batch_plan.read_text(encoding="utf-8"))
        jobs = plan.get("jobs") or []
        if not jobs:
            raise ValueError(tr('배치 분석 계획에 영상이 없습니다.'))
        for index, job in enumerate(jobs, start=1):
            print(f"NORT_BUNDLE={index}/{len(jobs)}:{job.get('name', '')}", flush=True)
            analyze_subject(
                Path(job["subject_root"]), Path(job["setup_root"]),
                Path(job["experiment_info"]), Path(job["output"]),
                float(job.get("pcutoff", args.pcutoff)),
                bool(job.get("orientation", args.orientation)),
                float(job.get("minimum_exploration", args.minimum_exploration)),
                float(job.get("bridge_gap", args.bridge_gap)),
                dict(job.get("bodypart_roles") or {}),
            )
        print(f"NORT_BATCH_COMPLETE={len(jobs)}", flush=True)
        return 0
    required = (args.subject_root, args.setup_root, args.experiment_info, args.output)
    if any(value is None for value in required):
        raise ValueError(tr('단일 분석에는 subject-root, setup-root, experiment-info, output이 필요합니다.'))
    analyze_subject(
        args.subject_root, args.setup_root, args.experiment_info, args.output,
        args.pcutoff, args.orientation,
        args.minimum_exploration, args.bridge_gap,
        {
            "nose": args.nose_bodypart,
            "head_center": args.head_bodypart,
            "body_center": args.body_bodypart,
            "rear_center": args.rear_bodypart,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
