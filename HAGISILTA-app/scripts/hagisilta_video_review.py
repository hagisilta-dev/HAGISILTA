# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from hagisilta_i18n import init_i18n, tr

import argparse
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from hagisilta_window_chrome import (
    HeaderContext,
    apply_hagisilta_window_identity, create_hagisilta_help_button,
    hagisilta_control_stylesheet, install_wheel_value_guard,
)
from hagisilta_runtime import project_output_root
from hagisilta_review_baseline import review_source_label
from hagisilta_playback import PLAYBACK_HELP, PlaybackSpeedControl, playback_interval_ms
from hagisilta_temporal_episodes import (
    DEFAULT_GT_INTERVAL_SECONDS,
    episode_frame_groups,
    episodes_overlap,
    make_episode,
    normalize_episode,
    sample_gt_frames,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ICON = REPO_ROOT / "assets" / "hagisilta-app.ico"


COLORS = [
    QColor("#ff5f7a"),
    QColor("#ffd166"),
    QColor("#45d6b0"),
    QColor("#75a7ff"),
    QColor("#c792ea"),
    QColor("#ff9f43"),
]


def review_playlist_caption(entry: dict, index: int, state: str) -> tuple[str, str]:
    """Keep video identity separate from result-type labels, without filesystem reads."""
    video = Path(str(entry.get("video", "") or ""))
    source = Path(str(entry.get("source_video", "") or video))
    label = str(entry.get("label", "") or "")
    pose = str(entry.get("pose", "") or "")
    stem = source.stem
    # Only remove generated HAGISILTA suffixes; preserve user timestamps/names.
    roi_match = re.search(r"(?:^|_)arena_(\d+)(?:_roi)?$", video.stem, re.IGNORECASE)
    source_roi = re.search(r"(?:^|_)arena_(\d+)(?:_roi)?$", stem, re.IGNORECASE)
    if source_roi:
        stem = stem[:source_roi.start()]
        stem = re.sub(r"_run\d{8}-\d{6}-\d+$", "", stem)
    roi = roi_match.group(1) if roi_match else entry.get("arena_id")
    if roi is None:
        legacy_roi = re.search(r"(?:^| · )ROI\s+(\d+)(?:$| · )", label, re.IGNORECASE)
        parent_roi = re.fullmatch(r"arena_(\d+)", video.parent.name, re.IGNORECASE)
        roi = legacy_roi.group(1) if legacy_roi else parent_roi.group(1) if parent_roi else None
    details = []
    if roi is not None:
        try:
            details.append(f"ROI {int(roi):02d}")
        except (TypeError, ValueError):
            pass
    shuffle = re.search(r"shuffle(\d+)", Path(pose).stem, re.IGNORECASE)
    if shuffle:
        details.append(f"Shuffle {int(shuffle.group(1))}")
    details.append(state)
    title = stem or video.stem or label or tr('영상 미지정')
    text = f"{index + 1:02d} · {title}\n{' · '.join(details)}"
    tooltip = [tr('영상: {p0}', p0=video)]
    if source != video:
        tooltip.append(tr('원본 영상: {p0}', p0=source))
    tooltip.append(tr('좌표: {p0}', p0=pose or tr('없음')))
    for detail in dict.fromkeys((label, str(entry.get("result_label", "") or ""))):
        if detail:
            tooltip.append(detail)
    return text, "\n".join(tooltip)


class ClickSeekSlider(QSlider):
    seek_requested = Signal(int)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.width() > 0:
            ratio = max(0.0, min(1.0, event.position().x() / self.width()))
            value = self.minimum() + round((self.maximum() - self.minimum()) * ratio)
            self.setValue(value)
            self.seek_requested.emit(value)
        super().mousePressEvent(event)


def read_pose(path: Path) -> tuple[pd.DataFrame, list[str]]:
    if path.suffix.lower() in {".h5", ".hdf5"}:
        frame = pd.read_hdf(path)
    else:
        frame = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    if not isinstance(frame.columns, pd.MultiIndex) or frame.columns.nlevels < 3:
        raise ValueError(tr('DLC MultiIndex CSV/H5 형식이 아닙니다.'))
    coord_level = frame.columns.nlevels - 1
    bodypart_level = frame.columns.nlevels - 2
    coords = set(map(str, frame.columns.get_level_values(coord_level)))
    if not {"x", "y"}.issubset(coords):
        raise ValueError(tr('좌표 열(x, y)을 찾을 수 없습니다.'))
    bodyparts = list(dict.fromkeys(map(str, frame.columns.get_level_values(bodypart_level))))
    return frame, bodyparts


def pose_column(frame: pd.DataFrame, bodypart: str, coord: str) -> pd.Series:
    bodypart_level = frame.columns.nlevels - 2
    coord_level = frame.columns.nlevels - 1
    matches = [
        column
        for column in frame.columns
        if str(column[bodypart_level]) == bodypart and str(column[coord_level]) == coord
    ]
    if not matches:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[matches[0]], errors="coerce")


ANOMALY_LABELS = {
    "swap": "신체점 교환 의심",
    "outside": "영상 밖 좌표",
    "missing": "좌표 없음",
    "spike_return": "순간 이탈·복귀",
    "motion_outlier": "공통 움직임 이탈",
    "structure": "신체점 배열 이상",
    "jump": "급격한 이동",
    "low_confidence": "낮은 신뢰도",
}
ANOMALY_ORDER = tuple(ANOMALY_LABELS)


def _robust_threshold(values: np.ndarray, minimum: float, deviations: float = 8.0) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return float(minimum)
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return max(float(minimum), median + deviations * 1.4826 * mad)


def detect_pose_anomalies(
    pose: pd.DataFrame,
    bodyparts: list[str],
    width: int,
    height: int,
    confidence_cutoff: float = 0.6,
    jump_threshold: float = 25.0,
    include_structure: bool = True,
) -> dict[int, list[dict]]:
    """Find review candidates without changing coordinates.

    The structural signals are bodypart-name agnostic.  They learn robust motion
    and pair-distance baselines from the current video and only add navigation
    candidates for a human reviewer.
    """
    frame_count = len(pose)
    point_count = len(bodyparts)
    if frame_count <= 0 or point_count <= 0:
        return {}

    coordinates = np.full((frame_count, point_count, 2), np.nan, dtype=float)
    likelihoods = np.full((frame_count, point_count), np.nan, dtype=float)
    for point_index, bodypart in enumerate(bodyparts):
        coordinates[:, point_index, 0] = pose_column(pose, bodypart, "x").to_numpy(float)
        coordinates[:, point_index, 1] = pose_column(pose, bodypart, "y").to_numpy(float)
        likelihoods[:, point_index] = pose_column(
            pose, bodypart, "likelihood"
        ).to_numpy(float)

    finite_xy = np.isfinite(coordinates).all(axis=2)
    dlc_missing = (
        finite_xy
        & np.isfinite(likelihoods)
        & (likelihoods < 0.0)
        & (coordinates[:, :, 0] < 0.0)
        & (coordinates[:, :, 1] < 0.0)
    )
    observed = finite_xy & ~dlc_missing
    details: dict[int, list[dict]] = {}

    def add(frame: int, code: str, points: list[int] | tuple[int, ...], value: float | None = None) -> None:
        if not 0 <= int(frame) < frame_count:
            return
        names = tuple(dict.fromkeys(bodyparts[index] for index in points))
        record = {"code": code, "bodyparts": names}
        if value is not None and np.isfinite(value):
            record["value"] = float(value)
        details.setdefault(int(frame), []).append(record)

    for frame, point in np.argwhere(~observed):
        add(int(frame), "missing", [int(point)])
    low_confidence = observed & np.isfinite(likelihoods) & (likelihoods < confidence_cutoff)
    for frame, point in np.argwhere(low_confidence):
        add(int(frame), "low_confidence", [int(point)], likelihoods[frame, point])

    if frame_count > 1:
        displacement = coordinates[1:] - coordinates[:-1]
        consecutive = observed[1:] & observed[:-1]
        distance = np.linalg.norm(displacement, axis=2)
        for frame, point in np.argwhere(consecutive & (distance > jump_threshold)):
            add(int(frame) + 1, "jump", [int(point)], distance[frame, point])
    else:
        displacement = np.empty((0, point_count, 2), dtype=float)
        consecutive = np.empty((0, point_count), dtype=bool)

    if not include_structure:
        return details

    outside = observed & (
        (coordinates[:, :, 0] < 0.0)
        | (coordinates[:, :, 0] >= max(int(width), 1))
        | (coordinates[:, :, 1] < 0.0)
        | (coordinates[:, :, 1] >= max(int(height), 1))
    )
    for frame, point in np.argwhere(outside):
        add(int(frame), "outside", [int(point)])

    pair_distances: dict[tuple[int, int], np.ndarray] = {}
    frame_extents = np.full(frame_count, np.nan, dtype=float)
    for left in range(point_count):
        for right in range(left + 1, point_count):
            pair_valid = observed[:, left] & observed[:, right]
            values = np.linalg.norm(
                coordinates[:, left] - coordinates[:, right], axis=1
            )
            values[~pair_valid] = np.nan
            pair_distances[(left, right)] = values
    if pair_distances:
        stacked_distances = np.column_stack(list(pair_distances.values()))
        valid_extent_rows = np.isfinite(stacked_distances).any(axis=1)
        frame_extents[valid_extent_rows] = np.nanmax(
            stacked_distances[valid_extent_rows], axis=1
        )
        finite_extents = frame_extents[np.isfinite(frame_extents)]
        body_scale = float(np.median(finite_extents)) if finite_extents.size else 0.0
    else:
        body_scale = 0.0
    if not np.isfinite(body_scale) or body_scale <= 0.0:
        body_scale = max(min(max(int(width), 1), max(int(height), 1)) * 0.1, 10.0)

    # A real animal translation moves most points together.  A label failure
    # leaves one point far from that common displacement.
    if displacement.size and point_count >= 2:
        common_motion = np.full((frame_count - 1, 2), np.nan, dtype=float)
        required_points = min(3, point_count)
        for frame in range(frame_count - 1):
            valid_points = consecutive[frame]
            if int(np.count_nonzero(valid_points)) >= required_points:
                common_motion[frame] = np.median(displacement[frame, valid_points], axis=0)
        motion_residual = np.linalg.norm(
            displacement - common_motion[:, None, :], axis=2
        )
        motion_residual[~consecutive] = np.nan
        minimum_motion = max(3.0, body_scale * 0.12)
        for point in range(point_count):
            threshold = _robust_threshold(motion_residual[:, point], minimum_motion)
            candidates = np.flatnonzero(motion_residual[:, point] > threshold)
            for frame in candidates:
                add(int(frame) + 1, "motion_outlier", [point], motion_residual[frame, point])

    # A one-frame hallucination is far from the midpoint of its two neighbours
    # and immediately returns.  The robust per-point baseline preserves genuine
    # fast motion and normal endpoint articulation.
    if frame_count >= 3:
        neighbour_valid = observed[:-2] & observed[1:-1] & observed[2:]
        midpoint = (coordinates[:-2] + coordinates[2:]) * 0.5
        midpoint_error = np.linalg.norm(coordinates[1:-1] - midpoint, axis=2)
        neighbour_span = np.linalg.norm(coordinates[2:] - coordinates[:-2], axis=2)
        midpoint_error[~neighbour_valid] = np.nan
        minimum_spike = max(3.0, body_scale * 0.10)
        for point in range(point_count):
            threshold = _robust_threshold(midpoint_error[:, point], minimum_spike)
            candidates = np.flatnonzero(
                (midpoint_error[:, point] > threshold)
                & (midpoint_error[:, point] > neighbour_span[:, point] * 0.75 + 2.0)
            )
            for offset in candidates:
                add(int(offset) + 1, "spike_return", [point], midpoint_error[offset, point])

        # Test every bodypoint pair as an unlabeled assignment.  Only a large,
        # unambiguous cost reduction is reported; coordinates remain untouched.
        if point_count >= 2:
            minimum_direct = max(8.0, body_scale * 0.22)
            minimum_improvement = max(5.0, body_scale * 0.12)
            for offset in range(frame_count - 2):
                best: tuple[float, int, int, float, float] | None = None
                predicted = midpoint[offset]
                current = coordinates[offset + 1]
                for left in range(point_count):
                    for right in range(left + 1, point_count):
                        if not (
                            neighbour_valid[offset, left]
                            and neighbour_valid[offset, right]
                        ):
                            continue
                        direct = float(
                            np.linalg.norm(current[left] - predicted[left])
                            + np.linalg.norm(current[right] - predicted[right])
                        )
                        swapped = float(
                            np.linalg.norm(current[right] - predicted[left])
                            + np.linalg.norm(current[left] - predicted[right])
                        )
                        improvement = direct - swapped
                        if (
                            direct >= minimum_direct
                            and improvement >= minimum_improvement
                            and swapped <= direct * 0.45
                            and (best is None or improvement > best[0])
                        ):
                            best = (improvement, left, right, direct, swapped)
                if best is not None:
                    improvement, left, right, _direct, _swapped = best
                    add(offset + 1, "swap", [left, right], improvement)

    # Rare global pair-distance failures catch persistent wrong placements that
    # are not a single-frame spike.  Requiring two extreme pairs around the same
    # point keeps ordinary curled postures from flooding the review list.
    if point_count >= 3 and pair_distances:
        per_frame_counts = np.zeros((frame_count, point_count), dtype=int)
        per_frame_score = np.zeros((frame_count, point_count), dtype=float)
        scale_floor = max(1.0, body_scale * 0.03)
        for (left, right), values in pair_distances.items():
            finite = values[np.isfinite(values)]
            if finite.size < 5:
                continue
            median = float(np.median(finite))
            mad = float(np.median(np.abs(finite - median)))
            scale = max(1.4826 * mad, scale_floor)
            score = np.abs(values - median) / scale
            extreme = np.isfinite(score) & (score > 10.0)
            per_frame_counts[extreme, left] += 1
            per_frame_counts[extreme, right] += 1
            per_frame_score[extreme, left] = np.maximum(
                per_frame_score[extreme, left], score[extreme]
            )
            per_frame_score[extreme, right] = np.maximum(
                per_frame_score[extreme, right], score[extreme]
            )
        for frame, point in np.argwhere(per_frame_counts >= 2):
            add(int(frame), "structure", [int(point)], per_frame_score[frame, point])

    return details


def format_anomaly_reasons(records: list[dict]) -> list[str]:
    grouped: dict[str, dict] = {}
    for record in records:
        code = str(record.get("code", ""))
        if code not in ANOMALY_LABELS:
            continue
        group = grouped.setdefault(code, {"bodyparts": [], "values": []})
        for bodypart in record.get("bodyparts", ()):
            if bodypart not in group["bodyparts"]:
                group["bodyparts"].append(str(bodypart))
        value = record.get("value")
        if value is not None and np.isfinite(value):
            group["values"].append(float(value))

    reasons: list[str] = []
    for code in ANOMALY_ORDER:
        if code not in grouped:
            continue
        group = grouped[code]
        names = group["bodyparts"]
        separator = " ↔ " if code == "swap" and len(names) == 2 else ", "
        text = tr(ANOMALY_LABELS[code])
        if names:
            text += f" · {separator.join(names)}"
        values = group["values"]
        if values:
            if code == "low_confidence":
                text += f" · {min(values):.2f}"
            elif code == "structure":
                text += tr(' · 편차 {p0:.1f}', p0=max(values))
            elif code == "swap":
                text += tr(' · 개선 {p0:.1f}px', p0=max(values))
            else:
                text += f" · {max(values):.1f}px"
        reasons.append(text)
    return reasons


def read_detections(path: Path | None) -> list[dict]:
    if path is None or not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(tr('Detector 결과 JSON 형식이 올바르지 않습니다.'))
    return [record if isinstance(record, dict) else {} for record in payload]


class PosePointItem(QGraphicsEllipseItem):
    def __init__(
        self, bodypart: str, color: QColor, diameter: int, callback, select_callback,
        *, movable: bool = True,
    ) -> None:
        radius = diameter / 2.0
        super().__init__(QRectF(-radius, -radius, diameter, diameter))
        self.bodypart = bodypart
        self.callback = callback
        self.select_callback = select_callback
        self._loading = True
        self.setBrush(color)
        self.setPen(QPen(QColor("#101621"), 2))
        self.setZValue(20)
        flags = QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIgnoresTransformations
        if movable:
            flags |= QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges
        self.setFlags(flags)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged and not self._loading:
            self.callback(self.bodypart, value)
        return super().itemChange(change, value)

    def mousePressEvent(self, event) -> None:
        self.select_callback(self.bodypart)
        super().mousePressEvent(event)


class VideoCanvas(QGraphicsView):
    point_requested = Signal(float, float)

    def __init__(self, allow_point_placement: bool = True) -> None:
        self.scene = QGraphicsScene()
        super().__init__(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.pixmap_item.setZValue(0)
        self.scene.addItem(self.pixmap_item)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor("#080c12"))
        self.setFrameShape(QFrame.NoFrame)
        self.setDragMode(QGraphicsView.NoDrag)
        self._fit_pending = True
        self._zoom_steps = 0
        self._maximum_zoom_steps = 12
        self._pan_active = False
        self._pan_position = None
        self.allow_point_placement = bool(allow_point_placement)

    def set_frame(self, image: QImage) -> None:
        self.pixmap_item.setPixmap(QPixmap.fromImage(image))
        self.scene.setSceneRect(QRectF(0, 0, image.width(), image.height()))
        if self._fit_pending:
            self.fit_to_view()
            self._fit_pending = False

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._zoom_steps == 0 and not self.scene.sceneRect().isEmpty():
            self.fit_to_view()

    def fit_to_view(self) -> None:
        if self.scene.sceneRect().isEmpty():
            return
        self.resetTransform()
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self._zoom_steps = 0

    def zoom_by_steps(self, steps: int, viewport_position=None) -> None:
        if not steps or self.scene.sceneRect().isEmpty():
            return
        next_steps = max(0, min(self._maximum_zoom_steps, self._zoom_steps + int(steps)))
        if next_steps == self._zoom_steps:
            return
        if next_steps == 0:
            self.fit_to_view()
            return
        anchor = viewport_position or self.viewport().rect().center()
        before = self.mapToScene(anchor)
        factor = 1.18 ** (next_steps - self._zoom_steps)
        self.scale(factor, factor)
        self._zoom_steps = next_steps
        after = self.mapToScene(anchor)
        delta = after - before
        self.translate(delta.x(), delta.y())

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if not delta:
            super().wheelEvent(event)
            return
        self.zoom_by_steps(1 if delta > 0 else -1, event.position().toPoint())
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self._pan_active = True
            self._pan_position = event.position().toPoint()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._pan_active and self._pan_position is not None:
            position = event.position().toPoint()
            delta = position - self._pan_position
            self._pan_position = position
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton and self._pan_active:
            self._pan_active = False
            self._pan_position = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if not self.allow_point_placement:
            super().mouseDoubleClickEvent(event)
            return
        point = self.mapToScene(event.position().toPoint())
        if self.scene.sceneRect().contains(point):
            self.point_requested.emit(point.x(), point.y())
        super().mouseDoubleClickEvent(event)


class VideoReviewPilot(QMainWindow):
    def __init__(
        self,
        video: Path,
        pose_path: Path | None,
        output: Path | None,
        scorer: str,
        playlist: list[dict] | None = None,
        playlist_index: int = 0,
        review_only: bool = False,
        draft_manifest: Path | None = None,
    ) -> None:
        super().__init__()
        self.playlist = playlist or [
            {
                "video": str(video),
                "pose": str(pose_path) if pose_path else "",
                "output": str(output) if output else "",
            }
        ]
        self.playlist_index = max(0, min(int(playlist_index), len(self.playlist) - 1))
        self.video = video.resolve()
        self.pose_path = pose_path.resolve() if pose_path else None
        self.scorer = scorer
        self.review_only = bool(review_only)
        self.draft_pose = None
        self.draft_source = None
        self.frame_baselines = {}
        self.previously_reviewed = set()
        self.output = (
            output.resolve() if output else
            (self.video.parent if self.review_only else self.default_output()).resolve()
        )
        if not self.review_only:
            self.output.mkdir(parents=True, exist_ok=True)

        self.capture = cv2.VideoCapture(str(self.video))
        if not self.capture.isOpened():
            raise RuntimeError(tr('영상을 열 수 없습니다: {p0}', p0=self.video))
        self.frame_count = max(1, int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS)) or 10.0
        self.playback_speed = 1.0
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.pose: pd.DataFrame | None = None
        self.bodyparts: list[str] = ["nose", "head_center", "body_center", "tail_base"]
        if self.pose_path:
            try:
                self.pose, self.bodyparts = read_pose(self.pose_path)
            except Exception:
                self.capture.release()
                raise
        self.pose_frames = min(len(self.pose), self.frame_count) if self.pose is not None else 0
        active_entry = self.playlist[self.playlist_index]
        draft_path = draft_manifest or active_entry.get("draft_manifest")
        if draft_path and not self.review_only:
            from hagisilta_review_baseline import load_review_draft
            try:
                self.draft_pose, self.draft_source = load_review_draft(self.video, self.pose_path, draft_path)
            except Exception:
                self.capture.release()
                raise
        self.label_frame_indices = sorted({
            int(value)
            for value in (active_entry.get("label_frames", []) or [])
            if 0 <= int(value) < self.frame_count
        })
        self.label_frame_set = set(self.label_frame_indices)
        self.labeling_mode = bool(self.label_frame_indices) and not self.review_only
        detection_value = str(active_entry.get("detections", "") or "")
        self.detection_path = Path(detection_value).expanduser().resolve() if detection_value else None
        self.detections = read_detections(self.detection_path)

        self.frame_index = 0
        self.selected_bodypart = self.bodyparts[0]
        self.visible_bodyparts = set(self.bodyparts)
        self.edits: dict[int, dict[str, dict]] = {}
        self.confirmed: set[int] = set()
        self.excluded: set[int] = set()
        self.temporal_episodes: list[dict] = []
        self.episode_draft_start: int | None = None
        self.selected_episode_id = ""
        self.dirty = False
        self.markers: dict[str, PosePointItem] = {}
        self.overlay_items: list[QGraphicsItem] = []
        self.flagged: list[int] = []
        self.flagged_set: set[int] = set()
        self.flag_starts: list[int] = []
        self.flag_start_set: set[int] = set()
        self.flag_details: dict[int, list[dict]] = {}

        self.setWindowTitle(
            f"HAGISILTA · {self.mode_title()} · {self.video.name}"
        )
        apply_hagisilta_window_identity(self)
        self.resize(1450, 900)
        self.build_ui()
        self.sync_playlist_selection()
        if not self.review_only:
            self.load_session()
            self.refresh_episode_list()
        self.update_draft_context()
        self.recompute_flags()
        self.show_frame(self.label_frame_indices[0] if self.labeling_mode else 0)

    def mode_title(self) -> str:
        if self.review_only:
            return tr('추론 영상 확인')
        if self.labeling_mode:
            return tr('프레임 라벨링')
        return tr('영상 교정')

    def is_editable_frame(self) -> bool:
        return not self.labeling_mode or self.frame_index in self.label_frame_set

    def require_editable_frame(self) -> bool:
        if self.review_only:
            return False
        if self.is_editable_frame():
            return True
        self.statusBar().showMessage(
            tr('현재 프레임은 영상 문맥 확인용입니다. Q/E로 라벨 프레임을 선택하세요.'), 3500
        )
        return False

    def default_output(self) -> Path:
        if self.video.parent.name.lower() == "videos":
            return project_output_root(self.video.parent.parent, create=True) / "video-review-pilot" / self.video.stem
        return self.video.parent / "video-review-pilot" / self.video.stem

    def build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("pageRoot")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QFrame(objectName="pageHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 10, 18, 10)
        header_layout.setSpacing(10)
        title_accent = QFrame(objectName="pageTitleAccent")
        title_accent.setFixedSize(3, 36)
        header_layout.addWidget(title_accent)
        title_stack = QVBoxLayout()
        title_stack.setContentsMargins(0, 0, 0, 0)
        title_stack.setSpacing(0)
        title_stack.addWidget(QLabel("HAGISILTA · POSE REVIEW", objectName="pageKicker"))
        title_stack.addWidget(QLabel(self.mode_title(), objectName="pageTitle"))
        header_layout.addLayout(title_stack)
        header_layout.addStretch(1)

        zoom_help = tr('마우스 휠 확대·축소 · 가운데 버튼 드래그 화면 이동 · Ctrl+0 화면 맞춤')
        if self.review_only:
            help_text = (
                tr('SuperAnimal 모델이 표시한 신체점의 위치와 움직임을 원본 그대로 확인합니다.\nQ/E 이전·다음 오류 후보 · Space 재생/정지 · A/D 한 프레임 이동\n{p0}\n\n이 화면에서는 점을 수정하거나 검토 상태를 저장하지 않습니다.', p0=zoom_help)
            )
        elif self.labeling_mode:
            help_text = (
                tr('실제 원본/ROI 영상을 재생하며 문맥을 확인합니다.\nQ/E 이전·다음 라벨 프레임 · Space 재생/정지 · A/D 한 프레임 이동\n라벨 프레임에서 점을 드래그하거나 더블클릭해 수정하며, 사이 프레임은 문맥 확인 전용입니다.\nX/Delete 선택 점 삭제 · Ctrl+S 중간 저장 · {p0}\n\n하단 ‘저장하고 닫기’는 수정 좌표를 학습 라벨에 반영한 뒤 창을 닫습니다.', p0=zoom_help)
            )
        else:
            help_text = (
                tr('영상을 넘겨보며 신체점 위치를 확인하고, 잘못된 점은 드래그하여 수정합니다.\nEnter 현재 프레임 검증 완료 · X/Delete 선택 점 삭제 · Ctrl+S 중간 저장\nQ/E 이전·다음 오류 · S/F 시계열 GT 구간 생성 · Shift+S/F 선택 구간 시작·끝 수정\nSpace 재생/정지 · {p0}\n\n하단 ‘저장하고 닫기’는 교정 세션을 저장한 뒤 창을 닫습니다.', p0=zoom_help)
            )
        header_layout.addWidget(create_hagisilta_help_button(
            self, tr('{p0} 도움말', p0=self.mode_title()), help_text + "\n\n" + tr(PLAYBACK_HELP),
        ))
        HeaderContext(header,
            tr('● 확인 중') if self.review_only else tr('● 라벨링 중') if self.labeling_mode else tr('● 교정 중'),
        )
        root.addWidget(header)

        context_bar = QFrame(objectName="contextBar")
        context_layout = QHBoxLayout(context_bar)
        context_layout.setContentsMargins(18, 6, 18, 6)
        context_layout.setSpacing(14)
        self.video_meta = QLabel(tr('영상 · {p0}', p0=self.video.name), objectName="contextMeta")
        self.video_meta.setMaximumWidth(360)
        self.video_meta.setToolTip(str(self.video))
        context_layout.addWidget(self.video_meta)
        self.pose_meta = QLabel(
            tr('예측 · {p0}', p0=self.pose_path.name if self.pose_path else tr('없음')),
            objectName="contextMeta",
        )
        self.pose_meta.setMaximumWidth(420)
        self.pose_meta.setToolTip(str(self.pose_path) if self.pose_path else tr('예측 좌표 없음'))
        context_layout.addWidget(self.pose_meta)
        context_layout.addStretch(1)
        root.addWidget(context_bar)

        content = QWidget(objectName="pageContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(8)
        root.addWidget(content, 1)

        footer = QFrame(objectName="detachedFooter")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(18, 9, 18, 9)
        self.footer_status = QLabel(
            tr('원본 추론 결과를 읽기 전용으로 확인하고 있습니다. 변경하거나 저장하는 내용은 없습니다.')
            if self.review_only else
            tr('교정 내용을 저장하면 다음에 같은 상태로 이어서 작업할 수 있습니다.')
        )
        self.footer_status.setObjectName("footerStatus")
        footer_layout.addWidget(self.footer_status)
        footer_layout.addStretch(1)
        self.save_button = QPushButton(
            tr('닫기') if self.review_only else tr('저장하고 닫기'), objectName="primaryButton"
        )
        self.save_button.setToolTip(
            tr('추론 결과 확인 창을 닫습니다.')
            if self.review_only else
            tr('점 이동·삭제, 판독 불가, 프레임 제외 상태를 교정 세션에 저장합니다. 일반적인 교정 작업에서는 이 버튼만 사용하면 됩니다.')
        )
        self.save_button.clicked.connect(self.close if self.review_only else self.save_and_close)
        footer_layout.addWidget(self.save_button)
        root.addWidget(footer)

        splitter = QSplitter(Qt.Horizontal)

        playlist_panel = QFrame(objectName="playlistPanel")
        playlist_panel.setMinimumWidth(220)
        playlist_panel.setMaximumWidth(480)
        playlist_layout = QVBoxLayout(playlist_panel)
        playlist_layout.setContentsMargins(10, 10, 10, 10)
        playlist_layout.setSpacing(7)
        playlist_layout.addWidget(QLabel(
            tr('추론 결과') if self.review_only else tr('라벨링 대상') if self.labeling_mode else tr('트래킹 결과'),
            objectName="sectionTitle",
        ))
        playlist_hint = QLabel(tr('항목을 클릭하면 이 창에서 바로 전환합니다.'), objectName="playlistHint")
        playlist_hint.setWordWrap(True)
        playlist_layout.addWidget(playlist_hint)
        self.playlist_widget = QListWidget(objectName="resultPlaylist")
        self.playlist_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.playlist_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.playlist_widget.setTextElideMode(Qt.ElideRight)
        self.playlist_widget.setSpacing(4)
        for index, entry in enumerate(self.playlist):
            pose_value = str(entry.get("pose", "") or "")
            label_frames = entry.get("label_frames", []) or []
            draft_source = entry.get("draft_source") or {"label": entry.get("draft_label", "")}
            pose_state = (tr('라벨 프레임 {p0}개', p0=len(label_frames)) if label_frames and not self.review_only
                          else tr('후처리 초안 · ') + review_source_label(draft_source) if entry.get("draft_manifest") and not self.review_only
                          else tr('원본 추론') if pose_value else tr('예측 좌표 없음'))
            caption, tooltip = review_playlist_caption(entry, index, pose_state)
            item = QListWidgetItem(caption)
            item.setToolTip(tooltip)
            self.playlist_widget.addItem(item)
        self.playlist_widget.currentRowChanged.connect(self.switch_playlist_row)
        playlist_layout.addWidget(self.playlist_widget, 1)
        splitter.addWidget(playlist_panel)

        self.canvas = VideoCanvas(allow_point_placement=not self.review_only)
        if not self.review_only:
            self.canvas.point_requested.connect(self.place_selected_point)
        splitter.addWidget(self.canvas)

        panel_scroll = QScrollArea(objectName="reviewSideScroll")
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        panel_scroll.setFrameShape(QFrame.NoFrame)
        panel_scroll.setMinimumWidth(320)
        panel_scroll.setMaximumWidth(430)
        panel = QWidget(objectName="reviewSidePanel")
        side = QVBoxLayout(panel)
        side.setContentsMargins(12, 12, 12, 12)
        side.setSpacing(6)
        side.addWidget(QLabel(tr('신체점'), objectName="sectionTitle"))
        self.bodypart_list = QListWidget()
        for index, bodypart in enumerate(self.bodyparts):
            item = QListWidgetItem(f"{index + 1}   {bodypart}")
            item.setData(Qt.UserRole, bodypart)
            if self.review_only:
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
            self.bodypart_list.addItem(item)
        self.bodypart_list.setCurrentRow(0)
        self.bodypart_list.setMinimumHeight(92)
        self.bodypart_list.setMaximumHeight(112)
        self.bodypart_list.currentRowChanged.connect(self.select_bodypart_index)
        if self.review_only:
            self.bodypart_list.itemChanged.connect(self.bodypart_visibility_changed)
        side.addWidget(self.bodypart_list)

        if self.review_only:
            visibility_tools = QHBoxLayout()
            show_all_points = QPushButton(tr('전체 표시'), objectName="toolButton")
            show_all_points.setToolTip(tr('모든 신체점을 영상에 표시합니다.'))
            show_all_points.clicked.connect(lambda: self.set_all_bodyparts_visible(True))
            hide_all_points = QPushButton(tr('전체 숨김'), objectName="toolButton")
            hide_all_points.setToolTip(tr('Detector 박스는 유지하고 모든 신체점만 숨깁니다.'))
            hide_all_points.clicked.connect(lambda: self.set_all_bodyparts_visible(False))
            visibility_tools.addWidget(show_all_points)
            visibility_tools.addWidget(hide_all_points)
            side.addLayout(visibility_tools)

        self.point_status = QLabel("-")
        self.point_status.setWordWrap(True)
        self.point_status.setObjectName("infoBox")
        side.addWidget(self.point_status)

        overlay_row = QGridLayout()
        self.show_detection_boxes = QCheckBox(tr('검출 박스'))
        self.show_detection_boxes.setChecked(
            bool(self.detections) and bool(self.playlist[self.playlist_index].get("show_bboxes", True))
        )
        self.show_detection_boxes.setEnabled(bool(self.detections))
        self.show_detection_boxes.setToolTip(tr('Detector가 해당 프레임에서 찾은 개체 영역과 신뢰도를 표시합니다.'))
        self.show_detection_boxes.toggled.connect(self.draw_pose)
        self.show_points = QCheckBox(tr('점 표시 (V)'))
        self.show_points.setChecked(True)
        self.show_points.toggled.connect(self.draw_pose)
        self.show_names = QCheckBox(tr('이름·연결선'))
        self.show_names.setChecked(True)
        self.show_names.toggled.connect(self.draw_pose)
        overlay_row.addWidget(self.show_detection_boxes, 0, 0)
        overlay_row.addWidget(self.show_points, 0, 1)
        overlay_row.addWidget(self.show_names, 1, 0, 1, 2)
        side.addLayout(overlay_row)
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel(tr('점 크기')))
        self.point_size = QSlider(Qt.Horizontal)
        self.point_size.setRange(3, 24)
        self.point_size.setValue(8)
        self.point_size.setToolTip(tr('화면에 표시되는 점의 지름입니다. 영상 좌표와 학습 데이터에는 영향을 주지 않습니다.'))
        self.point_size.valueChanged.connect(self.point_size_changed)
        size_row.addWidget(self.point_size, 1)
        self.point_size_label = QLabel("8 px")
        self.point_size_label.setMinimumWidth(42)
        size_row.addWidget(self.point_size_label)
        side.addLayout(size_row)

        if self.review_only:
            work_side = side
        else:
            self.review_tabs = QTabWidget(objectName="reviewToolTabs")
            self.review_tabs.setDocumentMode(True)
            self.review_tabs.tabBar().setExpanding(False)
            correction_tab = QWidget()
            work_side = QVBoxLayout(correction_tab)
            work_side.setContentsMargins(8, 10, 8, 8)
            work_side.setSpacing(6)
            self.review_tabs.addTab(correction_tab, tr('교정·오류 탐색'))
            side.addWidget(self.review_tabs, 1)

        if not self.review_only:
            invisible = QPushButton(tr('안 보임·선택 점 삭제 (X/Delete)'))
            invisible.setToolTip(
                tr('현재 프레임에서 선택한 신체점이 가려졌거나 판독되지 않을 때 사용합니다. 학습 데이터에는 좌표를 억지로 추정하지 않고 결측값(NaN)으로 저장합니다.')
            )
            invisible.clicked.connect(self.mark_selected_invisible)
            work_side.addWidget(invisible)
            reset_point = QPushButton(tr('선택 점을 교정 시작 좌표로 복원'))
            reset_point.clicked.connect(self.reset_selected_point)
            work_side.addWidget(reset_point)
            self.auto_confirm_edits = QCheckBox(tr('점을 수정하면 프레임 자동 검증 완료'))
            self.auto_confirm_edits.setChecked(True)
            self.auto_confirm_edits.setToolTip(
                tr('점을 드래그하거나 안 보임으로 삭제하면 해당 프레임을 즉시 Ground Truth 후보로 승인합니다. 수정과 프레임 승인을 분리하고 싶으면 끄세요.')
            )
            work_side.addWidget(self.auto_confirm_edits)

        self.error_search_title = QLabel(
            tr('라벨 프레임 탐색') if self.labeling_mode else tr('오류 후보 탐색'),
            objectName="sectionTitle",
        )
        work_side.addWidget(self.error_search_title)
        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel(tr('낮은 신뢰도')))
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.01, 0.99)
        self.confidence.setSingleStep(0.05)
        self.confidence.setValue(0.60)
        self.confidence.valueChanged.connect(self.recompute_flags)
        conf_row.addWidget(self.confidence)
        work_side.addLayout(conf_row)
        jump_row = QHBoxLayout()
        jump_row.addWidget(QLabel(tr('프레임 점프(px)')))
        self.jump_threshold = QDoubleSpinBox()
        self.jump_threshold.setRange(1, 500)
        self.jump_threshold.setValue(25)
        self.jump_threshold.valueChanged.connect(self.recompute_flags)
        jump_row.addWidget(self.jump_threshold)
        work_side.addLayout(jump_row)
        self.include_structural_flags = QCheckBox(tr('구조·경계 이상 포함'))
        self.include_structural_flags.setChecked(True)
        self.include_structural_flags.setToolTip(
            tr('영상 밖 좌표, 공통 움직임에서 혼자 벗어난 점, 순간 이탈 후 복귀, 신체점 배열 붕괴와 교환 의심을 함께 찾습니다. 신체점 이름을 사용하지 않으며 좌표를 자동으로 수정하지 않습니다.')
        )
        self.include_structural_flags.toggled.connect(self.recompute_flags)
        work_side.addWidget(self.include_structural_flags)
        self.auto_pause = QCheckBox(tr('재생 중 오류 후보에서 자동 정지'))
        self.auto_pause.setChecked(True)
        work_side.addWidget(self.auto_pause)
        flag_row = QHBoxLayout()
        previous_flag = QPushButton(tr('이전 라벨 (Q)') if self.labeling_mode else tr('이전 오류 (Q)'))
        previous_flag.clicked.connect(lambda: self.goto_label_frame(-1) if self.labeling_mode else self.goto_flag(-1))
        next_flag = QPushButton(tr('다음 라벨 (E)') if self.labeling_mode else tr('다음 오류 (E)'))
        next_flag.clicked.connect(lambda: self.goto_label_frame(1) if self.labeling_mode else self.goto_flag(1))
        flag_row.addWidget(previous_flag)
        flag_row.addWidget(next_flag)
        work_side.addLayout(flag_row)

        self.frame_state = QLabel("-")
        self.frame_state.setWordWrap(True)
        self.frame_state.setObjectName("infoBox")
        work_side.addWidget(self.frame_state)
        work_side.addStretch(1)
        if not self.review_only and not self.labeling_mode:
            temporal_tab = QWidget()
            temporal_side = QVBoxLayout(temporal_tab)
            temporal_side.setContentsMargins(8, 10, 8, 8)
            temporal_side.setSpacing(6)
            self.review_tabs.addTab(temporal_tab, tr('시계열 GT'))
            self.episode_summary = QLabel(tr('지정된 구간 없음'), objectName="infoBox")
            self.episode_summary.setWordWrap(True)
            temporal_side.addWidget(self.episode_summary)
            self.episode_list = QTableWidget(0, 3, objectName="episodeTable")
            self.episode_list.setHorizontalHeaderLabels([tr('구간'), tr('시간 범위'), tr('검증')])
            self.episode_list.verticalHeader().setVisible(False)
            self.episode_list.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.episode_list.setSelectionMode(QAbstractItemView.SingleSelection)
            self.episode_list.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.episode_list.setAlternatingRowColors(True)
            self.episode_list.setShowGrid(False)
            self.episode_list.setWordWrap(False)
            self.episode_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.episode_list.setMinimumHeight(145)
            self.episode_list.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
            self.episode_list.verticalHeader().setDefaultSectionSize(36)
            episode_header = self.episode_list.horizontalHeader()
            episode_header.setHighlightSections(False)
            episode_header.setSectionResizeMode(QHeaderView.Interactive)
            episode_header.setMinimumSectionSize(44)
            episode_header.setStretchLastSection(True)
            self.episode_list.setColumnWidth(0, 50)
            self.episode_list.setColumnWidth(1, 190)
            self.episode_list.setColumnWidth(2, 82)
            self.episode_list.currentCellChanged.connect(
                lambda row, _column, _previous_row, _previous_column: self.select_episode_row(row)
            )
            self.episode_list.cellClicked.connect(
                lambda row, _column: self.select_episode_row(row)
            )
            self.episode_list.setContextMenuPolicy(Qt.CustomContextMenu)
            self.episode_list.customContextMenuRequested.connect(
                self.show_episode_context_menu
            )
            temporal_side.addWidget(self.episode_list, 1)
            episode_range_tools = QHBoxLayout()
            episode_start = QPushButton(tr('시작 (S)'), objectName="toolButton")
            episode_start.setToolTip(tr('현재 프레임을 연속 오류 episode의 시작으로 지정합니다.'))
            episode_start.clicked.connect(self.start_temporal_episode)
            episode_end = QPushButton(tr('끝 (F)'), objectName="toolButton")
            episode_end.setToolTip(
                tr('현재 프레임까지를 episode로 저장하고 영상 시간 기준 0.1초 간격 GT 후보를 만듭니다.')
            )
            episode_end.clicked.connect(self.finish_temporal_episode)
            self.toggle_gt_candidate_button = QPushButton(tr('GT 추가 (G)'), objectName="toolButton")
            self.toggle_gt_candidate_button.setToolTip(
                tr('선택한 episode 안에서 현재 프레임을 GT 후보로 추가하거나 후보에서 제외합니다. 시작·끝 프레임은 항상 유지됩니다.')
            )
            self.toggle_gt_candidate_button.clicked.connect(self.toggle_current_gt_candidate)
            episode_range_tools.addWidget(episode_start)
            episode_range_tools.addWidget(episode_end)
            episode_range_tools.addWidget(self.toggle_gt_candidate_button)
            temporal_side.addLayout(episode_range_tools)
            episode_navigation = QHBoxLayout()
            previous_gt = QPushButton(tr('이전 (Z)'), objectName="toolButton")
            previous_gt.clicked.connect(lambda: self.goto_gt_candidate(-1))
            next_gt = QPushButton(tr('다음 (C)'), objectName="toolButton")
            next_gt.clicked.connect(lambda: self.goto_gt_candidate(1))
            self.confirm_gt_button = QPushButton(tr('검증 완료 (Enter)'), objectName="primaryButton")
            self.confirm_gt_button.setToolTip(
                tr('현재 GT 후보를 사람 검증 완료로 표시합니다. 이미 완료한 후보에서는 검증을 취소합니다.')
            )
            self.confirm_gt_button.clicked.connect(self.toggle_confirmed)
            episode_navigation.addWidget(previous_gt)
            episode_navigation.addWidget(next_gt)
            episode_navigation.addWidget(self.confirm_gt_button)
            temporal_side.addLayout(episode_navigation)
            episode_boundary_tools = QHBoxLayout()
            edit_episode_start = QPushButton(tr('시작 수정 (Shift+S)'), objectName="toolButton")
            edit_episode_start.setToolTip(
                tr('선택한 시계열 GT 구간의 시작을 현재 프레임으로 변경합니다. 범위 안의 기존 GT 후보와 검증 상태는 유지됩니다.')
            )
            edit_episode_start.clicked.connect(
                lambda: self.update_selected_episode_boundary("start")
            )
            edit_episode_end = QPushButton(tr('끝 수정 (Shift+F)'), objectName="toolButton")
            edit_episode_end.setToolTip(
                tr('선택한 시계열 GT 구간의 끝을 현재 프레임으로 변경합니다. 새로 확장된 범위에는 설정된 시간 간격으로 GT 후보가 추가됩니다.')
            )
            edit_episode_end.clicked.connect(
                lambda: self.update_selected_episode_boundary("end")
            )
            episode_boundary_tools.addWidget(edit_episode_start)
            episode_boundary_tools.addWidget(edit_episode_end)
            temporal_side.addLayout(episode_boundary_tools)
            remove_episode = QPushButton(tr('선택 구간 삭제'), objectName="dangerButton")
            remove_episode.setToolTip(
                tr('선택한 episode 지정만 삭제합니다. 이미 수정한 점과 프레임 검증 상태는 유지됩니다.')
            )
            remove_episode.clicked.connect(self.remove_selected_episode)
            exclude = QPushButton(tr('프레임 전체 제외'), objectName="dangerButton")
            exclude.clicked.connect(self.toggle_excluded)
            episode_destructive = QHBoxLayout()
            episode_destructive.addWidget(remove_episode)
            episode_destructive.addWidget(exclude)
            temporal_side.addLayout(episode_destructive)
        panel_scroll.setWidget(panel)
        # Translation can make the controls wider than the previous 350px pane.
        # Honor their layout minimum so disabling horizontal scrolling never
        # hides the right-hand actions or tabs.
        self.review_side_scroll = panel_scroll
        self.review_side_panel = panel
        self.review_splitter = splitter
        splitter.addWidget(panel_scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([250, 900, 350])
        content_layout.addWidget(splitter, 1)

        controls = QHBoxLayout()
        self.play_button = QPushButton(tr('▶ 재생 (Space)'))
        self.play_button.clicked.connect(self.toggle_play)
        controls.addWidget(self.play_button)
        back_ten = QPushButton("-10F (Shift+A)")
        back_ten.clicked.connect(lambda: self.step(-10))
        controls.addWidget(back_ten)
        back = QPushButton("-1F (A)")
        back.clicked.connect(lambda: self.step(-1))
        controls.addWidget(back)
        forward = QPushButton("+1F (D)")
        forward.clicked.connect(lambda: self.step(1))
        controls.addWidget(forward)
        forward_ten = QPushButton("+10F (Shift+D)")
        forward_ten.clicked.connect(lambda: self.step(10))
        controls.addWidget(forward_ten)
        self.playback_control = PlaybackSpeedControl(self)
        self.speed_combo = self.playback_control.combo
        self.playback_control.speedChanged.connect(self.set_playback_speed)
        controls.addWidget(self.playback_control)
        self.timeline = ClickSeekSlider(Qt.Horizontal)
        self.timeline.setRange(0, self.frame_count - 1)
        self.timeline.setToolTip(tr('바의 원하는 위치를 클릭하거나 손잡이를 드래그해 프레임을 이동합니다.'))
        self.timeline.seek_requested.connect(self.show_frame)
        self.timeline.sliderMoved.connect(self.show_frame)
        timeline_row = QHBoxLayout()
        timeline_row.addWidget(self.timeline, 1)
        self.time_label = QLabel("00:00.000 / 00:00.000")
        timeline_row.addWidget(self.time_label)
        content_layout.addLayout(timeline_row)
        controls.addStretch(1)
        content_layout.addLayout(controls)

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.setInterval(playback_interval_ms(self.fps, self.playback_speed))
        self.timer.timeout.connect(lambda: self.step(1, playback=True))
        self.install_shortcuts()
        self.apply_style()
        # Run after derived comparison windows add their controls and after the
        # stylesheet has contributed its padding to each widget's size hint.
        QTimer.singleShot(0, self.fit_review_sidebar)

    def fit_review_sidebar(self) -> None:
        panel = self.review_side_panel
        panel.ensurePolished()
        panel.layout().activate()
        scrollbar_width = self.review_side_scroll.verticalScrollBar().sizeHint().width()
        side_width = max(350, min(540, panel.minimumSizeHint().width() + scrollbar_width + 8))
        self.review_side_scroll.setMinimumWidth(side_width)
        self.review_side_scroll.setMaximumWidth(max(430, side_width))
        sizes = self.review_splitter.sizes()
        if len(sizes) == 3:
            self.review_splitter.setSizes([sizes[0], max(360, sizes[1]), side_width])

    def install_shortcuts(self) -> None:
        shortcuts = [
            ("Space", self.toggle_play),
            ("A", lambda: self.step(-1)),
            ("D", lambda: self.step(1)),
            ("Shift+A", lambda: self.step(-10)),
            ("Shift+D", lambda: self.step(10)),
            ("Q", lambda: self.goto_label_frame(-1) if self.labeling_mode else self.goto_flag(-1)),
            ("E", lambda: self.goto_label_frame(1) if self.labeling_mode else self.goto_flag(1)),
            ("V", lambda: self.show_points.setChecked(not self.show_points.isChecked())),
            ("Ctrl+0", self.canvas.fit_to_view),
            ("[", lambda: self.change_point_size(-1)),
            ("]", lambda: self.change_point_size(1)),
        ]
        if not self.review_only:
            shortcuts.extend([
                ("X", self.mark_selected_invisible),
                ("Delete", self.mark_selected_invisible),
                ("Backspace", self.mark_selected_invisible),
                ("Return", self.toggle_confirmed),
                ("Ctrl+S", self.save_session),
            ])
            if not self.labeling_mode:
                shortcuts.extend([
                    ("S", self.start_temporal_episode),
                    ("F", self.finish_temporal_episode),
                    ("Shift+S", lambda: self.update_selected_episode_boundary("start")),
                    ("Shift+F", lambda: self.update_selected_episode_boundary("end")),
                    ("Z", lambda: self.goto_gt_candidate(-1)),
                    ("C", lambda: self.goto_gt_candidate(1)),
                    ("G", self.toggle_current_gt_candidate),
                ])
        for key, callback in shortcuts:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(callback)
        for index in range(9):
            shortcut = QShortcut(QKeySequence(str(index + 1)), self)
            shortcut.activated.connect(lambda i=index: self.bodypart_list.setCurrentRow(i))

    def sync_playlist_selection(self) -> None:
        if not hasattr(self, "playlist_widget"):
            return
        self.playlist_widget.blockSignals(True)
        self.playlist_widget.setCurrentRow(self.playlist_index)
        self.playlist_widget.blockSignals(False)

    def switch_playlist_row(self, index: int) -> None:
        if index == self.playlist_index or not 0 <= index < len(self.playlist):
            return
        if not self.review_only and self.dirty:
            answer = QMessageBox.question(
                self,
                tr('변경 내용 저장'),
                tr('현재 영상의 저장하지 않은 교정 내용이 있습니다.'),
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if answer == QMessageBox.Cancel:
                self.sync_playlist_selection()
                return
            if answer == QMessageBox.Save:
                if self.save_session() is False:
                    self.sync_playlist_selection()
                    return

        entry = self.playlist[index]
        video = Path(str(entry.get("video", ""))).expanduser().resolve()
        pose_value = str(entry.get("pose", "") or "")
        pose_path = Path(pose_value).expanduser().resolve() if pose_value else None
        detection_value = str(entry.get("detections", "") or "")
        detection_path = Path(detection_value).expanduser().resolve() if detection_value else None
        output_value = str(entry.get("output", "") or "")
        output = Path(output_value).expanduser().resolve() if output_value else None
        new_capture = cv2.VideoCapture(str(video))
        if not new_capture.isOpened():
            QMessageBox.warning(self, tr('영상 전환 실패'), tr('영상을 열 수 없습니다.\n\n{p0}', p0=video))
            new_capture.release()
            self.sync_playlist_selection()
            return
        try:
            new_pose, new_bodyparts = read_pose(pose_path) if pose_path else (None, ["nose", "head_center", "body_center", "tail_base"])
            new_detections = read_detections(detection_path)
            new_draft, new_source = None, None
            if entry.get("draft_manifest") and not self.review_only:
                from hagisilta_review_baseline import load_review_draft
                new_draft, new_source = load_review_draft(video, pose_path, entry["draft_manifest"])
        except Exception as error:
            new_capture.release()
            QMessageBox.warning(self, tr('예측 좌표 전환 실패'), str(error))
            self.sync_playlist_selection()
            return

        self.timer.stop()
        self.capture.release()
        self.capture = new_capture
        self.video = video
        self.pose_path = pose_path
        self.output = (
            output.resolve() if output else
            (self.video.parent if self.review_only else self.default_output()).resolve()
        )
        if not self.review_only:
            self.output.mkdir(parents=True, exist_ok=True)
        self.pose = new_pose
        self.draft_pose, self.draft_source = new_draft, new_source
        self.frame_baselines = {}
        self.previously_reviewed = set()
        self.detection_path = detection_path
        self.detections = new_detections
        self.bodyparts = list(new_bodyparts)
        self.selected_bodypart = self.bodyparts[0]
        self.visible_bodyparts = set(self.bodyparts)
        self.frame_count = max(1, int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS)) or 10.0
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.pose_frames = min(len(self.pose), self.frame_count) if self.pose is not None else 0
        self.label_frame_indices = sorted({
            int(value)
            for value in (entry.get("label_frames", []) or [])
            if 0 <= int(value) < self.frame_count
        })
        self.label_frame_set = set(self.label_frame_indices)
        self.labeling_mode = bool(self.label_frame_indices) and not self.review_only
        self.frame_index = 0
        self.edits = {}
        self.confirmed = set()
        self.excluded = set()
        self.temporal_episodes = []
        self.episode_draft_start = None
        self.selected_episode_id = ""
        self.dirty = False
        self.flagged = []
        self.flagged_set = set()
        self.flag_starts = []
        self.flag_start_set = set()
        self.flag_details = {}
        self.playlist_index = index

        self.show_detection_boxes.blockSignals(True)
        self.show_detection_boxes.setEnabled(bool(self.detections))
        self.show_detection_boxes.setChecked(
            bool(self.detections) and bool(entry.get("show_bboxes", True))
        )
        self.show_detection_boxes.blockSignals(False)

        self.bodypart_list.blockSignals(True)
        self.bodypart_list.clear()
        for bodypart_index, bodypart in enumerate(self.bodyparts):
            item = QListWidgetItem(f"{bodypart_index + 1}   {bodypart}")
            item.setData(Qt.UserRole, bodypart)
            if self.review_only:
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
            self.bodypart_list.addItem(item)
        self.bodypart_list.setCurrentRow(0)
        self.bodypart_list.blockSignals(False)
        self.timeline.setRange(0, self.frame_count - 1)
        self.timer.setInterval(playback_interval_ms(self.fps, self.playback_speed))
        self.video_meta.setText(tr('영상 · {p0}', p0=self.video.name))
        self.video_meta.setToolTip(str(self.video))
        self.pose_meta.setText(tr('예측 · {p0}', p0=self.pose_path.name if self.pose_path else tr('없음')))
        self.pose_meta.setToolTip(str(self.pose_path) if self.pose_path else tr('예측 좌표 없음'))
        self.setWindowTitle(
            f"HAGISILTA · {self.mode_title()} · {self.video.name}"
        )
        if not self.review_only:
            self.load_session()
            self.refresh_episode_list()
        self.update_draft_context()
        self.recompute_flags()
        self.canvas._fit_pending = True
        self.show_frame(self.label_frame_indices[0] if self.labeling_mode else 0)

    def apply_style(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background:#23242b; color:#f0f1f2;
                font-family:'Segoe UI','Malgun Gothic'; font-size:13px;
            }
            QMainWindow, QWidget#pageRoot, QWidget#pageContent { background:#23242b; }
            QFrame#pageHeader { background:#202125; border-bottom:1px solid #41434a; }
            QFrame#pageTitleAccent { background:#5b82e8; border:none; }
            QLabel#pageKicker {
                background:transparent; color:#7f98d8; font-size:9px;
                font-weight:700; letter-spacing:1px;
            }
            QLabel#pageTitle {
                background:transparent; color:#f2f3f5; font-size:18px; font-weight:650;
            }
            QFrame#contextBar { background:#292a2f; border-bottom:1px solid #41434a; }
            QFrame#pageFooter { background:#202125; border-top:1px solid #41434a; }
            QLabel#footerStatus { background:transparent; color:#aeb2b9; }
            QFrame#pageFooter QPushButton#primaryButton { min-width:220px; }
            QLabel#contextGuide { background:transparent; color:#8cd5b8; font-weight:600; }
            QLabel#contextMeta {
                background:#232429; color:#c2c5ca; border:1px solid #3f4147;
                border-radius:2px; padding:4px 8px;
            }
            QToolButton, QPushButton {
                min-height:32px; background:#35363b; color:#eceef1;
                border:1px solid #55585f; border-radius:2px; padding:0 12px;
                font-weight:600;
            }
            QToolButton:hover, QPushButton:hover { background:#42444a; border-color:#70747d; }
            QToolButton::menu-indicator { subcontrol-position:right center; right:7px; }
            QMenu { background:#292a2f; border:1px solid #55585f; padding:5px; }
            QMenu::item { padding:8px 24px 8px 10px; }
            QMenu::item:selected { background:#3f6dcc; }
            QPushButton#primaryButton { background:#3f6dcc; border-color:#5c82d8; color:#ffffff; }
            QPushButton#primaryButton:hover { background:#4a78d8; border-color:#7194df; }
            QPushButton#dangerButton { background:#592d37; border-color:#8d4a5b; }
            QSplitter::handle { background:#41434a; width:1px; }
            QLabel#sectionTitle {
                background:#303136; color:#f0f1f2; border:1px solid #41434a;
                border-radius:2px; font-size:13px; font-weight:650; padding:7px 9px;
                margin-top:6px;
            }
            QLabel#infoBox {
                background:#232429; border:1px solid #3f4147;
                border-radius:2px; padding:8px 9px;
            }
            QListWidget {
                background:#202125; border:1px solid #3f4147;
                border-radius:2px; padding:3px;
            }
            QListWidget::item { padding:8px; border-radius:2px; }
            QListWidget::item:selected { background:#496bd0; color:#ffffff; }
            QGraphicsView { background:#101114; border:1px solid #3f4147; }
            QSlider::groove:horizontal { height:6px; background:#3b3d43; border-radius:3px; }
            QSlider::handle:horizontal { width:16px; margin:-5px 0; background:#63cbb0; border-radius:8px; }
            QDoubleSpinBox {
                min-height:30px; background:#202125; border:1px solid #4b4e55;
                border-radius:2px; padding:0 7px;
            }
            QCheckBox { background:transparent; spacing:7px; }
            QTabWidget#reviewToolTabs::pane {
                background:#24262b; border:1px solid #454952; top:-1px;
            }
            QTabWidget#reviewToolTabs QTabBar::tab {
                min-width:116px; min-height:32px; padding:0 16px;
                background:transparent; color:#aeb1b7; border:none;
                border-bottom:2px solid transparent; font-weight:550;
            }
            QTabWidget#reviewToolTabs QTabBar::tab:hover {
                background:#2c2d32; color:#ffffff;
            }
            QTabWidget#reviewToolTabs QTabBar::tab:selected {
                background:#292a2f; color:#ffffff;
                border-bottom:2px solid #5b82e8; font-weight:650;
            }
            QTableWidget#episodeTable {
                background:#202125; alternate-background-color:#25272d;
                border:1px solid #3f4147; selection-background-color:#496bd0;
                selection-color:#ffffff; outline:0;
            }
            QTableWidget#episodeTable::item { padding:0 6px; border:0; }
            QTableWidget#episodeTable::item:hover { background:#30343c; }
            QTableWidget#episodeTable::item:selected,
            QTableWidget#episodeTable::item:selected:hover {
                background:#3f67c1; color:#ffffff;
            }
            QTableWidget#episodeTable QHeaderView::section {
                min-height:30px; background:#303136; color:#c9ccd2;
                border:0; border-right:1px solid #41434a;
                border-bottom:1px solid #41434a; padding:0 6px; font-weight:650;
            }
        """)
        self.setStyleSheet(self.styleSheet() + hagisilta_control_stylesheet())

    def current_prediction(self, bodypart: str, frame_index: int | None = None) -> dict:
        frame_index = self.frame_index if frame_index is None else frame_index
        return self.prediction_from(self.pose, bodypart, frame_index)

    def prediction_from(self, pose, bodypart, frame_index):
        if pose is None or not 0 <= frame_index < min(len(pose), self.frame_count):
            return {"x": math.nan, "y": math.nan, "likelihood": math.nan, "visible": False}
        x = float(pose_column(pose, bodypart, "x").iloc[frame_index])
        y = float(pose_column(pose, bodypart, "y").iloc[frame_index])
        likelihood_series = pose_column(pose, bodypart, "likelihood")
        likelihood = float(likelihood_series.iloc[frame_index]) if not likelihood_series.isna().all() else 1.0
        visible = bool(
            np.isfinite(x)
            and np.isfinite(y)
            and np.isfinite(likelihood)
            and likelihood >= 0.0
            and 0.0 <= x < self.width
            and 0.0 <= y < self.height
        )
        return {"x": x, "y": y, "likelihood": likelihood, "visible": visible}

    def baseline_point(self, bodypart, frame_index):
        snapshot = self.frame_baselines.get(frame_index)
        if snapshot is not None:
            return dict(snapshot["points"][bodypart])
        if self.draft_pose is not None and frame_index not in self.previously_reviewed:
            return self.prediction_from(self.draft_pose, bodypart, frame_index)
        return self.current_prediction(bodypart, frame_index)

    def freeze_frame_baseline(self, frame_index):
        if self.review_only or frame_index in self.frame_baselines:
            return
        self.frame_baselines[frame_index] = {
            "points": {part: self.baseline_point(part, frame_index) for part in self.bodyparts},
            "source": dict(self.draft_source) if self.draft_source and frame_index not in self.previously_reviewed else
                      {"kind": "raw", "label": "Unfiltered", "source_pose": str(self.pose_path)},
        }

    def update_draft_context(self):
        if self.review_only or self.labeling_mode:
            return
        label = review_source_label(self.draft_source) if self.draft_source else tr('보정 전 좌표')
        self.pose_meta.setText(tr('교정 시작 · ') + label)
        self.pose_meta.setToolTip(tr('새 프레임의 시작 좌표입니다. 이미 저장한 교정 프레임의 좌표는 유지합니다.\n') +
                                 (str(self.draft_source) if self.draft_source else str(self.pose_path)))
        self.footer_status.setText(tr('기존 교정 유지 · 교정 저장 후 ‘새 수정 프레임 반영’으로 학습에 추가'))

    def reviewed_pose(self, include_edits=False):
        """The visible baseline, including snapshots of previously reviewed frames."""
        result = (self.draft_pose if self.draft_pose is not None else self.pose).copy().astype(float)
        for frame in self.previously_reviewed:
            if 0 <= frame < len(result):
                result.iloc[frame] = self.pose.iloc[frame]
        for frame, record in self.frame_baselines.items():
            if not 0 <= frame < len(result):
                continue
            for column in result.columns:
                part, coord = str(column[-2]), str(column[-1])
                point = record["points"].get(part, {})
                if coord in point:
                    result.at[result.index[frame], column] = point[coord]
        if include_edits:
            for frame, points in self.edits.items():
                if not 0 <= frame < len(result):
                    continue
                for column in result.columns:
                    point = points.get(str(column[-2]))
                    if point is not None and column[-1] in ("x", "y", "likelihood"):
                        coord = column[-1]
                        value = (1.0 if point.get("visible") else 0.0) if coord == "likelihood" else (
                            point.get(coord, math.nan) if point.get("visible") else math.nan)
                        result.at[result.index[frame], column] = value
        return result

    def current_point(self, bodypart: str, frame_index: int | None = None) -> dict:
        frame_index = self.frame_index if frame_index is None else frame_index
        edit = self.edits.get(frame_index, {}).get(bodypart)
        return dict(edit) if edit is not None else self.baseline_point(bodypart, frame_index)

    def show_frame(self, index: int) -> None:
        index = max(0, min(int(index), self.frame_count - 1))
        self.frame_index = index
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = self.capture.read()
        if not ok:
            self.statusBar().showMessage(tr('프레임 {p0} 읽기 실패', p0=index))
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
        self.canvas.set_frame(image)
        self.timeline.blockSignals(True)
        self.timeline.setValue(index)
        self.timeline.blockSignals(False)
        self.draw_pose()
        self.update_labels()

    def clear_overlay(self) -> None:
        for item in self.overlay_items:
            self.canvas.scene.removeItem(item)
        self.overlay_items.clear()
        self.markers.clear()

    def draw_pose(self) -> None:
        self.clear_overlay()
        if self.show_detection_boxes.isChecked() and self.frame_index < len(self.detections):
            record = self.detections[self.frame_index]
            boxes = record.get("bboxes", []) if isinstance(record, dict) else []
            scores = record.get("bbox_scores", []) if isinstance(record, dict) else []
            for box_index, box in enumerate(boxes if isinstance(boxes, list) else []):
                if not isinstance(box, (list, tuple)) or len(box) < 4:
                    continue
                x, y, width, height = map(float, box[:4])
                if not all(np.isfinite(value) for value in (x, y, width, height)) or width <= 0 or height <= 0:
                    continue
                rectangle = QGraphicsRectItem(x, y, width, height)
                pen = QPen(QColor("#ff5f67"), 2)
                pen.setCosmetic(True)
                rectangle.setPen(pen)
                rectangle.setZValue(8)
                self.canvas.scene.addItem(rectangle)
                self.overlay_items.append(rectangle)
                score = float(scores[box_index]) if isinstance(scores, list) and box_index < len(scores) else math.nan
                if np.isfinite(score) and score >= 0:
                    label = QGraphicsSimpleTextItem(f"Detector {score:.2f}")
                    label.setBrush(QColor("#ff7b82"))
                    label.setPos(QPointF(x + 3, max(0.0, y - 19)))
                    label.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
                    label.setZValue(9)
                    self.canvas.scene.addItem(label)
                    self.overlay_items.append(label)
        if not self.show_points.isChecked():
            return
        visible_points: list[tuple[str, QPointF, QColor]] = []
        for index, bodypart in enumerate(self.bodyparts):
            if self.review_only and bodypart not in self.visible_bodyparts:
                continue
            point = self.current_point(bodypart)
            if not point.get("visible") or not np.isfinite(point.get("x", np.nan)) or not np.isfinite(point.get("y", np.nan)):
                continue
            color = COLORS[index % len(COLORS)]
            position = QPointF(float(point["x"]), float(point["y"]))
            marker = PosePointItem(
                bodypart,
                color,
                self.point_size.value(),
                self.point_moved,
                self.select_bodypart_from_canvas,
                movable=not self.review_only,
            )
            marker._loading = True
            marker.setPos(position)
            marker._loading = False
            if bodypart == self.selected_bodypart:
                selected_size = self.point_size.value() + 4
                selected_radius = selected_size / 2.0
                marker.setRect(QRectF(-selected_radius, -selected_radius, selected_size, selected_size))
                marker.setPen(QPen(QColor("white"), 3))
            likelihood = float(point.get("likelihood", 1.0))
            if np.isfinite(likelihood) and likelihood < self.confidence.value():
                marker.setOpacity(0.55)
                marker.setPen(QPen(QColor("#ff445f"), 3, Qt.DashLine))
            self.canvas.scene.addItem(marker)
            self.overlay_items.append(marker)
            self.markers[bodypart] = marker
            if self.show_names.isChecked():
                label = QGraphicsSimpleTextItem(bodypart)
                label.setBrush(color)
                label.setPos(position + QPointF(7, -18))
                label.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
                label.setZValue(21)
                self.canvas.scene.addItem(label)
                self.overlay_items.append(label)
            visible_points.append((bodypart, position, color))
        if self.show_names.isChecked():
            for first, second in zip(visible_points, visible_points[1:]):
                line = QGraphicsLineItem(first[1].x(), first[1].y(), second[1].x(), second[1].y())
                line.setPen(QPen(QColor(235, 240, 255, 170), 2))
                line.setZValue(10)
                self.canvas.scene.addItem(line)
                self.overlay_items.append(line)

    def bodypart_visibility_changed(self, item: QListWidgetItem) -> None:
        if not self.review_only:
            return
        bodypart = str(item.data(Qt.UserRole) or "")
        if not bodypart:
            return
        if item.checkState() == Qt.Checked:
            self.visible_bodyparts.add(bodypart)
        else:
            self.visible_bodyparts.discard(bodypart)
        self.draw_pose()
        self.update_labels()

    def set_all_bodyparts_visible(self, visible: bool) -> None:
        if not self.review_only:
            return
        self.bodypart_list.blockSignals(True)
        for row in range(self.bodypart_list.count()):
            self.bodypart_list.item(row).setCheckState(Qt.Checked if visible else Qt.Unchecked)
        self.bodypart_list.blockSignals(False)
        self.visible_bodyparts = set(self.bodyparts) if visible else set()
        self.draw_pose()
        self.update_labels()

    def point_size_changed(self, value: int) -> None:
        self.point_size_label.setText(f"{value} px")
        self.draw_pose()

    def change_point_size(self, amount: int) -> None:
        self.point_size.setValue(self.point_size.value() + amount)

    def point_moved(self, bodypart: str, position: QPointF) -> None:
        if not self.require_editable_frame():
            return
        self.freeze_frame_baseline(self.frame_index)
        x = max(0.0, min(position.x(), self.width - 1.0))
        y = max(0.0, min(position.y(), self.height - 1.0))
        original = self.current_prediction(bodypart)
        self.edits.setdefault(self.frame_index, {})[bodypart] = {
            "x": x,
            "y": y,
            "likelihood": float(original.get("likelihood", 1.0)),
            "visible": True,
            "source": "human",
        }
        self.confirm_after_edit()
        self.dirty = True
        self.update_labels()

    def place_selected_point(self, x: float, y: float) -> None:
        if not self.require_editable_frame():
            return
        self.point_moved(self.selected_bodypart, QPointF(x, y))
        self.draw_pose()

    def select_bodypart(self, bodypart: str) -> None:
        self.selected_bodypart = bodypart
        self.bodypart_list.setCurrentRow(self.bodyparts.index(bodypart))
        self.draw_pose()

    def select_bodypart_from_canvas(self, bodypart: str) -> None:
        """Select a marker without rebuilding the scene during its mouse press.

        Rebuilding in ``mousePressEvent`` deleted the item before Qt could deliver
        move events, so the marker looked selectable but could never be dragged.
        """
        if bodypart not in self.bodyparts:
            return
        self.selected_bodypart = bodypart
        self.bodypart_list.blockSignals(True)
        self.bodypart_list.setCurrentRow(self.bodyparts.index(bodypart))
        self.bodypart_list.blockSignals(False)
        self.update_labels()

    def select_bodypart_index(self, index: int) -> None:
        if 0 <= index < len(self.bodyparts):
            self.selected_bodypart = self.bodyparts[index]
            self.draw_pose()

    def mark_selected_invisible(self) -> None:
        if not self.require_editable_frame():
            return
        self.freeze_frame_baseline(self.frame_index)
        original = self.current_prediction(self.selected_bodypart)
        self.edits.setdefault(self.frame_index, {})[self.selected_bodypart] = {
            "x": math.nan,
            "y": math.nan,
            "likelihood": float(original.get("likelihood", math.nan)),
            "visible": False,
            "source": "human_missing",
        }
        self.confirm_after_edit()
        self.dirty = True
        self.draw_pose()
        self.update_labels()

    def confirm_after_edit(self) -> None:
        if self.review_only:
            return
        if self.auto_confirm_edits.isChecked():
            self.confirmed.add(self.frame_index)
            self.excluded.discard(self.frame_index)
            self.refresh_episode_list()

    def reset_selected_point(self) -> None:
        if not self.require_editable_frame():
            return
        frame_edits = self.edits.get(self.frame_index, {})
        frame_edits.pop(self.selected_bodypart, None)
        if not frame_edits:
            self.edits.pop(self.frame_index, None)
        self.dirty = True
        self.draw_pose()
        self.update_labels()

    def toggle_confirmed(self) -> None:
        if not self.require_editable_frame():
            return
        if self.frame_index in self.confirmed:
            self.confirmed.remove(self.frame_index)
        else:
            self.freeze_frame_baseline(self.frame_index)
            self.confirmed.add(self.frame_index)
            self.excluded.discard(self.frame_index)
        self.dirty = True
        self.refresh_episode_list()
        self.update_labels()

    def toggle_excluded(self) -> None:
        if not self.require_editable_frame():
            return
        if self.frame_index in self.excluded:
            self.excluded.remove(self.frame_index)
        else:
            self.excluded.add(self.frame_index)
            self.confirmed.discard(self.frame_index)
        self.dirty = True
        self.refresh_episode_list()
        self.update_labels()

    def episode_by_id(self, episode_id: str) -> dict | None:
        return next(
            (
                episode
                for episode in self.temporal_episodes
                if str(episode.get("episode_id") or "") == str(episode_id)
            ),
            None,
        )

    def current_temporal_episode(self) -> dict | None:
        selected = self.episode_by_id(self.selected_episode_id)
        if selected is not None:
            return selected
        return next(
            (
                episode
                for episode in self.temporal_episodes
                if int(episode["start_frame"]) <= self.frame_index <= int(episode["end_frame"])
            ),
            None,
        )

    def start_temporal_episode(self) -> None:
        if self.review_only:
            return
        if self.timer.isActive():
            self.toggle_play()
        if hasattr(self, "review_tabs"):
            self.review_tabs.setCurrentIndex(1)
        self.episode_draft_start = self.frame_index
        self.dirty = True
        self.update_episode_summary()
        self.statusBar().showMessage(
            tr('시계열 GT 구간 시작을 {p0}프레임으로 지정했습니다.', p0=self.frame_index + 1), 3000
        )

    def finish_temporal_episode(self) -> None:
        if self.review_only:
            return
        if hasattr(self, "review_tabs"):
            self.review_tabs.setCurrentIndex(1)
        if self.episode_draft_start is None:
            self.statusBar().showMessage(tr('먼저 구간 시작 (S)을 지정하세요.'), 3000)
            return
        start, end = sorted((self.episode_draft_start, self.frame_index))
        if episodes_overlap(self.temporal_episodes, start, end):
            QMessageBox.warning(
                self,
                tr('겹치는 GT 구간'),
                tr('현재 범위가 이미 저장된 시계열 GT 구간과 겹칩니다. 기존 구간을 삭제하거나 겹치지 않는 범위를 지정하세요.'),
            )
            return
        existing_ids = {str(episode.get("episode_id") or "") for episode in self.temporal_episodes}
        serial = len(self.temporal_episodes) + 1
        episode = make_episode(
            start,
            end,
            self.fps,
            serial,
            DEFAULT_GT_INTERVAL_SECONDS,
        )
        while episode["episode_id"] in existing_ids:
            serial += 1
            episode = make_episode(
                start,
                end,
                self.fps,
                serial,
                DEFAULT_GT_INTERVAL_SECONDS,
            )
        self.temporal_episodes.append(episode)
        self.temporal_episodes.sort(key=lambda value: (value["start_frame"], value["end_frame"]))
        self.selected_episode_id = episode["episode_id"]
        self.episode_draft_start = None
        self.dirty = True
        self.refresh_episode_list()
        self.show_frame(episode["gt_frames"][0])
        self.statusBar().showMessage(
            tr('시계열 GT 구간을 저장했습니다. GT 후보 {p0}개', p0=len(episode['gt_frames'])), 4000
        )

    def select_episode_row(self, row: int) -> None:
        if self.review_only or not hasattr(self, "episode_list") or row < 0:
            return
        item = self.episode_list.item(row, 0)
        episode_id = str(item.data(Qt.UserRole) or "") if item else ""
        episode = self.episode_by_id(episode_id)
        if episode is None:
            return
        self.selected_episode_id = episode_id
        candidates = list(map(int, episode.get("gt_frames", []) or []))
        if candidates:
            unfinished = [
                frame
                for frame in candidates
                if frame not in self.confirmed and frame not in self.excluded
            ]
            self.show_frame(unfinished[0] if unfinished else candidates[0])
        else:
            self.update_episode_summary()

    def select_episode_row_for_context(self, row: int) -> dict | None:
        if self.review_only or not hasattr(self, "episode_list") or row < 0:
            return None
        item = self.episode_list.item(row, 0)
        episode_id = str(item.data(Qt.UserRole) or "") if item else ""
        episode = self.episode_by_id(episode_id)
        if episode is None:
            return None
        self.selected_episode_id = episode_id
        self.episode_list.blockSignals(True)
        self.episode_list.setCurrentCell(row, 0)
        self.episode_list.selectRow(row)
        self.episode_list.blockSignals(False)
        self.update_episode_summary()
        return episode

    def episode_context_menu(self, row: int) -> QMenu | None:
        episode = self.select_episode_row_for_context(row)
        if episode is None:
            return None
        candidates = set(map(int, episode.get("gt_frames", []) or []))
        unfinished = candidates - self.confirmed - self.excluded
        confirmed = candidates & self.confirmed

        menu = QMenu(self.episode_list)
        goto_unfinished = menu.addAction(tr('첫 미검증 GT로 이동'))
        goto_unfinished.setEnabled(bool(unfinished))
        goto_unfinished.triggered.connect(self.goto_first_unverified_in_selected_episode)
        menu.addSeparator()
        complete_all = menu.addAction(tr('이 구간 GT 전체 검증 완료'))
        complete_all.setEnabled(bool(unfinished))
        complete_all.triggered.connect(
            lambda: self.set_selected_episode_confirmation(True)
        )
        cancel_all = menu.addAction(tr('이 구간 GT 전체 검증 취소'))
        cancel_all.setEnabled(bool(confirmed))
        cancel_all.triggered.connect(
            lambda: self.set_selected_episode_confirmation(False)
        )
        menu.addSeparator()
        edit_start = menu.addAction(tr('현재 프레임을 구간 시작으로'))
        edit_start.triggered.connect(
            lambda: self.update_selected_episode_boundary("start")
        )
        edit_end = menu.addAction(tr('현재 프레임을 구간 끝으로'))
        edit_end.triggered.connect(
            lambda: self.update_selected_episode_boundary("end")
        )
        menu.addSeparator()
        remove = menu.addAction(tr('선택 구간 삭제'))
        remove.triggered.connect(self.remove_selected_episode)
        return menu

    def show_episode_context_menu(self, position) -> None:
        if self.review_only or not hasattr(self, "episode_list"):
            return
        index = self.episode_list.indexAt(position)
        if not index.isValid():
            return
        menu = self.episode_context_menu(index.row())
        if menu is not None:
            menu.exec(self.episode_list.viewport().mapToGlobal(position))

    def goto_first_unverified_in_selected_episode(self) -> None:
        episode = self.episode_by_id(self.selected_episode_id)
        if episode is None:
            return
        unfinished = sorted(
            set(map(int, episode.get("gt_frames", []) or []))
            - self.confirmed
            - self.excluded
        )
        if not unfinished:
            self.statusBar().showMessage(tr('선택한 구간에 미검증 GT 후보가 없습니다.'), 3000)
            return
        self.show_frame(unfinished[0])

    def set_selected_episode_confirmation(self, completed: bool) -> None:
        if self.review_only:
            return
        episode = self.episode_by_id(self.selected_episode_id)
        if episode is None:
            self.statusBar().showMessage(tr('검증 상태를 바꿀 시계열 GT 구간을 선택하세요.'), 3000)
            return
        candidates = set(map(int, episode.get("gt_frames", []) or []))
        before = set(self.confirmed)
        if completed:
            targets = candidates - self.excluded
            for frame in targets:
                self.freeze_frame_baseline(frame)
            self.confirmed.update(targets)
            message = tr('선택 구간의 GT 후보 {p0}개를 검증 완료했습니다.', p0=len(targets))
            excluded_count = len(candidates & self.excluded)
            if excluded_count:
                message += tr(' 제외 프레임 {p0}개는 유지했습니다.', p0=excluded_count)
        else:
            self.confirmed.difference_update(candidates)
            message = tr('선택 구간의 GT 후보 {p0}개 검증을 취소했습니다.', p0=len(candidates & before))
        if self.confirmed != before:
            self.dirty = True
        self.refresh_episode_list()
        self.update_labels()
        self.statusBar().showMessage(message, 4000)

    def update_selected_episode_boundary(self, boundary: str) -> None:
        if self.review_only:
            return
        episode = self.episode_by_id(self.selected_episode_id)
        if episode is None:
            self.statusBar().showMessage(tr('범위를 수정할 시계열 GT 구간을 선택하세요.'), 3000)
            return
        if boundary not in {"start", "end"}:
            raise ValueError(tr('지원하지 않는 구간 경계입니다: {p0}', p0=boundary))
        if self.timer.isActive():
            self.toggle_play()

        old_start = int(episode["start_frame"])
        old_end = int(episode["end_frame"])
        new_start = self.frame_index if boundary == "start" else old_start
        new_end = self.frame_index if boundary == "end" else old_end
        if new_start > new_end:
            target = tr('끝') if boundary == "start" else tr('시작')
            self.statusBar().showMessage(
                tr('구간 시작은 끝보다 뒤일 수 없고, 끝은 시작보다 앞일 수 없습니다. 현재 {p0} 경계를 확인하세요.', p0=target),
                4000,
            )
            return

        other_episodes = [value for value in self.temporal_episodes if value is not episode]
        if episodes_overlap(other_episodes, new_start, new_end):
            QMessageBox.warning(
                self,
                tr('겹치는 GT 구간'),
                tr('수정한 범위가 다른 시계열 GT 구간과 겹칩니다. 겹치지 않는 프레임으로 이동한 뒤 다시 수정하세요.'),
            )
            return

        interval = float(
            episode.get("gt_interval_seconds", DEFAULT_GT_INTERVAL_SECONDS)
        )
        old_candidates = set(map(int, episode.get("gt_frames", []) or []))
        preserved = {
            frame for frame in old_candidates if new_start <= frame <= new_end
        }
        sampled = set(sample_gt_frames(new_start, new_end, self.fps, interval))
        newly_exposed = {
            frame for frame in sampled if frame < old_start or frame > old_end
        }
        candidates = preserved | newly_exposed | {new_start, new_end}

        episode.update(
            {
                "start_frame": new_start,
                "end_frame": new_end,
                "start_time_s": new_start / self.fps,
                "end_time_s": new_end / self.fps,
                "gt_frames": sorted(candidates),
                "updated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            }
        )
        self.temporal_episodes.sort(
            key=lambda value: (int(value["start_frame"]), int(value["end_frame"]))
        )
        self.dirty = True
        self.refresh_episode_list()
        self.show_frame(self.frame_index)
        boundary_label = tr('시작') if boundary == "start" else tr('끝')
        self.statusBar().showMessage(
            tr('선택 구간의 {p0}을 {p1}F로 수정했습니다. 현재 GT 후보 {p2}개', p0=boundary_label, p1=self.frame_index + 1, p2=len(candidates)),
            4000,
        )

    def refresh_episode_list(self) -> None:
        if self.review_only or not hasattr(self, "episode_list"):
            return
        self.episode_list.blockSignals(True)
        self.episode_list.clearContents()
        self.episode_list.setRowCount(len(self.temporal_episodes))
        selected_row = -1
        for index, episode in enumerate(self.temporal_episodes):
            candidates = set(map(int, episode.get("gt_frames", []) or []))
            verified = len((candidates & self.confirmed) - self.excluded)
            start_text = self.format_time(int(episode["start_frame"]) / self.fps)
            end_text = self.format_time(int(episode["end_frame"]) / self.fps)
            tooltip = (
                tr('프레임 {p0}–{p1}\nGT 간격 {p2:.3f}초', p0=int(episode['start_frame']) + 1, p1=int(episode['end_frame']) + 1, p2=float(episode.get('gt_interval_seconds', DEFAULT_GT_INTERVAL_SECONDS)))
            )
            values = (
                f"{index + 1:02d}",
                f"{start_text}–{end_text}",
                f"✓ {verified}/{len(candidates)}" if candidates and verified == len(candidates)
                else f"{verified}/{len(candidates)}",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, episode["episode_id"])
                item.setToolTip(tooltip)
                item.setTextAlignment(Qt.AlignCenter)
                if column == 2:
                    item.setForeground(
                        QColor("#79d7ad")
                        if candidates and verified == len(candidates)
                        else QColor("#f0c36b")
                    )
                self.episode_list.setItem(index, column, item)
            self.episode_list.setRowHeight(index, 36)
            if episode["episode_id"] == self.selected_episode_id:
                selected_row = index
        if selected_row < 0 and self.temporal_episodes:
            selected_row = 0
            self.selected_episode_id = str(self.temporal_episodes[0]["episode_id"])
        if selected_row >= 0:
            self.episode_list.setCurrentCell(selected_row, 0)
            self.episode_list.selectRow(selected_row)
            self.episode_list.scrollToItem(self.episode_list.item(selected_row, 0))
        else:
            self.episode_list.clearSelection()
        self.episode_list.blockSignals(False)
        self.update_episode_summary()

    def update_episode_summary(self) -> None:
        if self.review_only or not hasattr(self, "episode_summary"):
            return
        total_candidates = sum(
            len(set(map(int, episode.get("gt_frames", []) or [])))
            for episode in self.temporal_episodes
        )
        total_verified = sum(
            len(
                (set(map(int, episode.get("gt_frames", []) or [])) & self.confirmed)
                - self.excluded
            )
            for episode in self.temporal_episodes
        )
        overall = (
            tr('구간 {p0}개 · 전체 GT {p1}/{p2} 검증', p0=len(self.temporal_episodes), p1=total_verified, p2=total_candidates)
        )
        if self.episode_draft_start is not None:
            draft_frame_count = abs(self.frame_index - self.episode_draft_start) + 1
            self.episode_summary.setText(
                tr('{p0}\n지정 중 · 시작 {p1}F · 현재 {p2}F · {p3}프레임 · 끝에서 F', p0=overall, p1=self.episode_draft_start + 1, p2=self.frame_index + 1, p3=draft_frame_count)
            )
        else:
            episode = self.current_temporal_episode()
            if episode is None:
                self.episode_summary.setText(
                    tr('{p0}\n오류 시작에서 S, 종료에서 F를 누르세요.', p0=overall)
                )
            else:
                candidates = set(map(int, episode.get("gt_frames", []) or []))
                episode_start = int(episode["start_frame"])
                episode_end = int(episode["end_frame"])
                episode_frame_count = episode_end - episode_start + 1
                inside = episode_start <= self.frame_index <= episode_end
                review_state = (
                    tr('검증 완료')
                    if self.frame_index in self.confirmed
                    else tr('프레임 제외')
                    if self.frame_index in self.excluded
                    else tr('미검증')
                )
                current = (
                    tr('GT 후보 · {p0}', p0=review_state)
                    if self.frame_index in candidates
                    else tr('문맥 프레임 · {p0}', p0=review_state)
                    if inside
                    else tr('선택 구간 밖')
                )
                self.episode_summary.setText(
                    tr('{p0}\n선택 {p1}–{p2}F · {p3}프레임 · 현재 {p4}', p0=overall, p1=episode_start + 1, p2=episode_end + 1, p3=episode_frame_count, p4=current)
                )
        episode = self.current_temporal_episode()
        if hasattr(self, "toggle_gt_candidate_button"):
            candidates = set(map(int, episode.get("gt_frames", []) or [])) if episode else set()
            self.toggle_gt_candidate_button.setText(
                tr('GT 제외 (G)') if self.frame_index in candidates else tr('GT 추가 (G)')
            )
            self.toggle_gt_candidate_button.setEnabled(
                episode is not None
                and int(episode["start_frame"]) <= self.frame_index <= int(episode["end_frame"])
            )
        if hasattr(self, "confirm_gt_button"):
            confirmed = self.frame_index in self.confirmed
            self.confirm_gt_button.setText(
                tr('검증 취소 (Enter)') if confirmed else tr('검증 완료 (Enter)')
            )

    def goto_gt_candidate(self, direction: int) -> None:
        if self.review_only:
            return
        episode = self.current_temporal_episode()
        if episode is None:
            self.statusBar().showMessage(tr('먼저 시계열 GT 구간을 선택하세요.'), 2500)
            return
        candidates = sorted(map(int, episode.get("gt_frames", []) or []))
        targets = (
            [frame for frame in candidates if frame > self.frame_index]
            if direction > 0
            else [frame for frame in candidates if frame < self.frame_index]
        )
        if not targets:
            self.statusBar().showMessage(tr('해당 방향에 GT 후보가 없습니다.'), 2500)
            return
        self.show_frame(min(targets) if direction > 0 else max(targets))

    def toggle_current_gt_candidate(self) -> None:
        if self.review_only:
            return
        episode = self.current_temporal_episode()
        if episode is None:
            self.statusBar().showMessage(tr('먼저 시계열 GT 구간을 선택하세요.'), 2500)
            return
        if not int(episode["start_frame"]) <= self.frame_index <= int(episode["end_frame"]):
            self.statusBar().showMessage(tr('현재 프레임은 선택한 구간 밖입니다.'), 2500)
            return
        candidates = set(map(int, episode.get("gt_frames", []) or []))
        boundaries = {int(episode["start_frame"]), int(episode["end_frame"])}
        if self.frame_index in candidates:
            if self.frame_index in boundaries:
                self.statusBar().showMessage(tr('구간 시작·끝 프레임은 GT 후보에서 제외할 수 없습니다.'), 3000)
                return
            candidates.remove(self.frame_index)
        else:
            candidates.add(self.frame_index)
        episode["gt_frames"] = sorted(candidates)
        episode["updated_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
        self.dirty = True
        self.refresh_episode_list()
        self.update_labels()

    def remove_selected_episode(self) -> None:
        if self.review_only:
            return
        episode = self.episode_by_id(self.selected_episode_id)
        if episode is None:
            self.statusBar().showMessage(tr('삭제할 시계열 GT 구간을 선택하세요.'), 2500)
            return
        answer = QMessageBox.question(
            self,
            tr('시계열 GT 구간 삭제'),
            tr('선택한 구간 지정만 삭제할까요?\n\n이미 수정한 신체점과 프레임 검증 상태는 유지됩니다.'),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self.temporal_episodes = [
            value for value in self.temporal_episodes if value is not episode
        ]
        self.selected_episode_id = ""
        self.dirty = True
        self.refresh_episode_list()
        self.update_labels()

    def set_playback_speed(self, speed: float) -> None:
        self.playback_speed = speed
        self.timer.setInterval(playback_interval_ms(self.fps, self.playback_speed))

    def toggle_play(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText(tr('▶ 재생 (Space)'))
        else:
            self.timer.start()
            self.play_button.setText(tr('Ⅱ 일시정지 (Space)'))

    def step(self, amount: int, playback: bool = False) -> None:
        target = self.frame_index + amount
        if target >= self.frame_count:
            if self.timer.isActive():
                self.toggle_play()
            return
        target = max(0, target)
        self.show_frame(target)
        if playback and self.auto_pause.isChecked() and target in self.flag_start_set and target not in self.confirmed:
            self.toggle_play()
            self.statusBar().showMessage(tr('오류 후보에서 자동 정지했습니다.'), 3000)

    def recompute_flags(self) -> None:
        if self.labeling_mode:
            self.flagged, self.flagged_set = [], set()
            self.flag_starts, self.flag_start_set = [], set()
            self.flag_details = {}
            if hasattr(self, "frame_state"):
                self.update_labels()
            return
        if self.pose is None:
            self.flagged, self.flagged_set = [], set()
            self.flag_starts, self.flag_start_set = [], set()
            self.flag_details = {}
            return
        self.flag_details = detect_pose_anomalies(
            self.reviewed_pose().iloc[: self.pose_frames],
            self.bodyparts,
            self.width,
            self.height,
            self.confidence.value(),
            self.jump_threshold.value(),
            self.include_structural_flags.isChecked(),
        )
        self.flagged = sorted(self.flag_details)
        self.flagged_set = set(self.flagged)
        merge_gap = max(1, int(round(self.fps * 0.10)))
        self.flag_starts = [
            frame for index, frame in enumerate(self.flagged)
            if index == 0 or frame - self.flagged[index - 1] > merge_gap
        ]
        self.flag_start_set = set(self.flag_starts)
        if hasattr(self, "frame_state"):
            self.update_labels()

    def goto_flag(self, direction: int) -> None:
        candidates = [value for value in self.flag_starts if value > self.frame_index] if direction > 0 else [value for value in self.flag_starts if value < self.frame_index]
        if not candidates:
            self.statusBar().showMessage(tr('해당 방향에 오류 후보가 없습니다.'), 2500)
            return
        self.show_frame(min(candidates) if direction > 0 else max(candidates))

    def goto_label_frame(self, direction: int) -> None:
        candidates = (
            [value for value in self.label_frame_indices if value > self.frame_index]
            if direction > 0 else
            [value for value in self.label_frame_indices if value < self.frame_index]
        )
        if not candidates:
            self.statusBar().showMessage(tr('해당 방향에 라벨 프레임이 없습니다.'), 2500)
            return
        self.show_frame(min(candidates) if direction > 0 else max(candidates))

    def update_labels(self) -> None:
        seconds = self.frame_index / self.fps
        total_seconds = (self.frame_count - 1) / self.fps
        self.time_label.setText(f"{self.format_time(seconds)} / {self.format_time(total_seconds)}  ·  {self.frame_index + 1}/{self.frame_count}")
        point = self.current_point(self.selected_bodypart)
        likelihood = point.get("likelihood", math.nan)
        likelihood_text = f"{likelihood:.3f}" if np.isfinite(likelihood) else "-"
        source = (
            tr('사람 수정') if self.selected_bodypart in self.edits.get(self.frame_index, {}) else
            tr('기존 사람 라벨') if self.labeling_mode and self.frame_index in self.label_frame_set else
            tr('모델 예측')
        )
        if (self.selected_bodypart not in self.edits.get(self.frame_index, {})
                and not (self.labeling_mode and self.frame_index in self.label_frame_set)
                and not self.review_only):
            baseline_source = self.frame_baselines.get(self.frame_index, {}).get("source") or (
                self.draft_source if self.frame_index not in self.previously_reviewed else None)
            if baseline_source and baseline_source.get("kind") == "postprocessed":
                source = tr('후처리 초안 · ') + review_source_label(baseline_source)
        visibility = tr('보임') if point.get("visible") else (tr('좌표 없음') if self.review_only else tr('판독 불가'))
        display_state = (
            tr(' · 표시 숨김')
            if self.review_only and self.selected_bodypart not in self.visible_bodyparts
            else ""
        )
        self.point_status.setText(
            tr('선택: {p0}\n상태: {p1} · {p2}{p3}\n모델 신뢰도: {p4}', p0=self.selected_bodypart, p1=visibility, p2=source, p3=display_state, p4=likelihood_text)
        )
        reasons = format_anomaly_reasons(self.flag_details.get(self.frame_index, []))
        flag = tr('오류 후보') if reasons else tr('일반 프레임')
        reason_lines = reasons[:2]
        if len(reasons) > 2:
            reason_lines.append(tr('그 외 {p0}개 신호', p0=len(reasons) - 2))
        reason_text = "\n".join(reason_lines)
        reason_block = f"\n{reason_text}" if reason_text else ""
        self.frame_state.setToolTip("\n".join(reasons))
        if self.review_only:
            any_visible = any(self.current_point(bodypart).get("visible") for bodypart in self.bodyparts)
            detection = tr('검출 있음') if any_visible else tr('검출 없음')
            self.frame_state.setText(
                tr('현재: {p0} · {p1}{p2}\n오류 후보 구간 {p3}개', p0=detection, p1=flag, p2=reason_block, p3=len(self.flag_starts))
            )
        elif self.labeling_mode:
            if self.frame_index in self.label_frame_set:
                position = self.label_frame_indices.index(self.frame_index) + 1
                state = tr('라벨 프레임 · 수정 가능')
                progress = tr('라벨 대상 {p0}/{p1}', p0=position, p1=len(self.label_frame_indices))
            else:
                state = tr('문맥 프레임 · 보기 전용')
                progress = tr('전체 라벨 대상 {p0}개', p0=len(self.label_frame_indices))
            self.frame_state.setText(tr('현재: {p0}\n{p1}', p0=state, p1=progress))
        else:
            state = tr('검증 완료') if self.frame_index in self.confirmed else tr('프레임 제외') if self.frame_index in self.excluded else tr('미검증')
            self.frame_state.setText(
                tr('현재: {p0} · {p1}{p2}\n검증 {p3} · 제외 {p4} · 오류 구간 {p5}', p0=state, p1=flag, p2=reason_block, p3=len(self.confirmed), p4=len(self.excluded), p5=len(self.flag_starts))
            )
            self.update_episode_summary()

    @staticmethod
    def format_time(seconds: float) -> str:
        minutes = int(seconds // 60)
        remainder = seconds - minutes * 60
        return f"{minutes:02d}:{remainder:06.3f}"

    @property
    def session_path(self) -> Path:
        return self.output / "review-session.json"

    def temporal_episode_payload(self) -> dict:
        episodes = []
        for episode in self.temporal_episodes:
            record = dict(episode)
            candidates = set(map(int, episode.get("gt_frames", []) or []))
            record["verified_gt_frames"] = sorted((candidates & self.confirmed) - self.excluded)
            record["edited_gt_frames"] = sorted(candidates & set(self.edits))
            record["excluded_gt_frames"] = sorted(candidates & self.excluded)
            episodes.append(record)
        return {
            "schema": "hagisilta-temporal-gt-episodes-v1",
            "video": str(self.video),
            "source_pose": str(self.pose_path) if self.pose_path else None,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "bodyparts": self.bodyparts,
            "episode_count": len(episodes),
            "candidate_gt_frames": sorted(episode_frame_groups(episodes)),
            "verified_gt_frames": sorted({
                frame for episode in episodes for frame in episode["verified_gt_frames"]
            }),
            "episodes": episodes,
        }

    def save_session(self) -> bool:
        if self.review_only:
            return False
        from PySide6.QtCore import QLockFile
        from hagisilta_atomic_io import atomic_write_bytes
        lock = QLockFile(str(self.session_path) + ".lock")
        if not lock.tryLock(0):
            QMessageBox.warning(self, tr('교정 저장 대기'), tr('다른 교정 창이 저장 중입니다. 잠시 후 다시 저장하세요.'))
            return False
        try:
            current = self.session_path.read_bytes() if self.session_path.exists() else None
            if getattr(self, "_session_load_failed", False) or current != getattr(self, "_loaded_session_bytes", None):
                raise ValueError(tr('다른 창에서 교정 내용이 변경되었거나 세션을 정상적으로 읽지 못했습니다. 기존 파일은 유지했습니다. 이 창을 닫기 전에 수정 내용을 확인하고 다시 여세요.'))
            result = self._write_session(atomic_write_bytes)
            return result
        except Exception as error:
            QMessageBox.warning(self, tr('교정 저장 실패'), str(error))
            return False
        finally:
            lock.unlock()

    def _write_session(self, atomic_write_bytes):
        for frame in set(self.edits) | self.confirmed:
            self.freeze_frame_baseline(frame)
        payload = {
            "version": 2,
            "video": str(self.video),
            "pose": str(self.pose_path) if self.pose_path else None,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "bodyparts": self.bodyparts,
            "confirmed": sorted(self.confirmed),
            "excluded": sorted(self.excluded),
            "edits": {str(frame): values for frame, values in self.edits.items()},
            "temporal_episodes": self.temporal_episodes,
            "episode_draft_start": self.episode_draft_start,
            "frame_baselines": {str(frame): record for frame, record in self.frame_baselines.items()
                                if frame in self.edits or frame in self.confirmed},
        }
        encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        episode_payload = self.temporal_episode_payload()
        atomic_write_bytes(self.output / "temporal-episodes.json",
                           json.dumps(episode_payload, ensure_ascii=False, indent=2).encode("utf-8"))
        episode_groups = episode_frame_groups(self.temporal_episodes)
        rows = []
        for frame, values in sorted(self.edits.items()):
            for bodypart, point in values.items():
                rows.append({
                    "frame": frame,
                    "time_s": frame / self.fps,
                    "episode_id": episode_groups.get(frame, ""),
                    "bodypart": bodypart,
                    **point,
                })
        atomic_write_bytes(self.output / "human-corrections.csv", pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"))
        atomic_write_bytes(self.session_path, encoded)
        self._loaded_session_bytes = encoded
        self.dirty = False
        self.footer_status.setText(
            tr('저장 완료 · 교정 {p0}프레임 · 시계열 GT 구간 {p1}개', p0=len(self.edits), p1=len(self.temporal_episodes))
        )
        return True

    def save_and_close(self) -> None:
        if self.review_only:
            self.close()
            return
        if self.save_session():
            self.close()

    def load_session(self) -> None:
        self._loaded_session_bytes = None
        self._session_load_failed = False
        if self.review_only:
            return
        if not self.session_path.exists():
            return
        try:
            self._loaded_session_bytes = self.session_path.read_bytes()
            payload = json.loads(self._loaded_session_bytes)
            if Path(payload.get("video", "")).resolve() != self.video:
                raise ValueError(tr('저장된 교정과 영상이 다릅니다.'))
            stored_pose = Path(payload["pose"]).resolve() if payload.get("pose") else None
            if stored_pose != self.pose_path:
                raise ValueError(tr('저장된 교정과 원본 추론 파일이 다릅니다. 교정 내용을 덮어쓰지 마세요.'))
            from hagisilta_review_baseline import validate_frame_baselines
            validate_frame_baselines(payload, self.bodyparts, self.frame_count)
            self.confirmed = set(map(int, payload.get("confirmed", [])))
            self.excluded = set(map(int, payload.get("excluded", [])))
            self.edits = {int(frame): values for frame, values in payload.get("edits", {}).items()}
            self.frame_baselines = {int(frame): record for frame, record in payload.get("frame_baselines", {}).items()}
            self.previously_reviewed = set(self.edits) | self.confirmed
            self.temporal_episodes = []
            for raw_episode in payload.get("temporal_episodes", []) or []:
                episode = normalize_episode(raw_episode, self.fps, self.frame_count)
                if episode is not None and not episodes_overlap(
                    self.temporal_episodes, episode["start_frame"], episode["end_frame"]
                ):
                    self.temporal_episodes.append(episode)
            self.temporal_episodes.sort(
                key=lambda value: (value["start_frame"], value["end_frame"])
            )
            draft = payload.get("episode_draft_start")
            self.episode_draft_start = (
                max(0, min(int(draft), self.frame_count - 1)) if draft is not None else None
            )
            self.selected_episode_id = (
                str(self.temporal_episodes[0]["episode_id"]) if self.temporal_episodes else ""
            )
            self.statusBar().showMessage(tr('이전 검토 세션을 불러왔습니다.'), 3000)
        except Exception as error:
            self._session_load_failed = True
            QMessageBox.warning(self, tr('세션 불러오기 실패'), str(error))

    def export_ground_truth(self) -> None:
        if not self.confirmed:
            QMessageBox.information(self, tr('내보낼 프레임 없음'), tr('Enter로 검증 완료한 프레임이 없습니다.'))
            return
        dataset_name = f"{self.video.stem}_video_review"
        labeled_dir = self.output / "labeled-data" / dataset_name
        labeled_dir.mkdir(parents=True, exist_ok=True)
        rows, indices = [], []
        for frame_index in sorted(self.confirmed):
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = self.capture.read()
            if not ok:
                continue
            filename = f"img{frame_index:06d}.png"
            success, encoded = cv2.imencode(".png", image)
            if not success:
                continue
            encoded.tofile(labeled_dir / filename)
            values = []
            for bodypart in self.bodyparts:
                point = self.current_point(bodypart, frame_index)
                if point.get("visible"):
                    values.extend([point.get("x", math.nan), point.get("y", math.nan)])
                else:
                    values.extend([math.nan, math.nan])
            rows.append(values)
            indices.append(("labeled-data", dataset_name, filename))
        columns = pd.MultiIndex.from_tuples(
            [(self.scorer, bodypart, coord) for bodypart in self.bodyparts for coord in ("x", "y")],
            names=["scorer", "bodyparts", "coords"],
        )
        index = pd.MultiIndex.from_tuples(indices)
        labels = pd.DataFrame(rows, index=index, columns=columns, dtype=float)
        csv_path = labeled_dir / f"CollectedData_{self.scorer}.csv"
        h5_path = labeled_dir / f"CollectedData_{self.scorer}.h5"
        labels.to_csv(csv_path)
        labels.to_hdf(h5_path, key="df_with_missing", mode="w")
        self.save_session()
        QMessageBox.information(
            self,
            tr('학습용 내보내기 완료'),
            tr('사람이 검증한 {p0}개 프레임을 별도 폴더에 저장했습니다.\n\n{p1}\n\n아직 현재 DLC 프로젝트에는 자동 반영하지 않았습니다.', p0=len(labels), p1=labeled_dir),
        )

    def export_corrected_track(self) -> None:
        if self.pose is None or self.pose_path is None:
            QMessageBox.information(self, tr('DLC 좌표 없음'), tr('전체 수정 트랙을 만들려면 DLC 추론 CSV/H5가 필요합니다.'))
            return
        # Human drag coordinates are float64; promote once to avoid per-cell
        # dtype coercion warnings and preserve the corrected subpixel values.
        corrected = self.reviewed_pose()
        bodypart_level = corrected.columns.nlevels - 2
        coord_level = corrected.columns.nlevels - 1

        def find_column(bodypart: str, coord: str):
            return next(
                (
                    column for column in corrected.columns
                    if str(column[bodypart_level]) == bodypart and str(column[coord_level]) == coord
                ),
                None,
            )

        correction_rows = []
        for frame_index, frame_edits in sorted(self.edits.items()):
            if not 0 <= frame_index < len(corrected):
                continue
            for bodypart, point in frame_edits.items():
                x_column = find_column(bodypart, "x")
                y_column = find_column(bodypart, "y")
                likelihood_column = find_column(bodypart, "likelihood")
                visible = bool(point.get("visible", False))
                x = float(point.get("x", math.nan)) if visible else math.nan
                y = float(point.get("y", math.nan)) if visible else math.nan
                if x_column is not None:
                    corrected.at[corrected.index[frame_index], x_column] = x
                if y_column is not None:
                    corrected.at[corrected.index[frame_index], y_column] = y
                if likelihood_column is not None:
                    corrected.at[corrected.index[frame_index], likelihood_column] = 1.0 if visible else 0.0
                correction_rows.append({
                    "frame": frame_index,
                    "time_s": frame_index / self.fps,
                    "bodypart": bodypart,
                    "x": x,
                    "y": y,
                    "visible": visible,
                    "source": point.get("source", "human"),
                })

        output = self.output / "corrected-tracking"
        output.mkdir(parents=True, exist_ok=True)
        stem = f"{self.video.stem}_{self.pose_path.stem}_corrected"
        csv_path = output / f"{stem}.csv"
        h5_path = output / f"{stem}.h5"
        corrected.to_csv(csv_path)
        corrected.to_hdf(h5_path, key="df_with_missing", mode="w")
        pd.DataFrame(correction_rows).to_csv(
            output / f"{stem}_human-edits.csv", index=False, encoding="utf-8-sig"
        )
        metadata = {
            "video": str(self.video),
            "source_pose": str(self.pose_path),
            "corrected_h5": str(h5_path),
            "frames": len(corrected),
            "edited_frames": len(self.edits),
            "edited_points": len(correction_rows),
        }
        (output / f"{stem}.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.save_session()
        QMessageBox.information(
            self,
            tr('전체 수정 트랙 내보내기 완료'),
            tr('총 {p0}프레임 중 사람이 수정한 {p1}프레임을 반영했습니다.\n\n{p2}', p0=len(corrected), p1=len(self.edits), p2=h5_path),
        )

    def closeEvent(self, event) -> None:
        if not self.review_only and self.dirty:
            answer = QMessageBox.question(
                self,
                tr('변경 내용 저장'),
                tr('저장하지 않은 수정 내용이 있습니다. 저장하고 닫을까요?'),
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if answer == QMessageBox.Cancel:
                event.ignore()
                return
            if answer == QMessageBox.Save:
                if self.save_session() is False:
                    event.ignore()
                    return
        self.timer.stop()
        self.capture.release()
        event.accept()


def choose_video() -> Path | None:
    path, _ = QFileDialog.getOpenFileName(None, tr('검토할 영상'), "", "Video (*.avi *.mp4 *.mov *.mkv)")
    return Path(path) if path else None


def choose_pose(video: Path) -> Path | None:
    candidates = []
    if video.parent.name.lower() == "videos":
        tracked = project_output_root(video.parent.parent) / "tracked"
        candidates = sorted(tracked.glob(f"{video.stem}*.h5"), key=lambda path: path.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    path, _ = QFileDialog.getOpenFileName(None, tr('DLC 추론 좌표(선택)'), str(video.parent), "DLC result (*.h5 *.hdf5 *.csv)")
    return Path(path) if path else None


def main() -> int:
    parser = argparse.ArgumentParser(description="HAGISILTA video ground-truth editor pilot")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--pose", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scorer", default="HAGISILTA")
    parser.add_argument("--playlist", type=Path)
    parser.add_argument("--playlist-index", type=int, default=0)
    parser.add_argument("--review-only", action="store_true")
    parser.add_argument("--draft-manifest", type=Path)
    args = parser.parse_args()

    app = QApplication(sys.argv)
    init_i18n(app)
    app.setApplicationName("HAGISILTA")
    if APP_ICON.is_file():
        app.setWindowIcon(QIcon(str(APP_ICON)))
    install_wheel_value_guard(app)
    video = args.video or choose_video()
    if not video:
        return 0
    pose = args.pose if args.pose and args.pose.exists() else choose_pose(video)
    playlist = None
    if args.playlist and args.playlist.is_file():
        try:
            loaded = json.loads(args.playlist.read_text(encoding="utf-8"))
            if isinstance(loaded, list) and loaded:
                playlist = [entry for entry in loaded if isinstance(entry, dict)]
        except Exception:
            playlist = None
    try:
        window = VideoReviewPilot(
        video, pose, args.output, args.scorer,
        playlist=playlist, playlist_index=args.playlist_index,
        review_only=args.review_only,
        draft_manifest=args.draft_manifest,
    )
    except Exception as error:
        QMessageBox.critical(
            None,
            tr('추론 영상 확인 실패') if args.review_only else tr('영상 교정기 실행 실패'),
            str(error),
        )
        return 1
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
