# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from hagisilta_i18n import init_i18n, tr
from hagisilta_render_text import draw_image_text

import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QImage, QKeySequence, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from hagisilta_window_chrome import (
    HeaderContext,
    apply_hagisilta_window_identity, create_hagisilta_header,
    create_hagisilta_help_button, create_hagisilta_settings_section,
    enable_full_window_controls, hagisilta_control_stylesheet,
)
from hagisilta_runtime import project_output_root, project_state_root
from hagisilta_playback import PLAYBACK_HELP, PLAYBACK_SPEEDS, PlaybackSpeedControl, playback_interval_ms
from hagisilta_watermaze_roi_overlay import load_water_roi_overlay, paint_water_roi_overlay
from hagisilta_result_composer import WaterMazePreviewLabel, WaterMazeVisualComposer


VIDEO_FILTER = "Videos (*.avi *.mp4 *.mov *.mkv *.wmv)"

TRIAL_TYPE_TOOLTIP = (
    "시험 유형은 선택 구간을 자르거나 좌표 계산 방식을 바꾸지 않습니다. "
    "네 유형 모두 같은 구간에서 궤적·거리·속도·사분면·벽면·플랫폼 지표를 계산하고, "
    "결과를 어떤 목적으로 해석할지 구분하는 메타데이터입니다.\n\n"
    "Hidden: 물속에 숨긴 플랫폼 위치를 주변 단서로 학습합니다. "
    "도달시간(Latency), 이동거리, 경로 효율과 Trial별 향상을 주로 봅니다.\n"
    "Probe: 플랫폼을 제거하고 이전 위치의 기억을 검사합니다. "
    "목표 사분면 체류율, 이전 플랫폼 위치 통과 횟수, 평균 근접거리가 핵심이며 "
    "플랫폼 도달 여부·Latency는 주 지표로 해석하지 않습니다.\n"
    "Visible: 동물이 볼 수 있는 플랫폼을 찾게 하여 시각·운동·수영 동기를 확인합니다. "
    "보이는 플랫폼 도달시간과 이동거리를 주로 봅니다.\n"
    "Reversal: 플랫폼을 새 위치로 옮긴 뒤 기존 기억을 억제하고 새 위치를 학습하는 "
    "인지적 유연성을 평가합니다. 새 플랫폼 도달시간과 Trial별 개선을 봅니다."
)

START_DIRECTION_TOOLTIP = (
    "동물을 수조에 놓은 시작 방향입니다(N/E/S/W). "
    "현재 위치 좌표나 사분면 계산을 회전시키지는 않으며, Trial 조건을 기록하고 "
    "같은 시작 방향끼리 비교할 때 사용하는 실험 메타데이터입니다."
)

TARGET_QUADRANT_TOOLTIP = (
    "플랫폼이 있거나 Probe 전에 플랫폼이 있었던 목표 사분면입니다. "
    "Target 체류시간·비율과 Probe 기억 지표의 기준이 됩니다. "
    "수조 편집기에서 설정한 방위 회전 기준과 실제 플랫폼 위치를 확인한 뒤 선택하세요."
)


def format_clock(seconds: float) -> str:
    value = max(float(seconds), 0.0)
    minutes = int(value // 60)
    return f"{minutes:02d}:{value - minutes * 60:06.3f}"


class ClickSeekSlider(QSlider):
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.maximum() > self.minimum():
            ratio = min(max(event.position().x() / max(self.width(), 1), 0.0), 1.0)
            self.setValue(round(self.minimum() + ratio * (self.maximum() - self.minimum())))
        super().mousePressEvent(event)


class WaterMazeDecisionViewer(QDialog):
    """Interactive QC playback for saved Water-maze frame decisions."""

    def __init__(
        self, video: Path, frames_file: Path, profile_file: Path,
        summary: dict, parent: QWidget | None = None,
        playlist: list[dict] | None = None,
    ) -> None:
        super().__init__(parent)
        self.video = video
        self.frames = pd.read_csv(frames_file)
        self.profile = json.loads(profile_file.read_text(encoding="utf-8"))
        self.summary = summary
        self.capture = cv2.VideoCapture(str(video))
        if not self.capture.isOpened() or self.frames.empty:
            raise ValueError(tr('원본 Arena 영상 또는 프레임 판정 CSV를 열 수 없습니다.'))
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or summary.get("fps", 1.0) or 1.0)
        self.frame_count = len(self.frames)
        self.capture_position = -1
        self.playing = False
        self.playback_speeds = list(PLAYBACK_SPEEDS)
        self.playback_speed = 1.0
        self.current_image: QImage | None = None
        self.show_pool = True
        self.show_platform = True
        self.show_labels = True
        self.show_banner = True
        self.show_points = True
        self.show_point_labels = True
        self.show_live_trail = True
        self.show_final_trail = False
        self.show_live_heatmap = False
        self.show_final_heatmap = False
        self.overlay_opacity = 0.52
        self.x = pd.to_numeric(self.frames.get("x"), errors="coerce").to_numpy(dtype=float)
        self.y = pd.to_numeric(self.frames.get("y"), errors="coerce").to_numpy(dtype=float)
        valid_series = self.frames.get("valid", pd.Series(False, index=self.frames.index))
        self.valid = valid_series.astype(str).str.lower().isin({"1", "true", "yes"}).to_numpy()
        self.final_heatmap: np.ndarray | None = None
        self.live_heat_counts: np.ndarray | None = None
        self.live_heat_index = -1
        raw_playlist = playlist or [{
            "video": str(video), "frames": str(frames_file),
            "profile": str(profile_file), "summary": dict(summary),
        }]
        self.playlist = [
            {
                **entry,
                "video": str(entry.get("video", "")),
                "frames": str(entry.get("frames", "")),
                "profile": str(entry.get("profile", "")),
                "summary": dict(entry.get("summary") or {}),
            }
            for entry in raw_playlist
        ]
        current_frames = str(Path(frames_file))
        self.current_playlist_index = next(
            (
                index for index, entry in enumerate(self.playlist)
                if str(Path(entry["frames"])) == current_frames
            ),
            0,
        )
        self.setWindowTitle(tr('HAGISILTA · Water maze 판정 검수 · {p0}', p0=video.name))
        apply_hagisilta_window_identity(self)
        enable_full_window_controls(self)
        self.resize(1240, 820)
        self.setMinimumSize(1120, 650)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        header, header_layout = create_hagisilta_header(
            tr('Water-maze 판정 검수'), "HAGISILTA · BEHAVIOR REVIEW",
        )
        help_text = (
            tr('저장된 프레임 판정을 원본 영상과 함께 재생합니다. 현재까지의 궤적·히트맵과 분석 구간 전체 결과를 따로 켜서 경계·누락·플랫폼 판정을 비교할 수 있습니다.\n\n재생/정지 (Space) · 1프레임 이동 (A/D) · 1초 이동 (Q/E)\n수조 ROI (R) · 플랫폼 (P) · 사분면 (L) · 판정 정보 (I)\n신체점 (K) · 신체점 이름 (N)\n현재/전체 궤적 (T/F) · 현재/전체 히트맵 (H/M)\n\n')
            + tr(PLAYBACK_HELP)
        )
        header_layout.addWidget(create_hagisilta_help_button(
            self, tr('Water-maze 판정 검수 도움말'), help_text,
        ))
        self.header_status = HeaderContext(
            header,
            f"{self.current_playlist_index + 1}/{len(self.playlist)} · 1/{self.frame_count}F",
        )
        root.addWidget(header)
        self.setStyleSheet(self.styleSheet() + hagisilta_control_stylesheet())

        body = QFrame(objectName="detachedBody")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(14, 12, 14, 12)
        body_layout.setSpacing(12)
        workspace = QFrame(objectName="reviewWorkspace")
        workspace.setMinimumWidth(520)
        workspace.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(8)
        sidebar = QFrame(objectName="detachedSidebar")
        sidebar.setFixedWidth(310)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        sidebar_layout.setSpacing(10)

        playlist_panel = QFrame(objectName="detachedSidebar")
        playlist_panel.setMinimumWidth(220)
        playlist_panel.setMaximumWidth(480)
        playlist_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        playlist_layout = QVBoxLayout(playlist_panel)
        playlist_layout.setContentsMargins(10, 9, 10, 10)
        playlist_layout.setSpacing(7)
        playlist_layout.addWidget(QLabel(tr('판정 영상·Trial'), objectName="settingsGroupTitle"))
        playlist_description = QLabel(
            tr('항목을 클릭하면 이 창에서 바로 전환합니다.'),
            objectName="settingsGroupDescription",
        )
        playlist_description.setWordWrap(True)
        playlist_layout.addWidget(playlist_description)
        self.playlist_widget = QListWidget()
        self.playlist_widget.setObjectName("decisionPlaylist")
        self.playlist_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.playlist_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.playlist_widget.setSpacing(4)
        for entry in self.playlist:
            payload = entry["summary"]
            bundle = str(payload.get("video_bundle") or Path(entry["video"]).stem)
            try:
                arena = int(float(payload.get("arena_id", 1)))
            except (TypeError, ValueError):
                arena = 1
            trial = str(payload.get("trial", "-") or "-")
            trial_type = str(payload.get("trial_type", "") or "")
            detail = f"Arena {arena:02d}  ·  Trial {trial}"
            if trial_type:
                detail += f"  ·  {trial_type}"
            text = f"{bundle}\n{detail}"
            item = QListWidgetItem(text)
            item.setSizeHint(QSize(0, 54))
            item.setToolTip(str(entry["video"]))
            self.playlist_widget.addItem(item)
        self.playlist_widget.setCurrentRow(self.current_playlist_index)
        self.playlist_widget.currentRowChanged.connect(self.select_playlist_entry)
        playlist_layout.addWidget(self.playlist_widget, 1)

        base_options = QFrame(objectName="settingsGroup")
        base_layout = QVBoxLayout(base_options)
        base_layout.setContentsMargins(10, 9, 10, 10)
        base_layout.setSpacing(7)
        base_layout.addWidget(QLabel(tr('기본 오버레이'), objectName="settingsGroupTitle"))
        base_description = QLabel(
            tr('영상 위에 계속 확인할 판정 기준을 선택합니다.'),
            objectName="settingsGroupDescription",
        )
        base_description.setWordWrap(True)
        base_layout.addWidget(base_description)

        path_options = QFrame(objectName="settingsGroup")
        path_layout = QVBoxLayout(path_options)
        path_layout.setContentsMargins(10, 9, 10, 10)
        path_layout.setSpacing(7)
        path_layout.addWidget(QLabel(tr('궤적·히트맵'), objectName="settingsGroupTitle"))
        path_description = QLabel(
            tr('현재 시점 누적값과 전체 분석 구간 결과를 비교합니다.'),
            objectName="settingsGroupDescription",
        )
        path_description.setWordWrap(True)
        path_layout.addWidget(path_description)

        specs = [
            ("pool_toggle", tr('수조 ROI (R)'), True, self.set_pool_visible,
             tr('수조 외곽과 벽면 선호 경계를 표시합니다. 단축키 R')),
            ("platform_toggle", tr('플랫폼 (P)'), True, self.set_platform_visible,
             tr('플랫폼 위치를 표시합니다. 단축키 P')),
            ("labels_toggle", tr('사분면 경계·이름 (L)'), True, self.set_labels_visible,
             tr('회전 기준에 맞춘 사분면 경계선과 NE/SE/SW/NW 이름을 함께 표시합니다. 단축키 L')),
            ("banner_toggle", tr('판정 정보 (I)'), True, self.set_banner_visible,
             tr('현재 구역·플랫폼 여부·분석 시간을 표시합니다. 단축키 I')),
            ("points_toggle", tr('신체점 (K)'), True, self.set_points_visible,
             tr('선택한 신체점과 중심 대표좌표의 점을 표시합니다. 단축키 K')),
            ("point_labels_toggle", tr('신체점 이름 (N)'), True, self.set_point_labels_visible,
             tr('각 신체점 옆의 이름을 표시합니다. 단축키 N')),
            ("live_trail_toggle", tr('현재까지 궤적 (T)'), True, self.set_live_trail_visible,
             tr('첫 프레임부터 현재 재생 위치까지 누적된 궤적입니다. 단축키 T')),
            ("final_trail_toggle", tr('최종 전체 궤적 (F)'), False, self.set_final_trail_visible,
             tr('현재 재생 위치와 관계없이 선택 분석 구간 전체 궤적을 표시합니다. 단축키 F')),
            ("live_heat_toggle", tr('현재까지 히트맵 (H)'), False, self.set_live_heat_visible,
             tr('첫 프레임부터 현재 위치까지의 체류 분포를 누적 표시합니다. 단축키 H')),
            ("final_heat_toggle", tr('최종 전체 히트맵 (M)'), False, self.set_final_heat_visible,
             tr('선택 분석 구간 전체의 체류 분포를 표시합니다. 단축키 M')),
        ]
        for index, (name, text, checked, slot, tip) in enumerate(specs):
            toggle = QCheckBox(text)
            toggle.setChecked(checked)
            toggle.setToolTip(tip)
            toggle.toggled.connect(slot)
            setattr(self, name, toggle)
            (base_layout if index < 6 else path_layout).addWidget(toggle)

        opacity_options = QFrame(objectName="settingsGroup")
        opacity_layout = QVBoxLayout(opacity_options)
        opacity_layout.setContentsMargins(10, 9, 10, 10)
        opacity_layout.setSpacing(7)
        opacity_layout.addWidget(QLabel(tr('표시 투명도'), objectName="settingsGroupTitle"))
        opacity_description = QLabel(
            tr('영상 위에 표시되는 ROI·판정·궤적·히트맵 전체에 적용됩니다.'),
            objectName="settingsGroupDescription",
        )
        opacity_description.setWordWrap(True)
        opacity_layout.addWidget(opacity_description)
        opacity_row = QHBoxLayout()
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(10, 85)
        self.opacity_slider.setValue(round(self.overlay_opacity * 100))
        self.opacity_slider.valueChanged.connect(self.set_overlay_opacity)
        opacity_row.addWidget(self.opacity_slider, 1)
        self.opacity_label = QLabel(f"{round(self.overlay_opacity * 100)}%")
        self.opacity_label.setMinimumWidth(36)
        self.opacity_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        opacity_row.addWidget(self.opacity_label)
        opacity_layout.addLayout(opacity_row)
        sidebar_layout.addWidget(base_options)
        sidebar_layout.addWidget(path_options)
        sidebar_layout.addWidget(opacity_options)
        sidebar_layout.addStretch()
        review_splitter = QSplitter(Qt.Horizontal)
        review_splitter.setObjectName("reviewPlaylistSplitter")
        review_splitter.setChildrenCollapsible(False)
        review_splitter.setHandleWidth(7)
        review_splitter.addWidget(playlist_panel)
        review_splitter.addWidget(workspace)
        review_splitter.setStretchFactor(0, 1)
        review_splitter.setStretchFactor(1, 3)
        review_splitter.setSizes([290, 700])
        body_layout.addWidget(review_splitter, 1)
        body_layout.addWidget(sidebar)
        root.addWidget(body, 1)

        self.preview = WaterMazePreviewLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(520, 360)
        self.preview.setStyleSheet("background:#090f18; border:1px solid #31445d;")
        self.preview.resized.connect(self.update_preview_pixmap)
        workspace_layout.addWidget(self.preview, 1)
        decision_panel = QFrame(objectName="statusSummaryBar")
        decision_layout = QVBoxLayout(decision_panel)
        decision_layout.setContentsMargins(10, 5, 10, 5)
        decision_layout.setSpacing(1)
        self.decision_label = QLabel(objectName="statusSummaryValue")
        self.decision_label.setWordWrap(True)
        decision_layout.addWidget(self.decision_label)
        workspace_layout.addWidget(decision_panel)
        self.timeline = ClickSeekSlider(Qt.Horizontal)
        self.timeline.setRange(0, max(0, self.frame_count - 1))
        self.timeline.valueChanged.connect(self.seek_frame)
        workspace_layout.addWidget(self.timeline)
        clock_row = QHBoxLayout()
        self.position_label = QLabel(objectName="fieldLabel")
        clock_row.addWidget(self.position_label)
        clock_row.addStretch()
        self.playback_control = PlaybackSpeedControl(self)
        self.speed_combo = self.playback_control.combo
        self.speed_combo.currentIndexChanged.connect(self.playback_speed_changed)
        clock_row.addWidget(self.playback_control)
        workspace_layout.addLayout(clock_row)
        controls = QHBoxLayout()
        self.play_button = QPushButton(tr('▶ 재생 (Space)'), objectName="primaryButton")
        self.play_button.clicked.connect(self.toggle_playback)
        controls.addWidget(self.play_button)
        for label, delta, seconds in (
            (tr('-1초 (Q)'), -1, True), ("-1F (A)", -1, False),
            ("+1F (D)", 1, False), (tr('+1초 (E)'), 1, True),
        ):
            button = QPushButton(label, objectName="secondaryButton")
            button.clicked.connect(
                lambda _checked=False, amount=delta, as_seconds=seconds: self.step_scaled(
                    round(self.fps * amount) if as_seconds else amount
                )
            )
            controls.addWidget(button)
        controls.addStretch()
        workspace_layout.addLayout(controls)
        footer_frame = QFrame(objectName="detachedFooter")
        footer = QHBoxLayout(footer_frame)
        footer.setContentsMargins(18, 9, 18, 9)
        footer.addStretch()
        close_button = QPushButton(tr('닫기'), objectName="secondaryButton")
        close_button.clicked.connect(self.accept)
        footer.addWidget(close_button)
        root.addWidget(footer_frame)
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(lambda: self.step_frames(1))
        self.update_playback_timer()
        self.seek_frame(0)

    @staticmethod
    def _truth(value) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes"}

    @staticmethod
    def _clock(seconds: float) -> str:
        minutes = int(max(seconds, 0.0) // 60)
        return f"{minutes:02d}:{max(seconds, 0.0) - minutes * 60:06.3f}"

    @staticmethod
    def _number(value) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    @classmethod
    def _quadrant_text(cls, value) -> str:
        if value is None:
            return tr('판정 불가')
        text = str(value).strip()
        return text if text and text.lower() not in {"nan", "none"} else tr('판정 불가')

    def trial_metric_overlay_lines(self) -> list[str]:
        normalized = str(self.summary.get("trial_type", "") or "").strip().lower()
        latency = self._number(self.summary.get("escape_latency_seconds"))
        distance = self._number(self.summary.get("path_length_cm"))
        efficiency = self._number(self.summary.get("path_efficiency"))
        speed = self._number(self.summary.get("mean_speed_cm_s"))
        target_percent = self._number(self.summary.get("target_quadrant_percent"))
        proximity = self._number(self.summary.get("mean_proximity_cm"))
        crossings = self._number(self.summary.get("platform_crossings"))
        latency_text = f"{latency:.2f}s" if latency is not None else "MISS"
        distance_text = f"{distance:,.1f}cm" if distance is not None else "-"
        if normalized == "probe":
            return [
                f"target {target_percent:.1f}% | crossings {round(crossings):d} | proximity {proximity:.1f}cm"
                if target_percent is not None and crossings is not None and proximity is not None
                else "target - | crossings - | proximity -",
            ]
        elif normalized == "visible":
            return [
                f"latency {latency_text} | distance {distance_text} | "
                f"speed {speed:.1f}cm/s" if speed is not None
                else f"latency {latency_text} | distance {distance_text} | speed -",
            ]
        return [
            f"latency {latency_text} | distance {distance_text} | "
            f"efficiency {efficiency:.3f}" if efficiency is not None
            else f"latency {latency_text} | distance {distance_text} | efficiency -",
        ]

    def select_playlist_entry(self, index: int) -> None:
        if not 0 <= index < len(self.playlist) or index == self.current_playlist_index:
            return
        entry = self.playlist[index]
        video = Path(entry["video"])
        frames_file = Path(entry["frames"])
        profile_file = Path(entry["profile"])
        try:
            frames = pd.read_csv(frames_file)
            profile = json.loads(profile_file.read_text(encoding="utf-8"))
            capture = cv2.VideoCapture(str(video))
            if not capture.isOpened() or frames.empty:
                capture.release()
                raise ValueError(tr('원본 Arena 영상 또는 프레임 판정 CSV를 열 수 없습니다.'))
        except Exception as error:
            self.playlist_widget.blockSignals(True)
            self.playlist_widget.setCurrentRow(self.current_playlist_index)
            self.playlist_widget.blockSignals(False)
            QMessageBox.warning(self, tr('판정 영상 전환 실패'), str(error))
            return

        self.timer.stop()
        self.playing = False
        self.play_button.setText(tr('▶ 재생 (Space)'))
        self.capture.release()
        self.video = video
        self.frames = frames
        self.profile = profile
        self.summary = dict(entry.get("summary") or {})
        self.capture = capture
        self.fps = float(
            self.capture.get(cv2.CAP_PROP_FPS) or self.summary.get("fps", 1.0) or 1.0
        )
        self.frame_count = len(self.frames)
        self.capture_position = -1
        self.current_image = None
        self.x = pd.to_numeric(self.frames.get("x"), errors="coerce").to_numpy(dtype=float)
        self.y = pd.to_numeric(self.frames.get("y"), errors="coerce").to_numpy(dtype=float)
        valid_series = self.frames.get("valid", pd.Series(False, index=self.frames.index))
        self.valid = valid_series.astype(str).str.lower().isin({"1", "true", "yes"}).to_numpy()
        self.final_heatmap = None
        self.live_heat_counts = None
        self.live_heat_index = -1
        self.current_playlist_index = index
        self.setWindowTitle(tr('HAGISILTA · Water maze 판정 검수 · {p0}', p0=video.name))
        self.timeline.blockSignals(True)
        self.timeline.setRange(0, max(0, self.frame_count - 1))
        self.timeline.setValue(0)
        self.timeline.blockSignals(False)
        self.update_playback_timer()
        self.seek_frame(0)

    def pool_geometry(self) -> tuple[float, float, float, float]:
        pool = self.profile["pool"]
        cx, cy = map(float, pool["center_px"])
        radius = float(pool.get("radius_px", 1.0))
        return cx, cy, float(pool.get("radius_x_px", radius)), float(pool.get("radius_y_px", radius))

    def pool_mask(self, shape: tuple[int, int]) -> np.ndarray:
        height, width = shape
        cx, cy, rx, ry = self.pool_geometry()
        yy, xx = np.mgrid[:height, :width]
        return ((xx - cx) / max(rx, 1e-9)) ** 2 + ((yy - cy) / max(ry, 1e-9)) ** 2 <= 1.0

    def heatmap(self, end_index: int | None, shape: tuple[int, int]) -> np.ndarray:
        height, width = shape
        if end_index is None and self.final_heatmap is not None:
            return self.final_heatmap
        limit = self.frame_count if end_index is None else min(end_index + 1, self.frame_count)
        counts = np.zeros((height, width), dtype=np.float32)
        indices = np.flatnonzero(
            self.valid[:limit] & np.isfinite(self.x[:limit]) & np.isfinite(self.y[:limit])
        )
        if len(indices):
            px = np.clip(np.rint(self.x[indices]).astype(int), 0, width - 1)
            py = np.clip(np.rint(self.y[indices]).astype(int), 0, height - 1)
            np.add.at(counts, (py, px), 1.0)
        sigma = max(5, round(min(width, height) * 0.025))
        kernel = sigma * 2 + 1
        blurred = cv2.GaussianBlur(counts, (kernel, kernel), 0)
        peak = float(blurred.max())
        normalized = np.zeros_like(blurred, dtype=np.uint8)
        if peak > 0:
            normalized = np.clip(blurred / peak * 255, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        if end_index is None:
            self.final_heatmap = colored
        return colored

    def draw_roi(self, frame: np.ndarray) -> None:
        cx, cy, rx, ry = self.pool_geometry()
        center = (round(cx), round(cy))
        axes = (max(1, round(rx)), max(1, round(ry)))
        if self.show_pool:
            cv2.ellipse(frame, center, axes, 0, 0, 360, (84, 220, 169), 2, cv2.LINE_AA)
            ring = WaterMazeCanvas.wall_ring_fraction(self.profile)
            cv2.ellipse(
                frame, center,
                (max(1, round(rx * (1 - ring))), max(1, round(ry * (1 - ring)))),
                0, 0, 360, (0, 211, 255), 1, cv2.LINE_AA,
            )
        angle_offset = float(self.profile.get("north_angle_degrees", 0.0))
        if self.show_labels:
            for boundary in (0.0, 90.0, 180.0, 270.0):
                angle = math.radians(angle_offset + boundary)
                endpoint = (
                    round(cx + rx * math.sin(angle)),
                    round(cy - ry * math.cos(angle)),
                )
                cv2.line(frame, center, endpoint, (205, 218, 232), 1, cv2.LINE_AA)
            for name, offset in (("NE", 45), ("SE", 135), ("SW", 225), ("NW", 315)):
                angle = math.radians(angle_offset + offset)
                point = (round(cx + rx * .58 * math.sin(angle)), round(cy - ry * .58 * math.cos(angle)))
                draw_image_text(frame, name, point, cv2.FONT_HERSHEY_SIMPLEX, .52, (245, 245, 245), 1, cv2.LINE_AA)
        if self.show_platform:
            platform = self.profile["platform"]
            points = np.asarray(platform.get("points_px") or [], dtype=np.int32)
            if str(platform.get("shape_type", "circle")) != "circle" and len(points) >= 3:
                cv2.polylines(frame, [points], True, (102, 102, 255), 2, cv2.LINE_AA)
            else:
                pc = tuple(round(float(v)) for v in platform["center_px"])
                cv2.circle(frame, pc, max(1, round(float(platform["radius_px"]))), (102, 102, 255), 2, cv2.LINE_AA)

    def draw_path(self, frame: np.ndarray, end: int, color: tuple[int, int, int], thickness: int) -> None:
        mask = self.valid[:end] & np.isfinite(self.x[:end]) & np.isfinite(self.y[:end])
        points = np.column_stack((self.x[:end][mask], self.y[:end][mask]))
        if len(points) >= 2:
            cv2.polylines(frame, [np.rint(points).astype(np.int32)], False, color, thickness, cv2.LINE_AA)

    def draw_points(self, frame: np.ndarray, row: pd.Series) -> None:
        if not self.show_points and not self.show_point_labels:
            return
        parts = [part.strip() for part in str(row.get("bodyparts_used", "")).split(",") if part.strip()]
        colors = [(100, 230, 175), (92, 202, 255), (138, 107, 255), (255, 168, 102)]
        for index, part in enumerate(parts):
            x = pd.to_numeric(pd.Series([row.get(f"{part}_x")]), errors="coerce").iloc[0]
            y = pd.to_numeric(pd.Series([row.get(f"{part}_y")]), errors="coerce").iloc[0]
            if pd.isna(x) or pd.isna(y):
                continue
            point = (round(float(x)), round(float(y)))
            if self.show_points:
                cv2.circle(frame, point, 4, colors[index % len(colors)], -1, cv2.LINE_AA)
            if self.show_point_labels:
                draw_image_text(frame, part, (point[0] + 6, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, .38, colors[index % len(colors)], 1, cv2.LINE_AA)
        x = pd.to_numeric(pd.Series([row.get("x")]), errors="coerce").iloc[0]
        y = pd.to_numeric(pd.Series([row.get("y")]), errors="coerce").iloc[0]
        if self.show_points and not pd.isna(x) and not pd.isna(y):
            point = (round(float(x)), round(float(y)))
            cv2.circle(frame, point, 7, (10, 18, 28), 2, cv2.LINE_AA)
            cv2.circle(frame, point, 4, (58, 230, 255), -1, cv2.LINE_AA)

    def draw_banner(self, frame: np.ndarray, row: pd.Series) -> None:
        if not self.show_banner:
            return
        height, width = frame.shape[:2]
        scale = max(0.75, min(width / 640.0, height / 480.0))
        margin = round(8 * scale)
        metric_lines = self.trial_metric_overlay_lines()
        panel_right = min(width - margin, round(632 * scale))
        panel_bottom = min(height - margin, round((63 + 19 * max(0, len(metric_lines) - 1)) * scale))
        overlay = frame.copy()
        cv2.rectangle(
            overlay, (margin, margin), (panel_right, panel_bottom), (7, 13, 22), -1,
        )
        cv2.addWeighted(overlay, .78, frame, .22, 0, frame)
        raw_zone = str(row.get("quadrant") or "").strip()
        zone = raw_zone if raw_zone and raw_zone.lower() not in {"nan", "none"} else "N/A"
        in_platform = self._truth(row.get("in_platform", False))
        analysis_time = float(row.get("analysis_time_seconds", 0.0) or 0.0)
        first_line = (
            f"ZONE {zone}  |  platform {'IN' if in_platform else 'OUT'}"
            f"  |  analysis {analysis_time:.2f}s"
        )
        draw_image_text(
            frame, first_line, (round(20 * scale), round(30 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, .50 * scale, (132, 236, 178),
            max(1, round(scale)), cv2.LINE_AA,
        )
        for index, text in enumerate(metric_lines):
            draw_image_text(
                frame, text,
                (round(20 * scale), round((53 + index * 19) * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, .42 * scale, (200, 207, 218),
                max(1, round(scale)), cv2.LINE_AA,
            )

    def seek_frame(self, index: int) -> None:
        index = min(max(int(index), 0), self.frame_count - 1)
        source_frame = int(float(self.frames.iloc[index].get("frame", index) or index))
        if source_frame != self.capture_position + 1:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
        ok, frame = self.capture.read()
        if not ok:
            return
        self.capture_position = source_frame
        row = self.frames.iloc[index]
        base_frame = frame.copy()
        overlay_frame = frame.copy()
        if self.show_final_heatmap or self.show_live_heatmap:
            heat = self.heatmap(None if self.show_final_heatmap else index, frame.shape[:2])
            mask = self.pool_mask(frame.shape[:2])
            overlay_frame[mask] = heat[mask]
        self.draw_roi(overlay_frame)
        if self.show_final_trail:
            self.draw_path(overlay_frame, self.frame_count, (140, 140, 140), 1)
        if self.show_live_trail:
            self.draw_path(overlay_frame, index + 1, (208, 201, 18), 2)
        self.draw_points(overlay_frame, row)
        self.draw_banner(overlay_frame, row)
        frame = cv2.addWeighted(
            overlay_frame, self.overlay_opacity,
            base_frame, 1.0 - self.overlay_opacity, 0,
        )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
        self.update_preview_pixmap()
        seconds = float(row.get("analysis_time_seconds", index / self.fps) or 0.0)
        self.position_label.setText(
            f"{self._clock(seconds)} / {self._clock(float(self.frames.iloc[-1].get('analysis_time_seconds', (self.frame_count - 1) / self.fps)))} · "
            f"{index + 1}/{self.frame_count}F · {self.playback_speed:g}×"
        )
        self.header_status.setText(
            f"{self.current_playlist_index + 1}/{len(self.playlist)} · "
            f"{index + 1}/{self.frame_count}F · {self.playback_speed:g}×"
        )
        self.decision_label.setText(
            tr('현재 프레임  ·  ZONE {p0} · 플랫폼 {p1} · 판정점 {p2}', p0=self._quadrant_text(row.get('quadrant')), p1=tr('안') if self._truth(row.get('in_platform', False)) else tr('밖'), p2=str(row.get('bodyparts_used', '') or '-'))
        )
        if index >= self.frame_count - 1 and self.playing:
            self.toggle_playback()

    def update_preview_pixmap(self) -> None:
        if self.current_image is not None:
            self.preview.setPixmap(
                QPixmap.fromImage(self.current_image).scaled(
                    self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )

    def redraw(self) -> None:
        self.seek_frame(self.timeline.value())

    def step_frames(self, delta: int) -> None:
        target = min(max(self.timeline.value() + int(delta), 0), self.frame_count - 1)
        if delta > 0 and target == self.timeline.value() == self.frame_count - 1 and self.playing:
            self.toggle_playback()
            return
        self.timeline.setValue(target)

    def step_scaled(self, delta: int) -> None:
        # Keep the historical method name for callers, not its speed multiplier.
        self.step_frames(delta)

    def toggle_playback(self) -> None:
        self.playing = not self.playing
        self.play_button.setText(tr('Ⅱ 일시정지 (Space)') if self.playing else tr('▶ 재생 (Space)'))
        self.timer.start() if self.playing else self.timer.stop()

    def update_playback_timer(self) -> None:
        self.timer.setInterval(playback_interval_ms(self.fps, self.playback_speed))

    def playback_speed_changed(self, index: int) -> None:
        self.playback_speed = float(self.speed_combo.itemData(index) or 1.0)
        self.update_playback_timer()
        self.redraw()

    def set_pool_visible(self, value: bool) -> None: self.show_pool = bool(value); self.redraw()
    def set_platform_visible(self, value: bool) -> None: self.show_platform = bool(value); self.redraw()
    def set_labels_visible(self, value: bool) -> None: self.show_labels = bool(value); self.redraw()
    def set_banner_visible(self, value: bool) -> None: self.show_banner = bool(value); self.redraw()
    def set_points_visible(self, value: bool) -> None: self.show_points = bool(value); self.redraw()
    def set_point_labels_visible(self, value: bool) -> None: self.show_point_labels = bool(value); self.redraw()
    def set_live_trail_visible(self, value: bool) -> None: self.show_live_trail = bool(value); self.redraw()
    def set_final_trail_visible(self, value: bool) -> None: self.show_final_trail = bool(value); self.redraw()
    def set_live_heat_visible(self, value: bool) -> None: self.show_live_heatmap = bool(value); self.redraw()
    def set_final_heat_visible(self, value: bool) -> None: self.show_final_heatmap = bool(value); self.redraw()

    def set_overlay_opacity(self, value: int) -> None:
        self.overlay_opacity = min(max(int(value) / 100, .1), .85)
        self.opacity_label.setText(f"{int(value)}%")
        self.redraw()

    def keyPressEvent(self, event) -> None:
        shortcuts = {
            Qt.Key_R: self.pool_toggle, Qt.Key_P: self.platform_toggle,
            Qt.Key_L: self.labels_toggle, Qt.Key_I: self.banner_toggle,
            Qt.Key_K: self.points_toggle, Qt.Key_N: self.point_labels_toggle,
            Qt.Key_T: self.live_trail_toggle,
            Qt.Key_F: self.final_trail_toggle, Qt.Key_H: self.live_heat_toggle,
            Qt.Key_M: self.final_heat_toggle,
        }
        if event.key() in shortcuts:
            shortcuts[event.key()].toggle(); return
        if event.key() == Qt.Key_Space:
            self.toggle_playback(); return
        if event.key() == Qt.Key_A: self.step_scaled(-1); return
        if event.key() == Qt.Key_D: self.step_scaled(1); return
        if event.key() == Qt.Key_Q: self.step_scaled(-round(self.fps)); return
        if event.key() == Qt.Key_E: self.step_scaled(round(self.fps)); return
        super().keyPressEvent(event)

    def done(self, result: int) -> None:
        self.timer.stop()
        self.capture.release()
        super().done(result)


from hagisilta_analysis_range import validate_range, frame_bounds, format_time
from hagisilta_analysis_range_ui import (
    AnalysisRangeControls, BatchAnalysisRangeDialog, RANGE_HELP, TimecodeSpin,
    create_range_inspector, create_range_clock, create_range_splitter,
)
from PySide6.QtGui import QShortcut, QValidator


class TrialBoundarySpin(TimecodeSpin):
    """An explicitly unset boundary; zero remains a real source-video time."""
    valueCommitted = Signal()

    def __init__(self):
        super().__init__()
        self.pending_text = None
        self.setMinimum(-1)
        self.setSpecialValueText(tr('미지정'))
        self.setValue(-1)
        self.lineEdit().textEdited.connect(self.remember_edit)
        self.editingFinished.connect(self.commit_text_edit)

    def remember_edit(self, text):
        self.pending_text = text

    def commit_text_edit(self):
        text, self.pending_text = self.pending_text, None
        if text is not None and self.validate(text, len(text))[0] == QValidator.Acceptable:
            self.valueCommitted.emit()

    def stepBy(self, steps):
        previous = self.value()
        super().stepBy(steps)
        self.pending_text = None
        if self.value() != previous:
            self.valueCommitted.emit()

    def validate(self, text, position):
        if text.strip() == self.specialValueText():
            return QValidator.Acceptable, text, position
        return super().validate(text, position)

    def valueFromText(self, text):
        if text.strip() == self.specialValueText():
            return -1.0
        return super().valueFromText(text)


class WaterTrialRangeControls(QWidget):
    """Independent boundaries for serial trials; no implicit duration lock."""
    rangeChanged = Signal()

    def __init__(self, duration, fps, current_seconds, seek_seconds, parent=None):
        super().__init__(parent)
        self.duration, self.fps = duration, fps
        self.current_seconds, self.seek_seconds = current_seconds, seek_seconds
        self._exact = {}
        self.start_spin, self.end_spin = TrialBoundarySpin(), TrialBoundarySpin()
        self.seek_section, seek_layout = create_hagisilta_settings_section(tr('영상 위치 이동'))
        seek_grid = QGridLayout()
        self.seek_basis = QComboBox()
        for text, key in ((tr('현재 위치 기준'), "current"), (tr('구간 시작 기준'), "start"), (tr('영상 처음 기준'), "zero")):
            self.seek_basis.addItem(text, key)
        self.seek_amount = TimecodeSpin(10)
        seek_grid.addWidget(self.seek_basis, 0, 0, 1, 3)
        seek_grid.addWidget(self.seek_amount, 1, 0)
        for column, (text, direction) in enumerate(((tr('뒤로 (Alt+A)'), -1), (tr('앞으로 (Alt+D)'), 1)), 1):
            button = QPushButton(text, objectName="toolButton")
            button.setAutoDefault(False)
            button.clicked.connect(lambda _checked=False, d=direction: self.jump(d))
            seek_grid.addWidget(button, 1, column)
        seek_layout.addLayout(seek_grid)
        self.seek_section.setMaximumWidth(480)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        for row, (label, spin, callback, text) in enumerate((
            (tr('시작'), self.start_spin, self.set_start_here, tr('현재 장면 → 시작 (S)')),
            (tr('종료'), self.end_spin, self.set_end_here, tr('현재 장면 → 종료 (F)')),
        )):
            grid.addWidget(QLabel(label, objectName="fieldLabel"), row, 0)
            grid.addWidget(spin, row, 1)
            button = QPushButton(text, objectName="toolButton")
            button.setFixedWidth(208)
            button.setAutoDefault(False)
            button.clicked.connect(callback)
            grid.addWidget(button, row, 2)
            setattr(self, "start_button" if row == 0 else "end_button", button)
        grid.setColumnStretch(3, 1)
        layout.addLayout(grid)
        self.summary = QLabel(objectName="trialRangeSummary")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.mark_feedback = QLabel()
        self.mark_feedback.setWordWrap(True)
        layout.addWidget(self.mark_feedback)
        self.duration_toggle = QToolButton()
        self.duration_toggle.setObjectName("toolButton")
        self.duration_toggle.setText(tr('일정 길이로 종료 설정 (선택)'))
        self.duration_toggle.setCheckable(True)
        self.duration_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.duration_toggle.setArrowType(Qt.RightArrow)
        layout.addWidget(self.duration_toggle, 0, Qt.AlignLeft)
        self.duration_fields = QWidget()
        duration_row = QHBoxLayout(self.duration_fields)
        duration_row.setContentsMargins(0, 0, 0, 0)
        self.length = TimecodeSpin(60)
        duration_row.addWidget(self.length)
        self.duration_button = QPushButton(tr('시작 + 이 길이 → 종료'), objectName="toolButton")
        self.duration_button.setAutoDefault(False)
        self.duration_button.clicked.connect(self.apply_duration)
        duration_row.addWidget(self.duration_button)
        duration_row.addStretch()
        self.duration_fields.hide()
        self.duration_toggle.toggled.connect(self.toggle_duration)
        layout.addWidget(self.duration_fields)
        for name, spin in (("start", self.start_spin), ("end", self.end_spin)):
            spin.valueChanged.connect(lambda _value, key=name: self.boundary_changed(key))
        self.end_spin.valueCommitted.connect(self.preview_end)
        self.end_spin.setToolTip(self.end_spin.toolTip() + tr('\n입력을 확정하면 종료 장면으로 이동합니다. 구간 등록 전에는 저장되지 않습니다.'))
        self.duration_button.setToolTip(tr('종료 = 시작 + 입력한 길이로 설정하고 해당 종료 장면으로 이동합니다.'))
        self.set_range(None, None)

    def toggle_duration(self, checked):
        self.duration_fields.setVisible(checked)
        self.duration_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    def boundary_changed(self, name):
        self._exact.pop(name, None)
        self.update_summary()
        self.rangeChanged.emit()

    def boundary(self, name):
        spin = self.start_spin if name == "start" else self.end_spin
        if spin.value() < 0:
            return None
        return self._exact.get(name, spin.value())

    def set_boundary(self, name, value):
        spin = self.start_spin if name == "start" else self.end_spin
        blocked = spin.blockSignals(True)
        spin.setValue(-1 if value is None else round(float(value) * 1000) / 1000)
        spin.blockSignals(blocked)
        spin.pending_text = None
        if value is not None:
            # Keep the exact clicked frame time behind its millisecond display.
            self._exact[name] = float(value)
        else:
            self._exact.pop(name, None)
        self.update_summary()
        self.rangeChanged.emit()

    def set_range(self, start, end):
        self.set_boundary("start", start)
        self.set_boundary("end", end)

    def preview_end(self):
        end = self.boundary("end")
        if end is None:
            return
        try:
            validate_range(self.boundary("start") or 0, end, self.duration)
        except ValueError as error:
            self.summary.setText(str(error))
            return
        # The viewer clamps duration itself to the last available frame without
        # changing the exact stored (exclusive) end time.
        self.seek_seconds(end)
        self.mark_feedback.setText(tr('종료 장면 확인 · {p0}', p0=format_time(end)))

    def finish_editing(self):
        for spin in (self.start_spin, self.end_spin, self.seek_amount, self.length):
            spin.interpretText()

    def set_start_here(self):
        self.mark_here("start")

    def set_end_here(self):
        self.mark_here("end")

    def mark_here(self, name):
        seconds = self.current_seconds()
        self.set_boundary(name, seconds)
        label = tr('시작') if name == "start" else tr('종료')
        self.mark_feedback.setText(
            tr('{p0} 지정 · {p1} · {p2}프레임', p0=label, p1=format_time(seconds), p2=round(seconds * self.fps) + 1)
        )

    def apply_duration(self):
        self.finish_editing()
        start = self.boundary("start")
        if start is None:
            self.summary.setText(tr('시작을 먼저 지정하세요.'))
            return
        end = start + self.length.value()
        try:
            validate_range(start, end, self.duration)
        except ValueError as error:
            self.summary.setText(str(error))
            return
        self.set_boundary("end", end)
        self.preview_end()

    def jump(self, direction):
        self.finish_editing()
        basis = self.seek_basis.currentData()
        start = self.boundary("start")
        if basis == "start" and start is None:
            self.summary.setText(tr('이동 기준이 될 시작을 먼저 지정하세요.'))
            return
        base = self.current_seconds() if basis == "current" else start if basis == "start" else 0
        self.seek_seconds(max(0, min(base + direction * self.seek_amount.value(), self.duration - 1 / self.fps)))

    def update_summary(self):
        if not hasattr(self, "summary"):
            return
        start, end = self.boundary("start"), self.boundary("end")
        if start is None or end is None:
            self.summary.setText(tr('분석 길이 · —'))
            return
        try:
            validate_range(start, end, self.duration)
            text = tr('분석 길이 · ') + format_time(end - start)
        except ValueError as error:
            text = str(error)
        self.summary.setText(text)


class WaterMazeAnalysisRangeDialog(QDialog):
    """Register individual trials in one continuous, unchanged Arena video."""
    EDITABLE_KEYS = (
        "segment_id", "subject_id", "group", "day", "trial", "trial_type",
        "start_direction", "target_quadrant", "analysis_start_seconds",
        "analysis_end_seconds",
    )

    def __init__(self, video: Path, segments: list[dict], selected_segment_id: str = "", parent=None, *, on_save=None, draft_defaults=None, profile_path: Path | None = None):
        super().__init__(parent)
        # Import lazily: the platform imports the Water-maze page during startup.
        from hagisilta_platform import MultiRowContextTableWidget, add_resizable_table
        self.host = getattr(parent, "host", None)
        self.on_save = on_save
        self.video = Path(video)
        self.capture = cv2.VideoCapture(str(self.video))
        if not self.capture.isOpened():
            raise OSError(tr('영상을 열 수 없습니다: {p0}', p0=self.video))
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 30)
        self.frame_count = max(1, int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
        self.duration = self.frame_count / self.fps
        self.current_frame, self.current_segment_index = 0, -1
        self.current_pixmap = QPixmap()
        self.roi_overlay = None
        roi_message = ""
        try:
            self.roi_overlay = load_water_roi_overlay(profile_path, self.video, (
                int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ))
        except (OSError, ValueError) as error:
            roi_message = str(error)
        self.loading_segment = False
        self.playback_speed = 1.0
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.advance_frame)
        self.segments = [{key: segment.get(key) for key in self.EDITABLE_KEYS} for segment in segments]
        for number, segment in enumerate(self.segments, 1):
            segment["segment_id"] = str(segment.get("segment_id") or f"trial_{number:03d}")
            for key, default in (("day", 1), ("trial", 1), ("trial_type", "Hidden"),
                                 ("start_direction", "N"), ("target_quadrant", "NE"),
                                 ("group", ""), ("subject_id", "")):
                if segment.get(key) is None:
                    segment[key] = default
            if segment.get("analysis_start_seconds") is None:
                segment["analysis_start_seconds"] = 0.0
            if segment.get("analysis_end_seconds") is None:
                segment["analysis_end_seconds"] = self.duration
        self.draft = None
        self.setObjectName("waterTrialEditor")
        self.setWindowTitle(tr('HAGISILTA · Water-maze 개체별 구간 등록 · {p0}', p0=self.video.name))
        self.resize(1440, 960)
        apply_hagisilta_window_identity(self)
        enable_full_window_controls(self)
        self.setStyleSheet(self.styleSheet() + hagisilta_control_stylesheet() + """
            QDialog#waterTrialEditor QLabel, QDialog#waterTrialEditor QPushButton,
            QDialog#waterTrialEditor QToolButton, QDialog#waterTrialEditor QLineEdit,
            QDialog#waterTrialEditor QComboBox, QDialog#waterTrialEditor QSpinBox,
            QDialog#waterTrialEditor QCheckBox,
            QDialog#waterTrialEditor QTableWidget {font-size:13px; font-style:normal;}
            QDialog#waterTrialEditor QLabel#detachedTitle {font-size:18px; font-weight:650;}
            QDialog#waterTrialEditor QLabel#detachedKicker {font-size:9px; font-weight:700;}
        """)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        header, header_layout = create_hagisilta_header(tr('Water-maze 개체별 구간 등록'), "HAGISILTA · ANALYSIS RANGE")
        help_text = (
            tr('한 영상에 같은 실험군의 여러 개체를 연속 촬영한 경우, 개체마다 시작·종료와 Trial을 등록합니다. 쥐를 교체하는 시간은 구간 밖에 둡니다. 원본 영상·수조 ROI·추론 좌표는 변경하지 않습니다.\n\n영상 위 ROI 표시는 이 Arena에 저장한 수조·플랫폼·사분면·벽면 구역입니다. 각 항목을 켜고 끌 수 있으며 표시만 바뀌고 저장된 ROI나 분석 설정은 바뀌지 않습니다. ROI가 없거나 영상과 맞지 않으면 표시하지 않습니다. 먼저 수조·플랫폼 편집에서 확인하세요.\n\n새 구간의 개체 ID는 실험군 이름 + 번호(예: 대조군_01)로 자동 입력됩니다. 구간을 등록한 뒤 다음 개체로 넘어가면 번호만 증가하며, 시작·종료는 미지정이고 Trial 기본값은 1입니다. 실험군·Day·시험 조건은 직전 구간에서 이어받으므로 다음 개체에 맞는지 확인하세요. 같은 개체의 재시험은 ID를 유지하고 Trial을 직접 지정합니다.\n자동 번호는 이 영상에 등록한 같은 군·Day의 마지막 번호 다음부터 제안합니다. 새 영상·다른 Day는 등록된 번호가 없으면 01부터 시작합니다. 촬영 순서가 다르면 번호를 직접 바꾸세요. ID를 직접 입력하면 자동 입력이 꺼집니다. 기존에 등록한 다른 개체 ID를 자동으로 바꾸지는 않습니다.\n시작과 종료는 독립적입니다. 일정 길이는 버튼을 눌렀을 때만 종료에 한 번 적용됩니다. 시간 입력은 분:초.밀리초입니다.\n\n종료 시간을 입력하고 Enter를 누르거나 다른 입력란으로 이동하면 영상도 종료 장면으로 이동합니다. 일정 길이를 적용해도 종료 장면으로 이동합니다. 다음 개체로 넘길 때는 등록한 종료 위치부터 탐색을 이어가며, 새 구간의 시작·종료는 미지정으로 유지됩니다. 종료가 영상 끝이면 마지막 프레임을 보여줍니다.\n\n시작·종료를 지정하면 해당 시각과 프레임을 바로 표시합니다. 기존 구간의 수정 값은 표에 *로 미리 표시되며, 수정 저장(Q) 또는 전체 저장 전에는 등록 데이터가 바뀌지 않습니다. 수정 취소는 원래 값으로 되돌립니다.\n\n재생/정지 (Space) · 1프레임 이동 (A/D 또는 ←/→) · 1초 이동 (Shift+A/D) · 빠른 이동량만큼 이동 (Alt+A/D) · 시작/종료 지정 (S/F) · 현재 구간 등록·수정 (Q) · 현재 구간 반영 후 다음 개체 (E) · 새 구간 (Ctrl+N) · 현재 등록·수정 버튼 (Ctrl+Enter) · 배속 ([/] 또는 Ctrl+[/Ctrl+]) · 저장만 (Ctrl+S)\n\n문자·숫자 입력란과 콤보박스에서는 일반 글자 단축키를 실행하지 않습니다. Ctrl 단축키는 입력 중에도 쓸 수 있습니다. 개체 ID 입력 후 Enter를 누르거나 영상을 클릭하면 영상 조작으로 돌아갑니다. 저장하고 닫기는 전체 구간을 저장한 후 창을 닫으며, Ctrl+S는 창을 닫지 않습니다. 취소는 마지막 저장 이후의 변경만 버립니다.\n\n')
            + tr(PLAYBACK_HELP)
        )
        header_layout.addWidget(create_hagisilta_help_button(self, tr('개체별 구간 등록 도움말'), help_text))
        self.header_status = HeaderContext(header, tr('{p0}개 구간', p0=len(self.segments)))
        outer.addWidget(header)
        context = QFrame()
        context_row = QHBoxLayout(context)
        context_row.setContentsMargins(18, 10, 18, 10)
        video_label = QLabel(tr('영상 · {p0}', p0=self.video.name))
        video_label.setToolTip(str(self.video))
        video_label.setMinimumWidth(0)
        video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        context_row.addWidget(video_label, 1)
        outer.addWidget(context)
        content_frame = QFrame(objectName="workspace")
        content = QHBoxLayout(content_frame)
        content.setContentsMargins(18, 12, 18, 12)
        video_widget = QWidget()
        video_panel = QVBoxLayout(video_widget)
        video_panel.setContentsMargins(0, 0, 0, 0)
        roi_row = QHBoxLayout()
        roi_row.setSpacing(12)
        roi_row.addWidget(QLabel(tr('ROI 표시')))
        self.roi_checks = {}
        available = self.roi_overlay.available if self.roi_overlay else frozenset()
        for key, label in (("pool", tr('수조')), ("platform", tr('플랫폼')),
                           ("quadrants", tr('사분면')), ("wall", tr('벽면 구역'))):
            check = QCheckBox(label)
            check.setObjectName(f"waterRangeRoi_{key}")
            check.setEnabled(key in available)
            check.setChecked(key in available)
            check.setToolTip(
                tr('저장된 {p0} 표시 켜기/끄기 · 보기만 바뀌며 ROI·분석 설정은 변경하지 않습니다.', p0=label)
                if key in available else tr('저장된 {p0} 설정이 없거나 표시할 수 없습니다.', p0=label)
            )
            check.toggled.connect(self.render_preview)
            self.roi_checks[key] = check
            roi_row.addWidget(check)
        roi_row.addStretch()
        video_panel.addLayout(roi_row)
        self.roi_message = QLabel(roi_message, video_widget)
        self.roi_message.setWordWrap(True)
        self.roi_message.setVisible(bool(roi_message))
        video_panel.addWidget(self.roi_message)
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(480, 280)
        self.preview.setFocusPolicy(Qt.StrongFocus)
        self.preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.preview.installEventFilter(self)
        self.preview.setStyleSheet("background:#111217;")
        video_panel.addWidget(self.preview, 1)
        self.slider = ClickSeekSlider(Qt.Horizontal)
        self.slider.setRange(0, self.frame_count - 1)
        self.slider.valueChanged.connect(self.seek_frame)
        video_panel.addWidget(self.slider)
        clock_row = QHBoxLayout()
        self.clock = create_range_clock()
        clock_row.addWidget(self.clock)
        clock_row.addStretch()
        self.playback_control = PlaybackSpeedControl(self)
        self.speed_combo = self.playback_control.combo
        self.speed_combo.currentTextChanged.connect(self.change_playback_speed)
        clock_row.addWidget(self.playback_control)
        video_panel.addLayout(clock_row)
        transport = QHBoxLayout()
        transport.setSpacing(8)
        self.play_button = QPushButton(tr('재생 (Space)'), objectName="secondaryButton")
        self.play_button.setFixedWidth(110)
        self.play_button.setAutoDefault(False)
        self.play_button.clicked.connect(self.toggle_play)
        transport.addWidget(self.play_button)
        for text, delta, width in ((tr('-1초 (Shift+A)'), -self.fps, 108), ("-1F (A)", -1, 72),
                                   ("+1F (D)", 1, 72), (tr('+1초 (Shift+D)'), self.fps, 108)):
            button = QPushButton(text, objectName="secondaryButton")
            button.setFixedWidth(width)
            button.setAutoDefault(False)
            button.clicked.connect(lambda _checked=False, step=delta: self.step_frames(round(step)))
            transport.addWidget(button)
        transport.addStretch()
        video_panel.addLayout(transport)
        self.range_controls = WaterTrialRangeControls(
            self.duration, self.fps, self.boundary_time,
            self.seek_seconds, self,
        )
        video_panel.addWidget(self.range_controls.seek_section)
        self.start_spin, self.end_spin = self.range_controls.start_spin, self.range_controls.end_spin
        editor_widget = QWidget()
        editor = QVBoxLayout(editor_widget)
        editor.setContentsMargins(12, 12, 12, 12)
        editor.setSpacing(12)
        heading = QHBoxLayout()
        heading.addWidget(QLabel(tr('등록한 구간'), objectName="sectionLabel"))
        self.count_label = QLabel()
        heading.addWidget(self.count_label)
        heading.addStretch()
        self.new_button = QPushButton(tr('새 구간 (Ctrl+N)'), objectName="toolButton")
        self.new_button.setAutoDefault(False)
        self.new_button.clicked.connect(self.new_segment)
        heading.addWidget(self.new_button)
        editor.addLayout(heading)
        self.segment_table = MultiRowContextTableWidget(0, 5)
        self.segment_table.setHorizontalHeaderLabels([tr('개체 ID'), "Trial", tr('시작'), tr('종료'), tr('길이')])
        self.segment_table.verticalHeader().setVisible(False)
        self.segment_table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.segment_table.verticalHeader().setDefaultSectionSize(36)
        self.segment_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.segment_table.horizontalHeader().setStretchLastSection(True)
        self.segment_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.segment_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.segment_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.segment_table.setAlternatingRowColors(True)
        for col, width in enumerate((116, 58, 116, 116, 110)):
            self.segment_table.setColumnWidth(col, width)
        self.segment_table.setSortingEnabled(True)
        self.segment_table.sortItems(2, Qt.AscendingOrder)
        self.segment_table.currentCellChanged.connect(self.select_segment)
        self.segment_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.segment_table.customContextMenuRequested.connect(self.show_segment_context_menu)
        add_resizable_table(editor, self.segment_table, "ui/table_heights/watermaze_segment_editor",
                            min_height=120, max_height=360, default_height=160)
        self.info_section, info_layout = create_hagisilta_settings_section(tr('새 구간 등록'))
        self.editor_title = self.info_section.findChild(QLabel)
        if self.editor_title:
            self.editor_title.setText(tr('새 구간 등록'))
        group_row = QHBoxLayout()
        group_row.setSpacing(12)
        group_row.addWidget(QLabel(tr('실험군'), objectName="fieldLabel"))
        self.group_edit = QLineEdit()
        self.group_edit.setFixedWidth(168)
        self.group_edit.setPlaceholderText(tr('예: 대조군, 약물군'))
        self.group_edit.setToolTip(tr('자동 ID 앞에 붙일 군 이름입니다. 다음 개체에도 이어집니다.\n현재 구간의 실험군만 수정하며 이미 등록한 다른 구간은 바꾸지 않습니다.'))
        self.group_edit.returnPressed.connect(lambda: self.preview.setFocus(Qt.ShortcutFocusReason))
        group_row.addWidget(self.group_edit)
        group_row.addStretch()
        info_layout.addLayout(group_row)
        self.subject_auto_row = QWidget()
        subject_auto_layout = QHBoxLayout(self.subject_auto_row)
        subject_auto_layout.setContentsMargins(0, 0, 0, 0)
        subject_auto_layout.setSpacing(14)
        self.subject_auto = QCheckBox(tr('군 이름 + 번호 자동 입력'))
        self.subject_auto.setChecked(True)
        self.subject_auto.setToolTip(tr('실험군을 입력하면 대조군_01처럼 ID를 만듭니다. 다음 개체 등록 시 번호만 증가합니다.\nID를 직접 입력하면 자동 입력이 꺼집니다. 기존 등록 ID는 바뀌지 않습니다.'))
        self.subject_number = QSpinBox()
        self.subject_number.setRange(1, 999999)
        self.subject_number.setFixedWidth(100)
        self.subject_number.setToolTip(tr('현재 마우스 번호입니다. 촬영 순서가 다르면 실제 개체의 번호로 수정하세요.'))
        subject_auto_layout.addWidget(self.subject_auto)
        subject_auto_layout.addWidget(QLabel(tr('마우스 번호')))
        subject_auto_layout.addWidget(self.subject_number)
        subject_auto_layout.addStretch()
        info_layout.addWidget(self.subject_auto_row)
        fields = QGridLayout()
        fields.setHorizontalSpacing(12)
        self.subject_edit = QLineEdit()
        self.subject_edit.setPlaceholderText(tr('개체 ID 입력'))
        self.subject_edit.setFixedWidth(168)
        self.subject_edit.setToolTip(tr('자동 입력된 ID도 직접 수정할 수 있습니다. 직접 입력하면 자동 입력을 끕니다.\n같은 쥐는 다른 Day에도 같은 ID를 유지하세요.'))
        self.subject_edit.returnPressed.connect(lambda: self.preview.setFocus(Qt.ShortcutFocusReason))
        self.day_spin = QSpinBox(); self.day_spin.setRange(1, 9999)
        self.trial_spin = QSpinBox(); self.trial_spin.setRange(1, 9999)
        self.trial_spin.setToolTip(tr('현재 개체의 Trial 번호입니다. 다음 개체는 1부터 시작하며 자동 증가하지 않습니다.'))
        self.trial_type_combo = QComboBox(); self.trial_type_combo.addItems(["Hidden", "Probe", "Visible", "Reversal"])
        self.start_direction_combo = QComboBox(); self.start_direction_combo.addItems(["N", "E", "S", "W"])
        self.target_quadrant_combo = QComboBox(); self.target_quadrant_combo.addItems(["NE", "SE", "SW", "NW"])
        self.trial_type_combo.setToolTip(tr(TRIAL_TYPE_TOOLTIP))
        self.start_direction_combo.setToolTip(tr(START_DIRECTION_TOOLTIP))
        self.target_quadrant_combo.setToolTip(tr(TARGET_QUADRANT_TOOLTIP))
        for index, (label, control) in enumerate((
            (tr('개체 ID'), self.subject_edit), ("Trial", self.trial_spin), ("Day", self.day_spin),
            (tr('시험 유형'), self.trial_type_combo), (tr('시작 방향'), self.start_direction_combo),
            (tr('목표 사분면'), self.target_quadrant_combo),
        )):
            row, col = divmod(index, 3)
            control.setFixedWidth(168 if col == 0 else 130)
            fields.addWidget(QLabel(label, objectName="fieldLabel"), row * 2, col)
            fields.addWidget(control, row * 2 + 1, col)
        fields.setColumnStretch(3, 1)
        info_layout.addLayout(fields)
        info_layout.addWidget(self.range_controls)
        actions = QHBoxLayout()
        self.commit_button = QPushButton(objectName="primaryButton")
        self.commit_button.setFixedWidth(220)
        self.commit_button.setAutoDefault(False)
        self.commit_button.clicked.connect(self.primary_action)
        self.cancel_edit_button = QPushButton(tr('수정 취소'), objectName="toolButton")
        self.cancel_edit_button.setAutoDefault(False)
        self.cancel_edit_button.clicked.connect(self.cancel_current_edit)
        self.next_button = QPushButton(tr('다음 개체 (E)'), objectName="toolButton")
        self.next_button.setAutoDefault(False)
        self.next_button.clicked.connect(self.add_segment)
        actions.addWidget(self.cancel_edit_button)
        actions.addWidget(self.next_button)
        actions.addStretch()
        actions.addWidget(self.commit_button)
        info_layout.addLayout(actions)
        editor.addWidget(self.info_section)
        self.message = QLabel(objectName="trialEditorMessage")
        self.message.setWordWrap(True)
        editor.addWidget(self.message)
        self.range_controls.rangeChanged.connect(self.preview_form_changes)
        for control in (self.subject_edit, self.group_edit):
            control.textChanged.connect(self.preview_form_changes)
        for control in (self.day_spin, self.trial_spin):
            control.valueChanged.connect(self.preview_form_changes)
        for control in (self.trial_type_combo, self.start_direction_combo, self.target_quadrant_combo):
            control.currentTextChanged.connect(self.preview_form_changes)
        self.subject_auto.toggled.connect(self.subject_auto_toggled)
        self.subject_number.valueChanged.connect(self.update_auto_subject)
        self.subject_edit.textEdited.connect(self.subject_edited_manually)
        self.group_edit.textChanged.connect(self.subject_context_changed)
        self.day_spin.valueChanged.connect(self.subject_context_changed)
        editor.addStretch()
        self.inspector = create_range_inspector(editor_widget, minimum_width=594)
        self.range_splitter = create_range_splitter(video_widget, self.inspector, inspector_width=650)
        self.range_splitter.splitterMoved.connect(lambda *_: self.render_preview())
        content.addWidget(self.range_splitter)
        outer.addWidget(content_frame, 1)
        footer_frame = QFrame(objectName="detachedFooter")
        footer = QHBoxLayout(footer_frame)
        footer.setContentsMargins(18, 9, 18, 9)
        footer.addWidget(QLabel(tr('원본 영상 유지 · 구간·개체 정보만 저장')))
        footer.addStretch()
        self.cancel_button = QPushButton(tr('취소'), objectName="secondaryButton")
        self.cancel_button.setAutoDefault(False)
        self.cancel_button.clicked.connect(self.reject)
        self.save_button = QPushButton(tr('저장하고 닫기'), objectName="primaryButton")
        self.save_button.setFixedWidth(220)
        self.save_button.setAutoDefault(False)
        self.save_button.clicked.connect(self.validate_and_accept)
        footer.addWidget(self.cancel_button)
        footer.addWidget(self.save_button)
        outer.addWidget(footer_frame)
        # Ctrl shortcuts work inside editors; unmodified letters never edit metadata.
        self.shortcuts = []
        bindings = {
            "Space": self.toggle_play, "A": lambda: self.step_frames(-1),
            "D": lambda: self.step_frames(1), "Left": lambda: self.step_frames(-1),
            "Right": lambda: self.step_frames(1),
            "Shift+A": lambda: self.step_frames(-round(self.fps)),
            "Shift+D": lambda: self.step_frames(round(self.fps)),
            "S": self.range_controls.set_start_here, "F": self.range_controls.set_end_here,
            "Q": self.apply_current_segment, "E": self.add_segment,
            "[": lambda: self.cycle_playback_speed(-1), "]": lambda: self.cycle_playback_speed(1),
            "Alt+A": lambda: self.range_controls.jump(-1), "Alt+D": lambda: self.range_controls.jump(1),
            "Ctrl+N": self.new_segment, "Ctrl+Return": self.primary_action,
            "Ctrl+Enter": self.primary_action, "Ctrl+S": self.save_progress,
        }
        for sequence, callback in bindings.items():
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.setAutoRepeat(sequence in ("A", "D", "Left", "Right", "Shift+A", "Shift+D"))
            shortcut.activated.connect(lambda fn=callback, key=sequence: self.trigger_shortcut(key, fn))
            self.shortcuts.append(shortcut)
        selected_index = next((i for i, item in enumerate(self.segments) if item["segment_id"] == selected_segment_id), 0)
        self.refresh_segment_table(selected_index if self.segments else None)
        if not self.segments:
            self.load_form(self._blank_draft(draft_defaults or {}))
            self.new_segment()
        self.seek_frame(round((self.segments[selected_index]["analysis_start_seconds"] if self.segments else 0) * self.fps))
        # No implicit Enter target (including the shared help button): Enter in
        # the ID editor returns focus to video, never opens/closes a dialog.
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.render_preview()

    def eventFilter(self, watched, event):
        if watched is getattr(self, "preview", None) and event.type() == QEvent.Resize:
            self.render_preview()
        return super().eventFilter(watched, event)

    def render_preview(self) -> None:
        if self.current_pixmap.isNull():
            return
        fitted = self.current_pixmap.scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.preview.setPixmap(paint_water_roi_overlay(
            fitted, self.roi_overlay,
            {key for key, check in self.roi_checks.items() if check.isChecked()},
        ))

    def seek_frame(self, frame_index: int) -> None:
        self.current_frame = min(max(int(frame_index), 0), self.frame_count - 1)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ok, frame = self.capture.read()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888).copy()
            self.current_pixmap = QPixmap.fromImage(image)
            self.render_preview()
        if self.slider.value() != self.current_frame:
            self.slider.blockSignals(True)
            self.slider.setValue(self.current_frame)
            self.slider.blockSignals(False)
        self.clock.setText(
            f"{format_clock(self.current_frame / self.fps)} / {format_clock(self.duration)}"
            f"  ·  {self.current_frame + 1}/{self.frame_count}F"
        )

    def step_frames(self, delta: int) -> None:
        self.stop_playback()
        self.slider.setValue(min(max(self.current_frame + delta, 0), self.frame_count - 1))

    def seek_seconds(self, seconds: float) -> None:
        self.stop_playback()
        self.seek_frame(round(seconds * self.fps))

    def toggle_play(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText(tr('재생 (Space)'))
        else:
            self.timer.start(self.playback_interval_ms())
            self.play_button.setText(tr('일시정지 (Space)'))

    def playback_interval_ms(self) -> int:
        return playback_interval_ms(self.fps, self.playback_speed)

    def change_playback_speed(self, text: str) -> None:
        try:
            self.playback_speed = float(text.replace("×", ""))
        except ValueError:
            self.playback_speed = 1.0
        self.timer.setInterval(self.playback_interval_ms())

    def cycle_playback_speed(self, direction: int) -> None:
        self.playback_control.cycle(direction)

    def trigger_shortcut(self, sequence, callback):
        focus = QApplication.focusWidget()
        if not sequence.startswith(("Ctrl+", "Alt+")) and isinstance(focus, (QLineEdit, QAbstractSpinBox, QComboBox)):
            return
        callback()

    def boundary_time(self):
        self.stop_playback()
        return self.current_frame / self.fps

    def stop_playback(self):
        self.timer.stop()
        self.play_button.setText(tr('재생 (Space)'))

    def advance_frame(self):
        if self.current_frame >= self.frame_count - 1:
            self.stop_playback()
        else:
            self.slider.setValue(self.current_frame + 1)

    def form_values(self, *, interpret=True):
        if interpret:
            self.range_controls.finish_editing()
            for spin in (self.day_spin, self.trial_spin, self.subject_number):
                spin.interpretText()
        return {
            "subject_id": self.subject_edit.text().strip(), "group": self.group_edit.text().strip(),
            "day": self.day_spin.value(), "trial": self.trial_spin.value(),
            "trial_type": self.trial_type_combo.currentText(),
            "start_direction": self.start_direction_combo.currentText(),
            "target_quadrant": self.target_quadrant_combo.currentText(),
            "analysis_start_seconds": self.range_controls.boundary("start"),
            "analysis_end_seconds": self.range_controls.boundary("end"),
        }

    @staticmethod
    def automatic_subject_id(group, number):
        group = str(group or "").strip()
        return f"{group}_{number:02d}" if group else ""

    def next_subject_number(self, group, day):
        prefix = str(group or "").strip() + "_"
        numbers = []
        for segment in self.segments:
            identifier = str(segment.get("subject_id") or "")
            if segment.get("group", "").strip() == str(group or "").strip() and segment.get("day", 1) == day and identifier.startswith(prefix):
                suffix = identifier[len(prefix):]
                if suffix.isascii() and suffix.isdecimal():
                    numbers.append(int(suffix))
        return min(max(numbers, default=0) + 1, self.subject_number.maximum())

    def draft_values(self):
        values = self.form_values()
        values.update(_auto_subject=self.subject_auto.isChecked(), _subject_number=self.subject_number.value())
        return values

    @classmethod
    def draft_has_content(cls, values):
        # An untouched, automatically suggested ID is not an unfinished Trial.
        generated = cls.automatic_subject_id(values.get("group"), values.get("_subject_number", 1))
        automatic = values.get("_auto_subject", False) and values.get("subject_id", "") == generated
        return (values.get("analysis_start_seconds") is not None or values.get("analysis_end_seconds") is not None
                or bool(values.get("subject_id")) and not automatic)

    def update_auto_subject(self, *_args):
        if self.loading_segment or self.current_segment_index >= 0:
            return
        automatic = self.subject_auto.isChecked()
        self.subject_number.setEnabled(automatic)
        if automatic:
            self.subject_edit.setText(self.automatic_subject_id(self.group_edit.text(), self.subject_number.value()))
        self.preview_form_changes()

    def subject_context_changed(self, *_args):
        if self.loading_segment or self.current_segment_index >= 0 or not self.subject_auto.isChecked():
            return
        self.subject_number.setValue(self.next_subject_number(self.group_edit.text(), self.day_spin.value()))
        self.update_auto_subject()

    def subject_auto_toggled(self, checked):
        if checked:
            self.subject_context_changed()
        self.update_auto_subject()

    def subject_edited_manually(self, *_args):
        if not self.loading_segment and self.current_segment_index < 0:
            self.subject_auto.setChecked(False)

    def _blank_draft(self, source):
        group, day = source.get("group", ""), source.get("day", 1)
        automatic = source.get("_auto_subject", self.subject_auto.isChecked())
        number = self.next_subject_number(group, day)
        return dict(subject_id=self.automatic_subject_id(group, number) if automatic else "", group=group, day=day,
                    trial=1, trial_type=source.get("trial_type", "Hidden"),
                    start_direction=source.get("start_direction", "N"),
                    target_quadrant=source.get("target_quadrant", "NE"),
                    analysis_start_seconds=None, analysis_end_seconds=None,
                    _auto_subject=automatic, _subject_number=number)

    def load_form(self, values):
        self.loading_segment = True
        self.subject_edit.setText(str(values.get("subject_id") or ""))
        self.group_edit.setText(str(values.get("group") or ""))
        self.day_spin.setValue(int(values.get("day") or 1))
        self.trial_spin.setValue(int(values.get("trial") or 1))
        self.trial_type_combo.setCurrentText(str(values.get("trial_type") or "Hidden"))
        self.start_direction_combo.setCurrentText(str(values.get("start_direction") or "N"))
        self.target_quadrant_combo.setCurrentText(str(values.get("target_quadrant") or "NE"))
        self.range_controls.set_range(values.get("analysis_start_seconds"), values.get("analysis_end_seconds"))
        editing = self.current_segment_index >= 0
        self.subject_auto_row.setVisible(not editing)
        if not editing:
            self.subject_auto.setChecked(values.get("_auto_subject", True))
            self.subject_number.setValue(values.get("_subject_number", self.next_subject_number(values.get("group"), values.get("day", 1))))
            self.subject_number.setEnabled(self.subject_auto.isChecked())
        self.loading_segment = False
        if self.editor_title:
            self.editor_title.setText(tr('선택 구간 수정') if editing else tr('새 구간 등록'))
        self.commit_button.setText(tr('수정 저장 (Q)') if editing else tr('구간 등록 · 다음 개체 (E)'))
        self.commit_button.setFixedWidth(max(220, min(420,
            self.commit_button.fontMetrics().horizontalAdvance(self.commit_button.text()) + 56)))
        self.cancel_edit_button.setVisible(editing)
        self.next_button.setVisible(editing)
        self.range_controls.mark_feedback.setText(tr('영상에서 장면을 찾은 뒤 시작(S)·종료(F)를 지정하세요.'))
        self.message.setText(tr('수정한 값은 표에 *로 미리 표시됩니다. 수정 저장(Q)으로 목록에 반영하세요.') if editing
                             else tr('미등록 구간입니다. 개체 ID와 시작·종료를 지정한 뒤 구간 등록을 누르세요.'))
        self.refresh_segment_table()

    def preview_form_changes(self, *_args):
        if self.loading_segment:
            return
        self.refresh_segment_table()
        if self.current_segment_index >= 0:
            current = self.segments[self.current_segment_index]
            pending = any(current.get(key) != value for key, value in self.form_values(interpret=False).items())
            self.message.setText(tr('* 수정 미반영 · 표는 입력값 미리보기입니다. 수정 저장(Q) 또는 전체 저장으로 반영하세요.')
                                 if pending else tr('현재 구간의 값과 같습니다. 다른 장면으로 이동한 뒤 시작·종료를 지정할 수 있습니다.'))
        else:
            self.message.setText(tr('아직 등록 전입니다. 개체 ID와 시작·종료를 지정한 뒤 구간 등록을 누르세요.'))

    def load_segment(self, index):
        if not 0 <= index < len(self.segments):
            return
        self.stop_playback()
        self.current_segment_index = index
        self.load_form(self.segments[index])
        self.seek_frame(round(self.start_spin.value() * self.fps))

    def _index_at_row(self, row):
        item = self.segment_table.item(row, 0)
        key = item.data(Qt.UserRole + 1) if item else None
        return next((i for i, segment in enumerate(self.segments) if segment["segment_id"] == key), -1)

    def refresh_segment_table(self, selected_index=None):
        from hagisilta_platform import SortableTableItem
        table = self.segment_table
        selected_keys = {table.item(index.row(), 0).data(Qt.UserRole + 1) for index in table.selectionModel().selectedRows()}
        current_key = table.item(table.currentRow(), 0).data(Qt.UserRole + 1) if table.currentRow() >= 0 and table.item(table.currentRow(), 0) else None
        if selected_index is not None and 0 <= selected_index < len(self.segments):
            current_key = self.segments[selected_index]["segment_id"]
            selected_keys = {current_key}
        sorting = table.isSortingEnabled()
        table.blockSignals(True)
        table.setSortingEnabled(False)
        table.setRowCount(len(self.segments))
        for row, saved_segment in enumerate(self.segments):
            segment = saved_segment
            pending = False
            if row == self.current_segment_index and not self.loading_segment:
                values = self.form_values(interpret=False)
                pending = any(saved_segment.get(key) != value for key, value in values.items())
                if pending:
                    segment = dict(saved_segment, **values)
            start, end = segment["analysis_start_seconds"], segment["analysis_end_seconds"]
            length = end - start if start is not None and end is not None and end > start else None
            display_time = lambda value: format_time(value) if value is not None else tr('미지정')
            values = (segment["subject_id"] + (" *" if pending else ""), segment["trial"], display_time(start),
                      display_time(end), format_time(length) if length is not None else "—")
            keys = (segment["subject_id"], segment["trial"], start if start is not None else -1,
                    end if end is not None else -1, length if length is not None else -1)
            tooltip = (tr('{p0} · Trial {p1} · 실험군 {p2}\nDay {p3} · {p4} · 시작 {p5} · 목표 {p6}\n{p7} → {p8}', p0=segment['subject_id'], p1=segment['trial'], p2=segment['group'] or tr('미지정'), p3=segment['day'], p4=segment['trial_type'], p5=segment['start_direction'], p6=segment['target_quadrant'], p7=display_time(start), p8=display_time(end))
                       + (tr('\n* 수정 미반영 · 수정 저장(Q)으로 목록에 반영') if pending else ""))
            for col, value in enumerate(values):
                item = SortableTableItem(str(value))
                item.setData(Qt.UserRole, keys[col])
                item.setData(Qt.UserRole + 1, segment["segment_id"])
                item.setToolTip(tooltip)
                item.setTextAlignment(Qt.AlignVCenter | (Qt.AlignLeft if col == 0 else Qt.AlignCenter))
                table.setItem(row, col, item)
        table.setSortingEnabled(sorting)
        table.clearSelection()
        from PySide6.QtCore import QItemSelectionModel
        for row in range(table.rowCount()):
            key = table.item(row, 0).data(Qt.UserRole + 1)
            index = table.model().index(row, 0)
            if key in selected_keys:
                table.selectionModel().select(index, QItemSelectionModel.Select | QItemSelectionModel.Rows)
            if key == current_key:
                table.selectionModel().setCurrentIndex(index, QItemSelectionModel.NoUpdate)
        table.blockSignals(False)
        self.count_label.setText(tr('{p0}개 구간 · {p1}개체', p0=len(self.segments), p1=len({s['subject_id'] for s in self.segments})))
        self.header_status.setText(tr('{p0}개 구간', p0=len(self.segments)))
        if selected_index is not None and 0 <= selected_index < len(self.segments):
            self.load_segment(selected_index)

    def update_segment_table_row(self, _index):
        self.refresh_segment_table()

    def _leave_existing(self):
        if self.current_segment_index < 0:
            self.draft = self.draft_values()
            return True
        values = self.form_values()
        current = self.segments[self.current_segment_index]
        if all(current.get(key) == value for key, value in values.items()):
            return True
        answer = QMessageBox.question(self, tr('구간 수정 확인'),
            tr('현재 구간의 수정 내용을 목록에 반영할까요?'),
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Cancel)
        if answer == QMessageBox.Cancel:
            return False
        return self.commit_current_segment() if answer == QMessageBox.Save else True

    def select_segment(self, current_row, _column, _previous_row, _previous_column):
        if self.loading_segment:
            return
        index = self._index_at_row(current_row)
        if index < 0 or index == self.current_segment_index:
            return
        if not self._leave_existing():
            table = self.segment_table
            table.blockSignals(True)
            table.clearSelection()
            for row in range(table.rowCount()):
                if self._index_at_row(row) == self.current_segment_index:
                    table.setCurrentCell(row, 0)
                    table.selectRow(row)
                    break
            table.blockSignals(False)
            return
        self.load_segment(index)

    def new_segment(self):
        if not self._leave_existing():
            return
        self.stop_playback()
        source = self.draft_values()
        self.current_segment_index = -1
        self.draft = self.draft if self.draft is not None else self._blank_draft(source)
        self.segment_table.blockSignals(True)
        self.segment_table.clearSelection()
        self.segment_table.setCurrentCell(-1, -1)
        self.segment_table.blockSignals(False)
        self.load_form(self.draft)
        if self.subject_auto.isChecked():
            (self.preview if self.subject_edit.text() else self.group_edit).setFocus()
        else:
            self.subject_edit.setFocus()

    def _validate_segment(self, segment, exclude=-1):
        if not segment["subject_id"]:
            raise ValueError(tr('개체 ID를 입력하세요.'))
        start, end = segment["analysis_start_seconds"], segment["analysis_end_seconds"]
        if start is None or end is None:
            raise ValueError(tr('시작과 종료 장면을 모두 지정하세요.'))
        frame_bounds(start, end, self.fps, self.frame_count)
        for index, other in enumerate(self.segments):
            if index == exclude:
                continue
            if start < other["analysis_end_seconds"] - 1e-9 and end > other["analysis_start_seconds"] + 1e-9:
                raise ValueError(tr('{p0} Trial {p1}과 시간이 겹칩니다. 교체 시간을 제외하고 지정하세요.', p0=other['subject_id'], p1=other['trial']))
            identity = ("subject_id", "day", "trial", "trial_type")
            if all(segment.get(key) == other.get(key) for key in identity):
                raise ValueError(tr('같은 개체·Day·시험 유형의 Trial이 이미 있습니다. Trial 번호를 확인하세요.'))

    def commit_current_segment(self):
        if self.loading_segment:
            return False
        values = self.form_values()
        try:
            self._validate_segment(values, self.current_segment_index)
        except (ValueError, TypeError) as error:
            QMessageBox.warning(self, tr('구간 등록 확인'), str(error))
            return False
        if self.current_segment_index < 0:
            used = {s["segment_id"] for s in self.segments}
            number = 1
            while f"trial_{number:03d}" in used:
                number += 1
            values["segment_id"] = f"trial_{number:03d}"
            self.segments.append(values)
            self.current_segment_index = len(self.segments) - 1
            self.draft = None
        else:
            self.segments[self.current_segment_index].update(values)
        frame = self.current_frame
        self.refresh_segment_table(self.current_segment_index)
        self.seek_frame(frame)
        self.message.setText(tr('구간 목록에 반영했습니다. 전체 저장은 Ctrl+S 또는 저장하고 닫기입니다.'))
        return True

    def apply_current_segment(self):
        return self.commit_current_segment()

    def primary_action(self):
        if self.current_segment_index < 0:
            return self.add_segment()
        return self.apply_current_segment()

    def add_segment(self):
        previous = self.draft_values()
        if not self.commit_current_segment():
            return False
        previous_end = self.segments[self.current_segment_index]["analysis_end_seconds"]
        self.draft = self._blank_draft(previous)
        self.new_segment()
        self.seek_seconds(previous_end)
        self.range_controls.mark_feedback.setText(
            tr('이전 구간 종료 · {p0} · 여기서 다음 개체 장면을 찾으세요.', p0=format_time(previous_end))
        )
        self.message.setText(tr('다음 개체 · {p0} · 시작·종료를 지정하세요.', p0=self.subject_edit.text())
                             if self.subject_auto.isChecked() and self.subject_edit.text() else
                             tr('다음 개체의 ID와 시작·종료를 지정하세요. Trial 기본값은 1입니다.'))
        return True

    def cancel_current_edit(self):
        if self.current_segment_index >= 0:
            self.load_segment(self.current_segment_index)

    def remove_segment(self):
        table_rows = [i.row() for i in self.segment_table.selectionModel().selectedRows()]
        indices = sorted({self._index_at_row(row) for row in table_rows} - {-1})
        if not indices:
            return
        if len(self.segments) - len(indices) < 1:
            QMessageBox.information(self, tr('구간 삭제'), tr('Arena에는 최소 한 개의 등록 구간이 필요합니다.'))
            return
        answer = QMessageBox.question(self, tr('Trial 구간 삭제'),
            tr('선택한 {p0}개 구간을 목록에서 삭제할까요?\n원본 영상·좌표는 유지됩니다. 전체 저장 전에는 프로젝트가 바뀌지 않습니다.', p0=len(indices)),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        for index in reversed(indices):
            self.segments.pop(index)
        self.current_segment_index = -1
        self.refresh_segment_table(min(indices[0], len(self.segments) - 1))

    def show_segment_context_menu(self, position):
        rows = self.segment_table.restore_right_click_rows(position)
        if not rows:
            return
        prefix = tr('이 행') if len(rows) == 1 else tr('선택한 {p0}개 행', p0=len(rows))
        menu = QMenu(self.segment_table)
        edit_action = menu.addAction(tr('{p0} 구간 편집', p0=prefix))
        edit_action.setEnabled(len(rows) == 1)
        menu.addSeparator()
        remove_action = menu.addAction(tr('{p0} Trial 삭제…', p0=prefix))
        chosen = menu.exec(self.segment_table.viewport().mapToGlobal(position))
        if chosen == edit_action:
            index = self._index_at_row(rows[0])
            if index >= 0 and self._leave_existing():
                self.load_segment(index)
        elif chosen == remove_action:
            self.remove_segment()

    def _ready_to_save(self):
        if self.current_segment_index >= 0:
            if not self.commit_current_segment():
                return False
            # A draft can be parked while the user inspects a registered row.
            # Never silently drop that hidden draft when saving and closing.
            if self.draft and self.draft_has_content(self.draft):
                self.new_segment()
                if not self.commit_current_segment():
                    return False
        else:
            values = self.draft_values()
            pending = self.draft_has_content(values)
            if pending and not self.commit_current_segment():
                return False
        if not self.segments:
            QMessageBox.warning(self, tr('구간 저장 확인'), tr('최소 한 개의 구간을 등록하세요.'))
            return False
        for index, segment in enumerate(self.segments):
            try:
                self._validate_segment(segment, index)
            except (ValueError, TypeError) as error:
                QMessageBox.warning(self, tr('구간 저장 확인'), tr('구간 {p0}: {p1}', p0=index + 1, p1=error))
                return False
        return True

    def save_progress(self):
        if not self._ready_to_save():
            return False
        if self.on_save is None:
            self.message.setText(tr('구간 목록에 반영했습니다. 저장하고 닫기로 부모 화면에 전달하세요.'))
            return False
        try:
            self.on_save(self.selected_segments)
        except Exception as error:
            QMessageBox.warning(self, tr('구간 저장 실패'), tr('저장하지 못했습니다. 입력 내용은 유지됩니다.\n{p0}', p0=error))
            return False
        self.message.setText(tr('{p0}개 구간 저장 완료 · 창은 계속 열려 있습니다.', p0=len(self.segments)))
        return True

    def validate_and_accept(self):
        if self.on_save is not None:
            if not self.save_progress():
                return
        elif not self._ready_to_save():
            return
        self.accept()

    def use_full_video(self):
        self.range_controls.set_range(0, self.duration)

    @property
    def selected_segments(self):
        return [dict(segment) for segment in self.segments]

    def closeEvent(self, event):
        self.timer.stop()
        self.capture.release()
        super().closeEvent(event)

    def done(self, result):
        self.timer.stop()
        self.capture.release()
        super().done(result)


def arena_number(path: Path) -> int:
    try:
        return int(path.name.rsplit("_", 1)[-1])
    except Exception:
        return 1


def arena_video(path: Path) -> Path | None:
    from hagisilta_labeled_video import source_video_candidates
    videos = source_video_candidates(path)
    return videos[0] if videos else None


def first_frame(video: Path) -> tuple[QPixmap, int, int]:
    return frame_at(video, 0.0)[:3]


def frame_at(video: Path, seconds: float) -> tuple[QPixmap, int, int, float, int]:
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
    frame_index = min(max(round(max(seconds, 0.0) * fps), 0), frame_count - 1)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise OSError(tr('영상을 읽을 수 없습니다: {p0}', p0=video))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888).copy()
    return QPixmap.fromImage(image), width, height, fps, frame_count


class WaterMazeCanvas(QWidget):
    def __init__(self, pixmap: QPixmap, profile: dict, parent=None) -> None:
        super().__init__(parent)
        self.pixmap = pixmap
        self.profile = profile
        self.tool = "select"
        self.interaction_enabled = True
        self.selected_shape = "pool"
        self.drag_mode = ""
        self.drag_start: QPointF | None = None
        self.drag_snapshot: dict = {}
        self.draw_shape_name = ""
        self.active_handle = -1
        self.rotation_start_angle = 0.0
        self.polygon_draft: list[list[float]] = []
        self.setMinimumSize(620, 500)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setCursor(Qt.ArrowCursor)

    def image_rect(self):
        scaled = self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        left = (self.width() - scaled.width()) / 2
        top = (self.height() - scaled.height()) / 2
        return scaled, left, top

    def to_image(self, point: QPointF) -> QPointF:
        scaled, left, top = self.image_rect()
        x = (point.x() - left) * self.pixmap.width() / max(scaled.width(), 1)
        y = (point.y() - top) * self.pixmap.height() / max(scaled.height(), 1)
        return QPointF(max(0, min(self.pixmap.width(), x)), max(0, min(self.pixmap.height(), y)))

    def to_widget(self, xy) -> QPointF:
        scaled, left, top = self.image_rect()
        return QPointF(
            left + float(xy[0]) * scaled.width() / max(self.pixmap.width(), 1),
            top + float(xy[1]) * scaled.height() / max(self.pixmap.height(), 1),
        )

    def view_scale(self) -> float:
        scaled, _left, _top = self.image_rect()
        return scaled.width() / max(self.pixmap.width(), 1)

    def platform_points(self) -> list[list[float]]:
        platform = self.profile.setdefault("platform", {})
        shape_type = str(platform.get("shape_type", "circle"))
        points = platform.get("points_px") or []
        if shape_type == "circle" or len(points) < 3:
            return []
        return [[float(point[0]), float(point[1])] for point in points]

    @staticmethod
    def pool_radii(target: dict) -> tuple[float, float]:
        fallback = float(target.get("radius_px", 0))
        return (
            float(target.get("radius_x_px", fallback)),
            float(target.get("radius_y_px", fallback)),
        )

    @staticmethod
    def wall_ring_fraction(profile: dict) -> float:
        pool = profile.get("pool", {}) or {}
        diameter_cm = max(float(pool.get("diameter_cm", 120.0)), 1e-6)
        if "wall_zone_width_cm" in profile:
            fraction = float(profile.get("wall_zone_width_cm", 10.0)) / (diameter_cm / 2.0)
        else:
            fraction = float(profile.get("peripheral_ring_percent", 0.15))
        return max(0.0, min(fraction, 0.95))

    def shape_handles(self, shape: str) -> list[QPointF]:
        target = self.profile.get(shape, {}) or {}
        if shape == "pool":
            center = target.get("center_px")
            radius_x, radius_y = self.pool_radii(target)
            if not center or radius_x <= 0 or radius_y <= 0:
                return []
            cx, cy = map(float, center)
            return [
                QPointF(cx - radius_x, cy - radius_y), QPointF(cx, cy - radius_y),
                QPointF(cx + radius_x, cy - radius_y), QPointF(cx + radius_x, cy),
                QPointF(cx + radius_x, cy + radius_y), QPointF(cx, cy + radius_y),
                QPointF(cx - radius_x, cy + radius_y), QPointF(cx - radius_x, cy),
            ]
        if str(target.get("shape_type", "circle")) == "circle":
            center = target.get("center_px")
            radius = float(target.get("radius_px", 0))
            if not center or radius <= 0:
                return []
            cx, cy = map(float, center)
            return [
                QPointF(cx - radius, cy - radius), QPointF(cx, cy - radius),
                QPointF(cx + radius, cy - radius), QPointF(cx + radius, cy),
                QPointF(cx + radius, cy + radius), QPointF(cx, cy + radius),
                QPointF(cx - radius, cy + radius), QPointF(cx - radius, cy),
            ]
        return [QPointF(float(point[0]), float(point[1])) for point in target.get("points_px", [])]

    def hit_handle(self, point: QPointF, shape: str) -> int:
        radius = 12.0 / max(self.view_scale(), 0.01)
        for index, handle in enumerate(self.shape_handles(shape)):
            if math.hypot(point.x() - handle.x(), point.y() - handle.y()) <= radius:
                return index
        return -1

    def north_handle(self) -> QPointF | None:
        pool = self.profile.get("pool", {}) or {}
        center = pool.get("center_px")
        radius_x, radius_y = self.pool_radii(pool)
        if not center or radius_x <= 0 or radius_y <= 0:
            return None
        angle = math.radians(float(self.profile.get("north_angle_degrees", 0.0)))
        return QPointF(
            float(center[0]) + radius_x * 0.72 * math.sin(angle),
            float(center[1]) - radius_y * 0.72 * math.cos(angle),
        )

    def hit_north_handle(self, point: QPointF) -> bool:
        handle = self.north_handle()
        if handle is None:
            return False
        radius = 14.0 / max(self.view_scale(), 0.01)
        return math.hypot(point.x() - handle.x(), point.y() - handle.y()) <= radius

    def selectable_shapes(self) -> tuple[str, ...]:
        stage = int(getattr(self.window(), "water_stage", 1))
        if stage == 1:
            return ("pool",)
        if stage == 2:
            return ("platform",)
        return ()

    def platform_rotation_points(self) -> tuple[QPointF, QPointF] | None:
        platform = self.profile.get("platform", {}) or {}
        if str(platform.get("shape_type", "circle")) == "circle":
            return None
        points = platform.get("points_px") or []
        center = platform.get("center_px")
        if len(points) < 3 or not center:
            return None
        cx, cy = float(center[0]), float(center[1])
        radius = max(math.hypot(float(x) - cx, float(y) - cy) for x, y in points)
        angle = math.radians(float(platform.get("rotation_degrees", 0.0)))
        direction_x, direction_y = math.sin(angle), -math.cos(angle)
        anchor = QPointF(cx + direction_x * radius, cy + direction_y * radius)
        handle = QPointF(
            cx + direction_x * (radius + 30.0 / max(self.view_scale(), 0.01)),
            cy + direction_y * (radius + 30.0 / max(self.view_scale(), 0.01)),
        )
        return anchor, handle

    def hit_platform_rotation_handle(self, point: QPointF) -> bool:
        rotation_points = self.platform_rotation_points()
        if rotation_points is None:
            return False
        handle = rotation_points[1]
        radius = 15.0 / max(self.view_scale(), 0.01)
        return math.hypot(point.x() - handle.x(), point.y() - handle.y()) <= radius

    def begin_platform_rotation(self, point: QPointF) -> None:
        platform = self.profile.get("platform", {}) or {}
        center = platform.get("center_px")
        if not center:
            return
        self.selected_shape = "platform"
        self.drag_mode = "rotate_platform"
        self.drag_snapshot = json.loads(json.dumps(platform))
        self.rotation_start_angle = math.degrees(
            math.atan2(point.x() - float(center[0]), -(point.y() - float(center[1])))
        )
        self.setCursor(Qt.ClosedHandCursor)

    def rotate_platform_to_pointer(self, point: QPointF) -> None:
        snapshot = self.drag_snapshot or {}
        center = snapshot.get("center_px")
        points = snapshot.get("points_px") or []
        if not center or len(points) < 3:
            return
        cx, cy = float(center[0]), float(center[1])
        pointer_angle = math.degrees(math.atan2(point.x() - cx, -(point.y() - cy)))
        delta = (pointer_angle - self.rotation_start_angle + 180.0) % 360.0 - 180.0
        radians = math.radians(delta)
        cosine, sine = math.cos(radians), math.sin(radians)
        platform = self.profile.setdefault("platform", {})
        platform["points_px"] = [
            [
                max(0.0, min(float(self.pixmap.width()), cx + (float(x) - cx) * cosine - (float(y) - cy) * sine)),
                max(0.0, min(float(self.pixmap.height()), cy + (float(x) - cx) * sine + (float(y) - cy) * cosine)),
            ]
            for x, y in points
        ]
        platform["rotation_degrees"] = (
            float(snapshot.get("rotation_degrees", 0.0)) + delta
        ) % 360.0
        self.update_platform_compatibility()

    def rotate_north(self, point: QPointF) -> None:
        pool = self.profile.get("pool", {}) or {}
        center = pool.get("center_px")
        radius_x, radius_y = self.pool_radii(pool)
        if not center or radius_x <= 0 or radius_y <= 0:
            return
        normalized_x = (point.x() - float(center[0])) / radius_x
        normalized_y = (point.y() - float(center[1])) / radius_y
        angle = math.degrees(math.atan2(normalized_x, -normalized_y)) % 360.0
        self.profile["north_angle_degrees"] = angle
        window = self.window()
        if hasattr(window, "north_angle"):
            window.north_angle.blockSignals(True)
            window.north_angle.setValue(angle)
            window.north_angle.blockSignals(False)
        if hasattr(window, "refresh_review_summary"):
            window.refresh_review_summary()

    def shape_contains(self, point: QPointF, shape: str) -> bool:
        target = self.profile.get(shape, {}) or {}
        if shape == "pool":
            center = target.get("center_px")
            radius_x, radius_y = self.pool_radii(target)
            if not center or radius_x <= 0 or radius_y <= 0:
                return False
            normalized = (
                ((point.x() - float(center[0])) / radius_x) ** 2
                + ((point.y() - float(center[1])) / radius_y) ** 2
            )
            return normalized <= 1.0
        if str(target.get("shape_type", "circle")) == "circle":
            center = target.get("center_px")
            radius = float(target.get("radius_px", 0))
            return bool(
                center and math.hypot(point.x() - float(center[0]), point.y() - float(center[1])) <= radius
            )
        points = self.shape_handles(shape)
        return len(points) >= 3 and QPolygonF(points).containsPoint(point, Qt.OddEvenFill)

    def begin_move(self, point: QPointF, shape: str) -> None:
        self.selected_shape = shape
        self.drag_mode = "move"
        self.drag_start = point
        self.drag_snapshot = json.loads(json.dumps(self.profile.get(shape, {})))

    def move_selected(self, point: QPointF) -> None:
        if not self.drag_start or not self.selected_shape:
            return
        dx = point.x() - self.drag_start.x()
        dy = point.y() - self.drag_start.y()
        target = self.profile.setdefault(self.selected_shape, {})
        snapshot = self.drag_snapshot
        if snapshot.get("center_px"):
            target["center_px"] = [
                max(0, min(self.pixmap.width(), float(snapshot["center_px"][0]) + dx)),
                max(0, min(self.pixmap.height(), float(snapshot["center_px"][1]) + dy)),
            ]
        if snapshot.get("points_px"):
            target["points_px"] = [
                [
                    max(0, min(self.pixmap.width(), float(value[0]) + dx)),
                    max(0, min(self.pixmap.height(), float(value[1]) + dy)),
                ]
                for value in snapshot["points_px"]
            ]
            self.update_platform_compatibility()

    def resize_selected(self, point: QPointF) -> None:
        target = self.profile.setdefault(self.selected_shape, {})
        if self.selected_shape == "pool":
            snapshot = self.drag_snapshot or target
            center = snapshot.get("center_px")
            radius_x, radius_y = self.pool_radii(snapshot)
            handle_signs = (
                (-1, -1), (0, -1), (1, -1), (1, 0),
                (1, 1), (0, 1), (-1, 1), (-1, 0),
            )
            if (
                center and radius_x > 0 and radius_y > 0
                and 0 <= self.active_handle < len(handle_signs)
            ):
                cx, cy = float(center[0]), float(center[1])
                sx, sy = handle_signs[self.active_handle]
                is_ellipse = str(snapshot.get("shape_type", "circle")) == "ellipse"
                new_radius_x, new_radius_y = radius_x, radius_y
                new_center = [cx, cy]
                if sx == 0:
                    anchor_y = cy - sy * radius_y
                    new_radius_y = max(3.0, abs(point.y() - anchor_y) / 2.0)
                    new_center[1] = anchor_y + sy * new_radius_y
                elif sy == 0:
                    anchor_x = cx - sx * radius_x
                    new_radius_x = max(3.0, abs(point.x() - anchor_x) / 2.0)
                    new_center[0] = anchor_x + sx * new_radius_x
                else:
                    anchor_x = cx - sx * radius_x
                    anchor_y = cy - sy * radius_y
                    candidate_x = max(3.0, abs(point.x() - anchor_x) / 2.0)
                    candidate_y = max(3.0, abs(point.y() - anchor_y) / 2.0)
                    if is_ellipse:
                        new_radius_x, new_radius_y = candidate_x, candidate_y
                    else:
                        new_radius_x = new_radius_y = min(candidate_x, candidate_y)
                    new_center = [
                        anchor_x + sx * new_radius_x,
                        anchor_y + sy * new_radius_y,
                    ]
                if not is_ellipse:
                    common_radius = new_radius_y if sx == 0 else new_radius_x
                    new_radius_x = new_radius_y = common_radius
                target["center_px"] = [
                    max(0.0, min(float(self.pixmap.width()), new_center[0])),
                    max(0.0, min(float(self.pixmap.height()), new_center[1])),
                ]
                target["radius_x_px"] = new_radius_x
                target["radius_y_px"] = new_radius_y
                target["radius_px"] = (new_radius_x + new_radius_y) / 2.0
        elif str(target.get("shape_type", "circle")) == "circle":
            center = target.get("center_px")
            if center:
                target["radius_px"] = max(
                    3.0, math.hypot(point.x() - float(center[0]), point.y() - float(center[1]))
                )
        else:
            points = target.get("points_px") or []
            if 0 <= self.active_handle < len(points):
                points[self.active_handle] = [
                    max(0, min(self.pixmap.width(), point.x())),
                    max(0, min(self.pixmap.height(), point.y())),
                ]
                target["points_px"] = points
                self.update_platform_compatibility()

    def update_platform_compatibility(self) -> None:
        platform = self.profile.setdefault("platform", {})
        points = platform.get("points_px") or []
        if len(points) < 3:
            return
        cx = sum(float(point[0]) for point in points) / len(points)
        cy = sum(float(point[1]) for point in points) / len(points)
        platform["center_px"] = [cx, cy]
        platform["radius_px"] = max(
            3.0, max(math.hypot(float(point[0]) - cx, float(point[1]) - cy) for point in points)
        )

    def rotate_platform(self, delta_degrees: float) -> bool:
        """Rotate a polygonal platform around its centroid in image coordinates."""
        platform = self.profile.setdefault("platform", {})
        if str(platform.get("shape_type", "circle")) == "circle":
            return False
        points = platform.get("points_px") or []
        if len(points) < 3:
            return False
        cx = sum(float(point[0]) for point in points) / len(points)
        cy = sum(float(point[1]) for point in points) / len(points)
        radians = math.radians(float(delta_degrees))
        cosine, sine = math.cos(radians), math.sin(radians)
        rotated = []
        for x, y in points:
            dx, dy = float(x) - cx, float(y) - cy
            rotated.append([
                max(0.0, min(float(self.pixmap.width()), cx + dx * cosine - dy * sine)),
                max(0.0, min(float(self.pixmap.height()), cy + dx * sine + dy * cosine)),
            ])
        platform["points_px"] = rotated
        platform["rotation_degrees"] = (
            float(platform.get("rotation_degrees", 0.0)) + float(delta_degrees)
        ) % 360.0
        self.update_platform_compatibility()
        self.update()
        return True

    def mousePressEvent(self, event) -> None:
        if not self.interaction_enabled:
            return
        if event.button() != Qt.LeftButton:
            return
        self.setFocus()
        point = self.to_image(event.position())
        if self.tool == "select":
            allowed_shapes = self.selectable_shapes()
            if "pool" in allowed_shapes and self.hit_north_handle(point):
                self.selected_shape = "pool"
                self.drag_mode = "rotate_north"
                self.drag_start = point
                self.setCursor(Qt.CrossCursor)
                return
            if "platform" in allowed_shapes and self.hit_platform_rotation_handle(point):
                self.begin_platform_rotation(point)
                return
            ordered_shapes = tuple(
                shape for shape in (self.selected_shape, *allowed_shapes)
                if shape in allowed_shapes
            )
            for shape in dict.fromkeys(ordered_shapes):
                if not shape:
                    continue
                handle = self.hit_handle(point, shape)
                if handle >= 0:
                    self.selected_shape = shape
                    self.active_handle = handle
                    self.drag_mode = "resize"
                    self.drag_start = point
                    self.drag_snapshot = json.loads(json.dumps(self.profile.get(shape, {})))
                    self.update()
                    return
            if "platform" in allowed_shapes and self.shape_contains(point, "platform"):
                self.begin_move(point, "platform")
            elif "pool" in allowed_shapes and self.shape_contains(point, "pool"):
                self.begin_move(point, "pool")
            else:
                self.selected_shape = ""
            self.update()
            return
        if self.tool == "platform_polygon":
            self.polygon_draft.append([point.x(), point.y()])
            self.selected_shape = "platform"
            self.update()
            return
        self.drag_start = point
        self.drag_mode = "draw"
        if self.tool.startswith("platform"):
            self.selected_shape = "platform"
        else:
            self.selected_shape = "pool"
        self.draw_shape_name = self.selected_shape
        self.drag_snapshot = json.loads(json.dumps(self.profile.get(self.draw_shape_name, {})))

    def mouseMoveEvent(self, event) -> None:
        if not self.interaction_enabled:
            return
        point = self.to_image(event.position())
        if self.tool == "select" and not self.drag_mode:
            if self.selected_shape == "platform" and self.hit_platform_rotation_handle(point):
                self.setCursor(Qt.OpenHandCursor)
                return
            handle = self.hit_handle(point, self.selected_shape) if self.selected_shape else -1
            if handle >= 0:
                if handle in (1, 5):
                    self.setCursor(Qt.SizeVerCursor)
                elif handle in (3, 7):
                    self.setCursor(Qt.SizeHorCursor)
                elif handle in (0, 4):
                    self.setCursor(Qt.SizeFDiagCursor)
                else:
                    self.setCursor(Qt.SizeBDiagCursor)
            elif (
                self.shape_contains(point, "platform")
                or self.shape_contains(point, "pool")
            ):
                self.setCursor(Qt.SizeAllCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        if self.drag_mode == "move":
            self.move_selected(point)
        elif self.drag_mode == "resize":
            self.resize_selected(point)
        elif self.drag_mode == "rotate_north":
            self.rotate_north(point)
        elif self.drag_mode == "rotate_platform":
            self.rotate_platform_to_pointer(point)
        elif self.drag_mode == "draw" and self.drag_start is not None:
            if self.tool in {"pool_circle", "platform_circle"}:
                self.set_circle(self.drag_start, point, "pool" if self.tool == "pool_circle" else "platform")
            elif self.tool == "pool_ellipse":
                self.set_pool_ellipse(self.drag_start, point)
            elif self.tool == "platform_rectangle":
                self.set_rectangle(self.drag_start, point)
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if not self.interaction_enabled:
            return
        if event.button() != Qt.LeftButton:
            return
        point = self.to_image(event.position())
        if self.drag_mode == "rotate_north":
            self.rotate_north(point)
            self.return_to_select()
        elif self.drag_mode == "rotate_platform":
            self.rotate_platform_to_pointer(point)
            self.setCursor(Qt.ArrowCursor)
        if self.drag_mode == "draw" and self.drag_start is not None:
            minimum_drag = 8.0 / max(self.view_scale(), 0.01)
            width = abs(point.x() - self.drag_start.x())
            height = abs(point.y() - self.drag_start.y())
            if min(width, height) < minimum_drag:
                self.cancel_drawing()
            else:
                if self.tool in {"pool_circle", "platform_circle"}:
                    self.set_circle(
                        self.drag_start,
                        point,
                        "pool" if self.tool == "pool_circle" else "platform",
                    )
                elif self.tool == "pool_ellipse":
                    self.set_pool_ellipse(self.drag_start, point)
                elif self.tool == "platform_rectangle":
                    self.set_rectangle(self.drag_start, point)
                self.return_to_select()
        self.drag_start = None
        self.drag_mode = ""
        self.draw_shape_name = ""
        self.drag_snapshot = {}
        self.active_handle = -1
        self.rotation_start_angle = 0.0
        self.update()

    def cancel_drawing(self) -> None:
        if self.draw_shape_name:
            self.profile[self.draw_shape_name] = json.loads(json.dumps(self.drag_snapshot))
        self.drag_start = None
        self.drag_mode = ""
        self.draw_shape_name = ""
        self.drag_snapshot = {}
        self.active_handle = -1
        self.update()

    def return_to_select(self) -> None:
        window = self.window()
        if hasattr(window, "set_tool"):
            window.set_tool("select")
        else:
            self.tool = "select"
            self.setCursor(Qt.ArrowCursor)

    def set_circle(self, start: QPointF, end: QPointF, shape: str) -> None:
        """Create a circle from the diagonal of its enclosing square.

        This matches the familiar shape-drawing interaction used by slide and
        graphics applications: drag from one edge/corner of the desired circle
        to the opposite edge/corner.  The smaller dragged dimension is used so
        an imprecise diagonal drag never expands the ROI beyond the intended
        boundary.
        """
        delta_x = end.x() - start.x()
        delta_y = end.y() - start.y()
        diameter = max(6.0, min(abs(delta_x), abs(delta_y)))
        direction_x = 1.0 if delta_x >= 0 else -1.0
        direction_y = 1.0 if delta_y >= 0 else -1.0
        center = QPointF(
            start.x() + direction_x * diameter / 2.0,
            start.y() + direction_y * diameter / 2.0,
        )
        radius = diameter / 2.0
        target = self.profile.setdefault(shape, {})
        if shape == "platform":
            target["shape_type"] = "circle"
            target.pop("points_px", None)
        target["center_px"] = [round(center.x(), 3), round(center.y(), 3)]
        target["radius_px"] = round(radius, 3)
        if shape == "pool":
            target["shape_type"] = "circle"
            target["radius_x_px"] = round(radius, 3)
            target["radius_y_px"] = round(radius, 3)
        self.update()

    def set_pool_ellipse(self, start: QPointF, end: QPointF) -> None:
        left, right = sorted((start.x(), end.x()))
        top, bottom = sorted((start.y(), end.y()))
        radius_x = max(3.0, (right - left) / 2.0)
        radius_y = max(3.0, (bottom - top) / 2.0)
        pool = self.profile.setdefault("pool", {})
        pool["shape_type"] = "ellipse"
        pool["center_px"] = [round((left + right) / 2.0, 3), round((top + bottom) / 2.0, 3)]
        pool["radius_x_px"] = round(radius_x, 3)
        pool["radius_y_px"] = round(radius_y, 3)
        pool["radius_px"] = round((radius_x + radius_y) / 2.0, 3)
        self.update()

    def set_rectangle(self, start: QPointF, end: QPointF) -> None:
        left, right = sorted((start.x(), end.x()))
        top, bottom = sorted((start.y(), end.y()))
        platform = self.profile.setdefault("platform", {})
        platform["shape_type"] = "rectangle"
        platform["points_px"] = [[left, top], [right, top], [right, bottom], [left, bottom]]
        platform["rotation_degrees"] = 0.0
        self.update_platform_compatibility()

    def finish_polygon(self) -> bool:
        if len(self.polygon_draft) < 3:
            return False
        platform = self.profile.setdefault("platform", {})
        platform["shape_type"] = "polygon"
        platform["points_px"] = [point[:] for point in self.polygon_draft]
        platform["rotation_degrees"] = 0.0
        self.polygon_draft.clear()
        self.update_platform_compatibility()
        self.tool = "select"
        self.selected_shape = "platform"
        self.update()
        return True

    def delete_platform(self) -> None:
        target = str(self.profile.get("platform", {}).get("target_quadrant", "NE"))
        self.profile["platform"] = {"shape_type": "circle", "target_quadrant": target}
        self.polygon_draft.clear()
        self.selected_shape = ""
        self.update()

    def keyPressEvent(self, event) -> None:
        if not self.interaction_enabled:
            super().keyPressEvent(event)
            return
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and self.tool == "platform_polygon":
            self.finish_polygon()
        elif event.key() == Qt.Key_Backspace and self.tool == "platform_polygon" and self.polygon_draft:
            self.polygon_draft.pop()
            self.update()
        elif event.key() == Qt.Key_Escape:
            if self.drag_mode == "draw":
                self.cancel_drawing()
                self.return_to_select()
            elif self.tool == "platform_polygon":
                self.polygon_draft.clear()
                self.return_to_select()
                self.update()
        elif event.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self.selected_shape == "platform":
            self.delete_platform()
        else:
            super().keyPressEvent(event)

    def draw_shape(self, painter: QPainter, shape: str, color: str, fill_alpha: int) -> None:
        target = self.profile.get(shape, {}) or {}
        painter.setPen(QPen(QColor(color), 3))
        painter.setBrush(QColor(*QColor(color).getRgb()[:3], fill_alpha))
        if shape == "pool":
            center = target.get("center_px")
            radius_x_px, radius_y_px = self.pool_radii(target)
            if not center or radius_x_px <= 0 or radius_y_px <= 0:
                return
            center_widget = self.to_widget(center)
            scale = self.view_scale()
            painter.drawEllipse(center_widget, radius_x_px * scale, radius_y_px * scale)
        elif str(target.get("shape_type", "circle")) == "circle":
            center = target.get("center_px")
            radius_px = float(target.get("radius_px", 0))
            if not center or radius_px <= 0:
                return
            center_widget = self.to_widget(center)
            scaled, _left, _top = self.image_rect()
            radius = radius_px * scaled.width() / max(self.pixmap.width(), 1)
            painter.drawEllipse(center_widget, radius, radius)
        else:
            points = [self.to_widget(point) for point in target.get("points_px", [])]
            if len(points) >= 3:
                painter.drawPolygon(QPolygonF(points))

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#07101d"))
        scaled, left, top = self.image_rect()
        painter.drawPixmap(round(left), round(top), scaled)
        pool = self.profile.get("pool", {})
        radius_x_px, radius_y_px = self.pool_radii(pool)
        if pool.get("center_px") and radius_x_px > 0 and radius_y_px > 0:
            self.draw_shape(painter, "pool", "#67a8ff", 24)
            center = self.to_widget(pool["center_px"])
            radius_x = radius_x_px * scaled.width() / max(self.pixmap.width(), 1)
            radius_y = radius_y_px * scaled.height() / max(self.pixmap.height(), 1)
            north_degrees = float(self.profile.get("north_angle_degrees", 0.0)) % 360.0

            def compass_point(angle_degrees: float, scale_factor: float = 1.0) -> QPointF:
                angle = math.radians(angle_degrees)
                return QPointF(
                    center.x() + radius_x * scale_factor * math.sin(angle),
                    center.y() - radius_y * scale_factor * math.cos(angle),
                )

            painter.setPen(QPen(QColor("#91a4bd"), 1, Qt.DashLine))
            for boundary in (north_degrees, north_degrees + 90.0):
                painter.drawLine(
                    compass_point(boundary, -1.0),
                    compass_point(boundary, 1.0),
                )
            labels = (
                ("NE", north_degrees + 45.0),
                ("SE", north_degrees + 135.0),
                ("SW", north_degrees + 225.0),
                ("NW", north_degrees + 315.0),
            )
            painter.setPen(QColor("#eaf2ff"))
            for text, angle in labels:
                point = compass_point(angle, 0.62)
                painter.drawText(point + QPointF(-10, 5), text)
            ring = self.wall_ring_fraction(self.profile)
            inner_x = radius_x * (1.0 - ring)
            inner_y = radius_y * (1.0 - ring)
            painter.setPen(QPen(QColor("#f2c14e"), 1, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(center, inner_x, inner_y)
        platform = self.profile.get("platform", {})
        if platform.get("center_px") and (
            platform.get("radius_px") or len(platform.get("points_px") or []) >= 3
        ):
            self.draw_shape(painter, "platform", "#ff6685", 60)
            center = self.to_widget(platform["center_px"])
            painter.setPen(QColor("#ffffff"))
            painter.drawText(center + QPointF(8, -8), "Platform")
        if self.polygon_draft:
            draft = [self.to_widget(point) for point in self.polygon_draft]
            painter.setPen(QPen(QColor("#ff6685"), 2, Qt.DashLine))
            for index in range(1, len(draft)):
                painter.drawLine(draft[index - 1], draft[index])
            for point in draft:
                painter.setBrush(QColor("#ff6685"))
                painter.drawEllipse(point, 4, 4)
        if self.tool == "select" and self.selected_shape:
            painter.setPen(QPen(QColor("#07101d"), 2))
            painter.setBrush(QColor("#f4f7fb"))
            for point in self.shape_handles(self.selected_shape):
                painter.drawEllipse(self.to_widget([point.x(), point.y()]), 6, 6)
            if self.selected_shape == "platform":
                rotation_points = self.platform_rotation_points()
                if rotation_points is not None:
                    anchor, handle = (self.to_widget([point.x(), point.y()]) for point in rotation_points)
                    painter.setPen(QPen(QColor("#dfe9f7"), 1))
                    painter.drawLine(anchor, handle)
                    painter.setPen(QPen(QColor("#243044"), 2))
                    painter.setBrush(QColor("#67a8ff"))
                    painter.drawEllipse(handle, 7, 7)
            north_handle = self.north_handle()
            if self.selected_shape == "pool" and north_handle is not None:
                handle = self.to_widget([north_handle.x(), north_handle.y()])
                painter.setPen(QPen(QColor("#07101d"), 2))
                painter.setBrush(QColor("#67a8ff"))
                painter.drawEllipse(handle, 8, 8)
                painter.setPen(QColor("#ffffff"))
                painter.drawText(handle + QPointF(11, 5), "N")


def watermaze_profile_matches_video(payload: dict, video: Path) -> bool:
    """Do not reuse another video's pool/platform geometry by path collision."""
    source_text = str(payload.get("source_video", "")).strip()
    if not source_text:
        return True
    source = Path(source_text)
    try:
        return str(source.resolve()).casefold() == str(video.resolve()).casefold()
    except OSError:
        return source.name.casefold() == video.name.casefold()


class WaterMazeEditorDialog(QDialog):
    def __init__(self, video: Path, profile_path: Path, parent=None) -> None:
        super().__init__(parent)
        enable_full_window_controls(self)
        self.video = video
        self.profile_path = profile_path
        pixmap, width, height, self.fps, self.frame_count = frame_at(video, 0.0)
        default_radius = min(width, height) * 0.44
        self.profile = {
            "profile_type": "hagisilta_watermaze_profile",
            "coordinate_space": "arena_crop_px",
            "source_video": str(video),
            "frame_size": [width, height],
            "pool": {
                "shape_type": "circle",
                "center_px": [width / 2, height / 2],
                "radius_px": default_radius,
                "radius_x_px": default_radius,
                "radius_y_px": default_radius,
                "diameter_cm": 120.0,
            },
            "platform": {
                "center_px": [width / 2 + default_radius * 0.45, height / 2 - default_radius * 0.45],
                "radius_px": max(5, default_radius * 0.075),
                "target_quadrant": "NE",
            },
            "north_angle_degrees": 0.0,
            "peripheral_ring_percent": 0.15,
            "wall_zone_width_cm": 10.0,
            "reference_seconds": 0.0,
        }
        loaded_profile = None
        if profile_path.is_file():
            try:
                loaded = json.loads(profile_path.read_text(encoding="utf-8"))
                if (
                    loaded.get("profile_type") == "hagisilta_watermaze_profile"
                    and watermaze_profile_matches_video(loaded, video)
                ):
                    loaded_profile = loaded
                    self.profile.update(loaded)
            except Exception:
                pass
        self.profile.setdefault("platform", {}).setdefault("shape_type", "circle")
        if loaded_profile is not None and "wall_zone_width_cm" not in loaded_profile:
            diameter_cm = float(self.profile.get("pool", {}).get("diameter_cm", 120.0))
            old_fraction = float(self.profile.get("peripheral_ring_percent", 0.15))
            self.profile["wall_zone_width_cm"] = diameter_cm * old_fraction / 2.0
        else:
            self.profile.setdefault("wall_zone_width_cm", 10.0)
        self.profile["peripheral_ring_percent"] = WaterMazeCanvas.wall_ring_fraction(self.profile)
        self.duration_seconds = max(0.0, (self.frame_count - 1) / max(self.fps, 1e-6))
        self.current_seconds = min(
            max(float(self.profile.get("reference_seconds", 0.0)), 0.0),
            self.duration_seconds,
        )
        self.water_stage = 1
        self._pending_seek_ms = round(self.current_seconds * 1000)
        self._seek_timer = QTimer(self)
        self._seek_timer.setSingleShot(True)
        self._seek_timer.setInterval(40)
        self._seek_timer.timeout.connect(self.apply_queued_timeline_seek)
        pixmap, width, height, self.fps, self.frame_count = frame_at(video, self.current_seconds)
        self.setWindowTitle(tr('HAGISILTA · Water maze · 수조·플랫폼 편집'))
        apply_hagisilta_window_identity(self)
        self.resize(1280, 800)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header, header_layout = create_hagisilta_header(
            tr('Water-maze 수조·플랫폼 편집'), "HAGISILTA · SPATIAL WORKSPACE",
        )
        header_layout.addWidget(create_hagisilta_help_button(
            self, tr('Water-maze 편집 도움말'),
            tr('1 수조 외곽을 지정합니다.\n2 플랫폼 위치와 모양을 지정합니다.\n3 목표 사분면과 벽면 선호 영역을 확인합니다.\n\nV 선택·이동 · C 수조 원 · E 수조 타원\nO 플랫폼 원 · R 플랫폼 사각형 · P 플랫폼 다각형\n사각형·다각형은 영상 위 파란 회전 핸들을 드래그\nDelete 플랫폼 삭제 · Ctrl+S 저장 · Alt+1~3 단계 이동'),
        ))
        self.status = HeaderContext(header, tr('편집 모드 · 선택·이동'))
        layout.addWidget(header)
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        sidebar = QFrame(objectName="sidebar")
        sidebar.setFixedWidth(184)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(18, 24, 18, 24)
        sidebar_layout.setSpacing(8)
        sidebar_layout.addWidget(QLabel("WORKFLOW", objectName="workflowTitle"))
        self.water_stage_buttons = []
        for stage, text in (
            (1, tr('01  수조 ROI')),
            (2, tr('02  플랫폼 ROI')),
            (3, tr('03  검토 및 저장')),
        ):
            button = QPushButton(text, objectName="workflowButton")
            button.setCheckable(True)
            button.setToolTip(tr('{p0} 단계로 이동 · Alt+{p1}', p0=text[4:], p1=stage))
            button.clicked.connect(lambda _checked=False, value=stage: self.set_water_stage(value))
            sidebar_layout.addWidget(button)
            self.water_stage_buttons.append((stage, button))
        sidebar_layout.addStretch()
        body.addWidget(sidebar)

        workspace = QFrame(objectName="workspace")
        workspace_layout = QHBoxLayout(workspace)
        workspace_layout.setContentsMargins(18, 18, 18, 18)
        workspace_layout.setSpacing(16)
        body.addWidget(workspace, 1)

        self.canvas = WaterMazeCanvas(pixmap, self.profile)
        canvas_frame = QFrame(objectName="canvasFrame")
        canvas_layout = QVBoxLayout(canvas_frame)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.addWidget(QLabel(self.video.name, objectName="canvasTitle"))
        canvas_layout.addWidget(self.canvas, 1)
        workspace_layout.addWidget(canvas_frame, 1)
        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setFrameShape(QFrame.NoFrame)
        side_scroll.setFixedWidth(380)
        side_scroll.setObjectName("panelScroll")
        side = QFrame(objectName="panel")
        side.setMinimumWidth(380)
        form = QVBoxLayout(side)
        form.setContentsMargins(18, 18, 18, 18)
        form.setSpacing(10)
        self.tool_buttons: dict[str, QPushButton] = {}
        self.water_stage_widgets: dict[int, list[QWidget]] = {0: [], 1: [], 2: [], 3: []}

        time_panel = QFrame(objectName="stagePanel")
        time_form = QVBoxLayout(time_panel)
        time_form.setContentsMargins(0, 0, 0, 0)
        time_form.setSpacing(10)
        self.time_section = QLabel(tr('기준 프레임'), objectName="toolGroupTitle")
        time_form.addWidget(self.time_section)
        self.time_label = QLabel(objectName="timeReadout")
        time_form.addWidget(self.time_label)
        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setRange(0, max(1, round(self.duration_seconds * 1000)))
        self.timeline.setValue(round(self.current_seconds * 1000))
        self.timeline.setToolTip(
            tr('ROI를 확인하기 좋은 영상 위치를 선택합니다. 프레임을 바꿔도 수조·플랫폼 좌표는 그대로 유지됩니다.')
        )
        self.timeline.valueChanged.connect(self.timeline_value_changed)
        self.timeline.sliderMoved.connect(self.queue_timeline_seek)
        self.timeline.sliderReleased.connect(self.seek_from_timeline)
        time_form.addWidget(self.timeline)
        seek_row = QHBoxLayout()
        for text, seconds, frames in (
            (tr('-1초'), -1.0, 0),
            ("‹ 1F", 0.0, -1),
            ("1F ›", 0.0, 1),
            (tr('+1초'), 1.0, 0),
        ):
            button = QPushButton(text, objectName="secondary")
            button.clicked.connect(
                lambda _checked=False, sec=seconds, frame_delta=frames: self.seek_delta(sec, frame_delta)
            )
            seek_row.addWidget(button)
        time_form.addLayout(seek_row)
        form.addWidget(time_panel)

        select_button = QPushButton(tr('편집 (V)'), objectName="shapeToolButton")
        select_button.setCheckable(True)
        select_button.setToolTip(tr('도형 내부를 드래그해 이동하고 흰 조절점을 드래그해 크기와 꼭짓점을 조절합니다.'))
        select_button.clicked.connect(lambda: self.set_tool("select"))
        self.tool_buttons["select"] = select_button
        self.select_tool_button = select_button

        pool_panel = QFrame(objectName="stagePanel")
        pool_form = QVBoxLayout(pool_panel)
        pool_form.setContentsMargins(0, 0, 0, 0)
        pool_form.setSpacing(10)
        self.pool_section = QLabel(tr('수조 ROI'), objectName="toolGroupTitle")
        pool_form.addWidget(self.pool_section)
        pool_shape_row = QHBoxLayout()
        pool_shape_row.addWidget(select_button)
        self.pool_shape_row = pool_shape_row
        pool_button = QPushButton(tr('원형 (C)'), objectName="shapeToolButton")
        pool_button.setCheckable(True)
        pool_button.setToolTip(
            tr('NORT 원형 오브젝트와 같은 방식입니다.\n수조를 감싸도록 드래그해 원을 만든 뒤, 원 안쪽을 끌어 이동하세요.\n선택된 원 주위의 흰 조절점 8개를 끌면 반대편은 고정된 채 크기와 중심이 함께 조절됩니다.')
        )
        pool_button.clicked.connect(lambda: self.set_tool("pool_circle"))
        self.tool_buttons["pool_circle"] = pool_button
        pool_shape_row.addWidget(pool_button)
        ellipse_button = QPushButton(tr('타원형 (E)'), objectName="shapeToolButton")
        ellipse_button.setCheckable(True)
        ellipse_button.setToolTip(
            tr('카메라 원근 때문에 원형 수조가 화면에서 타원처럼 보일 때 사용합니다.\n수조의 보이는 가로·세로 경계를 감싸도록 드래그한 뒤, NORT 도형처럼 8개 흰 조절점으로 위치와 가로·세로 크기를 각각 맞추세요.')
        )
        ellipse_button.clicked.connect(lambda: self.set_tool("pool_ellipse"))
        self.tool_buttons["pool_ellipse"] = ellipse_button
        pool_shape_row.addWidget(ellipse_button)
        pool_form.addLayout(pool_shape_row)
        pool_form.addWidget(QLabel(tr('실제 수조 지름 (cm)'), objectName="fieldLabel"))
        self.diameter = QDoubleSpinBox()
        self.diameter.setRange(20, 500); self.diameter.setDecimals(1)
        self.diameter.setValue(float(self.profile["pool"].get("diameter_cm", 120)))
        self.diameter.valueChanged.connect(self.update_diameter)
        pool_form.addWidget(self.diameter)
        pool_form.addWidget(QLabel(tr('방위 기준 · 북쪽 방향'), objectName="fieldLabel"))
        north_row = QGridLayout()
        self.north_angle = QDoubleSpinBox()
        self.north_angle.setRange(0.0, 359.9)
        self.north_angle.setDecimals(1)
        self.north_angle.setSuffix("°")
        self.north_angle.setValue(float(self.profile.get("north_angle_degrees", 0.0)) % 360.0)
        self.north_angle.setToolTip(
            tr('영상 화면 위쪽을 0°로 하고 시계방향으로 측정합니다.\n예: 위=0°, 오른쪽=90°, 아래=180°, 왼쪽=270°.\n설정한 북쪽을 기준으로 NE·SE·SW·NW 경계와 결과가 함께 회전합니다.')
        )
        self.north_angle.valueChanged.connect(self.set_north_angle)
        self.north_angle.setMinimumWidth(150)
        north_row.addWidget(self.north_angle, 0, 0, 1, 4)
        for column, (text, angle) in enumerate(
            ((tr('↑ 위'), 0.0), (tr('→ 오른쪽'), 90.0), (tr('↓ 아래'), 180.0), (tr('← 왼쪽'), 270.0))
        ):
            button = QPushButton(text, objectName="secondary")
            button.clicked.connect(lambda _checked=False, value=angle: self.north_angle.setValue(value))
            north_row.addWidget(button, 1, column)
        pool_form.addLayout(north_row)
        form.addWidget(pool_panel)
        self.water_stage_widgets[1].append(pool_panel)

        platform_panel = QFrame(objectName="stagePanel")
        platform_form = QVBoxLayout(platform_panel)
        platform_form.setContentsMargins(0, 0, 0, 0)
        platform_form.setSpacing(10)
        self.platform_section = QLabel(tr('플랫폼 ROI'), objectName="toolGroupTitle")
        platform_form.addWidget(self.platform_section)
        platform_row = QHBoxLayout()
        self.platform_shape_row = platform_row
        for text, tool in (
            (tr('원형 (O)'), "platform_circle"),
            (tr('사각 (R)'), "platform_rectangle"),
            (tr('다각 (P)'), "platform_polygon"),
        ):
            button = QPushButton(text, objectName="shapeToolButton")
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, value=tool: self.set_tool(value))
            self.tool_buttons[tool] = button
            platform_row.addWidget(button)
        platform_form.addLayout(platform_row)
        polygon_actions = QHBoxLayout()
        finish_polygon = QPushButton(tr('다각형 완성 (Enter)'), objectName="secondary")
        finish_polygon.clicked.connect(self.finish_polygon)
        delete_platform = QPushButton(tr('플랫폼 삭제'), objectName="danger")
        delete_platform.clicked.connect(self.canvas.delete_platform)
        polygon_actions.addWidget(finish_polygon)
        polygon_actions.addWidget(delete_platform)
        platform_form.addLayout(polygon_actions)
        platform_form.addWidget(QLabel(tr('플랫폼 목표 사분면'), objectName="fieldLabel"))
        self.target = QComboBox(); self.target.addItems(["NE", "SE", "SW", "NW"])
        self.target.setCurrentText(str(self.profile["platform"].get("target_quadrant", "NE")))
        self.target.currentTextChanged.connect(lambda text: self.profile["platform"].update({"target_quadrant": text}))
        platform_form.addWidget(self.target)
        form.addWidget(platform_panel)
        self.water_stage_widgets[2].append(platform_panel)

        review_panel = QFrame(objectName="stagePanel")
        review_form = QVBoxLayout(review_panel)
        review_form.setContentsMargins(0, 0, 0, 0)
        review_form.setSpacing(10)
        review_form.addWidget(QLabel(tr('검토 및 저장'), objectName="toolGroupTitle"))
        self.review_summary = QLabel(objectName="reviewSummary")
        self.review_summary.setWordWrap(True)
        review_form.addWidget(self.review_summary)
        review_form.addWidget(QLabel(tr('벽면 선호 영역 · Thigmotaxis'), objectName="toolGroupTitle"))
        review_form.addWidget(QLabel(tr('수조 벽에서 안쪽으로의 실제 폭'), objectName="fieldLabel"))
        self.wall_zone_cm = QDoubleSpinBox()
        self.wall_zone_cm.setRange(
            0.5,
            max(0.5, float(self.profile["pool"].get("diameter_cm", 120.0)) * 0.475),
        )
        self.wall_zone_cm.setDecimals(1)
        self.wall_zone_cm.setSingleStep(0.5)
        self.wall_zone_cm.setSuffix(" cm")
        self.wall_zone_cm.setValue(float(self.profile.get("wall_zone_width_cm", 10.0)))
        self.wall_zone_cm.setToolTip(
            tr('노란 점선과 수조 외곽선 사이를 thigmotaxis 영역으로 판정합니다.\n입력값은 면적 비율이 아니라 실제 수조 벽에서 안쪽으로 잰 거리입니다.\n대표 좌표가 이 띠 안에 있는 프레임의 시간과 비율을 결과에 기록합니다.')
        )
        self.wall_zone_cm.valueChanged.connect(self.update_wall_zone)
        review_form.addWidget(self.wall_zone_cm)
        self.wall_zone_conversion = QLabel(objectName="help")
        self.wall_zone_conversion.setWordWrap(True)
        review_form.addWidget(self.wall_zone_conversion)
        self.refresh_wall_zone_conversion()
        form.addWidget(review_panel)
        self.water_stage_widgets[3].append(review_panel)
        form.addStretch()
        side_scroll.setWidget(side)
        self.side_scroll = side_scroll
        workspace_layout.addWidget(side_scroll)
        layout.addLayout(body, 1)

        footer = QFrame(objectName="detachedFooter")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(18, 9, 18, 9)
        self.footer_context = QLabel(tr('수조 ROI · 1/3'))
        footer_layout.addWidget(self.footer_context)
        footer_layout.addStretch()
        self.footer_back = QPushButton(tr('← 이전'), objectName="secondaryButton")
        self.footer_back.clicked.connect(lambda: self.set_water_stage(self.water_stage - 1))
        self.footer_next = QPushButton(tr('다음 →'), objectName="primaryButton")
        self.footer_next.clicked.connect(self.next_water_stage)
        footer_layout.addWidget(self.footer_back)
        footer_layout.addWidget(self.footer_next)
        layout.addWidget(footer)

        self.apply_editor_style()
        self.update_time_label()
        self.set_tool("select")
        self.set_water_stage(1)

    def apply_editor_style(self) -> None:
        self.setStyleSheet("""
            QDialog, QWidget {
                background:#0d1420; color:#dfe8f5;
                font-family:'Segoe UI','Malgun Gothic'; font-size:13px;
            }
            #header { background:#111b2a; border:none; border-bottom:1px solid #223149; }
            #logo { color:#8fb1ff; font-size:18px; font-weight:800; letter-spacing:2px; }
            #subtitle { color:#718198; margin-left:8px; }
            #modeBadge {
                color:#9dc4ff; background:#172b46; border:1px solid #31547c;
                border-radius:12px; padding:6px 12px;
            }
            #sidebar { background:#101925; border:none; border-right:1px solid #223149; }
            #workspace { background:#0d1420; border:none; }
            #workflowTitle {
                color:#9ba1aa; background:transparent; font-size:10px;
                font-weight:700; letter-spacing:1px; margin-bottom:7px;
            }
            #footer { background:#101925; border:none; border-top:1px solid #223149; }
            #hint { color:#9ba1aa; line-height:1.5; }
            #canvasFrame { background:#0d1420; }
            #canvasTitle { color:#f0f1f2; font-size:14px; font-weight:650; }
            #panel { background:#101925; border:none; border-left:1px solid #223149; }
            QScrollArea#panelScroll { background:transparent; border:none; }
            QScrollArea#panelScroll > QWidget > QWidget { background:#101925; }
            #section {
                background:#0c1420; color:#f1f6fc; font-size:15px;
                font-weight:800; padding:8px;
            }
            #toolGroupTitle {
                background:transparent; color:#f1f3f6; border:none;
                font-size:14px; font-weight:800; padding:4px 1px;
            }
            #help {
                color:#aebfd3; background:#0d1622; border:1px solid #27384d;
                border-radius:6px; padding:10px;
            }
            #fieldLabel { color:#b7c7da; font-weight:700; }
            #timeReadout {
                color:#cbd9f4; background:#0c1521; border:1px solid #26394f;
                border-radius:5px; padding:7px 9px; font-family:Consolas;
            }
            QPushButton {
                background:#1a2a3d; color:#dfe8f5; border:1px solid #344b66;
                border-radius:3px; padding:0 13px; min-height:36px; font-weight:600;
            }
            QPushButton:hover { border-color:#7194e8; background:#20334a; }
            QPushButton#primary {
                background:#4d67d7; color:white; border-color:#607aed; font-weight:750;
            }
            QPushButton#primaryButton { background:#4c63cf; color:#ffffff; border:1px solid #6079d5; }
            QPushButton#primaryButton:hover { background:#5872df; border-color:#8297e0; }
            QPushButton#secondary { background:#17263a; color:#c7d5e6; }
            QPushButton#secondaryButton { background:#18263a; color:#dfe8f5; border:1px solid #304765; }
            QPushButton#secondaryButton:hover { background:#21334c; border-color:#45658d; }
            QPushButton#danger {
                background:#3a2028; color:#ffadb9; border-color:#713541;
            }
            QPushButton#toolButton:checked {
                background:#3f6dcc; color:#ffffff; border:1px solid #6f91df; font-weight:800;
            }
            QPushButton#shapeToolButton {
                padding:0 3px; min-height:36px; font-size:11px;
            }
            QPushButton#shapeToolButton:checked {
                background:#3f6dcc; color:#ffffff; border:1px solid #6f91df; font-weight:800;
            }
            QPushButton#workflowButton {
                min-height:44px; text-align:left; padding-left:12px; color:#c2c5ca;
                background:transparent; border:1px solid transparent; border-radius:3px;
            }
            QPushButton#workflowButton:hover { background:#172437; color:#ffffff; }
            QPushButton#workflowButton:checked {
                background:#334f9f; color:#ffffff;
                border:1px solid #5f7fd1; font-weight:650;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background:#101b2a; color:#eaf2fb; border:1px solid #344b66;
                border-radius:5px; padding:8px;
            }
            QComboBox QAbstractItemView {
                background:#101b2a; color:#eaf2fb; selection-background-color:#4d67d7;
            }
            QSlider::groove:horizontal {
                height:6px; background:#26364b; border-radius:3px;
            }
            QSlider::handle:horizontal {
                width:16px; margin:-5px 0; background:#6f91df; border-radius:8px;
            }
            QScrollBar:vertical { background:#0d1420; width:13px; }
            QScrollBar::handle:vertical {
                background:#40536b; min-height:28px; border-radius:5px; margin:2px;
            }
            QToolTip {
                color:#eaf2fb; background:#17263a; border:1px solid #49617e; padding:6px;
            }
        """)
        self.setStyleSheet(self.styleSheet() + """
            QDialog, QWidget { background:#23242b; color:#f0f1f2; }
            #sidebar, #panel, #footer { background:#27282d; border-color:#41434a; }
            #workspace, #canvasFrame { background:#202125; }
            #section { background:#303136; color:#f0f1f2; border:1px solid #41434a; border-radius:2px; }
            #help, #timeReadout { background:#232429; color:#c2c5ca; border:1px solid #3f4147; border-radius:2px; }
            QPushButton { background:#35363b; color:#eceef1; border:1px solid #55585f; border-radius:2px; }
            QPushButton:hover { background:#42444a; border-color:#70747d; }
            QPushButton#primary, QPushButton#primaryButton { background:#3f6dcc; border-color:#5c82d8; color:#ffffff; }
            QPushButton#secondaryButton, QPushButton#toolButton { background:#35363b; border-color:#55585f; }
            QPushButton#toolButton:checked { background:#3f6dcc; border-color:#6f91df; color:#ffffff; }
            QPushButton#shapeToolButton { background:#35363b; border-color:#55585f; padding:0 3px; font-size:11px; }
            QPushButton#shapeToolButton:checked { background:#3f6dcc; border-color:#6f91df; color:#ffffff; }
            QPushButton#workflowButton:checked { background:#334f9f; border-color:#5f7fd1; color:#ffffff; }
            QComboBox, QSpinBox, QDoubleSpinBox { background:#202125; color:#f0f1f2; border:1px solid #50535a; border-radius:2px; }
            #footer { border-top:1px solid #41434a; }
        """ + hagisilta_control_stylesheet())

    def show_shortcuts(self) -> None:
        QMessageBox.information(
            self,
            tr('Water maze ROI 단축키'),
            tr('Ctrl+S  설정 저장\nV  선택·이동\nC / E  수조 원형 / 타원형\nO / R / P  플랫폼 원형 / 사각형 / 다각형\n플랫폼 회전  영상 위 파란 핸들 드래그\nEnter  플랫폼 다각형 완성\nBackspace  마지막 점 취소\nDelete  선택 플랫폼 삭제\nEsc  다각형 입력 취소\nAlt+1·2·3  작업 단계 이동'),
        )

    @staticmethod
    def format_time(seconds: float) -> str:
        minutes = int(max(seconds, 0.0) // 60)
        remainder = max(seconds, 0.0) - minutes * 60
        return f"{minutes:02d}:{remainder:06.3f}"

    def update_time_label(self, preview_seconds: float | None = None) -> None:
        seconds = self.current_seconds if preview_seconds is None else preview_seconds
        frame_index = min(max(round(seconds * self.fps), 0), self.frame_count - 1)
        self.time_label.setText(
            f"{self.format_time(seconds)} / {self.format_time(self.duration_seconds)}"
            f"  ·  {frame_index + 1}/{self.frame_count}F"
        )

    def timeline_value_changed(self, value: int) -> None:
        self.update_time_label(value / 1000.0)

    def queue_timeline_seek(self, value: int) -> None:
        self._pending_seek_ms = int(value)
        self.update_time_label(value / 1000.0)
        self._seek_timer.start(40)

    def apply_queued_timeline_seek(self) -> None:
        self.seek_to(self._pending_seek_ms / 1000.0)

    def seek_from_timeline(self) -> None:
        self._seek_timer.stop()
        self.seek_to(self.timeline.value() / 1000.0)

    def seek_delta(self, seconds: float = 0.0, frames: int = 0) -> None:
        target = self.current_seconds + seconds + frames / max(self.fps, 1e-6)
        self.seek_to(target)

    def seek_to(self, seconds: float) -> None:
        target = min(max(float(seconds), 0.0), self.duration_seconds)
        try:
            pixmap, _width, _height, self.fps, self.frame_count = frame_at(self.video, target)
        except Exception as error:
            QMessageBox.warning(self, tr('프레임 이동 실패'), str(error))
            return
        self.current_seconds = target
        self.profile["reference_seconds"] = round(target, 6)
        self.canvas.pixmap = pixmap
        self.canvas.update()
        self.timeline.blockSignals(True)
        self.timeline.setValue(round(target * 1000))
        self.timeline.blockSignals(False)
        self.update_time_label()

    def set_tool(self, tool: str) -> None:
        self.canvas.tool = tool
        if tool == "select":
            self.canvas.setCursor(Qt.ArrowCursor)
            self.status.setText(tr('편집 모드 · 선택·이동'))
        else:
            self.canvas.setCursor(Qt.CrossCursor)
            labels = {
                "pool_circle": tr('편집 모드 · 수조 원형'),
                "pool_ellipse": tr('편집 모드 · 수조 타원형'),
                "platform_circle": tr('편집 모드 · 플랫폼 원형'),
                "platform_rectangle": tr('편집 모드 · 플랫폼 사각형'),
                "platform_polygon": tr('편집 모드 · 플랫폼 다각형'),
            }
            self.status.setText(labels.get(tool, tr('편집 모드')))
        if tool in {"pool_circle", "pool_ellipse"}:
            self.canvas.selected_shape = "pool"
        elif tool.startswith("platform"):
            self.canvas.selected_shape = "platform"
            if tool != "platform_polygon":
                self.canvas.polygon_draft.clear()
        for name, button in self.tool_buttons.items():
            button.setChecked(name == tool)
        if tool in {"pool_circle", "pool_ellipse"}:
            self._sync_water_stage(1)
        elif tool.startswith("platform"):
            self._sync_water_stage(2)
        self.canvas.update()

    def _sync_water_stage(self, stage: int) -> None:
        self.water_stage = max(1, min(int(stage), 3))
        self.canvas.interaction_enabled = self.water_stage in (1, 2)
        labels = {1: tr('수조 ROI'), 2: tr('플랫폼 ROI'), 3: tr('검토 및 저장')}
        stage_widgets = getattr(self, "water_stage_widgets", {})
        all_widgets = {
            widget
            for widgets in stage_widgets.values()
            for widget in widgets
        }
        visible_widgets = set(stage_widgets.get(self.water_stage, []))
        for widget in all_widgets:
            widget.setVisible(widget in visible_widgets)
        for button_stage, button in getattr(self, "water_stage_buttons", []):
            button.setChecked(button_stage == self.water_stage)
        if hasattr(self, "select_tool_button"):
            target_layout = self.pool_shape_row if self.water_stage == 1 else self.platform_shape_row
            target_layout.insertWidget(0, self.select_tool_button)
        if hasattr(self, "footer_context"):
            self.footer_context.setText(f"{labels[self.water_stage]} · {self.water_stage}/3")
            self.footer_back.setEnabled(self.water_stage > 1)
            self.footer_next.setText(tr('저장하고 닫기') if self.water_stage == 3 else tr('다음 →'))
        if self.water_stage == 3:
            self.refresh_review_summary()

    def refresh_review_summary(self) -> None:
        if not hasattr(self, "review_summary"):
            return
        platform = self.profile.get("platform", {}) or {}
        shape_labels = {
            "circle": tr('원형'),
            "rectangle": tr('사각형'),
            "polygon": tr('자유 다각형'),
        }
        shape = shape_labels.get(str(platform.get("shape_type", "circle")), tr('원형'))
        pool_shape = tr('타원형 · 원근 보정') if str(
            self.profile.get("pool", {}).get("shape_type", "circle")
        ) == "ellipse" else tr('원형')
        north_angle = float(self.profile.get("north_angle_degrees", 0.0)) % 360.0
        wall_cm = float(self.profile.get("wall_zone_width_cm", 10.0))
        wall_percent = WaterMazeCanvas.wall_ring_fraction(self.profile) * 100.0
        self.review_summary.setText(
            tr('기준 프레임  {p0}\n수조  {p1} · 실제 지름 {p2:.1f} cm\n북쪽  화면 위에서 시계방향 {p3:.1f}°\n플랫폼  {p4} · {p5} 사분면\n벽면 선호 영역  벽에서 {p6:.1f} cm · 반지름의 {p7:.1f}%', p0=self.format_time(self.current_seconds), p1=pool_shape, p2=float(self.profile.get('pool', {}).get('diameter_cm', 120.0)), p3=north_angle, p4=shape, p5=platform.get('target_quadrant', 'NE'), p6=wall_cm, p7=wall_percent)
        )

    def set_water_stage(self, stage: int) -> None:
        stage = max(1, min(int(stage), 3))
        self._sync_water_stage(stage)
        if stage == 1:
            pool_shape = str(self.profile.get("pool", {}).get("shape_type", "circle"))
            self.set_tool("pool_ellipse" if pool_shape == "ellipse" else "pool_circle")
            self.side_scroll.ensureWidgetVisible(self.pool_section)
        elif stage == 2:
            platform_shape = str(self.profile.get("platform", {}).get("shape_type", "circle"))
            tool = {
                "rectangle": "platform_rectangle",
                "polygon": "platform_polygon",
            }.get(platform_shape, "platform_circle")
            self.set_tool(tool)
            self.side_scroll.ensureWidgetVisible(self.platform_section)
        else:
            self.set_tool("select")
            self.canvas.selected_shape = ""
            self.canvas.update()
            self.side_scroll.ensureWidgetVisible(self.review_summary)
            self.status.setText(tr('검토 모드'))
        self._sync_water_stage(stage)

    def next_water_stage(self) -> None:
        if self.water_stage >= 3:
            self.save_profile()
        else:
            self.set_water_stage(self.water_stage + 1)

    def finish_polygon(self) -> None:
        if not self.canvas.finish_polygon():
            QMessageBox.information(self, tr('다각형 완성'), tr('플랫폼 다각형은 꼭짓점을 3개 이상 찍어야 합니다.'))
            return
        self.set_tool("select")

    def update_diameter(self, value: float) -> None:
        self.profile.setdefault("pool", {})["diameter_cm"] = float(value)
        if hasattr(self, "wall_zone_cm"):
            self.wall_zone_cm.setMaximum(max(0.5, float(value) * 0.475))
        self.profile["peripheral_ring_percent"] = WaterMazeCanvas.wall_ring_fraction(self.profile)
        self.refresh_wall_zone_conversion()
        self.refresh_review_summary()
        self.canvas.update()

    def set_north_angle(self, value: float) -> None:
        self.profile["north_angle_degrees"] = float(value) % 360.0
        self.refresh_review_summary()
        self.canvas.update()

    def refresh_wall_zone_conversion(self) -> None:
        if not hasattr(self, "wall_zone_conversion"):
            return
        diameter_cm = max(float(self.profile.get("pool", {}).get("diameter_cm", 120.0)), 1e-6)
        wall_cm = float(self.profile.get("wall_zone_width_cm", 10.0))
        radius_percent = wall_cm / (diameter_cm / 2.0) * 100.0
        area_percent = (1.0 - max(0.0, 1.0 - radius_percent / 100.0) ** 2) * 100.0
        self.wall_zone_conversion.setText(
            tr('현재 수조 지름 {p0:.1f} cm 기준 · 반지름의 {p1:.1f}% · 전체 수면 면적의 약 {p2:.1f}%', p0=diameter_cm, p1=radius_percent, p2=area_percent)
        )

    def update_wall_zone(self, value: float) -> None:
        self.profile["wall_zone_width_cm"] = float(value)
        self.profile["peripheral_ring_percent"] = WaterMazeCanvas.wall_ring_fraction(self.profile)
        self.refresh_wall_zone_conversion()
        self.refresh_review_summary()
        self.canvas.update()

    def keyPressEvent(self, event) -> None:
        if event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_S:
            self.save_profile()
        elif event.modifiers() == Qt.AltModifier and Qt.Key_1 <= event.key() <= Qt.Key_3:
            self.set_water_stage(event.key() - Qt.Key_0)
        elif event.modifiers() == Qt.NoModifier and event.key() == Qt.Key_V:
            self.set_tool("select")
        elif event.modifiers() == Qt.NoModifier and event.key() == Qt.Key_C:
            self.set_tool("pool_circle")
        elif event.modifiers() == Qt.NoModifier and event.key() == Qt.Key_E:
            self.set_tool("pool_ellipse")
        elif event.modifiers() == Qt.NoModifier and event.key() == Qt.Key_O:
            self.set_tool("platform_circle")
        elif event.modifiers() == Qt.NoModifier and event.key() == Qt.Key_R:
            self.set_tool("platform_rectangle")
        elif event.modifiers() == Qt.NoModifier and event.key() == Qt.Key_P:
            self.set_tool("platform_polygon")
        else:
            super().keyPressEvent(event)

    def save_profile(self) -> None:
        pool = self.profile.get("pool", {}) or {}
        platform = self.profile.get("platform", {}) or {}
        pool_radius_x, pool_radius_y = self.canvas.pool_radii(pool)
        if not pool.get("center_px") or pool_radius_x <= 0 or pool_radius_y <= 0:
            QMessageBox.information(self, tr('수조 ROI 필요'), tr('수조 ROI를 먼저 지정하세요.'))
            return
        platform_type = str(platform.get("shape_type", "circle"))
        platform_valid = (
            bool(platform.get("center_px")) and float(platform.get("radius_px", 0)) > 0
            if platform_type == "circle"
            else len(platform.get("points_px") or []) >= 3
        )
        if not platform_valid:
            QMessageBox.information(self, tr('플랫폼 ROI 필요'), tr('플랫폼 ROI를 원형·사각형·다각형 중 하나로 지정하세요.'))
            return
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        self.profile_path.write_text(json.dumps(self.profile, ensure_ascii=False, indent=2), encoding="utf-8")
        self.accept()


class WaterMazePage(QWidget):
    def __init__(self, host) -> None:
        super().__init__()
        self.host = host
        self._loaded_project_root: Path | None = None
        self._loaded_tracking_root: Path | None = None
        self.records: list[dict] = []
        self.visible_records: list[dict] = []
        self.bundles: list[dict] = []
        self.selected_bundle = ""
        self.checked_keys: set[str] = set()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        page, layout = self.host.workbench_page_shell(
            tr('Water maze · Morris 수중미로'),
            tr('Arena별 트래킹 좌표로 수조 탐색, 플랫폼 학습, Probe 기억을 분석합니다. Hidden·Probe·Visible·Reversal Trial을 한 프로젝트에서 함께 관리합니다.'),
            "BEHAVIOR ANALYSIS",
        )
        outer.addWidget(page)
        self.page_scroll_area = page.page_scroll_area
        layout.addWidget(self.host.workflow_ribbon((tr('영상·수조·Trial 설정'), tr('판정·분석'), tr('결과·QC'))))

        input_card, input_layout, self.input_badge = self.host.workflow_card(
            "01", tr('영상·수조·Trial 설정'),
            tr('프로젝트 영상을 선택하고 같은 작업 영역에서 Arena의 수조·플랫폼과 Trial 정보를 이어서 설정합니다.'),
            tr('입력 필요'), True,
        )
        row = QHBoxLayout()
        self.tracked_edit = QLineEdit(); self.tracked_edit.setPlaceholderText("…\\hagisilta-outputs\\tracked")
        row.addWidget(QLabel(tr('프로젝트 트래킹 폴더'), objectName="fieldLabel")); row.addWidget(self.tracked_edit, 1)
        choose = QPushButton(tr('찾기'), objectName="toolButton"); choose.clicked.connect(self.choose_root)
        row.addWidget(choose)
        input_layout.addLayout(row)
        actions = QHBoxLayout()
        actions.addWidget(QLabel(tr('프로젝트 원본 영상'), objectName="sectionLabel"), 1)
        add_video = QPushButton(tr('영상 추가'), objectName="toolButton")
        self.host.set_action_icon(add_video, "add")
        add_video.setToolTip(
            tr('프로젝트 공통 원본 영상 목록에 영상을 추가합니다. 이전에 목록에서 제외한 영상을 다시 고르면 제외 기록을 해제하고 기존 결과를 다시 표시합니다.')
        )
        add_video.clicked.connect(self.host.add_source_videos)
        refresh = QPushButton(tr('새로고침'), objectName="toolButton"); refresh.clicked.connect(self.refresh)
        self.host.set_action_icon(refresh, "refresh")
        select_all_videos = QPushButton(tr('영상 모두 사용'), objectName="toolButton")
        clear_all_videos = QPushButton(tr('영상 모두 제외'), objectName="toolButton")
        self.host.set_action_icon(select_all_videos, "include")
        self.host.set_action_icon(clear_all_videos, "exclude")
        select_all_videos.clicked.connect(lambda: self.check_all_videos(True))
        clear_all_videos.clicked.connect(lambda: self.check_all_videos(False))
        actions.addWidget(add_video); actions.addWidget(refresh)
        actions.addWidget(select_all_videos); actions.addWidget(clear_all_videos)
        input_layout.addLayout(actions)

        self.video_table = self.host.context_table(0, 7)
        self.video_table.setHorizontalHeaderLabels(
            [tr('사용'), tr('원본 영상'), tr('트래킹 묶음'), "Arena", tr('트래킹 좌표'), tr('수조 설정'), tr('상태')]
        )
        self.video_table.horizontalHeaderItem(4).setToolTip(
            tr('모델이 영상에서 계산한 신체점 위치 좌표가 각 Arena에 있는지 표시합니다.')
        )
        self.video_table.verticalHeader().setVisible(False)
        self.video_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.video_table.horizontalHeader().setStretchLastSection(True)
        self.video_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.video_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.video_table.setAlternatingRowColors(True)
        self.video_table.setMaximumHeight(175)
        self.video_table.cellClicked.connect(self.select_video_bundle)
        self.video_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.video_table.customContextMenuRequested.connect(self.show_video_context_menu)
        self.host.add_resizable_workbench_table(
            input_layout, self.video_table, "ui/table_heights/watermaze_video", default_height=200
        )

        config_layout = input_layout
        source_action_panel = QFrame(objectName="reviewSelectionPanel")
        source_action_layout = QHBoxLayout(source_action_panel)
        source_action_layout.setContentsMargins(12, 9, 12, 9)
        source_action_copy = QVBoxLayout()
        source_action_copy.setSpacing(2)
        source_action_copy.addWidget(QLabel(tr('선택 원본 영상 설정'), objectName="reviewSelectionEyebrow"))
        self.selected_video_action_title = QLabel(
            tr('위 표에서 편집할 원본 영상을 선택하세요.'), objectName="reviewSelectionTitle"
        )
        self.selected_video_action_title.setWordWrap(True)
        source_action_copy.addWidget(self.selected_video_action_title)
        source_action_layout.addLayout(source_action_copy, 1)
        edit = QPushButton(tr('수조·플랫폼 편집'), objectName="toolButton")
        self.host.set_action_icon(edit, "edit")
        edit.setToolTip(
            tr('파란 Trial 행에 연결된 Arena의 수조 원과 플랫폼을 편집합니다.')
        )
        edit.clicked.connect(self.edit_current)
        review = QPushButton(tr('Water maze 전체 영상 ROI 검수'), objectName="toolButton")
        self.host.set_action_icon(review, "inspect")
        review.setToolTip(tr('모든 영상의 수조 원과 플랫폼 위치를 한 화면에서 확인합니다.'))
        review.clicked.connect(self.open_overview)
        source_action_layout.addWidget(edit)
        source_action_layout.addWidget(review)
        config_layout.addWidget(source_action_panel)

        trial_panel = QFrame(objectName="reviewSelectionPanel")
        trial_panel_layout = QVBoxLayout(trial_panel)
        trial_panel_layout.setContentsMargins(12, 9, 12, 12)
        trial_panel_layout.setSpacing(8)
        arena_toolbar = QHBoxLayout()
        trial_heading = QVBoxLayout()
        trial_heading.setSpacing(2)
        trial_heading.addWidget(QLabel("ARENA · TRIAL", objectName="reviewSelectionEyebrow"))
        trial_heading.addWidget(QLabel(
            tr('Arena·Trial 구성'),
            objectName="reviewSelectionTitle",
        ))
        arena_toolbar.addLayout(trial_heading, 1)
        edit_range = QPushButton(tr('Trial 구간·실험 정보 편집'), objectName="toolButton")
        self.host.set_action_icon(edit_range, "edit")
        edit_range.setToolTip(
            tr('파란 Trial 행과 같은 Arena 영상에 포함된 모든 Trial 구간을 한 창에서 추가·삭제·편집합니다. 개체 ID·실험군·Day·Trial·시험 유형과 시작·종료 시각을 함께 저장합니다.')
        )
        edit_range.clicked.connect(self.edit_analysis_range)
        select_all = QPushButton(tr('Trial 모두 사용'), objectName="toolButton"); select_all.clicked.connect(lambda: self.check_all(True))
        clear_all = QPushButton(tr('Trial 모두 제외'), objectName="toolButton"); clear_all.clicked.connect(lambda: self.check_all(False))
        self.host.set_action_icon(select_all, "include")
        self.host.set_action_icon(clear_all, "exclude")
        arena_toolbar.addWidget(edit_range)
        batch_range = QPushButton(tr('구간 일괄 설정'), objectName="toolButton")
        self.host.set_action_icon(batch_range, "settings_tune")
        batch_range.setToolTip(tr('사용 중인 모든 Trial의 시작을 유지하고 길이를 맞추거나 기준 구간을 복사합니다.'))
        batch_range.clicked.connect(self.batch_analysis_ranges)
        arena_toolbar.addWidget(batch_range)
        arena_toolbar.addWidget(select_all)
        arena_toolbar.addWidget(clear_all)
        trial_panel_layout.addLayout(arena_toolbar)
        self.table = self.host.context_table(0, 12)
        self.table.setHorizontalHeaderLabels(
            [
                tr('사용'), tr('프로젝트 영상'), "Arena", tr('개체 ID'), tr('실험군'), "Day", "Trial",
                tr('시험 유형'), tr('시작 방향'), tr('플랫폼'), tr('분석 구간'), tr('수조 설정'),
            ]
        )
        self.table.horizontalHeaderItem(7).setToolTip(tr(TRIAL_TYPE_TOOLTIP))
        self.table.horizontalHeaderItem(8).setToolTip(tr(START_DIRECTION_TOOLTIP))
        self.table.horizontalHeaderItem(9).setToolTip(tr(TARGET_QUADRANT_TOOLTIP))
        self.table.horizontalHeaderItem(10).setToolTip(
            tr('이 Trial에서 실제로 분석할 시작·종료 시각입니다. 시험 유형과 무관하게 이 구간 전체가 CSV·궤적·히트맵·판정 영상에 사용됩니다.')
        )
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(36)
        self.table.verticalHeader().setMinimumSectionSize(36)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(190)
        self.table.cellDoubleClicked.connect(lambda _r, _c: self.edit_current())
        self.table.itemSelectionChanged.connect(self.update_range_summary)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_trial_context_menu)
        self.host.add_resizable_workbench_table(
            trial_panel_layout, self.table, "ui/table_heights/watermaze_trials", default_height=210
        )
        config_layout.addWidget(trial_panel)
        layout.addWidget(input_card)

        settings, settings_layout, self.analysis_badge = self.host.workflow_card(
            "02", tr('판정 설정·분석'),
            tr('DLC 좌표로 플랫폼 도달, 이동거리, 목표 사분면 체류와 벽면 선호를 계산합니다.'),
            tr('실행 전'), False,
        )
        grid = QGridLayout()
        bodypart_title = QLabel(tr('판정 신체점 · 체크한 점의 중심'), objectName="fieldLabel")
        bodypart_tip = (
            tr('사용 체크한 Arena의 실제 트래킹 좌표에서 위치 판정에 사용할 점을 체크합니다. 결과가 없으면 현재 트래킹 모델의 신체점을 표시합니다. 이름과 순서는 원본 정보를 그대로 사용합니다.\n\n계산 방식: 체크한 모든 점이 DLC 신뢰도 기준을 통과한 프레임만 유효합니다. 그 점들의 좌표 평균(중심)을 대표 수영 위치로 사용하여 거리·속도·사분면·플랫폼 도달을 계산합니다.\n\n한 점만 체크하면 기존 단일점 판정과 같습니다. 여러 점은 한 점의 순간적인 흔들림을 줄이지만, 한 점이라도 자주 가려지면 유효 좌표 비율이 낮아질 수 있습니다. 처음에는 모델에서 몸 중심에 해당하는 점을 권장하며 결과의 유효 좌표(%)와 판정 영상을 확인하세요. 실제 분석 좌표에 선택한 점이 없으면 모델과 트래킹 결과를 확인해야 합니다.')
        )
        bodypart_title.setToolTip(bodypart_tip)
        self.bodyparts = QFrame(objectName="reviewSelectionPanel")
        self.bodyparts.setFixedWidth(280)
        self.bodyparts.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.bodyparts.setToolTip(bodypart_tip)
        self._bodyparts_help_tooltip = bodypart_tip
        self.bodyparts_layout = QVBoxLayout(self.bodyparts)
        self.bodyparts_layout.setContentsMargins(10, 7, 10, 7)
        self.bodyparts_layout.setSpacing(4)
        self.bodypart_checks: dict[str, QCheckBox] = {}
        self.refresh_bodyparts()
        grid.addWidget(bodypart_title, 0, 0)
        grid.addWidget(self.bodyparts, 1, 0, 4, 1, Qt.AlignLeft | Qt.AlignTop)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        self.pcutoff = QDoubleSpinBox(); self.pcutoff.setRange(0, 1); self.pcutoff.setSingleStep(.05); self.pcutoff.setValue(.6)
        self.dwell = QDoubleSpinBox(); self.dwell.setRange(.1, 10); self.dwell.setValue(2.0); self.dwell.setSuffix(tr(' 초'))
        self.bridge = QDoubleSpinBox(); self.bridge.setRange(0, 2); self.bridge.setValue(.2); self.bridge.setSuffix(tr(' 초'))
        self.crossing_dwell = QDoubleSpinBox(); self.crossing_dwell.setRange(.1, 5); self.crossing_dwell.setSingleStep(.1); self.crossing_dwell.setValue(.3); self.crossing_dwell.setSuffix(tr(' 초'))
        self.crossing_gap = QDoubleSpinBox(); self.crossing_gap.setRange(0, 2); self.crossing_gap.setSingleStep(.1); self.crossing_gap.setValue(.2); self.crossing_gap.setSuffix(tr(' 초'))
        self.max_duration = QDoubleSpinBox(); self.max_duration.setRange(0, 7200); self.max_duration.setValue(0); self.max_duration.setSuffix(tr(' 초'))
        self.max_duration.setSpecialValueText(tr('구간 전체'))
        controls = [
            (
                tr('DLC 신뢰도'),
                self.pcutoff,
                tr('선택한 신체점의 likelihood가 이 값보다 낮으면 해당 프레임의 좌표를 누락으로 처리합니다. 너무 높이면 유효 좌표가 줄고, 너무 낮추면 오인식 좌표가 궤적에 포함될 수 있습니다. 결과의 유효 좌표(%)와 판정 영상을 함께 확인하세요.'),
            ),
            (
                tr('플랫폼 체류'),
                self.dwell,
                tr("신체점이 플랫폼 영역 안에 연속으로 머물러야 '플랫폼 도달'로 확정되는 최소 시간입니다. 짧게 스치거나 추적점이 순간적으로 튄 경우를 제외합니다. 이 값은 도달 여부와 Latency에만 영향을 주며, 사용자가 지정한 분석 종료 시각이나 판정 영상 길이를 줄이지 않습니다."),
            ),
            (
                tr('짧은 누락 연결'),
                self.bridge,
                tr('DLC 신뢰도가 낮아진 짧은 좌표 공백을 앞뒤 유효 좌표 사이의 직선으로 보간하는 최대 시간입니다. 그보다 긴 공백은 실제 위치를 추측하지 않고 누락으로 유지합니다. 값이 너무 크면 서로 떨어진 위치가 직선으로 연결되어 이동거리가 왜곡될 수 있습니다.'),
            ),
            (
                tr('플랫폼 교차 최소 유지'),
                self.crossing_dwell,
                tr('플랫폼 영역 안에 연속으로 머문 시간이 이 값 이상일 때만 플랫폼 교차 1회로 확정합니다. 플랫폼 도달 판정의 2초 체류와는 별도이며, 빠른 정상 통과는 살리고 한두 프레임의 좌표 흔들림은 제외합니다. Probe 시험의 이전 플랫폼 위치 통과 횟수에는 기본 0.3초를 권장합니다.'),
            ),
            (
                tr('동일 교차 연결 간격'),
                self.crossing_gap,
                tr("플랫폼을 통과하는 도중 좌표가 잠깐 플랫폼 밖으로 벗어난 간격이 이 값 이하이면 하나의 교차로 연결합니다. 서로 다른 교차 사건까지 합치지 않도록 기본 0.2초를 권장합니다. 좌표 자체를 보간하는 '짧은 누락 연결'과는 별도 기준입니다."),
            ),
            (
                tr('추가 최대시간 제한'),
                self.max_duration,
                tr('각 Arena에서 지정한 분석 구간을 기본으로 사용합니다. 0은 선택 구간 전체이며, 양수를 입력하면 선택 구간 시작부터 해당 시간까지만 분석하는 안전 제한입니다. Trial 편집기에서 정확한 시작·종료를 지정했다면 보통 0으로 둡니다.'),
            ),
        ]
        for index, (label, control, tip) in enumerate(controls):
            title = QLabel(label, objectName="fieldLabel"); title.setToolTip(tip); control.setToolTip(tip)
            column = 1 + index % 2
            row = (index // 2) * 2
            grid.addWidget(title, row, column)
            grid.addWidget(control, row + 1, column)
        settings_layout.addLayout(grid)
        self.analysis_selection_summary = QFrame(objectName="statusSummaryBar")
        summary_layout = QVBoxLayout(self.analysis_selection_summary)
        summary_layout.setContentsMargins(10, 6, 10, 6)
        summary_layout.setSpacing(1)
        self.analysis_selection_value = QLabel(tr('선택된 Trial 없음'), objectName="statusSummaryValue")
        self.analysis_selection_detail = QLabel(
            tr('위 표의 사용 체크로 실행 대상을 선택하세요.'), objectName="reviewSelectionPath"
        )
        self.analysis_selection_value.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.analysis_selection_detail.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        summary_layout.addWidget(QLabel(tr('실행 대상'), objectName="statusSummaryTitle"))
        summary_layout.addWidget(self.analysis_selection_value)
        summary_layout.addWidget(self.analysis_selection_detail)
        self.analysis_run_button = QPushButton(tr('Water maze 분석 실행'), objectName="primaryButton")
        self.analysis_run_button.clicked.connect(self.run_analysis)
        self.analysis_run_button.setFixedWidth(220)
        run_row = QHBoxLayout()
        run_row.setSpacing(10)
        run_row.addWidget(self.analysis_selection_summary, 1)
        run_row.addWidget(self.analysis_run_button, 0, Qt.AlignVCenter)
        settings_layout.addLayout(run_row)
        layout.addWidget(settings)

        result, result_layout, self.result_badge = self.host.workflow_card(
            "03", tr('결과·QC'),
            tr('Trial별 플랫폼 도달·거리·Probe 기억 지표를 확인하고 궤적과 히트맵을 함께 검수합니다.'),
            tr('결과 없음'), False,
        )
        result_toolbar = QHBoxLayout()
        result_toolbar.setContentsMargins(0, 0, 0, 0)
        result_toolbar.setSpacing(6)
        self.result_run_label = QLabel(tr('표시 결과 없음'), objectName="subtleLabel")
        self.result_run_label.setToolTip(tr('현재 표에 표시 중인 Water maze 분석 실행입니다.'))
        result_toolbar.addWidget(self.result_run_label)
        result_toolbar.addStretch()
        history = QPushButton(tr('분석 이력 불러오기'), objectName="toolButton"); history.clicked.connect(self.load_history)
        self.host.set_action_icon(history, "import")
        result_toolbar.addWidget(history)
        decision_video = QPushButton(tr('판정 영상 보기'), objectName="toolButton")
        self.host.set_action_icon(decision_video, "open_video")
        decision_video.setToolTip(
            tr('선택 Trial을 대화형 판정창으로 엽니다. 수조·플랫폼·판정점·정보판과 현재까지/최종 궤적·히트맵을 각각 켜거나 끌 수 있습니다.')
        )
        decision_video.clicked.connect(self.open_annotated_video)
        result_toolbar.addWidget(decision_video)
        visual = QPushButton(tr('궤적·히트맵 편집'), objectName="toolButton"); visual.clicked.connect(self.open_visual)
        self.host.set_action_icon(visual, "edit")
        result_toolbar.addWidget(visual)
        csv_button = QPushButton(tr('CSV 열기'), objectName="toolButton")
        self.host.set_action_icon(csv_button, "dataset")
        csv_button.setToolTip(tr('결과 표에서 선택한 Trial의 프레임별 판정 CSV를 엽니다.'))
        csv_button.clicked.connect(self.open_result_csv)
        result_toolbar.addWidget(csv_button)
        output = QPushButton(tr('결과 폴더 열기'), objectName="toolButton"); output.clicked.connect(self.open_output)
        self.host.set_action_icon(output, "folder")
        result_toolbar.addWidget(output)
        result_layout.addLayout(result_toolbar)
        self.result_table = self.host.context_table(0, 12)
        self.result_table.setHorizontalHeaderLabels([
            tr('영상'), "Arena", tr('개체 ID'), tr('실험군'), tr('유형'), "Day", "Trial",
            tr('도달'), "Latency(s)", tr('거리(cm)'), tr('효율'), tr('벽면(%)'),
        ])
        result_tips = {
            4: tr('선택한 Hidden·Probe·Visible·Reversal 시험 유형입니다. 계산 구간은 같지만 결과 해석 목적이 다릅니다.'),
            7: tr('설정한 최소 플랫폼 체류시간을 만족했는지 표시합니다.'),
            8: tr('분석 구간 시작부터 최초 확정 플랫폼 도달까지의 시간입니다.'),
            9: tr('유효한 추적 좌표를 따라 이동한 총거리입니다.'),
            10: tr('시작점에서 플랫폼까지의 직선거리 ÷ 실제 이동거리입니다.'),
            11: tr('수조 벽에서 설정한 폭 안에 머문 유효시간의 비율(thigmotaxis)입니다.'),
        }
        for column, tip in result_tips.items():
            self.result_table.horizontalHeaderItem(column).setToolTip(tip)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.result_table.horizontalHeader().setStretchLastSection(True)
        self.result_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.result_table.setAlternatingRowColors(True)
        self.result_table.setSortingEnabled(True)
        self.result_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.result_table.customContextMenuRequested.connect(self.show_result_context_menu)
        self.result_table.setMinimumHeight(220)
        self.host.add_resizable_workbench_table(
            result_layout, self.result_table, "ui/table_heights/watermaze_result", default_height=240
        )
        layout.addWidget(result)
        layout.addStretch()

    @property
    def project_root(self) -> Path:
        return self.host.config.parent if self.host.config else self.host.work_root

    def project_bodyparts(self) -> list[str]:
        """Read the checked analysis inputs before the selected model's schema."""
        self._bodyparts_resolution_error = ""
        self._bodyparts_resolution_detail = ""
        try:
            analysis_resolver = getattr(self.host, "analysis_bodypart_resolution", None)
            if callable(analysis_resolver):
                info = analysis_resolver(self.bodypart_arena_dirs()) or {}
                self._bodyparts_resolution_detail = str(info.get("detail", "") or "")
                values = info.get("bodyparts", []) or []
                return list(dict.fromkeys(str(value) for value in values if value is not None and str(value).strip()))
            resolver = getattr(self.host, "tracking_bodyparts", None)
            if not callable(resolver):
                return []
            values = resolver() or []
            return list(dict.fromkeys(str(value) for value in values if value is not None and str(value).strip()))
        except Exception as exc:
            self._bodyparts_resolution_error = str(exc)
            return []

    def bodypart_arena_dirs(self) -> list[Path]:
        """Never reuse a previous project's rows while a project is switching."""
        config = getattr(self.host, "config", None)
        loaded = getattr(self, "_loaded_project_root", None)
        if not config or loaded is None or Path(loaded).resolve() != Path(config).resolve().parent:
            return []
        tracked_edit = getattr(self, "tracked_edit", None)
        if tracked_edit is not None:
            text = tracked_edit.text().strip()
            tracking_root = getattr(self, "_loaded_tracking_root", None)
            if not text or tracking_root is None or Path(tracking_root).resolve() != Path(text).resolve():
                return []
        checked = getattr(self, "checked_keys", set())
        return list(dict.fromkeys(
            Path(record["arena"]).resolve()
            for record in getattr(self, "records", [])
            if record.get("key") in checked and record.get("arena")
        ))

    @staticmethod
    def default_bodypart_selection(values: list[str]) -> list[str]:
        if not values:
            return []
        # Only choose among actual names; do not synthesize renamed aliases.
        if "body_center" in values:
            return ["body_center"]
        centers = [value for value in values if value.casefold() in {"body_center", "bodycenter", "mouse_center"}]
        return centers if len(centers) == 1 else [values[0]]

    def refresh_bodyparts(self) -> None:
        if not hasattr(self, "bodyparts"):
            return
        values = self.project_bodyparts()
        config = getattr(self.host, "config", None)
        project_key = str(Path(config).resolve()) if config else ""
        context = (project_key, tuple(values))
        previous = getattr(self, "_bodyparts_context", None)
        if context == previous:
            self.update_bodypart_empty_hint()
            return
        selected = None
        if previous is None or previous[0] != project_key:
            selected = self.default_bodypart_selection(values)
        self.populate_bodyparts(values, selected)
        self._bodyparts_context = context
        self.update_bodypart_empty_hint()

    def update_bodypart_empty_hint(self) -> None:
        detail = getattr(self, "_bodyparts_resolution_detail", "")
        error = getattr(self, "_bodyparts_resolution_error", "")
        extra = "\n".join(value for value in (detail, error) if value)
        base = getattr(self, "_bodyparts_help_tooltip", "")
        self.bodyparts.setToolTip("\n\n".join(value for value in (base, extra) if value))
        hint = getattr(self, "bodyparts_empty_hint", None)
        if hint is not None:
            hint.setToolTip(
                tr('분석할 트래킹 좌표와 현재 모델에서 사용할 신체점을 확인하세요.')
                + ("\n" + extra if extra else "")
            )

    def populate_bodyparts(self, values: list[str], selected: list[str] | None = None) -> None:
        if not hasattr(self, "bodyparts"):
            return
        values = list(dict.fromkeys(str(value) for value in values if str(value).strip()))
        if selected is None:
            previous = getattr(self, "bodypart_checks", {})
            selected = [name for name, checkbox in previous.items() if checkbox.isChecked()]
            if not previous or (selected and not set(selected).intersection(values)):
                selected = self.default_bodypart_selection(values)
        selected_set = set(selected)
        while self.bodyparts_layout.count():
            item = self.bodyparts_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.bodypart_checks = {}
        self.bodyparts_empty_hint = None
        for value in values:
            checkbox = QCheckBox(value)
            checkbox.setToolTip(value)
            checkbox.setChecked(value in selected_set)
            self.bodyparts_layout.addWidget(checkbox)
            self.bodypart_checks[value] = checkbox
        if not values:
            self.bodyparts_empty_hint = QLabel(
                tr('신체점 정보 없음 · 결과·모델을 확인하세요'), objectName="settingsGroupDescription"
            )
            self.bodyparts_empty_hint.setWordWrap(True)
            self.bodyparts_layout.addWidget(self.bodyparts_empty_hint)
            self.update_bodypart_empty_hint()

    def selected_bodyparts(self) -> list[str]:
        return [
            name for name, checkbox in self.bodypart_checks.items() if checkbox.isChecked()
        ]

    @property
    def metadata_path(self) -> Path:
        return project_state_root(self.project_root) / "watermaze-trials.json"

    def profile_path(self, bundle: Path, arena: Path) -> Path:
        return project_state_root(self.project_root) / "watermaze_profiles" / f"{bundle.name}_arena_{arena_number(arena):02d}.json"

    def choose_root(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, tr('트래킹 결과 폴더'), self.tracked_edit.text())
        if selected:
            self.tracked_edit.setText(selected); self.refresh()

    def load_metadata(self) -> dict:
        try:
            return json.loads(self.metadata_path.read_text(encoding="utf-8")) if self.metadata_path.is_file() else {}
        except Exception:
            return {}

    @staticmethod
    def video_duration(video: Path) -> float:
        capture = cv2.VideoCapture(str(video))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        frames = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
        capture.release()
        return frames / max(fps, 1e-9)

    @staticmethod
    def record_source_path(record: dict) -> str:
        metadata_path = Path(record["arena"]) / "roi-metadata.json"
        try:
            return str((json.loads(metadata_path.read_text(encoding="utf-8")) or {}).get("source_video", ""))
        except Exception:
            return ""

    @staticmethod
    def record_has_dlc(record: dict) -> bool:
        arena = Path(record["arena"])
        return bool(
            list(arena.glob("*DLC*.h5"))
            or list(arena.glob("*DLC*.csv"))
            or list(arena.glob("*corrected*.h5"))
            or list(arena.glob("*corrected*.csv"))
        )

    def sync_visible_records(self) -> None:
        for row, record in enumerate(self.visible_records):
            checkbox = self.host.table_checkbox(self.table, row, 0)
            if checkbox and checkbox.isChecked():
                self.checked_keys.add(record["key"])
            else:
                self.checked_keys.discard(record["key"])
            try:
                record["subject_id"] = self.table.item(row, 3).text().strip()
                record["group"] = self.table.item(row, 4).text().strip()
                record["day"] = int(self.table.item(row, 5).text() or 1)
                record["trial"] = int(self.table.item(row, 6).text() or 1)
                record["trial_type"] = self.table.cellWidget(row, 7).currentText()
                record["start_direction"] = self.table.cellWidget(row, 8).currentText()
                record["target_quadrant"] = self.table.cellWidget(row, 9).currentText()
            except (AttributeError, TypeError, ValueError):
                continue

    def refresh(self) -> None:
        if hasattr(self, "bodyparts"):
            self.refresh_bodyparts()
        if not self.host.config:
            return
        self.sync_visible_records()
        current_project_root = self.project_root.resolve()
        if self._loaded_project_root != current_project_root:
            self._loaded_project_root = current_project_root
            self.tracked_edit.setText(str(project_output_root(current_project_root) / "tracked"))
        elif not self.tracked_edit.text().strip():
            self.tracked_edit.setText(str(project_output_root(current_project_root) / "tracked"))
        root = Path(self.tracked_edit.text().strip())
        bundles = self.host.discover_ymaze_bundles(root)
        hidden_sources = self.host.project_hidden_source_keys()
        bundle_sources: dict[str, str] = {}
        for bundle in bundles:
            for arena in sorted(path for path in bundle.glob("arena_*") if path.is_dir()):
                source_path = self.record_source_path({"arena": arena})
                if source_path:
                    bundle_sources[str(bundle)] = source_path
                    break
        bundles = [
            bundle for bundle in bundles
            if not bundle_sources.get(str(bundle))
            or self.host.normalized_source_key(bundle_sources[str(bundle)]) not in hidden_sources
        ]
        metadata = self.load_metadata()
        self.records = []
        for bundle in bundles:
            for arena in sorted(path for path in bundle.glob("arena_*") if path.is_dir()):
                video = arena_video(arena)
                if not video: continue
                base_key = f"{bundle.name}/arena_{arena_number(arena):02d}"
                saved = metadata.get(base_key, {})
                if not isinstance(saved, dict):
                    saved = {}
                duration = self.video_duration(video)
                segments = saved.get("segments")
                # Keep an Arena row for ROI setup, but never mistake an unsaved
                # whole-video placeholder for an explicitly registered Trial.
                registered = bool(segments) if isinstance(segments, list) else bool(saved)
                if not isinstance(segments, list) or not segments:
                    # Backward-compatible migration from the former one-Trial-per-Arena format.
                    segments = [{
                        "segment_id": "trial_001",
                        "subject_id": saved.get("subject_id", bundle.name if registered else ""),
                        "group": saved.get("group", ""),
                        "day": saved.get("day", 1),
                        "trial": saved.get("trial", arena_number(arena) if registered else 1),
                        "trial_type": saved.get("trial_type", "Hidden"),
                        "start_direction": saved.get("start_direction", "N"),
                        "target_quadrant": saved.get("target_quadrant", "NE"),
                        "analysis_start_seconds": saved.get("analysis_start_seconds", 0.0),
                        "analysis_end_seconds": saved.get("analysis_end_seconds", duration),
                    }]
                used_ids: set[str] = set()
                for index, segment in enumerate(segments, 1):
                    segment_id = str(segment.get("segment_id") or f"trial_{index:03d}")
                    if segment_id in used_ids:
                        segment_id = f"trial_{index:03d}"
                    used_ids.add(segment_id)
                    analysis_start = min(max(float(segment.get("analysis_start_seconds", 0.0)), 0.0), duration)
                    analysis_end = min(max(float(segment.get("analysis_end_seconds", duration)), 0.0), duration)
                    if analysis_end <= analysis_start:
                        analysis_start, analysis_end = 0.0, duration
                    self.records.append({
                        "key": f"{base_key}#{segment_id}",
                        "base_key": base_key,
                        "segment_id": segment_id,
                        "_range_registered": registered,
                        "bundle": bundle, "arena": arena, "video": video,
                        "subject_id": segment.get("subject_id", bundle.name),
                        "group": segment.get("group", ""),
                        "day": segment.get("day", 1),
                        "trial": segment.get("trial", index),
                        "trial_type": segment.get("trial_type", "Hidden"),
                        "start_direction": segment.get("start_direction", "N"),
                        "target_quadrant": segment.get("target_quadrant", "NE"),
                        "analysis_start_seconds": analysis_start,
                        "analysis_end_seconds": analysis_end,
                        "video_duration_seconds": duration,
                        "profile": self.profile_path(bundle, arena),
                    })
        self._loaded_tracking_root = root.resolve()
        current_keys = {record["key"] for record in self.records}
        if not self.checked_keys:
            self.checked_keys = set(current_keys)
        else:
            previous_keys = set(self.checked_keys)
            self.checked_keys &= current_keys
            if not self.checked_keys and any("#" not in key for key in previous_keys):
                self.checked_keys = set(current_keys)
        self.bundles = []
        for bundle in bundles:
            bundle_records = [record for record in self.records if record["bundle"] == bundle]
            if not bundle_records:
                continue
            source_path = bundle_sources.get(str(bundle)) or self.record_source_path(bundle_records[0])
            self.bundles.append({
                "path": bundle,
                "name": bundle.name,
                "source_path": source_path,
                "records": bundle_records,
            })
        if self.selected_bundle not in {bundle["name"] for bundle in self.bundles}:
            self.selected_bundle = self.bundles[0]["name"] if self.bundles else ""
        self.populate_video_table()
        self.populate_arena_table()
        self.refresh_results()

    def update_input_summary(self):
        physical_records = list({record["base_key"]: record for record in self.records}.values())
        profile_count = sum(record["profile"].is_file() for record in physical_records)
        ready = (bool(physical_records) and profile_count == len(physical_records)
                 and all(record.get("_range_registered", True) for record in self.records))
        self.host.set_workflow_badge(
            self.input_badge,
            tr('{p0}개 영상 · {p1}/{p2} Arena 설정 · {p3} Trial', p0=len(self.bundles), p1=profile_count, p2=len(physical_records), p3=sum((record.get('_range_registered', True) for record in self.records))),
            ready,
        )

    def populate_video_table(self) -> None:
        self.video_table.setSortingEnabled(False)
        self.video_table.setRowCount(0)
        self.video_table.setRowCount(len(self.bundles))
        selected_row = -1
        for row, bundle in enumerate(self.bundles):
            records = bundle["records"]
            physical_records = list({record["base_key"]: record for record in records}.values())
            checked = bool(records) and all(record["key"] in self.checked_keys for record in records)
            checkbox = self.host.set_centered_table_checkbox(self.video_table, row, 0, checked)
            checkbox.stateChanged.connect(
                lambda _state, name=bundle["name"], box=checkbox: self.set_bundle_checked(name, box.isChecked())
            )
            source_name = Path(bundle["source_path"]).name if bundle["source_path"] else bundle["name"]
            dlc_count = sum(self.record_has_dlc(record) for record in physical_records)
            profile_count = sum(record["profile"].is_file() for record in physical_records)
            trial_count = sum(record.get("_range_registered", True) for record in records)
            values = [
                source_name,
                bundle["name"],
                f"{len(physical_records)} Arena · {trial_count} Trial",
                (
                    tr('모두 있음') if physical_records and dlc_count == len(physical_records)
                    else tr('일부 있음 ({p0}/{p1})', p0=dlc_count, p1=len(physical_records)) if dlc_count
                    else tr('없음')
                ),
                tr('{p0}/{p1} 설정', p0=profile_count, p1=len(physical_records)),
                tr('구간 등록 필요') if trial_count != len(records) else
                tr('완료') if physical_records and dlc_count == len(physical_records) and profile_count == len(physical_records) else tr('확인 필요'),
            ]
            for column, value in enumerate(values, 1):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter if column != 1 else Qt.AlignVCenter | Qt.AlignLeft)
                if column == 1:
                    item.setData(Qt.UserRole, bundle["name"])
                self.video_table.setItem(row, column, item)
            self.video_table.setRowHeight(row, 36)
            if bundle["name"] == self.selected_bundle:
                selected_row = row
        for column, width in enumerate((76, 175, 150, 82, 105, 120, 125)):
            self.video_table.setColumnWidth(column, width)
        if selected_row >= 0:
            self.video_table.selectRow(selected_row)
        self.update_selected_video_action_title()
        self.update_input_summary()

    def update_selected_video_action_title(self) -> None:
        if hasattr(self, "selected_video_action_title"):
            bundle = next(
                (item for item in self.bundles if item["name"] == self.selected_bundle), None
            )
            if bundle:
                arena_count = len({record["base_key"] for record in bundle["records"]})
                self.selected_video_action_title.setText(
                    tr('{p0} · {p1} Arena 선택', p0=bundle['name'], p1=arena_count)
                )
            else:
                self.selected_video_action_title.setText(
                    tr('위 표에서 편집할 원본 영상을 선택하세요.')
                )

    def populate_arena_table(self) -> None:
        self.visible_records = [
            record for record in self.records if record["bundle"].name == self.selected_bundle
        ]
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        self.table.setRowCount(len(self.visible_records))
        for row, record in enumerate(self.visible_records):
            checkbox = self.host.set_centered_table_checkbox(
                self.table, row, 0, record["key"] in self.checked_keys
            )
            checkbox.stateChanged.connect(
                lambda _state, key=record["key"], box=checkbox: self.set_record_checked(key, box.isChecked())
            )
            values = [
                record["bundle"].name,
                f"Arena {arena_number(record['arena']):02d}",
                record["subject_id"],
                record.get("group", ""),
                str(record["day"]),
                str(record["trial"]),
            ]
            for column, value in enumerate(values, 1):
                self.table.setItem(row, column, QTableWidgetItem(value))
            for column, options, current in (
                (7, ["Hidden", "Probe", "Visible", "Reversal"], record["trial_type"]),
                (8, ["N", "E", "S", "W"], record["start_direction"]),
                (9, ["NE", "SE", "SW", "NW"], record["target_quadrant"]),
            ):
                combo = QComboBox(objectName="tableCellCombo")
                combo.addItems(options)
                combo.setCurrentText(current)
                combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                combo.setToolTip(
                    tr(TRIAL_TYPE_TOOLTIP) if column == 7 else
                    tr(START_DIRECTION_TOOLTIP) if column == 8 else
                    tr(TARGET_QUADRANT_TOOLTIP)
                )
                self.table.setCellWidget(row, column, combo)
                self.table.register_context_cell_widget(combo)
            interval = (
                f"{format_clock(record['analysis_start_seconds'])} → "
                f"{format_clock(record['analysis_end_seconds'])}"
            ) if record.get("_range_registered", True) else tr('미등록 · 구간 등록 필요')
            interval_item = QTableWidgetItem(interval)
            interval_item.setTextAlignment(Qt.AlignCenter)
            interval_item.setToolTip(
                tr('선택 구간 {p0:.3f}초', p0=record['analysis_end_seconds'] - record['analysis_start_seconds'])
                if record.get("_range_registered", True) else
                tr('영상 확인·구간 편집에서 개체별 시작과 종료를 등록하세요. 전체 영상은 자동 등록되지 않습니다.')
            )
            self.table.setItem(row, 10, interval_item)
            status = tr('완료') if record["profile"].is_file() else tr('설정 필요')
            item = QTableWidgetItem(status); item.setTextAlignment(Qt.AlignCenter); self.table.setItem(row, 11, item)
            self.table.setRowHeight(row, 36)
        for column, width in enumerate((72, 145, 95, 125, 110, 72, 72, 115, 105, 95, 205, 125)):
            self.table.setColumnWidth(column, width)
        if self.visible_records:
            self.table.selectRow(0)
        self.update_range_summary()
        self.update_analysis_selection_summary()

    def set_record_checked(self, key: str, checked: bool) -> None:
        if checked:
            self.checked_keys.add(key)
        else:
            self.checked_keys.discard(key)
        self.populate_video_table()
        self.update_analysis_selection_summary()

    def set_bundle_checked(self, name: str, checked: bool) -> None:
        bundle = next((item for item in self.bundles if item["name"] == name), None)
        if not bundle:
            return
        for record in bundle["records"]:
            if checked:
                self.checked_keys.add(record["key"])
            else:
                self.checked_keys.discard(record["key"])
        if name == self.selected_bundle:
            self.populate_arena_table()
        self.update_analysis_selection_summary()

    def select_video_bundle(self, row: int, _column: int = 0) -> None:
        item = self.video_table.item(row, 1)
        name = str(item.data(Qt.UserRole) or "") if item else ""
        if not name:
            return
        self.sync_visible_records()
        self.selected_bundle = name
        self.populate_arena_table()
        self.video_table.selectRow(row)
        self.update_selected_video_action_title()

    def check_all_videos(self, checked: bool) -> None:
        self.checked_keys = {record["key"] for record in self.records} if checked else set()
        self.populate_video_table()
        self.populate_arena_table()
        self.update_analysis_selection_summary()

    @staticmethod
    def analysis_duration_text(seconds: float) -> str:
        seconds = max(float(seconds), 0.0)
        if seconds < 60:
            return tr('{p0:.1f}초', p0=seconds)
        minutes, remainder = divmod(seconds, 60)
        return tr('{p0}분 {p1:.0f}초', p0=int(minutes), p1=remainder)

    def update_analysis_selection_summary(self) -> None:
        self.refresh_bodyparts()
        if not hasattr(self, "analysis_selection_value"):
            return
        selected = [record for record in self.records if record["key"] in self.checked_keys]
        unregistered_count = sum(not record.get("_range_registered", True) for record in selected)
        selected = [record for record in selected if record.get("_range_registered", True)]
        if not selected:
            self.analysis_selection_value.setText(tr('{p0} Arena · 등록한 Trial 없음', p0=unregistered_count) if unregistered_count else tr('선택된 Trial 없음'))
            self.analysis_selection_detail.setText(tr('개체별 시작·종료 구간을 먼저 등록하세요.') if unregistered_count else tr('위 표의 사용 체크로 실행 대상을 선택하세요.'))
            self.analysis_selection_summary.setToolTip("")
            return

        grouped: dict[str, list[dict]] = {}
        for record in selected:
            grouped.setdefault(record["bundle"].name, []).append(record)
        arena_count = len({record["base_key"] for record in selected})
        total_seconds = sum(
            max(
                float(record["analysis_end_seconds"])
                - float(record["analysis_start_seconds"]),
                0.0,
            )
            for record in selected
        )
        self.analysis_selection_value.setText(
            tr('{p0}개 영상 · {p1} Arena · {p2} Trial · 총 {p3}', p0=len(grouped), p1=arena_count, p2=len(selected), p3=self.analysis_duration_text(total_seconds))
            + (tr(' · {p0} Arena 구간 등록 필요', p0=unregistered_count) if unregistered_count else "")
        )
        details = []
        for bundle_name, records in grouped.items():
            bundle_arenas = len({record["base_key"] for record in records})
            bundle_seconds = sum(
                max(
                    float(record["analysis_end_seconds"])
                    - float(record["analysis_start_seconds"]),
                    0.0,
                )
                for record in records
            )
            details.append(
                f"{bundle_name} · {bundle_arenas} Arena · {len(records)} Trial · "
                f"{self.analysis_duration_text(bundle_seconds)}"
            )
        full_detail = "\n".join(details)
        inline_detail = "  |  ".join(details)
        if len(inline_detail) > 170:
            inline_detail = inline_detail[:167].rstrip() + "…"
        self.analysis_selection_detail.setText(inline_detail)
        self.analysis_selection_summary.setToolTip(full_detail)

    def checked(self, row: int) -> bool:
        checkbox = self.host.table_checkbox(self.table, row, 0)
        return bool(checkbox and checkbox.isChecked())

    def check_all(self, checked: bool) -> None:
        for row in range(self.table.rowCount()):
            checkbox = self.host.table_checkbox(self.table, row, 0)
            if checkbox: checkbox.setChecked(checked)

    def current_record(self) -> dict | None:
        row = self.table.currentRow()
        return self.visible_records[row] if 0 <= row < len(self.visible_records) else None

    def update_range_summary(self) -> None:
        if not hasattr(self, "range_summary"):
            return
        record = self.current_record()
        if not record:
            self.range_summary.setText(tr('Arena 행을 선택하면 분석 구간이 표시됩니다.'))
            return
        if not record.get("_range_registered", True):
            self.range_summary.setText(tr('미등록 · 영상 확인·구간 편집에서 개체 ID와 시작·종료를 등록하세요.'))
            return
        start = float(record.get("analysis_start_seconds", 0.0))
        end = float(record.get("analysis_end_seconds", record.get("video_duration_seconds", 0.0)))
        self.range_summary.setText(
            tr('{p0} · {p1} · Arena {p2:02d}  |  {p3} → {p4}  ·  {p5:.3f}초', p0=record['subject_id'], p1=record.get('group') or tr('실험군 미지정'), p2=arena_number(record['arena']), p3=format_clock(start), p4=format_clock(end), p5=max(end - start, 0.0))
        )

    def batch_analysis_ranges(self) -> None:
        self.sync_visible_records()
        selected = [record for record in self.records if record["key"] in self.checked_keys]
        if not selected:
            QMessageBox.information(self, tr('구간 일괄 설정'), tr('적용할 Trial의 사용 체크를 켜세요.'))
            return
        if any(not record.get("_range_registered", True) for record in selected):
            QMessageBox.information(self, tr('구간 등록 필요'), tr('미등록 Arena가 선택돼 있습니다. 개체별 구간을 먼저 등록한 뒤 일괄 수정하세요.'))
            return
        targets = [{"id": item["key"], "label": f"{item['bundle'].name} · {item['arena'].name} · Trial {item['trial']} · {item['subject_id']}",
                    "start": float(item["analysis_start_seconds"]), "end": float(item["analysis_end_seconds"]),
                    "duration": float(item["video_duration_seconds"])} for item in selected]
        reference = self.current_record() or selected[0]
        dialog = BatchAnalysisRangeDialog(targets, (reference["analysis_start_seconds"], reference["analysis_end_seconds"]), self)
        if dialog.exec() != QDialog.Accepted:
            return
        updates = {item["id"]: item for item in dialog.accepted_proposals()}
        before = [dict(record) for record in self.records]
        for record in self.records:
            if record["key"] in updates:
                item = updates[record["key"]]
                record.update(analysis_start_seconds=item["new_start"], analysis_end_seconds=item["new_end"])
        try:
            self.save_metadata(sync_table=False)
        except OSError as error:
            self.records = before
            self.populate_arena_table()
            QMessageBox.warning(self, tr('구간 저장 실패'), str(error))
            return
        self.populate_arena_table()
        self.update_analysis_selection_summary()
        QMessageBox.information(self, tr('구간 적용 결과'), tr('{p0}개 Trial 저장 · {p1}개 미적용', p0=len(updates), p1=len(targets) - len(updates)))

    def edit_analysis_range(self) -> None:
        self.sync_visible_records()
        record = self.current_record()
        if not record:
            QMessageBox.information(self, tr('분석 구간'), tr('01 표에서 파란 Arena 행을 먼저 선택하세요.'))
            return
        siblings = [item for item in self.records if item["base_key"] == record["base_key"]
                    and item.get("_range_registered", True)]
        try:
            dialog = WaterMazeAnalysisRangeDialog(
                record["video"],
                siblings,
                record.get("segment_id", ""),
                self,
                on_save=lambda segments: self.save_trial_editor_segments(record, segments),
                draft_defaults=record,
                profile_path=record.get("profile"),
            )
        except Exception as error:
            QMessageBox.critical(self, tr('영상 열기 실패'), str(error))
            return
        dialog.exec()

    def save_trial_editor_segments(self, record, segments):
        """Persist explicit saves only; keep every existing row's analysis inclusion."""
        previous_records, previous_checked = self.records, set(self.checked_keys)
        siblings = {item["segment_id"]: item for item in self.records if item["base_key"] == record["base_key"]}
        old_keys = {item["key"] for item in siblings.values()}
        updated_records = [item for item in self.records if item["base_key"] != record["base_key"]]
        updated_checked = self.checked_keys - old_keys
        for segment in segments:
            existing = siblings.get(segment["segment_id"])
            updated = dict(existing or record)
            updated.update(segment)
            updated["_range_registered"] = True
            updated["key"] = f"{record['base_key']}#{segment['segment_id']}"
            updated_records.append(updated)
            if (existing["key"] if existing else record["key"]) in previous_checked:
                updated_checked.add(updated["key"])
        self.records, self.checked_keys = updated_records, updated_checked
        try:
            # Visible cells still describe the previous list; never sync them over the edit.
            self.save_metadata(sync_table=False)
        except Exception:
            self.records, self.checked_keys = previous_records, previous_checked
            raise
        for bundle in self.bundles:
            bundle["records"] = [item for item in self.records if item["bundle"].name == bundle["name"]]
        self.populate_arena_table()
        self.populate_video_table()
        self.update_analysis_selection_summary()

    def remove_trial_segment(self) -> None:
        self.sync_visible_records()
        rows = self.host.selected_table_rows(self.table)
        selected = [self.visible_records[row] for row in rows if 0 <= row < len(self.visible_records)]
        if not selected:
            QMessageBox.information(self, tr('Trial 삭제'), tr('아래 표에서 삭제할 Trial 행을 파란색으로 선택하세요.'))
            return
        selected_keys = {record["key"] for record in selected}
        invalid_groups = {
            record["base_key"]
            for record in selected
            if not any(
                item["base_key"] == record["base_key"] and item["key"] not in selected_keys
                for item in self.records
            )
        }
        if invalid_groups:
            QMessageBox.information(
                self,
                tr('Trial 삭제'),
                tr('각 Arena에는 최소 한 개의 Trial 구간이 필요합니다. 선택 범위를 줄여 주세요. 영상 자체를 빼려면 위 영상 목록에서 제외하세요.'),
            )
            return
        detail = "\n".join(
            f"• Trial {record['trial']} · {format_clock(record['analysis_start_seconds'])} → "
            f"{format_clock(record['analysis_end_seconds'])}"
            for record in selected[:8]
        )
        if len(selected) > 8:
            detail += tr('\n• 그 외 {p0}개', p0=len(selected) - 8)
        answer = QMessageBox.question(
            self,
            tr('Trial 구간 삭제'),
            tr('선택한 {p0}개 Trial 구간을 삭제할까요?\n\n{p1}\n\n원본 영상과 DLC 좌표는 삭제되지 않습니다.', p0=len(selected), p1=detail),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self.records = [item for item in self.records if item["key"] not in selected_keys]
        self.checked_keys.difference_update(selected_keys)
        self.visible_records = [item for item in self.visible_records if item["key"] not in selected_keys]
        self.save_metadata(sync_table=False)
        self.refresh()

    def remove_selected_source(self) -> None:
        rows = self.host.selected_table_rows(self.video_table)
        bundle_names = {
            str(item.data(Qt.UserRole) or "")
            for row in rows
            if (item := self.video_table.item(row, 1)) is not None
        }
        paths = {
            str(bundle.get("source_path", ""))
            for bundle in self.bundles
            if bundle["name"] in bundle_names and str(bundle.get("source_path", "")).strip()
        }
        if not paths:
            QMessageBox.information(self, tr('목록에서 제외'), tr('위 영상 표에서 제외할 행을 파란색으로 선택하세요.'))
            return
        if self.host.remove_project_source_paths(paths):
            self.refresh()

    def show_video_context_menu(self, position) -> None:
        rows = self.video_table.restore_right_click_rows(position)
        if not rows:
            return
        prefix = self.host.context_row_prefix(rows)
        menu = QMenu(self.video_table)
        edit = menu.addAction(tr('{p0} 수조·플랫폼 편집', p0=prefix))
        edit.setEnabled(len(rows) == 1)
        menu.addSeparator()
        remove = menu.addAction(tr('{p0} 목록에서 제외…', p0=prefix))
        menu.addSeparator()
        review = menu.addAction(tr('Water maze 전체 영상 ROI 검수'))
        chosen = menu.exec(self.video_table.viewport().mapToGlobal(position))
        if chosen == edit:
            self.select_video_bundle(rows[0], 1)
            self.edit_current()
        elif chosen == remove:
            self.remove_selected_source()
        elif chosen == review:
            self.open_overview()

    def show_trial_context_menu(self, position) -> None:
        rows = self.table.restore_right_click_rows(position)
        if not rows:
            return
        prefix = self.host.context_row_prefix(rows)
        menu = QMenu(self.table)
        edit = menu.addAction(tr('{p0} Trial 구간·실험 정보 편집', p0=prefix))
        edit.setEnabled(len(rows) == 1)
        profile = menu.addAction(tr('{p0} 수조·플랫폼 편집', p0=prefix))
        profile.setEnabled(len(rows) == 1)
        menu.addSeparator()
        delete = menu.addAction(tr('{p0} Trial 삭제…', p0=prefix))
        menu.addSeparator()
        batch_range = menu.addAction(tr('사용 중인 {p0}개 항목 구간 일괄 설정', p0=len(self.checked_keys)))
        batch_range.setEnabled(bool(self.checked_keys))
        review = menu.addAction(tr('Water maze 전체 영상 ROI 검수'))
        chosen = menu.exec(self.table.viewport().mapToGlobal(position))
        if chosen == edit:
            self.edit_analysis_range()
        elif chosen == profile:
            self.edit_current()
        elif chosen == delete:
            self.remove_trial_segment()
        elif chosen == batch_range:
            self.batch_analysis_ranges()
        elif chosen == review:
            self.open_overview()

    def show_result_context_menu(self, position) -> None:
        rows = self.result_table.restore_right_click_rows(position)
        if not rows:
            return
        prefix = self.host.context_row_prefix(rows)
        menu = QMenu(self.result_table)
        video = menu.addAction(tr('{p0} 판정 영상 보기', p0=prefix))
        visual = menu.addAction(tr('{p0} 궤적·히트맵 편집', p0=prefix))
        csv_action = menu.addAction(tr('{p0} CSV 열기', p0=prefix))
        video.setEnabled(len(rows) == 1)
        visual.setEnabled(len(rows) == 1)
        csv_action.setEnabled(len(rows) == 1)
        menu.addSeparator()
        folder = menu.addAction(tr('현재 결과 폴더 열기'))
        chosen = menu.exec(self.result_table.viewport().mapToGlobal(position))
        if chosen == video:
            self.open_annotated_video()
        elif chosen == visual:
            self.open_visual()
        elif chosen == csv_action:
            self.open_result_csv()
        elif chosen == folder:
            self.open_output()

    def overview_profile(self, record: dict) -> dict | None:
        """Read only this Arena's saved setup, never a shared/template fallback."""
        from hagisilta_watermaze_roi_overlay import load_water_roi_overlay

        try:
            profile = json.loads(Path(record["profile"]).read_text(encoding="utf-8"))
            if not isinstance(profile, dict):
                return None
            video = Path(record["video"])
            stat = video.stat()
            size_key = (str(video), stat.st_mtime_ns, stat.st_size)
            frame_size = getattr(self, "_overview_frame_sizes", {}).get(size_key)
            # Before decoding a thumbnail, validate the saved structure. Once a
            # frame is read, the actual dimensions are cached and also checked.
            overlay = load_water_roi_overlay(record["profile"], video,
                                             frame_size or tuple(profile.get("frame_size") or (1, 1)))
            if "platform" not in overlay.available:
                return None
            return profile
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return None

    def overview_records(self) -> list[dict]:
        """One card per physical Arena, even when it has several mouse Trials."""
        records = {}
        for record in self.records:
            if record["base_key"] in records:
                continue
            ready = self.overview_profile(record) is not None
            records[record["base_key"]] = {
                **record,
                "review_key": record["base_key"],
                "display_name": f"{record['bundle'].name} · Arena {arena_number(record['arena']):02d}",
                "preview_tooltip": tr('클릭하면 이 Arena의 수조·플랫폼을 편집하고 전체 검수로 돌아옵니다.'),
                "reference_ready": ready,
                "status": tr('수조·플랫폼 설정 완료') if ready else (
                    tr('설정 확인 필요 · 이 Arena를 편집하세요') if record["profile"].is_file() else tr('수조·플랫폼 미설정')
                ),
            }
        return list(records.values())

    def preview_pixmap(self, record: dict, width: int = 390, height: int = 235) -> QPixmap:
        from hagisilta_watermaze_roi_overlay import load_water_roi_overlay, paint_water_roi_overlay

        profile = self.overview_profile(record)
        try:
            seconds = float((profile or {}).get("reference_seconds", 0.0))
            source, image_width, image_height, _fps, _count = frame_at(record["video"], seconds)
            stat = Path(record["video"]).stat()
            if not hasattr(self, "_overview_frame_sizes"):
                self._overview_frame_sizes = {}
            self._overview_frame_sizes[(str(record["video"]), stat.st_mtime_ns, stat.st_size)] = (image_width, image_height)
        except Exception:
            record["reference_ready"] = False
            record["status"] = tr('영상 확인 필요 · 미리보기를 읽을 수 없습니다')
            fallback = QPixmap(width, height)
            fallback.fill(QColor("#07101d"))
            painter = QPainter(fallback)
            painter.setPen(QColor("#ff6685"))
            painter.drawText(fallback.rect(), Qt.AlignCenter, "VIDEO READ ERROR")
            painter.end()
            return fallback
        preview = source.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        try:
            overlay = load_water_roi_overlay(record["profile"], record["video"], (image_width, image_height))
            preview = paint_water_roi_overlay(preview, overlay, {"pool", "platform"})
        except ValueError as error:
            record["reference_ready"] = False
            record["status"] = tr('수조·플랫폼 미설정') if not record["profile"].is_file() else tr('설정 확인 필요 · 이 Arena를 편집하세요')
            record["preview_tooltip"] = tr('{p0}\n클릭하면 이 Arena의 수조·플랫폼을 편집합니다.', p0=error)
        canvas = QPixmap(width, height)
        canvas.fill(QColor("#07101d"))
        left = (width - preview.width()) / 2
        top = (height - preview.height()) / 2
        painter = QPainter(canvas)
        painter.drawPixmap(round(left), round(top), preview)
        painter.end()
        return canvas

    def focus_record(self, record: dict, dialog: QDialog | None = None) -> None:
        self.sync_visible_records()
        self.selected_bundle = record["bundle"].name
        self.populate_video_table()
        self.populate_arena_table()
        for row, candidate in enumerate(self.visible_records):
            if candidate["key"] == record["key"]:
                self.table.selectRow(row)
                break
        if dialog is None:
            self.edit_current()
            return
        # A nested editor returns to the same contact sheet; its search, checked
        # targets, reference and scroll position remain part of this review.
        timer = getattr(dialog, "refresh_timer", None)
        if timer is not None:
            timer.stop()
        try:
            editor = WaterMazeEditorDialog(record["video"], record["profile"], dialog)
            try:
                if editor.exec():
                    self.populate_video_table()
                    self.populate_arena_table()
            finally:
                editor.deleteLater()
        except Exception as error:
            QMessageBox.critical(dialog, tr('영상 열기 실패'), str(error))
        finally:
            dialog.refresh_records()
            if timer is not None:
                timer.start()

    def open_overview(self) -> None:
        if not self.records:
            QMessageBox.information(self, tr('전체 검수'), tr('검수할 Water maze 영상이 없습니다.'))
            return
        dialog = self.create_overview_dialog()
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()

    def create_overview_dialog(self):
        # Deferred import: the main platform also imports this experiment page.
        from hagisilta_platform import RoiOverviewDialog

        guide = (
            tr('모든 영상의 수조 외곽과 플랫폼을 한 화면에서 비교합니다. 같은 카메라 배치의 영상은 기준 카드를 정하고 적용 대상을 체크해 한 번에 복사할 수 있습니다. 카드를 클릭하면 해당 Arena만 편집하고 이 목록으로 돌아옵니다. 여러 Trial이 있어도 같은 Arena는 한 카드로 표시합니다. 개별 저장 설정이 없는 Arena에는 다른 영상의 설정을 대신 표시하지 않습니다.')
        )
        dialog = RoiOverviewDialog(
            self.overview_records(), tr('Water maze 전체 영상 ROI 검수'), guide, self,
            records_provider=self.overview_records, group_apply=True,
            group_apply_label=tr('체크 Arena에 수조·플랫폼 적용'),
            group_apply_tooltip=(
                tr('‘이 설정을 기준으로’를 누른 Arena의 설정을 ‘적용 대상’으로 체크한 Arena에 복사합니다. 같은 카메라 배치에만 적용하세요. 체크하지 않은 Arena와 Trial 정보는 유지합니다.')
            ),
            preview_provider=self.preview_pixmap, record_unit="Arena",
            group_apply_handler=lambda source, targets, overwrite: self.apply_overview_profiles(
                source, targets, overwrite, dialog
            ),
        )
        dialog.overwrite_existing.setText(tr('기존 설정도 덮어쓰기'))
        dialog.overwrite_existing.setToolTip(
            tr('끄면 설정 파일이 없는 체크 Arena에만 적용합니다. 켜면 기존 설정도 확인 후 복사하며 백업을 보존합니다.')
        )
        missing_button = getattr(dialog, "select_missing_button", None)
        if missing_button is not None:
            missing_button.setText(tr('설정 없는 Arena 선택'))
        dialog.edit_requested.connect(lambda record: self.focus_record(record, dialog))
        return dialog

    def apply_overview_profiles(self, source: dict, targets: list[dict], overwrite: bool, dialog) -> bool:
        from hagisilta_atomic_io import atomic_json
        import time

        payload = self.overview_profile(source)
        if payload is None:
            QMessageBox.information(dialog, tr('기준 설정 필요'), tr('먼저 기준 Arena의 수조·플랫폼을 확인하고 저장하세요.'))
            return False
        eligible = {item["base_key"]: item for item in targets
                    if item["base_key"] != source["base_key"]
                    and (overwrite or not item["profile"].is_file())}
        if not eligible:
            QMessageBox.information(dialog, tr('적용 대상 없음'), tr('기존 설정을 바꾸려면 ‘기존 설정도 덮어쓰기’를 켜세요.'))
            return False
        existing = sum(item["profile"].is_file() for item in eligible.values())
        question = tr('체크한 {p0}개 Arena에 기준 수조·플랫폼을 적용할까요?', p0=len(eligible))
        if existing:
            question += tr('\n기존 설정 {p0}개는 백업한 뒤 덮어씁니다.', p0=existing)
        question += tr('\n원본 영상·트래킹 ROI·Trial 구간과 개체 정보는 유지합니다.')
        if QMessageBox.question(dialog, tr('수조·플랫폼 일괄 적용'), question,
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return False
        applied = 0
        try:
            for target in eligible.values():
                if not overwrite and target["profile"].is_file():
                    raise OSError(tr('적용 확인 이후 설정이 생성되어 덮어쓰지 않았습니다: {p0}', p0=target['profile'].name))
                target_payload = json.loads(json.dumps(payload))
                target_payload["source_video"] = str(target["video"])
                if target["profile"].is_file():
                    backup = target["profile"].parent / ".hagisilta-backups" / f"{target['profile'].stem}-{time.time_ns()}.json"
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(target["profile"], backup)
                atomic_json(target["profile"], target_payload)
                applied += 1
        except (OSError, ValueError) as error:
            QMessageBox.critical(dialog, tr('일괄 적용 중단'), tr('{p0}개 적용 후 중단했습니다. 기존 덮어쓴 설정은 백업에 보존했습니다.\n{p1}', p0=applied, p1=error))
            return False
        finally:
            self.populate_video_table()
            self.populate_arena_table()
        QMessageBox.information(dialog, tr('일괄 적용 완료'), tr('{p0}개 Arena에 수조·플랫폼 설정을 적용했습니다.', p0=applied))
        return True

    def edit_current(self) -> None:
        record = self.current_record()
        if not record:
            QMessageBox.information(self, tr('수조 설정'), tr('표에서 편집할 Trial 행을 선택하세요.'))
            return
        try:
            dialog = WaterMazeEditorDialog(record["video"], record["profile"], self)
        except Exception as error:
            QMessageBox.critical(self, tr('영상 열기 실패'), str(error)); return
        if dialog.exec():
            self.table.item(self.table.currentRow(), 11).setText(tr('완료'))
            self.populate_video_table()

    def apply_profile(self) -> None:
        self.sync_visible_records()
        record = self.current_record()
        if not record or not record["profile"].is_file():
            QMessageBox.information(self, tr('일괄 적용'), tr('설정이 완료된 기준 Trial 행을 먼저 선택하세요.'))
            return
        applied = 0
        target_records = list({item["base_key"]: item for item in self.records}.values())
        checked_bases = {
            item["base_key"] for item in self.records if item["key"] in self.checked_keys
        }
        for target in target_records:
            if target["base_key"] in checked_bases and target["base_key"] != record["base_key"]:
                target["profile"].parent.mkdir(parents=True, exist_ok=True)
                payload = json.loads(record["profile"].read_text(encoding="utf-8"))
                payload["source_video"] = str(target["video"])
                target["profile"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                applied += 1
        self.populate_video_table()
        self.populate_arena_table()
        QMessageBox.information(self, tr('일괄 적용'), tr('{p0}개 Arena에 수조·플랫폼 설정을 적용했습니다.', p0=applied))

    def save_metadata(self, sync_table: bool = True) -> None:
        if sync_table:
            self.sync_visible_records()
        payload: dict[str, dict] = {}
        for record in self.records:
            if not record.get("_range_registered", True):
                continue
            entry = payload.setdefault(record["base_key"], {"segments": []})
            entry["segments"].append({
                key: record[key]
                for key in (
                    "segment_id", "subject_id", "group", "day", "trial", "trial_type",
                    "start_direction", "target_quadrant", "analysis_start_seconds",
                    "analysis_end_seconds",
                )
            })
        from hagisilta_atomic_io import atomic_json
        atomic_json(self.metadata_path, payload)

    def run_analysis(self) -> None:
        self.sync_visible_records()
        selected_records = [record for record in self.records if record["key"] in self.checked_keys]
        if any(not record.get("_range_registered", True) for record in selected_records):
            QMessageBox.information(self, tr('구간 등록 필요'), tr('선택한 Arena에 미등록 구간이 있습니다. 개체별 시작·종료를 등록하거나 해당 Arena의 사용 체크를 해제하세요.'))
            return
        self.save_metadata(sync_table=False)
        overlap_messages: list[str] = []
        for base_key in sorted({record["base_key"] for record in selected_records}):
            segments = sorted(
                (record for record in selected_records if record["base_key"] == base_key),
                key=lambda record: float(record["analysis_start_seconds"]),
            )
            for previous, current in zip(segments, segments[1:]):
                if float(current["analysis_start_seconds"]) < float(previous["analysis_end_seconds"]):
                    overlap_messages.append(
                        f"{base_key}: {previous['subject_id']} ↔ {current['subject_id']}"
                    )
        if overlap_messages:
            answer = QMessageBox.question(
                self,
                tr('Trial 구간 겹침'),
                tr('같은 Arena에서 분석 구간이 겹치는 Trial이 있습니다.\n\n')
                + "\n".join(overlap_messages[:8])
                + ("\n…" if len(overlap_messages) > 8 else "")
                + tr('\n\n겹친 시간은 두 개체 결과에 모두 포함됩니다. 그대로 분석할까요?'),
            )
            if answer != QMessageBox.Yes:
                return
        items = []
        for record in selected_records:
            if not record["profile"].is_file():
                QMessageBox.information(self, tr('수조 설정 필요'), tr('{p0}의 수조·플랫폼을 먼저 설정하세요.', p0=record['key']))
                return
            items.append({
                "video_bundle": record["bundle"].name, "arena_id": arena_number(record["arena"]),
                "arena_dir": str(record["arena"]), "video": str(record["video"]), "profile": str(record["profile"]),
                "segment_id": record["segment_id"], "subject_id": record["subject_id"],
                "group": record.get("group", ""), "day": record["day"], "trial": record["trial"],
                "trial_type": record["trial_type"], "start_direction": record["start_direction"],
                "target_quadrant": record["target_quadrant"],
                "analysis_start_seconds": record["analysis_start_seconds"],
                "analysis_end_seconds": record["analysis_end_seconds"],
            })
        if not items:
            QMessageBox.information(self, tr('분석 대상'), tr('분석할 Trial을 한 개 이상 체크하세요.')); return
        selected_bodyparts = self.selected_bodyparts()
        if not selected_bodyparts:
            QMessageBox.information(self, tr('판정 신체점'), tr('판정에 사용할 신체점을 한 개 이상 체크하세요.'))
            return
        plan = {
            "items": items, "bodyparts": selected_bodyparts,
            "pcutoff": self.pcutoff.value(), "platform_dwell_seconds": self.dwell.value(),
            "bridge_seconds": self.bridge.value(),
            "platform_crossing_dwell_seconds": self.crossing_dwell.value(),
            "platform_crossing_gap_seconds": self.crossing_gap.value(),
            "max_duration_seconds": self.max_duration.value(),
        }
        plan_path = project_state_root(self.project_root, create=True) / "runtime" / "watermaze-plan.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        output = project_output_root(self.project_root, create=True) / "watermaze"
        self.host.set_workflow_badge(self.analysis_badge, tr('{p0}개 Trial 분석 중', p0=len(items)), True)
        self.host.start_process(
            self.host.SCRIPTS / "hagisilta_watermaze_analysis.py" if hasattr(self.host, "SCRIPTS") else Path(__file__).with_name("hagisilta_watermaze_analysis.py"),
            [str(plan_path), str(output)], self.analysis_finished,
        )

    def analysis_finished(self, code: int) -> None:
        if code == 0:
            self.host.set_workflow_badge(self.analysis_badge, tr('분석 완료'), True)
            self.refresh_results()
        else:
            self.host.set_workflow_badge(self.analysis_badge, tr('분석 실패'), False)

    def latest_run(self) -> Path | None:
        marker = project_output_root(self.project_root) / "watermaze" / "latest-run.json"
        try:
            path = Path(json.loads(marker.read_text(encoding="utf-8"))["run_dir"])
            return path if path.is_dir() else None
        except Exception:
            return None

    def refresh_results(self, run: Path | None = None) -> None:
        run = run or self.latest_run()
        self.current_run = run
        self.result_table.setRowCount(0)
        if not run:
            self.result_run_label.setText(tr('표시 결과 없음'))
            self.result_run_label.setToolTip(tr('저장된 분석 결과가 없습니다.'))
            self.host.set_workflow_badge(self.result_badge, tr('결과 없음'), False)
            return
        summary = run / "project-watermaze-summary.csv"
        if not summary.is_file():
            self.result_run_label.setText(tr('요약 결과 없음'))
            self.result_run_label.setToolTip(str(run))
            self.host.set_workflow_badge(self.result_badge, tr('결과 없음'), False)
            return
        import pandas as pd
        try: frame = pd.read_csv(summary)
        except Exception:
            self.host.set_workflow_badge(self.result_badge, tr('결과 읽기 실패'), False)
            return
        sorting_was_enabled = self.result_table.isSortingEnabled()
        self.result_table.setSortingEnabled(False)
        schema = self.result_schema(frame)
        self.result_table.clear()
        self.result_table.setColumnCount(len(schema))
        self.result_table.setHorizontalHeaderLabels([column[0] for column in schema])
        for column, (_, _, tooltip) in enumerate(schema):
            self.result_table.horizontalHeaderItem(column).setToolTip(tooltip)
        self.result_table.setRowCount(len(frame))
        for row, (_, record) in enumerate(frame.iterrows()):
            record_payload = record.where(pd.notna(record), "").to_dict()
            for column, (_, key, _) in enumerate(schema):
                item = QTableWidgetItem(self.format_result_value(key, record.get(key, "")))
                item.setData(Qt.UserRole, str(record.get("visual_file", "")))
                item.setData(Qt.UserRole + 1, str(record.get("annotated_video_file", "")))
                item.setData(Qt.UserRole + 2, record_payload)
                if column > 0: item.setTextAlignment(Qt.AlignCenter)
                self.result_table.setItem(row, column, item)
        self.result_table.setSortingEnabled(sorting_was_enabled)
        self.result_run_label.setText(tr('표시 결과 · {p0}개 Trial', p0=len(frame)))
        self.result_run_label.setToolTip(str(run))
        self.host.set_workflow_badge(self.result_badge, tr('{p0}개 결과', p0=len(frame)), bool(len(frame)))
        if len(frame):
            self.result_table.selectRow(0)

    @staticmethod
    def result_schema(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
        common = [
            (tr('영상'), "video_bundle", tr('분석한 원본 영상 묶음입니다.')),
            ("Arena", "arena_id", tr('원본 영상 안에서 분리된 Arena 번호입니다.')),
            (tr('개체 ID'), "subject_id", tr('이 Trial에 지정한 동물 개체 ID입니다.')),
            (tr('실험군'), "group", tr('이 Trial에 지정한 실험군입니다.')),
            (tr('유형'), "trial_type", tr(TRIAL_TYPE_TOOLTIP)),
            ("Day", "day", tr('학습 또는 검사 일차입니다.')),
            ("Trial", "trial", tr('같은 Day 안에서의 Trial 번호입니다.')),
        ]
        types = {
            str(value).strip().lower()
            for value in frame.get("trial_type", pd.Series(dtype=str)).dropna()
            if str(value).strip()
        }
        metric_tips = {
            "found_platform": tr('설정한 최소 플랫폼 체류시간을 만족했는지 표시합니다.'),
            "escape_latency_seconds": tr('분석 구간 시작부터 최초 확정 플랫폼 도달까지 걸린 시간입니다.'),
            "path_length_cm": tr('유효한 추적 좌표를 따라 이동한 총거리입니다.'),
            "mean_speed_cm_s": tr('유효한 추적 구간에서 계산한 평균 이동 속도입니다.'),
            "path_efficiency": tr('시작점과 플랫폼 사이 직선거리 ÷ 실제 이동거리입니다. 1에 가까울수록 효율적입니다.'),
            "target_quadrant_percent": tr('전체 유효 분석시간 중 목표 사분면에 머문 비율입니다.'),
            "platform_crossings": tr('이전 또는 목표 플랫폼 영역을 확정 통과한 횟수입니다.'),
            "mean_proximity_cm": tr('분석 중 플랫폼 중심까지의 평균 거리입니다. 낮을수록 목표 위치에 가깝습니다.'),
            "thigmotaxis_percent": tr('수조 벽에서 설정한 폭 안에 머문 유효시간 비율입니다.'),
            "valid_pose_percent": tr('전체 분석 프레임 중 선택한 신체점 조합이 신뢰도 기준을 통과한 비율입니다.'),
        }
        definitions = {
            "found_platform": (tr('도달'), "found_platform"),
            "escape_latency_seconds": ("Latency(s)", "escape_latency_seconds"),
            "path_length_cm": (tr('거리(cm)'), "path_length_cm"),
            "mean_speed_cm_s": (tr('속도(cm/s)'), "mean_speed_cm_s"),
            "path_efficiency": (tr('효율'), "path_efficiency"),
            "target_quadrant_percent": ("Target(%)", "target_quadrant_percent"),
            "platform_crossings": (tr('통과(회)'), "platform_crossings"),
            "mean_proximity_cm": (tr('근접(cm)'), "mean_proximity_cm"),
            "thigmotaxis_percent": (tr('벽면(%)'), "thigmotaxis_percent"),
            "valid_pose_percent": (tr('유효좌표(%)'), "valid_pose_percent"),
        }
        by_type = {
            "probe": [
                "target_quadrant_percent", "platform_crossings", "mean_proximity_cm",
                "thigmotaxis_percent", "valid_pose_percent",
            ],
            "hidden": [
                "found_platform", "escape_latency_seconds", "path_length_cm",
                "path_efficiency", "thigmotaxis_percent",
            ],
            "visible": [
                "found_platform", "escape_latency_seconds", "path_length_cm",
                "mean_speed_cm_s", "valid_pose_percent",
            ],
            "reversal": [
                "found_platform", "escape_latency_seconds", "path_length_cm",
                "path_efficiency", "target_quadrant_percent",
            ],
        }
        if len(types) == 1:
            metric_keys = by_type.get(next(iter(types)), by_type["hidden"])
        else:
            metric_keys = [
                "found_platform", "escape_latency_seconds", "path_length_cm",
                "mean_speed_cm_s", "path_efficiency", "target_quadrant_percent",
                "platform_crossings", "mean_proximity_cm", "thigmotaxis_percent",
                "valid_pose_percent",
            ]
        return common + [
            (definitions[key][0], definitions[key][1], metric_tips[key])
            for key in metric_keys
        ]

    @staticmethod
    def format_result_value(key: str, value: object) -> str:
        if key == "found_platform":
            normalized = str(value).strip().lower()
            if normalized in {"", "nan", "none"}:
                return "-"
            return tr('도달') if normalized in {"1", "true", "yes"} else tr('미도달')
        if key in {"video_bundle", "subject_id", "group", "trial_type"}:
            text = str(value).strip()
            return "-" if text.lower() in {"", "nan", "none"} else text
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return "-"
        if key in {"arena_id", "day", "trial", "platform_crossings"}:
            return str(int(round(float(numeric))))
        return f"{float(numeric):.2f}"

    def open_visual(self) -> None:
        row = self.result_table.currentRow()
        if row < 0:
            QMessageBox.information(self, tr('궤적·히트맵 편집'), tr('결과 표에서 Trial을 선택하세요.'))
            return
        entries: list[dict] = []
        current_index = 0
        for source_row in range(self.result_table.rowCount()):
            item = self.result_table.item(source_row, 0)
            payload = dict(item.data(Qt.UserRole + 2) or {}) if item else {}
            entry = self.decision_viewer_entry(payload) if payload else None
            if not entry:
                continue
            if source_row == row:
                current_index = len(entries)
            entries.append(entry)
        if not entries:
            QMessageBox.warning(
                self, tr('궤적·히트맵 편집'),
                tr('원본 영상·프레임 판정 CSV·ROI 프로필을 찾을 수 없습니다.'),
            )
            return
        try:
            dialog = WaterMazeVisualComposer(entries, current_index, self)
        except Exception as error:
            QMessageBox.warning(self, tr('궤적·히트맵 편집 실패'), str(error))
            return
        dialog.exec()

    def open_result_csv(self) -> None:
        row = self.result_table.currentRow()
        item = self.result_table.item(row, 0) if row >= 0 else None
        payload = item.data(Qt.UserRole + 2) if item else None
        if not isinstance(payload, dict):
            QMessageBox.information(self, tr('CSV 열기'), tr('결과 표에서 Trial을 선택하세요.'))
            return
        path = Path(str(payload.get("per_frame_file", "")))
        if not path.is_file():
            QMessageBox.information(
                self, tr('CSV 열기'), tr('선택한 Trial의 프레임별 판정 CSV를 찾을 수 없습니다.')
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def decision_viewer_entry(self, payload: dict) -> dict | None:
        video = Path(str(payload.get("source_video_file", "")))
        if not video.is_file() and payload.get("pose_file"):
            video = arena_video(Path(str(payload["pose_file"])).parent) or Path()
        frames = Path(str(payload.get("per_frame_file", "")))
        profile = Path(str(payload.get("profile_file", "")))
        if not profile.is_file():
            try:
                profile = (
                    project_state_root(self.project_root) / "watermaze_profiles"
                    / f"{payload.get('video_bundle')}_arena_{int(float(payload.get('arena_id', 1))):02d}.json"
                )
            except (TypeError, ValueError):
                profile = Path()
        if not (video.is_file() and frames.is_file() and profile.is_file()):
            return None
        return {
            "video": str(video), "frames": str(frames), "profile": str(profile),
            "summary": dict(payload),
        }

    def open_annotated_video(self) -> None:
        row = self.result_table.currentRow()
        item = self.result_table.item(row, 0) if row >= 0 else None
        payload = item.data(Qt.UserRole + 2) if item else None
        selected_entry = self.decision_viewer_entry(payload) if isinstance(payload, dict) else None
        if selected_entry:
            playlist: list[dict] = []
            seen_frames: set[str] = set()
            for result_row in range(self.result_table.rowCount()):
                result_item = self.result_table.item(result_row, 0)
                result_payload = result_item.data(Qt.UserRole + 2) if result_item else None
                entry = (
                    self.decision_viewer_entry(result_payload)
                    if isinstance(result_payload, dict) else None
                )
                if entry and entry["frames"] not in seen_frames:
                    playlist.append(entry)
                    seen_frames.add(entry["frames"])
            if selected_entry["frames"] not in seen_frames:
                playlist.append(selected_entry)
            try:
                viewer = WaterMazeDecisionViewer(
                    Path(selected_entry["video"]), Path(selected_entry["frames"]),
                    Path(selected_entry["profile"]), selected_entry["summary"], self,
                    playlist=playlist,
                )
                viewer.exec()
                return
            except Exception as error:
                QMessageBox.warning(self, tr('판정 영상'), tr('대화형 판정창을 열 수 없습니다.\n{p0}', p0=error))
                return
        legacy = Path(str(item.data(Qt.UserRole + 1))) if item else Path()
        if legacy.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(legacy)))
        else:
            QMessageBox.information(
                self,
                tr('판정 영상'),
                tr('새 판정 영상이 없습니다. Water maze 분석을 다시 실행하면 자동으로 생성됩니다.'),
            )

    def open_output(self) -> None:
        run = getattr(self, "current_run", None)
        if run and run.is_dir(): QDesktopServices.openUrl(QUrl.fromLocalFile(str(run)))
        else: QMessageBox.information(self, tr('결과 폴더'), tr('아직 분석 결과가 없습니다.'))

    def load_history(self) -> None:
        base = project_output_root(self.project_root) / "watermaze" / "runs"
        selected = QFileDialog.getExistingDirectory(self, tr('이전 Water maze 실행 선택'), str(base))
        if selected: self.refresh_results(Path(selected))
