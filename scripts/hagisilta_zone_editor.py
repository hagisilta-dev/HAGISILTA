# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from hagisilta_i18n import init_i18n, tr

import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QBrush, QFont, QIcon, QPen, QPolygonF, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGraphicsEllipseItem,
    QGraphicsItem, QGraphicsPolygonItem, QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem, QHBoxLayout, QInputDialog, QLabel,
    QLayout, QMainWindow, QMenu, QMessageBox, QPushButton, QScrollArea, QSlider, QVBoxLayout, QWidget,
)

from hagisilta_nort_editor import CanvasView, EditableShapeItem, PolygonObjectItem, cv_to_pixmap, frame_at
from hagisilta_window_chrome import (
    HeaderContext,
    apply_hagisilta_window_identity, create_hagisilta_header,
    create_hagisilta_help_button, enable_full_window_controls,
    hagisilta_control_stylesheet, install_wheel_value_guard,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ICON = REPO_ROOT / "assets" / "hagisilta-app.ico"


COLORS = [
    QColor("#55d6a9"), QColor("#ffca5c"), QColor("#ff6b8a"), QColor("#66a8ff"),
    QColor("#ba8cff"), QColor("#66e0e8"), QColor("#ff9e64"), QColor("#a8df65"),
]


class DraggableLandmarkItem(QGraphicsEllipseItem):
    def __init__(
        self, key: str, point: list[float], label: str, color: QColor,
        moved_callback, delete_callback, radius: float = 4.0, show_label: bool = True,
    ) -> None:
        radius = max(2.0, float(radius))
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.key = key
        self.moved_callback = moved_callback
        self.delete_callback = delete_callback
        self.setPos(float(point[0]), float(point[1]))
        self.setPen(QPen(QColor("#07101a"), 2.5))
        self.setBrush(color)
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.OpenHandCursor)
        self.setZValue(95)
        text = QGraphicsSimpleTextItem(label, self)
        text.setBrush(color)
        text.setPos(radius + 3, -radius - 6)
        text.setVisible(show_label)
        text.setAcceptedMouseButtons(Qt.NoButton)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        super().mouseReleaseEvent(event)
        self.setCursor(Qt.OpenHandCursor)
        position = self.scenePos()
        self.moved_callback(self.key, [float(position.x()), float(position.y())])

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        menu = QMenu()
        delete_action = menu.addAction(tr('이 점 삭제'))
        selected = menu.exec(event.screenPos())
        if selected == delete_action:
            self.delete_callback(self.key)
        event.accept()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HAGISILTA generic polygon ROI and zone editor")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--arena-profile", type=Path, help=tr('원본 영상의 Arena ROI 프로필'))
    parser.add_argument("--roi-only", action="store_true")
    parser.add_argument("--preset", choices=("ymaze",))
    parser.add_argument("--arena-id", type=int)
    parser.add_argument("--coordinate-space", choices=("source_px", "arena_crop_px"), default="source_px")
    return parser.parse_args()


def empty_ymaze_landmarks() -> dict:
    return {
        "center_triangle": [None, None, None],
        "entry_lines": {
            "arm_a": [None, None], "arm_b": [None, None], "arm_c": [None, None],
        },
        "arm_a": None, "arm_b": None, "arm_c": None,
    }


class ZoneStudio(QMainWindow):
    def __init__(
        self, video: Path, output: Path, profile: Path | None = None,
        roi_only: bool = False, preset: str | None = None,
        arena_id: int | None = None, coordinate_space: str = "source_px",
        arena_profile: Path | None = None,
    ) -> None:
        super().__init__()
        self.video = video.resolve()
        self.output = output.resolve()
        if arena_profile is not None and self.output == arena_profile.resolve():
            raise ValueError(
                tr('실험 공간 설정 출력은 트래킹 ROI 파일과 분리되어야 합니다. 트래킹 ROI는 읽기 전용 초기 경계로만 사용할 수 있습니다.')
            )
        self.roi_only = roi_only
        self.preset = preset
        self.arena_id = arena_id
        self.coordinate_space = coordinate_space
        self.single_arena = preset == "ymaze" and arena_id is not None
        self.frame, self.metadata = frame_at(self.video, 0.0)
        self.current_seconds = min(self.metadata["duration_seconds"] * 0.5, self.metadata["duration_seconds"])
        self.frame, self.metadata = frame_at(self.video, self.current_seconds)
        self.regions: list[dict] = []
        self.current_region = 0
        self.draw_target: tuple[str, str | None] = ("mask", None)
        self.tool_mode = "select"
        self.pending_points: list[QPointF] = []
        self.pending_point_size = 4
        self.guide_items = []
        self.rectangle_preview_item: QGraphicsRectItem | None = None
        self.landmark_target: str | None = None
        self.landmark_size = 4
        self.show_landmarks = True
        self.ymaze_stage = 0
        self.ymaze_stage_widgets: dict[int, list[QWidget]] = {0: [], 1: [], 2: []}
        # Rebuilding the scene after every landmark must not refit the view.
        # Otherwise the image moves under the cursor between consecutive clicks.
        self._scene_fitted = False
        self.setWindowTitle(
            tr('HAGISILTA · ROI Editor · 추적 영역 편집')
            if self.roi_only
            else tr('HAGISILTA · Y-maze · A/B/C 구역 편집')
        )
        apply_hagisilta_window_identity(self)
        enable_full_window_controls(self)
        self.resize(1500, 930)
        self.setMinimumSize(1120, 720)
        self._build_ui()
        self._apply_style()
        if profile and profile.is_file():
            self.load_profile(profile)
        elif arena_profile and arena_profile.is_file() and self.preset == "ymaze" and not self.roi_only:
            self.load_arena_profile(arena_profile)
        else:
            self.add_region()
            if self.preset == "ymaze" and not self.roi_only:
                self.add_ymaze_zones(self.region)
                self.refresh_zone_combo()
        if self.preset == "ymaze" and not self.roi_only:
            self.normalize_simple_ymaze_regions()
        self.render_scene()
        self.bind_shortcuts()
        self.update_tool_buttons()
        if self.preset == "ymaze" and not self.roi_only:
            self.set_ymaze_stage(self.infer_ymaze_stage())
        if (
            self.preset == "ymaze" and not self.roi_only and self.regions
            and not self.polygon(self.region.get("maze_outline"))
        ):
            QTimer.singleShot(0, lambda: self.begin_draw("maze_outline", None))

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header, header_layout = create_hagisilta_header(
            tr('추적 영역 편집') if self.roi_only else tr('Y-maze A/B/C·Center 편집'),
            "HAGISILTA · ROI WORKSPACE",
        )
        header_layout.addWidget(create_hagisilta_help_button(
            self,
            tr('ROI 편집 도움말') if self.roi_only else tr('Y-maze 구역 편집 도움말'),
            tr('Ctrl+S  저장\nEnter  다각형 완성\nBackspace  마지막 점 취소\nDelete  선택 점·도형 삭제\nEsc  그리기 취소\nF  화면 맞춤\n')
            + (
                tr('1 / 2 / 3  선택·사각형·다각형 도구')
                if self.roi_only else
                tr('Alt+1·2·3  작업 단계 이동\nO / C / G  Maze 외곽 / Center / A·B·C 자동 분할\nT / H / J  트래킹 ROI / Maze 외곽 / 구역 채움 표시')
            ),
        ))
        self.status = HeaderContext(header, tr('편집 중'))
        outer.addWidget(header)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        outer.addLayout(body, 1)

        self.ymaze_stage_buttons = []
        if self.preset == "ymaze" and not self.roi_only:
            sidebar = QFrame(objectName="sidebar")
            sidebar.setFixedWidth(184)
            side_layout = QVBoxLayout(sidebar)
            side_layout.setContentsMargins(18, 24, 18, 24)
            side_layout.setSpacing(8)
            side_layout.addWidget(QLabel("WORKFLOW", objectName="workflowTitle"))
            for index, text in enumerate((tr('01  Maze 외곽'), "02  Center", tr('03  결과 검토'))):
                button = QPushButton(text, objectName="workflowButton")
                button.setCheckable(True)
                button.setToolTip(tr('{p0} 단계로 이동 · Alt+{p1}', p0=text[4:], p1=index + 1))
                button.clicked.connect(lambda _checked=False, stage=index: self.set_ymaze_stage(stage))
                side_layout.addWidget(button)
                self.ymaze_stage_buttons.append(button)
            side_layout.addStretch()
            body.addWidget(sidebar)

        workspace = QFrame(objectName="workspace")
        workspace_layout = QHBoxLayout(workspace)
        workspace_layout.setContentsMargins(18, 18, 18, 18)
        workspace_layout.setSpacing(16)
        body.addWidget(workspace, 1)

        self.scene = QGraphicsScene(self)
        self.view = CanvasView(self.scene)
        self.view.canvas_clicked.connect(self.canvas_clicked)
        self.view.canvas_dragged.connect(self.canvas_dragged)
        self.view.canvas_drag_updated.connect(self.canvas_drag_updated)
        canvas_frame = QFrame(objectName="canvasFrame")
        canvas_layout = QVBoxLayout(canvas_frame)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.addWidget(QLabel(self.video.name, objectName="canvasTitle"))
        canvas_layout.addWidget(self.view, 1)
        workspace_layout.addWidget(canvas_frame, 1)

        panel = QFrame(objectName="panel")
        panel.setMinimumWidth(380)
        controls = QVBoxLayout(panel)
        controls.setContentsMargins(18, 18, 18, 18)
        controls.setSpacing(10)
        controls.setSizeConstraint(QLayout.SetMinimumSize)
        controls.addWidget(QLabel(tr('기준 프레임'), objectName="toolGroupTitle"))
        self.time_label = QLabel(objectName="timeReadout")
        controls.addWidget(self.time_label)
        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setRange(0, max(1, round(self.metadata["duration_seconds"] * 1000)))
        self.timeline.setValue(round(self.current_seconds * 1000))
        self._pending_seek_ms = int(self.timeline.value())
        self._seek_timer = QTimer(self)
        self._seek_timer.setSingleShot(True)
        self._seek_timer.setInterval(40)
        self._seek_timer.timeout.connect(self.apply_queued_seek)
        self.timeline.sliderMoved.connect(self.queue_seek_frame)
        self.timeline.sliderReleased.connect(self.seek_frame)
        controls.addWidget(self.timeline)

        seek_row = QHBoxLayout()
        self.seek_buttons: list[QPushButton] = []
        for text, seconds, frames in (
            (tr('-1초'), -1.0, 0),
            ("‹ 1F", 0.0, -1),
            ("1F ›", 0.0, 1),
            (tr('+1초'), 1.0, 0),
        ):
            button = QPushButton(text, objectName="secondary")
            button.clicked.connect(
                lambda _checked=False, sec=seconds, frame_delta=frames:
                    self.seek_delta(sec, frame_delta)
            )
            seek_row.addWidget(button)
            self.seek_buttons.append(button)
        controls.addLayout(seek_row)

        controls.addWidget(QLabel(
            (f"Arena {self.arena_id:02d}" if self.single_arena else tr('Y-maze 선택'))
            if self.preset == "ymaze" and not self.roi_only else tr('추적 영역'),
            objectName="toolGroupTitle",
        ))
        region_row = QHBoxLayout()
        self.region_combo = QComboBox()
        self.region_combo.currentIndexChanged.connect(self.region_changed)
        region_row.addWidget(self.region_combo, 1)
        add_region = QPushButton(tr('다음 미로 ＋') if self.preset == "ymaze" and not self.roi_only else "＋")
        add_region.setToolTip(tr('다음 Y-maze를 추가하거나 이동합니다.') if self.preset == "ymaze" and not self.roi_only else tr('화면 속 독립된 개체/미로/실험 영역을 추가합니다.'))
        add_region.clicked.connect(self.add_next_ymaze if self.preset == "ymaze" and not self.roi_only else self.add_region)
        remove_region = QPushButton("－")
        remove_region.clicked.connect(self.remove_region)
        region_row.addWidget(add_region)
        region_row.addWidget(remove_region)
        controls.addLayout(region_row)
        add_region.setVisible(not self.single_arena)
        remove_region.setVisible(not self.single_arena)
        self.add_region_button = add_region
        self.remove_region_button = remove_region
        self.region_count_label = QLabel("", objectName="regionCount")
        controls.addWidget(self.region_count_label)
        if self.preset == "ymaze" and not self.roi_only:
            self.ymaze_progress_label = QLabel(
                tr('Y-maze 1 · 0/3 완료'),
                objectName="ymazeProgress",
            )
            self.ymaze_progress_label.setWordWrap(False)
            controls.addWidget(self.ymaze_progress_label)
        point_size_row = QHBoxLayout()
        self.pending_point_size_label = QLabel(tr('입력점 크기  {p0}px', p0=self.pending_point_size))
        self.pending_point_size_label.setToolTip(tr('ROI·외곽·Center를 그릴 때 클릭한 꼭짓점의 화면 표시 크기입니다. 좌표에는 영향을 주지 않습니다.'))
        point_size_row.addWidget(self.pending_point_size_label)
        self.pending_point_size_slider = QSlider(Qt.Horizontal)
        self.pending_point_size_slider.setRange(2, 12)
        self.pending_point_size_slider.setValue(self.pending_point_size)
        self.pending_point_size_slider.setToolTip(tr('ROI 입력 꼭짓점 크기 · 2~12 px'))
        self.pending_point_size_slider.valueChanged.connect(self.set_pending_point_size)
        point_size_row.addWidget(self.pending_point_size_slider, 1)
        controls.addLayout(point_size_row)
        if self.roi_only:
            controls.addWidget(QLabel(tr('ROI 도형 도구'), objectName="section"))
            tool_row = QHBoxLayout()
            select_tool = QPushButton(tr('선택·이동 (1)'), objectName="toolButton")
            select_tool.setToolTip(tr('완성된 ROI 내부를 드래그해 이동하고, 테두리 핸들로 크기를 조절합니다.'))
            select_tool.clicked.connect(self.select_tool)
            rectangle_tool = QPushButton(tr('사각형 (2)'), objectName="toolButton")
            rectangle_tool.setToolTip(tr('영상 위에서 좌상단부터 우하단까지 드래그해 사각형 ROI를 만듭니다.'))
            rectangle_tool.clicked.connect(self.begin_rectangle)
            polygon_tool = QPushButton(tr('다각형 (3)'), objectName="toolButton")
            polygon_tool.setToolTip(tr('영상 위를 여러 번 클릭해 꼭짓점을 추가하고 Enter로 완성합니다.'))
            polygon_tool.clicked.connect(lambda: self.begin_draw("mask", None))
            self.tool_buttons = {
                "select": select_tool, "rectangle": rectangle_tool, "polygon": polygon_tool,
            }
            for button in self.tool_buttons.values():
                button.setCheckable(True)
            tool_row.addWidget(select_tool); tool_row.addWidget(rectangle_tool); tool_row.addWidget(polygon_tool)
            controls.addLayout(tool_row)
            clear_mask = QPushButton(tr('현재 ROI 도형 삭제'))
            clear_mask.clicked.connect(self.delete_current_mask)
            controls.addWidget(clear_mask)
        else:
            maze_tools = QHBoxLayout()
            maze_section = None
            if self.preset == "ymaze":
                maze_section = QLabel(tr('Maze 외곽 ROI'), objectName="toolGroupTitle")
                controls.addWidget(maze_section)
            select_maze = QPushButton(tr('편집 (V)'), objectName="shapeToolButton")
            select_maze.setToolTip(tr('완성된 외곽 내부를 드래그해 이동하고 흰 핸들로 꼭짓점을 조절합니다.'))
            select_maze.clicked.connect(self.select_tool)
            mask = QPushButton(
                tr('다각형 (O)') if self.preset == "ymaze" else tr('추적 마스크 새로 그리기'),
                objectName="shapeToolButton" if self.preset == "ymaze" else "primary",
            )
            mask.setToolTip(
                tr('잠긴 트래킹 ROI 안에서 검은 Y-maze 장치의 실제 외곽을 따라 다각형으로 그립니다.')
                if self.preset == "ymaze" else
                tr('다른 동물과 반사를 제외할 실제 추적 영역을 자유 다각형으로 지정합니다.')
            )
            mask.clicked.connect(
                (lambda: self.begin_draw("maze_outline", None))
                if self.preset == "ymaze" else (lambda: self.begin_draw("mask", None))
            )
            self.ymaze_mask_button = mask if self.preset == "ymaze" else None
            if self.preset == "ymaze":
                maze_tools.addWidget(select_maze, 1)
                maze_tools.addWidget(mask, 1)
                controls.addLayout(maze_tools)
                delete_outline = QPushButton(tr('Maze 외곽 삭제'), objectName="dangerButton")
                delete_outline.clicked.connect(self.delete_ymaze_outline)
                controls.addWidget(delete_outline)
            else:
                controls.addWidget(mask)
            if self.preset == "ymaze":
                self.ymaze_stage_widgets[0].extend(
                    [maze_section, select_maze, mask, delete_outline]
                )

        if not self.roi_only:
            zone_section = QLabel("Center ROI" if self.preset == "ymaze" else tr('행동 구역'), objectName="toolGroupTitle")
            controls.addWidget(zone_section)
            self.zone_combo = QComboBox()
            self.zone_combo.currentIndexChanged.connect(self.zone_changed)
            self.role_combo = QComboBox()
            self.role_combo.addItem(tr('선택 구역 · 진입/교대 계산'), "choice")
            self.role_combo.addItem(tr('중앙/전이 구역'), "center")
            self.role_combo.addItem(tr('기타 관찰 구역'), "other")
            self.role_combo.currentIndexChanged.connect(self.role_changed)
            if self.preset == "ymaze":
                self.zone_combo.setVisible(False)
                self.role_combo.setVisible(False)
                self.ymaze_zone_buttons = {}
                center_button = QPushButton(tr('다각형 (C)'), objectName="shapeToolButton")
                center_button.setCheckable(True)
                center_button.setToolTip(tr('세 Arm이 만나는 중앙 영역을 자유 다각형으로 그리고 Enter를 누릅니다.'))
                center_button.clicked.connect(self.begin_center_geometry)
                center_tools = QHBoxLayout()
                select_center = QPushButton(tr('편집 (V)'), objectName="shapeToolButton")
                select_center.setToolTip(tr('완성된 Center 내부를 드래그해 이동하고 흰 핸들로 꼭짓점을 조절합니다.'))
                select_center.clicked.connect(self.select_tool)
                center_tools.addWidget(select_center, 1)
                center_tools.addWidget(center_button, 1)
                controls.addLayout(center_tools)
                delete_center = QPushButton(tr('Center 삭제'), objectName="dangerButton")
                delete_center.clicked.connect(self.delete_ymaze_center)
                controls.addWidget(delete_center)
                self.ymaze_zone_buttons["center_geometry"] = center_button
                self.ymaze_stage_widgets[1].extend(
                    [zone_section, select_center, center_button, delete_center]
                )

                result_section = QLabel(tr('A/B/C 결과'), objectName="toolGroupTitle")
                controls.addWidget(result_section)
                auto_generate = QPushButton(tr('A/B/C 자동 분할 (G)'), objectName="secondaryButton")
                auto_generate.setToolTip(tr('Maze 외곽에서 Center를 제외한 세 Arm을 자동으로 나눕니다. 별도의 Arm 기준점은 필요 없습니다.'))
                auto_generate.clicked.connect(self.generate_ymaze_zones)
                controls.addWidget(auto_generate)
                self.ymaze_generate_button = auto_generate
                self.ymaze_stage_widgets[2].extend([result_section, auto_generate])

                rotate_labels = QPushButton(tr('A/B/C 이름 한 칸 회전 (R)'))
                rotate_labels.setToolTip(tr('영역 모양은 그대로 두고 Arm 이름만 A→B→C 순서로 한 칸 바꿉니다.'))
                rotate_labels.clicked.connect(self.rotate_ymaze_labels)
                controls.addWidget(rotate_labels)
                self.ymaze_rotate_button = rotate_labels
                self.ymaze_stage_widgets[2].append(rotate_labels)
                next_maze = QPushButton(tr('현재 미로 완료 → 다음 Y-maze'), objectName="primary")
                next_maze.clicked.connect(self.add_next_ymaze)
                controls.addWidget(next_maze)
                next_maze.setVisible(not self.single_arena)
                self.ymaze_stage_widgets[2].append(next_maze)
                self.next_ymaze_button = next_maze
            else:
                zone_row = QHBoxLayout()
                zone_row.addWidget(self.zone_combo, 1)
                add_zone = QPushButton(tr('구역 추가'))
                add_zone.clicked.connect(self.add_zone)
                delete_zone = QPushButton(tr('삭제'))
                delete_zone.clicked.connect(self.delete_zone)
                zone_row.addWidget(add_zone)
                zone_row.addWidget(delete_zone)
                controls.addLayout(zone_row)
                role_row = QHBoxLayout()
                role_row.addWidget(QLabel(tr('구역 역할')))
                role_row.addWidget(self.role_combo, 1)
                controls.addLayout(role_row)
                draw_zone = QPushButton(tr('선택 구역 새로 그리기'), objectName="secondaryButton")
                draw_zone.clicked.connect(self.begin_selected_zone)
                controls.addWidget(draw_zone)

        if self.preset == "ymaze" and not self.roi_only:
            controls.addWidget(QLabel(tr('참조 경계 표시'), objectName="toolGroupTitle"))
            self.overlay_toggles = {}
            for key, label in (("arena", tr('트래킹 ROI 표시 (T)')), ("outline", tr('Maze 외곽 표시 (H)')),
                               ("fill", tr('구역 채움 표시 (J)'))):
                toggle = QCheckBox(label)
                toggle.setChecked(True)
                toggle.toggled.connect(self.update_overlay_visibility)
                self.overlay_toggles[key] = toggle
                controls.addWidget(toggle)
        drawing_title = QLabel(tr('사용 방법') if self.roi_only else tr('다각형 그리기'), objectName="toolGroupTitle")
        controls.addWidget(drawing_title)
        self.draw_help = QLabel(
            ((tr('실제 Maze 외곽과 Center 다각형을 그리면 A/B/C가 자동으로 나뉩니다.\n') if self.preset == "ymaze" else "")
             if self.preset == "ymaze" and not self.roi_only else "") +
            (tr('사각형: 영상 위에서 드래그해 생성합니다.\n') if self.roi_only else "") +
            tr('왼쪽 클릭으로 꼭짓점 추가 · Enter로 완성\n완성 후 내부 드래그로 이동 · 테두리 핸들로 변형'),
            objectName="help",
        )
        self.draw_help.setWordWrap(True)
        controls.addWidget(self.draw_help)
        finish = QPushButton(tr('다각형 완성 (Enter)'), objectName="secondaryButton")
        finish.clicked.connect(self.finish_polygon)
        self.finish_polygon_button = finish
        controls.addWidget(finish)
        summary = QLabel(
            tr('ROI마다 한 마리씩 독립 추론됩니다. 서로 겹치는 외접 사각형 대신 실제 실험 장치의 모양을 따라 그리면 다른 동물을 제거할 수 있습니다.')
            if self.roi_only else
            ((tr('Arena {p0:02d}의 잘린 화면에서 A/B/C·Center를 지정합니다.\n같은 Arena 번호의 다른 영상에 이 템플릿이 재사용됩니다.', p0=self.arena_id)
              if self.single_arena else tr('Y-maze 1을 모두 지정한 뒤 ‘다음 Y-maze’를 누르세요.\nA/B/C 방향은 임의지만 모든 영상에서 같은 기준을 유지해야 합니다.'))
             if self.preset == "ymaze" else
             tr('범용 사용 예\nY/T-maze: 중앙 + 각 팔\nOpen field: 중앙 + 주변부\nEPM: 중앙 + open/closed arm\n임의 구역: 관심 영역을 원하는 수만큼 추가')),
            objectName="help",
        )
        summary.setWordWrap(True)
        self.editor_summary = summary
        controls.addWidget(summary)
        if self.preset == "ymaze" and not self.roi_only:
            drawing_title.hide()
            self.draw_help.hide()
            self.editor_summary.hide()
        controls.addStretch()
        panel_scroll = QScrollArea()
        panel_scroll.setObjectName("panelScroll")
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setFrameShape(QFrame.NoFrame)
        panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        panel_scroll.setWidget(panel)
        panel_scroll.setFixedWidth(380)
        workspace_layout.addWidget(panel_scroll)

        footer = QFrame(objectName="detachedFooter")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(18, 9, 18, 9)
        self.footer_context = QLabel(tr('Y-maze 설정 · 1/3') if self.preset == "ymaze" and not self.roi_only else tr('ROI 편집'))
        footer_layout.addWidget(self.footer_context)
        footer_layout.addStretch()
        self.footer_back = QPushButton(tr('← 이전'), footer, objectName="secondaryButton")
        self.footer_back.clicked.connect(self.previous_workflow_stage)
        self.footer_back.setVisible(not self.roi_only)
        self.footer_next = QPushButton(tr('저장하고 닫기') if self.roi_only else tr('다음 →'), objectName="primaryButton")
        self.footer_next.clicked.connect(self.next_workflow_stage)
        footer_layout.addWidget(self.footer_back)
        footer_layout.addWidget(self.footer_next)
        outer.addWidget(footer)

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QWidget { background:#0d1420; color:#dfe8f5; font-family:'Segoe UI','Malgun Gothic'; font-size:13px; }
            #header { background:#111b2a; border:none; border-bottom:1px solid #223149; }
            #logo { color:#69e0ad; font-size:18px; font-weight:800; letter-spacing:2px; }
            #subtitle { color:#718198; margin-left:8px; }
            #status { color:#70e2ae; background:#183529; border:1px solid #28513f; border-radius:12px; padding:5px 12px; }
            #modeBadge { color:#9dc4ff; background:#172b46; border:1px solid #31547c; border-radius:12px; padding:6px 12px; }
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
            #regionCount { color:#8fa5bd; background:#0c1521; border:1px solid #26394f; border-radius:5px; padding:7px 9px; }
            #ymazeProgress { color:#bcd0e7; background:#13243a; border:1px solid #345578; border-radius:6px; padding:10px; font-weight:700; }
            QPushButton#ymazeStepButton { text-align:left; padding:10px 12px; background:#17263a; border:1px solid #344e6b; }
            QPushButton#ymazeStepButton:hover { border-color:#63d5a6; background:#1b3045; }
            QPushButton#ymazeStepButton:checked { background:#315f52; color:#ffffff; border:2px solid #63d5a6; font-weight:800; }
            QPushButton#workflowButton {
                min-height:44px; text-align:left; padding-left:12px; color:#c2c5ca;
                background:transparent; border:1px solid transparent; border-radius:3px;
            }
            QPushButton#workflowButton:hover { background:#172437; color:#ffffff; }
            QPushButton#workflowButton:checked {
                background:#174c3d; color:#79e7b5; font-weight:650;
                border:1px solid #28745c;
            }
            #panel { background:#101925; border:none; border-left:1px solid #223149; }
            QScrollArea#panelScroll { background:transparent; border:none; }
            QScrollArea#panelScroll > QWidget > QWidget { background:#101925; }
            #section { background:#0c1420; color:#f1f6fc; font-size:15px; font-weight:800; padding:8px; }
            #help { color:#91a3b9; background:#0d1622; border:1px solid #27384d; border-radius:6px; padding:10px; }
            QPushButton { min-height:36px; background:#1a2a3d; border:1px solid #344b66; border-radius:3px; padding:0 13px; font-weight:600; }
            QPushButton:hover { border-color:#63d5a6; }
            QPushButton#primary { background:#4d67d7; color:white; border-color:#607aed; font-weight:750; }
            QPushButton#primary:checked { background:#315f52; border:2px solid #63d5a6; }
            QPushButton#primaryButton { background:#4c63cf; color:#ffffff; border:1px solid #6079d5; }
            QPushButton#primaryButton:hover { background:#5872df; border-color:#8297e0; }
            QPushButton#secondaryButton { background:#18263a; color:#dfe8f5; border:1px solid #304765; }
            QPushButton#secondaryButton:hover { background:#21334c; border-color:#45658d; }
            QPushButton#toolButton:checked { background:#174c3d; color:#79e7b5; border:1px solid #28745c; font-weight:800; }
            QComboBox { background:#101b2a; border:1px solid #344b66; border-radius:5px; padding:8px; }
            QSlider::groove:horizontal { height:6px; background:#26364b; border-radius:3px; }
            QSlider::handle:horizontal { width:16px; margin:-5px 0; background:#62d9aa; border-radius:8px; }
        """)
        # Discard the legacy navy editor theme above.  Detached workspaces must
        # start from the same neutral HAGISILTA surface instead of layering two
        # competing themes.
        self.setStyleSheet("")
        self.setStyleSheet("""
            QWidget {
                background:#23242b; color:#f0f1f2;
                font-family:'Segoe UI','Malgun Gothic'; font-size:13px;
            }
            QMainWindow { background:#23242b; }
            #header {
                background:#202125; border:none; border-bottom:1px solid #41434a;
            }
            #pageTitleAccent { background:#5b82e8; border:none; }
            #pageKicker {
                background:transparent; color:#7f98d8; font-size:9px;
                font-weight:700; letter-spacing:1px;
            }
            #pageWorkbenchTitle {
                background:transparent; color:#f2f3f5; font-size:18px; font-weight:650;
            }
            #stepBadge, #stepBadgeReady {
                border-radius:2px; padding:4px 9px; font-size:11px; font-weight:600;
            }
            #stepBadge { background:#292a2f; color:#c5c7cb; border:1px solid #4b4e55; }
            #stepBadgeReady { background:#263630; color:#8cd5b8; border:1px solid #3b5c50; }
            #sidebar, #panel, #footer {
                background:#27282d; border-color:#41434a;
            }
            #workspace, #canvasFrame { background:#202125; }
            QGraphicsView { background:#101114; border:1px solid #3f4147; }
            #canvasTitle { background:transparent; color:#f0f1f2; font-size:13px; font-weight:650; }
            QScrollArea#panelScroll { background:transparent; border:none; }
            QScrollArea#panelScroll > QWidget > QWidget { background:#27282d; }
            #section {
                background:#303136; color:#f0f1f2; border:1px solid #41434a;
                border-radius:2px; font-size:13px; font-weight:650; padding:7px 9px;
            }
            #toolGroupTitle {
                background:transparent; color:#f1f3f6; border:none;
                font-size:14px; font-weight:800; padding:4px 1px;
            }
            #help, #regionCount {
                color:#b5b8bd; background:#232429; border:1px solid #3f4147;
                border-radius:2px; padding:8px 9px;
            }
            #ymazeProgress {
                color:#d6d8dc; background:#2b2c31; border:1px solid #4b4e55;
                border-radius:2px; padding:8px; font-weight:650;
            }
            QPushButton {
                min-height:34px; padding:0 12px; border-radius:2px;
                background:#35363b; color:#eceef1; border:1px solid #55585f;
                font-weight:600;
            }
            QPushButton:hover { background:#42444a; border-color:#70747d; }
            QPushButton#primary, QPushButton#primaryButton {
                background:#3f6dcc; color:#ffffff; border:1px solid #5c82d8;
            }
            QPushButton#primary:hover, QPushButton#primaryButton:hover {
                background:#4b79da; border-color:#7d9ce2;
            }
            QPushButton#secondaryButton { background:#35363b; border:1px solid #55585f; }
            QPushButton#secondary { background:#35363b; border:1px solid #55585f; padding:0 5px; }
            QPushButton#shapeToolButton {
                background:#35363b; border:1px solid #55585f;
                padding:0 4px; font-size:11px;
            }
            QPushButton#shapeToolButton:checked {
                background:#3f6dcc; border-color:#6f91df; color:#ffffff;
            }
            QPushButton#toolButton {
                min-height:28px; max-height:28px; padding:0 9px;
                background:#303136; color:#d6d8dc; border:1px solid #4b4e55;
                border-radius:2px; font-size:11px; font-weight:550;
            }
            QPushButton#toolButton:hover { background:#41434a; border-color:#686c75; }
            QPushButton#toolButton:checked,
            QPushButton#primary:checked,
            QPushButton#ymazeStepButton:checked {
                background:#3b65c4; color:#ffffff; border:1px solid #718ee7;
            }
            QPushButton#workflowButton { border-radius:2px; }
            QPushButton#workflowButton:hover { background:#3a404b; }
            QPushButton#workflowButton:checked {
                background:#3b65c4; color:#ffffff; border:1px solid #718ee7;
            }
            QComboBox {
                background:#1f2024; color:#f0f1f2; border:1px solid #50535a;
                border-radius:2px; padding:7px 9px;
            }
            QSlider::groove:horizontal { height:5px; background:#3b3d43; border-radius:2px; }
            QSlider::handle:horizontal {
                width:15px; margin:-5px 0; background:#5b82e8;
                border:1px solid #89a2e7; border-radius:7px;
            }
            QToolTip { background:#303136; color:#f0f1f2; border:1px solid #62656d; padding:6px; }
        """)
        self.setStyleSheet(self.styleSheet() + hagisilta_control_stylesheet())

    def show_shortcuts(self) -> None:
        QMessageBox.information(
            self,
            tr('ROI 편집 단축키'),
            tr('Ctrl+S  저장\nEnter  다각형 완성\nBackspace  마지막 점 취소\nDelete  선택 점·도형 삭제\nEsc  그리기 취소\nF  화면 맞춤\n')
            + (
                tr('1 / 2 / 3  선택·사각형·다각형 도구')
                if self.roi_only else
                tr('Alt+1·2·3  작업 단계 이동\nO / C / G  Maze 외곽 / Center / A·B·C 자동 분할')
            ),
        )

    def bind_shortcuts(self) -> None:
        shortcuts = [
            ("Ctrl+S", self.save_profile), ("Return", self.finish_polygon),
            ("Enter", self.finish_polygon), ("Backspace", self.delete_selected_landmark_or_undo),
            ("Delete", self.delete_selected_landmark_or_undo),
            ("Escape", self.cancel_draw), ("F", self.fit_canvas),
        ]
        if self.roi_only:
            shortcuts += [
                ("1", self.select_tool), ("2", self.begin_rectangle),
                ("3", lambda: self.begin_draw("mask", None)),
            ]
        elif self.preset == "ymaze":
            shortcuts += ([
                ("T", lambda: self.overlay_toggles["arena"].toggle()),
                ("H", lambda: self.overlay_toggles["outline"].toggle()),
                ("J", lambda: self.overlay_toggles["fill"].toggle()),
                ("V", self.select_tool),
                ("O", lambda: self.begin_draw("maze_outline", None)),
                ("C", self.begin_center_geometry),
                ("G", self.generate_ymaze_zones),
                ("R", self.rotate_ymaze_labels),
                ("Alt+1", lambda: self.set_ymaze_stage(0)),
                ("Alt+2", lambda: self.set_ymaze_stage(1)),
                ("Alt+3", lambda: self.set_ymaze_stage(2)),
            ])
        for sequence, callback in shortcuts:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(callback)

    def add_region(self) -> None:
        if self.roi_only and len(self.regions) == 1 and not self.polygon(self.regions[0].get("mask")):
            self.current_region = 0
            self.renumber_regions()
            self.refresh_combos()
            if hasattr(self, "scene"):
                self.render_scene()
            return
        number = self.arena_id if self.single_arena else len(self.regions) + 1
        name = f"ROI {number}" if self.roi_only else (
            f"Arena {number:02d}" if self.single_arena else tr('추적 영역 {p0}', p0=number)
        )
        self.regions.append({
            "id": number, "name": name, "mask": None, "maze_outline": None, "zones": [],
            "mask_locked": False,
            "center_mode": "free_polygon",
            "arm_label_rotation": 0,
            "landmarks": empty_ymaze_landmarks(),
        })
        if self.preset == "ymaze" and not self.roi_only:
            self.add_ymaze_zones(self.regions[-1])
        self.current_region = len(self.regions) - 1
        self.renumber_regions()
        self.refresh_combos()
        if hasattr(self, "scene"):
            self.render_scene()

    def add_next_ymaze(self) -> None:
        if self.single_arena:
            QMessageBox.information(self, tr('Arena 전용 편집'), tr('현재 창은 선택한 Arena 하나만 편집합니다.'))
            return
        if self.preset != "ymaze" or self.roi_only:
            self.add_region()
            return
        if self.current_region + 1 < len(self.regions):
            self.region_combo.setCurrentIndex(self.current_region + 1)
        else:
            self.add_region()
        self.begin_draw("maze_outline", None)

    @staticmethod
    def add_ymaze_zones(region: dict) -> None:
        if region.get("zones"):
            return
        if not str(region.get("name", "")).startswith("Arena "):
            region["name"] = f"Y-maze {int(region.get('id', 1))}"
        region["zones"] = [
            {"id": "arm_a", "name": "Arm A", "role": "choice", "polygon": None},
            {"id": "arm_b", "name": "Arm B", "role": "choice", "polygon": None},
            {"id": "arm_c", "name": "Arm C", "role": "choice", "polygon": None},
            {"id": "center", "name": "Center", "role": "center", "polygon": None},
        ]

    def normalize_simple_ymaze_regions(self) -> None:
        """Migrate older point-heavy Y-maze profiles to the simple polygon workflow."""
        for region in self.regions:
            self.add_ymaze_zones(region)
            region["name"] = f"Arena {int(region.get('id', 1)):02d}"
            # These masks come from the tracking ROI profile and define the DLC
            # coordinate system. They are reference overlays, never editable here.
            if self.polygon(region.get("mask")):
                region["mask_locked"] = True
            center_zone = next((zone for zone in region.get("zones", []) if zone.get("id") == "center"), None)
            old_triangle = region.get("landmarks", {}).get("center_triangle", [])
            if center_zone is not None and not self.polygon(center_zone.get("polygon")) and len(old_triangle) == 3 and all(old_triangle):
                center_zone["polygon"] = [[float(x), float(y)] for x, y in old_triangle]
            region["center_mode"] = "free_polygon"
            region["arm_label_rotation"] = int(region.get("arm_label_rotation", 0)) % 3
            region["landmarks"] = empty_ymaze_landmarks()
            self.snap_ymaze_arm_center_edges(region)
        if hasattr(self, "region_combo"):
            self.refresh_combos()

    def remove_region(self) -> None:
        if self.region.get("mask_locked"):
            QMessageBox.information(
                self, tr('Arena ROI 잠금'),
                tr('이 Arena는 트래킹 좌표의 기준이므로 여기서 삭제할 수 없습니다. ROI 변경은 트래킹 단계에서 다시 지정하세요.'),
            )
            return
        if len(self.regions) <= 1:
            if not self.roi_only:
                QMessageBox.information(self, tr('추적 영역'), tr('추적 영역은 하나 이상 필요합니다.'))
                return
            self.regions[0] = {
                "id": 1, "name": "ROI 1", "mask": None, "maze_outline": None, "zones": [],
                "mask_locked": False, "center_mode": "free_polygon", "arm_label_rotation": 0,
                "landmarks": empty_ymaze_landmarks(),
            }
            self.current_region = 0
            self.refresh_combos()
            self.render_scene()
            self.status.setText(tr('● 모든 ROI를 지웠습니다 · 새 ROI는 ROI 1부터 시작합니다'))
            return
        self.regions.pop(self.current_region)
        self.current_region = min(self.current_region, len(self.regions) - 1)
        self.renumber_regions()
        self.refresh_combos(); self.render_scene()

    def renumber_regions(self) -> None:
        """Keep ROI identifiers and visible names contiguous after add/delete/load."""
        for index, region in enumerate(self.regions, start=1):
            region["id"] = index
            if self.roi_only:
                region["name"] = f"ROI {index}"

    def add_zone(self) -> None:
        name, ok = QInputDialog.getText(self, tr('행동 구역 추가'), tr('구역 이름:'), text=tr('구역 {p0}', p0=len(self.region['zones']) + 1))
        if not ok or not name.strip():
            return
        zone = {"id": f"zone_{len(self.region['zones']) + 1}", "name": name.strip(), "role": "choice", "polygon": None}
        self.region["zones"].append(zone)
        self.refresh_zone_combo(len(self.region["zones"]) - 1)
        self.begin_draw("zone", zone["id"])

    def delete_zone(self) -> None:
        index = self.zone_combo.currentIndex()
        if index < 0:
            return
        self.region["zones"].pop(index)
        self.refresh_zone_combo(); self.render_scene()

    def delete_ymaze_outline(self) -> None:
        if self.preset != "ymaze" or self.roi_only:
            return
        self.cancel_draw()
        self.region["maze_outline"] = None
        for zone in self.region.get("zones", []):
            if zone.get("id") in {"arm_a", "arm_b", "arm_c"}:
                zone["polygon"] = None
        self.set_ymaze_stage(0)
        self.render_scene()
        self.update_ymaze_steps()
        self.status.setText(tr('Maze 외곽을 삭제했습니다 · 다시 그리세요'))

    def delete_ymaze_center(self) -> None:
        if self.preset != "ymaze" or self.roi_only:
            return
        self.cancel_draw()
        for zone in self.region.get("zones", []):
            if zone.get("id") in {"center", "arm_a", "arm_b", "arm_c"}:
                zone["polygon"] = None
        self.set_ymaze_stage(1 if self.polygon(self.region.get("maze_outline")) else 0)
        self.render_scene()
        self.update_ymaze_steps()
        self.status.setText(tr('Center를 삭제했습니다 · 다시 그리세요'))

    @property
    def region(self) -> dict:
        return self.regions[self.current_region]

    def refresh_combos(self) -> None:
        self.region_combo.blockSignals(True)
        self.region_combo.clear()
        for region in self.regions:
            self.region_combo.addItem(region["name"])
        self.region_combo.setCurrentIndex(self.current_region)
        self.region_combo.blockSignals(False)
        self.update_region_count()
        if not self.roi_only:
            self.refresh_zone_combo()
        self.update_ymaze_steps()

    def refresh_zone_combo(self, selected: int = 0) -> None:
        self.zone_combo.blockSignals(True)
        self.zone_combo.clear()
        for zone in self.region["zones"]:
            self.zone_combo.addItem(zone["name"], zone["id"])
        if self.region["zones"]:
            self.zone_combo.setCurrentIndex(min(selected, len(self.region["zones"]) - 1))
        self.zone_combo.blockSignals(False)
        self.zone_changed()

    def update_region_count(self) -> None:
        if not hasattr(self, "region_count_label") or not self.regions:
            return
        completed = sum(
            bool(self.polygon(region.get("maze_outline" if self.preset == "ymaze" and not self.roi_only else "mask")))
            for region in self.regions
        )
        self.region_count_label.setText(
            tr('전체 {p0}개 · 완성 {p1}개 · 현재 {p2}/{p3}', p0=len(self.regions), p1=completed, p2=self.current_region + 1, p3=len(self.regions))
        )
        self.update_ymaze_steps()

    def update_mask_lock_controls(self) -> None:
        locked = bool(self.regions and self.region.get("mask_locked"))
        if getattr(self, "ymaze_mask_button", None):
            self.ymaze_mask_button.setEnabled(True)
            self.ymaze_mask_button.setToolTip(
                tr('점선 트래킹 ROI는 잠긴 상태로 유지됩니다. 그 안에서 실제 Y-maze 장치 외곽을 별도로 그립니다.')
                if locked else tr('Y-maze 장치의 실제 외곽을 자유 다각형으로 지정합니다.')
            )
        all_locked = bool(self.regions and all(region.get("mask_locked") for region in self.regions))
        if hasattr(self, "add_region_button"):
            self.add_region_button.setEnabled(not all_locked)
        if hasattr(self, "remove_region_button"):
            self.remove_region_button.setEnabled(not locked)

    def infer_ymaze_stage(self) -> int:
        if self.preset != "ymaze" or self.roi_only or not self.regions:
            return 0
        if not self.polygon(self.region.get("maze_outline")):
            return 0
        zones = {zone["id"]: zone for zone in self.region.get("zones", [])}
        if not self.polygon((zones.get("center") or {}).get("polygon")):
            return 1
        return 2

    def set_ymaze_stage(self, stage: int) -> None:
        if self.preset != "ymaze" or self.roi_only:
            return
        requested = max(0, min(int(stage), 2))
        available = self.infer_ymaze_stage()
        if requested > available:
            requested = available
            self.status.setText(tr('● 현재 단계를 먼저 완료하세요'))
        previous = self.ymaze_stage
        self.ymaze_stage = requested
        for index, widgets in self.ymaze_stage_widgets.items():
            for widget in widgets:
                widget.setVisible(index == requested)
        if hasattr(self, "next_ymaze_button"):
            self.next_ymaze_button.setVisible(requested == 2 and not self.single_arena)
        for index, button in enumerate(getattr(self, "ymaze_stage_buttons", [])):
            button.setChecked(index == requested)
            button.setEnabled(index <= available)
        if hasattr(self, "footer_context"):
            labels = (tr('Maze 외곽'), "Center", tr('결과 검토'))
            self.footer_context.setText(f"{labels[requested]} · {requested + 1}/3")
            self.footer_back.setEnabled(requested > 0)
            self.footer_next.setText(tr('저장하고 닫기') if requested == 2 else tr('다음 →'))
        if hasattr(self, "draw_help"):
            messages = (
                tr('실제 Maze 외곽을 따라 꼭짓점을 찍고 Enter를 누르세요. 완성 후에는 이동·변형할 수 있습니다.'),
                tr('세 Arm이 만나는 Center 영역을 자유 다각형으로 그리고 Enter를 누르세요. 외곽선은 잠겨 클릭을 통과합니다.'),
                tr('외곽과 Center는 잠겨 있습니다. 자동 분할 결과를 확인하고 A/B/C 방향만 다르면 R을 누르세요.'),
            )
            self.draw_help.setText(messages[requested])
            self.draw_help.setVisible(False)
        if hasattr(self, "finish_polygon_button"):
            self.finish_polygon_button.setVisible(requested < 2)
        if hasattr(self, "editor_summary"):
            self.editor_summary.setVisible(False)
        if previous != requested and hasattr(self, "scene"):
            self.render_scene()

    def previous_workflow_stage(self) -> None:
        if self.preset == "ymaze" and not self.roi_only:
            self.set_ymaze_stage(self.ymaze_stage - 1)

    def next_workflow_stage(self) -> None:
        if self.preset == "ymaze" and not self.roi_only:
            if self.ymaze_stage >= 2:
                self.save_profile(close_after=True)
            else:
                self.set_ymaze_stage(self.ymaze_stage + 1)
        else:
            self.save_profile(close_after=True)

    def update_ymaze_steps(self) -> None:
        if self.preset != "ymaze" or self.roi_only or not self.regions:
            return
        outline_done = bool(self.polygon(self.region.get("maze_outline")))
        zones = {zone["id"]: zone for zone in self.region.get("zones", [])}
        center_done = bool(self.polygon((zones.get("center") or {}).get("polygon")))
        arms_done = all(bool(self.polygon((zones.get(key) or {}).get("polygon"))) for key in ("arm_a", "arm_b", "arm_c"))
        completed = int(outline_done) + int(center_done) + int(arms_done)
        if getattr(self, "ymaze_mask_button", None):
            self.ymaze_mask_button.setText(
                (tr('다각형 다시 그리기') if outline_done else tr('다각형')) + " (O)"
            )
            self.ymaze_mask_button.setCheckable(True)
            self.ymaze_mask_button.setChecked(
                self.tool_mode == "polygon" and self.draw_target == ("maze_outline", None)
            )
        self.update_mask_lock_controls()
        center_button = getattr(self, "ymaze_zone_buttons", {}).get("center_geometry")
        if center_button:
            center_button.setChecked(self.tool_mode == "polygon" and self.draw_target == ("zone", "center"))
            center_button.setText(
                (tr('다각형 다시 그리기') if center_done else tr('다각형')) + " (C)"
            )
            center_button.setEnabled(outline_done)
        if hasattr(self, "ymaze_generate_button"):
            self.ymaze_generate_button.setEnabled(outline_done and center_done)
            self.ymaze_generate_button.setText(
                (tr('A/B/C 다시 계산') if arms_done else tr('A/B/C 자동 분할')) + " (G)"
            )
        if hasattr(self, "ymaze_rotate_button"):
            self.ymaze_rotate_button.setEnabled(arms_done)
        if hasattr(self, "ymaze_progress_label"):
            subject = f"Arena {self.arena_id:02d}" if self.single_arena else f"Y-maze {self.current_region + 1}"
            self.ymaze_progress_label.setText(
                tr('{p0} · {p1}/3 완료', p0=subject, p1=completed)
            )
        available = self.infer_ymaze_stage()
        for index, button in enumerate(getattr(self, "ymaze_stage_buttons", [])):
            button.setEnabled(index <= available)
            button.setChecked(index == self.ymaze_stage)

    def set_landmark_visibility(self, visible: bool) -> None:
        self.show_landmarks = bool(visible)
        self.render_scene()

    def set_landmark_size(self, value: int) -> None:
        self.landmark_size = max(2, min(int(value), 10))
        self.render_scene()

    def set_pending_point_size(self, value: int) -> None:
        self.pending_point_size = max(2, min(int(value), 12))
        if hasattr(self, "pending_point_size_label"):
            self.pending_point_size_label.setText(tr('입력점 크기  {p0}px', p0=self.pending_point_size))
        if hasattr(self, "scene"):
            self.render_pending()

    def center_mode_changed(self, *_args) -> None:
        if self.preset != "ymaze" or not self.regions or not hasattr(self, "center_mode_combo"):
            return
        new_mode = str(self.center_mode_combo.currentData() or "triangle")
        old_mode = str(self.region.get("center_mode", "triangle"))
        if new_mode == old_mode:
            return
        self.cancel_draw()
        self.region["center_mode"] = new_mode
        self.region["landmarks"]["center_triangle"] = [None, None, None]
        self.region["landmarks"]["entry_lines"] = {
            "arm_a": [None, None], "arm_b": [None, None], "arm_c": [None, None],
        }
        self.clear_generated_ymaze_zones(include_center=True)
        self.render_scene()
        self.update_ymaze_steps()
        self.status.setText(tr('● Center 지정 방식 변경됨 · 2번부터 다시 지정하세요'))

    def begin_center_geometry(self) -> None:
        if self.preset != "ymaze":
            return
        if not self.polygon(self.region.get("maze_outline")):
            QMessageBox.information(self, tr('Maze 외곽 필요'), tr('먼저 1번에서 실제 Y-maze 외곽을 그리세요.'))
            return
        self.clear_generated_ymaze_zones(include_center=True)
        self.region["center_mode"] = "free_polygon"
        self.begin_draw("zone", "center")
        self.status.setText(tr('● Center 영역을 자유 다각형으로 그리고 Enter를 누르세요'))

    def region_changed(self, index: int) -> None:
        if index < 0 or index >= len(self.regions):
            return
        self.cancel_draw()
        self.current_region = index
        if not self.roi_only:
            self.refresh_zone_combo()
        if self.preset == "ymaze" and hasattr(self, "center_mode_combo"):
            mode_index = self.center_mode_combo.findData(self.region.get("center_mode", "triangle"))
            self.center_mode_combo.blockSignals(True)
            self.center_mode_combo.setCurrentIndex(max(mode_index, 0))
            self.center_mode_combo.blockSignals(False)
        self.render_scene()
        self.update_mask_lock_controls()
        if self.preset == "ymaze" and not self.roi_only:
            self.set_ymaze_stage(self.infer_ymaze_stage())

    def zone_changed(self, *_args) -> None:
        index = self.zone_combo.currentIndex()
        enabled = index >= 0
        self.role_combo.setEnabled(enabled)
        if enabled:
            role = self.region["zones"][index].get("role", "choice")
            role_index = self.role_combo.findData(role)
            self.role_combo.blockSignals(True)
            self.role_combo.setCurrentIndex(max(role_index, 0))
            self.role_combo.blockSignals(False)
        self.update_ymaze_steps()

    def role_changed(self, *_args) -> None:
        index = self.zone_combo.currentIndex()
        if index >= 0:
            self.region["zones"][index]["role"] = self.role_combo.currentData()

    def begin_selected_zone(self) -> None:
        zone_id = self.zone_combo.currentData()
        if not zone_id:
            QMessageBox.information(self, tr('행동 구역'), tr('먼저 구역을 추가하세요.'))
            return
        self.begin_draw("zone", str(zone_id))

    def begin_ymaze_landmark(self, landmark_id: str) -> None:
        if self.preset != "ymaze":
            return
        if not self.polygon(self.region.get("maze_outline")):
            QMessageBox.information(self, tr('미로 외곽 필요'), tr('먼저 1번에서 Y-maze 전체 외곽을 다각형으로 그리세요.'))
            return
        labels = {
            "center_triangle": tr('이전 Center 기준점'), "arm_a": tr('Arm A 내부점'),
            "arm_b": tr('Arm B 내부점'), "arm_c": tr('Arm C 내부점'), "entry_lines": tr('Arm 입구선 끝점'),
        }
        if landmark_id == "center_triangle":
            triangle = self.region.setdefault("landmarks", empty_ymaze_landmarks()).setdefault(
                "center_triangle", [None, None, None]
            )
            if all(triangle):
                self.region["landmarks"]["center_triangle"] = [None, None, None]
                self.clear_generated_ymaze_zones()
                self.render_scene()
        elif landmark_id == "entry_lines":
            lines = self.region.setdefault("landmarks", empty_ymaze_landmarks()).setdefault(
                "entry_lines", {"arm_a": [None, None], "arm_b": [None, None], "arm_c": [None, None]}
            )
            if all(all(lines.get(arm, [])) for arm in ("arm_a", "arm_b", "arm_c")):
                self.region["landmarks"]["entry_lines"] = {
                    "arm_a": [None, None], "arm_b": [None, None], "arm_c": [None, None],
                }
                self.clear_generated_ymaze_zones()
                self.render_scene()
        self.cancel_draw()
        self.tool_mode = "landmark"
        self.landmark_target = landmark_id
        self.view.click_through_items = True
        self.view.setCursor(Qt.CrossCursor)
        if landmark_id == "center_triangle":
            count = sum(bool(point) for point in self.region["landmarks"]["center_triangle"])
            self.status.setText(tr('● 이전 Center 기준점 {p0}/3 클릭 대기', p0=count + 1))
        elif landmark_id == "entry_lines":
            lines = self.region["landmarks"]["entry_lines"]
            count = sum(bool(point) for arm in ("arm_a", "arm_b", "arm_c") for point in lines[arm])
            arm = ("A", "B", "C")[min(count // 2, 2)]
            endpoint = count % 2 + 1
            self.status.setText(tr('● Arm {p0} 입구선 끝점 {p1}/2 클릭 대기 · 총 {p2}/6', p0=arm, p1=endpoint, p2=count))
        else:
            self.status.setText(tr('● {p0} 클릭 대기', p0=labels[landmark_id]))
        self.update_ymaze_steps()

    def landmark_moved(self, landmark_id: str, point: list[float]) -> None:
        width, height = self.frame.shape[1], self.frame.shape[0]
        fixed = [
            min(max(float(point[0]), 0.0), float(width - 1)),
            min(max(float(point[1]), 0.0), float(height - 1)),
        ]
        landmarks = self.region.setdefault("landmarks", empty_ymaze_landmarks())
        if landmark_id.startswith("center_") and landmark_id.rsplit("_", 1)[-1].isdigit():
            index = int(landmark_id.rsplit("_", 1)[-1])
            landmarks.setdefault("center_triangle", [None, None, None])[index] = fixed
        elif landmark_id.startswith("line_"):
            _prefix, arm, index_text = landmark_id.rsplit("_", 2)
            if arm in ("a", "b", "c") and index_text.isdigit():
                landmarks.setdefault("entry_lines", {}).setdefault(f"arm_{arm}", [None, None])[int(index_text)] = fixed
        else:
            landmarks[landmark_id] = fixed
        self.status.setText(tr('● 기준점 이동 완료 · A/B/C·Center 다시 계산 중'))
        QTimer.singleShot(0, self.generate_ymaze_zones)

    def clear_generated_ymaze_zones(self, include_center: bool = False) -> None:
        for zone in self.region.get("zones", []):
            if zone.get("id") == "center" and self.region.get("center_mode") == "free_polygon" and not include_center:
                continue
            zone["polygon"] = None

    def delete_ymaze_landmark(self, landmark_id: str) -> None:
        if self.preset != "ymaze":
            return
        landmarks = self.region.get("landmarks", {})
        if landmark_id.startswith("center_") and landmark_id.rsplit("_", 1)[-1].isdigit():
            index = int(landmark_id.rsplit("_", 1)[-1])
            triangle = landmarks.get("center_triangle", [None, None, None])
            if not (0 <= index < len(triangle)):
                return
            triangle[index] = None
        elif landmark_id.startswith("line_"):
            _prefix, arm, index_text = landmark_id.rsplit("_", 2)
            lines = landmarks.get("entry_lines", {})
            key = f"arm_{arm}"
            if key not in lines or not index_text.isdigit():
                return
            lines[key][int(index_text)] = None
        elif landmark_id in landmarks:
            landmarks[landmark_id] = None
        else:
            return
        self.clear_generated_ymaze_zones()
        self.render_scene()
        self.update_ymaze_steps()
        labels = {"arm_a": "Arm A", "arm_b": "Arm B", "arm_c": "Arm C"}
        if landmark_id.startswith("center_"):
            labels[landmark_id] = tr('Center 꼭짓점')
        elif landmark_id.startswith("line_"):
            labels[landmark_id] = tr('Arm 입구선 끝점')
        self.status.setText(tr('● {p0} 기준점 삭제됨 · 해당 점을 다시 찍으세요', p0=labels.get(landmark_id, landmark_id)))

    def delete_selected_landmark_or_undo(self) -> None:
        if self.preset == "ymaze" and hasattr(self, "scene"):
            selected = next(
                (item for item in self.scene.selectedItems() if isinstance(item, DraggableLandmarkItem)),
                None,
            )
            if selected is not None:
                self.delete_ymaze_landmark(selected.key)
                return
        self.undo_point()

    def reset_ymaze_landmarks(self) -> None:
        if self.preset != "ymaze":
            return
        answer = QMessageBox.question(
            self, tr('기준점 전체 초기화'),
            tr('Center 다각형과 이전 Arm 기준점 및 자동 생성 영역을 모두 지울까요?\nMaze 외곽은 유지됩니다.'),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self.region["landmarks"] = empty_ymaze_landmarks()
        self.clear_generated_ymaze_zones(include_center=True)
        self.render_scene()
        self.update_ymaze_steps()
        self.status.setText(tr('● 기준점 초기화 완료 · Center 경계부터 다시 지정하세요'))

    @staticmethod
    def largest_contour_polygon(binary: np.ndarray) -> list[list[float]]:
        contours, _hierarchy = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        contour = max(contours, key=cv2.contourArea)
        epsilon = max(1.0, cv2.arcLength(contour, True) * 0.008)
        simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        return [[float(x), float(y)] for x, y in simplified] if len(simplified) >= 3 else []

    @staticmethod
    def line_intersection(line_1, line_2) -> list[float] | None:
        (x1, y1), (x2, y2) = line_1
        (x3, y3), (x4, y4) = line_2
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denominator) < 1e-6:
            return None
        det1 = x1 * y2 - y1 * x2
        det2 = x3 * y4 - y3 * x4
        px = (det1 * (x3 - x4) - (x1 - x2) * det2) / denominator
        py = (det1 * (y3 - y4) - (y1 - y2) * det2) / denominator
        return [float(px), float(py)]

    def center_polygon_points(self) -> tuple[list[list[float]], str | None]:
        mode = str(self.region.get("center_mode", "triangle"))
        landmarks = self.region.get("landmarks", {})
        if mode == "free_polygon":
            center_zone = next((zone for zone in self.region.get("zones", []) if zone.get("id") == "center"), None)
            points = self.polygon((center_zone or {}).get("polygon"))
            return (points, None) if len(points) >= 3 else ([], tr('Center 자유 다각형'))
        if mode == "entry_lines":
            lines = landmarks.get("entry_lines", {})
            values = [lines.get(arm, [None, None]) for arm in ("arm_a", "arm_b", "arm_c")]
            if any(len(line) != 2 or not all(line) for line in values):
                return [], tr('Arm 입구선 3개')
            # 각 입구선을 무한히 연장했을 때 생기는 세 교점을 Center 경계로 사용합니다.
            points = [
                self.line_intersection(values[0], values[1]),
                self.line_intersection(values[1], values[2]),
                self.line_intersection(values[2], values[0]),
            ]
            if not all(points):
                return [], tr('서로 평행하지 않은 Arm 입구선')
            width, height = self.frame.shape[1], self.frame.shape[0]
            if any(not (-width <= point[0] <= width * 2 and -height <= point[1] <= height * 2) for point in points):
                return [], tr('교점이 화면에서 너무 멀지 않은 입구선')
            if abs(cv2.contourArea(np.asarray(points, dtype=np.float32))) < 10:
                return [], tr('면적이 생기는 Arm 입구선')
            return points, None
        triangle = landmarks.get("center_triangle", [None, None, None])
        return (triangle, None) if len(triangle) == 3 and all(triangle) else ([], tr('Center 꼭짓점 3개'))

    @staticmethod
    def clip_polygon_outside_center_edge(
        polygon: list[list[float]], start: np.ndarray, end: np.ndarray, center: np.ndarray,
    ) -> list[list[float]]:
        """Center의 한 변을 그대로 공유하도록 외곽 다각형을 Arm 방향으로 자릅니다."""
        points = [np.asarray(point, dtype=np.float64) for point in polygon]
        if len(points) < 3:
            return []
        edge = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
        center_cross = float(np.cross(edge, np.asarray(center, dtype=np.float64) - start))
        side = 1.0 if center_cross >= 0 else -1.0

        def value(point: np.ndarray) -> float:
            return side * float(np.cross(edge, point - start))

        clipped: list[np.ndarray] = []
        previous = points[-1]
        previous_value = value(previous)
        previous_inside = previous_value <= 1e-6
        for current in points:
            current_value = value(current)
            current_inside = current_value <= 1e-6
            if current_inside != previous_inside:
                denominator = previous_value - current_value
                if abs(denominator) > 1e-12:
                    clipped.append(previous + (previous_value / denominator) * (current - previous))
            if current_inside:
                clipped.append(current)
            previous, previous_value, previous_inside = current, current_value, current_inside

        cleaned: list[list[float]] = []
        for point in clipped:
            pair = [float(point[0]), float(point[1])]
            if not cleaned or np.hypot(pair[0] - cleaned[-1][0], pair[1] - cleaned[-1][1]) > 1e-5:
                cleaned.append(pair)
        if len(cleaned) > 1 and np.hypot(cleaned[0][0] - cleaned[-1][0], cleaned[0][1] - cleaned[-1][1]) <= 1e-5:
            cleaned.pop()
        return cleaned

    @staticmethod
    def polygon_centroid(points: list[list[float]]) -> tuple[float, float]:
        array = np.asarray(points, dtype=np.float32)
        moments = cv2.moments(array)
        if abs(moments["m00"]) > 1e-6:
            return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])
        return float(array[:, 0].mean()), float(array[:, 1].mean())

    @staticmethod
    def replace_contour_center_edge(
        polygon: list[list[float]], start: list[float] | np.ndarray, end: list[float] | np.ndarray,
    ) -> list[list[float]]:
        """Replace the rasterized inner Arm edge with the exact Center edge."""
        if len(polygon) < 3:
            return polygon
        points = np.asarray(polygon, dtype=np.float64)
        start_array = np.asarray(start, dtype=np.float64)
        end_array = np.asarray(end, dtype=np.float64)
        start_index = int(np.argmin(np.linalg.norm(points - start_array, axis=1)))
        end_index = int(np.argmin(np.linalg.norm(points - end_array, axis=1)))
        if start_index == end_index:
            return polygon

        def cyclic_path(first: int, last: int) -> list[int]:
            result = [first]
            while result[-1] != last and len(result) <= len(points):
                result.append((result[-1] + 1) % len(points))
            return result

        forward = cyclic_path(start_index, end_index)
        reverse = cyclic_path(end_index, start_index)

        def path_length(indices: list[int]) -> float:
            return sum(float(np.linalg.norm(points[b] - points[a])) for a, b in zip(indices, indices[1:]))

        # The direct Center boundary is shorter than the route around the Arm.
        # Keep the long exterior route and close it with the exact Center edge.
        exterior = reverse if path_length(forward) <= path_length(reverse) else forward
        result = [[float(points[index, 0]), float(points[index, 1])] for index in exterior]
        if exterior[0] == end_index:
            result[0], result[-1] = end_array.tolist(), start_array.tolist()
        else:
            result[0], result[-1] = start_array.tolist(), end_array.tolist()
        return result

    @classmethod
    def snap_ymaze_arm_center_edges(cls, region: dict) -> bool:
        """Make every Arm share the exact same stored edge as Center.

        Older profiles and newly rasterized Arm contours can differ by a pixel at
        the junction.  Matching each Arm direction to one Center edge and
        replacing that contour segment removes both the visual gap and the
        analysis overlap/gap.
        """
        zones = {str(zone.get("id")): zone for zone in region.get("zones", [])}
        center_points = cls.polygon((zones.get("center") or {}).get("polygon"))
        arm_ids = ("arm_a", "arm_b", "arm_c")
        arm_points = {
            arm_id: cls.polygon((zones.get(arm_id) or {}).get("polygon"))
            for arm_id in arm_ids
        }
        outline = cls.polygon(region.get("maze_outline"))
        if len(outline) >= 3 and len(center_points) >= 3 and all(len(arm_points[key]) >= 3 for key in arm_ids):
            try:
                center, arms = cls.partition_ymaze(outline, center_points)
            except ValueError:
                return False
            # Preserve A/B/C labels while replacing the raster-derived boundaries.
            old_centers = [np.asarray(cls.polygon_centroid(arm_points[key])) for key in arm_ids]
            centers = [np.asarray(cls.polygon_centroid(points)) for points in arms]
            assignment = min(itertools.permutations(range(3)), key=lambda order:
                             sum(np.linalg.norm(old_centers[i] - centers[j]) for i, j in enumerate(order)))
            zones["center"]["polygon"] = center
            for key, index in zip(arm_ids, assignment):
                zones[key]["polygon"] = arms[index]
            return True
        if len(center_points) != 3 or any(len(arm_points[arm_id]) < 3 for arm_id in arm_ids):
            return False

        center_array = np.asarray(center_points, dtype=np.float64)
        center_xy = center_array.mean(axis=0)
        edges = [
            (center_array[index], center_array[(index + 1) % 3])
            for index in range(3)
        ]

        def direction(point: np.ndarray) -> np.ndarray:
            vector = point - center_xy
            length = float(np.linalg.norm(vector))
            return vector / length if length > 1e-9 else vector

        edge_directions = [direction((start + end) * 0.5) for start, end in edges]
        arm_directions = [
            direction(np.asarray(cls.polygon_centroid(arm_points[arm_id]), dtype=np.float64))
            for arm_id in arm_ids
        ]
        assignment = max(
            itertools.permutations(range(3)),
            key=lambda permutation: sum(
                float(np.dot(arm_directions[index], edge_directions[edge_index]))
                for index, edge_index in enumerate(permutation)
            ),
        )
        for arm_id, edge_index in zip(arm_ids, assignment):
            start, end = edges[edge_index]
            zones[arm_id]["polygon"] = cls.replace_contour_center_edge(
                arm_points[arm_id], start, end,
            )
        return True

    @staticmethod
    def partition_ymaze(outline, center):
        """Exact vector difference: no raster rounding or independent simplification."""
        from shapely.geometry import Polygon
        outer, middle = Polygon(outline), Polygon(center)
        if not outer.is_valid or not middle.is_valid or outer.area <= 0 or middle.area <= 0:
            raise ValueError(tr('Maze 외곽과 Center는 교차하지 않는 다각형으로 그리세요.'))
        middle = outer.intersection(middle)
        if middle.geom_type != "Polygon" or middle.is_empty or middle.interiors:
            raise ValueError(tr('Maze 안의 Center가 하나의 영역이 되도록 그리세요.'))
        remainder = outer.difference(middle)
        arms = list(remainder.geoms) if remainder.geom_type == "MultiPolygon" else [remainder]
        if len(arms) != 3 or any(part.geom_type != "Polygon" or part.area <= 0 or part.interiors for part in arms):
            raise ValueError(tr('Center가 세 Arm 입구를 모두 가로질러 Maze가 정확히 세 부분으로 나뉘도록 그리세요.'))
        points = lambda polygon: [[float(x), float(y)] for x, y in list(polygon.exterior.coords)[:-1]]
        return points(middle), [points(arm) for arm in arms]

    def generate_ymaze_zones(self) -> None:
        if self.preset != "ymaze":
            return
        outline = self.polygon(self.region.get("maze_outline"))
        center, error = self.center_polygon_points()
        if len(outline) < 3 or error:
            QMessageBox.information(self, tr('자동 생성 준비'), tr('Maze 외곽과 Center 다각형을 먼저 그리세요.'))
            return
        try:
            center, arms = self.partition_ymaze(outline, center)
        except ValueError as error:
            QMessageBox.warning(self, tr('자동 분할 확인'), str(error))
            return
        origin = np.asarray(self.polygon_centroid(center))
        arms.sort(key=lambda points: float(np.arctan2(*(np.asarray(self.polygon_centroid(points)) - origin)[::-1])))
        rotation = int(self.region.get("arm_label_rotation", 0)) % 3
        arms = arms[rotation:] + arms[:rotation]
        polygons = dict(zip(("arm_a", "arm_b", "arm_c"), arms))
        polygons["center"] = center
        for zone in self.region.get("zones", []):
            if zone.get("id") in polygons:
                zone["polygon"] = polygons[zone["id"]]
        self.render_scene()
        self.update_ymaze_steps()
        self.status.setText(tr('A/B/C 자동 분할 완료 · Center와 같은 경계 좌표 사용'))

    def rotate_ymaze_labels(self) -> None:
        if self.preset != "ymaze" or not self.regions:
            return
        zones = {zone["id"]: zone for zone in self.region.get("zones", [])}
        if not all(self.polygon((zones.get(key) or {}).get("polygon")) for key in ("arm_a", "arm_b", "arm_c")):
            QMessageBox.information(self, tr('A/B/C 이름'), tr('먼저 3번 A/B/C 자동 분할을 실행하세요.'))
            return
        self.region["arm_label_rotation"] = (int(self.region.get("arm_label_rotation", 0)) + 1) % 3
        self.generate_ymaze_zones()

    def begin_ymaze_zone(self, zone_id: str) -> None:
        if self.preset != "ymaze" or self.roi_only:
            return
        index = next(
            (index for index, zone in enumerate(self.region.get("zones", [])) if zone.get("id") == zone_id),
            -1,
        )
        if index < 0:
            return
        self.zone_combo.setCurrentIndex(index)
        self.begin_draw("zone", zone_id)
        zone_name = self.region["zones"][index]["name"]
        self.status.setText(tr('● {p0} 꼭짓점 입력 중 · Enter로 완성', p0=zone_name))

    def begin_draw(self, kind: str, zone_id: str | None) -> None:
        if kind == "mask" and self.region.get("mask_locked"):
            QMessageBox.information(
                self, tr('Arena ROI 잠금'),
                tr('이 외곽은 실제 트래킹 영상을 자를 때 사용한 ROI입니다.\n여기서 바꾸면 DLC 좌표와 맞지 않으므로 잠겨 있습니다. ROI 변경은 트래킹 단계에서 다시 지정하세요.'),
            )
            return
        self.cancel_draw()
        self.tool_mode = "polygon"
        self.draw_target = (kind, zone_id)
        self.status.setText(tr('● 꼭짓점 입력 중 · Enter로 완성'))
        self.view.creation_enabled = False
        self.view.setCursor(Qt.CrossCursor)
        self.update_tool_buttons()
        self.update_ymaze_steps()

    def select_tool(self) -> None:
        self.cancel_draw()
        self.tool_mode = "select"
        self.view.creation_enabled = False
        self.status.setText(tr('● 선택·이동 · 내부 드래그 / 핸들 크기 조절'))
        self.update_tool_buttons()

    def begin_rectangle(self) -> None:
        self.cancel_draw()
        self.tool_mode = "rectangle"
        self.draw_target = ("mask", None)
        self.view.creation_enabled = True
        self.view.setCursor(Qt.CrossCursor)
        self.status.setText(tr('● 사각형 입력 · 영상 위에서 드래그'))
        self.update_tool_buttons()

    def update_tool_buttons(self) -> None:
        for mode, button in getattr(self, "tool_buttons", {}).items():
            button.setChecked(mode == self.tool_mode)

    def canvas_dragged(self, start: QPointF, end: QPointF) -> None:
        self.clear_rectangle_preview()
        if self.tool_mode != "rectangle":
            return
        rect = QRectF(start, end).normalized().intersected(
            QRectF(0, 0, self.frame.shape[1], self.frame.shape[0])
        )
        if rect.width() < 8 or rect.height() < 8:
            return
        points = [rect.topLeft(), rect.topRight(), rect.bottomRight(), rect.bottomLeft()]
        self.region["mask"] = PolygonObjectItem(points, COLORS[0], "ROI")
        self.tool_mode = "select"
        self.view.creation_enabled = False
        self.view.unsetCursor()
        self.status.setText(tr('● ROI 완성 · 내부 이동 / 핸들 크기 조절'))
        self.update_tool_buttons()
        self.render_scene()
        mask = self.region.get("mask")
        if mask is not None and not isinstance(mask, list):
            mask.setSelected(True)

    def canvas_drag_updated(self, start: QPointF, end: QPointF) -> None:
        if self.tool_mode != "rectangle" or self.frame is None:
            self.clear_rectangle_preview()
            return
        rect = QRectF(start, end).normalized().intersected(
            QRectF(0, 0, self.frame.shape[1], self.frame.shape[0])
        )
        if self.rectangle_preview_item is None:
            preview = QGraphicsRectItem()
            pen = QPen(QColor("#7da2ff"), 2.0, Qt.DashLine)
            pen.setCosmetic(True)
            preview.setPen(pen)
            preview.setBrush(QBrush(QColor(91, 130, 232, 48)))
            preview.setZValue(120)
            preview.setAcceptedMouseButtons(Qt.NoButton)
            self.scene.addItem(preview)
            self.rectangle_preview_item = preview
        self.rectangle_preview_item.setRect(rect)
        self.rectangle_preview_item.setVisible(rect.width() > 1 and rect.height() > 1)

    def clear_rectangle_preview(self) -> None:
        preview = self.rectangle_preview_item
        self.rectangle_preview_item = None
        if preview is not None and preview.scene() is not None:
            preview.scene().removeItem(preview)

    def delete_current_mask(self) -> None:
        self.cancel_draw()
        self.region["mask"] = None
        self.render_scene()
        self.status.setText(tr('● ROI 삭제됨 · 새 도형을 선택하세요'))

    def canvas_clicked(self, point: QPointF) -> None:
        if self.tool_mode == "landmark" and self.landmark_target:
            target = self.landmark_target
            landmarks = self.region.setdefault("landmarks", empty_ymaze_landmarks())
            clicked = [float(point.x()), float(point.y())]
            if target == "center_triangle":
                triangle = landmarks.setdefault("center_triangle", [None, None, None])
                index = next((index for index, value in enumerate(triangle) if not value), -1)
                if index >= 0:
                    triangle[index] = clicked
                self.render_scene(); self.update_ymaze_steps()
                remaining = sum(not value for value in triangle)
                if remaining:
                    self.tool_mode = "landmark"
                    self.landmark_target = "center_triangle"
                    self.view.click_through_items = True
                    self.view.setCursor(Qt.CrossCursor)
                    self.status.setText(tr('● 이전 Center 기준점 {p0}/3 클릭 대기', p0=4 - remaining))
                    return
            elif target == "entry_lines":
                lines = landmarks.setdefault(
                    "entry_lines", {"arm_a": [None, None], "arm_b": [None, None], "arm_c": [None, None]}
                )
                slots = [(arm, index) for arm in ("arm_a", "arm_b", "arm_c") for index in range(2)]
                slot = next(((arm, index) for arm, index in slots if not lines[arm][index]), None)
                if slot:
                    lines[slot[0]][slot[1]] = clicked
                self.render_scene(); self.update_ymaze_steps()
                count = sum(bool(value) for arm, index in slots for value in [lines[arm][index]])
                if count < 6:
                    arm = ("A", "B", "C")[count // 2]
                    endpoint = count % 2 + 1
                    self.tool_mode = "landmark"
                    self.landmark_target = "entry_lines"
                    self.view.click_through_items = True
                    self.view.setCursor(Qt.CrossCursor)
                    self.status.setText(tr('● Arm {p0} 입구선 끝점 {p1}/2 클릭 대기 · 총 {p2}/6', p0=arm, p1=endpoint, p2=count))
                    return
            else:
                landmarks[target] = clicked
            self.landmark_target = None
            self.tool_mode = "select"
            self.view.click_through_items = False
            self.view.unsetCursor()
            self.render_scene()
            self.update_ymaze_steps()
            next_target = {
                "center_triangle": "arm_a", "entry_lines": "arm_a", "arm_a": "arm_b", "arm_b": "arm_c",
            }.get(target)
            if next_target:
                self.begin_ymaze_landmark(next_target)
            else:
                self.generate_ymaze_zones()
            return
        if self.tool_mode == "select" and self.preset == "ymaze" and not self.single_arena:
            for region_index, region in enumerate(self.regions):
                polygon = np.asarray(self.polygon(region.get("mask")), dtype=np.float32)
                if len(polygon) >= 3 and cv2.pointPolygonTest(
                    polygon, (float(point.x()), float(point.y())), False
                ) >= 0:
                    self.region_combo.setCurrentIndex(region_index)
                    self.status.setText(tr('● {p0} 선택 · Center/Arm을 지정하세요', p0=self.regions[region_index]['name']))
                    return
        if self.tool_mode != "polygon":
            return
        if not (0 <= point.x() < self.frame.shape[1] and 0 <= point.y() < self.frame.shape[0]):
            return
        self.pending_points.append(point)
        self.render_pending()

    def render_pending(self) -> None:
        for item in self.guide_items:
            self.scene.removeItem(item)
        self.guide_items.clear()
        if len(self.pending_points) >= 2:
            polygon = QGraphicsPolygonItem(QPolygonF(self.pending_points))
            polygon.setPen(QPen(QColor("#ffffff"), 1.5, Qt.DashLine))
            polygon.setBrush(QBrush(QColor(255, 255, 255, 20)))
            polygon.setZValue(80)
            self.scene.addItem(polygon); self.guide_items.append(polygon)
        for point in self.pending_points:
            radius = float(self.pending_point_size)
            marker = QGraphicsEllipseItem(-radius, -radius, radius * 2, radius * 2)
            marker.setPos(point)
            marker.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            marker.setAcceptedMouseButtons(Qt.NoButton)
            marker.setPen(QPen(QColor("#0b111b"), 1.2))
            marker.setBrush(QColor("#ffffff")); marker.setZValue(90)
            self.scene.addItem(marker); self.guide_items.append(marker)

    def finish_polygon(self) -> None:
        if len(self.pending_points) < 3:
            if self.pending_points:
                QMessageBox.information(self, tr('다각형'), tr('꼭짓점을 세 개 이상 지정하세요.'))
            return
        kind, zone_id = self.draw_target
        item_color = QColor("#ff9e64") if kind == "maze_outline" else COLORS[0]
        item = PolygonObjectItem(list(self.pending_points), item_color, kind)
        if kind == "mask":
            self.region["mask"] = item
        elif kind == "maze_outline":
            self.region["maze_outline"] = item
            self.region["landmarks"] = empty_ymaze_landmarks()
            self.region["arm_label_rotation"] = 0
            self.clear_generated_ymaze_zones(include_center=True)
        else:
            zone = next((zone for zone in self.region["zones"] if zone["id"] == zone_id), None)
            if zone:
                zone["polygon"] = item
        self.pending_points.clear()
        self.tool_mode = "select"
        self.view.creation_enabled = False
        self.status.setText(tr('● ROI 완성 · 내부 이동 / 핸들 크기 조절') if self.roi_only else tr('● 편집 중'))
        self.update_tool_buttons()
        self.view.unsetCursor()
        self.render_scene()
        completed = (
            self.region.get("mask") if kind == "mask" else
            self.region.get("maze_outline") if kind == "maze_outline" else
            (zone.get("polygon") if zone else None)
        )
        if completed is not None and not isinstance(completed, list):
            completed.setSelected(True)
        self.update_ymaze_steps()
        if self.preset == "ymaze" and not self.roi_only:
            if kind == "maze_outline":
                self.set_ymaze_stage(1)
                self.begin_center_geometry()
                return
            if kind == "zone" and zone_id == "center":
                self.generate_ymaze_zones()
                self.set_ymaze_stage(2)
                return
            self.status.setText(
                tr('● 현재 Arena 3/3 완료 · 저장') if self.single_arena
                else tr('● 현재 Y-maze 3/3 완료 · 다음 Y-maze 또는 저장')
            )

    def undo_point(self) -> None:
        if self.pending_points:
            self.pending_points.pop(); self.render_pending()

    def cancel_draw(self) -> None:
        self.clear_rectangle_preview()
        self.pending_points.clear()
        for item in self.guide_items:
            if item.scene(): self.scene.removeItem(item)
        self.guide_items.clear()
        if hasattr(self, "status"):
            self.status.setText(tr('● 편집 중'))
        if hasattr(self, "view"):
            self.view.creation_enabled = False
            self.view.click_through_items = False
            self.view.unsetCursor()
        self.tool_mode = "select"
        self.landmark_target = None
        self.update_tool_buttons()
        self.update_ymaze_steps()

    def queue_seek_frame(self, value: int) -> None:
        self._pending_seek_ms = int(value)
        self.current_seconds = value / 1000.0
        self.update_time_label()
        self._seek_timer.start()

    def update_time_label(self) -> None:
        fps = max(float(self.metadata.get("fps", 0.0) or 0.0), 1e-6)
        frame_count = max(int(self.metadata.get("frame_count", 1) or 1), 1)
        frame_index = min(max(round(self.current_seconds * fps), 0), frame_count - 1)
        self.time_label.setText(
            f"{self.format_time(self.current_seconds)} / "
            f"{self.format_time(self.metadata['duration_seconds'])}  ·  "
            f"{frame_index + 1}/{frame_count}F"
        )

    @staticmethod
    def format_time(seconds: float) -> str:
        value = max(float(seconds), 0.0)
        minutes = int(value // 60)
        return f"{minutes:02d}:{value - minutes * 60:06.3f}"

    def apply_queued_seek(self) -> None:
        self.seek_frame(self._pending_seek_ms)

    def seek_delta(self, seconds: float = 0.0, frames: int = 0) -> None:
        fps = max(float(self.metadata.get("fps", 0.0) or 0.0), 1e-6)
        target = self.current_seconds + float(seconds) + int(frames) / fps
        target = min(max(target, 0.0), float(self.metadata["duration_seconds"]))
        milliseconds = int(round(target * 1000.0))
        self.timeline.setValue(milliseconds)
        self.seek_frame(milliseconds)

    def seek_frame(self, milliseconds: int | None = None) -> None:
        self._seek_timer.stop()
        if milliseconds is not None:
            self.timeline.blockSignals(True)
            self.timeline.setValue(int(milliseconds))
            self.timeline.blockSignals(False)
        self.current_seconds = self.timeline.value() / 1000.0
        self.frame, _ = frame_at(self.video, self.current_seconds)
        self.render_scene()

    def fit_canvas(self) -> None:
        """Fit the video once on demand without disturbing point-entry clicks."""
        if hasattr(self, "view") and hasattr(self, "scene") and not self.scene.sceneRect().isEmpty():
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self._scene_fitted = True

    def render_scene(self) -> None:
        if not hasattr(self, "scene"):
            return
        self.clear_rectangle_preview()
        # QGraphicsScene.clear() deletes C++ items; preserve their edited scene
        # geometry as plain points before rebuilding the frame and overlays.
        for region in self.regions:
            mask = region.get("mask")
            if mask is not None and not isinstance(mask, list):
                region["mask"] = mask.scene_polygon()
            maze_outline = region.get("maze_outline")
            if maze_outline is not None and not isinstance(maze_outline, list):
                region["maze_outline"] = maze_outline.scene_polygon()
            for zone in region.get("zones", []):
                item = zone.get("polygon")
                if item is not None and not isinstance(item, list):
                    zone["polygon"] = item.scene_polygon()
            if self.preset == "ymaze" and not self.roi_only:
                # The three Arms and Center must use one canonical junction
                # boundary.  Reapply it after every scene edit so the rendered
                # outlines and the polygons used by analysis cannot drift apart.
                self.snap_ymaze_arm_center_edges(region)
        self.scene.clear(); self.guide_items.clear()
        self.reference_overlay_items = {"arena": [], "outline": [], "fill": []}
        self.scene.addPixmap(cv_to_pixmap(self.frame)).setZValue(0)
        self.scene.setSceneRect(0, 0, self.frame.shape[1], self.frame.shape[0])
        for region_index, region in enumerate(self.regions):
            raw_mask = region.get("mask")
            if not raw_mask:
                continue
            points = [QPointF(*point) for point in raw_mask] if isinstance(raw_mask, list) else [
                QPointF(*point) for point in raw_mask.scene_polygon()
            ]
            if region_index == self.current_region and not region.get("mask_locked"):
                mask = PolygonObjectItem(points, COLORS[0], region["name"])
                mask.setPen(QPen(COLORS[0], 3.0))
                mask.setBrush(QBrush(QColor(85, 214, 169, 42)))
                region["mask"] = mask
            else:
                mask = QGraphicsPolygonItem(QPolygonF(points))
                current_locked = region_index == self.current_region and region.get("mask_locked")
                mask.setPen(QPen(COLORS[0] if current_locked else QColor("#6682a6"), 3.0 if current_locked else 2.0, Qt.DashLine))
                mask.setBrush(QBrush(QColor(85, 214, 169, 30) if current_locked else QColor(70, 105, 145, 24)))
                mask.setAcceptedMouseButtons(Qt.NoButton)
                mask.setZValue(15)
            self.scene.addItem(mask)
            self.reference_overlay_items["arena"].append(mask)
            # Y-maze vertices are commonly placed near the junction, exactly
            # where a centered Arena label used to cover the image. The right
            # panel already identifies the active Arena, so keep the canvas
            # clear while drawing Y-maze geometry.
            if self.preset != "ymaze":
                bounds = QPolygonF(points).boundingRect()
                label = QGraphicsSimpleTextItem(
                    f"{region['name']}  {(tr('현재 편집') if region_index == self.current_region else '')}".strip()
                )
                label.setBrush(COLORS[0] if region_index == self.current_region else QColor("#8fa6c2"))
                label.setPos(bounds.center().x() - label.boundingRect().width() / 2, bounds.center().y() - 10)
                label.setAcceptedMouseButtons(Qt.NoButton)
                label.setZValue(45)
                self.scene.addItem(label)
            raw_outline = region.get("maze_outline")
            outline_points = (
                [QPointF(*point) for point in raw_outline]
                if isinstance(raw_outline, list) else
                ([QPointF(*point) for point in raw_outline.scene_polygon()] if raw_outline else [])
            )
            if len(outline_points) >= 3:
                if region_index == self.current_region and (
                    self.preset != "ymaze" or self.roi_only or self.ymaze_stage == 0
                ):
                    outline = PolygonObjectItem(outline_points, QColor("#ff9e64"), tr('Maze 외곽'))
                    outline.setPen(QPen(QColor("#ff9e64"), 3.0))
                    outline.setBrush(QBrush(QColor(255, 158, 100, 22)))
                    region["maze_outline"] = outline
                else:
                    outline = QGraphicsPolygonItem(QPolygonF(outline_points))
                    current_reference = region_index == self.current_region
                    outline.setPen(QPen(
                        QColor("#ff9e64") if current_reference else QColor("#9c7357"),
                        2.4 if current_reference else 2.0,
                        Qt.DashLine,
                    ))
                    outline.setBrush(QBrush(QColor(255, 158, 100, 10)))
                    outline.setAcceptedMouseButtons(Qt.NoButton)
                outline.setZValue(24)
                self.scene.addItem(outline)
                self.reference_overlay_items["outline"].append(outline)
        for index, zone in enumerate(self.region.get("zones", [])):
            item = zone.get("polygon")
            if isinstance(item, list):
                points = [QPointF(*point) for point in item]
                editable = not (
                    self.preset == "ymaze" and not self.roi_only
                    and (zone.get("id") != "center" or self.ymaze_stage != 1)
                )
                if editable:
                    item = PolygonObjectItem(points, COLORS[(index + 1) % len(COLORS)], zone["name"])
                    zone["polygon"] = item
                else:
                    item = QGraphicsPolygonItem(QPolygonF(points))
                    item.setAcceptedMouseButtons(Qt.NoButton)
            if item:
                color = COLORS[(index + 1) % len(COLORS)]
                item.setPen(QPen(color, 2.2)); item.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 45)))
                self.scene.addItem(item)
                self.reference_overlay_items["fill"].append((item, item.brush()))
                if self.preset == "ymaze" and not self.roi_only and zone.get("id") in {"arm_a", "arm_b", "arm_c", "center"}:
                    zone_points = self.polygon(zone.get("polygon"))
                    if len(zone_points) >= 3:
                        center_x, center_y = self.polygon_centroid(zone_points)
                        zone_text = {"arm_a": "A", "arm_b": "B", "arm_c": "C", "center": "CENTER"}[zone["id"]]
                        font = QFont("Segoe UI", 15 if zone["id"] != "center" else 10, QFont.Bold)
                        anchor = QGraphicsRectItem()
                        anchor.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
                        anchor.setAcceptedMouseButtons(Qt.NoButton)
                        anchor.setPos(center_x, center_y)
                        anchor.setZValue(70)
                        self.scene.addItem(anchor)
                        # A dark duplicate keeps the letter readable on both the
                        # black maze and the translucent colored overlay. Child
                        # offsets stay in screen pixels when the image is zoomed.
                        for offset, brush, z_value in ((QPointF(1.5, 1.5), QColor("#07111d"), 69), (QPointF(0, 0), QColor("#ffffff"), 70)):
                            label = QGraphicsSimpleTextItem(zone_text, anchor)
                            label.setFont(font); label.setBrush(brush)
                            label.setAcceptedMouseButtons(Qt.NoButton); label.setZValue(z_value)
                            bounds = label.boundingRect()
                            label.setPos(-bounds.width() / 2 + offset.x(), -bounds.height() / 2 + offset.y())
        if self.preset == "ymaze" and not self.roi_only:
            landmark_style = {
                "arm_a": ("A", QColor("#ffca5c")),
                "arm_b": ("B", QColor("#ff6b8a")),
                "arm_c": ("C", QColor("#66a8ff")),
            }
            landmarks = self.region.get("landmarks", {})
            triangle_points = [point for point in landmarks.get("center_triangle", []) if point]
            if len(triangle_points) >= 2:
                guide = QGraphicsPolygonItem(QPolygonF([QPointF(*point) for point in triangle_points]))
                guide.setPen(QPen(QColor("#70e2ae"), 2.0, Qt.DashLine))
                guide.setBrush(QBrush(QColor(112, 226, 174, 22)))
                guide.setAcceptedMouseButtons(Qt.NoButton); guide.setZValue(92)
                self.scene.addItem(guide)
            for index, point in enumerate(landmarks.get("center_triangle", [])):
                if not point:
                    continue
                if not self.show_landmarks:
                    continue
                marker = DraggableLandmarkItem(
                    f"center_{index}", point, f"C{index + 1}", QColor("#70e2ae"),
                    self.landmark_moved, self.delete_ymaze_landmark,
                    radius=self.landmark_size, show_label=self.show_landmarks,
                )
                self.scene.addItem(marker)
            line_colors = {"arm_a": QColor("#ffca5c"), "arm_b": QColor("#ff6b8a"), "arm_c": QColor("#66a8ff")}
            entry_lines = landmarks.get("entry_lines", {})
            for arm_id, label_text in (("arm_a", tr('A 입구')), ("arm_b", tr('B 입구')), ("arm_c", tr('C 입구'))):
                points = entry_lines.get(arm_id, [None, None])
                color = line_colors[arm_id]
                if len(points) == 2 and all(points):
                    guide = QGraphicsPolygonItem(QPolygonF([QPointF(*points[0]), QPointF(*points[1])]))
                    guide.setPen(QPen(color, 3.0, Qt.DashLine)); guide.setBrush(Qt.NoBrush)
                    guide.setAcceptedMouseButtons(Qt.NoButton); guide.setZValue(93)
                    self.scene.addItem(guide)
                for index, point in enumerate(points):
                    if not point:
                        continue
                    if not self.show_landmarks:
                        continue
                    marker = DraggableLandmarkItem(
                        f"line_{arm_id}_{index}", point, f"{label_text} {index + 1}", color,
                        self.landmark_moved, self.delete_ymaze_landmark,
                        radius=self.landmark_size, show_label=self.show_landmarks,
                    )
                    self.scene.addItem(marker)
            for key, point in landmarks.items():
                if not point or key not in landmark_style:
                    continue
                if not self.show_landmarks:
                    continue
                label_text, color = landmark_style[key]
                marker = DraggableLandmarkItem(
                    key, point, label_text, color,
                    self.landmark_moved, self.delete_ymaze_landmark,
                    radius=self.landmark_size, show_label=self.show_landmarks,
                )
                self.scene.addItem(marker)
        self.update_time_label()
        self.update_region_count()
        self.update_overlay_visibility()
        if not self._scene_fitted:
            self.fit_canvas()

    def update_overlay_visibility(self, *_args):
        toggles = getattr(self, "overlay_toggles", {})
        for key, items in getattr(self, "reference_overlay_items", {}).items():
            show = toggles[key].isChecked() if key in toggles else True
            for entry in items:
                item = entry[0] if key == "fill" else entry
                if self.preset == "ymaze" and not self.roi_only:
                    pen = item.pen(); pen.setCosmetic(True); item.setPen(pen)
                if key == "fill":
                    item.setBrush(entry[1] if show else QBrush(Qt.NoBrush))
                else:
                    item.setVisible(show)

    @staticmethod
    def polygon(item) -> list[list[float]]:
        if isinstance(item, list):
            return [[float(x), float(y)] for x, y in item]
        return item.scene_polygon() if item else []

    def save_profile(self, close_after: bool = False) -> None:
        incomplete = [region["name"] for region in self.regions if not self.polygon(region.get("mask"))]
        if incomplete:
            QMessageBox.information(self, tr('추적 마스크 미완성'), tr('추적 마스크가 없는 영역: ') + ", ".join(incomplete))
            return
        if self.preset == "ymaze" and not self.roi_only:
            missing_outlines = [
                region["name"] for region in self.regions
                if not self.polygon(region.get("maze_outline"))
            ]
            if missing_outlines:
                QMessageBox.information(
                    self, tr('Maze 외곽 미완성'),
                    tr('실제 Y-maze 외곽을 먼저 그리세요: ') + ", ".join(missing_outlines),
                )
                return
        empty_zones = [
            f"{region['name']} / {zone['name']}"
            for region in self.regions for zone in region["zones"] if not self.polygon(zone.get("polygon"))
        ]
        if empty_zones:
            QMessageBox.information(self, tr('행동 구역 미완성'), tr('다각형이 없는 구역: ') + ", ".join(empty_zones))
            return
        arenas = []
        for region in self.regions:
            mask = self.polygon(region["mask"])
            serialized_zones = [
                {
                    "id": zone["id"], "name": zone["name"],
                    "role": zone.get("role", "choice"),
                    "polygon": self.polygon(zone["polygon"]),
                }
                for zone in region["zones"]
            ]
            if self.preset == "ymaze" and not self.roi_only:
                # Serialize the shared Center/Arm edges from the same canonical
                # coordinates even when the live scene still owns graphics items.
                self.snap_ymaze_arm_center_edges({"zones": serialized_zones,
                                                 "maze_outline": self.polygon(region.get("maze_outline"))})
            arenas.append({
                "id": region["id"], "name": region["name"],
                "mask_locked": bool(region.get("mask_locked", False)),
                "center_mode": region.get("center_mode", "triangle"),
                "corners_source_px": mask,
                "mask_polygon_source_px": mask,
                "maze_outline_polygon_source_px": self.polygon(region.get("maze_outline")),
                "ymaze_landmarks_px": region.get("landmarks", {}),
                "zones": [
                    {
                        "id": zone["id"], "name": zone["name"], "role": zone.get("role", "choice"),
                        "polygon_source_px": zone["polygon"],
                    }
                    for zone in serialized_zones
                ],
            })
        payload = {
            "schema_version": 5,
            "profile_type": "hagisilta_zone_template",
            "coordinate_space": self.coordinate_space,
            "arena_id": self.arena_id,
            "source_video": str(self.video),
            "source_resolution_px": [self.metadata["width"], self.metadata["height"]],
            "setup_frame_seconds": self.current_seconds,
            "editor_display": {
                "show_landmarks": self.show_landmarks,
                "landmark_size": self.landmark_size,
            },
            "arenas": arenas,
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.status.setText(tr('● 템플릿 저장 완료'))
        print(f"ZONE_PROFILE={self.output}", flush=True)
        QMessageBox.information(
            self,
            tr('ROI 저장') if self.roi_only else tr('구역 템플릿 저장'),
            tr('저장했습니다.\n\n{p0}', p0=self.output),
        )
        if close_after:
            self.close()

    def load_profile(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        legacy_roi = payload.get("profile_type") in {"nort_arena_roi", "nort_camera_calibration"}
        if payload.get("profile_type") != "hagisilta_zone_template" and not (self.roi_only and legacy_roi):
            raise ValueError(tr('HAGISILTA 구역 템플릿이 아닙니다.'))
        expected = [self.metadata["width"], self.metadata["height"]]
        if list(payload.get("source_resolution_px", [])) != expected:
            raise ValueError(tr('템플릿과 영상 해상도가 다릅니다.'))
        if legacy_roi and self.output.resolve() == path.resolve():
            # Keep the original calibration/ROI schema for its existing users.
            candidate = path.with_name(f"{path.stem}__tracking-roi.json")
            counter = 2
            while candidate.exists():
                candidate = path.with_name(f"{path.stem}__tracking-roi-{counter}.json")
                counter += 1
            self.output = candidate
        display = payload.get("editor_display") or {}
        self.show_landmarks = bool(display.get("show_landmarks", self.show_landmarks))
        self.landmark_size = max(2, min(int(display.get("landmark_size", self.landmark_size)), 10))
        if hasattr(self, "landmark_visibility"):
            self.landmark_visibility.blockSignals(True)
            self.landmark_visibility.setChecked(self.show_landmarks)
            self.landmark_visibility.blockSignals(False)
        if hasattr(self, "landmark_size_slider"):
            self.landmark_size_slider.blockSignals(True)
            self.landmark_size_slider.setValue(self.landmark_size)
            self.landmark_size_slider.blockSignals(False)
        self.regions = []
        for index, arena in enumerate(payload.get("arenas", []), start=1):
            raw_landmarks = arena.get("ymaze_landmarks_px") or {}
            landmarks = empty_ymaze_landmarks()
            raw_triangle = raw_landmarks.get("center_triangle") or []
            for point_index, point in enumerate(raw_triangle[:3]):
                landmarks["center_triangle"][point_index] = point
            for arm_id in ("arm_a", "arm_b", "arm_c"):
                landmarks[arm_id] = raw_landmarks.get(arm_id)
            raw_lines = raw_landmarks.get("entry_lines") or {}
            for arm_id in ("arm_a", "arm_b", "arm_c"):
                points = raw_lines.get(arm_id) or []
                for point_index, point in enumerate(points[:2]):
                    landmarks["entry_lines"][arm_id][point_index] = point
            self.regions.append({
                "id": int(arena.get("id", index)), "name": arena.get("name", tr('추적 영역 {p0}', p0=index)),
                "mask_locked": bool(arena.get("mask_locked", False)),
                "center_mode": arena.get("center_mode", "triangle"),
                "mask": arena.get("mask_polygon_source_px") or arena.get("corners_source_px") or [],
                "maze_outline": arena.get("maze_outline_polygon_source_px") or [],
                "landmarks": landmarks,
                "zones": [
                    {
                        "id": zone.get("id", f"zone_{zone_index}"),
                        "name": zone.get("name", tr('구역 {p0}', p0=zone_index)),
                        "role": zone.get("role", "choice"),
                        "polygon": zone.get("polygon_source_px") or [],
                    }
                    for zone_index, zone in enumerate(arena.get("zones", []), start=1)
                ],
            })
        if not self.regions:
            self.add_region()
        # Loading an existing profile must not change Arena identity: tracking
        # results and experiment setups can refer to non-contiguous IDs.
        if self.preset == "ymaze" and not self.roi_only:
            for region in self.regions:
                if not self.polygon(region.get("maze_outline")):
                    region["center_mode"] = "triangle"
                    region["landmarks"] = empty_ymaze_landmarks()
                    for zone in region.get("zones", []):
                        zone["polygon"] = None
        if self.preset == "ymaze" and int(payload.get("schema_version", 1) or 1) < 3:
            for region in self.regions:
                for zone in region.get("zones", []):
                    zone["polygon"] = None
        self.current_region = 0
        self.current_seconds = float(payload.get("setup_frame_seconds", self.current_seconds))
        self.timeline.setValue(round(self.current_seconds * 1000))
        self.frame, _ = frame_at(self.video, self.current_seconds)
        self.refresh_combos()
        if self.preset == "ymaze" and hasattr(self, "center_mode_combo"):
            mode_index = self.center_mode_combo.findData(self.region.get("center_mode", "triangle"))
            self.center_mode_combo.blockSignals(True)
            self.center_mode_combo.setCurrentIndex(max(mode_index, 0))
            self.center_mode_combo.blockSignals(False)

    def load_arena_profile(self, path: Path) -> None:
        """Use the tracking ROI profile as the full-frame Y-maze editor background."""
        self.load_profile(path)
        for index, region in enumerate(self.regions, start=1):
            region["id"] = int(region.get("id", index))
            region["name"] = f"Arena {region['id']:02d}"
            region["mask_locked"] = True
            region["center_mode"] = "free_polygon"
            region["arm_label_rotation"] = 0
            region["maze_outline"] = []
            region["landmarks"] = empty_ymaze_landmarks()
            region["zones"] = []
            self.add_ymaze_zones(region)
        self.current_region = 0
        self.refresh_combos()
        self.render_scene()


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv)
    init_i18n(app)
    install_wheel_value_guard(app)
    app.setApplicationName("HAGISILTA")
    if APP_ICON.is_file():
        app.setWindowIcon(QIcon(str(APP_ICON)))
    window = ZoneStudio(
        args.video, args.output, args.profile, args.roi_only, args.preset,
        args.arena_id, args.coordinate_space, args.arena_profile,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
