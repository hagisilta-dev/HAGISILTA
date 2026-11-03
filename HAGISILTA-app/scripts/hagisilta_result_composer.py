# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
"""Shared result-image composers for NORT, Y-maze, and Water maze."""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import QPointF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QColorDialog, QComboBox,
    QDialog, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QScrollArea,
    QSizePolicy, QSlider, QSpinBox, QSplitter, QTabWidget, QToolButton,
    QVBoxLayout, QWidget,
)

from hagisilta_i18n import tr
from hagisilta_window_chrome import (
    HeaderContext, apply_hagisilta_window_identity, create_hagisilta_header,
    create_hagisilta_help_button, enable_full_window_controls,
    hagisilta_control_stylesheet,
)


class WaterMazePreviewLabel(QLabel):
    resized = Signal()
    clicked = Signal(QPointF)
    wheel_zoom = Signal(int, QPointF)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.resized.emit()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit(event.position())
        super().mousePressEvent(event)

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta:
            self.wheel_zoom.emit(1 if delta > 0 else -1, event.position())
            event.accept()
            return
        super().wheelEvent(event)


class WaterMazePreviewScrollArea(QScrollArea):
    resized = Signal()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.resized.emit()


class WaterMazeVisualComposer(QDialog):
    """Compose and batch-export Water-maze trajectory and heatmap figures."""

    EXPERIMENT_KIND = "watermaze"
    DEFAULT_COLORS = {
        "canvas_background": (0.99, 0.0, 0.0),
        "trail": (0.78, 0.16, 200.0),
        "pool": (0.82, 0.13, 165.0),
        "wall": (0.76, 0.16, 75.0),
        "trajectory_fill": (0.30, 0.025, 250.0),
        "heatmap_fill": (0.27, 0.045, 290.0),
        "platform": (0.72, 0.20, 20.0),
        "quadrant_line": (0.88, 0.03, 250.0),
        "title_text": (0.22, 0.025, 255.0),
        "panel_text": (0.34, 0.025, 255.0),
        "metric_text": (0.28, 0.025, 255.0),
        "quadrant_text": (0.90, 0.025, 250.0),
        "colorbar_text": (0.25, 0.02, 255.0),
    }

    def __init__(self, entries: list[dict], current_index: int = 0, parent=None) -> None:
        super().__init__(parent)
        if not entries:
            raise ValueError(tr('표시할 Water-maze 결과가 없습니다.'))
        self.entries = entries
        self.current_index = min(max(int(current_index), 0), len(entries) - 1)
        self.data_cache: dict[int, dict] = {}
        self.current_result: QImage | None = None
        self.color_specs = dict(self.DEFAULT_COLORS)
        self.customized_colors: set[str] = set()
        self.background_mode_restore: dict[str, int] | None = None
        self.loading_color_controls = False
        self.export_width = 3840
        self.export_custom_width = 1920
        self.preview_zoom = 1.0
        self.preview_status_base = ""
        self.canvas_style_editor: QDialog | None = None
        self.render_debounce = QTimer(self)
        self.render_debounce.setSingleShot(True)
        self.render_debounce.setInterval(55)
        self.render_debounce.timeout.connect(self.render_current)
        self.setWindowTitle("HAGISILTA · " + self.composer_title())
        apply_hagisilta_window_identity(self)
        enable_full_window_controls(self)
        self.resize(1380, 860)
        self.setMinimumSize(1050, 680)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        header, header_layout = create_hagisilta_header(
            self.composer_title(),
            "HAGISILTA · RESULT COMPOSER",
        )
        help_text = (
            tr('Trial별 궤적·히트맵 결과 이미지를 다시 구성합니다. 파란 행은 현재 미리보기이고, 체크는 일괄 저장 대상입니다. 원본 분석 CSV와 프로필은 변경하지 않습니다.\n\n표시 요소와 배치, 히트맵 색상표, OKLCH 색상, 투명도와 궤적 두께를 바꾸면 중앙 미리보기에 즉시 반영됩니다.')
        )
        header_layout.addWidget(create_hagisilta_help_button(
            self, tr('궤적·히트맵 편집 도움말'), help_text,
        ))
        self.header_status = HeaderContext(header, f"{self.current_index + 1}/{len(entries)}")
        root.addWidget(header)
        self.setStyleSheet(self.styleSheet() + hagisilta_control_stylesheet())

        body = QFrame(objectName="detachedBody")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(14, 12, 14, 12)
        body_layout.setSpacing(12)

        playlist_panel = QFrame(objectName="detachedSidebar")
        playlist_panel.setMinimumWidth(230)
        playlist_panel.setMaximumWidth(480)
        playlist_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        playlist_layout = QVBoxLayout(playlist_panel)
        playlist_layout.setContentsMargins(10, 9, 10, 10)
        playlist_layout.setSpacing(7)
        playlist_layout.addWidget(QLabel(self.composer_playlist_title(), objectName="settingsGroupTitle"))
        playlist_description = QLabel(
            tr('파란 행은 미리보기 · 체크는 일괄 저장 대상'),
            objectName="settingsGroupDescription",
        )
        playlist_description.setWordWrap(True)
        playlist_layout.addWidget(playlist_description)
        use_row = QHBoxLayout()
        use_all = QPushButton(tr('모두 선택'), objectName="toolButton")
        use_none = QPushButton(tr('모두 해제'), objectName="toolButton")
        use_all.clicked.connect(lambda: self.set_all_checked(True))
        use_none.clicked.connect(lambda: self.set_all_checked(False))
        use_row.addWidget(use_all)
        use_row.addWidget(use_none)
        playlist_layout.addLayout(use_row)
        self.playlist_widget = QListWidget()
        self.playlist_widget.setObjectName("decisionPlaylist")
        self.playlist_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.playlist_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.playlist_widget.setSpacing(4)
        for entry in self.entries:
            summary = dict(entry.get("summary") or {})
            bundle = str(summary.get("video_bundle") or Path(entry.get("video", "")).stem)
            arena = self._integer(summary.get("arena_id"), 1)
            trial = self._integer(summary.get("trial"), 1)
            trial_type = str(summary.get("trial_type", "") or "-")
            item = QListWidgetItem(self.playlist_item_text(entry, summary))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setSizeHint(QSize(0, 58))
            item.setToolTip(str(entry.get("video", "")))
            self.playlist_widget.addItem(item)
        self.playlist_widget.setCurrentRow(self.current_index)
        self.playlist_widget.currentRowChanged.connect(self.select_entry)
        playlist_layout.addWidget(self.playlist_widget, 1)
        navigation = QHBoxLayout()
        previous = QPushButton(tr('← 이전'), objectName="secondaryButton")
        following = QPushButton(tr('다음 →'), objectName="secondaryButton")
        previous.clicked.connect(lambda: self.move_entry(-1))
        following.clicked.connect(lambda: self.move_entry(1))
        navigation.addWidget(previous)
        navigation.addWidget(following)
        playlist_layout.addLayout(navigation)

        workspace = QFrame(objectName="reviewWorkspace")
        workspace.setMinimumWidth(420)
        workspace.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(8)
        self.preview_scroll = WaterMazePreviewScrollArea()
        self.preview_scroll.setWidgetResizable(False)
        self.preview_scroll.setAlignment(Qt.AlignCenter)
        self.preview_scroll.setMinimumSize(520, 420)
        self.preview_scroll.setFrameShape(QFrame.NoFrame)
        self.preview_scroll.setStyleSheet("background:#090f18; border:1px solid #31445d;")
        self.preview = WaterMazePreviewLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(1, 1)
        self.preview.setStyleSheet("background:#090f18; border:0;")
        self.preview_scroll.setWidget(self.preview)
        self.preview_scroll.resized.connect(self.update_preview_pixmap)
        self.preview.clicked.connect(self.handle_preview_click)
        self.preview.wheel_zoom.connect(self.adjust_preview_zoom)
        workspace_layout.addWidget(self.preview_scroll, 1)
        self.preview_status = QLabel(objectName="statusSummaryValue")
        self.preview_status.setWordWrap(True)
        status_panel = QFrame(objectName="statusSummaryBar")
        status_layout = QVBoxLayout(status_panel)
        status_layout.setContentsMargins(10, 6, 10, 6)
        status_layout.addWidget(self.preview_status)
        workspace_layout.addWidget(status_panel)

        review_splitter = QSplitter(Qt.Horizontal)
        review_splitter.setObjectName("reviewPlaylistSplitter")
        review_splitter.setChildrenCollapsible(False)
        review_splitter.setHandleWidth(7)
        review_splitter.addWidget(playlist_panel)
        review_splitter.addWidget(workspace)
        review_splitter.setStretchFactor(0, 0)
        review_splitter.setStretchFactor(1, 1)
        review_splitter.setSizes([280, 760])
        settings_panel = QFrame(objectName="detachedSidebar")
        settings_panel.setMinimumWidth(350)
        settings_panel.setMaximumWidth(460)
        settings_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        settings_outer = QVBoxLayout(settings_panel)
        settings_outer.setContentsMargins(0, 0, 0, 0)
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QFrame.NoFrame)
        settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        settings_body = QWidget()
        settings_layout = QVBoxLayout(settings_body)
        settings_layout.setContentsMargins(10, 10, 10, 10)
        settings_layout.setSpacing(10)

        settings_panel.setStyleSheet(settings_panel.styleSheet() + """
            QFrame#propertySection { background:#24262b; border:1px solid #454952; border-radius:2px; }
            QToolButton#propertySectionHeader {
                background:#2c2f35; color:#eef0f3; border:0; border-bottom:1px solid #454952;
                min-height:28px; padding:0 8px; font-weight:600; text-align:left;
            }
            QToolButton#propertySectionHeader:hover { background:#343841; }
            QFrame#propertySectionBody { background:#24262b; border:0; }
            QLabel#propertyFieldLabel { color:#aeb3bd; background:transparent; border:0; }
            QTabWidget#propertyTabs::pane {
                background:#24262b; border:1px solid #454952; top:-1px;
            }
            QTabWidget#propertyTabs QTabBar::tab {
                min-width:84px; min-height:28px; padding:0 10px;
                color:#b8bdc7; background:#202228; border:1px solid #3d4149;
            }
            QTabWidget#propertyTabs QTabBar::tab:hover { background:#30343c; color:#f4f6fa; }
            QTabWidget#propertyTabs QTabBar::tab:selected {
                color:#ffffff; background:#466ed1; border-color:#6488e4;
            }
        """)

        components_group, components_layout = self.create_property_section(tr('레이어'), True)
        self.component_checks: dict[str, QCheckBox] = {}
        component_specs = [
            ("background", tr('원본 이미지'), True),
            ("background_color", tr('배경색'), True),
            ("pool", tr('수조 ROI'), True),
            ("pool_fill", tr('사분면 내부 채우기'), False),
            ("platform", tr('플랫폼'), True),
            ("quadrants", tr('사분면 경계·이름'), True),
            ("trail", tr('전체 궤적'), True),
            ("heatmap", tr('히트맵'), True),
            ("information", tr('제목·시험 정보·주요 지표'), True),
        ]
        if self.EXPERIMENT_KIND == "watermaze":
            component_specs.insert(3, ("wall", tr('벽면 구역'), True))
        for key, label, checked in component_specs:
            checkbox = QCheckBox(self.style_label(key, label))
            checkbox.setChecked(checked)
            checkbox.setFixedHeight(23)
            checkbox.toggled.connect(self.schedule_render)
            self.component_checks[key] = checkbox
            components_layout.addWidget(checkbox)
        self.component_checks["background"].toggled.connect(self.apply_background_mode_defaults)
        settings_layout.addWidget(components_group)

        render_group, render_layout = self.create_property_section(tr('히트맵·배치'), True)
        render_layout.addWidget(QLabel(tr('결과 배치'), objectName="propertyFieldLabel"))
        self.layout_combo = QComboBox()
        self.layout_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.layout_combo.setMinimumContentsLength(12)
        self.layout_combo.addItem(tr('궤적·히트맵 나란히'), "compare")
        self.layout_combo.addItem(tr('한 화면에 합성'), "merged")
        self.layout_combo.currentIndexChanged.connect(self.schedule_render)
        render_layout.addWidget(self.layout_combo)
        render_layout.addWidget(QLabel(tr('히트맵 색상'), objectName="propertyFieldLabel"))
        self.heatmap_combo = QComboBox()
        self.heatmap_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.heatmap_combo.setMinimumContentsLength(12)
        self.heatmap_combo.addItem("Turbo", "turbo")
        self.heatmap_combo.addItem("Viridis", "viridis")
        self.heatmap_combo.addItem("Plasma", "plasma")
        self.heatmap_combo.addItem("Inferno", "inferno")
        self.heatmap_combo.addItem("Magma", "magma")
        self.heatmap_combo.addItem("Cividis", "cividis")
        self.heatmap_combo.currentIndexChanged.connect(self.schedule_render)
        render_layout.addWidget(self.heatmap_combo)
        render_layout.addWidget(QLabel(tr('히트맵 투명도'), objectName="propertyFieldLabel"))
        heat_row = QHBoxLayout()
        self.heat_opacity = QSlider(Qt.Horizontal)
        self.heat_opacity.setRange(0, 100)
        self.heat_opacity.setValue(28)
        self.heat_opacity.setToolTip(tr('0% 불투명 · 100% 완전 투명'))
        self.heat_opacity.valueChanged.connect(self.schedule_render)
        self.heat_opacity_label = QLabel("28%")
        self.heat_opacity.valueChanged.connect(
            lambda value: self.heat_opacity_label.setText(f"{value}%")
        )
        heat_row.addWidget(self.heat_opacity, 1)
        heat_row.addWidget(self.heat_opacity_label)
        render_layout.addLayout(heat_row)
        render_layout.addWidget(QLabel(tr('선·궤적 투명도'), objectName="propertyFieldLabel"))
        overlay_row = QHBoxLayout()
        self.overlay_opacity = QSlider(Qt.Horizontal)
        self.overlay_opacity.setRange(0, 100)
        self.overlay_opacity.setValue(12)
        self.overlay_opacity.setToolTip(tr('0% 불투명 · 100% 완전 투명'))
        self.overlay_opacity.valueChanged.connect(self.schedule_render)
        self.overlay_opacity_label = QLabel("12%")
        self.overlay_opacity.valueChanged.connect(
            lambda value: self.overlay_opacity_label.setText(f"{value}%")
        )
        overlay_row.addWidget(self.overlay_opacity, 1)
        overlay_row.addWidget(self.overlay_opacity_label)
        render_layout.addLayout(overlay_row)
        settings_layout.addWidget(render_group)

        style_group, style_layout = self.create_property_section(tr('선·글자 스타일'), False)
        self.style_group = style_group
        style_tabs = QTabWidget(objectName="propertyTabs")
        style_tabs.setDocumentMode(True)
        style_tabs.tabBar().setExpanding(True)
        self.style_tabs = style_tabs
        self.color_buttons: dict[str, QPushButton] = {}

        def make_style_tab() -> tuple[QWidget, QGridLayout]:
            tab = QWidget()
            grid = QGridLayout(tab)
            grid.setContentsMargins(10, 12, 10, 12)
            grid.setHorizontalSpacing(9)
            grid.setVerticalSpacing(10)
            grid.setColumnStretch(0, 1)
            grid.setAlignment(Qt.AlignTop)
            return tab, grid

        def add_style_row(
            grid: QGridLayout, row: int, label: str, attribute: str,
            minimum: int, maximum: int, value: int, color_key: str | None = None,
        ) -> None:
            label = self.style_label(color_key, label)
            grid.addWidget(QLabel(label, objectName="propertyFieldLabel"), row, 0)
            control = QSpinBox()
            control.setRange(minimum, maximum)
            control.setValue(value)
            control.setSuffix(" px")
            control.setFixedHeight(30)
            control.valueChanged.connect(self.schedule_render)
            setattr(self, attribute, control)
            grid.addWidget(control, row, 1)
            if color_key:
                color_button = QPushButton("", objectName="colorChip")
                color_button.setFixedSize(42, 30)
                color_button.setToolTip(tr('{p0} 색상', p0=label))
                color_button.clicked.connect(
                    lambda _checked=False, key=color_key, name=label: self.choose_color_for_key(key, name)
                )
                self.color_buttons[color_key] = color_button
                grid.addWidget(color_button, row, 2)

        text_tab, text_grid = make_style_tab()
        for row, spec in enumerate((
            (tr('제목'), "title_text_size", 8, 48, 15, "title_text"),
            (tr('패널명'), "panel_text_size", 8, 36, 12, "panel_text"),
            (tr('주요 지표'), "metric_text_size", 8, 36, 12, "metric_text"),
            (tr('사분면'), "quadrant_text_size", 8, 36, 14, "quadrant_text"),
            (tr('컬러바'), "colorbar_text_size", 8, 32, 11, "colorbar_text"),
        )):
            add_style_row(text_grid, row, *spec)
        style_tabs.addTab(text_tab, tr('글자'))

        line_tab, line_grid = make_style_tab()
        line_specs = [
            (tr('수조 ROI'), "pool_line_width", 1, 12, 2, "pool"),
            (tr('플랫폼'), "platform_line_width", 1, 12, 2, "platform"),
            (tr('사분면'), "quadrant_line_width", 1, 12, 1, "quadrant_line"),
            (tr('궤적'), "trail_width", 1, 16, 2, "trail"),
            (tr('컬러바 폭'), "colorbar_width", 8, 48, 18, None),
        ]
        if self.EXPERIMENT_KIND == "watermaze":
            line_specs.insert(1, (tr('벽면 구역'), "wall_line_width", 1, 12, 2, "wall"))
        for row, spec in enumerate(line_specs):
            add_style_row(line_grid, row, *spec)
        style_tabs.addTab(line_tab, tr('선'))

        fill_tab, fill_grid = make_style_tab()
        background_label = QLabel(tr('배경색'), objectName="propertyFieldLabel")
        fill_grid.addWidget(background_label, 0, 0)
        background_hint = QLabel(tr('레이어에서 켜기'), objectName="propertyFieldLabel")
        fill_grid.addWidget(background_hint, 0, 1)
        background_button = QPushButton("", objectName="colorChip")
        background_button.setFixedSize(42, 30)
        background_button.setToolTip(tr('배경색 선택'))
        background_button.clicked.connect(
            lambda _checked=False: self.choose_color_for_key("canvas_background", tr('배경색'))
        )
        self.color_buttons["canvas_background"] = background_button
        fill_grid.addWidget(background_button, 0, 2)
        for row, (label, attribute, value, color_key) in enumerate((
            (tr('궤적 내부 채움'), "trajectory_fill_transparency", 72, "trajectory_fill"),
            (tr('히트맵 내부 채움'), "heatmap_fill_transparency", 72, "heatmap_fill"),
        ), 1):
            label = self.style_label(color_key, label)
            fill_grid.addWidget(QLabel(label, objectName="propertyFieldLabel"), row, 0)
            transparency = QSpinBox()
            transparency.setRange(0, 100)
            transparency.setValue(value)
            transparency.setSuffix(" %")
            transparency.setFixedHeight(30)
            transparency.setToolTip(tr('0% 불투명 · 100% 완전 투명'))
            transparency.valueChanged.connect(self.schedule_render)
            setattr(self, attribute, transparency)
            fill_grid.addWidget(transparency, row, 1)
            color_button = QPushButton("", objectName="colorChip")
            color_button.setFixedSize(42, 30)
            color_button.setToolTip(tr('{p0} 색상', p0=label))
            color_button.clicked.connect(
                lambda _checked=False, key=color_key, name=label: self.choose_color_for_key(key, name)
            )
            self.color_buttons[color_key] = color_button
            fill_grid.addWidget(color_button, row, 2)
        style_tabs.addTab(fill_tab, tr('채움'))
        style_tab_heights = (272, 272 + 40 * (len(line_specs) - 5), 202)
        style_tabs.currentChanged.connect(
            lambda index: style_tabs.setFixedHeight(style_tab_heights[max(0, min(index, 2))])
        )
        style_tabs.setFixedHeight(style_tab_heights[0])
        style_layout.addWidget(style_tabs)
        # Capture the actual controls' initial defaults once, before user edits.
        self.style_tab_defaults = []
        for index in range(style_tabs.count()):
            tab = style_tabs.widget(index)
            self.style_tab_defaults.append({
                "values": [(control, control.value()) for control in tab.findChildren(QSpinBox)],
                "colors": [key for key, button in self.color_buttons.items() if tab.isAncestorOf(button)],
            })
        self.reset_style_button = QPushButton(objectName="toolButton")
        self.reset_style_button.clicked.connect(self.reset_current_style_tab)
        style_tabs.currentChanged.connect(self.update_style_reset_button)
        self.update_style_reset_button()
        style_layout.addWidget(self.reset_style_button)
        settings_layout.addWidget(style_group)

        settings_layout.addStretch()
        settings_scroll.setWidget(settings_body)
        settings_outer.addWidget(settings_scroll)
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setObjectName("composerMainSplitter")
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setHandleWidth(7)
        main_splitter.addWidget(review_splitter)
        main_splitter.addWidget(settings_panel)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 0)
        main_splitter.setSizes([920, 390])
        body_layout.addWidget(main_splitter, 1)
        root.addWidget(body, 1)

        footer_frame = QFrame(objectName="detachedFooter")
        footer = QHBoxLayout(footer_frame)
        footer.setContentsMargins(18, 9, 18, 9)
        self.save_current_button = QPushButton(tr('현재 이미지 저장'), objectName="secondaryButton")
        self.save_batch_button = QPushButton(tr('체크 항목 일괄 저장'), objectName="primaryButton")
        close_button = QPushButton(tr('닫기'), objectName="secondaryButton")
        self.save_current_button.clicked.connect(self.save_current)
        self.save_batch_button.clicked.connect(self.save_checked)
        close_button.clicked.connect(self.accept)
        footer.addStretch()
        footer.addWidget(self.save_current_button)
        footer.addWidget(self.save_batch_button)
        footer.addWidget(close_button)
        root.addWidget(footer_frame)
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)
        self.refresh_color_buttons()
        self.render_current()

    def create_property_section(self, title: str, expanded: bool = True) -> tuple[QFrame, QVBoxLayout]:
        section = QFrame(objectName="propertySection")
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.setSpacing(0)
        header = QToolButton(objectName="propertySectionHeader")
        header.setText(title)
        header.setCheckable(True)
        header.setChecked(expanded)
        header.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        body = QFrame(section, objectName="propertySectionBody")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(9, 8, 9, 9)
        body_layout.setSpacing(6)
        body.setVisible(expanded)

        def toggle_section(checked: bool) -> None:
            body.setVisible(checked)
            header.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

        header.toggled.connect(toggle_section)
        section_layout.addWidget(header)
        section_layout.addWidget(body)
        return section, body_layout

    @staticmethod
    def _integer(value, fallback: int = 0) -> int:
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _number(value) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    def playlist_item_text(self, entry: dict, summary: dict) -> str:
        bundle = str(summary.get("video_bundle") or Path(entry.get("video", "")).stem)
        arena = self._integer(summary.get("arena_id"), 1)
        trial = self._integer(summary.get("trial"), 1)
        trial_type = str(summary.get("trial_type", "") or "-")
        return f"{bundle}\nArena {arena:02d}  ·  Trial {trial}  ·  {trial_type}"

    def result_title_text(self, data: dict) -> str:
        summary = data["summary"]
        bundle = str(summary.get("video_bundle", "") or Path(data["entry"]["video"]).stem)
        arena = self._integer(summary.get("arena_id"), 1)
        trial = self._integer(summary.get("trial"), 1)
        trial_type = str(summary.get("trial_type", "") or "-")
        return f"{bundle} · Arena {arena:02d} · Trial {trial} · {trial_type}"

    def preview_status_text(self, data: dict) -> str:
        summary = data["summary"]
        bundle = str(summary.get("video_bundle", "") or "-")
        arena = self._integer(summary.get("arena_id"), 1)
        trial = self._integer(summary.get("trial"), 1)
        return f"{bundle} · Arena {arena:02d} · Trial {trial}"

    @staticmethod
    def oklch_rgb(spec: tuple[float, float, float]) -> tuple[int, int, int]:
        lightness, chroma, hue = spec
        radians = math.radians(hue)
        lab_a = chroma * math.cos(radians)
        lab_b = chroma * math.sin(radians)
        l_value = lightness + 0.3963377774 * lab_a + 0.2158037573 * lab_b
        m_value = lightness - 0.1055613458 * lab_a - 0.0638541728 * lab_b
        s_value = lightness - 0.0894841775 * lab_a - 1.2914855480 * lab_b
        l_value, m_value, s_value = l_value ** 3, m_value ** 3, s_value ** 3
        linear = (
            4.0767416621 * l_value - 3.3077115913 * m_value + 0.2309699292 * s_value,
            -1.2684380046 * l_value + 2.6097574011 * m_value - 0.3413193965 * s_value,
            -0.0041960863 * l_value - 0.7034186147 * m_value + 1.7076147010 * s_value,
        )

        def transfer(channel: float) -> int:
            channel = 12.92 * channel if channel <= 0.0031308 else 1.055 * max(channel, 0.0) ** (1 / 2.4) - 0.055
            return round(min(max(channel, 0.0), 1.0) * 255)

        return tuple(transfer(channel) for channel in linear)

    @staticmethod
    def rgb_oklch(red: int, green: int, blue: int) -> tuple[float, float, float]:
        def linearize(channel: int) -> float:
            value = min(max(channel / 255.0, 0.0), 1.0)
            return value / 12.92 if value <= .04045 else ((value + .055) / 1.055) ** 2.4

        red_l, green_l, blue_l = map(linearize, (red, green, blue))
        l_value = 0.4122214708 * red_l + 0.5363325363 * green_l + 0.0514459929 * blue_l
        m_value = 0.2119034982 * red_l + 0.6806995451 * green_l + 0.1073969566 * blue_l
        s_value = 0.0883024619 * red_l + 0.2817188376 * green_l + 0.6299787005 * blue_l
        l_root, m_root, s_root = np.cbrt((l_value, m_value, s_value))
        lightness = 0.2104542553 * l_root + 0.7936177850 * m_root - 0.0040720468 * s_root
        lab_a = 1.9779984951 * l_root - 2.4285922050 * m_root + 0.4505937099 * s_root
        lab_b = 0.0259040371 * l_root + 0.7827717662 * m_root - 0.8086757660 * s_root
        chroma = math.hypot(lab_a, lab_b)
        hue = math.degrees(math.atan2(lab_b, lab_a)) % 360.0 if chroma > 1e-9 else 0.0
        return float(lightness), float(chroma), float(hue)

    def color_bgr(self, key: str) -> tuple[int, int, int]:
        red, green, blue = self.oklch_rgb(self.color_specs[key])
        return blue, green, red

    def surface_color_bgr(self, key: str, light_surface: bool = False) -> tuple[int, int, int]:
        if light_surface and key not in self.customized_colors:
            automatic = {
                "quadrant_line": (0.48, 0.02, 255.0),
                "quadrant_text": (0.34, 0.025, 255.0),
            }.get(key)
            if automatic is not None:
                red, green, blue = self.oklch_rgb(automatic)
                return blue, green, red
        return self.color_bgr(key)

    def color_qcolor(self, key: str) -> QColor:
        red, green, blue = self.oklch_rgb(self.color_specs[key])
        return QColor(red, green, blue)

    def choose_color_for_key(self, key: str, label: str = "") -> None:
        label = self.style_label(key, label)
        red, green, blue = self.oklch_rgb(self.color_specs[key])
        chosen = QColorDialog.getColor(
            QColor(red, green, blue), self,
            tr('{p0} 색상 선택', p0=label or key),
            QColorDialog.ColorDialogOption.DontUseNativeDialog,
        )
        if not chosen.isValid():
            return
        self.color_specs[key] = self.rgb_oklch(chosen.red(), chosen.green(), chosen.blue())
        self.customized_colors.add(key)
        if key in {"trajectory_fill", "heatmap_fill"}:
            self.component_checks["pool_fill"].setChecked(True)
        elif key == "canvas_background":
            self.component_checks["background_color"].setChecked(True)
        self.refresh_color_buttons()
        self.render_current()

    def refresh_color_buttons(self) -> None:
        for key, button in self.color_buttons.items():
            red, green, blue = self.oklch_rgb(self.color_specs[key])
            lightness, chroma, hue = self.color_specs[key]
            button.setStyleSheet(
                f"background:rgb({red},{green},{blue}); border:1px solid #747983; border-radius:2px;"
            )
            button.setToolTip(
                tr('클릭하여 색상 선택\nOKLCH: L {p0:.2f} · C {p1:.2f} · H {p2:.0f}°', p0=lightness, p1=chroma, p2=hue)
            )

    def update_style_reset_button(self, *_args) -> None:
        name = self.style_tabs.tabText(self.style_tabs.currentIndex())
        self.reset_style_button.setText(tr('{p0} 설정 초기화', p0=name))
        self.reset_style_button.setToolTip(
            tr('확인 후 ‘{p0}’ 탭의 값과 색상만 기본값으로 되돌립니다. 다른 탭과 레이어 설정은 유지합니다.', p0=name)
        )

    def confirm_style_reset(self, tab_name: str) -> bool:
        message = QMessageBox(self)
        apply_hagisilta_window_identity(message)
        message.setStyleSheet(
            "QMessageBox {background:#23242a; color:#e4e5e8;} "
            "QMessageBox QLabel {color:#e4e5e8;}" + hagisilta_control_stylesheet()
        )
        message.setWindowTitle(tr('{p0} 설정 초기화', p0=tab_name))
        message.setIcon(QMessageBox.Question)
        message.setText(tr('‘{p0}’ 탭의 값과 색상을 기본값으로 되돌릴까요?', p0=tab_name))
        message.setInformativeText(
            tr('다른 탭·레이어·히트맵 설정은 유지합니다.\n원본 영상·분석 데이터와 이미 저장한 이미지는 변경하지 않습니다.')
        )
        reset = message.addButton(tr('초기화'), QMessageBox.AcceptRole)
        cancel = message.addButton(tr('취소'), QMessageBox.RejectRole)
        message.setDefaultButton(cancel)
        message.setEscapeButton(cancel)
        message.exec()
        confirmed = message.clickedButton() == reset
        message.deleteLater()
        return confirmed

    def reset_current_style_tab(self) -> None:
        index = self.style_tabs.currentIndex()
        if not self.confirm_style_reset(self.style_tabs.tabText(index)):
            return
        defaults = self.style_tab_defaults[index]
        for control, value in defaults["values"]:
            control.setValue(value)
        for key in defaults["colors"]:
            self.color_specs[key] = self.DEFAULT_COLORS[key]
        self.customized_colors.difference_update(defaults["colors"])
        self.refresh_color_buttons()
        self.render_current()

    def apply_background_mode_defaults(self, original_enabled: bool) -> None:
        if original_enabled:
            restore = self.background_mode_restore
            if restore:
                if self.heat_opacity.value() == 0:
                    self.heat_opacity.setValue(restore["heat"])
                if self.overlay_opacity.value() == 0:
                    self.overlay_opacity.setValue(restore["overlay"])
            self.background_mode_restore = None
            return
        self.background_mode_restore = {
            "heat": self.heat_opacity.value(),
            "overlay": self.overlay_opacity.value(),
        }
        self.heat_opacity.setValue(0)
        self.overlay_opacity.setValue(0)

    def expand_style_section(self, tab_index: int) -> None:
        header = self.style_group.findChild(QToolButton, "propertySectionHeader")
        if header is not None:
            header.setChecked(True)
        self.style_tabs.setCurrentIndex(tab_index)

    def style_label(self, key: str | None, default: str) -> str:
        """Name the edited property, not just the shared source UI text."""
        return tr('사분면 이름') if key == "quadrant_text" else default

    def composer_title(self) -> str:
        return {
            "nort": tr('NORT 궤적·히트맵 편집'),
            "ymaze": tr('Y-maze 궤적·히트맵 편집'),
        }.get(self.EXPERIMENT_KIND, tr('Water-maze 궤적·히트맵 편집'))

    def composer_playlist_title(self) -> str:
        return tr('결과 영상·Trial') if self.EXPERIMENT_KIND == "watermaze" else tr('결과 영상·Arena')

    def open_canvas_style_editor(self, key: str) -> None:
        definitions = {
            "canvas_background": (tr('배경색'), None, "", 2),
            "title_text": (tr('제목 글자'), "title_text_size", tr('크기'), 0),
            "panel_text": (tr('패널명 글자'), "panel_text_size", tr('크기'), 0),
            "metric_text": (tr('주요 지표 글자'), "metric_text_size", tr('크기'), 0),
            "quadrant_text": (tr('사분면 글자'), "quadrant_text_size", tr('크기'), 0),
            "colorbar_text": (tr('컬러바 글자'), "colorbar_text_size", tr('크기'), 0),
            "pool": (tr('수조 ROI 선'), "pool_line_width", tr('두께'), 1),
            "wall": (tr('벽면 구역 경계선'), "wall_line_width", tr('두께'), 1),
            "platform": (tr('플랫폼 선'), "platform_line_width", tr('두께'), 1),
            "quadrant_line": (tr('사분면 선'), "quadrant_line_width", tr('두께'), 1),
            "trail": (tr('궤적 선'), "trail_width", tr('두께'), 1),
            "trajectory_fill": (tr('궤적 사분면 내부'), "trajectory_fill_transparency", tr('투명도'), 2),
            "heatmap_fill": (tr('히트맵 사분면 내부'), "heatmap_fill_transparency", tr('투명도'), 2),
        }
        if key not in definitions or (key == "wall" and "wall" not in self.component_checks):
            return
        label, attribute, value_label, tab_index = definitions[key]
        label = self.style_label(key, label)
        self.expand_style_section(tab_index)
        if self.canvas_style_editor is not None:
            self.canvas_style_editor.close()
        dialog = QDialog(self)
        self.canvas_style_editor = dialog
        apply_hagisilta_window_identity(dialog)
        dialog.setStyleSheet(
            "QDialog {background:#23242a; color:#e4e5e8;} "
            "QLabel {color:#e4e5e8;}" + hagisilta_control_stylesheet()
        )
        dialog.setWindowTitle(tr('{p0} 스타일', p0=label))
        dialog.setModal(False)
        dialog.setFixedWidth(340)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(9)
        layout.addWidget(QLabel(label, objectName="settingsGroupTitle"))
        note = QLabel(tr('표시 스타일만 변경합니다. ROI·구역의 위치와 모양은 바뀌지 않습니다.'))
        note.setWordWrap(True)
        note.setObjectName("settingsGroupDescription")
        layout.addWidget(note)
        row = QHBoxLayout()
        if attribute:
            row.addWidget(QLabel(value_label, objectName="propertyFieldLabel"))
            source_control = getattr(self, attribute)
            editor = QSpinBox()
            editor.setRange(source_control.minimum(), source_control.maximum())
            editor.setValue(source_control.value())
            editor.setSuffix(" %" if tab_index == 2 else " px")
            editor.valueChanged.connect(source_control.setValue)
            source_control.valueChanged.connect(editor.setValue)
            row.addWidget(editor, 1)
        else:
            row.addStretch()
        color = QPushButton(tr('색상 선택…'), objectName="colorChip")

        def choose_context_color() -> None:
            self.choose_color_for_key(key, label)
            color.setStyleSheet(
                f"background:{self.color_qcolor(key).name()}; border:1px solid #747983;"
            )

        color.clicked.connect(choose_context_color)
        choose_context_color.__name__ = "choose_context_color"
        color.setStyleSheet(
            f"background:{self.color_qcolor(key).name()}; border:1px solid #747983;"
        )
        row.addWidget(color)
        layout.addLayout(row)
        close = QPushButton(tr('닫기'), objectName="primaryButton")
        close.clicked.connect(dialog.close)
        layout.addWidget(close, alignment=Qt.AlignRight)
        dialog.finished.connect(lambda _result: setattr(self, "canvas_style_editor", None))
        dialog.show()

    @staticmethod
    def point_segment_distance(
        point: tuple[float, float], start: tuple[float, float], end: tuple[float, float],
    ) -> float:
        px, py = point
        ax, ay = start
        bx, by = end
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-9:
            return math.hypot(px - ax, py - ay)
        ratio = min(max(((px - ax) * dx + (py - ay) * dy) / length_sq, 0.0), 1.0)
        return math.hypot(px - (ax + ratio * dx), py - (ay + ratio * dy))

    def preview_layer_at(self, x_value: float, y_value: float) -> str:
        if self.current_result is None:
            return ""
        data = self.entry_data(self.current_index)
        output_width, output_height = self.current_result.width(), self.current_result.height()
        source_height, source_width = data["background"].shape[:2]
        trail_enabled = self.component_checks["trail"].isChecked()
        heat_enabled = self.component_checks["heatmap"].isChecked()
        compare = self.layout_combo.currentData() == "compare" and trail_enabled and heat_enabled
        panel_count = 2 if compare else 1
        base_gap = 10 if compare else 0
        base_legend = self.heat_legend_base_width() if heat_enabled else 0
        base_width = source_width * panel_count + base_gap + base_legend
        render_scale = output_width / max(base_width, 1)
        gap = max(1, round(base_gap * render_scale)) if compare else 0
        legend_width = max(70, round(base_legend * render_scale)) if heat_enabled else 0
        panel_space = max(panel_count, output_width - gap - legend_width)
        first_width = max(1, panel_space // panel_count)
        panel_widths = [first_width]
        if compare:
            panel_widths.append(max(1, panel_space - first_width))
        panel_height = max(round(source_height * width / max(source_width, 1)) for width in panel_widths)
        information_scale = first_width / max(source_width, 1)
        show_information = self.component_checks["information"].isChecked()
        header_height = round((self.title_text_size.value() + self.panel_text_size.value() + 21) * information_scale) if show_information else 0
        footer_height = round((self.metric_text_size.value() + 26) * information_scale) if show_information else 0
        if show_information and y_value < header_height:
            title_cut = (self.title_text_size.value() + 9) * information_scale
            return "title_text" if y_value <= title_cut else "panel_text"
        if show_information and y_value >= output_height - footer_height:
            return "metric_text"
        if heat_enabled and x_value >= output_width - legend_width:
            return "colorbar_text"
        panel_index = 0
        panel_left = 0
        if compare and x_value >= first_width + gap:
            panel_index = 1
            panel_left = first_width + gap
        panel_width = panel_widths[panel_index]
        local_x = (x_value - panel_left) * source_width / max(panel_width, 1)
        local_y = (y_value - header_height) * source_width / max(panel_width, 1)
        point = (local_x, local_y)
        profile = data["profile"]
        pool = profile["pool"]
        cx, cy = map(float, pool["center_px"])
        radius = float(pool.get("radius_px", 1.0))
        rx = float(pool.get("radius_x_px", radius))
        ry = float(pool.get("radius_y_px", radius))
        platform = profile["platform"]
        if self.component_checks["platform"].isChecked():
            if str(platform.get("shape_type", "circle")) == "circle":
                pcx, pcy = map(float, platform["center_px"])
                pr = float(platform.get("radius_px", 1.0))
                if abs(math.hypot(local_x - pcx, local_y - pcy) - pr) <= 10:
                    return "platform"
            else:
                points = np.asarray(platform.get("points_px") or [], dtype=np.float32)
                if len(points) >= 3 and abs(cv2.pointPolygonTest(points, point, True)) <= 10:
                    return "platform"
        if panel_index == 0 and trail_enabled and self.component_checks["trail"].isChecked():
            valid_points = np.column_stack((data["x"][data["valid"]], data["y"][data["valid"]]))
            if len(valid_points):
                step = max(1, len(valid_points) // 700)
                if np.min(np.hypot(valid_points[::step, 0] - local_x, valid_points[::step, 1] - local_y)) <= 9:
                    return "trail"
        normalized_radius = math.sqrt(((local_x - cx) / max(rx, 1e-9)) ** 2 + ((local_y - cy) / max(ry, 1e-9)) ** 2)
        if normalized_radius > 1.05:
            return "canvas_background"
        if self.component_checks["pool"].isChecked() and abs(normalized_radius - 1.0) <= .045:
            return "pool"
        wall = data.get("wall_zone", {})
        wall_check = self.component_checks.get("wall")
        if wall_check is not None and wall_check.isChecked() and wall.get("available"):
            if abs(normalized_radius - (1.0 - wall["fraction"])) * min(rx, ry) <= 7:
                return "wall"
        if self.component_checks["quadrants"].isChecked():
            angle_offset = float(profile.get("north_angle_degrees", 0.0))
            for offset in (0.0, 90.0, 180.0, 270.0):
                angle = math.radians(angle_offset + offset)
                endpoint = (cx + rx * math.sin(angle), cy - ry * math.cos(angle))
                if self.point_segment_distance(point, (cx, cy), endpoint) <= 8:
                    return "quadrant_line"
            for offset in (45.0, 135.0, 225.0, 315.0):
                angle = math.radians(angle_offset + offset)
                label_point = (cx + rx * .58 * math.sin(angle), cy - ry * .58 * math.cos(angle))
                if math.hypot(local_x - label_point[0], local_y - label_point[1]) <= 22:
                    return "quadrant_text"
        return "heatmap_fill" if (compare and panel_index == 1) or (not compare and heat_enabled) else "trajectory_fill"

    def handle_preview_click(self, position: QPointF) -> None:
        if self.current_result is None or self.preview.width() <= 0 or self.preview.height() <= 0:
            return
        x_value = position.x() * self.current_result.width() / self.preview.width()
        y_value = position.y() * self.current_result.height() / self.preview.height()
        self.open_canvas_style_editor(self.preview_layer_at(x_value, y_value))

    def heat_legend_base_width(self) -> int:
        return max(
            116,
            self.colorbar_width.value() + self.colorbar_text_size.value() * 5 + 42,
        )

    def choose_export_width(self) -> int | None:
        dialog = QDialog(self)
        dialog.setWindowTitle(tr('저장 해상도'))
        dialog.setModal(True)
        dialog.setFixedWidth(380)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        layout.addWidget(QLabel(tr('저장할 이미지 너비'), objectName="settingsGroupTitle"))
        layout.addWidget(QLabel(
            tr('영상 프레임만 보간하고 선·글자·히트맵은 선택 해상도에서 다시 그립니다.'),
            objectName="settingsGroupDescription",
        ))
        resolution = QComboBox()
        for label, value in (
            (tr('원본 해상도'), 0), (tr('1920px 너비'), 1920),
            (tr('2560px 너비'), 2560), (tr('3840px 너비 · 4K'), 3840),
            (tr('사용자 지정'), -1),
        ):
            resolution.addItem(label, value)
        known_index = resolution.findData(self.export_width)
        resolution.setCurrentIndex(known_index if known_index >= 0 else resolution.findData(-1))
        layout.addWidget(resolution)
        custom = QSpinBox()
        custom.setRange(320, 7680)
        custom.setValue(self.export_custom_width)
        custom.setSuffix(tr(' px 너비'))
        layout.addWidget(custom)

        def update_custom_visibility() -> None:
            custom.setVisible(int(resolution.currentData() or 0) < 0)
            dialog.adjustSize()

        resolution.currentIndexChanged.connect(update_custom_visibility)
        update_custom_visibility()
        buttons = QHBoxLayout()
        cancel = QPushButton(tr('취소'), objectName="secondaryButton")
        proceed = QPushButton(tr('계속'), objectName="primaryButton")
        cancel.clicked.connect(dialog.reject)
        proceed.clicked.connect(dialog.accept)
        buttons.addStretch()
        buttons.addWidget(cancel)
        buttons.addWidget(proceed)
        layout.addLayout(buttons)
        if dialog.exec() != QDialog.Accepted:
            return None
        selected = int(resolution.currentData() or 0)
        if selected < 0:
            self.export_custom_width = custom.value()
            selected = self.export_custom_width
            self.export_width = -1
        else:
            self.export_width = selected
        return selected

    def set_all_checked(self, checked: bool) -> None:
        state = Qt.Checked if checked else Qt.Unchecked
        for index in range(self.playlist_widget.count()):
            self.playlist_widget.item(index).setCheckState(state)

    def move_entry(self, delta: int) -> None:
        row = min(max(self.playlist_widget.currentRow() + int(delta), 0), len(self.entries) - 1)
        self.playlist_widget.setCurrentRow(row)

    def select_entry(self, index: int) -> None:
        if not 0 <= index < len(self.entries):
            return
        self.current_index = index
        for cached_index, cached_data in self.data_cache.items():
            if cached_index != index:
                cached_data.get("spatial_cache", {}).clear()
                cached_data.get("background_cache", {}).clear()
        self.header_status.setText(f"{index + 1}/{len(self.entries)}")
        self.render_current()

    @staticmethod
    def wall_zone_geometry(profile: dict, summary: dict | None = None,
                           frame_size: tuple[int, int] | None = None,
                           video: Path | None = None) -> dict:
        """Use saved analysis thresholds, then explicit ROI/legacy values only."""
        summary = summary or {}
        unavailable = {"available": False, "fraction": None}

        def present(value) -> bool:
            if value is None or (isinstance(value, str) and not value.strip()):
                return False
            try:
                return not math.isnan(float(value))
            except (ValueError, TypeError):
                return True

        def number(value) -> float:
            result = float(value)
            if not math.isfinite(result):
                raise ValueError(tr('유한한 숫자가 필요합니다.'))
            return result

        try:
            pool = profile.get("pool") or {}
            cx, cy = map(number, pool["center_px"])
            radius = pool.get("radius_px")
            rx = number(pool.get("radius_x_px", radius))
            ry = number(pool.get("radius_y_px", radius))
            if rx <= 0 or ry <= 0:
                raise ValueError(tr('수조 반지름이 올바르지 않습니다.'))
            if profile.get("coordinate_space") not in (None, "", "arena_crop_px"):
                raise ValueError(tr('수조 좌표계가 ROI 영상과 다릅니다.'))
            if video and profile.get("source_video") and Path(profile["source_video"]).resolve() != Path(video).resolve():
                raise ValueError(tr('수조 설정이 다른 영상의 설정입니다.'))
            if frame_size is not None and profile.get("frame_size"):
                saved_size = tuple(map(number, profile["frame_size"]))
                if saved_size != tuple(frame_size):
                    raise ValueError(tr('수조 설정의 영상 크기가 현재 영상과 다릅니다.'))
            if present(summary.get("wall_zone_radius_percent")):
                fraction = number(summary["wall_zone_radius_percent"]) / 100.0
                if not 0 <= fraction <= .95:
                    raise ValueError(tr('분석 결과의 벽면 구역 비율이 올바르지 않습니다.'))
                source = tr('분석 당시 기준')
            elif present(summary.get("wall_zone_width_cm")) or present(profile.get("wall_zone_width_cm")):
                diameter = number(pool.get("diameter_cm"))
                if diameter <= 0:
                    raise ValueError(tr('벽면 폭을 환산할 저장 수조 지름이 없습니다.'))
                from_summary = present(summary.get("wall_zone_width_cm"))
                width = number(summary["wall_zone_width_cm"] if from_summary else profile["wall_zone_width_cm"])
                if width < 0:
                    raise ValueError(tr('벽면 구역 폭은 음수일 수 없습니다.'))
                fraction = min(width / (diameter / 2.0), .95)
                source = tr('분석 당시 폭') if from_summary else tr('저장된 ROI 폭')
            elif present(profile.get("peripheral_ring_percent")):
                fraction = number(profile["peripheral_ring_percent"])
                if not 0 <= fraction <= 1:
                    raise ValueError(tr('이전 ROI의 벽면 구역 비율이 올바르지 않습니다.'))
                fraction = min(fraction, .95)
                source = tr('이전 ROI의 반지름 비율')
            else:
                return {**unavailable, "detail": tr('벽면 구역 기준이 저장되어 있지 않아 표시하지 않습니다.')}
            if fraction == 0:
                return {**unavailable, "detail": tr('저장된 벽면 구역 폭이 0이므로 표시하지 않습니다.')}
            return {
                "available": True, "fraction": fraction,
                "center": (cx, cy), "inner_radii": (rx * (1.0 - fraction), ry * (1.0 - fraction)),
                "detail": tr('{p0} · 반지름의 {p1:.1f}% · 점선과 수조 외곽 사이가 벽면 구역입니다.', p0=source, p1=fraction * 100),
            }
        except (KeyError, TypeError, ValueError, OSError) as error:
            return {**unavailable, "detail": tr('벽면 구역 미표시 · {p0}', p0=error)}

    def update_wall_layer_controls(self, data: dict) -> None:
        check = self.component_checks.get("wall")
        if check is None:
            return
        wall = data.get("wall_zone", {})
        available = bool(wall.get("available"))
        check.setEnabled(available)
        check.setToolTip(wall.get("detail", tr('벽면 구역 기준이 없습니다.')))
        self.wall_line_width.setEnabled(available)
        self.color_buttons["wall"].setEnabled(available)

    def entry_data(self, index: int) -> dict:
        if index in self.data_cache:
            return self.data_cache[index]
        entry = self.entries[index]
        frames = pd.read_csv(Path(entry["frames"]))
        profile = json.loads(Path(entry["profile"]).read_text(encoding="utf-8"))
        if frames.empty:
            raise ValueError(tr('프레임 판정 CSV가 비어 있습니다.'))
        video = Path(entry["video"])
        capture = cv2.VideoCapture(str(video))
        video_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(video_fps) or video_fps <= 0:
            video_fps = float(profile.get("fps") or (entry.get("summary") or {}).get("fps") or 30.0)
        source_frame = self._integer(frames.iloc[0].get("frame"), 0)
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(source_frame, 0))
        ok, background = capture.read()
        capture.release()
        if not ok:
            frame_size = profile.get("frame_size") or [640, 480]
            background = np.full((int(frame_size[1]), int(frame_size[0]), 3), 35, dtype=np.uint8)
        x_values = pd.to_numeric(frames.get("x"), errors="coerce").to_numpy(dtype=float)
        y_values = pd.to_numeric(frames.get("y"), errors="coerce").to_numpy(dtype=float)
        valid_series = frames.get("valid", pd.Series(False, index=frames.index))
        valid = valid_series.astype(str).str.lower().isin({"1", "true", "yes"}).to_numpy()
        valid &= np.isfinite(x_values) & np.isfinite(y_values)
        height, width = background.shape[:2]
        counts = np.zeros((height, width), dtype=np.float32)
        if valid.any():
            px = np.clip(np.rint(x_values[valid]).astype(int), 0, width - 1)
            py = np.clip(np.rint(y_values[valid]).astype(int), 0, height - 1)
            np.add.at(counts, (py, px), 1.0)
        sigma = max(5, round(min(width, height) * .025))
        blurred = cv2.GaussianBlur(counts, (sigma * 2 + 1, sigma * 2 + 1), 0)
        peak = float(blurred.max())
        heat_gray = np.zeros_like(blurred, dtype=np.uint8)
        if peak > 0:
            heat_gray = np.clip(blurred / peak * 255, 0, 255).astype(np.uint8)
        heat_peak_seconds = peak / max(video_fps, 1e-9)
        pool = profile["pool"]
        cx, cy = map(float, pool["center_px"])
        radius = float(pool.get("radius_px", 1.0))
        rx = float(pool.get("radius_x_px", radius))
        ry = float(pool.get("radius_y_px", radius))
        yy, xx = np.mgrid[:height, :width]
        pool_mask = ((xx - cx) / max(rx, 1e-9)) ** 2 + ((yy - cy) / max(ry, 1e-9)) ** 2 <= 1.0
        data = {
            "entry": entry, "summary": dict(entry.get("summary") or {}),
            "frames": frames, "profile": profile, "background": background,
            "x": x_values, "y": y_values, "valid": valid,
            "heat_gray": heat_gray, "pool_mask": pool_mask,
            "video_fps": video_fps, "heat_peak_seconds": heat_peak_seconds,
            "spatial_cache": {}, "background_cache": {},
            "wall_zone": self.wall_zone_geometry(profile, dict(entry.get("summary") or {}), (width, height), video),
        }
        self.data_cache[index] = data
        return data

    def spatial_layers(self, data: dict, width: int, height: int) -> tuple[np.ndarray, np.ndarray, float, float]:
        cache = data.setdefault("spatial_cache", {})
        cache_key = (int(width), int(height))
        if cache_key in cache:
            return cache[cache_key]
        source_height, source_width = data["background"].shape[:2]
        scale_x = width / max(source_width, 1)
        scale_y = height / max(source_height, 1)
        counts = np.zeros((height, width), dtype=np.float32)
        valid = data["valid"]
        if valid.any():
            px = np.clip(np.rint(data["x"][valid] * scale_x).astype(int), 0, width - 1)
            py = np.clip(np.rint(data["y"][valid] * scale_y).astype(int), 0, height - 1)
            np.add.at(counts, (py, px), 1.0)
        sigma = max(5, round(min(width, height) * .025))
        blurred = cv2.GaussianBlur(counts, (sigma * 2 + 1, sigma * 2 + 1), 0)
        peak = float(blurred.max())
        heat_gray = np.zeros_like(blurred, dtype=np.uint8)
        if peak > 0:
            heat_gray = np.clip(blurred / peak * 255, 0, 255).astype(np.uint8)
        pool = data["profile"]["pool"]
        cx, cy = map(float, pool["center_px"])
        radius = float(pool.get("radius_px", 1.0))
        rx = float(pool.get("radius_x_px", radius))
        ry = float(pool.get("radius_y_px", radius))
        yy, xx = np.mgrid[:height, :width]
        pool_mask = (
            ((xx - cx * scale_x) / max(rx * scale_x, 1e-9)) ** 2
            + ((yy - cy * scale_y) / max(ry * scale_y, 1e-9)) ** 2
            <= 1.0
        )
        result = (heat_gray, pool_mask, scale_x, scale_y)
        if len(cache) >= 2:
            cache.clear()
        cache[cache_key] = result
        return result

    def draw_geometry(
        self, image: np.ndarray, data: dict, draw_trail: bool,
        scale_x: float = 1.0, scale_y: float = 1.0,
        light_surface: bool = False,
    ) -> None:
        height, width = image.shape[:2]
        scale = max(.75, min(width / 640.0, height / 480.0))
        profile = data["profile"]
        pool = profile["pool"]
        cx, cy = map(float, pool["center_px"])
        radius = float(pool.get("radius_px", 1.0))
        rx = float(pool.get("radius_x_px", radius))
        ry = float(pool.get("radius_y_px", radius))
        center = (round(cx * scale_x), round(cy * scale_y))
        axes = (max(1, round(rx * scale_x)), max(1, round(ry * scale_y)))
        if self.component_checks["pool"].isChecked():
            cv2.ellipse(image, center, axes, 0, 0, 360, self.color_bgr("pool"), max(1, round(self.pool_line_width.value() * scale)), cv2.LINE_AA)
        wall = data.get("wall_zone", {})
        wall_check = self.component_checks.get("wall")
        if wall_check is not None and wall_check.isChecked() and wall.get("available"):
            inner_rx, inner_ry = wall["inner_radii"]
            inner_axes = (max(1, round(inner_rx * scale_x)), max(1, round(inner_ry * scale_y)))
            for angle in range(0, 360, 12):
                cv2.ellipse(image, center, inner_axes, 0, angle, angle + 7,
                            self.color_bgr("wall"), max(1, round(self.wall_line_width.value() * scale)), cv2.LINE_AA)
        if self.component_checks["platform"].isChecked():
            platform = profile["platform"]
            points = np.asarray(platform.get("points_px") or [], dtype=float)
            if len(points):
                points[:, 0] *= scale_x
                points[:, 1] *= scale_y
                points = np.rint(points).astype(np.int32)
            color = self.color_bgr("platform")
            if str(platform.get("shape_type", "circle")) != "circle" and len(points) >= 3:
                cv2.polylines(image, [points], True, color, max(1, round(self.platform_line_width.value() * scale)), cv2.LINE_AA)
            else:
                pc = (
                    round(float(platform["center_px"][0]) * scale_x),
                    round(float(platform["center_px"][1]) * scale_y),
                )
                platform_scale = (scale_x + scale_y) / 2.0
                cv2.circle(image, pc, max(1, round(float(platform["radius_px"]) * platform_scale)), color, max(1, round(self.platform_line_width.value() * scale)), cv2.LINE_AA)
        if self.component_checks["quadrants"].isChecked():
            line_color = self.surface_color_bgr("quadrant_line", light_surface)
            text_color = self.surface_color_bgr("quadrant_text", light_surface)
            angle_offset = float(profile.get("north_angle_degrees", 0.0))
            for offset in (0.0, 90.0, 180.0, 270.0):
                angle = math.radians(angle_offset + offset)
                endpoint = (
                    round((cx + rx * math.sin(angle)) * scale_x),
                    round((cy - ry * math.cos(angle)) * scale_y),
                )
                cv2.line(image, center, endpoint, line_color, max(1, round(self.quadrant_line_width.value() * scale)), cv2.LINE_AA)
            for name, offset in (("NE", 45), ("SE", 135), ("SW", 225), ("NW", 315)):
                angle = math.radians(angle_offset + offset)
                point = (
                    round((cx + rx * .58 * math.sin(angle)) * scale_x),
                    round((cy - ry * .58 * math.cos(angle)) * scale_y),
                )
                font_scale = self.quadrant_text_size.value() / 30.0 * scale
                text_weight = max(1, round(self.quadrant_text_size.value() / 14.0 * scale))
                cv2.putText(image, name, point, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_weight, cv2.LINE_AA)
        if draw_trail and self.component_checks["trail"].isChecked():
            valid = data["valid"]
            points = np.column_stack((data["x"][valid] * scale_x, data["y"][valid] * scale_y))
            if len(points) >= 2:
                cv2.polylines(
                    image, [np.rint(points).astype(np.int32)], False,
                    self.color_bgr("trail"),
                    max(1, round(self.trail_width.value() * scale)), cv2.LINE_AA,
                )

    def render_panel(
        self, data: dict, include_trail: bool, include_heatmap: bool,
        target_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        source = data["background"]
        source_height, source_width = source.shape[:2]
        width, height = target_size or (source_width, source_height)
        if width == source_width and height == source_height:
            background = source.copy()
        else:
            background_cache = data.setdefault("background_cache", {})
            cache_key = (int(width), int(height))
            if cache_key not in background_cache:
                if len(background_cache) >= 2:
                    background_cache.clear()
                background_cache[cache_key] = cv2.resize(
                    source, (width, height), interpolation=cv2.INTER_CUBIC,
                )
            background = background_cache[cache_key].copy()
        if width == source_width and height == source_height:
            heat_gray, pool_mask, scale_x, scale_y = data["heat_gray"], data["pool_mask"], 1.0, 1.0
        else:
            heat_gray, pool_mask, scale_x, scale_y = self.spatial_layers(data, width, height)
        original_enabled = self.component_checks["background"].isChecked()
        background_color_enabled = self.component_checks["background_color"].isChecked()
        base_opaque = original_enabled or background_color_enabled
        if original_enabled:
            panel = background.copy()
            alpha_mask = np.full(background.shape[:2], 255, dtype=np.uint8)
        elif background_color_enabled:
            panel = np.empty_like(background)
            panel[:] = self.color_bgr("canvas_background")
            alpha_mask = np.full(background.shape[:2], 255, dtype=np.uint8)
        else:
            panel = np.zeros_like(background)
            alpha_mask = np.zeros(background.shape[:2], dtype=np.uint8)
        if self.component_checks["pool_fill"].isChecked():
            fill = np.empty_like(panel)
            fill_key = "heatmap_fill" if include_heatmap else "trajectory_fill"
            transparency = (
                self.heatmap_fill_transparency.value()
                if include_heatmap else self.trajectory_fill_transparency.value()
            )
            fill[:] = self.color_bgr(fill_key)
            alpha = 1.0 - transparency / 100.0
            if base_opaque:
                blended = cv2.addWeighted(fill, alpha, panel, 1.0 - alpha, 0)
                panel[pool_mask] = blended[pool_mask]
            else:
                panel[pool_mask] = fill[pool_mask]
                alpha_mask[pool_mask] = np.maximum(alpha_mask[pool_mask], round(alpha * 255))
        if include_heatmap and self.component_checks["heatmap"].isChecked():
            color_maps = {
                "turbo": cv2.COLORMAP_TURBO,
                "viridis": cv2.COLORMAP_VIRIDIS,
                "plasma": cv2.COLORMAP_PLASMA,
                "inferno": cv2.COLORMAP_INFERNO,
                "magma": cv2.COLORMAP_MAGMA,
                "cividis": cv2.COLORMAP_CIVIDIS,
            }
            palette_key = str(self.heatmap_combo.currentData())
            color_map = color_maps.get(palette_key, cv2.COLORMAP_TURBO)
            palette_gamma = {
                "turbo": 1.0, "viridis": .68, "plasma": .76,
                "inferno": .74, "magma": .74, "cividis": .64,
            }.get(palette_key, 1.0)
            display_gray = heat_gray
            if palette_gamma != 1.0:
                display_gray = np.clip(
                    np.power(heat_gray.astype(np.float32) / 255.0, palette_gamma) * 255.0,
                    0, 255,
                ).astype(np.uint8)
            heat = cv2.applyColorMap(display_gray, color_map)
            alpha = 1.0 - self.heat_opacity.value() / 100.0
            if base_opaque:
                blended = cv2.addWeighted(heat, alpha, panel, 1.0 - alpha, 0)
                panel[pool_mask] = blended[pool_mask]
            else:
                if alpha > 0:
                    blended = cv2.addWeighted(heat, alpha, panel, 1.0 - alpha, 0)
                    panel[pool_mask] = blended[pool_mask]
                    alpha_mask[pool_mask] = np.maximum(
                        alpha_mask[pool_mask], round(alpha * 255),
                    )
        overlay = panel.copy()
        light_surface = (
            not original_enabled
            and background_color_enabled
            and not include_heatmap
            and self.color_qcolor("canvas_background").lightnessF() >= .65
        )
        self.draw_geometry(
            overlay, data, include_trail, scale_x, scale_y,
            light_surface=light_surface,
        )
        alpha = 1.0 - self.overlay_opacity.value() / 100.0
        if base_opaque:
            panel = cv2.addWeighted(overlay, alpha, panel, 1.0 - alpha, 0)
        else:
            changed = np.any(overlay != panel, axis=2)
            panel[changed] = overlay[changed]
            alpha_mask[changed] = np.maximum(alpha_mask[changed], round(alpha * 255))
        return np.dstack((panel, alpha_mask))

    def metric_text(self, summary: dict) -> str:
        trial_type = str(summary.get("trial_type", "") or "").strip().lower()
        latency = self._number(summary.get("escape_latency_seconds"))
        distance = self._number(summary.get("path_length_cm"))
        if trial_type == "probe":
            target = self._number(summary.get("target_quadrant_percent"))
            crossings = self._number(summary.get("platform_crossings"))
            proximity = self._number(summary.get("mean_proximity_cm"))
            return (
                tr('목표 사분면 {p0:.1f}% · 플랫폼 위치 통과 {p1:d}회 · 평균 근접거리 {p2:.1f}cm', p0=target, p1=round(crossings), p2=proximity)
                if target is not None and crossings is not None and proximity is not None else tr('Probe 주요 결과 없음')
            )
        latency_text = tr('{p0:.2f}초', p0=latency) if latency is not None else tr('미도달')
        distance_text = f"{distance:,.1f}cm" if distance is not None else "-"
        if trial_type == "visible":
            speed = self._number(summary.get("mean_speed_cm_s"))
            return tr('도달시간 {p0} · 이동거리 {p1} · 평균 속도 {p2:.1f}cm/s', p0=latency_text, p1=distance_text, p2=speed) if speed is not None else tr('도달시간 {p0} · 이동거리 {p1}', p0=latency_text, p1=distance_text)
        efficiency = self._number(summary.get("path_efficiency"))
        return tr('도달시간 {p0} · 이동거리 {p1} · 경로효율 {p2:.3f}', p0=latency_text, p1=distance_text, p2=efficiency) if efficiency is not None else tr('도달시간 {p0} · 이동거리 {p1}', p0=latency_text, p1=distance_text)

    def compose_image(self, index: int, target_width: int = 0) -> QImage:
        data = self.entry_data(index)
        trail_enabled = self.component_checks["trail"].isChecked()
        heat_enabled = self.component_checks["heatmap"].isChecked()
        compare = self.layout_combo.currentData() == "compare" and trail_enabled and heat_enabled
        source_height, source_width = data["background"].shape[:2]
        panel_count = 2 if compare else 1
        base_gap = 10 if compare else 0
        base_legend_width = self.heat_legend_base_width() if heat_enabled else 0
        base_width = source_width * panel_count + base_gap + base_legend_width
        render_width = target_width if target_width > 0 else base_width
        render_scale = render_width / max(base_width, 1)
        gap = max(1, round(base_gap * render_scale)) if compare else 0
        legend_width = max(70, round(base_legend_width * render_scale)) if heat_enabled else 0
        panel_space = max(panel_count, render_width - gap - legend_width)
        first_width = max(1, panel_space // panel_count)
        panel_widths = [first_width]
        if compare:
            panel_widths.append(max(1, panel_space - first_width))
        panel_heights = [
            max(1, round(source_height * panel_width / max(source_width, 1)))
            for panel_width in panel_widths
        ]
        panel_height = max(panel_heights)
        information_scale = first_width / max(source_width, 1)
        panels: list[tuple[str, np.ndarray]] = []
        if compare:
            panels = [
                (tr('궤적'), self.render_panel(data, True, False, (panel_widths[0], panel_heights[0]))),
                (tr('히트맵'), self.render_panel(data, False, True, (panel_widths[1], panel_heights[1]))),
            ]
        else:
            panels = [(
                tr('결과'),
                self.render_panel(data, trail_enabled, heat_enabled, (panel_widths[0], panel_heights[0])),
            )]
        widths = [panel.shape[1] for _, panel in panels]
        show_information = self.component_checks["information"].isChecked()
        header_base = self.title_text_size.value() + self.panel_text_size.value() + 21
        footer_base = self.metric_text_size.value() + 26
        header_height = max(1, round(header_base * information_scale)) if show_information else 0
        footer_height = max(1, round(footer_base * information_scale)) if show_information else 0
        background_color_enabled = self.component_checks["background_color"].isChecked()
        canvas = np.zeros(
            (panel_height + header_height + footer_height, render_width, 4), dtype=np.uint8,
        )
        if background_color_enabled:
            blue, green, red = self.color_bgr("canvas_background")
            canvas[:] = (blue, green, red, 255)
        positions = []
        left = 0
        for label, panel in panels:
            top = header_height + (panel_height - panel.shape[0]) // 2
            canvas[top:top + panel.shape[0], left:left + panel.shape[1]] = panel
            positions.append((label, left, panel.shape[1], top, panel.shape[0]))
            left += panel.shape[1] + gap
        rgba = cv2.cvtColor(canvas, cv2.COLOR_BGRA2RGBA)
        image = QImage(
            rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0],
            QImage.Format_RGBA8888,
        ).copy()
        painter = QPainter(image)
        painter.setRenderHint(QPainter.TextAntialiasing)
        if heat_enabled:
            color_maps = {
                "turbo": cv2.COLORMAP_TURBO, "viridis": cv2.COLORMAP_VIRIDIS,
                "plasma": cv2.COLORMAP_PLASMA, "inferno": cv2.COLORMAP_INFERNO,
                "magma": cv2.COLORMAP_MAGMA, "cividis": cv2.COLORMAP_CIVIDIS,
            }
            color_map = color_maps.get(str(self.heatmap_combo.currentData()), cv2.COLORMAP_TURBO)
            bar_height = max(120, round(panel_height * .66))
            bar_width = max(8, round(self.colorbar_width.value() * information_scale))
            bar_x = render_width - legend_width + max(10, round(12 * information_scale))
            bar_y = header_height + (panel_height - bar_height) // 2
            gradient = np.linspace(255, 0, bar_height, dtype=np.uint8).reshape(-1, 1)
            gradient = cv2.applyColorMap(gradient, color_map)
            gradient = cv2.resize(gradient, (bar_width, bar_height), interpolation=cv2.INTER_NEAREST)
            gradient_rgb = cv2.cvtColor(gradient, cv2.COLOR_BGR2RGB)
            gradient_image = QImage(
                gradient_rgb.data, bar_width, bar_height, gradient_rgb.strides[0], QImage.Format_RGB888,
            ).copy()
            painter.drawImage(bar_x, bar_y, gradient_image)
            painter.setPen(QPen(self.color_qcolor("colorbar_text"), max(1, round(information_scale))))
            painter.drawRect(bar_x, bar_y, bar_width, bar_height)
            tick_font = QFont("Arial")
            tick_font.setPixelSize(max(8, round(self.colorbar_text_size.value() * information_scale)))
            painter.setFont(tick_font)
            peak_seconds = max(float(data.get("heat_peak_seconds") or 0.0), 0.0)
            decimals = 3 if peak_seconds < .1 else 2 if peak_seconds < 1.0 else 1
            tick_length = max(4, round(6 * information_scale))
            tick_gap = max(4, round(5 * information_scale))
            for tick_index in range(6):
                ratio = tick_index / 5.0
                tick_y = round(bar_y + ratio * bar_height)
                tick_value = peak_seconds * (1.0 - ratio)
                painter.drawLine(bar_x + bar_width, tick_y, bar_x + bar_width + tick_length, tick_y)
                painter.drawText(
                    bar_x + bar_width + tick_length + tick_gap,
                    tick_y + max(4, round(4 * information_scale)),
                    f"{tick_value:.{decimals}f}",
                )
            painter.save()
            axis_inset = max(
                12,
                round((self.colorbar_text_size.value() + 8) * information_scale),
            )
            painter.translate(render_width - axis_inset, bar_y + bar_height // 2)
            painter.rotate(-90)
            axis_font = QFont("Arial")
            axis_font.setPixelSize(max(8, round(self.colorbar_text_size.value() * information_scale)))
            painter.setFont(axis_font)
            axis_text = "Dwell time (s / spatial bin)"
            axis_width = painter.fontMetrics().horizontalAdvance(axis_text)
            painter.drawText(-axis_width // 2, 0, axis_text)
            painter.restore()
        if show_information:
            summary = data["summary"]
            painter.setPen(self.color_qcolor("title_text"))
            title_font = QFont("Malgun Gothic"); title_font.setPixelSize(max(8, round(self.title_text_size.value() * information_scale))); title_font.setBold(True); painter.setFont(title_font)
            inset = max(8, round(16 * information_scale))
            title_baseline = max(10, round((self.title_text_size.value() + 7) * information_scale))
            painter.drawText(inset, title_baseline, self.result_title_text(data))
            label_font = QFont("Malgun Gothic"); label_font.setPixelSize(max(8, round(self.panel_text_size.value() * information_scale))); label_font.setBold(False); painter.setFont(label_font)
            painter.setPen(self.color_qcolor("panel_text"))
            panel_baseline = max(
                title_baseline + 8,
                round((self.title_text_size.value() + self.panel_text_size.value() + 14) * information_scale),
            )
            for label, position, _panel_width, _top, _height in positions:
                painter.drawText(position + inset, panel_baseline, label)
            painter.setPen(self.color_qcolor("metric_text"))
            metric_font = QFont("Malgun Gothic")
            metric_font.setPixelSize(max(8, round(self.metric_text_size.value() * information_scale)))
            painter.setFont(metric_font)
            painter.drawText(inset, image.height() - max(7, round(13 * information_scale)), self.metric_text(summary))
        painter.end()
        return image

    def schedule_render(self, *_args) -> None:
        self.render_debounce.start()

    def render_current(self, *_args) -> None:
        self.render_debounce.stop()
        try:
            self.current_result = self.compose_image(self.current_index, 3840)
        except Exception as error:
            self.current_result = None
            self.preview.clear()
            self.preview.setText(tr('결과 이미지를 만들 수 없습니다.\n{p0}', p0=error))
            self.preview_status.setText(str(error))
            return
        self.update_preview_pixmap()
        data = self.entry_data(self.current_index)
        self.update_wall_layer_controls(data)
        self.preview_status_base = (
            tr('{p0}  |  고해상도 미리보기 {p1}×{p2}px{p3}', p0=self.preview_status_text(data), p1=self.current_result.width(), p2=self.current_result.height(), p3=tr('  ·  투명 배경') if not self.component_checks['background'].isChecked() and (not self.component_checks['background_color'].isChecked()) else '')
        )
        if "wall" in self.component_checks and not data.get("wall_zone", {}).get("available"):
            self.preview_status_base += "  ·  " + data.get("wall_zone", {}).get("detail", tr('벽면 구역 기준 없음'))
        self.refresh_preview_status()

    def refresh_preview_status(self) -> None:
        if self.preview_status_base:
            self.preview_status.setText(
                tr('{p0}  ·  화면 확대 {p1}%', p0=self.preview_status_base, p1=round(self.preview_zoom * 100))
            )

    def adjust_preview_zoom(self, direction: int, position: QPointF) -> None:
        old_zoom = self.preview_zoom
        factor = 1.18 if direction > 0 else 1 / 1.18
        new_zoom = min(max(old_zoom * factor, 1.0), 6.0)
        if abs(new_zoom - old_zoom) < 1e-6:
            return
        horizontal = self.preview_scroll.horizontalScrollBar()
        vertical = self.preview_scroll.verticalScrollBar()
        viewport_point = QPointF(position.x() - horizontal.value(), position.y() - vertical.value())
        ratio = new_zoom / old_zoom
        self.preview_zoom = new_zoom
        self.update_preview_pixmap()
        horizontal.setValue(round(position.x() * ratio - viewport_point.x()))
        vertical.setValue(round(position.y() * ratio - viewport_point.y()))
        self.refresh_preview_status()

    def update_preview_pixmap(self) -> None:
        if self.current_result is not None:
            viewport = self.preview_scroll.viewport().size()
            image_width = max(self.current_result.width(), 1)
            image_height = max(self.current_result.height(), 1)
            fit_scale = min(
                max(viewport.width() - 4, 1) / image_width,
                max(viewport.height() - 4, 1) / image_height,
            )
            display_width = max(1, round(image_width * fit_scale * self.preview_zoom))
            display_height = max(1, round(image_height * fit_scale * self.preview_zoom))
            display_size = QSize(display_width, display_height)
            self.preview.resize(display_size)
            result = QPixmap.fromImage(self.current_result).scaled(
                display_size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation,
            )
            if self.component_checks["background_color"].isChecked():
                self.preview.setPixmap(result)
                return
            checker = QPixmap(display_size)
            checker.fill(QColor("#303239"))
            painter = QPainter(checker)
            tile = 16
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#24262b"))
            for y_value in range(0, checker.height(), tile):
                for x_value in range(0, checker.width(), tile):
                    if (x_value // tile + y_value // tile) % 2:
                        painter.drawRect(x_value, y_value, tile, tile)
            painter.drawPixmap(0, 0, result)
            painter.end()
            self.preview.setPixmap(checker)

    def default_output_path(self, index: int) -> Path:
        entry = self.entries[index]
        visual = Path(str((entry.get("summary") or {}).get("visual_file", "")))
        if visual.name:
            return visual.with_name(f"{visual.stem}_custom.png")
        return Path(entry["frames"]).with_name(f"{Path(entry['frames']).stem}_custom.png")

    def save_current(self) -> None:
        if self.current_result is None:
            return
        output_width = self.choose_export_width()
        if output_width is None:
            return
        suggested = self.default_output_path(self.current_index)
        path, _ = QFileDialog.getSaveFileName(
            self, tr('현재 결과 이미지 저장'), str(suggested), tr('PNG 이미지 (*.png)'),
        )
        if not path:
            return
        target = Path(path)
        if target.suffix.lower() != ".png":
            target = target.with_suffix(".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        output_image = self.compose_image(self.current_index, output_width)
        if not output_image.save(str(target), "PNG"):
            QMessageBox.warning(self, tr('이미지 저장 실패'), str(target))
            return
        self.preview_status.setText(tr('저장 완료 · {p0}', p0=target))

    def save_checked(self) -> None:
        indices = [
            index for index in range(self.playlist_widget.count())
            if self.playlist_widget.item(index).checkState() == Qt.Checked
        ]
        if not indices:
            QMessageBox.information(self, tr('일괄 저장'), tr('왼쪽 목록에서 저장할 항목을 체크하세요.'))
            return
        output_width = self.choose_export_width()
        if output_width is None:
            return
        default_folder = self.default_output_path(indices[0]).parent
        folder = QFileDialog.getExistingDirectory(self, tr('체크 항목 저장 폴더'), str(default_folder))
        if not folder:
            return
        output = Path(folder)
        output.mkdir(parents=True, exist_ok=True)
        failures = []
        for order, index in enumerate(indices, 1):
            QApplication.processEvents()
            try:
                image = self.compose_image(index, output_width)
                target = output / self.default_output_path(index).name
                if not image.save(str(target), "PNG"):
                    raise ValueError(tr('PNG 저장 실패'))
                self.preview_status.setText(tr('일괄 저장 {p0}/{p1} · {p2}', p0=order, p1=len(indices), p2=target.name))
            except Exception as error:
                failures.append(f"{index + 1}: {error}")
        if failures:
            QMessageBox.warning(self, tr('일괄 저장 일부 실패'), "\n".join(failures))
        else:
            QMessageBox.information(self, tr('일괄 저장 완료'), tr('{p0}개 이미지를 저장했습니다.\n{p1}', p0=len(indices), p1=output))


class NortVisualComposer(WaterMazeVisualComposer):
    """Water-maze result composer adapted to NORT Arena/Object geometry."""

    EXPERIMENT_KIND = "nort"

    def __init__(self, entries: list[dict], current_index: int = 0, parent=None) -> None:
        normalized: list[dict] = []
        for source in entries:
            entry = dict(source)
            summary = dict(entry.get("summary") or {})
            summary.setdefault("arena_id", summary.get("arena", 1))
            summary.setdefault("trial", 1)
            summary.setdefault("trial_type", "NORT")
            entry["summary"] = summary
            normalized.append(entry)
        super().__init__(normalized, current_index, parent)
        self.setWindowTitle(tr('HAGISILTA · NORT 궤적·히트맵 편집'))
        self.render_current()

    def style_label(self, key: str | None, default: str) -> str:
        return {
            "pool": tr('Arena 경계'),
            "pool_fill": tr('Arena 내부 채우기'),
            "quadrants": tr('탐색 판정 영역·역할'),
            "platform": tr('실제 오브젝트'),
            "quadrant_text": tr('역할 이름'),
            "quadrant_line": tr('탐색 판정 영역'),
            "trajectory_fill": tr('궤적 Arena 내부'),
            "heatmap_fill": tr('히트맵 Arena 내부'),
        }.get(key, default)

    def playlist_item_text(self, entry: dict, summary: dict) -> str:
        bundle = str(summary.get("video_bundle") or Path(entry.get("video", "")).stem)
        arena = self._integer(summary.get("arena_id"), 1)
        subject = str(summary.get("subject_id") or tr('개체 미지정'))
        return f"{bundle}\nArena {arena:02d}  ·  {subject}"

    def result_title_text(self, data: dict) -> str:
        summary = data["summary"]
        bundle = str(summary.get("video_bundle") or Path(data["entry"]["video"]).stem)
        arena = self._integer(summary.get("arena_id"), 1)
        subject = str(summary.get("subject_id") or tr('개체 미지정'))
        return f"{bundle} · Arena {arena:02d} · {subject} · NORT"

    def preview_status_text(self, data: dict) -> str:
        summary = data["summary"]
        return (
            f"{summary.get('video_bundle', '-')} · Arena "
            f"{self._integer(summary.get('arena_id'), 1):02d} · {summary.get('subject_id', '-')}"
        )

    def entry_data(self, index: int) -> dict:
        if index in self.data_cache:
            return self.data_cache[index]
        entry = self.entries[index]
        frames = pd.read_csv(Path(entry["frames"]))
        setup = json.loads(Path(entry["setup"]).read_text(encoding="utf-8"))
        if frames.empty:
            raise ValueError(tr('프레임 판정 CSV가 비어 있습니다.'))
        capture = cv2.VideoCapture(str(entry["video"]))
        video_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        source_frame = self._integer(frames.iloc[0].get("frame"), 0)
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(source_frame, 0))
        ok, background = capture.read()
        capture.release()
        if not ok:
            video_info = setup.get("video") or {}
            background = np.full(
                (int(video_info.get("height", 480)), int(video_info.get("width", 640)), 3),
                35, dtype=np.uint8,
            )
        x_values = pd.to_numeric(frames.get("body_center_x"), errors="coerce").to_numpy(dtype=float)
        y_values = pd.to_numeric(frames.get("body_center_y"), errors="coerce").to_numpy(dtype=float)
        valid_source = frames.get("body_pose_valid", frames.get("valid_pose", pd.Series(True, index=frames.index)))
        valid = valid_source.astype(str).str.lower().isin({"1", "true", "yes"}).to_numpy()
        valid &= np.isfinite(x_values) & np.isfinite(y_values)
        height, width = background.shape[:2]
        arena = (setup.get("arenas") or [{}])[0]
        arena_polygon = np.asarray(arena.get("corners_source_px") or [], dtype=float)
        if len(arena_polygon) < 3:
            arena_polygon = np.asarray([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=float)
        arena_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(arena_mask, [np.rint(arena_polygon).astype(np.int32)], 1)
        counts = np.zeros((height, width), dtype=np.float32)
        if valid.any():
            px = np.clip(np.rint(x_values[valid]).astype(int), 0, width - 1)
            py = np.clip(np.rint(y_values[valid]).astype(int), 0, height - 1)
            np.add.at(counts, (py, px), 1.0)
        counts[arena_mask == 0] = 0.0
        sigma = max(5, round(min(width, height) * .025))
        blurred = cv2.GaussianBlur(counts, (sigma * 2 + 1, sigma * 2 + 1), 0)
        blurred[arena_mask == 0] = 0.0
        peak = float(blurred.max())
        heat_gray = np.clip(blurred / peak * 255, 0, 255).astype(np.uint8) if peak > 0 else np.zeros_like(blurred, dtype=np.uint8)
        data = {
            "entry": entry, "summary": dict(entry.get("summary") or {}), "frames": frames,
            "setup": setup, "arena": arena, "background": background,
            "x": x_values, "y": y_values, "valid": valid,
            "heat_gray": heat_gray, "pool_mask": arena_mask.astype(bool),
            "video_fps": video_fps, "heat_peak_seconds": peak / max(video_fps, 1e-9),
            "spatial_cache": {}, "background_cache": {},
        }
        self.data_cache[index] = data
        return data

    def spatial_layers(self, data: dict, width: int, height: int):
        cache = data.setdefault("spatial_cache", {})
        key = (int(width), int(height))
        if key in cache:
            return cache[key]
        source_height, source_width = data["background"].shape[:2]
        scale_x, scale_y = width / max(source_width, 1), height / max(source_height, 1)
        polygon = np.asarray(data["arena"].get("corners_source_px") or [], dtype=float)
        if len(polygon) < 3:
            polygon = np.asarray([[0, 0], [source_width - 1, 0], [source_width - 1, source_height - 1], [0, source_height - 1]], dtype=float)
        polygon[:, 0] *= scale_x; polygon[:, 1] *= scale_y
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
        mask = mask.astype(bool)
        counts = np.zeros((height, width), dtype=np.float32)
        valid = data["valid"]
        if valid.any():
            px = np.clip(np.rint(data["x"][valid] * scale_x).astype(int), 0, width - 1)
            py = np.clip(np.rint(data["y"][valid] * scale_y).astype(int), 0, height - 1)
            np.add.at(counts, (py, px), 1.0)
        counts[~mask] = 0.0
        sigma = max(5, round(min(width, height) * .025))
        blurred = cv2.GaussianBlur(counts, (sigma * 2 + 1, sigma * 2 + 1), 0)
        blurred[~mask] = 0.0
        peak = float(blurred.max())
        heat_gray = np.clip(blurred / peak * 255, 0, 255).astype(np.uint8) if peak > 0 else np.zeros_like(blurred, dtype=np.uint8)
        result = (heat_gray, mask, scale_x, scale_y)
        if len(cache) >= 2:
            cache.clear()
        cache[key] = result
        return result

    def draw_geometry(self, image: np.ndarray, data: dict, draw_trail: bool,
                      scale_x: float = 1.0, scale_y: float = 1.0,
                      light_surface: bool = False) -> None:
        height, width = image.shape[:2]
        scale = max(.75, min(width / 640.0, height / 480.0))
        summary = data["summary"]
        prepared: list[tuple[str, str, np.ndarray, np.ndarray]] = []
        for label, payload in (data["arena"].get("objects") or {}).items():
            key = str(label).lower()
            role = str(summary.get(f"object_{key}_role") or f"Object {label}")
            object_polygon = np.asarray(payload.get("polygon_source_px") or [], dtype=float)
            explore_polygon = np.asarray(payload.get("exploration_boundary_source_px") or payload.get("polygon_source_px") or [], dtype=float)
            if len(object_polygon) >= 3:
                object_polygon[:, 0] *= scale_x; object_polygon[:, 1] *= scale_y
            if len(explore_polygon) >= 3:
                explore_polygon[:, 0] *= scale_x; explore_polygon[:, 1] *= scale_y
            prepared.append((str(label), role, object_polygon, explore_polygon))
        # Keep the two spatial meanings distinct in trajectory and heatmap
        # panels: pale outer exploration range, stronger inner object body.
        if self.component_checks["quadrants"].isChecked():
            exploration_overlay = image.copy(); has_exploration = False
            exploration_color = self.surface_color_bgr("quadrant_line", light_surface)
            for _label, _role, _object_polygon, explore_polygon in prepared:
                if len(explore_polygon) >= 3:
                    cv2.fillPoly(exploration_overlay, [np.rint(explore_polygon).astype(np.int32)], exploration_color, cv2.LINE_AA)
                    has_exploration = True
            if has_exploration:
                cv2.addWeighted(exploration_overlay, .12, image, .88, 0, image)
        if self.component_checks["platform"].isChecked():
            object_overlay = image.copy(); has_object = False
            object_color = self.color_bgr("platform")
            for _label, _role, object_polygon, _explore_polygon in prepared:
                if len(object_polygon) >= 3:
                    cv2.fillPoly(object_overlay, [np.rint(object_polygon).astype(np.int32)], object_color, cv2.LINE_AA)
                    has_object = True
            if has_object:
                cv2.addWeighted(object_overlay, .32, image, .68, 0, image)
        if draw_trail and self.component_checks["trail"].isChecked():
            valid = data["valid"]
            points = np.column_stack((data["x"][valid] * scale_x, data["y"][valid] * scale_y))
            if len(points) >= 2:
                cv2.polylines(image, [np.rint(points).astype(np.int32)], False, self.color_bgr("trail"), max(1, round(self.trail_width.value() * scale)), cv2.LINE_AA)
        arena_polygon = np.asarray(data["arena"].get("corners_source_px") or [], dtype=float)
        if len(arena_polygon) >= 3 and self.component_checks["pool"].isChecked():
            arena_polygon[:, 0] *= scale_x; arena_polygon[:, 1] *= scale_y
            cv2.polylines(image, [np.rint(arena_polygon).astype(np.int32)], True, self.color_bgr("pool"), max(1, round(self.pool_line_width.value() * scale)), cv2.LINE_AA)
        for _label, _role, object_polygon, explore_polygon in prepared:
            if self.component_checks["platform"].isChecked() and len(object_polygon) >= 3:
                cv2.polylines(image, [np.rint(object_polygon).astype(np.int32)], True, self.color_bgr("platform"), max(1, round(self.platform_line_width.value() * scale)), cv2.LINE_AA)
            if self.component_checks["quadrants"].isChecked() and len(explore_polygon) >= 3:
                cv2.polylines(image, [np.rint(explore_polygon).astype(np.int32)], True, self.surface_color_bgr("quadrant_line", light_surface), max(1, round(self.quadrant_line_width.value() * scale)), cv2.LINE_AA)
        if self.component_checks["quadrants"].isChecked():
            font_scale = self.quadrant_text_size.value() / 30.0 * scale
            weight = max(1, round(self.quadrant_text_size.value() / 14.0 * scale))
            color = self.surface_color_bgr("quadrant_text", light_surface)
            for _label, role, object_polygon, explore_polygon in prepared:
                polygon = object_polygon if len(object_polygon) >= 3 else explore_polygon
                if len(polygon) < 3:
                    continue
                center = polygon.mean(axis=0)
                text = role
                size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, weight)
                cv2.putText(image, text, (round(center[0] - size[0] / 2), round(center[1] + (size[1] - baseline) / 2)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, weight, cv2.LINE_AA)

    def metric_text(self, summary: dict) -> str:
        novel = self._number(summary.get("novel_seconds"))
        familiar = self._number(summary.get("familiar_seconds"))
        di = self._number(summary.get("discrimination_index"))
        valid = self._number(summary.get("valid_pose_percent"))
        return (
            tr('Novel 탐색 {p0:.2f}초 · Familiar 탐색 {p1:.2f}초 · 변별지수 {p2:.3f} · 유효 좌표 {p3:.1f}%', p0=novel, p1=familiar, p2=di, p3=valid)
            if None not in (novel, familiar, di, valid) else tr('NORT 주요 결과 없음')
        )

    def preview_layer_at(self, x_value: float, y_value: float) -> str:
        if self.current_result is None:
            return ""
        x_ratio = x_value / max(self.current_result.width(), 1)
        y_ratio = y_value / max(self.current_result.height(), 1)
        if y_ratio < .13:
            return "title_text"
        if y_ratio > .90:
            return "metric_text"
        if x_ratio > .91 and self.component_checks["heatmap"].isChecked():
            return "colorbar_text"
        compare = self.layout_combo.currentData() == "compare"
        if compare and x_ratio > .48:
            return "heatmap_fill"
        return "trail" if self.component_checks["trail"].isChecked() else "trajectory_fill"

    def default_output_path(self, index: int) -> Path:
        frames = Path(self.entries[index]["frames"])
        return frames.with_name(f"{frames.stem}_nort_custom.png")


class YMazeVisualComposer(WaterMazeVisualComposer):
    """Water-maze parity composer adapted to polygonal Y-maze geometry."""

    EXPERIMENT_KIND = "ymaze"

    def __init__(self, entries: list[dict], current_index: int = 0, parent=None) -> None:
        normalized: list[dict] = []
        for source in entries:
            entry = dict(source)
            summary = dict(entry.get("summary") or {})
            summary.setdefault("arena_id", summary.get("arena", 1))
            summary.setdefault("trial", 1)
            summary.setdefault("trial_type", "자발적 교대")
            entry["summary"] = summary
            normalized.append(entry)
        super().__init__(normalized, current_index, parent)
        self.setWindowTitle(tr('HAGISILTA · Y-maze 궤적·히트맵 편집'))

        self.render_current()

    def style_label(self, key: str | None, default: str) -> str:
        return {
            "pool": tr('Maze 외곽'),
            "pool_fill": tr('구역 내부 채우기'),
            "quadrants": tr('A/B/C·Center 경계·이름'),
            "platform": "Center",
            "quadrant_text": tr('구역 이름'),
            "quadrant_line": tr('구역'),
            "trajectory_fill": tr('궤적 구역 내부'),
            "heatmap_fill": tr('히트맵 구역 내부'),
        }.get(key, default)

    def playlist_item_text(self, entry: dict, summary: dict) -> str:
        bundle = str(summary.get("video_bundle") or Path(entry.get("video", "")).stem)
        arena = self._integer(summary.get("arena_id"), 1)
        return tr("{bundle}\nArena {arena:02d} · 자발적 교대", bundle=bundle, arena=arena)

    def result_title_text(self, data: dict) -> str:
        summary = data["summary"]
        bundle = str(summary.get("video_bundle") or Path(data["entry"]["video"]).stem)
        arena = self._integer(summary.get("arena_id"), 1)
        return tr("{bundle} · Arena {arena:02d} · 자발적 교대", bundle=bundle, arena=arena)

    def entry_data(self, index: int) -> dict:
        if index in self.data_cache:
            return self.data_cache[index]
        entry = self.entries[index]
        frames = pd.read_csv(Path(entry["frames"]))
        if frames.empty:
            raise ValueError(tr('프레임 판정 CSV가 비어 있습니다.'))
        capture = cv2.VideoCapture(str(entry["video"]))
        video_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        source_frame = self._integer(frames.iloc[0].get("frame"), 0)
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(source_frame, 0))
        ok, background = capture.read()
        capture.release()
        if not ok:
            background = np.full((480, 640, 3), 35, dtype=np.uint8)
        x_values = pd.to_numeric(frames.get("x"), errors="coerce").to_numpy(dtype=float)
        y_values = pd.to_numeric(frames.get("y"), errors="coerce").to_numpy(dtype=float)
        valid_series = frames.get(
            "valid_pose", frames.get("valid", pd.Series(True, index=frames.index))
        )
        valid = valid_series.astype(str).str.lower().isin({"1", "true", "yes"}).to_numpy()
        valid &= np.isfinite(x_values) & np.isfinite(y_values)
        height, width = background.shape[:2]
        maze_mask = np.zeros((height, width), dtype=np.uint8)
        zones = list(entry.get("zones") or [])
        for zone in zones:
            polygon = np.asarray(zone.get("polygon") or [], dtype=float)
            if len(polygon) >= 3:
                cv2.fillPoly(maze_mask, [np.rint(polygon).astype(np.int32)], 1)
        counts = np.zeros((height, width), dtype=np.float32)
        if valid.any():
            px = np.clip(np.rint(x_values[valid]).astype(int), 0, width - 1)
            py = np.clip(np.rint(y_values[valid]).astype(int), 0, height - 1)
            np.add.at(counts, (py, px), 1.0)
        if np.any(maze_mask):
            counts[maze_mask == 0] = 0.0
        sigma = max(5, round(min(width, height) * .025))
        blurred = cv2.GaussianBlur(counts, (sigma * 2 + 1, sigma * 2 + 1), 0)
        if np.any(maze_mask):
            blurred[maze_mask == 0] = 0.0
        peak = float(blurred.max())
        heat_gray = (
            np.clip(blurred / peak * 255, 0, 255).astype(np.uint8)
            if peak > 0 else np.zeros_like(blurred, dtype=np.uint8)
        )
        summary = dict(entry.get("summary") or {})
        data = {
            "entry": entry, "summary": summary, "frames": frames,
            "background": background, "x": x_values, "y": y_values,
            "valid": valid, "heat_gray": heat_gray, "pool_mask": maze_mask.astype(bool),
            "zones": zones, "video_fps": video_fps,
            "heat_peak_seconds": peak / max(video_fps, 1e-9),
            "spatial_cache": {}, "background_cache": {},
        }
        self.data_cache[index] = data
        return data

    def spatial_layers(self, data: dict, width: int, height: int):
        cache = data.setdefault("spatial_cache", {})
        key = (int(width), int(height))
        if key in cache:
            return cache[key]
        source_height, source_width = data["background"].shape[:2]
        scale_x, scale_y = width / max(source_width, 1), height / max(source_height, 1)
        mask = np.zeros((height, width), dtype=np.uint8)
        for zone in data["zones"]:
            polygon = np.asarray(zone.get("polygon") or [], dtype=float)
            if len(polygon) >= 3:
                polygon[:, 0] *= scale_x
                polygon[:, 1] *= scale_y
                cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
        mask = mask.astype(bool)
        counts = np.zeros((height, width), dtype=np.float32)
        valid = data["valid"]
        if valid.any():
            px = np.clip(np.rint(data["x"][valid] * scale_x).astype(int), 0, width - 1)
            py = np.clip(np.rint(data["y"][valid] * scale_y).astype(int), 0, height - 1)
            np.add.at(counts, (py, px), 1.0)
        if np.any(mask):
            counts[~mask] = 0.0
        sigma = max(5, round(min(width, height) * .025))
        blurred = cv2.GaussianBlur(counts, (sigma * 2 + 1, sigma * 2 + 1), 0)
        if np.any(mask):
            blurred[~mask] = 0.0
        peak = float(blurred.max())
        heat_gray = (
            np.clip(blurred / peak * 255, 0, 255).astype(np.uint8)
            if peak > 0 else np.zeros_like(blurred, dtype=np.uint8)
        )
        result = (heat_gray, mask, scale_x, scale_y)
        if len(cache) >= 2:
            cache.clear()
        cache[key] = result
        return result

    def draw_geometry(
        self, image: np.ndarray, data: dict, draw_trail: bool,
        scale_x: float = 1.0, scale_y: float = 1.0,
        light_surface: bool = False,
    ) -> None:
        height, width = image.shape[:2]
        scale = max(.75, min(width / 640.0, height / 480.0))
        line_color = self.surface_color_bgr("quadrant_line", light_surface)
        text_color = self.surface_color_bgr("quadrant_text", light_surface)
        center_color = self.color_bgr("platform")
        outer_color = self.color_bgr("pool")
        prepared_zones: list[tuple[str, str, np.ndarray, np.ndarray]] = []
        for zone in data["zones"]:
            polygon = np.asarray(zone.get("polygon") or [], dtype=float)
            if len(polygon) < 3:
                continue
            polygon[:, 0] *= scale_x
            polygon[:, 1] *= scale_y
            points = np.rint(polygon).astype(np.int32)
            zone_id = str(zone.get("id", ""))
            label = str(zone.get("label") or zone.get("name") or zone_id.replace("arm_", "").upper())
            prepared_zones.append((zone_id, label, polygon, points))

        # Draw data first and structural lines second. Labels are intentionally
        # rendered last so dense trails can never cover A/B/C/Center names.
        if draw_trail and self.component_checks["trail"].isChecked():
            valid = data["valid"]
            points = np.column_stack((data["x"][valid] * scale_x, data["y"][valid] * scale_y))
            if len(points) >= 2:
                cv2.polylines(image, [np.rint(points).astype(np.int32)], False, self.color_bgr("trail"), max(1, round(self.trail_width.value() * scale)), cv2.LINE_AA)

        for zone_id, _label, _polygon, points in prepared_zones:
            if zone_id == "center":
                if self.component_checks["platform"].isChecked():
                    cv2.polylines(image, [points], True, center_color, max(1, round(self.platform_line_width.value() * scale)), cv2.LINE_AA)
            elif self.component_checks["quadrants"].isChecked():
                cv2.polylines(image, [points], True, line_color, max(1, round(self.quadrant_line_width.value() * scale)), cv2.LINE_AA)
        if self.component_checks["pool"].isChecked() and prepared_zones:
            union = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(union, [zone[3] for zone in prepared_zones], 1)
            contours, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, outer_color, max(1, round(self.pool_line_width.value() * scale)), cv2.LINE_AA)
        if self.component_checks["quadrants"].isChecked():
            font_scale = self.quadrant_text_size.value() / 30.0 * scale
            weight = max(1, round(self.quadrant_text_size.value() / 14.0 * scale))
            for _zone_id, label, polygon, _points in prepared_zones:
                centroid = polygon.mean(axis=0)
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, weight,
                )
                origin = (
                    round(float(centroid[0]) - text_width / 2),
                    round(float(centroid[1]) + (text_height - baseline) / 2),
                )
                cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, weight, cv2.LINE_AA)

    def metric_text(self, summary: dict) -> str:
        percent = self._number(summary.get("alternation_percent"))
        successes = self._number(summary.get("alternation_successes"))
        opportunities = self._number(summary.get("alternation_opportunities"))
        entries = self._number(summary.get("total_entries"))
        text = tr('자발적 교대 {p0:.1f}%', p0=percent) if percent is not None else tr('자발적 교대 -')
        if successes is not None and opportunities is not None:
            text += tr(' · 교대 성공 {p0:d}/{p1:d}', p0=round(successes), p1=round(opportunities))
        if entries is not None:
            text += tr(' · 총 진입 {p0:d}회', p0=round(entries))
        return text

    def preview_layer_at(self, x_value: float, y_value: float) -> str:
        if self.current_result is None:
            return ""
        x_ratio = x_value / max(self.current_result.width(), 1)
        y_ratio = y_value / max(self.current_result.height(), 1)
        if y_ratio < .13:
            return "title_text"
        if y_ratio > .90:
            return "metric_text"
        if x_ratio > .91 and self.component_checks["heatmap"].isChecked():
            return "colorbar_text"
        compare = self.layout_combo.currentData() == "compare"
        if compare and x_ratio > .48:
            return "heatmap_fill"
        return "trail" if self.component_checks["trail"].isChecked() else "trajectory_fill"

    def default_output_path(self, index: int) -> Path:
        frames = Path(self.entries[index]["frames"])
        return frames.with_name(f"{frames.stem}_ymaze_custom.png")
