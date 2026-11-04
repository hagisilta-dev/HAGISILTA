# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
"""Frame-locked comparison and non-destructive per-bodypart filter composition."""
from __future__ import annotations

from hagisilta_i18n import init_i18n, tr

import argparse
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np
from PySide6.QtCore import QTimer, Signal, Qt
from PySide6.QtGui import QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMenu,
    QMessageBox, QPushButton, QSizePolicy, QSplitter, QTabWidget, QToolButton, QVBoxLayout, QWidget,
)

from hagisilta_atomic_io import atomic_json
from hagisilta_pose_postprocess import (
    WINDOWS, load_comparison_family, load_comparison_entry, median_predictions, window_description,
)
from hagisilta_pose_postprocess import (
    DEFAULT_RECIPE_NAME, compose_predictions, notify_compositions, read_recipe,
    recipe_display_name, save_recipe_preset, validate_recipe,
)
from hagisilta_playback import PLAYBACK_HELP
from hagisilta_video_review import VideoCanvas, VideoReviewPilot, pose_column, review_playlist_caption
from hagisilta_window_chrome import (
    HeaderContext,
    create_hagisilta_header, create_hagisilta_help_button, create_hagisilta_settings_section,
    install_wheel_value_guard,
)


class PoseRecipeMixin:
    def init_recipe_support(self, output_root, recipe_manifest, recipe_result=None, initial_window=0):
        self.recipe_editor = recipe_result is not None
        if self.recipe_editor and output_root is None:
            raise ValueError(tr('보정 설정 저장 폴더가 필요합니다.'))
        self.composition_root = output_root if self.recipe_editor else None
        self.initial_recipe_manifest = recipe_manifest
        self.recipe_result = recipe_result
        self.initial_window = initial_window
        self.recipe_stage = 0  # Compare ranges first, then assemble a recipe.
        self.recipe, self.recipe_name_value = {}, DEFAULT_RECIPE_NAME
        self.accepted_recipe = False
        self.preset_path = None

    def recipe_candidate(self, entry, record, original, initial=False):
        parts = list(dict.fromkeys(original.columns.get_level_values(1)))
        recipe, name = {part: self.initial_window for part in parts}, DEFAULT_RECIPE_NAME
        if self.recipe_editor:
            if not initial and self.recipe:
                # One reusable recipe across preview videos, not per-video settings.
                recipe, name = self.recipe, self.recipe_name_value
            elif self.initial_recipe_manifest:
                recipe, name = read_recipe(self.initial_recipe_manifest)
        elif entry.get("composition"):
            recipe, name = read_recipe(Path(entry["composition"]))
        recipe = validate_recipe(recipe, parts)
        recipe = {part: recipe[part] for part in parts}
        composed, _ = compose_predictions(original, recipe)
        return recipe, name, composed

    def prepare_recipe(self, entry, initial=False, candidate=None):
        if not self.recipe_editor and not entry.get("composition"):
            self.MAX_FILTERS = 3
            return
        self.MAX_FILTERS = 3 if self.recipe_editor and self.recipe_stage == 0 else 2
        recipe, name, composed = candidate or self.recipe_candidate(
            entry, self.comparison, self.comparison_frames[0], initial)
        self.recipe, self.recipe_name_value = recipe, name
        if initial:
            self.initial_recipe = dict(recipe), name
        self.comparison_frames[-1] = composed
        if self.recipe_editor:
            for size in sorted({3, 5, 7} | getattr(self, "visible_filters", set())):
                if size <= len(self.comparison_frames[0]):
                    self.ensure_preview_filter(size)

    def ensure_preview_filter(self, size):
        """Calculate from this video's baseline in memory; never save a result."""
        if size in self.variants:
            return False
        filtered, _summary = median_predictions(self.comparison_frames[0], size)
        self.variants[size] = {"window": size, "preview": True, "pose": filtered}
        self.comparison_frames[size] = filtered
        return True

    def build_range_controls(self):
        page = QWidget(objectName="filterInspector")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(12)
        section, body = create_hagisilta_settings_section(tr('비교할 중앙값 범위'))
        section.setToolTip(tr('보정 전과 최대 3개 범위를 같은 프레임에서 비교합니다. 저장된 보정본 없이 원본 좌표로 미리 계산합니다. 중앙값 범위는 3~31 홀수 프레임입니다.'))
        layout.addWidget(section)
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        self.range_checks = {}
        for index, size in enumerate(WINDOWS):
            check = QCheckBox(tr('{v0}프레임', v0=size))
            check.setChecked(size in self.visible_filters)
            check.toggled.connect(lambda checked, value=size: self.set_filter_visible(value, checked))
            grid.addWidget(check, index // 3, index % 3)
            self.range_checks[size] = check
        body.addLayout(grid)
        next_button = QPushButton(tr('신체점별 조합으로'), objectName="toolButton")
        next_button.clicked.connect(lambda: self.recipe_tabs.setCurrentIndex(1))
        next_button.setToolTip(tr('비교한 범위를 바탕으로 점마다 보정 범위를 선택하고 조합 결과를 미리 봅니다.'))
        body.addWidget(next_button)
        layout.addStretch(1)
        return page

    def set_recipe_stage(self, index):
        self.filter_picker.setVisible(index != 0)
        if index not in (0, 1):
            return
        self.recipe_stage = index
        self.MAX_FILTERS = 3 if index == 0 else 2
        self.visible_filters = set(sorted(self.visible_filters)[:self.MAX_FILTERS])
        self.recipe_use_button.setEnabled(index == 1)
        self.reflow_views()
        self.draw_pose()
        self.update_labels()

    def sync_range_controls(self):
        for size, check in getattr(self, "range_checks", {}).items():
            check.blockSignals(True)
            check.setChecked(size in self.visible_filters)
            check.setEnabled(size <= len(self.comparison_frames[0]) and (
                size in self.visible_filters or len(self.visible_filters) < self.MAX_FILTERS))
            check.setToolTip(window_description(size, self.fps))
            check.blockSignals(False)

    def build_recipe_controls(self):
        page = QWidget(objectName="filterInspector")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(10)
        section, body = create_hagisilta_settings_section(tr('신체점별 보정 범위'))
        section.setToolTip(tr('같은 신체점 구성을 가진 다른 영상에도 재사용하는 설정입니다. 영상 전환 시 같은 조합으로 미리 봅니다. 실제 보정은 메인 화면 2단계에서 실행합니다.'))
        layout.addWidget(section)
        body.addWidget(QLabel(tr('조합 이름'), objectName="settingsGroupDescription"))
        self.recipe_name = QLineEdit(recipe_display_name(self.recipe_name_value))
        self.recipe_name.setPlaceholderText(tr('조합 이름'))
        self.recipe_name.setMaxLength(120)
        self.recipe_name.textChanged.connect(self.recipe_name_changed)
        body.addWidget(self.recipe_name)
        self.recipe_fields = QWidget(objectName="recipeFields")
        self.recipe_grid = QGridLayout(self.recipe_fields)
        self.recipe_grid.setContentsMargins(0, 0, 0, 0)
        self.recipe_grid.setColumnStretch(1, 1)
        self.recipe_grid.setVerticalSpacing(10)
        self.recipe_grid.setHorizontalSpacing(12)
        body.addWidget(self.recipe_fields)
        self.recipe_status = QLabel(objectName="pageText")
        self.recipe_status.setWordWrap(True)
        body.addWidget(self.recipe_status)
        layout.addStretch(1)
        self.rebuild_recipe_controls()
        return page

    def rebuild_recipe_controls(self):
        if not hasattr(self, "recipe_grid"):
            return
        while self.recipe_grid.count():
            widget = self.recipe_grid.takeAt(0).widget()
            if widget:
                widget.hide()
                widget.deleteLater()
        self.recipe_combos = {}
        self.recipe_name.blockSignals(True)
        self.recipe_name.setText(recipe_display_name(self.recipe_name_value))
        self.recipe_name.blockSignals(False)
        for row, (part, value) in enumerate(self.recipe.items()):
            label = QLabel(part)
            label.setToolTip(part)
            combo = QComboBox()
            combo.setMinimumWidth(136)
            combo.addItem(tr('보정 전 유지'), 0)
            for size in range(3, 32, 2):
                combo.addItem(tr('중앙값 {v0}프레임', v0=size), size)
                if size > len(self.comparison_frames[0]):
                    combo.model().item(combo.count() - 1).setEnabled(False)
            combo.setCurrentIndex(combo.findData(value))
            combo.view().setMinimumWidth(164)
            combo.currentIndexChanged.connect(lambda _index, name=part: self.recipe_changed(name))
            self.recipe_grid.addWidget(label, row, 0)
            self.recipe_grid.addWidget(combo, row, 1)
            self.recipe_combos[part] = combo
        self.update_recipe_status()

    def recipe_name_changed(self, value):
        self.recipe_name_value = value

    def recipe_changed(self, part):
        candidate = {**self.recipe, part: self.recipe_combos[part].currentData()}
        try:
            composed, _summary = compose_predictions(self.comparison_frames[0], candidate)
        except Exception as error:
            combo = self.recipe_combos[part]
            combo.blockSignals(True)
            combo.setCurrentIndex(combo.findData(self.recipe[part]))
            combo.blockSignals(False)
            QMessageBox.warning(self, tr('보정 설정 확인'), str(error))
            return
        self.recipe = candidate
        self.comparison_frames[-1] = composed
        self.update_recipe_status()
        self.select_bodypart(part)
        self.update_labels()

    def update_recipe_status(self):
        if not hasattr(self, "recipe_status"):
            return
        count = sum(bool(value) for value in self.recipe.values())
        self.recipe_status.setText(tr('미리보기 · {v0:g} FPS · 보정 {v1}점 / 유지 {v2}점', v0=self.fps, v1=count, v2=len(self.recipe) - count))
        descriptions = []
        for part, value in self.recipe.items():
            description = window_description(value, self.fps) if value else tr('보정 전 유지')
            self.recipe_combos[part].setToolTip(description)
            descriptions.append(f"{part}: {description}")
        self.recipe_status.setToolTip("\n".join(descriptions) + tr('\n다른 영상에도 같은 프레임 수를 사용하며 FPS에 따라 시간 범위가 달라집니다.'))

    def use_recipe(self):
        if not self.recipe_editor:
            return
        self.timer.stop()
        self.play_button.setText(tr('▶ 재생 (Space)'))
        try:
            recipe = validate_recipe(self.recipe, self.bodyparts)
            # No dependency on this preview video's path or coordinates.
            if self.preset_path is None or read_recipe(self.preset_path) != (recipe, self.recipe_name_value.strip() or DEFAULT_RECIPE_NAME):
                self.preset_path = save_recipe_preset(self.composition_root, recipe, self.recipe_name_value)
            atomic_json(self.recipe_result, {"preset": str(self.preset_path)})
        except Exception as error:
            QMessageBox.warning(self, tr('보정 설정 저장 실패'), str(error))
            return
        # Advisory only: the parent also consumes the result file.
        try:
            notify_compositions(self.composition_root)
        except OSError:
            pass
        self.accepted_recipe = True
        self.close()

    def closeEvent(self, event):
        dirty = self.recipe_editor and not self.accepted_recipe and (
            self.recipe != self.initial_recipe[0] or self.recipe_name_value != self.initial_recipe[1])
        if dirty and QMessageBox.question(self, tr("사용하지 않은 설정"), tr("변경한 보정 설정을 사용하지 않고 닫을까요?"),
                                         QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Cancel) != QMessageBox.Discard:
            event.ignore()
            return
        if self.recipe_editor and not self.accepted_recipe:
            try:
                atomic_json(self.recipe_result, {"cancelled": True})
            except OSError:
                pass
        super().closeEvent(event)


class ComparisonCanvas(VideoCanvas):
    """Notify sibling views after view-only zoom/pan, not point movement."""
    view_changed = Signal()

    def __init__(self) -> None:
        super().__init__(allow_point_placement=False)
        self.displayed_frame = -1
        self._view_center = None

    def publish_view(self) -> None:
        self._view_center = self.mapToScene(self.viewport().rect().center())
        self.view_changed.emit()

    def fit_to_view(self) -> None:
        super().fit_to_view()
        self.publish_view()

    def zoom_by_steps(self, steps: int, viewport_position=None) -> None:
        super().zoom_by_steps(steps, viewport_position)
        self.publish_view()

    def resizeEvent(self, event) -> None:
        # Keep the same relative zoom after changing the number of panels.
        center, steps = self._view_center, self._zoom_steps
        blocked = self.blockSignals(True)
        try:
            super().resizeEvent(event)
            if steps and not self.scene.sceneRect().isEmpty():
                VideoCanvas.fit_to_view(self)
                VideoCanvas.zoom_by_steps(self, steps)
                if center is not None:
                    self.centerOn(center)
        finally:
            self.blockSignals(blocked)
        self.publish_view()

    def mouseMoveEvent(self, event) -> None:
        panning = self._pan_active
        super().mouseMoveEvent(event)
        if panning:
            self.publish_view()


class FilterComparisonWindow(PoseRecipeMixin, VideoReviewPilot):
    MAX_FILTERS = 3

    def __init__(self, manifest: Path | None = None, *, playlist: list[dict] | None = None,
                 playlist_index: int = 0, output_root: Path | None = None,
                 recipe_manifest: Path | None = None, recipe_result: Path | None = None,
                 initial_window: int = 0) -> None:
        self.init_recipe_support(output_root, recipe_manifest, recipe_result, initial_window)
        if not playlist:
            if manifest is None:
                raise ValueError(tr('비교 결과 또는 영상 목록이 필요합니다.'))
            record, _original, _variants = load_comparison_family(manifest)
            playlist = [{"video": record["source_video"], "pose": record["source_pose"],
                         "manifest": str(manifest.resolve())}]
        if self.recipe_editor:
            # Settings always preview current raw predictions, never archived filter results.
            playlist = [{k: v for k, v in entry.items() if k not in ("manifest", "composition")}
                        for entry in playlist]
        self.comparison_entries = [dict(entry) for entry in playlist]
        playlist_index = max(0, min(playlist_index, len(playlist) - 1))
        entry = self.comparison_entries[playlist_index]
        self.comparison, original, self.variants, pose_path = load_comparison_entry(entry)
        self.manifest = Path(entry["manifest"]).resolve() if entry.get("manifest") else None
        self.comparison_frames = {0: original, **{key: item["pose"] for key, item in self.variants.items()}}
        self.prepare_recipe(entry, initial=True)
        self.visible_filters = self.default_filters()
        self.views = {}
        self._switching_comparison = False
        self._syncing_views = False
        super().__init__(Path(self.comparison["source_video"]), pose_path,
                         None, "HAGISILTA", playlist=[dict(row) for row in playlist],
                         playlist_index=playlist_index, review_only=True)
        self.finish_comparison_switch()

    def default_filters(self) -> set[int]:
        if self.recipe_editor:
            return set(key for key in (3, 5, 7) if key in self.variants)
        selected = self.comparison.get("window")
        keys = ([selected] if selected in self.variants else []) + list(self.variants)
        return set(list(dict.fromkeys(keys))[:1 if self.composition_root else 2])

    def finish_comparison_switch(self) -> None:
        self.frame_count = min(self.frame_count, len(self.comparison_frames[0]))
        self.pose_frames = self.frame_count
        self.timeline.blockSignals(True)
        self.timeline.setRange(0, self.frame_count - 1)
        self.timeline.blockSignals(False)
        self.auto_pause.setChecked(False)
        self.play_button.setText(tr('▶ 재생 (Space)'))
        self.pose_meta.setText(tr('모든 화면 · 같은 프레임 · 확대/이동 동기화'))
        self.pose_meta.setToolTip(tr('보정 전 추론: {v0}', v0=self.comparison['source_pose']))
        for index, entry in enumerate(self.comparison_entries):
            item = self.playlist_widget.item(index)
            state = (tr('설정 미리보기') if self.recipe_editor else
                     tr('신체점별 보정 조합') if entry.get("composition") else
                     tr('보정 전 · 필터 비교') if entry.get("manifest") else tr('보정 전만 · 보정본 없음'))
            caption, tooltip = review_playlist_caption(entry, index, state)
            item.setText(caption)
            item.setToolTip(tooltip)
        self.sync_playlist_selection()
        self.show_frame(0)

    def switch_playlist_row(self, index: int) -> None:
        if index == self.playlist_index or not 0 <= index < len(self.comparison_entries):
            return
        self.timer.stop()
        self.play_button.setText(tr('▶ 재생 (Space)'))
        entry = self.comparison_entries[index]
        try:
            record, original, variants, pose_path = load_comparison_entry(entry)
            candidate = (self.recipe_candidate(entry, record, original)
                         if self.composition_root is not None or entry.get("composition") else None)
            probe = cv2.VideoCapture(str(record["source_video"]))
            try:
                ok, _frame = probe.read()
                if not ok:
                    raise ValueError(tr('선택한 영상의 첫 프레임을 읽을 수 없습니다.'))
            finally:
                probe.release()
        except Exception as error:
            self.sync_playlist_selection()
            QMessageBox.warning(self, tr('비교 영상 전환 실패'), str(error))
            return
        # Let the common viewer reset decoder/FPS/bodyparts/QC, but do not draw
        # the old comparison family's coordinates on the new video.
        self._switching_comparison = True
        self.playlist[index] = {**entry, "pose": str(pose_path)}
        try:
            super().switch_playlist_row(index)
        finally:
            self._switching_comparison = False
        if self.playlist_index != index:
            return
        previous = set(self.visible_filters)
        self.comparison, self.variants = record, variants
        self.manifest = Path(entry["manifest"]).resolve() if entry.get("manifest") else None
        self.comparison_frames = {0: original, **{key: item["pose"] for key, item in variants.items()}}
        self.prepare_recipe(entry, candidate=candidate)
        self.visible_filters = set(sorted((previous & set(self.variants)) or self.default_filters())[:self.MAX_FILTERS])
        self.rebuild_comparison_views()
        self.rebuild_filter_menu()
        self.reflow_views()
        self.rebuild_recipe_controls()
        self.finish_comparison_switch()

    def mode_title(self) -> str:
        return tr('보정 범위 비교·신체점 조합') if self.recipe_editor else tr('좌표 후처리 나란히 비교')

    def build_ui(self) -> None:
        super().build_ui()
        root = self.centralWidget().layout()
        old_header = root.takeAt(0).widget()
        old_header.hide()
        old_header.deleteLater()
        header, header_layout = create_hagisilta_header(self.mode_title(), "HAGISILTA · POSE COMPARISON")
        header_layout.addWidget(create_hagisilta_help_button(
            self, tr('좌표 후처리 비교 도움말'),
            tr('보정 전 추론과 여러 필터 결과를 같은 프레임에서 동시에 재생합니다.\n') +
            (tr('‘범위 비교’에서 보정 전과 최대 3개 범위를 비교한 뒤 ‘신체점 조합’으로 이동합니다.\n')
             if self.recipe_editor else tr('‘비교할 필터’에서 저장된 결과를 최대 {v0}개까지 선택합니다. 보정 전은 항상 표시합니다.\n', v0=self.MAX_FILTERS)) +
            tr('왼쪽 영상 목록을 클릭하면 이 창에서 전환합니다.\nSpace 재생/정지 · A/D 이전/다음 프레임 · Shift+A/D 10프레임 이동\n')
            + tr(PLAYBACK_HELP) + "\n"
            + (tr('신체점별 설정: 점마다 보정 전 유지 또는 중앙값 범위를 고르면 미리보기에 즉시 반영됩니다.\n조합 중에는 보정 전·조합 미리보기·최대 2개 비교 범위를 함께 볼 수 있습니다.\n‘이 설정 사용’을 누르면 재사용할 설정만 저장하고 메인 화면 2단계에서 선택됩니다.\n같은 신체점 구성을 가진 다른 영상에도 재사용합니다. 같은 프레임 수라도 FPS에 따라 시간 범위가 다릅니다.\n2단계에서 선택한 결과를 일괄 보정하고, 3단계에서 분석 입력을 지정합니다.\n')
               if self.recipe_editor else tr('저장된 결과를 비교하는 읽기 전용 창입니다. 설정·보정·분석 입력 지정은 메인 화면에서 진행합니다.\n')) +
            tr('휠 확대·축소 · 가운데 버튼 드래그 이동 · Ctrl+0 화면 맞춤: 모든 화면에 함께 적용\n\n') +
            (tr('미리보기는 현재 영상의 보정 전 좌표로 계산하며 파일을 생성하지 않습니다.\n') if self.recipe_editor else
             tr('같은 영상과 같은 보정 전 좌표로 만든 결과만 비교합니다. 같은 범위는 최신 결과를 사용하되 직접 고른 실행을 유지합니다. 새 범위를 먼저 시험하려면 메인 화면의 ‘비교하고 조합 만들기’를 사용하세요.\n')) +
            tr('오류 후보 탐색은 보정 전 좌표 기준입니다. likelihood는 보정 전 값입니다. 변경량은 정답 대비 오차가 아닙니다. 영상 양 끝과 좌표 누락 주변은 원래 좌표를 유지합니다.'),
        ))
        HeaderContext(header, tr('설정 미리보기') if self.recipe_editor else tr('동기 재생 · 읽기 전용'))
        root.insertWidget(0, header)

        old_canvas = self.canvas
        splitter = old_canvas.parentWidget()
        assert isinstance(splitter, QSplitter)
        self.mosaic = QWidget(objectName="comparisonMosaic")
        self.grid = QGridLayout(self.mosaic)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(8)
        splitter.replaceWidget(splitter.indexOf(old_canvas), self.mosaic)
        old_canvas.hide()
        old_canvas.deleteLater()
        self.rebuild_comparison_views()
        # The inherited shortcut was connected before the canvas was replaced.
        for shortcut in self.findChildren(QShortcut):
            if shortcut.key() == QKeySequence("Ctrl+0"):
                shortcut.activated.disconnect()
                shortcut.activated.connect(self.fit_all_views)

        toolbar = QFrame(objectName="contextBar")
        tools = QHBoxLayout(toolbar)
        tools.setContentsMargins(18, 6, 18, 6)
        self.filter_picker = QToolButton(objectName="toolButton")
        self.filter_picker.setText(tr('비교할 필터'))
        self.filter_picker.setMinimumWidth(124)
        self.filter_picker.setPopupMode(QToolButton.InstantPopup)
        self.filter_picker.setToolTip(tr('같은 보정 전 좌표로 만든 필터 결과를 최대 {v0}개까지 함께 표시합니다.', v0=self.MAX_FILTERS))
        self.rebuild_filter_menu()
        tools.addWidget(self.filter_picker)
        self.filter_picker.setVisible(not self.recipe_editor)
        if self.recipe_editor:
            tools.addWidget(QLabel(tr('확인할 신체점')))
            self.point_picker = QComboBox()
            self.point_picker.addItems(self.bodyparts)
            self.point_picker.setMinimumWidth(140)
            self.point_picker.currentTextChanged.connect(self.select_comparison_bodypart)
            tools.addWidget(self.point_picker)
        self.visible_summary = QLabel(objectName="comparisonSummary")
        tools.addWidget(self.visible_summary, 1)
        reset = QPushButton(tr('화면 맞춤 (Ctrl+0)'), objectName="toolButton")
        reset.clicked.connect(self.fit_all_views)
        tools.addWidget(reset)
        root.insertWidget(2, toolbar)
        self.playlist_widget.parentWidget().setMinimumWidth(240)
        self.playlist_widget.parentWidget().setMaximumWidth(400)
        self.playlist_widget.setTextElideMode(Qt.ElideRight)
        self.save_button.setFixedWidth(88)
        self.save_button.setObjectName("toolButton")
        self.save_button.setText(tr('취소') if self.recipe_editor else tr('닫기'))
        self.save_button.setToolTip(tr('설정 선택을 바꾸지 않고 닫습니다.') if self.recipe_editor else tr('비교 창을 닫습니다.'))
        self.pose_meta.setText(tr('모든 화면 · 같은 프레임 · 확대/이동 동기화'))
        self.pose_meta.setMaximumWidth(16777215)
        self.pose_meta.setToolTip(tr('생성 당시 보정 전 추론: {v0}', v0=self.comparison['source_pose']))
        self.footer_status.setWordWrap(True)
        footer_layout = self.footer_status.parentWidget().layout()
        footer_layout.setStretch(0, 1)
        footer_layout.setStretch(1, 0)
        self.footer_status.setToolTip(tr('선택한 신체점의 보정 전 대비 좌표 변경량입니다. 정답 대비 오차가 아닙니다.'))
        self.time_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        inspector = splitter.widget(2)
        inspector.setMinimumWidth(360 if self.composition_root else 280)
        inspector.setMaximumWidth(460 if self.composition_root else 360)
        if self.composition_root is not None:
            previous = inspector.takeWidget()
            tabs = QTabWidget(objectName="reviewToolTabs")
            tabs.setDocumentMode(True)
            tabs.tabBar().setDrawBase(False)
            tabs.addTab(self.build_range_controls(), tr('1 · 범위 비교'))
            tabs.addTab(self.build_recipe_controls(), tr('2 · 신체점 조합'))
            tabs.addTab(previous, tr('표시·탐색'))
            inspector.setWidget(tabs)
            self.recipe_tabs = tabs
            self.recipe_use_button = QPushButton(tr('이 설정 사용'), objectName="primaryButton")
            self.recipe_use_button.setFixedWidth(220)
            self.recipe_use_button.setEnabled(False)
            self.recipe_use_button.setToolTip(tr('설정만 저장해 메인 화면에 선택하고 닫습니다. 실제 보정은 메인 화면 2단계에서 실행합니다.'))
            self.recipe_use_button.clicked.connect(self.use_recipe)
            footer_layout.insertWidget(footer_layout.indexOf(self.save_button), self.recipe_use_button)
            tabs.currentChanged.connect(self.set_recipe_stage)
        splitter.setSizes([230, 900, 280])
        self.show_detection_boxes.hide()
        self.show_names.setChecked(self.recipe_editor)
        self.error_search_title.setText(tr('오류 후보 · 보정 전 기준'))
        self.setStyleSheet(self.styleSheet() + """
            QFrame#comparisonPanel { background:#101114; border:none; }
            QLabel#comparisonTitle { background:#303136; color:#f0f1f2; padding:9px 12px;
                                     font-size:13px; font-weight:650; border:none; }
            QWidget#comparisonMosaic { background:transparent; }
            QTabWidget#reviewToolTabs::pane { background:#23242b; border:none; }
            QTabWidget#reviewToolTabs QTabBar::tab { min-width:0px; padding:0 9px; }
        """)
        self.reflow_views()

    def select_comparison_bodypart(self, name):
        if name in self.bodyparts:
            self.select_bodypart(name)
            self.update_labels()

    def rebuild_comparison_views(self) -> None:
        while self.grid.count():
            self.grid.takeAt(0)
        for entry in self.views.values():
            entry["panel"].hide()
            entry["panel"].deleteLater()
        self.views = {}
        for key in self.comparison_frames:
            self.add_comparison_view(key)
        self.canvas = self.views[0]["canvas"]
        self.markers = self.views[0]["markers"]
        self.overlay_items = self.views[0]["overlays"]

    def add_comparison_view(self, key):
        panel = QFrame(self.mosaic, objectName="comparisonPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        title = (tr('조합 미리보기') if self.recipe_editor else tr('저장한 보정 조합')) if key == -1 else tr('보정 전') if key == 0 else tr('중앙값 {v0}프레임', v0=key)
        label = QLabel(title, objectName="comparisonTitle")
        label.setToolTip(tr('현재 설정을 보정 전 좌표에 적용한 미리보기') if key == -1 and self.recipe_editor else
                        tr('저장된 신체점별 보정 결과') if key == -1 else
                        (tr('필터 생성 당시 보관한 DLC 추론 좌표') if self.manifest else tr('DLC 추론 좌표')) if key == 0 else
                        window_description(key, self.fps) + "\n" + self.variants[key].get("manifest", tr('현재 영상에서 계산한 미리보기 · 저장 전')))
        layout.addWidget(label)
        canvas = ComparisonCanvas()
        canvas.setMinimumSize(160, 140)
        layout.addWidget(canvas, 1)
        self.views[key] = {"panel": panel, "canvas": canvas, "markers": {}, "overlays": [], "label": label}
        canvas.view_changed.connect(lambda view=canvas: self.sync_views(view))

    def rebuild_filter_menu(self) -> None:
        previous = self.filter_picker.menu()
        menu = QMenu(self.filter_picker)
        self.filter_actions = {}
        for key in (WINDOWS if self.recipe_editor else self.variants):
            action = menu.addAction(tr('중앙값 {v0}프레임', v0=key))
            action.setCheckable(True)
            action.setChecked(key in self.visible_filters)
            if self.recipe_editor:
                action.setToolTip(window_description(key, self.fps) + tr('\n미리보기 · 결과 파일을 저장하지 않습니다.'))
            else:
                created = datetime.fromisoformat(self.variants[key]["created_at"]).astimezone().strftime("%Y-%m-%d %H:%M:%S")
                action.setToolTip(window_description(key, self.fps) + tr('\n저장: {v0}', v0=created))
            action.toggled.connect(lambda checked, value=key: self.set_filter_visible(value, checked))
            self.filter_actions[key] = action
        menu.setToolTipsVisible(True)
        self.filter_picker.setMenu(menu)
        self.filter_picker.setEnabled(bool(self.variants))
        if previous:
            previous.deleteLater()

    def visible_keys(self) -> list[int]:
        show_recipe = -1 in self.comparison_frames and (not self.recipe_editor or self.recipe_stage == 1)
        return [0, *([-1] if show_recipe else []), *sorted(self.visible_filters)]

    def set_filter_visible(self, key: int, checked: bool) -> None:
        if checked and key not in self.visible_filters and len(self.visible_filters) >= self.MAX_FILTERS:
            action = self.filter_actions[key]
            action.blockSignals(True)
            action.setChecked(False)
            action.blockSignals(False)
            self.sync_range_controls()
            return
        if checked:
            if self.recipe_editor:
                try:
                    if self.ensure_preview_filter(key):
                        self.add_comparison_view(key)
                        self.show_frame(self.frame_index)
                        self.sync_views(self.canvas)
                except Exception as error:
                    self.filter_actions[key].blockSignals(True)
                    self.filter_actions[key].setChecked(False)
                    self.filter_actions[key].blockSignals(False)
                    self.sync_range_controls()
                    QMessageBox.warning(self, tr('비교 범위 확인'), str(error))
                    return
            self.visible_filters.add(key)
        else:
            self.visible_filters.discard(key)
        self.reflow_views()
        self.draw_pose()
        self.update_labels()

    def reflow_views(self) -> None:
        while self.grid.count():
            self.grid.takeAt(0)
        for entry in self.views.values():
            entry["panel"].hide()
        keys = self.visible_keys()
        columns = 2 if len(keys) == 4 else len(keys)
        for index in range(3):
            self.grid.setColumnStretch(index, 1 if index < columns else 0)
        for index in range(2):
            self.grid.setRowStretch(index, 1 if index == 0 or len(keys) == 4 else 0)
        for index, key in enumerate(keys):
            panel = self.views[key]["panel"]
            self.grid.addWidget(panel, index // columns, index % columns)
            panel.show()
        for key, action in self.filter_actions.items():
            action.blockSignals(True)
            action.setChecked(key in self.visible_filters)
            action.setEnabled((not self.recipe_editor or key <= len(self.comparison_frames[0])) and (
                key in self.visible_filters or len(self.visible_filters) < self.MAX_FILTERS))
            action.blockSignals(False)
        self.sync_range_controls()
        self.filter_picker.setToolTip(tr('보정 전과 함께 비교할 범위를 최대 {v0}개 선택합니다.', v0=self.MAX_FILTERS))
        self.visible_summary.setText(tr('보정 전 + ') + ", ".join((tr('설정 미리보기') if self.recipe_editor else tr('저장한 보정 조합')) if key == -1 else tr('{v0}프레임', v0=key) for key in keys[1:])
                                     if len(keys) > 1 else tr('보정 전'))
        QTimer.singleShot(0, lambda: self.sync_views(self.canvas))

    def fit_all_views(self) -> None:
        self.canvas.fit_to_view()

    def sync_views(self, source: ComparisonCanvas) -> None:
        if (self._syncing_views or not any(entry["canvas"] is source for entry in self.views.values())
                or source.scene.sceneRect().isEmpty()):
            return
        self._syncing_views = True
        try:
            center = source.mapToScene(source.viewport().rect().center())
            steps = source._zoom_steps
            for entry in self.views.values():
                canvas = entry["canvas"]
                if canvas is source or canvas.scene.sceneRect().isEmpty():
                    continue
                canvas.fit_to_view()
                if steps:
                    canvas.zoom_by_steps(steps)
                    canvas.centerOn(center)
                    canvas.publish_view()
        finally:
            self._syncing_views = False

    def show_frame(self, index: int) -> None:
        if self._switching_comparison:
            return
        index = max(0, min(int(index), self.frame_count - 1))
        if index >= len(self.comparison_frames[0]):
            return
        # One decode, one image, one clock. Never start a player per panel.
        if int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) != index:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = self.capture.read()
        if not ok:
            if self.timer.isActive():
                self.toggle_play()
            self.statusBar().showMessage(tr('프레임 {v0} 읽기 실패', v0=index + 1))
            self.timeline.setValue(self.frame_index)
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
        self.frame_index = index
        self._syncing_views = True
        try:
            for entry in self.views.values():
                entry["canvas"].set_frame(image)
                entry["canvas"].displayed_frame = index
        finally:
            self._syncing_views = False
        self.timeline.blockSignals(True)
        self.timeline.setValue(index)
        self.timeline.blockSignals(False)
        self.draw_pose()
        self.update_labels()

    @contextmanager
    def drawing_view(self, key: int):
        """Reuse common marker drawing; restore the baseline before any UI event."""
        saved = self.canvas, self.pose, self.markers, self.overlay_items
        entry = self.views[key]
        self.canvas, self.pose = entry["canvas"], self.comparison_frames[key]
        self.markers, self.overlay_items = entry["markers"], entry["overlays"]
        try:
            yield
        finally:
            self.canvas, self.pose, self.markers, self.overlay_items = saved

    def draw_pose(self) -> None:
        if self._switching_comparison or not self.views:
            return
        for key in self.visible_keys():
            with self.drawing_view(key):
                super().draw_pose()

    def update_labels(self) -> None:
        if self._switching_comparison:
            return
        super().update_labels()
        if self.frame_index >= len(self.comparison_frames[0]):
            return
        self.point_status.setText(self.point_status.text().replace(tr('모델 예측'), tr('보정 전 추론')))
        if hasattr(self, "point_picker"):
            self.point_picker.blockSignals(True)
            self.point_picker.setCurrentText(self.selected_bodypart)
            self.point_picker.blockSignals(False)
        xy = lambda frame: np.array([pose_column(frame, self.selected_bodypart, coord).iloc[self.frame_index]
                                     for coord in ("x", "y")], dtype=float)
        original = xy(self.comparison_frames[0])
        changes = []
        for key in self.visible_keys()[1:]:
            delta = float(np.linalg.norm(original - xy(self.comparison_frames[key])))
            label = tr('조합') if key == -1 else f"{key}F"
            changes.append(f"{label}: {delta:.2f}px" if np.isfinite(delta) else tr('{v0}: 좌표 누락', v0=label))
        self.footer_status.setText(tr('{v0} · 보정 전 대비 변경량 · ', v0=self.selected_bodypart) + " / ".join(changes)
                                   if changes else tr('보정 전 추론'))


def main() -> int:
    parser = argparse.ArgumentParser(description="HAGISILTA read-only pose filter comparison")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--playlist", type=Path)
    parser.add_argument("--playlist-index", type=int, default=0)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--recipe-manifest", type=Path)
    parser.add_argument("--recipe-result", type=Path, help="Open settings-only preview and return the chosen preset here")
    parser.add_argument("--initial-window", type=int, default=0)
    args = parser.parse_args()
    app = QApplication(sys.argv)
    init_i18n(app)
    app.setApplicationName("HAGISILTA")
    install_wheel_value_guard(app)
    try:
        playlist = json.loads(args.playlist.read_text(encoding="utf-8")) if args.playlist else None
        window = FilterComparisonWindow(args.manifest, playlist=playlist, playlist_index=args.playlist_index,
                                        output_root=args.output_root, recipe_manifest=args.recipe_manifest,
                                        recipe_result=args.recipe_result, initial_window=args.initial_window)
    except Exception as error:
        QMessageBox.critical(None, tr('좌표 후처리 비교 실패'), str(error))
        return 1
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
