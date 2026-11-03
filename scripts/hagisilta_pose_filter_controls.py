# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
"""Shared recipe and explicit analysis-selection controls for both workbenches."""

from hagisilta_i18n import tr
import json
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton, QTableWidgetItem, QVBoxLayout, QWidget

from hagisilta_pose_postprocess import (
    choose_analysis_pose, list_compositions, list_recipe_presets, load_composition, read_recipe, recipe_label, recipe_display_name,
)
from hagisilta_pose_postprocess import list_filter_tests, read_predictions
from hagisilta_pose_filter_status import FilterStatusReader
from hagisilta_window_chrome import create_hagisilta_settings_section


def filter_history(root, video, pose):
    return sorted(list_filter_tests(root, video, pose) + list_compositions(root, video, pose),
                  key=lambda row: row["created_at"], reverse=True)


def history_label(record):
    return recipe_display_name(record.get("name")) if "recipe" in record else tr('중앙값 {v0}프레임', v0=record['window'])


def matches_settings(record, recipe, window):
    return (record.get("recipe") == recipe if recipe else
            record.get("window") == window or
            bool(record.get("recipe") and set(record["recipe"].values()) == {window}))


class PoseFilterControls:
    def filter_status_reader(self):
        root = self.evaluation_filter_root()
        reader = getattr(self, "_filter_status_reader", None)
        if reader is None or reader.root != root:
            reader = self._filter_status_reader = FilterStatusReader(root)
        return reader

    def filter_status_cells(self, payload, refresh=False):
        reader = self.filter_status_reader()
        video, pose = payload["video"], payload["pose"]
        states = (reader.recent(video, pose, refresh=refresh), reader.analysis(video, pose))
        cells = []
        for state in states:
            item = QTableWidgetItem(state.title)
            item.setToolTip(state.detail)
            cells.append(item)
        return cells

    def refresh_filter_status_tables(self, rescan=False):
        """Update provenance cells in place; keep sorting, selection and scroll."""
        if not self.config:
            return
        for name in ("track_review_table", "eval_pilot_review_table"):
            table = getattr(self, name, None)
            if table is None or table.columnCount() < 7:
                continue
            blocked, sorting = table.blockSignals(True), table.isSortingEnabled()
            scroll = table.horizontalScrollBar().value(), table.verticalScrollBar().value()
            try:
                table.setSortingEnabled(False)
                for row in range(table.rowCount()):
                    item = table.item(row, 0)
                    payload = item.data(Qt.UserRole) if item else None
                    if not payload:
                        continue
                    for column, fresh in enumerate(self.filter_status_cells(payload, refresh=rescan), 5):
                        cell = table.item(row, column)
                        if cell is None:
                            table.setItem(row, column, fresh)
                        elif cell.text() != fresh.text() or cell.toolTip() != fresh.toolTip():
                            cell.setText(fresh.text())
                            cell.setToolTip(fresh.toolTip())
            finally:
                table.setSortingEnabled(sorting)
                table.blockSignals(blocked)
                table.horizontalScrollBar().setValue(scroll[0])
                table.verticalScrollBar().setValue(scroll[1])

    def build_filter_workflow(self, scope):
        """One ordered workflow shared by tracking and evaluation."""
        panel = QFrame(objectName="filterWorkflow")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        def stage(number, title):
            box, inner = create_hagisilta_settings_section(title, number)
            layout.addWidget(box)
            return inner
        settings_controls = self.build_composition_controls(scope)
        controls = self.composition_controls[scope]
        comparison = stage(1, tr('범위 비교·조합 만들기'))
        preview_row = QHBoxLayout()
        preview_target = QLabel(objectName="pageText"); preview_target.setWordWrap(True)
        preview_row.addWidget(preview_target, 1)
        preview_row.addWidget(controls["edit"])
        controls["preview_target"] = preview_target
        comparison.addLayout(preview_row)

        settings = stage(2, tr('보정 설정·일괄 실행'))
        settings.addWidget(settings_controls)
        uniform = QWidget(objectName="filterUniformFields")
        row = QHBoxLayout(uniform); row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel(tr('중앙값 범위')))
        window = self.pose_filter_window_combo()
        setattr(self, scope + "_filter_window", window)
        window.currentIndexChanged.connect(self.update_tracking_filter_window if scope == "track" else self.update_evaluation_filter_window)
        window.currentIndexChanged.connect(lambda _index: self.update_composition_status(scope))
        row.addWidget(window)
        timing = QLabel(objectName="pageText"); timing.setWordWrap(True)
        setattr(self, scope + "_filter_time", timing)
        row.addWidget(timing, 1)
        controls["settings_row"].insertWidget(2, uniform, 1)
        controls["uniform"] = uniform
        controls["workflow_panel"] = panel
        action = QHBoxLayout()
        status = QLabel(objectName="pageText"); status.setWordWrap(True)
        setattr(self, scope + "_filter_status", status)
        action.addWidget(status, 1)
        run = QPushButton(tr('선택 결과 보정'), objectName="primaryButton")
        self.fit_dynamic_action_button(run)
        run.clicked.connect(self.run_tracking_filter if scope == "track" else self.run_evaluation_filter)
        setattr(self, scope + "_filter_run", run); action.addWidget(run)
        settings.addLayout(action)
        results_stage = stage(3, tr('보정 결과 확인·교정') if scope == "eval" else tr('보정 결과 확인·활용'))
        saved_results = QVBoxLayout()
        results_stage.addLayout(saved_results)
        controls["saved_results_layout"] = saved_results
        saved = QHBoxLayout(); saved.addWidget(QLabel(tr('저장본 보기')))
        results = QComboBox(); results.setMinimumWidth(240)
        results.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        results.setToolTip(tr('이미 저장한 결과를 확인합니다. 여기서 고른 저장본은 일괄 보정 설정을 바꾸지 않습니다.'))
        setattr(self, scope + "_filter_results", results)
        results.currentIndexChanged.connect(self.update_tracking_filter_result if scope == "track" else self.update_evaluation_filter_result)
        saved.addWidget(results, 1)
        for key, label, callback, icon in (
            ("open", tr('저장본 비교 열기'), self.open_tracking_filter_comparison if scope == "track" else self.open_evaluation_filter_comparison, "open_video"),
            ("folder", tr('보정 폴더 열기'), self.open_tracking_filter_folder if scope == "track" else self.open_evaluation_filter_folder, "folder"),
        ):
            button = QPushButton(label, objectName="toolButton"); button.setEnabled(False)
            self.set_action_icon(button, icon, 14); button.clicked.connect(callback)
            setattr(self, scope + "_filter_" + key, button); saved.addWidget(button)
        saved_results.addLayout(saved)
        comparison_summary = QLabel(objectName="pageText"); comparison_summary.setWordWrap(True)
        saved_results.addWidget(comparison_summary)
        controls["comparison_summary"] = comparison_summary
        review_row = QHBoxLayout()
        review_note = QLabel(tr('교정 저장 후 ‘새 수정 프레임 반영’으로 학습에 추가합니다.'), objectName="pageText")
        review_note.setWordWrap(True)
        review_row.addWidget(review_note, 1)
        review = QPushButton(tr('후처리본으로 교정'), objectName="toolButton")
        self.set_action_icon(review, "edit", 14)
        review.setToolTip(tr('‘저장본 보기’에서 고른 좌표로 교정을 시작합니다. 기존 수동 교정은 유지합니다. 학습·행동 분석에는 자동 적용하지 않습니다.'))
        review.clicked.connect(lambda: self.open_filtered_video_review(scope))
        controls["review"] = review
        review_row.addWidget(review)
        results_stage.addLayout(review_row)
        if scope == "track":
            results_stage.addWidget(controls["analysis_panel"])
        else:
            controls["analysis_panel"].setParent(panel)
            controls["analysis_panel"].hide()
        self.update_composition_status(scope)
        return panel

    def build_composition_controls(self, scope):
        if not hasattr(self, "composition_controls"):
            self.composition_controls = {}
            self.composition_poll = QTimer(self)
            self.composition_poll.setInterval(2000)
            self.composition_poll.timeout.connect(self.poll_composition_changes)
            self.composition_poll.start()
        panel = QWidget(objectName="filterSettingsFields")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel(tr('보정 방식')))
        mode = QComboBox()
        mode.addItem(tr('전체 신체점 동일'), "uniform")
        mode.addItem(tr('신체점별 설정'), "parts")
        mode.setFixedWidth(180)
        row.addWidget(mode)
        combo = QComboBox()
        combo.setMinimumWidth(240)
        combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.addItem(tr('저장한 신체점 설정 선택'), None)
        combo.setToolTip(tr('프로젝트 공용 설정입니다. 같은 신체점 이름을 가진 다른 영상에도 재사용합니다. 선택만으로 좌표는 바뀌지 않으며 2단계에서 선택 결과를 보정합니다.'))
        row.addWidget(combo, 1)
        edit = QPushButton(tr('비교하고 조합 만들기'), objectName="toolButton")
        edit.setFixedWidth(220)
        self.set_action_icon(edit, "edit", 14)
        edit.setToolTip(tr('대표 결과 한 개에서 보정 전·여러 중앙값 범위를 먼저 비교한 뒤 신체점별로 조합합니다. 저장된 보정본 없이도 미리 볼 수 있습니다.'))
        edit.clicked.connect(lambda: self.open_recipe_editor(scope))
        layout.addLayout(row)
        recipe_summary = QLabel(objectName="pageText")
        recipe_summary.setWordWrap(True)
        layout.addWidget(recipe_summary)
        settings_row = row
        analysis_panel = QWidget(objectName="filterAnalysisFields")
        row = QHBoxLayout(analysis_panel)
        row.setContentsMargins(0, 0, 0, 0)
        status = QLabel(objectName="pageText")
        status.setWordWrap(True)
        row.addWidget(status, 1)
        use = QPushButton(tr('선택 결과를 행동 분석에 사용'), objectName="primaryButton")
        use.setFixedWidth(220)
        self.set_action_icon(use, "confirm", 14)
        use.clicked.connect(lambda: self.choose_workbench_analysis(scope, False))
        row.addWidget(use)
        original = QPushButton(tr('보정 전 사용'), objectName="toolButton")
        original.clicked.connect(lambda: self.choose_workbench_analysis(scope, True))
        original.setToolTip(tr('선택한 ROI의 다음 행동 분석에 원본 추론 좌표를 사용합니다. 보정본과 기존 분석 결과는 삭제하지 않습니다.'))
        row.addWidget(original)
        self.composition_controls[scope] = {"combo": combo, "mode": mode, "edit": edit, "use": use, "original": original,
                                            "status": status, "analysis_panel": analysis_panel,
                                            "settings_row": settings_row, "recipe_summary": recipe_summary}
        mode.currentIndexChanged.connect(lambda _index: self.update_composition_status(scope))
        combo.currentIndexChanged.connect(lambda _index: self.recipe_selection_changed(scope))
        return panel

    def recipe_selection_changed(self, scope):
        controls = self.composition_controls[scope]
        if controls["combo"].currentData():
            controls["mode"].setCurrentIndex(1)
        self.update_composition_status(scope)

    def composition_payloads(self, scope):
        return (self.tracking_review_selected_payloads() if scope == "track" else
                self.evaluation_pilot_review_selected_payloads()) if self.config else []

    def chosen_recipe_manifest(self, scope):
        controls = getattr(self, "composition_controls", {}).get(scope)
        return controls["combo"].currentData() if controls and controls["mode"].currentData() == "parts" else None

    def poll_composition_changes(self):
        if not self.config or not hasattr(self, "composition_controls"):
            return
        self.consume_recipe_editor_results()
        root = self.evaluation_filter_root()
        marker = root / "composition-revision.json"
        stamp = (str(root), marker.stat().st_mtime_ns if marker.is_file() else None)
        if stamp != getattr(self, "_composition_revision", None):
            if self.refresh_composition_presets():
                self._composition_revision = stamp
                self.refresh_filter_status_tables(rescan=True)
                self.refresh_evaluation_filter()
                self.refresh_tracking_filter()
        else:
            self.refresh_filter_status_tables()
            for scope in self.composition_controls:
                self.update_composition_status(scope)

    def refresh_composition_presets(self, force=False):
        controls = getattr(self, "composition_controls", {})
        # Do not destroy a popup or replace a user's in-progress choice.
        histories = [getattr(self, scope + "_filter_results", None) for scope in controls]
        if (any(row["combo"].view().isVisible() for row in controls.values()) or
                any(combo is not None and combo.view().isVisible() for combo in histories)):
            return False
        records = list_recipe_presets(self.evaluation_filter_root()) if self.config else []
        project_changed = str(self.config) != getattr(self, "_composition_project", None)
        self._composition_project = str(self.config)
        recipes = {}
        for record in records:
            identity = (record["name"], json.dumps(record["recipe"], sort_keys=True))
            recipes.setdefault(identity, record)
        for scope, row in controls.items():
            combo = row["combo"]
            previous = None if project_changed else combo.currentData()
            old_recipe = None if project_changed else combo.currentData(Qt.UserRole + 1)
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(tr('저장한 신체점 설정 선택'), None)
            for record in recipes.values():
                created = datetime.fromisoformat(record["created_at"]).astimezone().strftime("%Y-%m-%d %H:%M")
                combo.addItem(recipe_display_name(record["name"]) + " · " + created, record["manifest"])
                combo.setItemData(combo.count() - 1, record["recipe"], Qt.UserRole + 1)
                combo.setItemData(combo.count() - 1, recipe_label(record["recipe"]), Qt.ToolTipRole)
            index = combo.findData(previous)
            if index < 0 and old_recipe:
                index = next((i for i in range(1, combo.count()) if combo.itemData(i, Qt.UserRole + 1) == old_recipe), 0)
            combo.setCurrentIndex(max(0, index))
            combo.blockSignals(False)
            self.update_composition_status(scope)
        return True

    def update_composition_status(self, scope):
        controls = getattr(self, "composition_controls", {}).get(scope)
        if not controls:
            return
        payloads = self.composition_payloads(scope)
        count = len(payloads)
        results = getattr(self, scope + "_filter_results", None)
        manifest = results.currentData() if results else None
        recipe = self.chosen_recipe_manifest(scope)
        parts_mode = controls["mode"].currentData() == "parts"
        busy = bool(getattr(self, "_" + scope + "_filter_busy", False))
        controls["mode"].setEnabled(not busy)
        controls["combo"].setVisible(parts_mode)
        if "uniform" in controls:
            controls["uniform"].setVisible(not parts_mode)
        controls["combo"].setEnabled(not busy)
        controls["edit"].setEnabled(count == 1 and not busy)
        if "preview_target" in controls:
            controls["preview_target"].setText(
                tr('대표 결과 · {v0}', v0=Path(payloads[0]['video']).name) if count == 1 else
                tr('비교할 대표 결과 한 개를 선택하세요.'))
        controls["recipe_summary"].setVisible(parts_mode)
        selected_recipe = controls["combo"].currentData(Qt.UserRole + 1) or {}
        controls["recipe_summary"].setText(
            tr('일괄 보정에 사용할 설정 · {v0}', v0=recipe_label(selected_recipe)) if recipe else
            tr('1단계에서 비교·조합한 설정을 선택하세요.'))
        controls["original"].setEnabled(count > 0 and not busy)
        if hasattr(self, scope + "_filter_run"):
            ready = (bool(count) if scope == "track" else
                     bool(getattr(self, "_eval_filter_ready", count == 1)))
            getattr(self, scope + "_filter_run").setEnabled(ready and not busy and (not parts_mode or bool(recipe)))
        controls["use"].setEnabled(not busy and bool(count) and
                                  (bool(manifest) if count == 1 else not parts_mode or bool(recipe)))
        controls["use"].setText(tr('선택한 {v0}개를 행동 분석에 사용', v0=count) if count else tr('선택 결과를 행동 분석에 사용'))
        self.fit_dynamic_action_button(controls["use"])
        controls["use"].setToolTip(tr('2단계 보정 설정과 일치하는 각 영상의 저장본을 분석 입력으로 지정합니다. ‘저장본 보기’의 선택과는 별개입니다.'))
        if "review" in controls:
            controls["review"].setEnabled(count == 1 and bool(manifest) and not busy)
        run = getattr(self, scope + "_filter_run", None)
        if run is not None:
            chosen_text = recipe_label(selected_recipe) if parts_mode and recipe else (
                tr('중앙값 {v0}프레임', v0=getattr(self, scope + '_filter_window').currentData()) if not parts_mode else tr('설정 선택 필요'))
            run.setToolTip(tr('바로 위 보정 설정으로 선택 결과를 처리합니다.\n') + chosen_text)
        window = getattr(self, scope + "_filter_window", None)
        if window is None:
            return
        window.setEnabled(not busy and not parts_mode)
        if parts_mode and not recipe and hasattr(self, scope + "_filter_run"):
            getattr(self, scope + "_filter_run").setEnabled(False)
        reader = self.filter_status_reader() if payloads else None
        comparison_summary = controls.get("comparison_summary")
        if comparison_summary is not None:
            if manifest and count == 1:
                saved = reader.describe(manifest, payloads[0]["video"], payloads[0]["pose"])
                comparison_summary.setText(tr('이 보정본의 설정 · ') + saved.settings)
                comparison_summary.setToolTip(saved.detail)
            else:
                comparison_summary.setText("")
                comparison_summary.setToolTip("")
            comparison_summary.setVisible(bool(manifest and count == 1))
        text = tr('행동 분석 입력: 결과 선택')
        controls["status"].setToolTip("")
        if count == 1:
            state = reader.analysis(payloads[0]["video"], payloads[0]["pose"])
            text = tr('행동 분석 입력: ') + state.settings
            controls["status"].setToolTip(state.detail)
        elif count > 1:
            counts = Counter(reader.analysis(row["video"], row["pose"]).settings for row in payloads)
            groups = [tr('{v0} {v1}개', v0=label, v1=n) for label, n in counts.most_common()]
            text = tr('현재 분석 입력 · ') + " / ".join(groups[:3])
            if len(groups) > 3:
                text += tr(' / 외 {v0}종', v0=len(groups) - 3)
            controls["status"].setToolTip("\n".join(groups))
        controls["status"].setText(text)
        if recipe:
            getattr(self, scope + "_filter_time").setText(tr('신체점별 조합 범위 적용 · 영상별 FPS 기준'))
        elif scope == "track":
            self.update_tracking_filter_window()
        else:
            self.update_evaluation_filter_window()

    def open_filtered_video_review(self, scope):
        payloads = self.composition_payloads(scope)
        results = getattr(self, scope + "_filter_results", None)
        manifest = results.currentData() if results else None
        if len(payloads) != 1 or not manifest or getattr(self, "_" + scope + "_filter_busy", False):
            return
        self.launch_video_review(payloads[0], draft_manifest=Path(manifest))

    def open_recipe_editor(self, scope):
        payloads = self.composition_payloads(scope)
        if len(payloads) != 1:
            return
        table = self.track_review_table if scope == "track" else self.eval_pilot_review_table
        self.launch_filter_comparison(table, payloads[0], None, self.chosen_recipe_manifest(scope), recipe_scope=scope)

    def consume_recipe_editor_results(self):
        """Return only settings to the launching project/scope; never start a job."""
        pending = []
        for request in getattr(self, "_recipe_editor_requests", []):
            if request["config"] != str(self.config):
                continue
            if getattr(self, "_" + request["scope"] + "_filter_busy", False):
                pending.append(request)
                continue
            result = Path(request["result"])
            if not result.is_file():
                pending.append(request)
                continue
            try:
                response = json.loads(result.read_text(encoding="utf-8"))
                if response.get("cancelled"):
                    continue
                manifest = str(Path(response["preset"]).resolve())
                read_recipe(Path(manifest))
                if not self.refresh_composition_presets():
                    pending.append(request)
                    continue
                row = self.composition_controls[request["scope"]]
                index = row["combo"].findData(manifest)
                if index < 0:
                    raise ValueError(tr('저장한 신체점 설정을 목록에서 찾을 수 없습니다.'))
                row["combo"].setCurrentIndex(index)
                row["mode"].setCurrentIndex(1)
                self.update_composition_status(request["scope"])
            except (OSError, ValueError, KeyError, TypeError) as error:
                QMessageBox.warning(self, tr('보정 설정 선택 실패'), str(error))
        self._recipe_editor_requests = pending

    def choose_workbench_analysis(self, scope, original):
        payloads = self.composition_payloads(scope)
        if not payloads:
            return
        if getattr(self, "_" + scope + "_filter_busy", False):
            return
        try:
            targets, arenas = [], set()
            recipe_path = self.chosen_recipe_manifest(scope)
            if not original and self.composition_controls[scope]["mode"].currentData() == "parts" and not recipe_path:
                raise ValueError(tr('적용할 신체점별 설정을 먼저 선택하세요.'))
            recipe = read_recipe(Path(recipe_path))[0] if recipe_path and not original else None
            for payload in payloads:
                video, pose = Path(payload["video"]), Path(payload["pose"])
                arena = video.resolve().parent
                if arena in arenas:
                    raise ValueError(tr('같은 ROI의 모델 결과가 여러 개 선택되었습니다. ROI마다 사용할 결과 하나만 선택하세요.'))
                arenas.add(arena)
                manifest = None
                if not original:
                    window = getattr(self, scope + "_filter_window").currentData() or 3
                    matches = [row for row in filter_history(self.evaluation_filter_root(), video, pose)
                               if matches_settings(row, recipe, window)]
                    if not matches:
                        raise ValueError(tr('현재 설정과 일치하는 보정본이 없습니다: {v0}\n2단계에서 선택 결과를 먼저 보정하세요. 아무 ROI도 분석 입력을 변경하지 않았습니다.', v0=video.name))
                    manifest = Path(matches[0]["manifest"])
                    if len(payloads) == 1:
                        selected = getattr(self, scope + "_filter_results").currentData()
                        if selected and any(Path(row["manifest"]) == Path(selected) for row in matches):
                            manifest = Path(selected)
                    record = json.loads(manifest.read_text(encoding="utf-8"))
                    if Path(record["source_video"]).resolve() != video.resolve() or Path(record["source_pose"]).resolve() != pose.resolve():
                        raise ValueError(tr('분석 입력의 원본 영상·추론이 일치하지 않습니다.'))
                targets.append((video, pose, manifest))
        except Exception as error:
            QMessageBox.warning(self, tr('분석 입력 확인'), str(error))
            return
        label = tr('보정 전 좌표') if original else (tr('신체점별 설정 · ') + recipe_label(recipe) if recipe else tr('중앙값 {v0}프레임 보정본', v0=window))
        if QMessageBox.question(self, tr("분석 입력 지정"), tr("선택한 {v0}개 ROI의 다음 행동 분석에 {v1}을 사용합니다.\n기존 분석 결과는 유지됩니다.", v0=len(targets), v1=label)) != QMessageBox.Yes:
            return
        if len(targets) > 1:
            self.start_analysis_batch(scope, targets)
            return
        done = 0
        try:
            for target in targets:
                choose_analysis_pose(*target)
                done += 1
        except Exception as error:
            QMessageBox.warning(self, tr('분석 입력 지정 중단'), tr('{v0}/{v1}개 지정 완료 · {v2}', v0=done, v1=len(targets), v2=error))
        self.refresh_filter_status_tables()
        for key in self.composition_controls:
            self.update_composition_status(key)

    def start_analysis_batch(self, scope, targets):
        from PySide6.QtCore import QProcess
        if self.process and self.process.state() != QProcess.NotRunning:
            QMessageBox.information(self, tr('작업 실행 중'), tr('현재 작업이 끝난 뒤 분석 입력을 지정하세요.'))
            return
        runtime = self.evaluation_filter_root().parent / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="analysis-inputs-", dir=runtime))
        plan, report = directory / "plan.json", directory / "report.json"
        plan.write_text(json.dumps({"targets": [{"video": str(v), "pose": str(p), "manifest": str(m) if m else None}
                                                for v, p, m in targets]}, ensure_ascii=False), encoding="utf-8")
        setattr(self, "_" + scope + "_filter_busy", True)
        self.update_composition_status(scope)
        handled = False
        def finished(code):
            nonlocal handled
            if handled:
                return
            handled = True
            setattr(self, "_" + scope + "_filter_busy", False)
            try:
                result = json.loads(report.read_text(encoding="utf-8"))
                if code or not result.get("complete"):
                    raise ValueError(result.get("error", tr('작업 중단')))
                text = tr('{v0}개 ROI 분석 입력 지정 완료', v0=result['count'])
            except Exception as error:
                text = tr('분석 입력 지정 실패 · {v0}', v0=error)
                QMessageBox.warning(self, tr('분석 입력 지정'), text)
            self.refresh_filter_status_tables()
            for key in self.composition_controls:
                self.update_composition_status(key)
            if scope == "track":
                self.refresh_tracking_filter()
            else:
                self.refresh_evaluation_filter()
            status = self.composition_controls[scope]["status"]
            status.setText(text + "\n" + status.text())
        try:
            self.start_process(Path(__file__).with_name("hagisilta_pose_postprocess.py"),
                               ["apply", "--analysis-plan", str(plan), "--report", str(report)], finished)
            if self.process:
                self.process.errorOccurred.connect(
                    lambda error: finished(1) if error == QProcess.FailedToStart else None)
        except Exception:
            finished(1)
            raise
