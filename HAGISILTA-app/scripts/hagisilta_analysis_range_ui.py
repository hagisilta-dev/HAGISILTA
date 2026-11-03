# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
"""Shared range controls for NORT, Y-maze and Water maze."""
from __future__ import annotations

from hagisilta_i18n import tr

import re
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QListWidget, QPlainTextEdit, QPushButton, QScrollArea, QSplitter,
    QStyle, QStyleOptionViewItem, QVBoxLayout, QWidget,
)
from hagisilta_analysis_range import format_time, propose_ranges, validate_range
from hagisilta_window_chrome import (
    apply_hagisilta_window_identity, create_hagisilta_header, create_hagisilta_help_button,
    enable_full_window_controls, hagisilta_control_stylesheet, create_hagisilta_settings_section,
)

RANGE_HELP = (
    "영상과 설정 사이의 세로 분할 바를 드래그하면 두 영역의 폭을 조절할 수 있습니다.\n"
    "빠른 이동은 재생 위치만 바꾸며 분석 구간은 변경하지 않습니다. 기준을 현재 위치·분석 시작·영상 처음 중 고르세요.\n"
    "시간은 분:초.밀리초로 입력합니다. 예: 10분 = 10:00.000, 13초 = 00:13.000. 분은 59를 넘어도 됩니다.\n"
    "시작 장면에서 '현재 위치 → 시작'을 누른 뒤 실험 길이를 입력하세요. 길이 고정 중에는 시작을 바꾸면 종료도 함께 이동합니다.\n"
    "종료를 직접 지정하려면 길이 고정을 끄거나 '현재 위치 → 종료'를 누르세요. 영상보다 긴 구간은 저장할 수 없습니다.\n"
    "원본 영상·좌표는 자르거나 재인코딩하지 않습니다. 저장한 구간은 다음 분석 실행부터 반영됩니다."
)

SINGLE_RANGE_HELP = (
    "ROI마다 한 마리의 시작·종료 한 쌍을 저장합니다. 앞뒤 대기시간을 제외하세요.\n"
    "시간은 분:초.밀리초로 입력합니다. 예: 10분 = 10:00.000.\n"
    "시작과 종료는 독립적입니다. 한쪽을 바꿔도 다른 쪽은 자동으로 움직이지 않습니다.\n"
    "직접 입력을 확정하면 해당 장면으로 이동합니다. '기준 시각 + 시간'의 입력 하나를 위치 보기와 종료 지정에 함께 씁니다. "
    "예: 분석 시작 기준 + 10:00.000 → '위치 보기'는 영상만 이동, '종료로 지정'은 종료도 변경합니다.\n"
    "왼쪽 ROI 행을 누르면 같은 재생 시각에서 다른 ROI를 봅니다. 수정 중인 구간은 유지됩니다. "
    "체크는 구간 복사 대상이며 행 선택·메인 표의 사용 체크와 별개입니다.\n"
    "'체크한 N개 ROI에 이 구간 적용'은 창 안의 초안만 변경합니다. 목록의 미저장 표시와 구간을 확인하고 "
    "다른 ROI도 개별 조정하세요. 다른 영상은 바뀌지 않습니다.\n"
    "Ctrl+S는 변경된 모든 ROI를 저장만, '저장하고 닫기'는 저장 후 종료합니다. "
    "체크를 해제해도 이미 수정한 초안은 저장 대상입니다. 취소는 마지막 저장 이후 변경만 버립니다.\n"
    "원본 영상·좌표는 자르지 않습니다. 저장한 구간은 다음 분석 실행부터 반영됩니다."
)


class TimecodeSpin(QDoubleSpinBox):
    """A single mm:ss.mmm field with a seconds-valued API (no 24 h wrap)."""
    def __init__(self, seconds=0, parent=None):
        super().__init__(parent)
        self.setObjectName("analysisTimecode")
        self.setRange(0, 604800)
        self.setDecimals(3)
        self.setKeyboardTracking(False)
        self.setFixedWidth(168)
        self.setMinimumHeight(34)
        self.setStyleSheet("QDoubleSpinBox#analysisTimecode {font-family:Consolas; font-size:16px; font-weight:500; font-style:normal; color:#f2f3f5;}")
        self.setToolTip(tr('분:초.밀리초 · 예: 10:00.000 = 10분\n위/아래 화살표는 커서가 있는 분·초·밀리초 단위를 바꿉니다.'))
        self.setValue(seconds)

    def textFromValue(self, value):
        return format_time(value)

    def valueFromText(self, text):
        match = re.fullmatch(r"(\d{1,5}):([0-5]\d)(?:\.(\d{1,3}))?", text.strip())
        if not match:
            return self.value()
        return int(match[1]) * 60 + int(match[2]) + int((match[3] or "").ljust(3, "0")) / 1000

    def validate(self, text, position):
        candidate = text.strip()
        if re.fullmatch(r"\d{1,5}:[0-5]\d(?:\.\d{1,3})?", candidate):
            state = QValidator.Acceptable if self.minimum() <= self.valueFromText(candidate) <= self.maximum() else QValidator.Invalid
        elif re.fullmatch(r"\d{0,5}(?::[0-5]?\d?(?:\.\d{0,3})?)?", candidate):
            state = QValidator.Intermediate
        else:
            state = QValidator.Invalid
        return state, text, position

    def stepBy(self, steps):
        if self.isReadOnly():
            return
        position = self.lineEdit().cursorPosition()
        text = self.lineEdit().text()
        unit = 60 if position <= text.find(":") else .001 if position > text.find(".") >= 0 else 1
        self.interpretText()
        self.setValue(self.value() + steps * unit)
        self.lineEdit().setCursorPosition(min(position, len(self.lineEdit().text())))

    def set_value(self, seconds):
        self.setValue(seconds)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.interpretText()
            self.editingFinished.emit()
            event.accept()
            return  # Commit this field, never activate a dialog's default button.
        super().keyPressEvent(event)

    def wheelEvent(self, event):
        event.ignore()


def create_range_inspector(content, minimum_width=438):
    """Resizable inspector; reserve its scrollbar gutter to keep inputs steady."""
    scroll = QScrollArea(objectName="analysisRangeInspector")
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setMinimumWidth(minimum_width)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setStyleSheet("QScrollArea#analysisRangeInspector {background:#27282d; border:none;}")
    scroll.setWidget(content)
    return scroll


def create_range_splitter(video, inspector, inspector_width=448):
    splitter = QSplitter(Qt.Horizontal, objectName="analysisRangeSplitter")
    splitter.setChildrenCollapsible(False)
    splitter.setHandleWidth(10)
    splitter.setStyleSheet("""
        QSplitter#analysisRangeSplitter::handle:horizontal {background:#484d56; margin:0 3px;}
        QSplitter#analysisRangeSplitter::handle:horizontal:hover,
        QSplitter#analysisRangeSplitter::handle:horizontal:pressed {background:#8a9098;}
    """)
    splitter.addWidget(video)
    splitter.addWidget(inspector)
    splitter.setStretchFactor(0, 1)
    splitter.setStretchFactor(1, 0)
    splitter.setSizes([720, inspector_width])
    splitter.handle(1).setCursor(Qt.SizeHorCursor)
    splitter.handle(1).setToolTip(tr('드래그하여 영상과 설정 패널의 폭 조절'))
    return splitter


def create_range_clock():
    label = QLabel("00:00.000 / 00:00.000", objectName="rangePlaybackTime")
    label.setStyleSheet("QLabel#rangePlaybackTime {font-family:Consolas; font-size:18px; font-weight:600; font-style:normal; color:#f2f3f5; background:transparent;}")
    label.setMinimumHeight(30)
    label.setToolTip(tr('현재 재생 위치 / 전체 영상 길이 · 분:초.밀리초'))
    return label


class AnalysisRangeControls(QWidget):
    def __init__(self, duration, fps, start, end, current_seconds, seek_seconds, parent=None):
        super().__init__(parent)
        self.duration, self.fps = duration, fps
        self.current_seconds, self.seek_seconds = current_seconds, seek_seconds
        self._updating = True
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        seek_section, seek_layout = create_hagisilta_settings_section(tr('빠른 이동'))
        seek_row = QGridLayout()
        seek_row.setHorizontalSpacing(8)
        self.seek_basis = QComboBox()
        for label, key in ((tr('현재 위치 기준'), "current"), (tr('분석 시작 기준'), "start"), (tr('영상 처음 기준'), "zero")):
            self.seek_basis.addItem(label, key)
        self.seek_amount = TimecodeSpin(10)
        seek_row.addWidget(self.seek_basis, 0, 0, 1, 3)
        seek_row.addWidget(self.seek_amount, 1, 0)
        for column, (label, direction) in enumerate(((tr('뒤로 이동'), -1), (tr('앞으로 이동'), 1)), 1):
            button = QPushButton(label, objectName="toolButton")
            button.setFixedWidth(100)
            button.setAutoDefault(False)
            button.clicked.connect(lambda _checked=False, d=direction: self.jump(d))
            seek_row.addWidget(button, 1, column)
        seek_layout.addLayout(seek_row)
        layout.addWidget(seek_section)
        self.seek_basis.setToolTip(tr('이동 기준만 고릅니다. 영상 이동은 분석 시작·종료를 바꾸지 않습니다.'))
        range_section, range_layout = create_hagisilta_settings_section(tr('분석 구간'))
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        self.start_spin, self.end_spin = TimecodeSpin(), TimecodeSpin()
        for row, (label, spin, callback, button_text) in enumerate((
            (tr('분석 시작'), self.start_spin, self.set_start_here, tr('현재 위치 → 시작 (S)')),
            (tr('분석 종료'), self.end_spin, self.set_end_here, tr('현재 위치 → 종료 (F)')),
        )):
            grid.addWidget(QLabel(label), row * 2, 0, 1, 2)
            grid.addWidget(spin, row * 2 + 1, 0)
            button = QPushButton(button_text, objectName="toolButton")
            button.setFixedWidth(208)
            button.setAutoDefault(False)
            button.clicked.connect(callback)
            grid.addWidget(button, row * 2 + 1, 1)
        range_layout.addLayout(grid)
        layout.addWidget(range_section)
        length_section, length_layout = create_hagisilta_settings_section(tr('실험 길이'))
        length_row = QHBoxLayout()
        self.length = TimecodeSpin()
        length_row.addWidget(self.length)
        self.lock_length = QCheckBox(tr('길이 고정'))
        self.lock_length.setToolTip(tr('시작을 옮겨도 입력한 실험 길이를 유지합니다. 해제하면 종료 시각을 직접 지정합니다.'))
        self.lock_length.setChecked(True)
        length_row.addWidget(self.lock_length)
        full = QPushButton(tr('영상 전체'), objectName="toolButton")
        full.setFixedWidth(100)
        full.setAutoDefault(False)
        full.clicked.connect(lambda: self.set_range(0, self.duration))
        length_row.addStretch()
        length_layout.addLayout(length_row)
        self.summary = QLabel(objectName="pageText")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("font-size:13px; color:#cbd0d8;")
        length_layout.addWidget(self.summary)
        length_layout.addWidget(full, 0, Qt.AlignRight)
        layout.addWidget(length_section)
        self.start_spin.valueChanged.connect(self.start_changed)
        self.end_spin.valueChanged.connect(self.end_changed)
        self.length.valueChanged.connect(self.length_changed)
        self.lock_length.toggled.connect(self.lock_changed)
        self._updating = False
        self.set_range(start, end)

    def set_range(self, start, end):
        self._updating = True
        self.start_spin.setValue(start)
        self.end_spin.setValue(end)
        self.length.set_value(end - start)
        self._updating = False
        self.end_spin.setReadOnly(self.lock_length.isChecked())
        self.update_summary()

    def start_changed(self):
        if self._updating:
            return
        if self.lock_length.isChecked():
            self.end_spin.setValue(self.start_spin.value() + self.length.value())
        else:
            self.end_changed()
        self.update_summary()

    def end_changed(self):
        if self._updating:
            return
        if not self.lock_length.isChecked():
            self._updating = True
            self.length.set_value(self.end_spin.value() - self.start_spin.value())
            self._updating = False
        self.update_summary()

    def length_changed(self):
        if self._updating:
            return
        self.lock_length.setChecked(True)
        self.start_changed()

    def lock_changed(self):
        self.end_spin.setReadOnly(self.lock_length.isChecked())
        self.start_changed()

    def set_start_here(self):
        self.start_spin.setValue(self.current_seconds())

    def set_end_here(self):
        self.lock_length.setChecked(False)
        self.end_spin.setValue(self.current_seconds())

    def jump(self, direction):
        basis = {"current": self.current_seconds(), "start": self.start_spin.value(), "zero": 0.0}[self.seek_basis.currentData()]
        target = max(0, min(basis + direction * self.seek_amount.value(), self.duration - 1 / self.fps))
        self.seek_seconds(target)

    def update_summary(self):
        start, end = self.start_spin.value(), self.end_spin.value()
        text = tr('{v0} → {v1}\n길이 {v2}', v0=format_time(start), v1=format_time(end), v2=format_time(max(0, end - start)))
        try:
            validate_range(start, end, self.duration)
        except ValueError as error:
            text += f"\n{error}"
        self.summary.setText(text)


class CommittedTimecodeSpin(TimecodeSpin):
    """Distinguish an actual user commit from loading/restoring a boundary."""
    valueCommitted = Signal()

    def __init__(self):
        super().__init__()
        self.pending_text = None
        self.lineEdit().textEdited.connect(self.remember_edit)
        self.editingFinished.connect(self.commit_text)

    def remember_edit(self, text):
        self.pending_text = text

    def commit_text(self):
        text, self.pending_text = self.pending_text, None
        if text is not None and self.validate(text, len(text))[0] == QValidator.Acceptable:
            self.valueCommitted.emit()

    def stepBy(self, steps):
        before = self.value()
        super().stepBy(steps)
        self.pending_text = None
        if before != self.value():
            self.valueCommitted.emit()


class RoiRangeList(QListWidget):
    """Checkbox clicks change batch scope, never the currently viewed ROI."""
    def check_rect(self, item):
        option = QStyleOptionViewItem()
        self.itemDelegate().initStyleOption(option, self.indexFromItem(item))
        option.rect = self.visualItemRect(item)
        return self.style().subElementRect(QStyle.SE_ItemViewItemCheckIndicator, option, self)

    def mousePressEvent(self, event):
        item = self.itemAt(event.position().toPoint())
        self._check_pressed = bool(event.button() == Qt.LeftButton and item is not None
                                   and self.check_rect(item).contains(event.position().toPoint()))
        if self._check_pressed:
            item.setCheckState(Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if getattr(self, "_check_pressed", False):
            self._check_pressed = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        # The navigator must not consume the editor's playback/mark shortcuts
        # as list text search or toggle a copy checkbox on playback Space.
        if event.modifiers() in (Qt.NoModifier, Qt.ShiftModifier) and event.key() in (
            Qt.Key_Space, Qt.Key_A, Qt.Key_D, Qt.Key_Q, Qt.Key_E, Qt.Key_S, Qt.Key_F, Qt.Key_G,
        ):
            event.ignore()
            return
        super().keyPressEvent(event)


class SingleAnalysisRangeControls(QWidget):
    """One mouse per ROI: independent boundaries, optional one-shot duration."""
    rangeChanged = Signal()

    def __init__(self, duration, fps, start, end, current_seconds, seek_seconds, parent=None):
        super().__init__(parent)
        self.duration, self.fps = duration, fps
        self.current_seconds, self.seek_seconds = current_seconds, seek_seconds
        self._exact = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        section, fields = create_hagisilta_settings_section(tr('분석 시작·종료'))
        self.start_spin, self.end_spin = CommittedTimecodeSpin(), CommittedTimecodeSpin()
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        for row, (name, label, spin, key) in enumerate((
            ("start", tr('시작'), self.start_spin, "S"), ("end", tr('종료'), self.end_spin, "F"),
        )):
            grid.addWidget(QLabel(label, objectName="fieldLabel"), row * 2, 0, 1, 2)
            grid.addWidget(spin, row * 2 + 1, 0)
            button = QPushButton(tr('현재 장면 → {v0} ({v1})', v0=label, v1=key), objectName="toolButton")
            button.setFixedWidth(208)
            button.setAutoDefault(False)
            button.clicked.connect(lambda _checked=False, n=name: self.mark_here(n))
            setattr(self, name + "_button", button)
            grid.addWidget(button, row * 2 + 1, 1)
            spin.valueChanged.connect(lambda _value, n=name: self.boundary_changed(n))
            spin.valueCommitted.connect(lambda n=name: self.preview_boundary(n))
            spin.setToolTip(spin.toolTip() + tr('\n입력 확정 시 {v0} 장면으로 이동합니다. 다른 경계는 바뀌지 않습니다.', v0=label))
        fields.addLayout(grid)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        fields.addWidget(self.summary)
        self.mark_feedback = QLabel()
        self.mark_feedback.setWordWrap(True)
        fields.addWidget(self.mark_feedback)
        full = QPushButton(tr('영상 전체'), objectName="toolButton")
        full.setFixedWidth(100)
        full.setAutoDefault(False)
        full.clicked.connect(lambda: self.set_range(0, self.duration))
        fields.addWidget(full, 0, Qt.AlignRight)
        layout.addWidget(section)
        section, fields = create_hagisilta_settings_section(tr('기준 시각 + 시간'))
        grid = QGridLayout()
        self.seek_basis = QComboBox()
        for label, key in ((tr('현재 위치 기준'), "current"), (tr('분석 시작 기준'), "start"), (tr('영상 처음 기준'), "zero")):
            self.seek_basis.addItem(label, key)
        self.seek_basis.setCurrentIndex(1)
        self.seek_amount = TimecodeSpin(max(0, end - start))
        self.length = self.seek_amount  # One widget, not two independently entered times.
        grid.addWidget(self.seek_basis, 0, 0, 1, 2)
        grid.addWidget(self.seek_amount, 1, 0)
        back = QPushButton(tr('이 시간만큼 뒤로 (Alt+A)'), objectName="toolButton")
        back.setAutoDefault(False)
        back.clicked.connect(lambda: self.jump(-1))
        grid.addWidget(back, 1, 1)
        fields.addLayout(grid)
        actions = QHBoxLayout()
        self.offset_preview_button = QPushButton(tr('위치 보기 (Alt+D)'), objectName="toolButton")
        self.offset_preview_button.setAutoDefault(False)
        self.offset_preview_button.clicked.connect(lambda: self.jump(1))
        self.duration_button = QPushButton(tr('종료로 지정'), objectName="toolButton")
        self.duration_button.setAutoDefault(False)
        self.duration_button.clicked.connect(self.apply_duration)
        self.offset_preview_button.setToolTip(tr('기준 시각 + 입력한 시간의 장면만 봅니다. 시작·종료는 바뀌지 않습니다.'))
        self.duration_button.setToolTip(tr('기준 시각 + 입력한 시간을 종료로 지정하고 그 장면으로 이동합니다.'))
        actions.addWidget(self.offset_preview_button)
        actions.addWidget(self.duration_button)
        fields.addLayout(actions)
        layout.addWidget(section)
        self.set_range(start, end)

    def boundary(self, name):
        return self._exact.get(name, getattr(self, name + "_spin").value())

    def boundary_changed(self, name):
        self._exact.pop(name, None)
        self.update_summary()
        self.rangeChanged.emit()

    def set_boundary(self, name, value):
        spin = getattr(self, name + "_spin")
        blocked = spin.blockSignals(True)
        spin.setValue(value)
        spin.blockSignals(blocked)
        spin.pending_text = None
        self._exact[name] = float(value)
        self.update_summary()
        self.rangeChanged.emit()

    def set_range(self, start, end):
        self.set_boundary("start", start)
        self.set_boundary("end", end)

    def finish_editing(self):
        for spin in (self.start_spin, self.end_spin, self.seek_amount):
            spin.interpretText()

    def mark_here(self, name):
        seconds = self.current_seconds()
        self.set_boundary(name, seconds)
        label = tr('시작') if name == "start" else tr('종료')
        self.mark_feedback.setText(tr('{v0} 지정 · {v1} · {v2}프레임', v0=label, v1=format_time(seconds), v2=round(seconds * self.fps) + 1))

    def set_start_here(self):
        self.mark_here("start")

    def set_end_here(self):
        self.mark_here("end")

    def preview_boundary(self, name):
        value = self.boundary(name)
        if not 0 <= value <= self.duration + .00051:
            return
        if name == "end":
            try:
                validate_range(self.boundary("start"), value, self.duration)
            except ValueError:
                return
        self.seek_seconds(value)
        label = tr('시작') if name == "start" else tr('종료')
        self.mark_feedback.setText(tr('{v0} 장면 확인 · {v1}', v0=label, v1=format_time(value)))

    def apply_duration(self):
        self.finish_editing()
        end = self.offset_target(1)
        try:
            validate_range(self.boundary("start"), end, self.duration)
        except ValueError as error:
            self.mark_feedback.setText(str(error))
            return
        self.set_boundary("end", end)
        self.preview_boundary("end")

    def jump(self, direction):
        self.finish_editing()
        self.seek_seconds(max(0, min(self.offset_target(direction), self.duration - 1 / self.fps)))

    def offset_target(self, direction):
        basis = {"current": self.current_seconds(), "start": self.boundary("start"), "zero": 0}[self.seek_basis.currentData()]
        return basis + direction * self.seek_amount.value()

    def update_summary(self):
        start, end = self.boundary("start"), self.boundary("end")
        try:
            validate_range(start, end, self.duration)
            text = tr('분석 길이 · {v0}', v0=format_time(end - start))
        except ValueError as error:
            text = str(error)
        self.summary.setText(text)


class BatchAnalysisRangeDialog(QDialog):
    """Explicit target preview; invalid targets are never silently shortened."""
    def __init__(self, records, reference=(0.0, 600.0), parent=None):
        super().__init__(parent)
        self.records, self.reference = records, reference
        self.setWindowTitle(tr('HAGISILTA · 분석 구간 일괄 설정'))
        apply_hagisilta_window_identity(self)
        enable_full_window_controls(self)
        self.setStyleSheet(self.styleSheet() + hagisilta_control_stylesheet())
        self.resize(850, 580)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        header, heading = create_hagisilta_header(tr('분석 구간 일괄 설정'), "HAGISILTA · ANALYSIS RANGE")
        heading.addWidget(create_hagisilta_help_button(self, tr('구간 일괄 설정 도움말'), tr(RANGE_HELP) + tr('\n기본은 각 대상의 시작을 유지하고 길이만 맞춥니다. 목록을 확인한 뒤 적용하세요. 부족한 항목은 직접 제외를 선택해야 합니다.')))
        root.addWidget(header)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(18, 16, 18, 16)
        self.mode = QComboBox()
        self.mode.addItem(tr('각 시작 유지 · 길이만 적용'), "duration")
        self.mode.addItem(tr('기준 항목의 시작·종료 시각 복사'), "absolute")
        layout.addWidget(self.mode)
        row = QHBoxLayout()
        row.addWidget(QLabel(tr('실험 길이')))
        self.length = TimecodeSpin(max(0, reference[1] - reference[0]))
        row.addWidget(self.length)
        row.addStretch()
        layout.addLayout(row)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        layout.addWidget(self.preview, 1)
        self.skip_invalid = QCheckBox(tr('부족·오류 항목은 제외하고 적용'))
        layout.addWidget(self.skip_invalid)
        self.summary = QLabel()
        layout.addWidget(self.summary)
        root.addWidget(body, 1)
        footer = QFrame(objectName="detachedFooter")
        buttons = QHBoxLayout(footer)
        buttons.setContentsMargins(18, 9, 18, 9)
        buttons.addStretch()
        self.apply_button = QPushButton(tr('확인한 구간 적용'), objectName="primaryButton")
        self.apply_button.setFixedWidth(220)
        self.apply_button.clicked.connect(self.accept)
        cancel = QPushButton(tr('취소'), objectName="toolButton")
        cancel.clicked.connect(self.reject)
        cancel.setDefault(True)
        self.apply_button.setAutoDefault(False)
        buttons.addWidget(self.apply_button)
        buttons.addWidget(cancel)
        root.addWidget(footer)
        for signal in (self.mode.currentIndexChanged, self.length.valueChanged, self.skip_invalid.toggled):
            signal.connect(self.update_preview)
        self.update_preview()

    def update_preview(self):
        self.length.setEnabled(self.mode.currentData() == "duration")
        self.proposals = propose_ranges(self.records, self.mode.currentData(), self.length.value(), self.reference)
        lines = []
        for item in self.proposals:
            lines.append(f"{item['label']}\n  {format_time(item['start'])} → {format_time(item['end'])}  ⇒  "
                         f"{format_time(item['new_start'])} → {format_time(item['new_end'])}"
                         + (tr('\n  적용 불가: {v0}', v0=item['error']) if item['error'] else ""))
        self.preview.setPlainText("\n\n".join(lines))
        bad = sum(bool(item["error"]) for item in self.proposals)
        good = len(self.proposals) - bad
        self.summary.setText(tr('대상 {v0}개 · 적용 가능 {v1}개 · 부족/오류 {v2}개', v0=len(self.proposals), v1=good, v2=bad))
        self.apply_button.setEnabled(good > 0 and (not bad or self.skip_invalid.isChecked()))

    def accepted_proposals(self):
        return [item for item in self.proposals if not item["error"]]
