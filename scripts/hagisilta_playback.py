# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
"""Shared preview-only playback speed controls; never change analysis timing."""
from __future__ import annotations

from hagisilta_i18n import tr

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSizePolicy, QWidget


PLAYBACK_SPEEDS = (0.25, 0.5, 1.0, 1.5, 2.0, 4.0)
PLAYBACK_HELP = (
    "재생 배속: 0.25×~4× · 감속 (Ctrl+[) / 가속 (Ctrl+])\n"
    "배속은 미리보기 재생에만 적용됩니다. 프레임·시간 이동 폭, 원본 FPS와 분석 시간은 바꾸지 않습니다."
)


def playback_interval_ms(fps: float, speed: float = 1.0) -> int:
    """Keep each decoded frame, including QC stop points, at the requested rate."""
    fps, speed = float(fps), float(speed)
    if not math.isfinite(fps) or fps <= 0:
        fps = 1.0
    if not math.isfinite(speed) or speed <= 0:
        speed = 1.0
    return max(1, round(1000.0 / (fps * speed)))


class _SpeedCombo(QComboBox):
    def wheelEvent(self, event):
        # Scrolling a nearby inspector must not change preview speed by accident.
        event.ignore()


class PlaybackSpeedControl(QWidget):
    speedChanged = Signal(float)

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setObjectName("playbackSpeedControl")
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.label = QLabel(tr('재생 배속'), self, objectName="fieldLabel")
        row.addWidget(self.label)
        self.combo = _SpeedCombo(self)
        self.combo.setObjectName("playbackSpeedCombo")
        self.combo.setAccessibleName(tr('재생 배속'))
        self.combo.setFixedWidth(84)
        self.combo.setMinimumHeight(28)
        for speed in PLAYBACK_SPEEDS:
            self.combo.addItem(f"{speed:g}×", speed)
        self.combo.setCurrentIndex(PLAYBACK_SPEEDS.index(1.0))
        self.label.setBuddy(self.combo)
        row.addWidget(self.combo)
        self.setToolTip(tr(PLAYBACK_HELP) + tr('\n영상 처리 성능에 따라 실제 재생 속도는 느려질 수 있습니다.'))
        self.combo.setToolTip(self.toolTip())
        self.combo.currentIndexChanged.connect(lambda _index: self.speedChanged.emit(self.speed()))
        self.shortcuts = []
        for key, direction in (("Ctrl+[", -1), ("Ctrl+]", 1)):
            shortcut = QShortcut(QKeySequence(key), parent)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(lambda step=direction: self.cycle(step))
            self.shortcuts.append(shortcut)

    def speed(self) -> float:
        return float(self.combo.currentData() or 1.0)

    def cycle(self, direction: int) -> None:
        index = max(0, min(self.combo.count() - 1, self.combo.currentIndex() + int(direction)))
        self.combo.setCurrentIndex(index)
