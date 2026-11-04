# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from hagisilta_i18n import tr

import ctypes
import sys
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QTimer, Qt
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)
from PySide6.QtGui import QIcon


class WheelValueChangeGuard(QObject):
    """Prevent page scrolling from changing hovered numeric/select controls."""

    def eventFilter(self, watched, event) -> bool:
        if event.type() != QEvent.Wheel or not isinstance(
            watched, (QAbstractSpinBox, QComboBox)
        ):
            return super().eventFilter(watched, event)

        # Keep the scrolling gesture useful by moving the nearest scroll page.
        parent = watched.parentWidget()
        while parent is not None and not isinstance(parent, QAbstractScrollArea):
            parent = parent.parentWidget()
        if isinstance(parent, QAbstractScrollArea):
            delta = event.angleDelta()
            if abs(delta.x()) > abs(delta.y()):
                bar = parent.horizontalScrollBar()
                amount = delta.x()
            else:
                bar = parent.verticalScrollBar()
                amount = delta.y()
            if amount:
                steps = amount / 120.0
                bar.setValue(round(bar.value() - steps * bar.singleStep() * 3))
        event.accept()
        return True


def install_wheel_value_guard(app: QApplication | None = None) -> None:
    """Install one application-wide accidental wheel-change guard."""
    app = app or QApplication.instance()
    if app is None or getattr(app, "_hagisilta_wheel_value_guard", None):
        return
    guard = WheelValueChangeGuard(app)
    app._hagisilta_wheel_value_guard = guard
    app.installEventFilter(guard)


def hagisilta_action_stylesheet() -> str:
    """The same borderless action/status palette in main and detached windows.

    Transparent borders retain control geometry; keyboard focus remains visible.
    Do not apply this to image overlays, table grids or editable input outlines.
    """
    neutral = ", ".join(("QPushButton", "QToolButton", *(
        f"QPushButton#{name}" for name in
        ("secondaryButton", "secondary", "toolButton", "stageActionButton", "stepButton", "shapeToolButton")
    ), "QToolButton#toolButton"))
    primary = ", ".join(f"QPushButton#{name}" for name in
                        ("primaryButton", "primary", "stageActionNext", "quickRunButton", "quickReviewButton"))
    danger = ", ".join(f"QPushButton#{name}" for name in
                       ("dangerButton", "jobStopButton", "jobRailStopButton"))

    def states(selectors, suffix):
        return ", ".join(selector + suffix for selector in selectors.split(", "))

    return f"""
        QPushButton, QToolButton {{ min-height:32px; padding:0 12px; font-weight:600; }}
        {neutral} {{
            background:#4d535e; color:#ffffff; border:1px solid transparent;
            border-radius:2px;
        }}
        {states(neutral, ':hover')} {{ background:#606976; border-color:transparent; }}
        {states(neutral, ':pressed')} {{ background:#3d434d; border-color:transparent; }}
        QPushButton#toolButton, QToolButton#toolButton {{
            min-height:28px; max-height:28px; padding:0 9px; font-size:11px;
        }}
        QPushButton#toolButton:checked, QPushButton#shapeToolButton:checked {{
            background:#3f6dcc; color:#ffffff; border-color:transparent;
        }}
        {primary} {{ background:#3f6dcc; color:#ffffff; border:1px solid transparent; }}
        {states(primary, ':hover')} {{ background:#4b79da; border-color:transparent; }}
        {states(primary, ':pressed')} {{ background:#315cad; border-color:transparent; }}
        {danger} {{ background:#803c4a; color:#ffffff; border:1px solid transparent; }}
        {states(danger, ':hover')} {{ background:#984959; border-color:transparent; }}
        {states(danger, ':pressed')} {{ background:#6c303d; border-color:transparent; }}
        {states(neutral + ', ' + primary + ', ' + danger, ':focus')} {{ border-color:#9ab8ff; }}
        {states(neutral + ', ' + primary + ', ' + danger, ':disabled')} {{
            background:#343840; color:#969da8; border-color:transparent;
        }}
        QLabel#stepBadge, QLabel#stepBadgeReady, QLabel#detachedStatus,
        QLabel#experimentBadge, QLabel#labelSummaryBadge, QLabel#jobStatusBadge {{
            background:#424951; color:#ffffff; border:1px solid transparent;
            border-radius:2px;
        }}
        QLabel#stepBadgeReady {{ background:#26784f; color:#ffffff; }}
        QLabel#jobStatusBadge[active="true"] {{ background:#345794; color:#ffffff; }}
        QLabel#jobStatusBadge[error="true"] {{ background:#803c4a; color:#ffffff; }}
        QLabel#pipelineChip {{ background:#353a43; color:#ffffff; border:1px solid transparent; }}
        QLabel#workflowStepNumber {{ background:#424951; color:#ffffff; border:1px solid transparent; }}
        QFrame#statusSummaryBar, QFrame#reviewSelectionPanel {{
            background:#2c3037; border:none; border-radius:2px;
        }}
        QLabel#contextMeta, QLabel#infoBox {{ background:#2c3037; border:1px solid transparent; }}
        QFrame#filterWorkflow {{ background:transparent; border:none; }}
        QFrame#settingsSection, QFrame#settingsSectionHeader {{ background:transparent; border:none; }}
        QLabel#settingsSectionTitle {{ background:transparent; color:#c5c9d0; font-size:12px; font-weight:600; }}
        QLabel#settingsSectionNumber {{
            background:transparent; color:#969da8; border:none;
            font-size:11px; font-weight:600;
        }}
        QFrame#settingsSectionRule {{ background:#3c4148; border:none; }}
        QFrame#settingsSectionContent, QWidget#filterSettingsFields,
        QWidget#filterUniformFields, QWidget#filterAnalysisFields,
        QWidget#filterInspector, QWidget#recipeFields {{ background:transparent; }}
        QFrame#settingsSectionContent QLabel[objectName=""] {{ background:transparent; }}
        QFrame#settingsSectionContent QLabel#pageText {{ background:transparent; color:#b5b8bd; font-size:11px; }}
    """


def create_hagisilta_settings_section(title: str, step: int | None = None) -> tuple[QFrame, QVBoxLayout]:
    """Quiet subsection: text and a hairline, never a second workflow header."""
    section = QFrame(objectName="settingsSection")
    layout = QVBoxLayout(section)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    header = QFrame(objectName="settingsSectionHeader")
    heading = QHBoxLayout(header)
    heading.setContentsMargins(4, 4, 4, 4)
    heading.setSpacing(8)
    if step is not None:
        number = QLabel(f"{step}.", objectName="settingsSectionNumber")
        number.setFixedWidth(14)
        number.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        heading.addWidget(number)
    heading.addWidget(QLabel(title, objectName="settingsSectionTitle"))
    rule = QFrame(objectName="settingsSectionRule")
    rule.setFixedHeight(1)
    heading.addWidget(rule, 1)
    layout.addWidget(header)
    content = QFrame(objectName="settingsSectionContent")
    body = QVBoxLayout(content)
    body.setContentsMargins(4, 8, 4, 4)
    body.setSpacing(10)
    layout.addWidget(content)
    return section, body


def hagisilta_control_stylesheet() -> str:
    """Shared native-control styling for detached HAGISILTA work windows."""
    assets = Path(__file__).resolve().parents[1] / "assets"
    spin_up = (assets / "spin-up.svg").as_posix()
    spin_down = (assets / "spin-down.svg").as_posix()
    check_icon = (assets / "check.svg").as_posix()
    return """
        QFrame#detachedHeader {
            background:#202125; border:none; border-bottom:1px solid #41434a;
        }
        QFrame#detachedTitleAccent { background:#5b82e8; border:none; }
        QLabel#detachedKicker {
            background:transparent; color:#7f98d8; font-size:9px;
            font-weight:700; letter-spacing:1px;
        }
        QLabel#detachedTitle {
            background:transparent; color:#f2f3f5; font-size:18px; font-weight:650;
        }
        QLabel#detachedStatus {
            background:#292a2f; color:#c5c7cb; border:1px solid #4b4e55;
            border-radius:2px; padding:4px 9px; font-size:11px; font-weight:600;
        }
        QFrame#detachedBody { background:#23242a; border:none; }
        QFrame#reviewWorkspace { background:transparent; border:none; }
        QFrame#detachedSidebar {
            background:#202125; border:1px solid #41434a; border-radius:2px;
        }
        QFrame#settingsGroup {
            background:#27282d; border:1px solid #464950; border-radius:2px;
        }
        QLabel#settingsGroupTitle {
            background:transparent; color:#e4e5e8; font-size:12px; font-weight:650;
        }
        QLabel#settingsGroupDescription {
            background:transparent; color:#969aa2; font-size:11px;
        }
        QLabel#sectionTitle { border:1px solid transparent; }
        QListWidget#decisionPlaylist {
            background:#202125; border:none; outline:none; padding:0;
        }
        QListWidget#decisionPlaylist::item {
            background:#292a2f; color:#d9dbe0; border:1px solid #42454c;
            border-radius:2px; padding:6px 9px;
        }
        QListWidget#decisionPlaylist::item:hover {
            background:#343740; border-color:#66718b;
        }
        QListWidget#decisionPlaylist::item:selected {
            background:#4567c8; color:#ffffff; border-color:#6f8ce5;
        }
        QSplitter#reviewPlaylistSplitter::handle:horizontal {
            background:#35383f; border-left:1px solid #50545c;
            border-right:1px solid #24262b;
        }
        QSplitter#reviewPlaylistSplitter::handle:horizontal:hover,
        QSplitter#reviewPlaylistSplitter::handle:horizontal:pressed {
            background:#5b82e8; border-color:#7f9ded;
        }
        QFrame#statusSummaryBar {
            background:#202125; border:1px solid #41434a; border-radius:2px;
        }
        QLabel#statusSummaryTitle {
            background:transparent; color:#aeb2b9; font-size:10px; font-weight:650;
        }
        QLabel#statusSummaryValue {
            background:transparent; color:#eceef1; font-size:12px; font-weight:600;
        }
        QFrame#detachedFooter {
            background:#202125; border:none; border-top:1px solid #41434a;
            min-height:48px;
        }
        QFrame#detachedFooter QPushButton#primaryButton { min-width:220px; }
        QFrame#detachedFooter QPushButton#secondaryButton { min-width:88px; }
        QCheckBox { spacing:8px; background:transparent; color:#e4e5e7; }
        QCheckBox::indicator {
            width:17px; height:17px; border:1px solid #6b7480;
            border-radius:3px; background:#23242b;
        }
        QCheckBox::indicator:hover { border-color:#9baeea; background:#343943; }
        QCheckBox::indicator:checked {
            image:url(%s); border-color:#7188da; background:#4560c4;
        }
        QListWidget::indicator {
            width:17px; height:17px; border:1px solid #6b7480;
            border-radius:3px; background:#23242b;
        }
        QListWidget::indicator:hover { border-color:#9baeea; background:#343943; }
        QListWidget::indicator:checked {
            image:url(%s); border-color:#7188da; background:#4560c4;
        }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background:#1f2024; color:#f0f1f2; border:1px solid #50535a;
            border-radius:2px; padding:4px 7px; min-height:21px;
            selection-background-color:#4560c4;
        }
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color:#7288da; }
        QComboBox QAbstractItemView {
            background:#202125; color:#f0f1f2; border:1px solid #66738a;
            selection-background-color:#4560c4; selection-color:#ffffff;
        }
        QSpinBox, QDoubleSpinBox { padding-right:34px; }
        QSpinBox::up-button, QDoubleSpinBox::up-button,
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            width:28px; background:#34363b; border-left:1px solid #50535a;
            border-radius:0;
        }
        QSpinBox::up-button, QDoubleSpinBox::up-button { border-bottom:1px solid #50535a; }
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background:#484b52; }
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
            image:url(%s); width:12px; height:8px;
        }
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
            image:url(%s); width:12px; height:8px;
        }
        QComboBox { padding-right:30px; }
        QComboBox::drop-down {
            subcontrol-origin:padding; subcontrol-position:top right;
            width:27px; border-left:1px solid #50535a; background:#34363b;
        }
        QComboBox::drop-down:hover { background:#484b52; }
        QComboBox::down-arrow { image:url(%s); width:12px; height:8px; }
        QSlider::groove:horizontal {
            height:6px; background:#3b3d43; border-radius:3px;
        }
        QSlider::sub-page:horizontal { background:#4560c4; border-radius:3px; }
        QSlider::handle:horizontal {
            width:16px; margin:-5px 0; background:#f0f1f2;
            border:2px solid #7188da; border-radius:8px;
        }
        QScrollBar:vertical {
            background:#18191c; width:14px; margin:0; border:1px solid #36383e;
        }
        QScrollBar::handle:vertical {
            background:#767b84; min-height:34px; border:1px solid #949aa4;
            border-radius:3px; margin:2px;
        }
        QScrollBar::handle:vertical:hover { background:#9298a2; border-color:#b1b6bf; }
        QScrollBar:horizontal {
            background:#18191c; height:14px; margin:0; border:1px solid #36383e;
        }
        QScrollBar::handle:horizontal {
            background:#767b84; min-width:34px; border:1px solid #949aa4;
            border-radius:3px; margin:2px;
        }
        QScrollBar::handle:horizontal:hover { background:#9298a2; border-color:#b1b6bf; }
        QScrollBar::add-page, QScrollBar::sub-page { background:#18191c; }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width:0; background:transparent; border:none;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height:0; background:transparent; border:none;
        }
        QScrollBar::left-arrow, QScrollBar::right-arrow,
        QScrollBar::up-arrow, QScrollBar::down-arrow { image:none; }
    """ % (check_icon, check_icon, spin_up, spin_down, spin_down) + hagisilta_action_stylesheet()


class HeaderContext(QObject):
    """Non-visual header feedback; existing editor updates become hover details."""

    def __init__(self, header: QFrame, text: str = "") -> None:
        super().__init__(header)
        self.setText(text)

    def setText(self, text: str) -> None:
        self._text = str(text)
        self.parent().setToolTip(self._text)

    def text(self) -> str:
        return self._text


def create_hagisilta_header(title: str, kicker: str = "HAGISILTA WORKSPACE") -> tuple[QFrame, QHBoxLayout]:
    """Build the title/help shell. Context belongs in hover text, not a status badge."""
    header = QFrame(objectName="detachedHeader")
    layout = QHBoxLayout(header)
    layout.setContentsMargins(18, 10, 18, 10)
    layout.setSpacing(10)
    accent = QFrame(objectName="detachedTitleAccent")
    accent.setFixedSize(3, 36)
    layout.addWidget(accent)
    title_stack = QVBoxLayout()
    title_stack.setContentsMargins(0, 0, 0, 0)
    title_stack.setSpacing(0)
    title_stack.addWidget(QLabel(kicker, objectName="detachedKicker"))
    title_stack.addWidget(QLabel(title, objectName="detachedTitle"))
    layout.addLayout(title_stack)
    layout.addStretch(1)
    return header, layout


def create_hagisilta_help_button(parent, title: str, text: str) -> QPushButton:
    """Create the one standard help entry used by detached work windows."""
    button = QPushButton(tr('도움말'), objectName="toolButton")
    button.setFixedWidth(88)
    button.setToolTip(tr('이 창의 작업 방법과 단축키를 확인합니다.'))
    button.clicked.connect(lambda: QMessageBox.information(parent, title, text))
    return button


def apply_hagisilta_window_identity(widget) -> None:
    """Give every detached HAGISILTA window the same icon and dark title bar."""
    icon_path = Path(__file__).resolve().parents[1] / "assets" / "hagisilta-app.ico"
    if icon_path.is_file():
        widget.setWindowIcon(QIcon(str(icon_path)))
    apply_hagisilta_titlebar(widget)


def enable_full_window_controls(widget) -> None:
    """Give tool dialogs the same minimize/maximize/close controls as app windows."""
    widget.setWindowFlags(
        Qt.Window
        | Qt.WindowTitleHint
        | Qt.WindowSystemMenuHint
        | Qt.WindowMinimizeButtonHint
        | Qt.WindowMaximizeButtonHint
        | Qt.WindowCloseButtonHint
    )


def apply_hagisilta_titlebar(widget) -> None:
    """Apply the HAGISILTA dark caption palette on supported Windows builds."""
    if sys.platform != "win32":
        return

    def apply() -> None:
        try:
            hwnd = int(widget.winId())
            dwm = ctypes.windll.dwmapi
            dark = ctypes.c_int(1)
            for attribute in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE
                try:
                    dwm.DwmSetWindowAttribute(
                        hwnd, attribute, ctypes.byref(dark), ctypes.sizeof(dark)
                    )
                    break
                except Exception:
                    continue

            # COLORREF is 0x00BBGGRR.
            colors = {
                35: 0x00252120,  # DWMWA_CAPTION_COLOR -> #202125
                36: 0x00F5F3F2,  # DWMWA_TEXT_COLOR -> #f2f3f5
                34: 0x004A4341,  # DWMWA_BORDER_COLOR -> #41434a
            }
            for attribute, color_value in colors.items():
                color = ctypes.c_int(color_value)
                dwm.DwmSetWindowAttribute(
                    hwnd, attribute, ctypes.byref(color), ctypes.sizeof(color)
                )
        except Exception:
            # Older Windows builds simply keep the native caption.
            pass

    QTimer.singleShot(0, apply)
