# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)
from hagisilta_window_chrome import install_wheel_value_guard
from hagisilta_startup import StartupReadiness
from hagisilta_i18n import init_i18n, tr


ROOT = Path(__file__).resolve().parents[1]
APP_SCRIPT = ROOT / "scripts" / "hagisilta_platform.py"
SPLASH_IMAGE = ROOT / "assets" / "hagisilta-splash-artwork-v4.png"
VERSION_FILE = ROOT / "VERSION.txt"
APP_ICON = ROOT / "assets" / "hagisilta-app.ico"
SPLASH_CROP_SIDE = 140


class StartupSplash(QWidget):
    def __init__(self) -> None:
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setObjectName("startupSplash")
        self.setFixedSize(1110, 512)
        self.setStyleSheet(
            """
            QWidget#startupSplash {
                background: #090f18;
                border: none;
                font-family: "Segoe UI";
            }
            QFrame#startupPanel {
                background: #0b121d;
                border: none;
                border-right: 1px solid #243248;
            }
            QLabel#startupAppIcon {
                background: transparent;
                border: none;
            }
            QLabel#startupTitle {
                color: #f2f6fb;
                font-size: 29px;
                font-weight: 700;
                letter-spacing: 1px;
            }
            QLabel#startupSubtitle {
                color: #a8b6c8;
                font-size: 12px;
                letter-spacing: 1px;
            }
            QLabel#startupDescription {
                color: #708198;
                font-size: 11px;
                line-height: 1.45;
            }
            QLabel#startupVersion {
                color: #718198;
                font-size: 10px;
                font-weight: 700;
            }
            QLabel#startupCopyright {
                color: #53657c;
                font-size: 9px;
                font-weight: 600;
                letter-spacing: .3px;
            }
            QLabel#startupArtworkCredit {
                color: #687a91;
                font-size: 9px;
                font-weight: 600;
                letter-spacing: .3px;
            }
            QLabel#startupStatus {
                color: #d9e3ee;
                font-size: 11px;
                font-weight: 600;
            }
            QProgressBar#startupProgress {
                min-height: 3px;
                max-height: 3px;
                border: none;
                background: #1a2636;
                text-align: center;
            }
            QProgressBar#startupProgress::chunk {
                background: #63dcb4;
                border: none;
            }
            """
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        panel = QFrame(objectName="startupPanel")
        panel.setFixedWidth(330)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(28, 28, 28, 25)
        panel_layout.setSpacing(0)

        monogram_row = QHBoxLayout()
        app_icon = QLabel(objectName="startupAppIcon")
        app_icon.setFixedSize(48, 48)
        app_icon.setAlignment(Qt.AlignCenter)
        app_icon.setPixmap(QIcon(str(APP_ICON)).pixmap(44, 44))
        monogram_row.addWidget(app_icon)
        monogram_row.addStretch()
        panel_layout.addLayout(monogram_row)
        panel_layout.addSpacing(58)

        panel_layout.addWidget(QLabel("HAGISILTA", objectName="startupTitle"))
        panel_layout.addSpacing(5)
        panel_layout.addWidget(QLabel("Behavior Analysis Platform", objectName="startupSubtitle"))
        panel_layout.addSpacing(22)
        description = QLabel(
            "Tracking · Model Development · Experiment Analysis",
            objectName="startupDescription",
        )
        description.setWordWrap(False)
        panel_layout.addWidget(description)
        panel_layout.addStretch()

        panel_layout.addWidget(
            QLabel("Artwork by Song Jun-Hwi.", objectName="startupArtworkCredit")
        )
        panel_layout.addSpacing(6)
        copyright_label = QLabel(
            "Copyright © 2026 Ok Seungwon and Song Jun-Hwi.\n"
            "HAGISILTA: Apache License 2.0.\n"
            "Qt: © The Qt Company Ltd.\n"
            "DeepLabCut: © A. & M.W. Mathis Labs.\n"
            "Qt / DLC: LGPLv3. See LICENSE_GUIDE.md.",
            objectName="startupCopyright",
        )
        copyright_label.setWordWrap(True)
        panel_layout.addWidget(copyright_label)
        panel_layout.addSpacing(14)

        self.status_label = QLabel(tr("환경과 분석 모듈을 준비하는 중…"), objectName="startupStatus")
        self.status_label.setWordWrap(True)
        panel_layout.addWidget(self.status_label)
        panel_layout.addSpacing(11)
        self.progress = QProgressBar(objectName="startupProgress")
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        panel_layout.addWidget(self.progress)
        panel_layout.addSpacing(15)
        version = VERSION_FILE.read_text(encoding="utf-8").strip() if VERSION_FILE.is_file() else "-"
        self.version_label = QLabel(
            f"v{version}   ·   DeepLabCut 3.0   ·   PyTorch",
            objectName="startupVersion",
        )
        panel_layout.addWidget(self.version_label)
        layout.addWidget(panel)

        artwork = QLabel()
        artwork.setFixedSize(780, 512)
        artwork.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap(str(SPLASH_IMAGE))
        if not pixmap.isNull():
            pixmap = pixmap.copy(
                SPLASH_CROP_SIDE,
                0,
                pixmap.width() - (SPLASH_CROP_SIDE * 2),
                pixmap.height(),
            )
            scaled_artwork = pixmap.scaled(
                artwork.size(),
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation,
            )
            crop_x = max(0, (scaled_artwork.width() - artwork.width()) // 2)
            crop_y = max(0, (scaled_artwork.height() - artwork.height()) // 2)
            artwork.setPixmap(
                scaled_artwork.copy(crop_x, crop_y, artwork.width(), artwork.height())
            )
        else:
            fallback = QPixmap(artwork.size())
            fallback.fill(QColor("#0b1422"))
            painter = QPainter(fallback)
            painter.setPen(QColor("#55d6a6"))
            painter.drawText(fallback.rect(), Qt.AlignCenter, "HAGISILTA")
            painter.end()
            artwork.setPixmap(fallback)
        layout.addWidget(artwork)

        self.child: subprocess.Popen | None = None
        self.readiness: StartupReadiness | None = None
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(180)
        self.poll_timer.timeout.connect(self.poll_child)
        self.elapsed_ticks = 0

    def start_application(self) -> None:
        pythonw = Path(sys.executable).with_name("pythonw.exe")
        executable = pythonw if pythonw.is_file() else Path(sys.executable)
        try:
            self.readiness = StartupReadiness()
            self.child = subprocess.Popen(
                [str(executable), "-X", "utf8", str(APP_SCRIPT)],
                cwd=str(ROOT),
                env=self.readiness.child_environment(),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError:
            self.show_startup_failure()
            return
        self.poll_timer.start()

    def release_readiness(self) -> None:
        if self.readiness is not None:
            self.readiness.close()
            self.readiness = None

    def closeEvent(self, event) -> None:
        self.poll_timer.stop()
        self.release_readiness()
        super().closeEvent(event)

    def show_startup_failure(self) -> None:
        self.poll_timer.stop()
        self.release_readiness()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status_label.setText(tr("실행하지 못했습니다. 설치 복구를 실행하세요."))
        QTimer.singleShot(3500, QApplication.instance().quit)

    def poll_child(self) -> None:
        if self.child is None:
            return
        self.elapsed_ticks += 1
        if self.elapsed_ticks == 6:
            self.status_label.setText(tr("프로젝트와 장치 상태를 확인하는 중…"))
        elif self.elapsed_ticks == 14:
            self.status_label.setText(tr("HAGISILTA 작업공간을 여는 중…"))
        if self.child.poll() is not None:
            self.show_startup_failure()
            return
        if self.readiness is not None and self.readiness.is_ready(self.child.pid):
            self.poll_timer.stop()
            self.close()
            QApplication.instance().quit()


def main() -> int:
    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "HAGISILTA.BehaviorAnalysis.Platform"
            )
        except Exception:
            pass
    app = QApplication(sys.argv)
    init_i18n(app)
    install_wheel_value_guard(app)
    app.setApplicationName("HAGISILTA Startup")
    if APP_ICON.is_file():
        app.setWindowIcon(QIcon(str(APP_ICON)))
    splash = StartupSplash()
    app.aboutToQuit.connect(splash.release_readiness)
    splash.show()
    splash.move(
        splash.screen().availableGeometry().center() - splash.rect().center()
    )
    QTimer.singleShot(80, splash.start_application)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
