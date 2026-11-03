# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
"""Per-launch readiness handshake, emitted only after the main content paints."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys
import tempfile

from PySide6.QtCore import QEvent, QObject, QTimer
from PySide6.QtWidgets import QApplication, QWidget


READY_DIRECTORY_ENV = "HAGISILTA_STARTUP_READY_DIRECTORY"
READY_FILENAME = "main-window-ready"


class StartupReadiness:
    """Private, short-lived IPC directory; never uses machine/project settings."""

    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="hagisilta-startup-")
        self.directory = Path(self._temporary.name)

    def child_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment[READY_DIRECTORY_ENV] = str(self.directory)
        return environment

    def is_ready(self, process_id: int) -> bool:
        try:
            return (self.directory / READY_FILENAME).read_bytes() == str(process_id).encode("ascii")
        except OSError:
            return False

    def close(self) -> None:
        self._temporary.cleanup()


class FirstPaintNotifier(QObject):
    """Watch only the completed main window, not arbitrary app-owned windows."""

    def __init__(self, window: QWidget, content: QWidget, directory: str) -> None:
        super().__init__(window)
        self.window = window
        self.content = content
        self.directory = Path(directory)
        self.completed = False
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.publish_ready)
        content.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:
        if (watched is self.content and event.type() == QEvent.Paint
                and not self.completed and not self.timer.isActive()
                and self.window.isVisible() and not self.window.isMinimized()):
            # A queued callback runs after this paint and its child paints return.
            self.timer.start(0)
        return super().eventFilter(watched, event)

    def publish_ready(self) -> None:
        if self.completed or not self.window.isVisible() or self.window.isMinimized():
            return
        if sys.platform == "win32" and QApplication.platformName() == "windows":
            try:
                # Wait for the compositor to present the frame before uncovering it.
                ctypes.windll.dwmapi.DwmFlush()
            except (AttributeError, OSError):
                pass
        temporary = self.directory / f"ready-{os.getpid()}.tmp"
        try:
            temporary.write_bytes(str(os.getpid()).encode("ascii"))
            temporary.replace(self.directory / READY_FILENAME)
        except OSError:
            # The splash may have been closed by the user; never interrupt the app.
            if self.directory.is_dir():
                self.timer.start(180)
            return
        self.completed = True
        self.content.removeEventFilter(self)


def watch_main_window(window: QWidget, directory: str | None) -> FirstPaintNotifier | None:
    """Direct app launches need no handshake; a launcher opts in via its child env."""
    if not directory:
        return None
    central = getattr(window, "centralWidget", lambda: None)()
    return FirstPaintNotifier(window, central if central is not None else window, directory)
