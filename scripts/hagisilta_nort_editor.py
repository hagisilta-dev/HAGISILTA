# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from hagisilta_i18n import init_i18n, tr

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal, QTimer
from PySide6.QtGui import (
    QColor, QBrush, QImage, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QPolygonF, QShortcut
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from hagisilta_window_chrome import (
    HeaderContext,
    apply_hagisilta_window_identity, create_hagisilta_header,
    hagisilta_control_stylesheet, install_wheel_value_guard,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "setups"
CAMERA_PROFILE_DIR = REPO_ROOT / "analysis" / "camera_profiles"
COLORS = {"A": QColor("#ffb020"), "B": QColor("#4aa8ff")}


def cosmetic_pen(color: QColor, width: float = 2.0, style=Qt.SolidLine) -> QPen:
    pen = QPen(color, width, style)
    pen.setCosmetic(True)
    return pen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HAGISILTA NORT region and calibration setup")
    parser.add_argument("video", nargs="?")
    parser.add_argument("--time", type=float, default=275.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--roi-only", action="store_true", help="Only define and save arena ROIs")
    parser.add_argument("--profile-output", type=Path)
    parser.add_argument("--profile", type=Path, help="Existing ROI/camera profile to edit")
    parser.add_argument(
        "--single-arena", action="store_true",
        help="Treat an already-cropped arena video as one full-frame ROI and begin at cm calibration",
    )
    parser.add_argument("--arena-id", type=int, default=1, help="Original arena number for saved metadata")
    parser.add_argument(
        "--bundle-manifest", type=Path,
        help="Open one source video with every tracked Arena locked and save one setup per cropped Arena",
    )
    return parser.parse_args()


def frame_at(video: Path, seconds: float) -> tuple[np.ndarray, dict]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(tr('영상을 열 수 없습니다: {p0}', p0=video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = count / fps if fps else 0.0
    seconds = min(max(seconds, 0.0), max(duration - 1.0 / max(fps, 1.0), 0.0))
    capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(tr('{p0:.2f}초 프레임을 읽지 못했습니다.', p0=seconds))
    return frame, {
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "fps": fps,
        "frame_count": count,
        "duration_seconds": duration,
        "setup_frame_seconds": seconds,
    }


def cv_to_pixmap(frame: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(image.copy())


class CanvasView(QGraphicsView):
    canvas_clicked = Signal(QPointF)
    canvas_dragged = Signal(QPointF, QPointF)
    canvas_drag_updated = Signal(QPointF, QPointF)
    canvas_painted = Signal(QPointF, bool)
    canvas_hovered = Signal(QPointF)
    brush_size_delta = Signal(int)

    def __init__(self, scene: QGraphicsScene) -> None:
        super().__init__(scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(QColor("#090e17"))
        self._press: QPointF | None = None
        self._pan_start = None
        self.creation_enabled = False
        self.paint_enabled = False
        self.click_through_items = False

    def wheelEvent(self, event) -> None:  # noqa: N802
        if self.paint_enabled and event.modifiers() & Qt.ControlModifier:
            self.brush_size_delta.emit(1 if event.angleDelta().y() > 0 else -1)
            event.accept()
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        point = self.mapToScene(event.position().toPoint())
        if event.button() == Qt.MiddleButton:
            self._pan_start = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if self.paint_enabled and event.button() in (Qt.LeftButton, Qt.RightButton):
            self.canvas_painted.emit(point, event.button() == Qt.LeftButton)
            event.accept()
            return
        hit_item = self.itemAt(event.position().toPoint())
        if event.button() == Qt.LeftButton and (
            self.click_through_items or not hasattr(hit_item, "scene_polygon")
        ):
            self._press = point
            if self.click_through_items:
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MiddleButton and self._pan_start is not None:
            self._pan_start = None
            self.unsetCursor()
            event.accept()
            return
        if self.paint_enabled and event.button() in (Qt.LeftButton, Qt.RightButton):
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._press is not None:
            end = self.mapToScene(event.position().toPoint())
            start = self._press
            self._press = None
            if self.creation_enabled and abs(end.x() - start.x()) + abs(end.y() - start.y()) > 8:
                self.canvas_dragged.emit(start, end)
            else:
                self.canvas_clicked.emit(end)
            if self.click_through_items:
                event.accept()
                return
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._pan_start is not None:
            current = event.position().toPoint()
            delta = current - self._pan_start
            self._pan_start = current
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        if self.paint_enabled and event.buttons() & (Qt.LeftButton | Qt.RightButton):
            point = self.mapToScene(event.position().toPoint())
            self.canvas_painted.emit(point, bool(event.buttons() & Qt.LeftButton))
            event.accept()
            return
        if self.creation_enabled and self._press is not None and event.buttons() & Qt.LeftButton:
            self.canvas_drag_updated.emit(
                self._press, self.mapToScene(event.position().toPoint())
            )
            event.accept()
            return
        self.canvas_hovered.emit(self.mapToScene(event.position().toPoint()))
        super().mouseMoveEvent(event)


class EditableShapeItem(QGraphicsItem):
    handle_size = 10.0
    handle_hit_size = 18.0
    rotation_hit_size = 20.0
    rotate_offset = 28.0

    def __init__(self, shape: str, rect: QRectF, color: QColor, label: str) -> None:
        super().__init__()
        self.shape_kind = shape
        self.rect = QRectF(-rect.width() / 2, -rect.height() / 2, rect.width(), rect.height())
        self.color = color
        self.label = label
        self.interaction_mode = "edit"
        self.mode: str | None = None
        self.active_handle: tuple[int, int] | None = None
        self.start_rect = QRectF()
        self.start_scene_angle = 0.0
        self.start_rotation = 0.0
        self.hovered_handle: tuple[int, int] | str | None = None
        self.setPos(rect.center())
        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsFocusable)
        self.setAcceptHoverEvents(True)
        self.setZValue(30)

    def boundingRect(self) -> QRectF:  # noqa: N802
        return self.rect.adjusted(-160, -160, 160, 160)

    def view_scale(self) -> float:
        if self.scene() and self.scene().views():
            return max(abs(self.scene().views()[0].transform().m11()), 0.05)
        return 1.0

    def shape_path(self) -> QPainterPath:
        path = QPainterPath()
        if self.shape_kind == "ellipse":
            path.addEllipse(self.rect)
        else:
            path.addRect(self.rect)
        return path

    def shape(self) -> QPainterPath:
        path = self.shape_path()
        if self.isSelected() and self.interaction_mode == "edit":
            radius = self.handle_hit_size / self.view_scale()
            for point in self.handles().values():
                path.addEllipse(point, radius, radius)
            path.addEllipse(self.rotate_handle(), radius * 1.2, radius * 1.2)
        return path

    def handles(self) -> dict[tuple[int, int], QPointF]:
        r = self.rect
        return {
            (-1, -1): r.topLeft(), (0, -1): QPointF(r.center().x(), r.top()), (1, -1): r.topRight(),
            (1, 0): QPointF(r.right(), r.center().y()), (1, 1): r.bottomRight(),
            (0, 1): QPointF(r.center().x(), r.bottom()), (-1, 1): r.bottomLeft(),
            (-1, 0): QPointF(r.left(), r.center().y()),
        }

    def rotate_handle(self) -> QPointF:
        return QPointF(self.rect.center().x(), self.rect.top() - self.rotate_offset / self.view_scale())

    def paint(self, painter: QPainter, _option, _widget=None) -> None:
        scale = self.view_scale()
        outline = QPen(self.color, 2.2)
        outline.setCosmetic(True)
        painter.setPen(outline)
        fill = QColor(self.color)
        fill.setAlpha(42)
        painter.setBrush(fill)
        painter.drawPath(self.shape_path())
        painter.setPen(QPen(self.color, 1.5))
        painter.drawText(self.rect.topLeft() + QPointF(6, 18), f"Object {self.label}")
        if self.isSelected() and self.interaction_mode == "edit":
            selection_pen = QPen(QColor("#dfe9f7"), 1)
            selection_pen.setCosmetic(True)
            painter.setPen(selection_pen)
            painter.drawLine(QPointF(self.rect.center().x(), self.rect.top()), self.rotate_handle())
            for point in self.handles().values():
                painter.setBrush(QColor("#ffffff"))
                handle_pen = QPen(QColor("#243044"), 1.5)
                handle_pen.setCosmetic(True)
                painter.setPen(handle_pen)
                painter.drawEllipse(point, 5 / scale, 5 / scale)
            if isinstance(self.hovered_handle, tuple):
                hover_point = self.handles()[self.hovered_handle]
                hover_color = QColor("#72e5b2")
                hover_color.setAlpha(80)
                painter.setPen(Qt.NoPen)
                painter.setBrush(hover_color)
                painter.drawEllipse(hover_point, 10 / scale, 10 / scale)
            painter.setBrush(QColor("#55d98a"))
            painter.drawEllipse(self.rotate_handle(), 6 / scale, 6 / scale)
            if self.hovered_handle == "rotate":
                hover_color = QColor("#72e5b2")
                hover_color.setAlpha(80)
                painter.setBrush(hover_color)
                painter.drawEllipse(self.rotate_handle(), 11 / scale, 11 / scale)

    def hit_handle(self, point: QPointF) -> tuple[int, int] | None:
        hit_radius = self.handle_hit_size / self.view_scale()
        for key, location in self.handles().items():
            if math.hypot(point.x() - location.x(), point.y() - location.y()) <= hit_radius:
                return key
        return None

    def hoverMoveEvent(self, event) -> None:  # noqa: N802
        self.hovered_handle = None
        if self.interaction_mode == "edit":
            local = event.pos()
            if math.hypot(
                local.x() - self.rotate_handle().x(), local.y() - self.rotate_handle().y()
            ) <= self.rotation_hit_size / self.view_scale():
                self.hovered_handle = "rotate"
                self.setCursor(Qt.OpenHandCursor)
            else:
                handle = self.hit_handle(local)
                if handle:
                    self.hovered_handle = handle
                    sx, sy = handle
                    if sx == 0:
                        self.setCursor(Qt.SizeVerCursor)
                    elif sy == 0:
                        self.setCursor(Qt.SizeHorCursor)
                    elif sx == sy:
                        self.setCursor(Qt.SizeFDiagCursor)
                    else:
                        self.setCursor(Qt.SizeBDiagCursor)
                else:
                    self.setCursor(Qt.SizeAllCursor)
        self.update()
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # noqa: N802
        self.hovered_handle = None
        self.unsetCursor()
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.setSelected(True)
        local = event.pos()
        if self.interaction_mode == "edit" and math.hypot(
            local.x() - self.rotate_handle().x(), local.y() - self.rotate_handle().y()
        ) <= self.rotation_hit_size / self.view_scale():
            self.mode = "rotate"
            center = self.mapToScene(self.rect.center())
            scene = event.scenePos()
            self.start_scene_angle = math.degrees(math.atan2(scene.y() - center.y(), scene.x() - center.x()))
            self.start_rotation = self.rotation()
            event.accept()
            return
        handle = self.hit_handle(local) if self.interaction_mode == "edit" else None
        if handle:
            self.mode = "resize"
            self.active_handle = handle
            self.start_rect = QRectF(self.rect)
            event.accept()
            return
        self.mode = "move"
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self.mode == "rotate":
            center = self.mapToScene(self.rect.center())
            scene = event.scenePos()
            angle = math.degrees(math.atan2(scene.y() - center.y(), scene.x() - center.x()))
            self.setRotation(self.start_rotation + angle - self.start_scene_angle)
            self.update()
            return
        if self.mode == "resize" and self.active_handle:
            point = event.pos()
            sx, sy = self.active_handle
            r = QRectF(self.start_rect)
            minimum = 12.0
            if sx < 0:
                r.setLeft(min(point.x(), r.right() - minimum))
            elif sx > 0:
                r.setRight(max(point.x(), r.left() + minimum))
            if sy < 0:
                r.setTop(min(point.y(), r.bottom() - minimum))
            elif sy > 0:
                r.setBottom(max(point.y(), r.top() + minimum))
            old_center_scene = self.mapToScene(self.rect.center())
            self.prepareGeometryChange()
            self.rect = r
            new_center = self.rect.center()
            self.setPos(self.pos() + (old_center_scene - self.mapToScene(new_center)))
            self.update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self.mode = None
        self.active_handle = None
        super().mouseReleaseEvent(event)

    def scene_polygon(self, vertices: int = 64) -> list[list[float]]:
        if self.shape_kind == "ellipse":
            points = [
                QPointF(
                    self.rect.center().x() + self.rect.width() * 0.5 * math.cos(2 * math.pi * i / vertices),
                    self.rect.center().y() + self.rect.height() * 0.5 * math.sin(2 * math.pi * i / vertices),
                )
                for i in range(vertices)
            ]
        else:
            points = [self.rect.topLeft(), self.rect.topRight(), self.rect.bottomRight(), self.rect.bottomLeft()]
        return [[self.mapToScene(p).x(), self.mapToScene(p).y()] for p in points]


class PolygonObjectItem(QGraphicsPolygonItem):
    def __init__(self, points: list[QPointF], color: QColor, label: str) -> None:
        super().__init__(QPolygonF(points))
        self.shape_kind = "polygon"
        self.label = label
        fill = QColor(color)
        fill.setAlpha(42)
        self.setPen(cosmetic_pen(color, 2.2))
        self.setBrush(fill)
        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsFocusable)
        self.setAcceptHoverEvents(True)
        self.setZValue(30)
        self.mode: str | None = None
        self.active_handle: tuple[int, int] | str | None = None
        self.start_polygon = QPolygonF()
        self.start_bounds = QRectF()
        self.start_scene_angle = 0.0
        self.start_rotation = 0.0

    def view_scale(self) -> float:
        if self.scene() and self.scene().views():
            return max(abs(self.scene().views()[0].transform().m11()), 0.05)
        return 1.0

    def handles(self) -> dict[tuple[int, int], QPointF]:
        r = self.polygon().boundingRect()
        return {
            (-1, -1): r.topLeft(), (0, -1): QPointF(r.center().x(), r.top()), (1, -1): r.topRight(),
            (1, 0): QPointF(r.right(), r.center().y()), (1, 1): r.bottomRight(),
            (0, 1): QPointF(r.center().x(), r.bottom()), (-1, 1): r.bottomLeft(),
            (-1, 0): QPointF(r.left(), r.center().y()),
        }

    def rotate_handle(self) -> QPointF:
        bounds = self.polygon().boundingRect()
        return QPointF(bounds.center().x(), bounds.top() - 28.0 / self.view_scale())

    def hit_handle(self, point: QPointF) -> tuple[int, int] | str | None:
        radius = 18.0 / self.view_scale()
        if math.hypot(point.x() - self.rotate_handle().x(), point.y() - self.rotate_handle().y()) <= radius:
            return "rotate"
        for key, location in self.handles().items():
            if math.hypot(point.x() - location.x(), point.y() - location.y()) <= radius:
                return key
        return None

    def boundingRect(self) -> QRectF:  # noqa: N802
        return super().boundingRect().adjusted(-80, -80, 80, 80)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addPolygon(self.polygon())
        if self.isSelected():
            radius = 18.0 / self.view_scale()
            for point in self.handles().values():
                path.addEllipse(point, radius, radius)
            path.addEllipse(self.rotate_handle(), radius, radius)
        return path

    def paint(self, painter: QPainter, option, widget=None) -> None:
        super().paint(painter, option, widget)
        if not self.isSelected():
            return
        scale = self.view_scale()
        painter.setPen(cosmetic_pen(QColor("#dfe9f7"), 1))
        bounds = self.polygon().boundingRect()
        painter.drawLine(QPointF(bounds.center().x(), bounds.top()), self.rotate_handle())
        for point in self.handles().values():
            painter.setBrush(QColor("#ffffff"))
            painter.setPen(cosmetic_pen(QColor("#243044"), 1.5))
            painter.drawEllipse(point, 5 / scale, 5 / scale)
        painter.setBrush(QColor("#55d98a"))
        painter.drawEllipse(self.rotate_handle(), 6 / scale, 6 / scale)

    def hoverMoveEvent(self, event) -> None:  # noqa: N802
        handle = self.hit_handle(event.pos())
        if handle == "rotate":
            self.setCursor(Qt.OpenHandCursor)
        elif isinstance(handle, tuple):
            sx, sy = handle
            if sx == 0:
                self.setCursor(Qt.SizeVerCursor)
            elif sy == 0:
                self.setCursor(Qt.SizeHorCursor)
            elif sx == sy:
                self.setCursor(Qt.SizeFDiagCursor)
            else:
                self.setCursor(Qt.SizeBDiagCursor)
        else:
            self.setCursor(Qt.SizeAllCursor)
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # noqa: N802
        self.unsetCursor()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.setSelected(True)
        handle = self.hit_handle(event.pos())
        if handle == "rotate":
            self.mode = "rotate"
            center = self.mapToScene(self.polygon().boundingRect().center())
            scene = event.scenePos()
            self.start_scene_angle = math.degrees(math.atan2(scene.y() - center.y(), scene.x() - center.x()))
            self.start_rotation = self.rotation()
            self.setTransformOriginPoint(self.polygon().boundingRect().center())
            event.accept()
            return
        if isinstance(handle, tuple):
            self.mode = "resize"
            self.active_handle = handle
            self.start_polygon = QPolygonF(self.polygon())
            self.start_bounds = self.start_polygon.boundingRect()
            event.accept()
            return
        self.mode = "move"
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self.mode == "rotate":
            center = self.mapToScene(self.polygon().boundingRect().center())
            scene = event.scenePos()
            angle = math.degrees(math.atan2(scene.y() - center.y(), scene.x() - center.x()))
            self.setRotation(self.start_rotation + angle - self.start_scene_angle)
            self.update()
            return
        if self.mode == "resize" and isinstance(self.active_handle, tuple):
            sx, sy = self.active_handle
            bounds = QRectF(self.start_bounds)
            point = event.pos()
            minimum = 8.0
            if sx < 0:
                bounds.setLeft(min(point.x(), bounds.right() - minimum))
            elif sx > 0:
                bounds.setRight(max(point.x(), bounds.left() + minimum))
            if sy < 0:
                bounds.setTop(min(point.y(), bounds.bottom() - minimum))
            elif sy > 0:
                bounds.setBottom(max(point.y(), bounds.top() + minimum))
            source = self.start_bounds
            new_points = []
            for original in self.start_polygon:
                nx = (original.x() - source.left()) / max(source.width(), 1e-6)
                ny = (original.y() - source.top()) / max(source.height(), 1e-6)
                new_points.append(QPointF(bounds.left() + nx * bounds.width(), bounds.top() + ny * bounds.height()))
            self.setPolygon(QPolygonF(new_points))
            self.setTransformOriginPoint(self.polygon().boundingRect().center())
            self.update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self.mode = None
        self.active_handle = None
        super().mouseReleaseEvent(event)

    def scene_polygon(self, _vertices: int = 64) -> list[list[float]]:
        return [[self.mapToScene(point).x(), self.mapToScene(point).y()] for point in self.polygon()]


class GridObjectItem(QGraphicsPathItem):
    def __init__(
        self,
        cells: set[tuple[int, int]],
        cell_px: float,
        image_size: tuple[int, int],
        color: QColor,
        label: str,
    ) -> None:
        super().__init__()
        self.shape_kind = "grid"
        self.label = label
        self.cells = cells
        self.cell_px = cell_px
        self.image_size = image_size
        fill = QColor(color)
        fill.setAlpha(90)
        self.setPen(cosmetic_pen(color, 1.2))
        self.setBrush(fill)
        self.setFlags(QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsFocusable)
        self.setZValue(30)
        self.rebuild()

    def rebuild(self) -> None:
        path = QPainterPath()
        for column, row in self.cells:
            path.addRect(QRectF(column * self.cell_px, row * self.cell_px, self.cell_px, self.cell_px))
        self.setPath(path)

    def scene_polygon(self, _vertices: int = 64) -> list[list[float]]:
        width, height = self.image_size
        mask = np.zeros((height, width), dtype=np.uint8)
        for column, row in self.cells:
            x0 = max(0, int(math.floor(column * self.cell_px)))
            y0 = max(0, int(math.floor(row * self.cell_px)))
            x1 = min(width - 1, int(math.ceil((column + 1) * self.cell_px)))
            y1 = min(height - 1, int(math.ceil((row + 1) * self.cell_px)))
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
        return contour.tolist()


class NortStudio(QMainWindow):
    steps = ["기준 프레임", "Arena ROI", "cm 기준", "Object A/B", "검토·저장"]

    def __init__(
        self, video: Path | None, seconds: float, output: Path | None,
        roi_only: bool = False, profile_output: Path | None = None, initial_profile: Path | None = None,
        single_arena: bool = False, arena_id: int = 1, bundle_manifest: Path | None = None,
    ) -> None:
        super().__init__()
        self.steps = [tr(title) for title in type(self).steps]
        self.roi_only = roi_only
        self.profile_output = profile_output
        if roi_only:
            self.steps = [tr('기준 프레임'), "Arena ROI"]
        self.bundle_manifest_path = bundle_manifest.resolve() if bundle_manifest else None
        self.bundle_payload = self.load_bundle_manifest(self.bundle_manifest_path)
        self.bundle_records = list(self.bundle_payload.get("arenas", []))
        self.bundle_mode = bool(self.bundle_records and not roi_only)
        if self.bundle_mode:
            bundle_video = Path(str(self.bundle_payload.get("source_video", "")))
            if bundle_video.is_file():
                video = bundle_video
            seconds = float(self.bundle_payload.get("setup_frame_seconds", seconds))
        self.single_arena = bool(single_arena and not roi_only and not self.bundle_mode)
        self.arena_id = max(1, int(arena_id))
        self.setWindowTitle(tr('HAGISILTA · NORT · Arena ROI 편집') if roi_only else tr('HAGISILTA · NORT · Arena·Object 편집'))
        apply_hagisilta_window_identity(self)
        self.resize(1440, 900)
        self.setMinimumSize(1120, 720)
        self.video = video
        self.seconds = seconds
        self.output = output
        self.frame: np.ndarray | None = None
        self.video_meta: dict = {}
        self.distance_cm = 2.0
        self.arena_count = len(self.bundle_records) if self.bundle_mode else (1 if self.single_arena else 4)
        self.camera_locked = self.bundle_mode
        self.camera_profile_path: Path | None = None
        self.camera_profile_resolution: tuple[int, int] | None = None
        self.stage = 0
        self._pending_preview_ms = int(round(self.seconds * 1000))
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(40)
        self._preview_timer.timeout.connect(self.apply_queued_preview_time)
        self.current_arena = 0
        if self.bundle_mode:
            start_arena_id = int(self.bundle_payload.get("start_arena_id", 1) or 1)
            self.current_arena = next(
                (
                    index for index, record in enumerate(self.bundle_records)
                    if int(record.get("arena_id", index + 1)) == start_arena_id
                ),
                0,
            )
        self.current_object = "A"
        self.point_buffer: list[QPointF] = []
        self.object_mode = "edit"
        self.grid_brush_cells = 3
        self.brush_cursor: QGraphicsItem | None = None
        self.object_drag_preview: QGraphicsItem | None = None
        self.object_point_buffers: dict[tuple[int, str], list[QPointF]] = {}
        self.grid_cells: dict[tuple[int, str], set[tuple[int, int]]] = {}
        if self.bundle_mode:
            self.arenas = [
                {
                    "corners": record.get("corners_source_px", []),
                    "calibration_points": [], "reference_cm": 10.0, "objects": {},
                }
                for record in self.bundle_records
            ]
        else:
            self.arenas = [
                {"corners": [], "calibration_points": [], "reference_cm": 10.0, "objects": {}}
                for _ in range(self.arena_count)
            ]
        self.scene = QGraphicsScene(self)
        self.view = CanvasView(self.scene)
        self.pixmap_item: QGraphicsPixmapItem | None = None
        self.guide_items: list[QGraphicsItem] = []
        self.object_items: dict[tuple[int, str], EditableShapeItem] = {}
        self.step_labels: list[QPushButton] = []
        self._build_ui()
        self._apply_style()
        self._install_shortcuts()
        self.view.canvas_clicked.connect(self.on_canvas_click)
        self.view.canvas_dragged.connect(self.on_canvas_drag)
        self.view.canvas_drag_updated.connect(self.on_canvas_drag_updated)
        self.view.canvas_painted.connect(self.on_canvas_paint)
        self.view.canvas_hovered.connect(self.update_brush_cursor)
        self.view.brush_size_delta.connect(self.adjust_grid_brush_size)
        if self.video:
            self.load_video(self.video)
            if self.single_arena and self.frame is not None:
                height, width = self.frame.shape[:2]
                self.arenas[0]["corners"] = [
                    [0.0, 0.0], [float(width - 1), 0.0],
                    [float(width - 1), float(height - 1)], [0.0, float(height - 1)],
                ]
                self.stage = 2
                self.status_badge.setText(tr('●  Arena {p0:02d} · cm 보정부터 시작', p0=self.arena_id))
            elif self.bundle_mode and self.frame is not None:
                self.stage = 2
                self.status_badge.setText(tr('●  {p0}개 Arena · ROI 잠금 · cm 보정부터 시작', p0=self.arena_count))
                self.load_bundle_setups()
                self.point_buffer = [
                    QPointF(*point) for point in self.arenas[self.current_arena]["calibration_points"]
                ]
        if initial_profile and initial_profile.is_file():
            self.apply_camera_profile(initial_profile)
        if self.output and self.output.is_file() and not self.roi_only:
            self.load_saved_setup(self.output)
        self.refresh()

    @staticmethod
    def load_bundle_manifest(path: Path | None) -> dict:
        if not path or not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            raise RuntimeError(tr('NORT Arena 묶음 파일을 읽을 수 없습니다: {p0}', p0=error)) from error
        if not payload.get("source_video") or not payload.get("arenas"):
            raise RuntimeError(tr('NORT Arena 묶음에 원본 영상 또는 Arena 정보가 없습니다.'))
        return payload

    def load_saved_setup(self, path: Path) -> None:
        """Restore an arena setup so object positions can be adjusted later."""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if self.video and not self.setup_matches_video(payload, self.video):
                raise ValueError(
                    tr('이 설정은 다른 영상에서 저장되어 현재 영상에 적용하지 않았습니다.')
                )
            saved_arenas = payload.get("arenas") or []
            if not saved_arenas:
                raise ValueError(tr('저장 파일에 arena 설정이 없습니다.'))
            if self.single_arena:
                matched = next(
                    (arena for arena in saved_arenas if int(arena.get("id", 1)) == self.arena_id),
                    saved_arenas[0],
                )
                saved_arenas = [matched]
            self.clear_all_objects()
            restored = []
            self.distance_cm = float(payload.get("exploration_distance_cm", 2.0))
            for arena_index, data in enumerate(saved_arenas):
                calibration = data.get("calibration") or {}
                arena = {
                    "corners": data.get("corners_source_px", []),
                    "calibration_points": calibration.get("points_source_px", []),
                    "reference_cm": float(calibration.get("actual_length_cm", 10.0)),
                    "objects": {},
                }
                restored.append(arena)
                for label in ("A", "B"):
                    object_data = (data.get("objects") or {}).get(label) or {}
                    points = [QPointF(float(x), float(y)) for x, y in object_data.get("polygon_source_px", [])]
                    if len(points) < 3:
                        continue
                    shape_kind = str(object_data.get("shape", "polygon"))
                    if shape_kind in {"ellipse", "rectangle"}:
                        cloud = np.asarray([[point.x(), point.y()] for point in points], dtype=np.float32)
                        (cx, cy), (width, height), angle = cv2.minAreaRect(cloud)
                        rect = QRectF(
                            float(cx - width / 2), float(cy - height / 2),
                            max(float(width), 12.0), max(float(height), 12.0),
                        )
                        item = EditableShapeItem(shape_kind, rect, COLORS[label], label)
                        item.setRotation(float(angle))
                    else:
                        item = PolygonObjectItem(points, COLORS[label], label)
                    self.scene.addItem(item)
                    self.object_items[(arena_index, label)] = item
                    arena["objects"][label] = {"item": item}
            self.arenas = restored
            self.arena_count = len(restored)
            self.current_arena = 0
            self.current_object = "A"
            self.stage = 3
            self.status_badge.setText(tr('●  저장된 설정 불러옴 · 오브젝트 조정 가능'))
        except Exception as error:
            QMessageBox.warning(self, tr('설정 불러오기'), tr('저장된 설정을 불러오지 못했습니다.\n{p0}', p0=error))

    @staticmethod
    def shifted_points(points: list, dx: float, dy: float) -> list[list[float]]:
        return [[float(point[0]) + dx, float(point[1]) + dy] for point in points]

    @staticmethod
    def setup_matches_video(payload: dict, video: Path) -> bool:
        """Prevent a stale NORT setup from being loaded for another ROI video."""
        source_text = str(payload.get("source_video", "")).strip()
        if not source_text:
            return True
        source = Path(source_text)
        try:
            return str(source.resolve()).casefold() == str(video.resolve()).casefold()
        except OSError:
            return source.name.casefold() == video.name.casefold()

    def add_restored_object(
        self, arena_index: int, label: str, object_data: dict, dx: float = 0.0, dy: float = 0.0,
    ) -> None:
        points = [
            QPointF(float(x) + dx, float(y) + dy)
            for x, y in object_data.get("polygon_source_px", [])
        ]
        if len(points) < 3:
            return
        shape_kind = str(object_data.get("shape", "polygon"))
        if shape_kind in {"ellipse", "rectangle"}:
            cloud = np.asarray([[point.x(), point.y()] for point in points], dtype=np.float32)
            (cx, cy), (width, height), angle = cv2.minAreaRect(cloud)
            rect = QRectF(
                float(cx - width / 2), float(cy - height / 2),
                max(float(width), 12.0), max(float(height), 12.0),
            )
            item = EditableShapeItem(shape_kind, rect, COLORS[label], label)
            item.setRotation(float(angle))
        else:
            item = PolygonObjectItem(points, COLORS[label], label)
        self.scene.addItem(item)
        self.object_items[(arena_index, label)] = item
        self.arenas[arena_index]["objects"][label] = {"item": item}

    def load_bundle_setups(self) -> None:
        """Restore per-crop NORT settings into the locked full-source coordinate view."""
        if not self.bundle_mode:
            return
        self.clear_all_objects()
        loaded = 0
        for arena_index, record in enumerate(self.bundle_records):
            output = Path(str(record.get("output", "")))
            if not output.is_file():
                continue
            try:
                payload = json.loads(output.read_text(encoding="utf-8"))
                roi_video = Path(str(record.get("roi_video", "")))
                if roi_video.is_file() and not self.setup_matches_video(payload, roi_video):
                    print(
                        f"NORT_BUNDLE_SETUP_SOURCE_MISMATCH={output}:{roi_video}",
                        flush=True,
                    )
                    continue
                data = (payload.get("arenas") or [{}])[0]
                calibration = data.get("calibration") or {}
                offset = record.get("source_offset_px") or [0, 0]
                dx, dy = float(offset[0]), float(offset[1])
                self.distance_cm = float(payload.get("exploration_distance_cm", self.distance_cm))
                self.arenas[arena_index]["calibration_points"] = self.shifted_points(
                    calibration.get("points_source_px", []), dx, dy
                )
                self.arenas[arena_index]["reference_cm"] = float(
                    calibration.get("actual_length_cm", 10.0)
                )
                for label in ("A", "B"):
                    object_data = (data.get("objects") or {}).get(label) or {}
                    self.add_restored_object(arena_index, label, object_data, dx, dy)
                loaded += 1
            except Exception as error:
                print(f"NORT_BUNDLE_SETUP_WARNING={output}:{error}", flush=True)
        if loaded:
            self.status_badge.setText(
                tr('●  원본 화면에 {p0}/{p1}개 Arena 설정 불러옴', p0=loaded, p1=self.arena_count)
            )

    def _install_shortcuts(self) -> None:
        self.shortcuts: list[QShortcut] = []

        def bind(keys: str, callback) -> QShortcut:
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.setContext(Qt.ApplicationShortcut)
            shortcut.activated.connect(callback)
            self.shortcuts.append(shortcut)
            return shortcut

        bind("Ctrl+O", self.choose_video)
        bind("Ctrl+S", self.shortcut_save)
        bind("Ctrl+0", self.fit_canvas)
        bind("F", self.fit_canvas)
        bind("Ctrl++", lambda: self.view.scale(1.2, 1.2))
        bind("Ctrl+=", lambda: self.view.scale(1.2, 1.2))
        bind("Ctrl+-", lambda: self.view.scale(1 / 1.2, 1 / 1.2))
        bind("Ctrl+Z", self.undo_action)
        bind("Alt+Left", self.go_back)
        bind("Alt+Right", self.go_next)
        bind("Q", self.go_back)
        bind("E", self.enter_action)
        bind("W", lambda: self.pan_canvas(0, -70))
        bind("A", lambda: self.pan_canvas(-70, 0))
        bind("S", lambda: self.pan_canvas(0, 70))
        bind("D", lambda: self.pan_canvas(70, 0))
        bind("R", self.reset_current)
        bind("Z", self.undo_action)
        bind("X", lambda: self.remove_current_object() if self.stage == 3 else None)
        bind("Tab", self.toggle_object_focus)
        bind("Up", lambda: self.nudge_selected(0, -1))
        bind("Down", lambda: self.nudge_selected(0, 1))
        bind("Left", lambda: self.nudge_selected(-1, 0))
        bind("Right", lambda: self.nudge_selected(1, 0))
        bind("Shift+Up", lambda: self.nudge_selected(0, -10))
        bind("Shift+Down", lambda: self.nudge_selected(0, 10))
        bind("Shift+Left", lambda: self.nudge_selected(-10, 0))
        bind("Shift+Right", lambda: self.nudge_selected(10, 0))
        bind("Return", self.enter_action)
        bind("Enter", self.enter_action)
        bind("Escape", lambda: self.set_object_mode("edit") if self.stage == 3 else None)
        bind("?", self.show_gamer_shortcuts)
        bind("H", self.show_gamer_shortcuts)
        bind("F1", self.show_gamer_shortcuts)
        bind("J", lambda: self.step_preview_time(-1.0))
        bind("L", lambda: self.step_preview_time(1.0))
        bind(",", lambda: self.step_preview_frame(-1))
        bind(".", lambda: self.step_preview_frame(1))
        first_visible_stage = 0 if self.roi_only else 1
        for target_stage in range(first_visible_stage, len(self.steps)):
            shortcut_number = target_stage + 1 if self.roi_only else target_stage
            bind(f"Alt+{shortcut_number}", lambda stage=target_stage: self.jump_to_stage(stage))
        self.tool_shortcuts: list[QShortcut] = []
        for key, mode in {
            "1": "edit", "2": "ellipse", "3": "rectangle", "4": "polygon_points",
            "5": "circle_points", "6": "rectangle_points", "7": "grid",
        }.items():
            shortcut = bind(key, lambda value=mode: self.set_object_mode(value))
            shortcut.setEnabled(False)
            self.tool_shortcuts.append(shortcut)

    def pan_canvas(self, dx: int, dy: int) -> None:
        self.view.horizontalScrollBar().setValue(self.view.horizontalScrollBar().value() + dx)
        self.view.verticalScrollBar().setValue(self.view.verticalScrollBar().value() + dy)

    def reset_current(self) -> None:
        if self.stage in (1, 2):
            self.clear_current_points()
        elif self.stage == 3:
            self.remove_current_object()
            self.status_badge.setText(tr('●  현재 오브젝트 초기화'))

    def toggle_object_focus(self) -> None:
        if self.stage == 3:
            self.current_object = "B" if self.current_object == "A" else "A"
            self.object_mode = "edit"
            self.refresh()

    def nudge_selected(self, dx: int, dy: int) -> None:
        if self.stage != 3:
            return
        for item in self.scene.selectedItems():
            if hasattr(item, "scene_polygon") and not isinstance(item, GridObjectItem):
                item.moveBy(dx, dy)
                self.status_badge.setText(tr('●  도형 이동  {p0:+d}, {p1:+d}px', p0=dx, p1=dy))

    def fit_canvas(self) -> None:
        if self.frame is not None:
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def shortcut_save(self) -> None:
        if self.stage == 4:
            self.save_setup()
        else:
            self.status_badge.setText(tr('●  검토 단계에서 저장 가능'))

    def camera_profile_ready(self) -> bool:
        return bool(self.arenas) and all(
            len(arena["corners"]) >= 3 and len(arena["calibration_points"]) == 2
            for arena in self.arenas
        )

    def roi_profile_ready(self) -> bool:
        return bool(self.arenas) and all(len(arena["corners"]) == 4 for arena in self.arenas)

    def save_roi_profile(self) -> bool:
        if not self.roi_profile_ready() or self.frame is None or self.video is None:
            QMessageBox.information(self, tr('ROI 미완성'), tr('모든 Arena ROI의 꼭짓점 네 개를 지정하세요.'))
            return False
        payload = {
            "schema_version": 1,
            "profile_type": "nort_arena_roi",
            "created_by": "HAGISILTA",
            "created_from_video": str(self.video.resolve()),
            "source_resolution_px": [int(self.frame.shape[1]), int(self.frame.shape[0])],
            "setup_frame_seconds": float(self.seconds),
            "arena_count": self.arena_count,
            "arenas": [
                {
                    "id": index + 1,
                    "corners_source_px": arena["corners"],
                    "corner_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
                }
                for index, arena in enumerate(self.arenas)
            ],
        }
        path = self.profile_output
        if path is None:
            CAMERA_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
            default_name = CAMERA_PROFILE_DIR / f"{self.video.stem}_roi_profile.json"
            selected, _ = QFileDialog.getSaveFileName(
                self, tr('ROI 영역 저장'), str(default_name), "HAGISILTA ROI profile (*.json)"
            )
            if not selected:
                return False
            path = Path(selected)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.camera_profile_path = path
        self.status_badge.setText(tr('●  빠른 추론 ROI 저장됨'))
        print(f"ROI_PROFILE={path.resolve()}", flush=True)
        QMessageBox.information(self, tr('ROI 저장 완료'), tr('{p0}개 영역을 저장했습니다.\n\n{p1}', p0=self.arena_count, p1=path))
        return True

    def clear_all_objects(self) -> None:
        for item in list(self.object_items.values()):
            try:
                if item.scene() is self.scene:
                    self.scene.removeItem(item)
            except RuntimeError:
                pass
        self.object_items.clear()
        self.object_point_buffers.clear()
        self.grid_cells.clear()
        for arena in self.arenas:
            arena["objects"] = {}

    def camera_profile_payload(self) -> dict:
        if not self.camera_profile_ready() or self.frame is None or self.video is None:
            raise ValueError(tr('Arena ROI와 cm 기준 보정을 먼저 완료하세요.'))
        return {
            "schema_version": 1,
            "profile_type": "nort_camera_calibration",
            "created_by": "HAGISILTA",
            "created_from_video": str(self.video.resolve()),
            "source_resolution_px": [int(self.frame.shape[1]), int(self.frame.shape[0])],
            "arena_count": self.arena_count,
            "arenas": [
                {
                    "id": index + 1,
                    "corners_source_px": arena["corners"],
                    "corner_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
                    "calibration": {
                        "method": "local_reference_line",
                        "points_source_px": arena["calibration_points"],
                        "actual_length_cm": float(arena["reference_cm"]),
                        "plane": "arena_floor",
                    },
                }
                for index, arena in enumerate(self.arenas)
            ],
        }

    def save_camera_profile(self) -> None:
        try:
            payload = self.camera_profile_payload()
        except ValueError as error:
            QMessageBox.information(self, tr('카메라 템플릿'), str(error))
            return
        CAMERA_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        default_name = CAMERA_PROFILE_DIR / f"{self.video.stem}_camera_profile.json"
        selected, _ = QFileDialog.getSaveFileName(
            self, tr('카메라·Arena 템플릿 저장'), str(default_name), "HAGISILTA camera profile (*.json)"
        )
        if not selected:
            return
        path = Path(selected)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.camera_profile_path = path
        self.camera_profile_resolution = tuple(payload["source_resolution_px"])
        self.camera_locked = True
        self.status_badge.setText(tr('●  카메라 템플릿 저장됨'))
        QMessageBox.information(self, tr('템플릿 저장 완료'), tr('카메라 템플릿을 저장했습니다.\n\n{p0}', p0=path))

    def load_camera_profile(self) -> None:
        CAMERA_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        selected, _ = QFileDialog.getOpenFileName(
            self, tr('카메라·Arena 템플릿 불러오기'), str(CAMERA_PROFILE_DIR), "HAGISILTA camera profile (*.json)"
        )
        if not selected:
            return
        self.apply_camera_profile(Path(selected))

    def apply_camera_profile(self, path: Path) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            profile_type = payload.get("profile_type")
            allowed = {"nort_camera_calibration", "nort_arena_roi"} if self.roi_only else {"nort_camera_calibration"}
            if profile_type not in allowed:
                raise ValueError(tr('HAGISILTA ROI/카메라 템플릿 파일이 아닙니다.'))
            resolution = tuple(int(value) for value in payload["source_resolution_px"])
            if self.frame is not None and resolution != (self.frame.shape[1], self.frame.shape[0]):
                raise ValueError(
                    tr('템플릿 해상도 {p0}×{p1}와 현재 영상 {p2}×{p3}가 다릅니다.', p0=resolution[0], p1=resolution[1], p2=self.frame.shape[1], p3=self.frame.shape[0])
                )
            arenas = []
            for data in payload["arenas"]:
                calibration = data.get("calibration", {})
                arenas.append({
                    "corners": data["corners_source_px"],
                    "calibration_points": calibration.get("points_source_px", []),
                    "reference_cm": float(calibration.get("actual_length_cm", 10.0)),
                    "objects": {},
                })
            if not arenas:
                raise ValueError(tr('템플릿에 Arena가 없습니다.'))
        except Exception as error:
            QMessageBox.critical(self, tr('템플릿 오류'), str(error))
            return
        self.clear_all_objects()
        self.arenas = arenas
        self.arena_count = len(arenas)
        self.camera_profile_path = path
        self.camera_profile_resolution = resolution
        self.camera_locked = not self.roi_only
        self.current_arena = 0
        self.current_object = "A"
        self.point_buffer.clear()
        self.stage = 0
        self.status_badge.setText(tr('●  ROI 템플릿 적용됨') if self.roi_only else tr('●  카메라 템플릿 적용됨'))
        self.refresh()

    def unlock_camera_profile(self) -> None:
        self.camera_locked = False
        self.camera_profile_path = None
        self.status_badge.setText(tr('●  ROI/cm 보정 편집 가능'))
        self.refresh()

    def apply_current_camera_to_new_video(self) -> None:
        if not self.camera_profile_ready() or self.frame is None:
            return
        selected, _ = QFileDialog.getOpenFileName(
            self, tr('같은 카메라의 새 영상 선택'), "", "Videos (*.avi *.mp4 *.mov *.mkv *.wmv)"
        )
        if not selected:
            return
        candidate = Path(selected)
        try:
            candidate_frame, _ = frame_at(candidate, self.seconds)
        except Exception as error:
            QMessageBox.critical(self, tr('영상 오류'), str(error))
            return
        expected = (self.frame.shape[1], self.frame.shape[0])
        actual = (candidate_frame.shape[1], candidate_frame.shape[0])
        if actual != expected:
            QMessageBox.critical(
                self, tr('해상도 불일치'), tr('기존 {p0}×{p1} / 새 영상 {p2}×{p3}', p0=expected[0], p1=expected[1], p2=actual[0], p3=actual[1])
            )
            return
        self.clear_all_objects()
        self.load_video(candidate)
        self.camera_profile_resolution = expected
        self.camera_locked = True
        self.stage = 0
        self.current_arena = 0
        self.current_object = "A"
        self.object_mode = "edit"
        self.status_badge.setText(tr('●  새 영상 · 오브젝트만 설정'))
        self.refresh()

    def enter_action(self) -> None:
        if self.stage == 3 and self.object_mode in ("polygon_points", "circle_points", "rectangle_points"):
            self.finalize_point_object()
        else:
            self.go_next()

    def undo_action(self) -> None:
        if self.point_buffer:
            self.point_buffer.pop()
            self.refresh()
            return
        if self.stage == 3:
            key = (self.current_arena, self.current_object)
            points = self.object_point_buffers.get(key, [])
            if points:
                points.pop()
                self.refresh()
                return
        self.status_badge.setText(tr('●  취소할 점 없음'))

    def show_shortcuts(self) -> None:
        QMessageBox.information(
            self,
            tr('HAGISILTA ROI 편집기 단축키'),
            tr('도구\n1 이동·크기·회전   2 원형   3 사각형   4 점 외곽선\n5 점 원형     6 점 사각형     7 그리드\n\n작업\nEnter 확정/다음   Ctrl+Z 또는 Backspace 마지막 점   Delete 오브젝트 삭제\nAlt+숫자 작업 단계 이동   Alt+← 이전   Alt+→ 다음   Esc 선택 도구\n\n화면/파일\nF 또는 Ctrl+0 화면 맞춤   Ctrl++ / Ctrl+- 확대·축소\nJ / L ±1초   , / . 이전·다음 프레임\nCtrl+O 영상 열기   Ctrl+S 저장   H 또는 ? 도움말'),
        )

    @staticmethod
    def format_time(seconds: float) -> str:
        minutes = int(seconds // 60)
        remainder = seconds - minutes * 60
        return f"{minutes:02d}:{remainder:06.3f}"

    def show_gamer_shortcuts(self) -> None:
        QMessageBox.information(
            self,
            tr('HAGISILTA ROI 편집기 단축키'),
            tr('게이머 조작\nWASD  화면 이동     Q  이전     E  확정/다음     R  현재 작업 리셋\nZ  실행취소     X  삭제     Tab  Object A/B 전환\n방향키  도형 1px 이동     Shift+방향키  10px 이동\n\n워크플로우\nAlt+숫자  해당 작업 단계로 이동\n\n도구 슬롯\n1  이동·크기·회전     2  원형     3  사각형     4  점 외곽선\n5  점 원형     6  점 사각형     7  그리드\n\n화면 및 파일\nF  화면 맞춤     휠  확대/축소     가운데 드래그  화면 이동\n그리드에서 Ctrl+휠  브러시 크기 조절\nJ/L  ±1초     ,/.  이전/다음 프레임\nCtrl+O  영상 열기     Ctrl+S  저장     F1/H/?  도움말'),
        )

    def set_time_controls(self, seconds: float) -> None:
        self.seconds = float(seconds)
        if hasattr(self, "time_slider"):
            self.time_slider.blockSignals(True)
            self.time_slider.setValue(int(round(self.seconds * 1000)))
            self.time_slider.blockSignals(False)
        if getattr(self, "time_spin", None) is not None:
            self.time_spin.blockSignals(True)
            self.time_spin.setValue(self.seconds)
            self.time_spin.blockSignals(False)
        if hasattr(self, "time_readout"):
            self.time_readout.setText(self.format_time(self.seconds))

    def preview_slider_moved(self, milliseconds: int) -> None:
        seconds = milliseconds / 1000.0
        if getattr(self, "time_spin", None) is not None:
            self.time_spin.blockSignals(True)
            self.time_spin.setValue(seconds)
            self.time_spin.blockSignals(False)
        self.time_readout.setText(self.format_time(seconds))
        self._pending_preview_ms = int(milliseconds)
        self._preview_timer.start()

    def apply_queued_preview_time(self) -> None:
        self.apply_preview_time(self._pending_preview_ms / 1000.0)

    def apply_slider_time(self) -> None:
        self._preview_timer.stop()
        self.apply_preview_time(self.time_slider.value() / 1000.0)

    def apply_spin_time(self) -> None:
        self.apply_preview_time(float(self.time_spin.value()))

    def apply_preview_time(self, seconds: float) -> None:
        if not self.video:
            return
        duration = float(self.video_meta.get("duration_seconds", 0.0))
        seconds = min(max(seconds, 0.0), max(duration - 1.0 / max(self.video_meta.get("fps", 1.0), 1.0), 0.0))
        try:
            frame, meta = frame_at(self.video, seconds)
        except Exception as error:
            self.status_badge.setText(tr('●  프레임 이동 실패: {p0}', p0=error))
            return
        self.seconds = float(meta.get("setup_frame_seconds", seconds))
        self.frame = frame
        self.video_meta.update(meta)
        if self.pixmap_item is None:
            self.pixmap_item = self.scene.addPixmap(cv_to_pixmap(frame))
            self.pixmap_item.setZValue(-100)
        else:
            self.pixmap_item.setPixmap(cv_to_pixmap(frame))
        self.scene.setSceneRect(QRectF(0, 0, frame.shape[1], frame.shape[0]))
        self.canvas_title.setText(tr('{p0}  ·  {p1:.1f}초', p0=self.video.name, p1=self.seconds))
        self.set_time_controls(self.seconds)
        self.view.viewport().update()

    def step_preview_time(self, delta_seconds: float) -> None:
        if self.video:
            self.apply_preview_time(self.seconds + delta_seconds)

    def step_preview_frame(self, direction: int) -> None:
        fps = float(self.video_meta.get("fps", 0.0))
        if self.video and fps > 0:
            self.apply_preview_time(self.seconds + direction / fps)

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header, header_layout = create_hagisilta_header(
            tr('Arena ROI 편집') if self.roi_only else tr('NORT Arena·Object 편집'),
            "HAGISILTA · SPATIAL WORKSPACE",
        )
        self.status_badge = HeaderContext(header, tr('편집 중'))
        help_button = QPushButton(tr('도움말'), objectName="toolButton")
        help_button.setFixedWidth(88)
        help_button.clicked.connect(self.show_gamer_shortcuts)
        header_layout.addWidget(help_button)
        outer.addWidget(header)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        outer.addLayout(body, 1)

        sidebar = QFrame(objectName="sidebar")
        sidebar.setFixedWidth(184)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(18, 24, 18, 24)
        side_layout.addWidget(QLabel("WORKFLOW", objectName="sectionTitle"))
        for index, title in enumerate(self.steps):
            display_index = index if not self.roi_only else index + 1
            display_title = title
            if not self.roi_only and index == 0:
                display_title = tr('영상 준비')
            button = QPushButton(f"{display_index:02d}   {display_title}", objectName="step")
            button.setMinimumHeight(44)
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            shortcut_number = index + 1 if self.roi_only else index
            button.setToolTip(tr('{p0} 단계로 이동 · Alt+{p1}', p0=title, p1=shortcut_number))
            button.clicked.connect(lambda _checked=False, target=index: self.jump_to_stage(target))
            self.step_labels.append(button)
            side_layout.addWidget(button)
            if not self.roi_only and index == 0:
                button.setVisible(False)
        side_layout.addStretch()
        body.addWidget(sidebar)

        center = QFrame(objectName="canvasFrame")
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(18, 18, 18, 18)
        self.canvas_title = QLabel(tr('영상 미리보기'), objectName="canvasTitle")
        center_layout.addWidget(self.canvas_title)
        center_layout.addWidget(self.view, 1)
        body.addWidget(center, 1)

        inspector_scroll = QScrollArea(objectName="panelScroll")
        inspector_scroll.setWidgetResizable(True)
        inspector_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inspector_scroll.setFixedWidth(380)
        self.inspector = QFrame(objectName="inspector")
        self.inspector.setMinimumWidth(360)
        self.inspector_layout = QVBoxLayout(self.inspector)
        self.inspector_layout.setContentsMargins(18, 18, 18, 18)
        inspector_scroll.setWidget(self.inspector)
        body.addWidget(inspector_scroll)

        footer = QFrame(objectName="detachedFooter")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(18, 9, 18, 9)
        self.context_label = QLabel(tr('새 프로젝트'), objectName="context")
        self.back_button = QPushButton(tr('←  이전'), objectName="secondaryButton")
        self.next_button = QPushButton(tr('다음  →'), objectName="primaryButton")
        self.back_button.clicked.connect(self.go_back)
        self.next_button.clicked.connect(self.go_next)
        footer_layout.addWidget(self.context_label)
        footer_layout.addStretch()
        footer_layout.addWidget(self.back_button)
        footer_layout.addWidget(self.next_button)
        outer.addWidget(footer)

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QWidget {
                background:#0d1420; color:#dfe8f5;
                font-family:'Segoe UI','Malgun Gothic'; font-size:13px;
            }
            QMainWindow, QDialog { background:#0d1420; }
            #header { background:#111b2a; border:none; border-bottom:1px solid #223149; }
            #logo { color:#69e0ad; font-size:18px; font-weight:800; letter-spacing:2px; }
            #subtitle { color:#718198; margin-left:8px; }
            #status {
                background:#183529; color:#70e2ae; border:1px solid #28513f;
                border-radius:12px; padding:5px 12px;
            }
            #modeBadge {
                color:#9dc4ff; background:#172b46; border:1px solid #31547c;
                border-radius:12px; padding:6px 12px;
            }
            #sidebar { background:#101925; border:none; border-right:1px solid #223149; }
            #sectionTitle {
                color:#9ba1aa; font-size:10px; font-weight:700;
                letter-spacing:1px; margin-bottom:7px;
            }
            #step {
                color:#c2c5ca; border-radius:3px; padding-left:12px;
                border:1px solid transparent; text-align:left; background:transparent;
            }
            #step:hover { background:#172437; color:#ffffff; }
            #step[active='true'] {
                background:#174c3d; color:#79e7b5; font-weight:650;
                border:1px solid #28745c;
            }
            #step[done='true'] { color:#8ce0bd; }
            #hint { color:#9ba1aa; line-height:1.5; }
            #canvasFrame { background:#0d1420; }
            #canvasTitle { color:#f0f1f2; font-size:14px; font-weight:650; }
            QGraphicsView { background:#121212; border:1px solid #565f6c; border-radius:3px; }
            #inspector { background:#101925; border:none; border-left:1px solid #223149; }
            #panelTitle { font-size:14px; font-weight:650; color:#f0f1f2; margin-top:4px; }
            #panelText { color:#a9adb4; line-height:1.45; }
            #card { background:#111b2a; border:1px solid #283a53; border-radius:5px; }
            QPushButton { min-height:36px; border-radius:3px; padding:0 13px; font-weight:600; }
            #primaryButton { background:#4c63cf; color:#ffffff; border:1px solid #6079d5; }
            #primaryButton:hover { background:#5872df; border-color:#8297e0; }
            #primaryButton:pressed { background:#3b51b5; }
            #secondaryButton { background:#18263a; color:#dfe8f5; border:1px solid #304765; }
            #secondaryButton:hover { background:#21334c; border-color:#45658d; }
            #toolButton { background:#172437; color:#d1dbea; border:1px solid #304765; }
            #toolButton:hover { background:#4a515d; color:#ffffff; }
            #toolButton:checked { background:#174c3d; color:#79e7b5; border:1px solid #28745c; }
            #shapeToolButton {
                background:#18263a; color:#dfe8f5; border:1px solid #304765;
            }
            #shapeToolButton:hover { background:#21334c; border-color:#45658d; }
            #shapeToolButton:checked {
                background:#4c63cf; color:#ffffff; border:1px solid #6f91df;
            }
            QPushButton:disabled { background:#343840; color:#777d86; border-color:#444a54; }
            QDoubleSpinBox, QSpinBox {
                background:#0f1825; color:#dfe8f5; border:1px solid #304765;
                border-radius:3px; padding:7px; min-height:23px;
                selection-background-color:#4560c4;
            }
            QDoubleSpinBox:focus, QSpinBox:focus { border:1px solid #7288da; }
            QSlider::groove:horizontal { height:6px; background:#424a54; border-radius:3px; }
            QSlider::sub-page:horizontal { background:#4560c4; border-radius:3px; }
            QSlider::handle:horizontal {
                width:16px; margin:-5px 0; background:#f0f1f2;
                border:2px solid #7188da; border-radius:8px;
            }
            QScrollArea { border:none; }
            QScrollBar:vertical { background:#23242b; width:12px; border:none; }
            QScrollBar::handle:vertical { background:#565f6c; min-height:28px; border-radius:5px; margin:2px; }
            QScrollBar::handle:vertical:hover { background:#737d89; }
            QScrollBar::add-line, QScrollBar::sub-line { width:0; height:0; }
            #footer { background:#101925; border:none; border-top:1px solid #223149; }
            #context { color:#a9adb4; }
            QToolTip { background:#3a404b; color:#f0f1f2; border:1px solid #6b7480; padding:7px; }
        """)
        self.setStyleSheet(self.styleSheet() + """
            QWidget { background:#23242b; color:#f0f1f2; }
            QMainWindow, QDialog { background:#23242b; }
            #sidebar, #inspector, #footer { background:#27282d; border-color:#41434a; }
            #canvasFrame { background:#202125; }
            #sectionTitle { color:#aeb2b9; }
            #card { background:#292a2f; border:1px solid #41434a; border-radius:2px; }
            #step:hover { background:#3a404b; }
            #step[active='true'] { background:#3b65c4; color:#ffffff; border-color:#718ee7; }
            #footer { border-top:1px solid #41434a; }
            QPushButton { border-radius:2px; background:#35363b; color:#eceef1; border:1px solid #55585f; }
            QPushButton:hover { background:#42444a; border-color:#70747d; }
            #primaryButton { background:#3f6dcc; color:#ffffff; border-color:#5c82d8; }
            #secondaryButton, #toolButton { background:#35363b; color:#eceef1; border-color:#55585f; }
            QDoubleSpinBox, QSpinBox, QComboBox {
                background:#202125; color:#f0f1f2; border:1px solid #50535a; border-radius:2px;
            }
        """ + hagisilta_control_stylesheet())

    def clear_layout(self, layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            child_layout = item.layout()
            if child_layout:
                self.clear_layout(child_layout)

    def panel_label(self, title: str, text: str) -> None:
        heading = QLabel(title, objectName="panelTitle")
        heading.setToolTip(text)
        self.inspector_layout.addWidget(heading)

    def load_video(self, path: Path) -> None:
        try:
            frame, meta = frame_at(path, self.seconds)
        except Exception as error:
            QMessageBox.critical(self, tr('영상 오류'), str(error))
            return
        self.video = path
        self.frame = frame
        self.video_meta = meta
        self.seconds = float(meta["setup_frame_seconds"])
        self.guide_items.clear()
        self.pixmap_item = None
        self.scene.clear()
        self.brush_cursor = None
        self.object_drag_preview = None
        self.pixmap_item = self.scene.addPixmap(cv_to_pixmap(frame))
        self.pixmap_item.setZValue(-100)
        self.scene.setSceneRect(QRectF(0, 0, frame.shape[1], frame.shape[0]))
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.status_badge.setText(tr('●  영상 로드됨'))
        self.canvas_title.setText(tr('{p0}  ·  {p1:.1f}초', p0=path.name, p1=self.seconds))

    def choose_video(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, tr('HAGISILTA 영상 선택'), "", "Videos (*.avi *.mp4 *.mov *.mkv *.wmv)")
        if selected:
            self.load_video(Path(selected))
            self.refresh()

    def refresh(self) -> None:
        for index, label in enumerate(self.step_labels):
            label.setProperty("active", index == self.stage)
            label.setProperty("done", index < self.stage)
            label.setChecked(index == self.stage)
            label.style().unpolish(label)
            label.style().polish(label)
        self.back_button.setEnabled(
            (self.stage > 0 or self.current_arena > 0)
            and not ((self.single_arena or self.bundle_mode) and self.stage == 2 and self.current_arena == 0)
        )
        if (self.single_arena or self.bundle_mode) and len(self.step_labels) > 1:
            self.step_labels[1].setEnabled(False)
            self.step_labels[1].setToolTip(
                tr('트래킹에서 잘라낸 Arena 전체 화면이 ROI 기준입니다. 좌표 정합성을 위해 여기서는 수정할 수 없습니다.')
            )
        if self.roi_only and self.stage == 1 and self.current_arena == self.arena_count - 1:
            self.next_button.setText(tr('저장하고 닫기'))
        elif self.stage == 4:
            self.next_button.setText(tr('저장하고 닫기'))
        elif self.stage == 0 and (self.bundle_mode or self.single_arena):
            self.next_button.setText(tr('cm 보정  →'))
        elif self.stage == 0 and self.camera_locked:
            self.next_button.setText(tr('오브젝트 편집  →'))
        else:
            self.next_button.setText(tr('다음  →'))
        self.view.creation_enabled = self.stage == 3 and self.object_mode in ("ellipse", "rectangle")
        self.view.paint_enabled = self.stage == 3 and self.object_mode == "grid"
        if not self.view.creation_enabled:
            self.clear_object_drag_preview()
        if not self.view.paint_enabled and self.brush_cursor is not None:
            try:
                if self.brush_cursor.scene() is self.scene:
                    self.scene.removeItem(self.brush_cursor)
            except RuntimeError:
                pass
            self.brush_cursor = None
        for key, item in self.object_items.items():
            editable = self.stage == 3 and self.object_mode == "edit" and key[0] == self.current_arena
            item.setOpacity(1.0 if key[0] == self.current_arena else 0.42)
            item.setFlag(QGraphicsItem.ItemIsSelectable, editable)
            if self.stage == 3 and self.object_mode == "edit":
                if isinstance(item, EditableShapeItem):
                    item.interaction_mode = "edit"
                    item.setFlag(QGraphicsItem.ItemIsMovable, editable)
                    item.update()
                elif isinstance(item, PolygonObjectItem):
                    item.setFlag(QGraphicsItem.ItemIsMovable, editable)
        for shortcut in self.tool_shortcuts:
            shortcut.setEnabled(self.stage == 3)
        self.sync_status_badge()
        self.rebuild_panel()
        self.redraw_guides()

    def sync_status_badge(self) -> None:
        if self.roi_only:
            status = tr('기준 프레임') if self.stage == 0 else "Arena ROI"
        else:
            status = {
                0: tr('기준 프레임'),
                1: "Arena ROI",
                2: tr('cm 기준'),
                3: f"Object {self.current_object}",
                4: tr('저장 준비'),
            }.get(self.stage, tr('편집 중'))
        self.status_badge.setText(tr('편집 중 · {p0}', p0=status))

    def add_timeline_controls(self, include_spin: bool = False) -> None:
        if not self.video or not self.video_meta:
            return
        timeline_header = QHBoxLayout()
        timeline_header.addWidget(QLabel(tr('기준 프레임'), objectName="toolGroupTitle"))
        timeline_header.addStretch()
        self.time_readout = QLabel(self.format_time(self.seconds), objectName="timeReadout")
        timeline_header.addWidget(self.time_readout)
        self.inspector_layout.addLayout(timeline_header)

        self.time_slider = QSlider(Qt.Horizontal)
        duration_ms = max(1, int(round(float(self.video_meta["duration_seconds"]) * 1000)))
        self.time_slider.setRange(0, duration_ms)
        self.time_slider.setValue(int(round(self.seconds * 1000)))
        self.time_slider.setSingleStep(max(1, int(round(1000 / max(self.video_meta["fps"], 1.0)))))
        self.time_slider.sliderMoved.connect(self.preview_slider_moved)
        self.time_slider.sliderReleased.connect(self.apply_slider_time)
        self.inspector_layout.addWidget(self.time_slider)

        frame_buttons = QHBoxLayout()
        frame_buttons.setSpacing(8)
        for text, callback in (
            (tr('-1초'), lambda: self.step_preview_time(-1.0)),
            ("‹ 1F", lambda: self.step_preview_frame(-1)),
            ("1F ›", lambda: self.step_preview_frame(1)),
            (tr('+1초'), lambda: self.step_preview_time(1.0)),
        ):
            button = QPushButton(text, objectName="secondaryButton")
            button.clicked.connect(lambda _checked=False, action=callback: action())
            frame_buttons.addWidget(button)
        self.inspector_layout.addLayout(frame_buttons)

        if include_spin:
            self.time_spin = QDoubleSpinBox()
            self.time_spin.setRange(0.0, float(self.video_meta["duration_seconds"]))
            self.time_spin.setDecimals(3)
            self.time_spin.setSingleStep(1.0 / max(float(self.video_meta["fps"]), 1.0))
            self.time_spin.setSuffix(tr(' 초'))
            self.time_spin.setValue(self.seconds)
            self.time_spin.editingFinished.connect(self.apply_spin_time)
            self.time_spin.setToolTip(tr('정확한 기준 프레임 시각을 직접 입력합니다.'))
            self.inspector_layout.addWidget(self.time_spin)

    def rebuild_panel(self) -> None:
        self.time_spin = None
        self.clear_layout(self.inspector_layout)
        self.add_timeline_controls(include_spin=self.stage == 0)
        if self.bundle_mode and self.stage >= 2:
            self.panel_label(
                tr('Arena 선택'),
                tr('원본 영상의 모든 Arena를 함께 표시합니다. 현재 Arena만 밝게 편집되며 트래킹 ROI는 잠겨 있습니다.'),
            )
            arena_combo = QComboBox()
            for index, record in enumerate(self.bundle_records):
                arena_id = int(record.get("arena_id", index + 1))
                arena_combo.addItem(f"Arena {arena_id:02d}", index)
            arena_combo.setCurrentIndex(self.current_arena)
            arena_combo.currentIndexChanged.connect(self.select_bundle_arena)
            self.inspector_layout.addWidget(arena_combo)
            self.add_progress_card(
                tr('현재 {p0}/{p1}', p0=self.current_arena + 1, p1=self.arena_count),
                tr('다른 Arena는 흐리게 표시되며 실수로 이동할 수 없습니다.'),
            )
        if self.stage == 0:
            self.panel_label(
                tr('영상·Arena 설정'),
                tr('ROI·기준선·오브젝트를 가장 정확히 구분할 수 있는 영상 위치를 선택합니다. 트래킹 결과에서 불러온 Arena ROI 좌표는 프레임을 바꿔도 유지됩니다.'),
            )
            open_button = QPushButton(tr('영상 선택'), objectName="secondaryButton")
            open_button.clicked.connect(self.choose_video)
            self.inspector_layout.addWidget(open_button)
            profile_button = QPushButton(tr('ROI 템플릿 불러오기') if self.roi_only else tr('카메라 템플릿 불러오기'), objectName="secondaryButton")
            profile_button.clicked.connect(self.load_camera_profile)
            self.inspector_layout.addWidget(profile_button)
            video_label = QLabel(str(self.video) if self.video else tr('선택된 영상 없음'), objectName="panelText")
            video_label.setWordWrap(True)
            self.inspector_layout.addWidget(video_label)
            self.inspector_layout.addWidget(QLabel(tr('영상 속 Arena 수')))
            arena_spin = QSpinBox()
            arena_spin.setRange(1, 16)
            arena_spin.setValue(self.arena_count)
            arena_spin.setSuffix(tr(' 개'))
            arena_spin.setEnabled(not self.camera_locked)
            arena_spin.valueChanged.connect(self.set_arena_count)
            self.inspector_layout.addWidget(arena_spin)
            if self.camera_locked:
                profile_name = self.camera_profile_path.name if self.camera_profile_path else tr('현재 설정')
                self.add_progress_card(tr('카메라 설정 잠김'), tr('{p0}\n다음 단계는 오브젝트 설정입니다.', p0=profile_name))
                unlock = QPushButton(tr('ROI / cm 보정 잠금 해제'), objectName="secondaryButton")
                unlock.clicked.connect(self.unlock_camera_profile)
                self.inspector_layout.addWidget(unlock)
        elif self.stage == 1:
            self.panel_label(f"Arena ROI  ·  {self.current_arena + 1}/{self.arena_count}", tr('화면에서 보이는 실험 영역을 좌상단 → 우상단 → 우하단 → 좌하단 순으로 클릭하세요.'))
            self.add_progress_card(tr('선택한 점  {p0}/4', p0=len(self.point_buffer)))
            clear = QPushButton(tr('현재 점 초기화'), objectName="secondaryButton")
            clear.clicked.connect(self.clear_current_points)
            self.inspector_layout.addWidget(clear)
        elif self.stage == 2:
            self.panel_label(tr('cm 기준 보정  ·  {p0}/{p1}', p0=self.current_arena + 1, p1=self.arena_count), tr('바닥과 같은 높이에 있는 실제 길이를 아는 기준선의 양 끝을 클릭하세요.'))
            self.add_progress_card(tr('기준점  {p0}/2', p0=len(self.point_buffer)), tr('오브젝트 가까운 위치일수록 좋습니다.'))
            self.inspector_layout.addWidget(QLabel(tr('선택한 기준의 실제 길이 (cm)')))
            spin = QDoubleSpinBox()
            spin.setRange(0.1, 100.0)
            spin.setDecimals(2)
            spin.setValue(float(self.arenas[self.current_arena]["reference_cm"]))
            spin.valueChanged.connect(self.update_reference_cm)
            self.inspector_layout.addWidget(spin)
            clear = QPushButton(tr('기준선 다시 선택'), objectName="secondaryButton")
            clear.clicked.connect(self.clear_current_points)
            self.inspector_layout.addWidget(clear)
        elif self.stage == 3:
            self.panel_label(
                f"Object {self.current_object}  ·  Arena {self.current_arena + 1}/{self.arena_count}",
                tr('도형을 드래그해 만든 뒤 내부 이동, 흰 손잡이 크기 조절, 초록 손잡이 회전으로 맞추세요.'),
            )
            self.add_exploration_distance_control()
            self.tool_group = QButtonGroup(self)
            self.tool_group.setExclusive(True)
            tools = (
                (tr('편집 (1)'), "edit", tr('도형 내부를 끌어 이동하고 조절점으로 크기·회전을 바꿉니다.')),
                (tr('원형 (2)'), "ellipse", tr('영상 위에서 드래그해 원형 Object를 만듭니다.')),
                (tr('사각형 (3)'), "rectangle", tr('영상 위에서 드래그해 사각형 Object를 만듭니다.')),
                (tr('다각형 (4)'), "polygon_points", tr('꼭짓점을 클릭하고 Enter로 다각형을 완성합니다.')),
                (tr('점 원형 (5)'), "circle_points", tr('중심과 가장자리 점으로 원형을 만듭니다.')),
                (tr('3점 사각형 (6)'), "rectangle_points", tr('세 점으로 회전된 사각형을 만듭니다.')),
                (tr('그리드 (7)'), "grid", tr('0.5 cm 그리드를 칠해 불규칙한 Object를 만듭니다.')),
            )
            tool_grid = QGridLayout()
            tool_grid.setHorizontalSpacing(8)
            tool_grid.setVerticalSpacing(8)
            for index, (text, mode, tooltip) in enumerate(tools):
                button = QPushButton(text, objectName="shapeToolButton")
                button.setCheckable(True)
                button.setChecked(self.object_mode == mode)
                button.setToolTip(tooltip)
                button.clicked.connect(lambda _checked=False, value=mode: self.set_object_mode(value))
                self.tool_group.addButton(button)
                row, column = divmod(index, 3)
                tool_grid.addWidget(button, row, column)
            self.inspector_layout.addLayout(tool_grid)
            if self.object_mode in ("polygon_points", "circle_points", "rectangle_points"):
                count = len(self.object_point_buffers.get((self.current_arena, self.current_object), []))
                self.add_progress_card(tr('선택한 점  {p0}', p0=count), tr('좌클릭 추가 · Backspace 마지막 점 취소'))
                finish = QPushButton(tr('점 도형 확정 (Enter)'), objectName="primaryButton")
                finish.clicked.connect(self.finalize_point_object)
                self.inspector_layout.addWidget(finish)
            elif self.object_mode == "grid":
                brush_cm = self.grid_brush_cells * 0.5
                self.add_progress_card(
                    tr('그리드 브러시  {p0}×{p1}셀', p0=self.grid_brush_cells, p1=self.grid_brush_cells),
                    tr('브러시 지름 약 {p0:g} cm · 좌드래그 칠하기 · 우드래그 지우기', p0=brush_cm),
                )
                self.inspector_layout.addWidget(QLabel(tr('브러시 크기')))
                self.brush_slider = QSlider(Qt.Horizontal)
                self.brush_slider.setRange(1, 9)
                self.brush_slider.setValue(self.grid_brush_cells)
                self.brush_slider.setTickInterval(1)
                self.brush_slider.valueChanged.connect(self.set_grid_brush_size)
                self.inspector_layout.addWidget(self.brush_slider)
                self.brush_size_readout = QLabel(tr('현재 지름: {p0:g} cm', p0=self.grid_brush_cells * 0.5))
                self.brush_size_readout.setStyleSheet("color:#72e5b2;font-weight:700")
                self.inspector_layout.addWidget(self.brush_size_readout)
            remove = QPushButton(tr('현재 오브젝트 삭제'), objectName="secondaryButton")
            remove.clicked.connect(lambda: self.remove_current_object())
            self.inspector_layout.addWidget(remove)
        else:
            self.panel_label(
                tr('검토 및 저장'),
                tr('잠긴 Arena ROI, cm 기준선, 오브젝트와 탐색영역을 확인한 뒤 저장하세요.')
                if self.bundle_mode else
                tr('ROI, 초록 cm 기준선, 오브젝트와 탐색 판정 범위를 확인한 뒤 저장하세요.'),
            )
            complete = sum(1 for arena in self.arenas if len(arena["objects"]) == 2)
            self.add_progress_card(tr('완료된 Arena  {p0}/{p1}', p0=complete, p1=self.arena_count))
            self.add_exploration_distance_control()
            if not self.bundle_mode:
                save_profile = QPushButton(tr('카메라 템플릿으로 저장'), objectName="secondaryButton")
                save_profile.clicked.connect(self.save_camera_profile)
                self.inspector_layout.addWidget(save_profile)
                next_video = QPushButton(tr('새 영상에 적용 · 오브젝트만'), objectName="primaryButton")
                next_video.clicked.connect(self.apply_current_camera_to_new_video)
                self.inspector_layout.addWidget(next_video)
            self.status_badge.setText(tr('●  저장 준비'))
        self.inspector_layout.addStretch()
        self.context_label.setText(self.context_text())

    def add_exploration_distance_control(self) -> None:
        """Expose the nose-to-object exploration boundary where it is edited and reviewed."""
        self.inspector_layout.addWidget(QLabel(tr('탐색 판정 거리'), objectName="toolGroupTitle"))
        spin = QDoubleSpinBox()
        spin.setRange(0.5, 5.0)
        spin.setDecimals(1)
        spin.setSingleStep(0.5)
        spin.setSuffix(" cm")
        spin.setValue(self.distance_cm)
        spin.setToolTip(
            tr('Object 외곽에서 바깥쪽으로 확장할 탐색 판정 범위입니다. 코 좌표가 이 경계 안에 들어온 프레임을 탐색 후보로 판정합니다.')
        )
        spin.valueChanged.connect(self.set_exploration_distance)
        self.inspector_layout.addWidget(spin)

    def set_exploration_distance(self, value: float) -> None:
        self.distance_cm = float(value)
        self.redraw_guides()

    def select_bundle_arena(self, index: int) -> None:
        if not self.bundle_mode or not 0 <= int(index) < self.arena_count:
            return
        index = int(index)
        if index == self.current_arena:
            return
        if self.stage == 2 and len(self.point_buffer) == 2:
            self.arenas[self.current_arena]["calibration_points"] = [
                [point.x(), point.y()] for point in self.point_buffer
            ]
        self.current_arena = index
        self.current_object = "A"
        self.object_mode = "edit"
        self.point_buffer = (
            [QPointF(*point) for point in self.arenas[index]["calibration_points"]]
            if self.stage == 2 else []
        )
        self.status_badge.setText(tr('●  Arena {p0:02d} 선택', p0=index + 1))
        self.refresh()

    def arena_at_point(self, point: QPointF) -> int | None:
        for index, arena in enumerate(self.arenas):
            corners = arena.get("corners") or []
            if len(corners) < 3:
                continue
            if cv2.pointPolygonTest(
                np.asarray(corners, dtype=np.float32),
                (float(point.x()), float(point.y())), False,
            ) >= 0:
                return index
        return None

    def jump_to_stage(self, target: int) -> None:
        """Navigate from the workflow sidebar while protecting incomplete prerequisites."""
        if self.roi_only:
            target = min(max(int(target), 0), 1)
        else:
            target = min(max(int(target), 0), 4)
        if (self.single_arena or self.bundle_mode) and target == 1:
            QMessageBox.information(
                self, tr('Arena ROI 잠금'),
                tr('이 영상은 트래킹 단계에서 이미 Arena ROI가 확정되었습니다.\n좌표 정합성을 위해 ROI 외곽은 수정할 수 없습니다.'),
            )
            return
        if target == self.stage:
            return
        if target > 0 and self.frame is None:
            QMessageBox.information(self, tr('영상 필요'), tr('먼저 프로젝트 단계에서 영상을 선택하세요.'))
            return

        # Preserve a complete draft before leaving its screen. An incomplete
        # draft never blocks moving backwards, but cannot unlock a later stage.
        if self.stage == 1 and len(self.point_buffer) == 4:
            self.arenas[self.current_arena]["corners"] = [[p.x(), p.y()] for p in self.point_buffer]
        elif self.stage == 2 and len(self.point_buffer) == 2:
            self.arenas[self.current_arena]["calibration_points"] = [[p.x(), p.y()] for p in self.point_buffer]

        if target >= 2:
            missing = [index + 1 for index, arena in enumerate(self.arenas) if len(arena["corners"]) < 3]
            if missing:
                QMessageBox.information(
                    self, tr('Arena ROI 미완성'),
                    tr('먼저 Arena ROI를 완료하세요.\n미완성 Arena: {p0}', p0=', '.join(map(str, missing))),
                )
                return
        if target >= 3:
            missing = [
                index + 1 for index, arena in enumerate(self.arenas)
                if len(arena["calibration_points"]) != 2
            ]
            if missing:
                QMessageBox.information(
                    self, tr('cm 보정 미완성'),
                    tr('먼저 기준선을 지정하세요.\n미완성 Arena: {p0}', p0=', '.join(map(str, missing))),
                )
                return
        if target >= 4:
            missing = [
                index + 1 for index in range(self.arena_count)
                if any((index, label) not in self.object_items for label in ("A", "B"))
            ]
            if missing:
                QMessageBox.information(
                    self, tr('오브젝트 미완성'),
                    tr('Object A/B를 모두 지정하세요.\n미완성 Arena: {p0}', p0=', '.join(map(str, missing))),
                )
                return

        self.stage = target
        self.current_arena = 0
        self.current_object = "A"
        self.object_mode = "edit"
        if target == 1:
            next_missing = next(
                (index for index, arena in enumerate(self.arenas) if len(arena["corners"]) < 3), 0
            )
            self.current_arena = next_missing
            self.point_buffer = [QPointF(*p) for p in self.arenas[next_missing]["corners"]]
        elif target == 2:
            next_missing = next(
                (index for index, arena in enumerate(self.arenas) if len(arena["calibration_points"]) != 2), 0
            )
            self.current_arena = next_missing
            self.point_buffer = [QPointF(*p) for p in self.arenas[next_missing]["calibration_points"]]
        else:
            self.point_buffer = []
        if target == 3:
            for arena_index in range(self.arena_count):
                for label in ("A", "B"):
                    if (arena_index, label) not in self.object_items:
                        self.current_arena = arena_index
                        self.current_object = label
                        break
                else:
                    continue
                break
        if target == 4:
            for item in self.object_items.values():
                item.setSelected(False)
        self.refresh()

    def add_progress_card(self, main: str, sub: str | None = None) -> None:
        card = QFrame(objectName="card")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(12, 8, 12, 8)
        label = QLabel(main)
        label.setStyleSheet("font-size:13px;font-weight:700;color:#eef5fd")
        layout.addWidget(label)
        if sub:
            card.setToolTip(sub)
            label.setToolTip(sub)
        self.inspector_layout.addWidget(card)

    def context_text(self) -> str:
        if self.stage in (1, 2):
            return f"Arena {self.current_arena + 1} / {self.arena_count}"
        if self.stage == 3:
            return f"Arena {self.current_arena + 1} / {self.arena_count}  ·  Object {self.current_object}"
        return self.steps[self.stage]

    def clear_guides(self) -> None:
        for item in self.guide_items:
            if item.scene() is self.scene:
                self.scene.removeItem(item)
        self.guide_items.clear()

    def add_dot(self, point: QPointF, color: QColor, radius: float = 5) -> None:
        # Keep control points at a constant screen size, independent of zoom.
        item = self.scene.addEllipse(
            -radius,
            -radius,
            radius * 2,
            radius * 2,
            cosmetic_pen(QColor("#0b111b"), 2),
            QBrush(color),
        )
        item.setPos(point)
        item.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        item.setZValue(20)
        self.guide_items.append(item)

    def redraw_guides(self) -> None:
        self.clear_guides()
        if self.frame is None:
            return
        for index, arena in enumerate(self.arenas):
            is_current_draft = self.stage == 1 and index == self.current_arena and bool(self.point_buffer)
            corners = [[point.x(), point.y()] for point in self.point_buffer] if is_current_draft else arena["corners"]
            if len(corners) >= 3:
                polygon = QPolygonF([QPointF(*p) for p in corners])
                color = QColor("#ffcf5a") if is_current_draft else (
                    QColor("#55d98a") if index == self.current_arena else QColor("#52657c")
                )
                item = self.scene.addPolygon(
                    polygon,
                    cosmetic_pen(color, 3 if is_current_draft else 2),
                    QBrush(QColor(color.red(), color.green(), color.blue(), 42 if is_current_draft else 24)),
                )
                item.setZValue(5)
                self.guide_items.append(item)
                center_x = sum(float(value[0]) for value in corners) / len(corners)
                center_y = sum(float(value[1]) for value in corners) / len(corners)
                label = self.scene.addText(
                    f"Arena {index + 1:02d}" + (tr('  현재 편집') if self.bundle_mode and index == self.current_arena else "")
                )
                label.setDefaultTextColor(color)
                label.setPos(center_x - 24, center_y - 13)
                label.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
                label.setZValue(21)
                self.guide_items.append(label)
            elif is_current_draft and len(corners) >= 2:
                points = [QPointF(*value) for value in corners]
                for start, end in zip(points, points[1:]):
                    line = self.scene.addLine(
                        start.x(), start.y(), end.x(), end.y(),
                        cosmetic_pen(QColor("#ffcf5a"), 2, Qt.DashLine),
                    )
                    line.setZValue(19)
                    self.guide_items.append(line)
            cal = arena["calibration_points"]
            if len(cal) == 2:
                cal_color = QColor("#55d98a") if index == self.current_arena else QColor("#52657c")
                line = self.scene.addLine(
                    cal[0][0], cal[0][1], cal[1][0], cal[1][1],
                    cosmetic_pen(cal_color, 3 if index == self.current_arena else 2)
                )
                line.setZValue(18)
                self.guide_items.append(line)
        for point in self.point_buffer:
            self.add_dot(point, QColor("#ffcf5a"))
        if self.stage == 3:
            object_points = self.object_point_buffers.get((self.current_arena, self.current_object), [])
            for point in object_points:
                self.add_dot(point, COLORS[self.current_object], 4)
            if len(object_points) >= 2:
                preview = self.scene.addPolygon(
                    QPolygonF(object_points), cosmetic_pen(COLORS[self.current_object], 1.5, Qt.DashLine)
                )
                preview.setZValue(19)
                self.guide_items.append(preview)
        if self.stage in (3, 4):
            self.draw_exploration_zones()

    def on_canvas_click(self, point: QPointF) -> None:
        self.clear_object_drag_preview()
        if self.frame is None or not self.scene.sceneRect().contains(point):
            return
        if self.bundle_mode and self.stage in (2, 3):
            clicked_arena = self.arena_at_point(point)
            if clicked_arena is not None and clicked_arena != self.current_arena:
                self.select_bundle_arena(clicked_arena)
                return
        required = 4 if self.stage == 1 else 2 if self.stage == 2 else 0
        if required and len(self.point_buffer) < required:
            self.point_buffer.append(point)
            if self.stage == 1 and len(self.point_buffer) == 4:
                self.status_badge.setText(tr('편집 중 · Arena {p0} ROI', p0=self.current_arena + 1))
            self.refresh()
            return
        if self.stage == 3 and self.object_mode in ("polygon_points", "circle_points", "rectangle_points"):
            key = (self.current_arena, self.current_object)
            points = self.object_point_buffers.setdefault(key, [])
            points.append(point)
            required_points = {"circle_points": 2, "rectangle_points": 3}.get(self.object_mode)
            if required_points and len(points) >= required_points:
                self.finalize_point_object()
            else:
                self.refresh()

    def on_canvas_drag(self, start: QPointF, end: QPointF) -> None:
        self.clear_object_drag_preview()
        if self.stage != 3 or self.object_mode not in ("ellipse", "rectangle"):
            return
        width, height = abs(end.x() - start.x()), abs(end.y() - start.y())
        if width < 8 or height < 8:
            return
        self.remove_current_object()
        rect = QRectF(min(start.x(), end.x()), min(start.y(), end.y()), width, height)
        item = EditableShapeItem(self.object_mode, rect, COLORS[self.current_object], self.current_object)
        self.scene.addItem(item)
        item.setSelected(True)
        self.object_items[(self.current_arena, self.current_object)] = item
        self.arenas[self.current_arena]["objects"][self.current_object] = {"item": item}
        self.object_mode = "edit"
        self.refresh()

    def clear_object_drag_preview(self) -> None:
        preview = self.object_drag_preview
        self.object_drag_preview = None
        if preview is None:
            return
        try:
            if preview.scene() is self.scene:
                self.scene.removeItem(preview)
        except RuntimeError:
            pass

    def on_canvas_drag_updated(self, start: QPointF, end: QPointF) -> None:
        """Draw the current ellipse/rectangle continuously until release."""
        if self.stage != 3 or self.object_mode not in ("ellipse", "rectangle") or self.frame is None:
            self.clear_object_drag_preview()
            return
        rect = QRectF(start, end).normalized().intersected(self.scene.sceneRect())
        if rect.width() < 1 or rect.height() < 1:
            self.clear_object_drag_preview()
            return
        expected_type = QGraphicsEllipseItem if self.object_mode == "ellipse" else QGraphicsRectItem
        if not isinstance(self.object_drag_preview, expected_type):
            self.clear_object_drag_preview()
            self.object_drag_preview = expected_type()
            self.object_drag_preview.setZValue(120)
            self.scene.addItem(self.object_drag_preview)
        color = QColor(COLORS[self.current_object])
        fill = QColor(color)
        fill.setAlpha(48)
        self.object_drag_preview.setPen(cosmetic_pen(color, 2.0, Qt.DashLine))
        self.object_drag_preview.setBrush(QBrush(fill))
        self.object_drag_preview.setRect(rect)

    def on_canvas_paint(self, point: QPointF, add: bool) -> None:
        if self.stage != 3 or self.object_mode != "grid" or self.frame is None:
            return
        arena = self.arenas[self.current_arena]
        if len(arena["calibration_points"]) != 2 or len(arena["corners"]) < 3:
            return
        inside = cv2.pointPolygonTest(
            np.asarray(arena["corners"], dtype=np.float32), (float(point.x()), float(point.y())), False
        )
        if inside < 0:
            return
        cell_px = max(self.calibration_ppcm(arena) * 0.5, 2.0)
        cell = (int(point.x() // cell_px), int(point.y() // cell_px))
        key = (self.current_arena, self.current_object)
        cells = self.grid_cells.setdefault(key, set())
        radius = self.grid_brush_cells * 0.5
        reach = int(math.ceil(radius))
        brush_cells: set[tuple[int, int]] = set()
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                if math.hypot(dx, dy) > radius:
                    continue
                candidate = (cell[0] + dx, cell[1] + dy)
                candidate_center = (
                    (candidate[0] + 0.5) * cell_px,
                    (candidate[1] + 0.5) * cell_px,
                )
                if cv2.pointPolygonTest(
                    np.asarray(arena["corners"], dtype=np.float32), candidate_center, False
                ) >= 0:
                    brush_cells.add(candidate)
        if add:
            cells.update(brush_cells)
        else:
            cells.difference_update(brush_cells)
        existing = self.object_items.get(key)
        if not cells:
            if existing and existing.scene() is self.scene:
                self.scene.removeItem(existing)
            self.object_items.pop(key, None)
            arena["objects"].pop(self.current_object, None)
            return
        if isinstance(existing, GridObjectItem):
            existing.cell_px = cell_px
            existing.rebuild()
        elif cells:
            self.remove_current_object(clear_drafts=False)
            item = GridObjectItem(
                cells,
                cell_px,
                (self.frame.shape[1], self.frame.shape[0]),
                COLORS[self.current_object],
                self.current_object,
            )
            self.scene.addItem(item)
            self.object_items[key] = item
            arena["objects"][self.current_object] = {"item": item}

    def clear_current_points(self) -> None:
        self.point_buffer.clear()
        if self.stage == 1:
            self.arenas[self.current_arena]["corners"] = []
        elif self.stage == 2:
            self.arenas[self.current_arena]["calibration_points"] = []
        self.refresh()

    def update_reference_cm(self, value: float) -> None:
        self.arenas[self.current_arena]["reference_cm"] = float(value)

    def set_arena_count(self, value: int) -> None:
        value = int(value)
        if value == self.arena_count:
            return
        if value > self.arena_count:
            self.arenas.extend(
                {"corners": [], "calibration_points": [], "reference_cm": 10.0, "objects": {}}
                for _ in range(value - self.arena_count)
            )
        else:
            for key, item in list(self.object_items.items()):
                if key[0] >= value:
                    if item.scene() is self.scene:
                        self.scene.removeItem(item)
                    self.object_items.pop(key, None)
            self.object_point_buffers = {
                key: points for key, points in self.object_point_buffers.items() if key[0] < value
            }
            self.grid_cells = {key: cells for key, cells in self.grid_cells.items() if key[0] < value}
            self.arenas = self.arenas[:value]
        self.arena_count = value
        self.current_arena = min(self.current_arena, value - 1)
        self.context_label.setText(self.context_text())
        self.redraw_guides()

    def set_object_mode(self, mode: str) -> None:
        self.clear_object_drag_preview()
        self.object_mode = mode
        self.view.creation_enabled = mode in ("ellipse", "rectangle")
        self.view.paint_enabled = mode == "grid"
        if mode == "edit":
            for item in self.object_items.values():
                if isinstance(item, EditableShapeItem):
                    item.interaction_mode = "edit"
                    item.setFlag(QGraphicsItem.ItemIsMovable, True)
                    item.update()
                elif isinstance(item, PolygonObjectItem):
                    item.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.rebuild_panel()
        self.redraw_guides()

    def set_grid_brush_size(self, value: int) -> None:
        self.grid_brush_cells = int(value)
        if hasattr(self, "brush_slider"):
            self.brush_slider.blockSignals(True)
            self.brush_slider.setValue(self.grid_brush_cells)
            self.brush_slider.blockSignals(False)
        if hasattr(self, "brush_size_readout"):
            self.brush_size_readout.setText(tr('현재 지름: {p0:g} cm', p0=self.grid_brush_cells * 0.5))
        self.status_badge.setText(tr('●  브러시 {p0}×{p1}셀', p0=self.grid_brush_cells, p1=self.grid_brush_cells))

    def adjust_grid_brush_size(self, delta: int) -> None:
        if self.stage == 3 and self.object_mode == "grid":
            self.set_grid_brush_size(min(max(self.grid_brush_cells + int(delta), 1), 9))

    def update_brush_cursor(self, point: QPointF) -> None:
        if self.brush_cursor is not None:
            try:
                if self.brush_cursor.scene() is self.scene:
                    self.scene.removeItem(self.brush_cursor)
            except RuntimeError:
                pass
            self.brush_cursor = None
        if self.stage != 3 or self.object_mode != "grid":
            return
        arena = self.arenas[self.current_arena]
        if len(arena["calibration_points"]) != 2:
            return
        cell_px = max(self.calibration_ppcm(arena) * 0.5, 2.0)
        diameter = self.grid_brush_cells * cell_px
        snapped = QPointF(
            (int(point.x() // cell_px) + 0.5) * cell_px,
            (int(point.y() // cell_px) + 0.5) * cell_px,
        )
        cursor = self.scene.addEllipse(
            snapped.x() - diameter * 0.5,
            snapped.y() - diameter * 0.5,
            diameter,
            diameter,
            cosmetic_pen(QColor("#ffffff"), 1.5, Qt.DashLine),
            QBrush(Qt.NoBrush),
        )
        cursor.setZValue(60)
        self.brush_cursor = cursor

    def remove_current_object(self, clear_drafts: bool = True) -> None:
        key = (self.current_arena, self.current_object)
        item = self.object_items.pop(key, None)
        if item and item.scene() is self.scene:
            self.scene.removeItem(item)
        self.arenas[self.current_arena]["objects"].pop(self.current_object, None)
        if clear_drafts:
            self.object_point_buffers.pop(key, None)
            self.grid_cells.pop(key, None)
        self.redraw_guides()

    def finalize_point_object(self) -> None:
        key = (self.current_arena, self.current_object)
        points = self.object_point_buffers.get(key, [])
        mode = self.object_mode
        needed = 3 if mode in ("polygon_points", "rectangle_points") else 2
        if len(points) < needed:
            QMessageBox.information(self, tr('점 부족'), tr('이 도형에는 최소 {p0}개의 점이 필요합니다.', p0=needed))
            return
        self.remove_current_object(clear_drafts=False)
        color = COLORS[self.current_object]
        if mode == "circle_points":
            center, edge = points[:2]
            radius = math.hypot(edge.x() - center.x(), edge.y() - center.y())
            rect = QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2)
            item: QGraphicsItem = EditableShapeItem("ellipse", rect, color, self.current_object)
        elif mode == "rectangle_points":
            first, adjacent, opposite = points[:3]
            dx, dy = adjacent.x() - first.x(), adjacent.y() - first.y()
            width = math.hypot(dx, dy)
            if width < 2:
                return
            nx, ny = -dy / width, dx / width
            signed_height = (opposite.x() - first.x()) * nx + (opposite.y() - first.y()) * ny
            height = abs(signed_height)
            center = QPointF(
                (first.x() + adjacent.x()) * 0.5 + nx * signed_height * 0.5,
                (first.y() + adjacent.y()) * 0.5 + ny * signed_height * 0.5,
            )
            rect = QRectF(center.x() - width / 2, center.y() - height / 2, width, max(height, 8))
            item = EditableShapeItem("rectangle", rect, color, self.current_object)
            item.setRotation(math.degrees(math.atan2(dy, dx)))
        else:
            item = PolygonObjectItem(points, color, self.current_object)
        self.scene.addItem(item)
        item.setSelected(True)
        self.object_items[key] = item
        self.arenas[self.current_arena]["objects"][self.current_object] = {"item": item}
        self.object_point_buffers.pop(key, None)
        self.object_mode = "edit"
        self.refresh()

    def commit_points(self) -> bool:
        if self.stage == 1:
            if len(self.point_buffer) != 4:
                QMessageBox.information(self, tr('ROI 미완성'), tr('바닥 영역의 점 네 개를 먼저 선택하세요.'))
                return False
            self.arenas[self.current_arena]["corners"] = [[p.x(), p.y()] for p in self.point_buffer]
        elif self.stage == 2:
            if len(self.point_buffer) != 2:
                QMessageBox.information(self, tr('기준선 미완성'), tr('기준 길이의 양 끝점을 선택하세요.'))
                return False
            self.arenas[self.current_arena]["calibration_points"] = [[p.x(), p.y()] for p in self.point_buffer]
        return True

    def go_next(self) -> None:
        if self.stage == 0:
            if self.frame is None:
                QMessageBox.information(self, tr('영상 필요'), tr('먼저 분석할 영상을 선택하세요.'))
                return
            if self.bundle_mode or self.single_arena:
                self.stage = 2
                self.current_arena = 0
                self.point_buffer = [
                    QPointF(*point) for point in self.arenas[0]["calibration_points"]
                ]
            elif self.camera_locked:
                expected = self.camera_profile_resolution
                actual = (self.frame.shape[1], self.frame.shape[0])
                if expected and actual != expected:
                    QMessageBox.critical(
                        self, tr('해상도 불일치'), tr('템플릿 {p0}×{p1} / 영상 {p2}×{p3}', p0=expected[0], p1=expected[1], p2=actual[0], p3=actual[1])
                    )
                    return
                if not self.camera_profile_ready():
                    QMessageBox.critical(self, tr('템플릿 오류'), tr('ROI 또는 cm 보정값이 누락됐습니다.'))
                    return
                self.stage = 3
                self.current_arena = 0
                self.current_object = "A"
                self.point_buffer = []
            else:
                self.stage = 1
                self.current_arena = 0
                self.point_buffer = [QPointF(*p) for p in self.arenas[0]["corners"]]
        elif self.stage in (1, 2):
            if not self.commit_points():
                return
            if self.current_arena < self.arena_count - 1:
                self.current_arena += 1
                key = "corners" if self.stage == 1 else "calibration_points"
                self.point_buffer = [QPointF(*p) for p in self.arenas[self.current_arena][key]]
            else:
                if self.roi_only and self.stage == 1:
                    if self.save_roi_profile():
                        self.close()
                    return
                self.stage += 1
                self.current_arena = 0
                self.point_buffer = [QPointF(*p) for p in self.arenas[0]["calibration_points"]] if self.stage == 2 else []
        elif self.stage == 3:
            current_item = self.object_items.get((self.current_arena, self.current_object))
            if current_item is None or not current_item.scene_polygon():
                QMessageBox.information(self, tr('오브젝트 미완성'), tr('Object {p0} 도형을 먼저 만드세요.', p0=self.current_object))
                return
            if self.current_object == "A":
                self.current_object = "B"
            elif self.current_arena < self.arena_count - 1:
                self.current_arena += 1
                self.current_object = "A"
            else:
                self.stage = 4
                for item in self.object_items.values():
                    item.setSelected(False)
        else:
            self.save_setup(close_after=True)
            return
        self.object_mode = "edit"
        self.refresh()

    def go_back(self) -> None:
        if self.stage == 0:
            return
        if (self.single_arena or self.bundle_mode) and self.stage == 2 and self.current_arena == 0:
            self.stage = 0
            self.point_buffer = []
            self.refresh()
            return
        if self.stage in (1, 2) and self.current_arena > 0:
            self.current_arena -= 1
            key = "corners" if self.stage == 1 else "calibration_points"
            self.point_buffer = [QPointF(*p) for p in self.arenas[self.current_arena][key]]
        elif self.stage == 3:
            if self.current_object == "B":
                self.current_object = "A"
            elif self.current_arena > 0:
                self.current_arena -= 1
                self.current_object = "B"
            else:
                if self.bundle_mode:
                    self.stage = 2
                    self.current_arena = 0
                    self.point_buffer = [
                        QPointF(*p) for p in self.arenas[0]["calibration_points"]
                    ]
                elif self.camera_locked:
                    self.stage = 0
                    self.current_arena = 0
                    self.point_buffer = []
                else:
                    self.stage = 2
                    self.current_arena = self.arena_count - 1
                    self.point_buffer = [
                        QPointF(*p) for p in self.arenas[self.current_arena]["calibration_points"]
                    ]
        elif self.stage == 4:
            self.stage = 3
            self.current_arena = self.arena_count - 1
            self.current_object = "B"
        else:
            self.stage -= 1
            self.current_arena = self.arena_count - 1
            key = "corners" if self.stage == 1 else "calibration_points"
            self.point_buffer = [QPointF(*p) for p in self.arenas[self.current_arena][key]]
        self.refresh()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Backspace and self.point_buffer:
            self.point_buffer.pop()
            self.refresh()
            return
        if event.key() == Qt.Key_Backspace and self.stage == 3:
            key = (self.current_arena, self.current_object)
            points = self.object_point_buffers.get(key, [])
            if points:
                points.pop()
                self.refresh()
                return
        if event.key() == Qt.Key_Delete and self.stage == 3:
            self.remove_current_object()
            return
        super().keyPressEvent(event)

    def calibration_ppcm(self, arena: dict) -> float:
        points = np.asarray(arena["calibration_points"], dtype=np.float64)
        return float(np.linalg.norm(points[1] - points[0]) / float(arena["reference_cm"]))

    def expanded_polygon(self, polygon: list[list[float]], radius_px: int) -> list[list[float]]:
        assert self.frame is not None
        mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius_px * 2 + 1, radius_px * 2 + 1))
        expanded = cv2.dilate(mask, kernel)
        contours, _ = cv2.findContours(expanded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float).tolist()

    def draw_exploration_zones(self) -> None:
        for arena_index, arena in enumerate(self.arenas):
            if len(arena["calibration_points"]) != 2:
                continue
            radius = max(1, int(round(self.distance_cm * self.calibration_ppcm(arena))))
            for label in ("A", "B"):
                item = self.object_items.get((arena_index, label))
                if not item:
                    continue
                expanded = self.expanded_polygon(item.scene_polygon(), radius)
                polygon = QPolygonF([QPointF(*p) for p in expanded])
                zone = self.scene.addPolygon(
                    polygon, cosmetic_pen(COLORS[label], 1.5, Qt.DashLine), QBrush(Qt.NoBrush)
                )
                zone.setZValue(12)
                self.guide_items.append(zone)

    def save_bundle_setups(self) -> bool:
        """Save the full-source editor state as one crop-local setup per tracked Arena."""
        saved: list[Path] = []
        for index, (arena, record) in enumerate(zip(self.arenas, self.bundle_records)):
            output_text = str(record.get("output", "")).strip()
            if not output_text:
                continue
            output = Path(output_text)
            offset = record.get("source_offset_px") or [0, 0]
            dx, dy = float(offset[0]), float(offset[1])
            crop_size = record.get("crop_size_px") or [0, 0]
            width, height = int(crop_size[0]), int(crop_size[1])
            ppcm = self.calibration_ppcm(arena)
            radius = max(1, int(round(self.distance_cm * ppcm)))

            def crop_local(points: list) -> list[list[float]]:
                return self.shifted_points(points, -dx, -dy)

            objects = {}
            for label in ("A", "B"):
                item = self.object_items[(index, label)]
                polygon_source = item.scene_polygon()
                objects[label] = {
                    "shape": item.shape_kind,
                    "polygon_source_px": crop_local(polygon_source),
                    "rotation_degrees": float(item.rotation() % 360),
                    "exploration_boundary_source_px": crop_local(
                        self.expanded_polygon(polygon_source, radius)
                    ),
                    "selection_reference": "floor_footprint",
                }
            crop_corners = record.get("crop_corners_px") or [
                [0.0, 0.0], [float(max(width - 1, 0)), 0.0],
                [float(max(width - 1, 0)), float(max(height - 1, 0))],
                [0.0, float(max(height - 1, 0))],
            ]
            arena_id = int(record.get("arena_id", index + 1))
            payload = {
                "schema_version": 3,
                "created_by": "HAGISILTA · NORT multi-Arena editor",
                "source_video": str(Path(str(record.get("roi_video") or self.video)).resolve()),
                "source_full_video": str(self.video.resolve()) if self.video else "",
                "video": record.get("video_meta") or {
                    "width": width, "height": height,
                },
                "arena_dimensions_required": False,
                "arena_count": 1,
                "exploration_distance_cm": self.distance_cm,
                "distance_reference": "object_boundary",
                "metric_calibration": "local_reference_line_per_arena",
                "camera_profile": str(self.bundle_manifest_path) if self.bundle_manifest_path else None,
                "camera_profile_locked": True,
                "coordinate_transform": {
                    "display_space": "source_video_px",
                    "saved_space": "arena_crop_px",
                    "source_offset_px": [dx, dy],
                },
                "arenas": [{
                    "id": arena_id,
                    "corners_source_px": crop_corners,
                    "corner_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
                    "pixels_per_cm": ppcm,
                    "calibration": {
                        "method": "local_reference_line",
                        "points_source_px": crop_local(arena["calibration_points"]),
                        "actual_length_cm": arena["reference_cm"],
                        "pixel_length": ppcm * arena["reference_cm"],
                        "pixels_per_cm": ppcm,
                        "plane": "arena_floor",
                    },
                    "objects": objects,
                }],
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            saved.append(output)
        self.status_badge.setText(tr('●  {p0}개 Arena 저장 완료', p0=len(saved)))
        print(f"NORT_BUNDLE_SAVED={len(saved)}", flush=True)
        QMessageBox.information(
            self, tr('저장 완료'),
            tr('원본 화면에서 편집한 설정을 {p0}개 Arena 파일로 나누어 저장했습니다.', p0=len(saved))
        )
        return bool(saved)

    def save_setup(self, close_after: bool = False) -> None:
        if not self.video:
            return
        if self.bundle_mode:
            if self.save_bundle_setups() and close_after:
                self.close()
            return
        output = self.output or DEFAULT_OUTPUT_DIR / f"{self.video.stem}_nort_setup.json"
        if self.output is None:
            selected, _ = QFileDialog.getSaveFileName(self, tr('NORT 공간·오브젝트 설정 저장'), str(output), "JSON (*.json)")
            if not selected:
                return
            output = Path(selected)
        payload_arenas = []
        for index, arena in enumerate(self.arenas):
            ppcm = self.calibration_ppcm(arena)
            radius = max(1, int(round(self.distance_cm * ppcm)))
            objects = {}
            for label in ("A", "B"):
                item = self.object_items[(index, label)]
                polygon = item.scene_polygon()
                objects[label] = {
                    "shape": item.shape_kind,
                    "polygon_source_px": polygon,
                    "rotation_degrees": float(item.rotation() % 360),
                    "exploration_boundary_source_px": self.expanded_polygon(polygon, radius),
                    "selection_reference": "floor_footprint",
                }
                if isinstance(item, GridObjectItem):
                    objects[label]["grid_cell_cm"] = 0.5
                    objects[label]["grid_cell_px"] = item.cell_px
                    objects[label]["selected_grid_cells"] = [list(cell) for cell in sorted(item.cells)]
            payload_arenas.append({
                "id": self.arena_id if self.single_arena else index + 1,
                "corners_source_px": arena["corners"],
                "corner_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
                "pixels_per_cm": ppcm,
                "calibration": {
                    "method": "local_reference_line",
                    "points_source_px": arena["calibration_points"],
                    "actual_length_cm": arena["reference_cm"],
                    "pixel_length": ppcm * arena["reference_cm"],
                    "pixels_per_cm": ppcm,
                    "plane": "arena_floor",
                },
                "objects": objects,
            })
        payload = {
            "schema_version": 3,
            "created_by": "HAGISILTA · NORT Arena·Object editor",
            "source_video": str(self.video.resolve()),
            "video": self.video_meta,
            "arena_dimensions_required": False,
            "arena_count": self.arena_count,
            "exploration_distance_cm": self.distance_cm,
            "distance_reference": "object_boundary",
            "metric_calibration": "local_reference_line_per_arena",
            "camera_profile": str(self.camera_profile_path.resolve()) if self.camera_profile_path else None,
            "camera_profile_locked": self.camera_locked,
            "arenas": payload_arenas,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.status_badge.setText(tr('●  저장 완료'))
        QMessageBox.information(self, tr('저장 완료'), tr('NORT 공간·오브젝트 설정을 저장했습니다.\n\n{p0}', p0=output))
        if close_after:
            self.close()


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv)
    init_i18n(app)
    install_wheel_value_guard(app)
    app.setApplicationName("HAGISILTA")
    window = NortStudio(
        Path(args.video) if args.video else None,
        args.time,
        args.output,
        roi_only=args.roi_only,
        profile_output=args.profile_output,
        initial_profile=args.profile,
        single_arena=args.single_arena,
        arena_id=args.arena_id,
        bundle_manifest=args.bundle_manifest,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
