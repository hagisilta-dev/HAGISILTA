# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
"""Read-only, Arena-local Water-maze ROI overlays for video viewers.

Never construct default geometry or update a saved profile while viewing it.
Coordinates refer to the original Arena frame, not the letterboxed widget.
"""
from __future__ import annotations

from hagisilta_i18n import tr

from dataclasses import dataclass
import json
import math
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap, QPolygonF


@dataclass(frozen=True)
class WaterRoiOverlay:
    frame_size: tuple[int, int]
    pool_center: tuple[float, float]
    pool_radii: tuple[float, float]
    platform_center: tuple[float, float] | None
    platform_radius: float | None
    platform_points: tuple[tuple[float, float], ...]
    north_angle: float | None
    wall_fraction: float | None

    @property
    def available(self) -> frozenset[str]:
        keys = {"pool"}
        if self.platform_center is not None or self.platform_points:
            keys.add("platform")
        if self.north_angle is not None:
            keys.add("quadrants")
        if self.wall_fraction is not None and self.wall_fraction > 0:
            keys.add("wall")
        return frozenset(keys)


def _number(value, *, positive=False) -> float:
    if isinstance(value, bool):
        raise ValueError(tr('잘못된 좌표'))
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(tr('잘못된 좌표'))
    return number


def _point(value) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(tr('잘못된 좌표'))
    return _number(value[0]), _number(value[1])


def load_water_roi_overlay(profile_path: Path | None, video: Path,
                           frame_size: tuple[int, int]) -> WaterRoiOverlay:
    """Load only this Arena's saved geometry; legacy omitted space/size is allowed.

An explicit source, coordinate-space or frame-size mismatch is never remapped.
Missing optional components stay absent rather than becoming guessed ROIs.
"""
    if profile_path is None or not Path(profile_path).is_file():
        raise ValueError(tr('저장된 ROI 없음 · 수조·플랫폼 설정 후 표시됩니다.'))
    try:
        payload = json.loads(Path(profile_path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise ValueError(tr('ROI 파일을 읽을 수 없습니다 · 수조·플랫폼 설정을 확인하세요.')) from error
    if not isinstance(payload, dict) or payload.get("profile_type") != "hagisilta_watermaze_profile":
        raise ValueError(tr('Water-maze ROI 파일이 아닙니다.'))
    source = str(payload.get("source_video") or "").strip()
    if source and str(Path(source).resolve()).casefold() != str(Path(video).resolve()).casefold():
        raise ValueError(tr('다른 영상의 ROI입니다 · 이 Arena의 설정을 확인하세요.'))
    if payload.get("coordinate_space", "arena_crop_px") != "arena_crop_px":
        raise ValueError(tr('ROI 좌표 기준이 다릅니다 · 이 Arena의 설정을 확인하세요.'))
    try:
        width, height = frame_size
        if width <= 0 or height <= 0:
            raise ValueError(tr('영상 크기를 확인할 수 없습니다.'))
        if "frame_size" in payload and _point(payload["frame_size"]) != (width, height):
            raise ValueError(tr('ROI와 영상 크기가 다릅니다 · 이 Arena의 설정을 확인하세요.'))
        pool = payload["pool"]
        if pool.get("shape_type", "circle") not in ("circle", "ellipse"):
            raise ValueError(tr('지원하지 않는 수조 모양입니다.'))
        center = _point(pool["center_px"])
        radii = (_number(pool.get("radius_x_px", pool.get("radius_px")), positive=True),
                 _number(pool.get("radius_y_px", pool.get("radius_px")), positive=True))
        platform_center, platform_radius, platform_points = None, None, ()
        platform = payload.get("platform")
        if platform:
            shape = platform.get("shape_type", "circle")
            if shape == "circle":
                platform_center = _point(platform["center_px"])
                platform_radius = _number(platform["radius_px"], positive=True)
            elif shape in ("rectangle", "polygon"):
                platform_points = tuple(_point(point) for point in platform["points_px"])
                if len(platform_points) < 3:
                    raise ValueError(tr('플랫폼 꼭짓점이 부족합니다.'))
                area = sum(a[0] * b[1] - b[0] * a[1] for a, b in
                           zip(platform_points, platform_points[1:] + platform_points[:1]))
                if abs(area) < 1e-6:
                    raise ValueError(tr('플랫폼 영역이 없습니다.'))
            else:
                raise ValueError(tr('지원하지 않는 플랫폼 모양입니다.'))
        north = _number(payload["north_angle_degrees"]) if "north_angle_degrees" in payload else None
        wall = None
        if "wall_zone_width_cm" in payload:
            diameter = _number(pool["diameter_cm"], positive=True)
            wall = _number(payload["wall_zone_width_cm"]) / (diameter / 2)
        elif "peripheral_ring_percent" in payload:
            wall = _number(payload["peripheral_ring_percent"])
        if wall is not None:
            wall = max(0., min(wall, .95))
        return WaterRoiOverlay((width, height), center, radii, platform_center,
                               platform_radius, platform_points, north, wall)
    except (KeyError, TypeError, AttributeError, OverflowError, ValueError) as error:
        raise ValueError(tr('ROI 표시 불가 · {v0} · 수조·플랫폼 설정을 확인하세요.', v0=error)) from error


def paint_water_roi_overlay(scaled_frame: QPixmap, overlay: WaterRoiOverlay | None,
                            visible: set[str]) -> QPixmap:
    """Paint onto a fresh, already aspect-fitted frame; keep screen line widths stable."""
    result = scaled_frame.copy()
    if result.isNull() or overlay is None or not visible:
        return result
    visible = visible & overlay.available
    sx = result.width() / overlay.frame_size[0]
    sy = result.height() / overlay.frame_size[1]

    def point(x, y):
        return QPointF(x * sx, y * sy)

    def ellipse(center, radii):
        x, y = center
        rx, ry = radii
        return QRectF((x - rx) * sx, (y - ry) * sy, rx * 2 * sx, ry * 2 * sy)

    painter = QPainter(result)
    try:
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(Qt.NoBrush)
        cx, cy = overlay.pool_center
        rx, ry = overlay.pool_radii
        if "quadrants" in visible:
            painter.setPen(QPen(QColor("#d5e1ef"), 1.5, Qt.DashLine))
            for boundary in (0, 90, 180, 270):
                angle = math.radians(overlay.north_angle + boundary)
                painter.drawLine(point(cx, cy), point(cx + rx * math.sin(angle), cy - ry * math.cos(angle)))
            font = painter.font()
            font.setPixelSize(13)
            font.setBold(True)
            painter.setFont(font)
            for label, offset in (("NE", 45), ("SE", 135), ("SW", 225), ("NW", 315)):
                angle = math.radians(overlay.north_angle + offset)
                position = point(cx + .60 * rx * math.sin(angle), cy - .60 * ry * math.cos(angle))
                box = QRectF(position.x() - 20, position.y() - 12, 40, 24)
                painter.setPen(QColor("#111217"))
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    painter.drawText(box.translated(dx, dy), Qt.AlignCenter, label)
                painter.setPen(QColor("#ffffff"))
                painter.drawText(box, Qt.AlignCenter, label)
        if "wall" in visible:
            painter.setPen(QPen(QColor("#ffd35c"), 2, Qt.DashLine))
            painter.drawEllipse(ellipse(overlay.pool_center,
                                        (rx * (1 - overlay.wall_fraction), ry * (1 - overlay.wall_fraction))))
        if "pool" in visible:
            painter.setPen(QPen(QColor("#60ddb1"), 2))
            painter.drawEllipse(ellipse(overlay.pool_center, overlay.pool_radii))
        if "platform" in visible:
            painter.setPen(QPen(QColor("#ff667e"), 2.5))
            if overlay.platform_points:
                painter.drawPolygon(QPolygonF([point(x, y) for x, y in overlay.platform_points]))
            else:
                painter.drawEllipse(ellipse(overlay.platform_center,
                                            (overlay.platform_radius, overlay.platform_radius)))
    finally:
        painter.end()
    return result
