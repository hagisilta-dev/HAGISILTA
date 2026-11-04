# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
"""Font fallback for analysis figures and text drawn on image arrays."""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def korean_font_path() -> str | None:
    """Find an installed Hangul font without requiring a Qt application."""
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font

    for family in (
        "Malgun Gothic", "Noto Sans CJK KR", "Noto Sans KR", "NanumGothic",
        "Apple SD Gothic Neo", "AppleGothic",
    ):
        try:
            path = font_manager.findfont(
                font_manager.FontProperties(family=family), fallback_to_default=False,
            )
            if FT2Font(path).get_char_index(ord("한")):
                return path
        except (ValueError, OSError, RuntimeError):
            continue
    return None


def configure_plot_fonts() -> None:
    """Keep the current Latin font and append an installed Hangul fallback."""
    import matplotlib
    from matplotlib import font_manager

    path = korean_font_path()
    if path is None:
        return
    family = font_manager.FontProperties(fname=path).get_name()
    families = list(matplotlib.rcParams["font.family"])
    if family not in families:
        matplotlib.rcParams["font.family"] = [*families, family]


@lru_cache(maxsize=64)
def _image_font(pixel_size: int):
    from PIL import ImageFont

    path = korean_font_path()
    if path is None:
        from matplotlib import font_manager
        path = font_manager.findfont(font_manager.FontProperties())
    return ImageFont.truetype(path, pixel_size)


def _image_text_metrics(text: str, font_scale: float, thickness: int):
    font = _image_font(max(1, round(30 * font_scale)))
    stroke = max(0, (thickness - 1) // 2)
    bounds = font.getbbox(text, anchor="ls", stroke_width=stroke)
    return font, stroke, bounds


def image_text_size(text: str, font_face: int, font_scale: float, thickness: int = 1):
    """Return OpenCV-style width, ascent and baseline for the actual renderer."""
    import cv2

    if text.isascii():
        return cv2.getTextSize(text, font_face, font_scale, thickness)
    _font, _stroke, (left, top, right, bottom) = _image_text_metrics(text, font_scale, thickness)
    return (right - left, max(0, -top)), max(0, bottom)


def draw_image_text(
    image, text: str, origin: tuple[int, int], font_face: int,
    font_scale: float, color, thickness: int = 1, line_type: int | None = None,
):
    """Draw in place on a BGR/BGRA image, retaining OpenCV's exact ASCII path.

    Unicode labels use a system TrueType font with the same baseline origin.
    Only the text's clipped bounding rectangle is converted for Pillow.
    """
    import cv2

    if line_type is None:
        line_type = cv2.LINE_8
    if text.isascii():
        return cv2.putText(image, text, origin, font_face, font_scale, color, thickness, line_type)

    import numpy as np
    from PIL import Image, ImageDraw

    font, stroke, bounds = _image_text_metrics(text, font_scale, thickness)
    x, y = origin
    height, width = image.shape[:2]
    left, top = max(0, x + bounds[0]), max(0, y + bounds[1])
    right, bottom = min(width, x + bounds[2]), min(height, y + bounds[3])
    if right <= left or bottom <= top:
        return image
    channels = [2, 1, 0, 3] if image.shape[2] == 4 else [2, 1, 0]
    tile = Image.fromarray(np.ascontiguousarray(image[top:bottom, left:right, channels]))
    fill = tuple(int(color[index]) for index in channels)
    ImageDraw.Draw(tile).text(
        (x - left, y - top), text, font=font, fill=fill, anchor="ls",
        stroke_width=stroke, stroke_fill=fill,
    )
    image[top:bottom, left:right] = np.asarray(tile)[..., channels]
    return image
