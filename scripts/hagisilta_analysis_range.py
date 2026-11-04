# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
"""Non-destructive analysis intervals; seconds are on the source video clock."""
from __future__ import annotations

from hagisilta_i18n import tr

import json
import math
import os
from pathlib import Path

from hagisilta_atomic_io import atomic_json


def format_time(seconds: float) -> str:
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return tr('확인 필요')
    if not math.isfinite(seconds) or seconds < 0:
        return tr('확인 필요')
    milliseconds = round(seconds * 1000)
    minutes, remainder = divmod(milliseconds, 60000)
    return f"{minutes:02d}:{remainder / 1000:06.3f}"


def validate_range(start: float, end: float, duration: float) -> None:
    if not all(math.isfinite(v) for v in (start, end, duration)) or duration <= 0:
        raise ValueError(tr('영상 길이 또는 구간 값을 확인할 수 없습니다.'))
    if start < 0 or end <= start:
        raise ValueError(tr('시작은 0 이상, 종료는 시작보다 뒤여야 합니다.'))
    # UI stores milliseconds; tolerate only its rounding, not a missing frame.
    if end > duration + 0.00051:
        raise ValueError(tr('영상 길이가 {v0:.3f}초 부족합니다. 구간을 수정하세요.', v0=end - duration))


def frame_bounds(start: float, end: float | None, fps: float, count: int) -> tuple[int, int]:
    if not math.isfinite(fps) or fps <= 0 or count <= 0:
        raise ValueError(tr('FPS 또는 프레임 수를 확인할 수 없습니다.'))
    end = count / fps if end is None else float(end)
    validate_range(float(start), end, count / fps)
    # Half-open interval [start, end): no preceding frame and no end-frame leak.
    first = max(0, math.ceil(float(start) * fps - 1e-8))
    last = min(count, math.ceil(end * fps - 1e-8))
    if first >= last:
        raise ValueError(tr('분석 구간에 포함된 프레임이 없습니다.'))
    return first, last


def load_ymaze_range(arena: Path) -> dict:
    path = Path(arena) / "ymaze-analysis-range.json"
    if not path.is_file():
        return {}  # Legacy projects still mean the full video, with no write.
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(tr('구간 설정 형식 오류: {v0}', v0=path))
    return payload


def save_ymaze_range(arena: Path, start: float, end: float, duration: float) -> None:
    validate_range(start, end, duration)
    atomic_json(Path(arena) / "ymaze-analysis-range.json", {
        "schema_version": 1, "experiment_type": "Y-maze", "time_basis": "source_video_seconds",
        "start_seconds": start, "end_seconds": end, "duration_seconds": end - start,
    })


def same_video_ymaze_arenas(arena: Path) -> list[Path]:
    """Stay in the selected recording bundle; reject conflicting/broken provenance.

    Legacy bundles without metadata still group their own arena_* children.
    Never match other recordings by filename stems or by analysis-use checks.
    """
    arena = Path(arena)

    def source(path):
        metadata = path / "roi-metadata.json"
        if not metadata.is_file():
            return ""
        payload = json.loads(metadata.read_text(encoding="utf-8-sig"))
        value = str(payload.get("source_video") or "")
        return os.path.normcase(os.path.abspath(value)) if value else ""

    try:
        reference = source(arena)
    except (OSError, ValueError, AttributeError):
        return [arena]
    result = [arena]
    for candidate in sorted(arena.parent.glob("arena_*")):
        if candidate == arena or not candidate.is_dir():
            continue
        try:
            if source(candidate) == reference:
                result.append(candidate)
        except (OSError, ValueError, AttributeError):
            continue
    return result


def propose_ranges(records: list[dict], mode: str, length: float, reference: tuple[float, float]) -> list[dict]:
    """Return proposals/errors without mutating records or shortening requests."""
    proposals = []
    for record in records:
        start, end = float("nan"), float("nan")
        error = str(record.get("source_error", record.get("error")) or "")
        try:
            start = float(record["start"]) if mode == "duration" else float(reference[0])
            end = start + length if mode == "duration" else float(reference[1])
            validate_range(start, end, float(record["duration"]))
        except (ValueError, TypeError) as exc:
            error = error or str(exc)
        proposals.append({**record, "new_start": start, "new_end": end, "error": error})
    return proposals
