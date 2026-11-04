# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from hagisilta_i18n import tr

from datetime import datetime
from typing import Any


DEFAULT_GT_INTERVAL_SECONDS = 0.1


def sample_gt_frames(
    start_frame: int,
    end_frame: int,
    fps: float,
    interval_seconds: float = DEFAULT_GT_INTERVAL_SECONDS,
) -> list[int]:
    """Sample an inclusive episode on a time grid rather than a frame grid."""
    start, end = sorted((max(0, int(start_frame)), max(0, int(end_frame))))
    if start == end:
        return [start]
    fps = float(fps)
    interval_seconds = float(interval_seconds)
    if fps <= 0 or interval_seconds <= 0:
        raise ValueError(tr('FPS와 GT 간격은 0보다 커야 합니다.'))
    duration_seconds = (end - start) / fps
    steps = int(duration_seconds / interval_seconds)
    frames = {start, end}
    for index in range(1, steps + 1):
        frame = round(start + index * interval_seconds * fps)
        if start < frame < end:
            frames.add(int(frame))
    return sorted(frames)


def make_episode(
    start_frame: int,
    end_frame: int,
    fps: float,
    serial: int,
    interval_seconds: float = DEFAULT_GT_INTERVAL_SECONDS,
) -> dict[str, Any]:
    start, end = sorted((max(0, int(start_frame)), max(0, int(end_frame))))
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "episode_id": f"episode-{start:09d}-{end:09d}-{max(1, int(serial)):03d}",
        "start_frame": start,
        "end_frame": end,
        "start_time_s": start / float(fps),
        "end_time_s": end / float(fps),
        "gt_interval_seconds": float(interval_seconds),
        "gt_frames": sample_gt_frames(start, end, fps, interval_seconds),
        "created_at": now,
        "updated_at": now,
    }


def normalize_episode(
    payload: dict[str, Any], fps: float, frame_count: int
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    try:
        start = max(0, min(int(payload.get("start_frame", 0)), max(0, frame_count - 1)))
        end = max(0, min(int(payload.get("end_frame", start)), max(0, frame_count - 1)))
        start, end = sorted((start, end))
        interval = float(payload.get("gt_interval_seconds", DEFAULT_GT_INTERVAL_SECONDS))
    except (TypeError, ValueError):
        return None
    if fps <= 0 or interval <= 0:
        return None
    candidates = payload.get("gt_frames")
    if not isinstance(candidates, list):
        candidates = sample_gt_frames(start, end, fps, interval)
    candidates = sorted({
        int(frame) for frame in candidates
        if isinstance(frame, (int, float)) and start <= int(frame) <= end
    })
    candidates = sorted(set(candidates) | {start, end})
    normalized = dict(payload)
    normalized.update(
        {
            "episode_id": str(payload.get("episode_id") or f"episode-{start:09d}-{end:09d}"),
            "start_frame": start,
            "end_frame": end,
            "start_time_s": start / fps,
            "end_time_s": end / fps,
            "gt_interval_seconds": interval,
            "gt_frames": candidates,
        }
    )
    return normalized


def episode_frame_groups(episodes: list[dict[str, Any]]) -> dict[int, str]:
    result: dict[int, str] = {}
    for episode in episodes:
        episode_id = str(episode.get("episode_id") or "")
        if not episode_id:
            continue
        for frame in episode.get("gt_frames", []) or []:
            result[int(frame)] = episode_id
    return result


def episodes_overlap(episodes: list[dict[str, Any]], start_frame: int, end_frame: int) -> bool:
    start, end = sorted((int(start_frame), int(end_frame)))
    return any(
        start <= int(episode.get("end_frame", -1))
        and end >= int(episode.get("start_frame", 0))
        for episode in episodes
    )
