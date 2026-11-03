# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from hagisilta_i18n import tr

import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, Ellipse, Polygon

from hagisilta_nort_analysis import (
    arena_video, available_bodyparts, coordinate, load_pose, save_figure_resilient, seconds_heatmap,
)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    starts = np.flatnonzero(values & ~np.r_[False, values[:-1]])
    ends = np.flatnonzero(values & ~np.r_[values[1:], False]) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def connect_short_gaps(mask: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Connect brief exits between two platform-presence runs."""
    values = np.asarray(mask, dtype=bool).copy()
    max_gap_frames = max(int(max_gap_frames), 0)
    if max_gap_frames <= 0 or values.size == 0:
        return values
    for start, end in runs(~values):
        if start > 0 and end < len(values) and end - start <= max_gap_frames:
            values[start:end] = True
    return values


def interpolate_short(values: np.ndarray, valid: np.ndarray, limit: int) -> np.ndarray:
    series = pd.Series(np.where(valid, values, np.nan))
    return series.interpolate(limit=max(limit, 0), limit_area="inside").to_numpy(dtype=float)


def selected_bodypart_coordinates(
    pose: pd.DataFrame, pose_path: Path, settings: dict,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Resolve every selected point before computing a strict multi-point centroid."""
    available = available_bodyparts(pose)
    context = (
        tr('좌표 파일: {v0}\n사용 가능한 신체점: {v1}', v0=pose_path, v1=', '.join(available) or tr('없음'))
    )
    # Old analysis plans did not store a selection and used body_center. Keep
    # only that missing-key compatibility; an explicit empty selection is not
    # permission to choose a different bodypart on the user's behalf.
    requested = settings.get("bodyparts", ["body_center"])
    if not isinstance(requested, (list, tuple)):
        raise ValueError(tr('Water-maze 판정 신체점을 목록으로 지정하세요.\n{v0}', v0=context))
    requested = [str(value) for value in requested if value is not None and str(value).strip()]
    if not requested:
        raise ValueError(tr('Water-maze 판정에 사용할 신체점을 한 개 이상 선택하세요.\n{v0}', v0=context))

    selected: list[str] = []
    points: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for requested_name in requested:
        if requested_name in available:
            canonical_name = requested_name
        else:
            matching = [name for name in available if name.lower() == requested_name.lower()]
            if len(matching) != 1:
                reason = tr('이름이 여러 신체점과 일치합니다') if matching else tr('찾을 수 없습니다')
                raise ValueError(
                    tr("Water-maze 판정 신체점 '{v0}'을(를) {v1}.\n{v2}", v0=requested_name, v1=reason, v2=context)
                )
            canonical_name = matching[0]
        if canonical_name in selected:
            continue
        values: list[np.ndarray] = []
        for coord_name in ("x", "y", "likelihood"):
            try:
                value = coordinate(pose, canonical_name, coord_name)
                if value.ndim != 1 or len(value) != len(pose):
                    raise ValueError(tr('좌표 길이가 프레임 수와 일치하지 않습니다.'))
            except (KeyError, ValueError, TypeError, IndexError) as exc:
                raise ValueError(
                    tr("Water-maze 판정 신체점 '{v0}'의 '{v1}' 좌표가 없거나 올바르지 않습니다.\n{v2}", v0=canonical_name, v1=coord_name, v2=context)
                ) from exc
            values.append(value)
        # Append only a complete x/y/likelihood triplet. A missing column must
        # never silently drop a checked point or misalign the centroid arrays.
        points.append((values[0], values[1], values[2]))
        selected.append(canonical_name)
    return (
        selected,
        np.vstack([point[0] for point in points]),
        np.vstack([point[1] for point in points]),
        np.vstack([point[2] for point in points]),
    )


def write_annotated_video(
    video: Path,
    output_file: Path,
    fps: float,
    frame_count: int,
    x: np.ndarray,
    y: np.ndarray,
    usable: np.ndarray,
    quadrant: np.ndarray,
    in_platform: np.ndarray,
    pool_center: tuple[float, float],
    pool_radii: tuple[float, float],
    platform: dict,
    north_angle: float,
    peripheral_ring: float,
    start_frame: int = 0,
) -> None:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(start_frame), 0))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    writer = cv2.VideoWriter(
        str(output_file),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(float(fps), 1.0),
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise OSError(f"Could not create annotated video: {output_file}")

    cx, cy = pool_center
    radius_x, radius_y = pool_radii
    center = (int(round(cx)), int(round(cy)))
    axes = (max(1, int(round(radius_x))), max(1, int(round(radius_y))))
    ring_axes = (
        max(1, int(round(radius_x * (1.0 - peripheral_ring)))),
        max(1, int(round(radius_y * (1.0 - peripheral_ring)))),
    )
    platform_type = str(platform.get("shape_type", "circle"))
    platform_points = np.asarray(platform.get("points_px") or [], dtype=np.int32)
    platform_center = tuple(int(round(value)) for value in platform["center_px"])
    platform_radius = max(1, int(round(float(platform["radius_px"]))))
    entries = np.asarray(in_platform[:frame_count], dtype=bool) & ~np.r_[
        False, np.asarray(in_platform[:frame_count], dtype=bool)[:-1]
    ]
    cumulative_entries = np.cumsum(entries)
    trail_frames = max(1, int(round(fps * 2.0)))

    try:
        for index in range(frame_count):
            ok, frame = cap.read()
            if not ok:
                break
            overlay = frame.copy()
            cv2.ellipse(overlay, center, axes, 0, 0, 360, (84, 220, 169), 2, cv2.LINE_AA)
            cv2.ellipse(overlay, center, ring_axes, 0, 0, 360, (0, 211, 255), 1, cv2.LINE_AA)

            for boundary in (0.0, 90.0, 180.0, 270.0):
                angle = math.radians(north_angle + boundary)
                endpoint = (
                    int(round(cx + radius_x * math.sin(angle))),
                    int(round(cy - radius_y * math.cos(angle))),
                )
                cv2.line(overlay, center, endpoint, (165, 180, 195), 1, cv2.LINE_AA)
            for name, offset in (("NE", 45.0), ("SE", 135.0), ("SW", 225.0), ("NW", 315.0)):
                angle = math.radians(north_angle + offset)
                label_point = (
                    int(round(cx + radius_x * 0.56 * math.sin(angle))),
                    int(round(cy - radius_y * 0.56 * math.cos(angle))),
                )
                cv2.putText(
                    overlay, name, label_point, cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (245, 245, 245), 1, cv2.LINE_AA,
                )

            if platform_type != "circle" and platform_points.ndim == 2 and len(platform_points) >= 3:
                cv2.polylines(overlay, [platform_points], True, (102, 102, 255), 2, cv2.LINE_AA)
            else:
                cv2.circle(overlay, platform_center, platform_radius, (102, 102, 255), 2, cv2.LINE_AA)

            start = max(0, index - trail_frames)
            trail_valid = np.asarray(usable[start : index + 1], dtype=bool)
            trail_x = np.asarray(x[start : index + 1])
            trail_y = np.asarray(y[start : index + 1])
            points = np.column_stack((trail_x[trail_valid], trail_y[trail_valid]))
            if len(points) >= 2:
                cv2.polylines(overlay, [np.round(points).astype(np.int32)], False, (208, 201, 18), 2, cv2.LINE_AA)
            if index < len(usable) and usable[index]:
                current = (int(round(x[index])), int(round(y[index])))
                cv2.circle(overlay, current, 6, (58, 230, 255), -1, cv2.LINE_AA)
                cv2.circle(overlay, current, 8, (12, 20, 28), 2, cv2.LINE_AA)

            panel = overlay.copy()
            cv2.rectangle(panel, (8, 8), (min(width - 8, 460), 88), (7, 13, 22), -1)
            cv2.addWeighted(panel, 0.78, overlay, 0.22, 0, overlay)
            current_zone = str(quadrant[index]) if index < len(quadrant) and quadrant[index] else "N/A"
            cv2.putText(
                overlay, f"ZONE {current_zone}", (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (132, 236, 178), 2, cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                f"time {(start_frame + index) / max(fps, 1e-9):06.1f}s"
                f"   analysis {index / max(fps, 1e-9):06.1f}s"
                f"   platform crossings {int(cumulative_entries[index])}",
                (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (238, 238, 238), 1, cv2.LINE_AA,
            )
            writer.write(overlay)
    finally:
        cap.release()
        writer.release()


def analyze(item: dict, settings: dict, output: Path) -> dict:
    arena_dir = Path(item["arena_dir"])
    profile = json.loads(Path(item["profile"]).read_text(encoding="utf-8"))
    pose, pose_path = load_pose(arena_dir)
    used_bodyparts, all_x, all_y, all_likelihood = selected_bodypart_coordinates(
        pose, pose_path, settings,
    )
    video = Path(item.get("video", "")) if item.get("video") else arena_video(arena_dir)
    selected_video = pose.attrs.get("hagisilta_analysis_pose", {}).get("source_video")
    if selected_video and video and Path(selected_video).resolve() != video.resolve():
        raise ValueError(tr('Water maze Trial 영상과 선택된 분석 좌표의 영상이 다릅니다.'))
    if not video or not video.is_file():
        raise FileNotFoundError(tr('Arena 영상이 없습니다: {v0}', v0=arena_dir))
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or profile["frame_size"][0])
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or profile["frame_size"][1])
    video_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or len(pose)))
    count = min(len(pose), video_frames)
    requested_start = max(float(item.get("analysis_start_seconds", 0.0) or 0.0), 0.0)
    requested_end = float(item.get("analysis_end_seconds", count / fps) or count / fps)
    start_frame = min(max(int(math.floor(requested_start * fps)), 0), max(count - 1, 0))
    end_frame = min(max(int(math.ceil(requested_end * fps)), start_frame + 1), count)
    max_duration = float(settings.get("max_duration_seconds", 0.0) or 0.0)
    if max_duration > 0:
        end_frame = min(end_frame, start_frame + max(1, round(max_duration * fps)))
    if end_frame <= start_frame:
        raise ValueError(
            tr('분석 종료는 시작보다 뒤여야 합니다: {v0:.3f}s → {v1:.3f}s', v0=requested_start, v1=requested_end)
        )
    max_frames = end_frame - start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, frame0 = cap.read()
    cap.release()
    if not ok:
        frame0 = np.full((height, width, 3), 35, dtype=np.uint8)
    background = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    pcutoff = float(settings.get("pcutoff", .6))
    x_stack = all_x[:, start_frame:end_frame]
    y_stack = all_y[:, start_frame:end_frame]
    l_stack = all_likelihood[:, start_frame:end_frame]
    point_valid = (l_stack >= pcutoff) & np.isfinite(x_stack) & np.isfinite(y_stack)
    # Multi-point decisions follow the same strict rule exposed by the Y-maze UI:
    # every checked point must be usable.  The representative swimming position
    # is the centroid of those points, which is less sensitive to head/tail sway.
    valid = point_valid.all(axis=0)
    x = np.where(valid, np.mean(x_stack, axis=0), np.nan)
    y = np.where(valid, np.mean(y_stack, axis=0), np.nan)
    bridge = round(float(settings.get("bridge_seconds", .2)) * fps)
    x = interpolate_short(x, valid, bridge); y = interpolate_short(y, valid, bridge)
    valid = np.isfinite(x) & np.isfinite(y)

    pool = profile["pool"]; platform = profile["platform"]
    cx, cy = map(float, pool["center_px"])
    radius = float(pool["radius_px"])
    radius_x = float(pool.get("radius_x_px", radius))
    radius_y = float(pool.get("radius_y_px", radius))
    px, py = map(float, platform["center_px"]); platform_radius = float(platform["radius_px"])
    diameter_cm = float(pool.get("diameter_cm", 120.0))
    cm_per_px_x = diameter_cm / max(2 * radius_x, 1)
    cm_per_px_y = diameter_cm / max(2 * radius_y, 1)
    normalized_x = (x - cx) / max(radius_x, 1e-9)
    normalized_y = (y - cy) / max(radius_y, 1e-9)
    normalized_radius = np.hypot(normalized_x, normalized_y)
    inside_pool = valid & (normalized_radius <= 1.0)
    theta = (
        np.degrees(np.arctan2(normalized_x, -normalized_y))
        - float(profile.get("north_angle_degrees", 0))
    ) % 360
    quadrant = np.full(max_frames, "", dtype=object)
    quadrant[inside_pool & (theta < 90)] = "NE"
    quadrant[inside_pool & (theta >= 90) & (theta < 180)] = "SE"
    quadrant[inside_pool & (theta >= 180) & (theta < 270)] = "SW"
    quadrant[inside_pool & (theta >= 270)] = "NW"
    platform_type = str(platform.get("shape_type", "circle"))
    platform_points = np.asarray(platform.get("points_px") or [], dtype=float)
    if platform_type != "circle" and platform_points.ndim == 2 and len(platform_points) >= 3:
        in_platform = inside_pool & MplPath(platform_points).contains_points(
            np.column_stack((x, y)), radius=1e-9
        )
    else:
        platform_type = "circle"
        in_platform = inside_pool & (np.hypot(x - px, y - py) <= platform_radius)
    dwell_frames = max(1, round(float(settings.get("platform_dwell_seconds", 2)) * fps))
    platform_runs = runs(in_platform)
    confirmed = [(start, end) for start, end in platform_runs if end - start >= dwell_frames]
    found = bool(confirmed)
    escape_frame = confirmed[0][0] if found else max_frames
    crossing_gap_frames = max(0, round(float(settings.get("platform_crossing_gap_seconds", .2)) * fps))
    crossing_dwell_frames = max(1, round(float(settings.get("platform_crossing_dwell_seconds", .3)) * fps))
    crossing_presence = connect_short_gaps(in_platform, crossing_gap_frames)
    crossing_runs = [
        (start, end)
        for start, end in runs(crossing_presence)
        if end - start >= crossing_dwell_frames
    ]
    confirmed_crossing_mask = np.zeros_like(in_platform, dtype=bool)
    for start, end in crossing_runs:
        confirmed_crossing_mask[start:end] = True
    trial_type = str(item.get("trial_type", "Hidden"))
    # The interval explicitly selected by the user is authoritative. Platform
    # acquisition remains available as found_platform/escape_latency_seconds,
    # but must not silently shorten trajectory, heatmap, CSV, or QC video output.
    analysis_end = max_frames
    usable = inside_pool[:analysis_end]
    dx = np.diff(x[:analysis_end]); dy = np.diff(y[:analysis_end])
    step_valid = usable[1:] & usable[:-1]
    step_cm = np.where(
        step_valid,
        np.hypot(dx * cm_per_px_x, dy * cm_per_px_y),
        0,
    )
    path_cm = float(np.nansum(step_cm))
    duration = analysis_end / fps
    speed = path_cm / max(float(step_valid.sum()) / fps, 1e-9)
    first_valid = int(np.flatnonzero(usable)[0]) if usable.any() else 0
    straight_cm = (
        float(np.hypot(
            (x[first_valid] - px) * cm_per_px_x,
            (y[first_valid] - py) * cm_per_px_y,
        ))
        if usable.any() else math.nan
    )
    efficiency = straight_cm / path_cm if path_cm > 0 and np.isfinite(straight_cm) else math.nan
    target = str(item.get("target_quadrant") or platform.get("target_quadrant", "NE"))
    target_mask = usable & (quadrant[:analysis_end] == target)
    if "wall_zone_width_cm" in profile:
        wall_zone_width_cm = float(profile.get("wall_zone_width_cm", 10.0))
        ring = wall_zone_width_cm / max(diameter_cm / 2.0, 1e-9)
    else:
        ring = float(profile.get("peripheral_ring_percent", .15))
        wall_zone_width_cm = ring * diameter_cm / 2.0
    ring = max(0.0, min(ring, 0.95))
    peripheral = usable & (normalized_radius[:analysis_end] >= 1.0 - ring)
    proximity = np.hypot(
        (x[:analysis_end] - px) * cm_per_px_x,
        (y[:analysis_end] - py) * cm_per_px_y,
    )
    frame_rows = pd.DataFrame({
        "frame": start_frame + np.arange(analysis_end),
        "time_seconds": (start_frame + np.arange(analysis_end)) / fps,
        "analysis_time_seconds": np.arange(analysis_end) / fps,
        "x": x[:analysis_end], "y": y[:analysis_end], "valid": usable,
        "bodyparts_used": ",".join(used_bodyparts),
        "quadrant": quadrant[:analysis_end], "in_platform": in_platform[:analysis_end],
        "confirmed_platform_crossing": confirmed_crossing_mask[:analysis_end],
        "in_peripheral_ring": peripheral,
    })
    for part_index, bodypart in enumerate(used_bodyparts):
        frame_rows[f"{bodypart}_x"] = x_stack[part_index, :analysis_end]
        frame_rows[f"{bodypart}_y"] = y_stack[part_index, :analysis_end]
        frame_rows[f"{bodypart}_likelihood"] = l_stack[part_index, :analysis_end]
    segment_label = str(item.get("segment_id") or f"trial_{item.get('trial', 1)}")
    safe_segment = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", segment_label).strip("._-") or "trial"
    stem = f"{item['video_bundle']}_arena_{int(item['arena_id']):02d}_{safe_segment}"
    frame_file = output / f"{stem}_watermaze_frames.csv"; frame_rows.to_csv(frame_file, index=False, encoding="utf-8-sig")
    crossing_file = output / f"{stem}_platform_crossings.csv"
    crossing_columns = [
        "start_seconds", "end_seconds", "analysis_start_seconds",
        "analysis_end_seconds", "duration_seconds",
    ]
    pd.DataFrame([
        {
            "start_seconds": (start_frame + a) / fps,
            "end_seconds": (start_frame + b) / fps,
            "analysis_start_seconds": a / fps,
            "analysis_end_seconds": b / fps,
            "duration_seconds": (b - a) / fps,
        }
        for a, b in crossing_runs
    ], columns=crossing_columns).to_csv(crossing_file, index=False, encoding="utf-8-sig")

    figure, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=150)
    for ax in axes:
        ax.imshow(background); ax.set_xlim(0, width); ax.set_ylim(height, 0); ax.axis("off")
        ax.add_patch(Ellipse(
            (cx, cy),
            2 * radius_x,
            2 * radius_y,
            fill=False,
            color="#55d6ad",
            linewidth=1.8,
        ))
        if platform_type == "circle":
            ax.add_patch(Circle((px, py), platform_radius, fill=False, color="#ff6685", linewidth=1.5))
        else:
            ax.add_patch(Polygon(platform_points, closed=True, fill=False, color="#ff6685", linewidth=1.5))
        for boundary in (0.0, 90.0, 180.0, 270.0):
            angle = math.radians(float(profile.get("north_angle_degrees", 0.0)) + boundary)
            endpoint = (
                cx + radius_x * math.sin(angle),
                cy - radius_y * math.cos(angle),
            )
            ax.plot([cx, endpoint[0]], [cy, endpoint[1]], color="#d6e0ea", linewidth=.65, alpha=.8)
        for name, offset in (("NE", 45.0), ("SE", 135.0), ("SW", 225.0), ("NW", 315.0)):
            angle = math.radians(float(profile.get("north_angle_degrees", 0.0)) + offset)
            ax.text(
                cx + radius_x * .58 * math.sin(angle),
                cy - radius_y * .58 * math.cos(angle),
                name, color="white", fontsize=8, ha="center", va="center",
            )
    axes[0].plot(x[:analysis_end][usable], y[:analysis_end][usable], color="#12c9d0", linewidth=.8, alpha=.85)
    axes[0].set_title(
        f"{item['video_bundle']} · Arena {item['arena_id']} · "
        f"{item.get('subject_id', safe_segment)} · {trial_type}\nTrajectory"
    )
    heat = seconds_heatmap(x[:analysis_end], y[:analysis_end], usable, width, height, fps)
    heat_up = cv2.resize(heat, (width, height), interpolation=cv2.INTER_CUBIC)
    # Cubic interpolation can create tiny negative values around empty bins.
    # Clamp them so every unvisited point inside the pool is rendered with the
    # colormap's true zero color instead of becoming a transparent hole.
    heat_up = np.maximum(heat_up, 0.0)
    yy, xx = np.mgrid[:height, :width]
    pool_mask = (
        ((xx - cx) / max(radius_x, 1e-9)) ** 2
        + ((yy - cy) / max(radius_y, 1e-9)) ** 2
    ) <= 1.0
    masked = np.ma.masked_where(~pool_mask, heat_up)
    heat_max = max(float(np.max(heat_up[pool_mask])), 1.0 / max(fps, 1e-9))
    heat_image = axes[1].imshow(
        masked,
        cmap="turbo",
        alpha=.78,
        vmin=0.0,
        vmax=heat_max,
    )
    colorbar = figure.colorbar(heat_image, ax=axes[1], fraction=.046, pad=.03)
    colorbar.set_label("Dwell time (s / spatial bin)")
    axes[1].set_title("Occupancy heatmap")
    visual_file = output / f"{stem}_watermaze_visual.png"
    save_figure_resilient(figure, visual_file, bbox_inches="tight", facecolor="white"); plt.close(figure)
    annotated_video_file = output / f"{stem}_watermaze_annotated.mp4"
    write_annotated_video(
        video=video,
        output_file=annotated_video_file,
        fps=fps,
        frame_count=analysis_end,
        x=x,
        y=y,
        usable=usable,
        quadrant=quadrant[:analysis_end],
        in_platform=confirmed_crossing_mask[:analysis_end],
        pool_center=(cx, cy),
        pool_radii=(radius_x, radius_y),
        platform=platform,
        north_angle=float(profile.get("north_angle_degrees", 0.0)) % 360.0,
        peripheral_ring=ring,
        start_frame=start_frame,
    )

    row = {
        **{
            key: item.get(key)
            for key in (
                "video_bundle", "arena_id", "segment_id", "subject_id", "group",
                "day", "trial", "trial_type", "start_direction",
            )
        },
        "target_quadrant": target, "found_platform": found,
        "pool_shape": str(pool.get("shape_type", "circle")),
        "north_angle_degrees": float(profile.get("north_angle_degrees", 0.0)) % 360.0,
        "wall_zone_width_cm": wall_zone_width_cm,
        "wall_zone_radius_percent": ring * 100.0,
        "analysis_start_seconds": start_frame / fps,
        "analysis_end_seconds": (start_frame + analysis_end) / fps,
        "selected_end_seconds": end_frame / fps,
        "escape_latency_seconds": escape_frame / fps if found else math.nan,
        "path_length_cm": path_cm, "mean_speed_cm_s": speed, "path_efficiency": efficiency,
        "mean_proximity_cm": float(np.nanmean(proximity[usable])) if usable.any() else math.nan,
        "target_quadrant_seconds": float(target_mask.sum() / fps),
        "target_quadrant_percent": float(target_mask.sum() / max(usable.sum(), 1) * 100),
        "platform_crossings": len(crossing_runs),
        "first_platform_crossing_seconds": crossing_runs[0][0] / fps if crossing_runs else math.nan,
        "platform_dwell_seconds": float(settings.get("platform_dwell_seconds", 2)),
        "platform_crossing_dwell_seconds": float(settings.get("platform_crossing_dwell_seconds", .3)),
        "platform_crossing_gap_seconds": float(settings.get("platform_crossing_gap_seconds", .2)),
        "bridge_seconds": float(settings.get("bridge_seconds", .2)),
        "pcutoff": pcutoff,
        "thigmotaxis_seconds": float(peripheral.sum() / fps),
        "thigmotaxis_percent": float(peripheral.sum() / max(usable.sum(), 1) * 100),
        "valid_pose_percent": float(valid[:analysis_end].mean() * 100),
        "duration_seconds": duration, "fps": fps,
        "bodyparts_used": ",".join(used_bodyparts),
        "source_video_file": str(video.resolve()),
        "profile_file": str(Path(item["profile"]).resolve()),
        "visual_file": str(visual_file.resolve()),
        "annotated_video_file": str(annotated_video_file.resolve()),
        "per_frame_file": str(frame_file.resolve()),
        "platform_crossings_file": str(crossing_file.resolve()), "pose_file": str(pose_path.resolve()),
        "pose_postprocessing": json.dumps(pose.attrs.get("hagisilta_analysis_pose", {}), ensure_ascii=False),
    }
    for name in ("NE", "SE", "SW", "NW"):
        mask = usable & (quadrant[:analysis_end] == name)
        row[f"quadrant_{name.lower()}_seconds"] = float(mask.sum() / fps)
        row[f"quadrant_{name.lower()}_percent"] = float(mask.sum() / max(usable.sum(), 1) * 100)
    return row


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: hagisilta_watermaze_analysis.py PLAN_JSON OUTPUT_ROOT", flush=True); return 2
    plan_path, output_root = Path(sys.argv[1]), Path(sys.argv[2])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    stamp = datetime.now().strftime("run_%Y%m%d-%H%M%S")
    run_dir = output_root / "runs" / stamp
    counter = 2
    while run_dir.exists():
        run_dir = output_root / "runs" / f"{stamp}_{counter}"; counter += 1
    run_dir.mkdir(parents=True)
    rows = []
    items = plan.get("items", [])
    for index, item in enumerate(items, 1):
        print(
            f"WATERMAZE_TRIAL={index}/{len(items)}:{item.get('video_bundle')}:"
            f"{item.get('arena_id')}:{item.get('subject_id', item.get('segment_id', ''))}",
            flush=True,
        )
        rows.append(analyze(item, plan, run_dir))
    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "project-watermaze-summary.csv", index=False, encoding="utf-8-sig")
    (run_dir / "project-watermaze-summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=lambda value: None if pd.isna(value) else value),
        encoding="utf-8",
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "latest-run.json").write_text(json.dumps({"run_dir": str(run_dir.resolve())}, indent=2), encoding="utf-8")
    print(f"WATERMAZE_RUN={run_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
