# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from hagisilta_i18n import tr
from hagisilta_render_text import configure_plot_fonts

import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hagisilta_nort_analysis import arena_id, arena_video, coordinate, load_pose, seconds_heatmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze arbitrary HAGISILTA polygon zones")
    parser.add_argument("tracked_root", type=Path)
    parser.add_argument("profile", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pcutoff", type=float, default=0.6)
    parser.add_argument("--bodypart", default="body_center")
    parser.add_argument("--minimum-dwell", type=float, default=0.2)
    return parser.parse_args()


def segments(values: np.ndarray) -> list[tuple[int, int, str]]:
    if not len(values):
        return []
    result: list[tuple[int, int, str]] = []
    start = 0
    current = str(values[0])
    for index in range(1, len(values)):
        value = str(values[index])
        if value != current:
            result.append((start, index, current))
            start, current = index, value
    result.append((start, len(values), current))
    return result


def alternation_score(sequence: list[str]) -> tuple[int, int, float]:
    opportunities = max(len(sequence) - 2, 0)
    successful = sum(len(set(sequence[index:index + 3])) == 3 for index in range(opportunities))
    percent = successful / opportunities * 100.0 if opportunities else math.nan
    return successful, opportunities, percent


def video_frame(video: Path | None, frame_index: int, width: int, height: int) -> np.ndarray:
    if video:
        capture = cv2.VideoCapture(str(video))
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(frame_index, 0))
        ok, frame = capture.read(); capture.release()
        if ok:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return np.full((max(height, 1), max(width, 1), 3), 35, dtype=np.uint8)


def analyze_region(
    arena_dir: Path, region: dict, output: Path,
    pcutoff: float, bodypart: str, minimum_dwell: float,
) -> dict:
    pose, pose_path = load_pose(arena_dir)
    metadata_path = arena_dir / "roi-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    fps = float(metadata.get("fps", 0.0) or 0.0)
    if fps <= 0:
        video = arena_video(arena_dir)
        capture = cv2.VideoCapture(str(video)) if video else None
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) if capture else 0.0
        if capture: capture.release()
    if fps <= 0:
        raise ValueError(tr('FPS를 확인할 수 없습니다: {v0}', v0=arena_dir))
    x = coordinate(pose, bodypart, "x")
    y = coordinate(pose, bodypart, "y")
    likelihood = coordinate(pose, bodypart, "likelihood")
    offset = np.asarray(metadata.get("source_offset_px", [0, 0]), dtype=float)
    source_x, source_y = x + offset[0], y + offset[1]
    valid = np.isfinite(x) & np.isfinite(y) & (likelihood >= pcutoff)
    zone_ids = np.full(len(pose), "", dtype=object)
    zones = region.get("zones", [])
    for zone in zones:
        polygon = np.asarray(zone.get("polygon_source_px") or [], dtype=np.float32)
        if len(polygon) < 3:
            continue
        for frame_index in np.flatnonzero(valid & (zone_ids == "")):
            if cv2.pointPolygonTest(polygon, (float(source_x[frame_index]), float(source_y[frame_index])), False) >= 0:
                zone_ids[frame_index] = str(zone["id"])

    minimum_frames = max(1, round(minimum_dwell * fps))
    accepted = []
    entry_counts = {str(zone["id"]): 0 for zone in zones}
    first_entry = {str(zone["id"]): math.nan for zone in zones}
    previous_accepted = None
    for start, end, zone_id in segments(zone_ids):
        if not zone_id or end - start < minimum_frames:
            continue
        if zone_id != previous_accepted:
            entry_counts[zone_id] += 1
            if math.isnan(first_entry[zone_id]):
                first_entry[zone_id] = start / fps
            accepted.append({"frame": start, "time_seconds": start / fps, "zone_id": zone_id})
            previous_accepted = zone_id

    role_by_id = {str(zone["id"]): str(zone.get("role", "choice")) for zone in zones}
    choice_sequence = [entry["zone_id"] for entry in accepted if role_by_id.get(entry["zone_id"]) == "choice"]
    success, opportunities, alternation = alternation_score(choice_sequence)
    distance = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    distance_valid = valid[1:] & valid[:-1]
    distance_px = float(np.nansum(distance[distance_valid]))

    output.mkdir(parents=True, exist_ok=True)
    region_id = int(region["id"])
    per_frame = pd.DataFrame({
        "frame": np.arange(len(pose)), "time_seconds": np.arange(len(pose)) / fps,
        f"{bodypart}_x": x, f"{bodypart}_y": y, "source_x": source_x, "source_y": source_y,
        "likelihood": likelihood, "valid_pose": valid, "zone_id": zone_ids,
    })
    per_frame_path = output / f"region_{region_id:02d}_zone_frames.csv"
    per_frame.to_csv(per_frame_path, index=False, encoding="utf-8-sig")
    transitions_path = output / f"region_{region_id:02d}_transitions.csv"
    pd.DataFrame(accepted).to_csv(transitions_path, index=False, encoding="utf-8-sig")

    rows = []
    for zone in zones:
        zone_id = str(zone["id"])
        rows.append({
            "region_id": region_id, "region_name": region.get("name", f"Region {region_id}"),
            "zone_id": zone_id, "zone_name": zone.get("name", zone_id), "zone_role": zone.get("role", "choice"),
            "occupancy_seconds": float(np.count_nonzero(zone_ids == zone_id) / fps),
            "entries": int(entry_counts.get(zone_id, 0)), "first_entry_seconds": first_entry.get(zone_id, math.nan),
        })

    video = arena_video(arena_dir)
    if video:
        capture = cv2.VideoCapture(str(video))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
    else:
        width = int(np.nanmax(x) + 1); height = int(np.nanmax(y) + 1)
    background = video_frame(video, 0, width, height)
    configure_plot_fonts()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=170)
    axes[0].imshow(background)
    line_x, line_y = x.copy(), y.copy(); line_x[~valid] = np.nan; line_y[~valid] = np.nan
    axes[0].plot(line_x, line_y, color="#00e5e5", linewidth=0.55, alpha=0.65)
    heat = seconds_heatmap(x, y, valid, width, height, fps)
    axes[1].imshow(background)
    axes[1].imshow(np.ma.masked_less_equal(heat, 0), extent=(0, width, height, 0), cmap="turbo", alpha=0.78)
    colors = ("#ffd166", "#ff6b8a", "#66a8ff", "#70e2ae", "#bb8cff")
    for zone_index, zone in enumerate(zones):
        polygon = np.asarray(zone.get("polygon_source_px") or [], dtype=float) - offset
        if len(polygon) < 3: continue
        closed = np.vstack([polygon, polygon[0]])
        for ax in axes:
            ax.plot(closed[:, 0], closed[:, 1], color=colors[zone_index % len(colors)], linewidth=1.5)
            center = polygon.mean(axis=0)
            ax.text(center[0], center[1], str(zone.get("name", zone["id"])), color="white", fontsize=8,
                    ha="center", va="center", fontweight="bold")
    for ax, title in zip(axes, (f"{bodypart} trajectory", "Occupancy heatmap")):
        ax.set_title(title); ax.set_xlim(0, width); ax.set_ylim(height, 0); ax.axis("off")
    fig.suptitle(f"{arena_dir.parent.name} · {region.get('name', f'Region {region_id}')}")
    fig.tight_layout()
    visual_path = output / f"region_{region_id:02d}_zone_visual.png"
    fig.savefig(visual_path, bbox_inches="tight"); plt.close(fig)
    return {
        "region_id": region_id, "region_name": region.get("name", f"Region {region_id}"),
        "pose_file": str(pose_path.resolve()), "fps": fps, "frames": len(pose),
        "valid_pose_percent": float(valid.mean() * 100.0), "distance_px": distance_px,
        "choice_sequence": choice_sequence, "alternation_successes": success,
        "alternation_opportunities": opportunities, "alternation_percent": alternation,
        "zone_rows": rows, "per_frame_file": str(per_frame_path.resolve()),
        "transitions_file": str(transitions_path.resolve()), "visual_image": str(visual_path.resolve()),
    }


def main() -> int:
    args = parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    if profile.get("profile_type") != "hagisilta_zone_template":
        raise ValueError(tr('HAGISILTA 구역 템플릿이 아닙니다.'))
    regions = {int(region["id"]): region for region in profile.get("arenas", [])}
    results = []
    arena_dirs = sorted(path for path in args.tracked_root.glob("arena_*") if path.is_dir())
    for index, arena_dir in enumerate(arena_dirs, start=1):
        region_id = arena_id(arena_dir)
        if region_id not in regions:
            continue
        print(f"ZONE_REGION={index}/{len(arena_dirs)}:{region_id}", flush=True)
        results.append(analyze_region(
            arena_dir, regions[region_id], args.output,
            min(max(args.pcutoff, 0.0), 1.0), args.bodypart, max(args.minimum_dwell, 0.0),
        ))
    if not results:
        raise ValueError(tr('템플릿과 일치하는 추적 영역이 없습니다.'))
    zone_rows = [row for result in results for row in result.pop("zone_rows")]
    zone_summary = args.output / "zone-summary.csv"
    pd.DataFrame(zone_rows).to_csv(zone_summary, index=False, encoding="utf-8-sig")
    experiment_summary = args.output / "experiment-summary.json"
    experiment_summary.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    print(f"ZONE_SUMMARY={zone_summary.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
