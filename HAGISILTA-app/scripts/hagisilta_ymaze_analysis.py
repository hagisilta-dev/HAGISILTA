# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from hagisilta_i18n import tr

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hagisilta_nort_analysis import (
    RESULT_RENDER_DPI, RESULT_RENDER_WIDTH_IN,
    arena_id, arena_video, coordinate, load_pose, save_figure_resilient,
    seconds_heatmap, write_csv_resilient, write_text_resilient,
)


ARM_IDS = ("arm_a", "arm_b", "arm_c")
ARM_LABELS = {"arm_a": "A", "arm_b": "B", "arm_c": "C"}


def canonical_arena_name(value: object, arena_id: int) -> str:
    """Return the experiment-facing Y-maze Arena label.

    Older Y-maze profiles could inherit ``ROI 1``/``ROI 2`` from the
    tracking crop profile.  ROI is the tracking-stage term; once the crop is
    used as a Y-maze experiment region it is consistently called Arena.
    """
    text = str(value or "").strip()
    match = re.fullmatch(r"(?:arena|roi|y-maze)\s*0*(\d+)", text, re.IGNORECASE)
    number = int(match.group(1)) if match else int(arena_id)
    return f"Arena {number:02d}"


def parse_single_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HAGISILTA Y-maze spontaneous alternation pilot")
    parser.add_argument("tracked_root", type=Path)
    parser.add_argument("profile", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pcutoff", type=float, default=0.6)
    parser.add_argument("--bodypart", default="body_center")
    parser.add_argument("--bodyparts", default="")
    parser.add_argument("--minimum-dwell", type=float, default=0.3)
    parser.add_argument("--bridge-gap", type=float, default=0.2)
    parser.add_argument("--reentry-center-dwell", type=float, default=0.2)
    # Kept only so older saved command lines still parse. Re-entry now uses
    # the same bodyparts selected for arm-entry judgement.
    parser.add_argument("--reentry-bodyparts", default="")
    parser.add_argument("--strict-entry", action="store_true")
    return parser.parse_args(argv)


def runs(values: np.ndarray) -> list[tuple[int, int, str]]:
    if not len(values):
        return []
    result: list[tuple[int, int, str]] = []
    start, current = 0, str(values[0])
    for index in range(1, len(values)):
        value = str(values[index])
        if value != current:
            result.append((start, index, current))
            start, current = index, value
    result.append((start, len(values), current))
    return result


def bridge_short_gaps(values: np.ndarray, maximum_frames: int) -> np.ndarray:
    fixed = values.copy()
    if maximum_frames <= 0:
        return fixed
    for start, end, value in runs(fixed):
        if value or end - start > maximum_frames or start == 0 or end == len(fixed):
            continue
        before, after = str(fixed[start - 1]), str(fixed[end])
        if before and before == after:
            fixed[start:end] = before
    return fixed


def alternation(sequence: list[str]) -> tuple[int, int, float, list[bool]]:
    opportunities = max(len(sequence) - 2, 0)
    decisions = [len(set(sequence[index:index + 3])) == 3 for index in range(opportunities)]
    successes = int(sum(decisions))
    percent = successes / opportunities * 100.0 if opportunities else math.nan
    return successes, opportunities, percent, decisions


def point_zones(x: np.ndarray, y: np.ndarray, valid: np.ndarray, zones: list[dict]) -> np.ndarray:
    result = np.full(len(x), "", dtype=object)
    for zone in zones:
        zone_id = str(zone.get("id", ""))
        polygon = np.asarray(zone.get("polygon_source_px") or [], dtype=np.float32)
        if len(polygon) < 3:
            continue
        for frame in np.flatnonzero(valid & (result == "")):
            if cv2.pointPolygonTest(polygon, (float(x[frame]), float(y[frame])), False) >= 0:
                result[frame] = zone_id
    return result


def validate_region(region: dict) -> None:
    zones = region.get("zones", [])
    available = {str(zone.get("id", "")) for zone in zones}
    missing = [zone_id for zone_id in (*ARM_IDS, "center") if zone_id not in available]
    if missing:
        raise ValueError(
            tr('{v0}에 필요한 구역이 없습니다: {v1}. Y-maze 파일럿 편집기에서 Arm A/B/C와 Center를 모두 그리세요.', v0=region.get('name', 'Y-maze'), v1=', '.join(missing))
        )


from hagisilta_analysis_range import frame_bounds, load_ymaze_range


def analyze_arena(
    arena_dir: Path, region: dict, output: Path, pcutoff: float,
    bodyparts: list[str] | tuple[str, ...] | str,
    minimum_dwell: float, bridge_gap: float, strict_entry: bool = False,
    coordinate_space: str = "source_px",
    reentry_center_dwell: float = 0.2,
    analysis_range: dict | None = None,
) -> dict:
    validate_region(region)
    pose, pose_path = load_pose(arena_dir)
    metadata_path = arena_dir / "roi-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    video = arena_video(arena_dir)
    fps = float(metadata.get("fps", 0) or 0)
    if fps <= 0 and video:
        capture = cv2.VideoCapture(str(video)); fps = float(capture.get(cv2.CAP_PROP_FPS) or 0); capture.release()
    if fps <= 0:
        raise ValueError(tr('FPS를 확인할 수 없습니다: {v0}', v0=arena_dir))

    selected_range = load_ymaze_range(arena_dir) if analysis_range is None else analysis_range
    range_start = float(selected_range.get("start_seconds", 0.0))
    range_end = selected_range.get("end_seconds")
    start_frame, end_frame = frame_bounds(range_start, range_end, fps, len(pose))
    in_analysis_range = np.zeros(len(pose), dtype=bool)
    in_analysis_range[start_frame:end_frame] = True

    offset = np.asarray(metadata.get("source_offset_px", [0, 0]), dtype=float)
    crop_local = coordinate_space == "arena_crop_px"
    zones = region.get("zones", [])
    selected_bodyparts = [str(bodyparts)] if isinstance(bodyparts, str) else [
        str(part) for part in bodyparts if str(part).strip()
    ]
    if not selected_bodyparts:
        selected_bodyparts = ["body_center"]
    if strict_entry and "tail_base" not in selected_bodyparts:
        selected_bodyparts.append("tail_base")
    point_x: list[np.ndarray] = []
    point_y: list[np.ndarray] = []
    point_likelihood: list[np.ndarray] = []
    point_zone_ids: list[np.ndarray] = []
    point_valid: list[np.ndarray] = []
    for bodypart in selected_bodyparts:
        current_x = coordinate(pose, bodypart, "x")
        current_y = coordinate(pose, bodypart, "y")
        current_likelihood = coordinate(pose, bodypart, "likelihood")
        current_valid = (
            np.isfinite(current_x) & np.isfinite(current_y)
            & (current_likelihood >= pcutoff)
            & in_analysis_range
        )
        zone_x, zone_y = (
            (current_x, current_y)
            if crop_local else (current_x + offset[0], current_y + offset[1])
        )
        point_x.append(current_x)
        point_y.append(current_y)
        point_likelihood.append(current_likelihood)
        point_valid.append(current_valid)
        point_zone_ids.append(point_zones(zone_x, zone_y, current_valid, zones))
    x = np.nanmean(np.vstack(point_x), axis=0)
    y = np.nanmean(np.vstack(point_y), axis=0)
    likelihood = np.nanmin(np.vstack(point_likelihood), axis=0)
    valid = np.logical_and.reduce(point_valid)
    consensus_zone_ids = point_zone_ids[0].copy()
    for current_zones in point_zone_ids[1:]:
        consensus_zone_ids = np.where(
            consensus_zone_ids == current_zones, consensus_zone_ids, ""
        )
    arm_consensus = np.isin(consensus_zone_ids, ARM_IDS) & valid
    center_consensus = (consensus_zone_ids == "center") & valid
    center_reference_mask = center_consensus
    center_reference_zone_ids = np.where(center_consensus, "center", "")
    center_reference = "all:" + "+".join(selected_bodyparts)
    zone_ids = np.where(
        arm_consensus,
        consensus_zone_ids,
        np.where(
            center_reference_mask,
            np.where(center_consensus, "center", "transition"),
            "",
        ),
    )
    zone_ids = bridge_short_gaps(zone_ids, max(0, round(bridge_gap * fps)))

    minimum_frames = max(1, round(minimum_dwell * fps))
    # A repeated entry into the same arm is accepted after every judgement
    # bodypart has stayed outside the previous arm for a short, independent
    # dwell. Requiring simultaneous Center consensus is too strict for an
    # elongated animal: the leading point can already enter the next arm while
    # the trailing point is still clearing the previous arm, even though the
    # animal has unambiguously left it. Do not reuse bridge_gap here;
    # bridge_gap repairs brief pose dropouts inside an arm, while this value
    # confirms a genuine exit from the previous arm.
    exit_frames = max(1, round(max(reentry_center_dwell, 0.04) * fps))
    events: list[dict] = []
    sequence: list[str] = []
    previous_event_end = 0
    previous_zone_id = ""
    for start, end, zone_id in runs(zone_ids):
        if zone_id not in ARM_IDS or end - start < minimum_frames:
            continue
        arm = ARM_LABELS[zone_id]
        if sequence and arm == sequence[-1]:
            interval = slice(previous_event_end, start)
            clear_previous_arm = valid.copy()
            for current_zones in point_zone_ids:
                clear_previous_arm &= current_zones != previous_zone_id
            clear_previous_arm = clear_previous_arm[interval]
            exited_long_enough = any(
                value == "True" and run_end - run_start >= exit_frames
                for run_start, run_end, value in runs(clear_previous_arm.astype(object))
            )
            if not exited_long_enough:
                continue
        sequence.append(arm)
        events.append({
            "event_index": len(events) + 1,
            "arm": arm,
            "zone_id": zone_id,
            "entry_frame": start,
            "entry_seconds": start / fps,
            "entry_elapsed_seconds": (start - start_frame) / fps,
            "exit_frame": end - 1,
            "exit_seconds": end / fps,
            "exit_elapsed_seconds": (end - start_frame) / fps,
            "dwell_seconds": (end - start) / fps,
        })
        previous_event_end = end
        previous_zone_id = zone_id

    successes, opportunities, percent, decisions = alternation(sequence)
    for index, success in enumerate(decisions):
        events[index + 2]["triad"] = "".join(sequence[index:index + 3])
        events[index + 2]["alternation"] = bool(success)

    output.mkdir(parents=True, exist_ok=True)
    region_id = int(region.get("id", arena_id(arena_dir)))
    zones_path = write_text_resilient(
        output / f"arena_{region_id:02d}_zones.json",
        json.dumps(
            {"coordinate_space": coordinate_space, "region": region},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    event_path = output / f"arena_{region_id:02d}_arm_events.csv"
    frame_path = output / f"arena_{region_id:02d}_frames.csv"
    event_path = write_csv_resilient(pd.DataFrame(events, columns=[
        "event_index", "arm", "zone_id", "entry_frame", "entry_seconds", "entry_elapsed_seconds",
        "exit_frame", "exit_seconds", "exit_elapsed_seconds", "dwell_seconds", "triad", "alternation",
    ]), event_path, index=False, encoding="utf-8-sig")
    frame_columns = {
        "frame": np.arange(len(pose)),
        "time_seconds": np.arange(len(pose)) / fps,
        "in_analysis_range": in_analysis_range,
        "analysis_elapsed_seconds": np.where(in_analysis_range, (np.arange(len(pose)) - start_frame) / fps, np.nan),
        "x": x, "y": y, "likelihood": likelihood,
        "valid_pose": valid, "zone_id": zone_ids,
        "bodyparts_used": ",".join(selected_bodyparts),
        "reentry_bodyparts_used": ",".join(selected_bodyparts),
    }
    for index, bodypart in enumerate(selected_bodyparts):
        frame_columns[f"{bodypart}_x"] = point_x[index]
        frame_columns[f"{bodypart}_y"] = point_y[index]
        frame_columns[f"{bodypart}_likelihood"] = point_likelihood[index]
        frame_columns[f"{bodypart}_zone"] = point_zone_ids[index]
    frame_columns["reentry_reference_x"] = x
    frame_columns["reentry_reference_y"] = y
    frame_columns["reentry_reference_likelihood"] = likelihood
    frame_columns["reentry_reference_zone"] = center_reference_zone_ids
    frame_path = write_csv_resilient(
        pd.DataFrame(frame_columns), frame_path, index=False, encoding="utf-8-sig"
    )

    occupancy = {
        zone_id: float(np.count_nonzero(zone_ids == zone_id) / fps)
        for zone_id in ARM_IDS
    }
    occupancy["center"] = float(np.count_nonzero(center_reference_mask) / fps)
    entries = {arm: sequence.count(arm) for arm in ("A", "B", "C")}
    distance = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    distance_valid = valid[1:] & valid[:-1]

    visual_path = output / f"arena_{region_id:02d}_ymaze_visual.png"
    if video:
        capture = cv2.VideoCapture(str(video)); ok, background = capture.read(); capture.release()
        if ok:
            background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
            height, width = background.shape[:2]
            fig, axes = plt.subplots(
                1, 2,
                figsize=(RESULT_RENDER_WIDTH_IN, 7.2),
                dpi=RESULT_RENDER_DPI,
                constrained_layout=True,
            )
            for axis in axes:
                axis.imshow(background)
            plot_x, plot_y = x.copy(), y.copy(); plot_x[~valid] = np.nan; plot_y[~valid] = np.nan
            axes[0].plot(plot_x, plot_y, color="#00e5e5", linewidth=0.55, alpha=0.7)
            heat = seconds_heatmap(x, y, valid, width, height, fps)
            colors = {"arm_a": "#ffca5c", "arm_b": "#ff6b8a", "arm_c": "#66a8ff", "center": "#70e2ae"}
            roi_mask = np.zeros((height, width), dtype=np.uint8)
            for zone in zones:
                polygon = np.asarray(zone.get("polygon_source_px") or [], dtype=float)
                if not crop_local:
                    polygon = polygon - offset
                if len(polygon) < 3: continue
                cv2.fillPoly(
                    roi_mask,
                    [np.rint(polygon).astype(np.int32).reshape((-1, 1, 2))],
                    1,
                )
                closed = np.vstack([polygon, polygon[0]])
                for axis in axes:
                    axis.plot(closed[:, 0], closed[:, 1], color=colors.get(str(zone.get("id")), "white"), linewidth=1.5)
                    center = polygon.mean(axis=0)
                    axis.text(center[0], center[1], str(zone.get("name", "")), color="white", ha="center", va="center", fontsize=8)
            # Keep zero-second bins visible inside the maze. Only pixels
            # outside the configured A/B/C/Center ROI are transparent.
            heat = cv2.resize(heat, (width, height), interpolation=cv2.INTER_CUBIC)
            heat = np.maximum(heat, 0.0)
            roi_values = heat[roi_mask != 0]
            heat_max = max(
                float(np.max(roi_values)) if roi_values.size else 0.0,
                1.0 / max(fps, 1e-9),
            )
            heat = np.ma.masked_where(roi_mask == 0, heat)
            heat_image = axes[1].imshow(
                heat,
                cmap="turbo",
                alpha=0.78,
                vmin=0.0,
                vmax=heat_max,
                interpolation="bilinear",
            )
            colorbar = fig.colorbar(heat_image, ax=axes[1], fraction=0.046, pad=0.03)
            colorbar.set_label("Dwell time (s / spatial bin)")
            axes[0].set_title("Trajectory")
            axes[1].set_title("Occupancy heatmap")
            for axis in axes: axis.set_xlim(0, width); axis.set_ylim(height, 0); axis.axis("off")
            fig.suptitle(f"{arena_dir.parent.name} · {region.get('name', f'Y-maze {region_id}')} · alternation {percent:.1f}%" if not math.isnan(percent) else f"{arena_dir.parent.name} · alternation N/A")
            visual_path = save_figure_resilient(fig, visual_path, facecolor="white"); plt.close(fig)

    return {
        "arena": region_id,
        "arena_name": canonical_arena_name(region.get("name"), region_id),
        "bodyparts_used": ",".join(selected_bodyparts),
        "reentry_bodyparts_used": ",".join(selected_bodyparts),
        "center_reference": center_reference,
        "pose_file": str(pose_path.resolve()),
        "pose_postprocessing": json.dumps(pose.attrs.get("hagisilta_analysis_pose", {}), ensure_ascii=False),
        "sequence": "-".join(sequence),
        "total_entries": len(sequence),
        "alternation_successes": successes,
        "alternation_opportunities": opportunities,
        "alternation_percent": percent,
        "entries_a": entries["A"], "entries_b": entries["B"], "entries_c": entries["C"],
        "dwell_a_seconds": occupancy["arm_a"], "dwell_b_seconds": occupancy["arm_b"],
        "dwell_c_seconds": occupancy["arm_c"], "center_seconds": occupancy["center"],
        "analysis_start_seconds": start_frame / fps,
        "analysis_end_seconds": end_frame / fps,
        "analysis_duration_seconds": (end_frame - start_frame) / fps,
        "requested_start_seconds": range_start,
        "requested_end_seconds": len(pose) / fps if range_end is None else float(range_end),
        "analysis_start_frame": start_frame, "analysis_end_frame_exclusive": end_frame,
        "valid_pose_percent": float(valid[in_analysis_range].mean() * 100.0),
        "distance_px": float(np.nansum(distance[distance_valid])),
        "events_file": str(event_path.resolve()), "frames_file": str(frame_path.resolve()),
        "zones_file": str(zones_path.resolve()),
        "visual_image": str(visual_path.resolve()) if visual_path.is_file() else "",
    }


def single_main(argv: list[str] | None = None) -> int:
    args = parse_single_args(argv)
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    if profile.get("profile_type") != "hagisilta_zone_template":
        raise ValueError(tr('HAGISILTA 구역 템플릿이 아닙니다.'))
    regions = {int(region["id"]): region for region in profile.get("arenas", [])}
    arena_dirs = sorted(path for path in args.tracked_root.glob("arena_*") if path.is_dir())
    results = []
    for index, arena_dir in enumerate(arena_dirs, start=1):
        region_id = arena_id(arena_dir)
        if region_id not in regions:
            continue
        print(f"YMAZE_ARENA={index}/{len(arena_dirs)}:{region_id}", flush=True)
        selected_bodyparts = [
            part.strip() for part in str(args.bodyparts).split(",") if part.strip()
        ] or [str(args.bodypart)]
        results.append(analyze_arena(
            arena_dir, regions[region_id], args.output,
            min(max(args.pcutoff, 0.0), 1.0), selected_bodyparts,
            max(args.minimum_dwell, 0.0), max(args.bridge_gap, 0.0), args.strict_entry,
            str(profile.get("coordinate_space", "source_px")),
            max(args.reentry_center_dwell, 0.0),
        ))
    if not results:
        raise ValueError(tr('템플릿과 일치하는 Y-maze Arena가 없습니다.'))
    summary_path = args.output / "ymaze-summary.csv"
    summary_path = write_csv_resilient(
        pd.DataFrame(results), summary_path, index=False, encoding="utf-8-sig"
    )
    write_text_resilient(
        args.output / "ymaze-summary.json",
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8",
    )
    print(f"YMAZE_SUMMARY={summary_path.resolve()}", flush=True)
    return 0


def parse_batch_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HAGISILTA multi-video Y-maze batch analysis")
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args(argv)


def batch_main(argv: list[str] | None = None) -> int:
    args = parse_batch_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    items = list(plan.get("items", []) or [])
    if not items:
        # Legacy whole-frame plan support.
        profile_path = Path(str(plan.get("profile", "")))
        items = [
            {"bundle": str(bundle), "profile": str(profile_path), "arena_id": None}
            for bundle in plan.get("bundles", [])
        ]
    items = [item for item in items if Path(str(item.get("bundle", ""))).is_dir()]
    if not items:
        raise ValueError(tr('분석할 영상 묶음이 없습니다.'))

    args.output.mkdir(parents=True, exist_ok=True)
    base_run_id = "run_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = base_run_id; counter = 2
    run_output = args.output / "runs" / run_id
    while run_output.exists():
        run_id = f"{base_run_id}_{counter}"; counter += 1
        run_output = args.output / "runs" / run_id
    run_output.mkdir(parents=True, exist_ok=False)
    write_text_resilient(
        run_output / "analysis-settings.json",
        json.dumps({
            "run_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "pcutoff": float(plan.get("pcutoff", 0.6)),
            "bodyparts": list(plan.get("bodyparts", []) or [str(plan.get("bodypart", "body_center"))]),
            "minimum_dwell": float(plan.get("minimum_dwell", 0.3)),
            "bridge_gap": float(plan.get("bridge_gap", 0.2)),
            "reentry_center_dwell": float(plan.get("reentry_center_dwell", 0.2)),
            "strict_entry": bool(plan.get("strict_entry", False)),
            "planned_items": len(items),
            "items": items,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"YMAZE_RUN={run_output.resolve()}", flush=True)
    all_results: list[dict] = []
    bundle_names = list(dict.fromkeys(str(Path(str(item["bundle"]))) for item in items))
    for bundle_index, bundle_text in enumerate(bundle_names, start=1):
        bundle = Path(bundle_text)
        print(f"YMAZE_BUNDLE={bundle_index}/{len(bundle_names)}:{bundle.name}", flush=True)
        bundle_output = run_output / bundle.name
        bundle_items = [item for item in items if Path(str(item["bundle"])) == bundle]
        arena_dirs = []
        for item in bundle_items:
            requested_id = item.get("arena_id")
            candidates = sorted(path for path in bundle.glob("arena_*") if path.is_dir())
            if requested_id is not None:
                candidates = [path for path in candidates if arena_id(path) == int(requested_id)]
            for arena_dir in candidates:
                arena_dirs.append((arena_dir, Path(str(item.get("profile", ""))), item.get("analysis_range")))
        for arena_index, (arena_dir, profile_path, analysis_range) in enumerate(arena_dirs, start=1):
            region_id = arena_id(arena_dir)
            if not profile_path.is_file():
                raise FileNotFoundError(tr('Arena {v0} Y-maze 템플릿이 없습니다: {v1}', v0=region_id, v1=profile_path))
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            if profile.get("profile_type") != "hagisilta_zone_template":
                raise ValueError(tr('HAGISILTA 구역 템플릿이 아닙니다: {v0}', v0=profile_path))
            regions = {int(region["id"]): region for region in profile.get("arenas", [])}
            if region_id not in regions:
                raise ValueError(tr('{v0}에 Arena {v1} 구역이 없습니다.', v0=profile_path.name, v1=region_id))
            print(f"YMAZE_ARENA={arena_index}/{len(arena_dirs)}:{region_id}", flush=True)
            result = analyze_arena(
                arena_dir, regions[region_id], bundle_output,
                min(max(float(plan.get("pcutoff", 0.6)), 0.0), 1.0),
                list(plan.get("bodyparts", []) or [str(plan.get("bodypart", "body_center"))]),
                max(float(plan.get("minimum_dwell", 0.3)), 0.0),
                max(float(plan.get("bridge_gap", 0.2)), 0.0),
                bool(plan.get("strict_entry", False)),
                str(profile.get("coordinate_space", "source_px")),
                max(float(plan.get("reentry_center_dwell", 0.2)), 0.0),
                analysis_range=analysis_range,
            )
            result["video_bundle"] = bundle.name
            result["tracked_bundle"] = str(bundle.resolve())
            all_results.append(result)
        if arena_dirs:
            bundle_rows = [row for row in all_results if row["video_bundle"] == bundle.name]
            write_csv_resilient(
                pd.DataFrame(bundle_rows), bundle_output / "ymaze-summary.csv",
                index=False, encoding="utf-8-sig",
            )

    if not all_results:
        raise ValueError(tr('템플릿 Arena 번호와 일치하는 추적 결과가 없습니다.'))
    summary_path = run_output / "project-ymaze-summary.csv"
    summary_path = write_csv_resilient(
        pd.DataFrame(all_results), summary_path, index=False, encoding="utf-8-sig"
    )
    summary_json = write_text_resilient(
        run_output / "project-ymaze-summary.json",
        json.dumps(all_results, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8",
    )
    # A tiny pointer is never opened by spreadsheet software.  It lets the UI
    # follow a versioned summary when the conventional CSV/JSON is locked.
    pointer = {
        "csv": str(summary_path.resolve()),
        "json": str(summary_json.resolve()),
    }
    write_text_resilient(
        run_output / "project-ymaze-summary.latest.json",
        json.dumps(pointer, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    write_text_resilient(
        args.output / "latest-run.json",
        json.dumps({
            "run_id": run_id,
            "run_directory": str(run_output.resolve()),
            "summary_csv": str(summary_path.resolve()),
            "summary_json": str(summary_json.resolve()),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"YMAZE_PROJECT_SUMMARY={summary_path.resolve()}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HAGISILTA Y-maze analysis")
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("single", add_help=False, help="Analyze one tracked video bundle")
    modes.add_parser("batch", add_help=False, help="Analyze the items in a project plan")
    args, remaining = parser.parse_known_args(argv)
    return single_main(remaining) if args.mode == "single" else batch_main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
