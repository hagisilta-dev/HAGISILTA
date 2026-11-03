# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from hagisilta_i18n import tr

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# Persisted report text must be independent of the interface language.
QC_WARNING = "Confidence is not the same as coordinate accuracy; also review the label overlays."


def load_dlc_csv(path: Path) -> tuple[pd.DataFrame, int, int]:
    if path.suffix.lower() in {".h5", ".hdf5"}:
        data = pd.read_hdf(path)
        coord_level = next(
            level
            for level in range(data.columns.nlevels)
            if "likelihood" in set(map(str, data.columns.get_level_values(level)))
        )
        return data, coord_level - 1, coord_level
    last_error: Exception | None = None
    for header_rows in (3, 4):
        try:
            data = pd.read_csv(path, header=list(range(header_rows)), index_col=0)
            coord_level = next(
                level
                for level in range(data.columns.nlevels)
                if "likelihood" in set(map(str, data.columns.get_level_values(level)))
            )
            if coord_level == 0:
                continue
            return data, coord_level - 1, coord_level
        except Exception as error:
            last_error = error
    raise ValueError(tr('DLC CSV 헤더를 해석할 수 없습니다: {v0}', v0=path)) from last_error


def bodypart_arrays(
    data: pd.DataFrame, bodypart_level: int, coord_level: int
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    bodyparts = list(dict.fromkeys(map(str, data.columns.get_level_values(bodypart_level))))
    for bodypart in bodyparts:
        columns = [column for column in data.columns if str(column[bodypart_level]) == bodypart]
        coords: dict[str, np.ndarray] = {}
        for coordinate in ("x", "y", "likelihood"):
            matches = [column for column in columns if str(column[coord_level]) == coordinate]
            if matches:
                coords[coordinate] = pd.to_numeric(data[matches[0]], errors="coerce").to_numpy(float)
        if all(name in coords for name in ("x", "y", "likelihood")):
            result[bodypart] = coords
    if not result:
        raise ValueError(tr('CSV에서 x/y/likelihood 신체점을 찾지 못했습니다.'))
    return result


def finite_percentile(values: np.ndarray, percentile: float) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, percentile)) if finite.size else None


def observed_mask(values: dict[str, np.ndarray]) -> np.ndarray:
    """Return frames containing an actual DLC observation, excluding -1 sentinels."""
    return (
        np.isfinite(values["x"])
        & np.isfinite(values["y"])
        & np.isfinite(values["likelihood"])
        & (values["likelihood"] >= 0.0)
    )


def run_lengths(mask: np.ndarray) -> list[int]:
    """Return lengths of consecutive True runs."""
    values = np.asarray(mask, dtype=bool)
    if not values.size:
        return []
    padded = np.concatenate(([False], values, [False])).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [int(end - start) for start, end in zip(starts, ends)]


def analyze_pose_csv(path: Path, cutoff: float = 0.6, hard_frame_count: int = 50) -> dict:
    data, bodypart_level, coord_level = load_dlc_csv(path)
    points = bodypart_arrays(data, bodypart_level, coord_level)
    frame_count = len(data)
    metrics: dict[str, dict] = {}
    hard_score = np.zeros(frame_count, dtype=float)
    trusted_masks: dict[str, np.ndarray] = {}
    jump_candidate_masks: dict[str, np.ndarray] = {}
    frame_rows: list[dict] = []

    for bodypart, values in points.items():
        likelihood = values["likelihood"]
        observed = observed_mask(values)
        trusted = observed & (likelihood >= cutoff)
        trusted_masks[bodypart] = trusted
        coverage = float(np.count_nonzero(trusted) / frame_count) if frame_count else 0.0
        detection_coverage = float(np.count_nonzero(observed) / frame_count) if frame_count else 0.0
        xy = np.column_stack([values["x"], values["y"]])
        consecutive = trusted[1:] & trusted[:-1]
        all_jumps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        jumps = all_jumps[consecutive]
        jump95 = finite_percentile(jumps, 95)
        jump99 = finite_percentile(jumps, 99)
        finite_jumps = jumps[np.isfinite(jumps)]
        if finite_jumps.size:
            jump_median = float(np.median(finite_jumps))
            jump_mad = float(np.median(np.abs(finite_jumps - jump_median)))
            jump_threshold = max(25.0, jump_median + 6.0 * 1.4826 * jump_mad)
        else:
            jump_threshold = 25.0
        jump_candidates = np.zeros(frame_count, dtype=bool)
        if frame_count > 1:
            jump_candidates[1:] = consecutive & np.isfinite(all_jumps) & (all_jumps > jump_threshold)
        jump_candidate_masks[bodypart] = jump_candidates
        dropout_runs = run_lengths(~trusted)
        valid_runs = run_lengths(trusted)
        metrics[bodypart] = {
            "coverage_at_cutoff": coverage,
            "detection_coverage": detection_coverage,
            "median_likelihood": finite_percentile(likelihood[observed], 50),
            "jump_p95_px_per_frame": jump95,
            "jump_p99_px_per_frame": jump99,
            "trusted_consecutive_jumps": int(np.count_nonzero(consecutive)),
            "trusted_frames": int(np.count_nonzero(trusted)),
            "low_confidence_frames": int(np.count_nonzero(observed & ~trusted)),
            "missing_frames": int(np.count_nonzero(~observed)),
            "dropout_segment_count": len(dropout_runs),
            "longest_dropout_frames": max(dropout_runs, default=0),
            "mean_dropout_frames": float(np.mean(dropout_runs)) if dropout_runs else 0.0,
            "jump_threshold_px": float(jump_threshold),
            "jump_candidate_count": int(np.count_nonzero(jump_candidates)),
            "longest_valid_run_frames": max(valid_runs, default=0),
        }
        previous_xy = np.vstack(([np.nan, np.nan], xy[:-1]))
        jump_values = np.linalg.norm(xy - previous_xy, axis=1)
        for frame_index in range(frame_count):
            frame_rows.append({
                "frame_index": frame_index,
                "bodypart": bodypart,
                "x": float(values["x"][frame_index]) if np.isfinite(values["x"][frame_index]) else "",
                "y": float(values["y"][frame_index]) if np.isfinite(values["y"][frame_index]) else "",
                "likelihood": float(likelihood[frame_index]) if np.isfinite(likelihood[frame_index]) else "",
                "observed": int(observed[frame_index]),
                "trusted": int(trusted[frame_index]),
                "dropout": int(not trusted[frame_index]),
                "jump_px": float(jump_values[frame_index]) if np.isfinite(jump_values[frame_index]) else "",
                "jump_candidate": int(jump_candidates[frame_index]),
            })
        point_weight = 1.0 / max(len(points), 1)
        confidence_penalty = np.ones(frame_count, dtype=float)
        confidence_penalty[observed] = np.clip(1.0 - likelihood[observed], 0.0, 1.0)
        hard_score += confidence_penalty * point_weight
        if jumps.size and consecutive.any():
            scale = max(finite_percentile(jumps, 95) or 1.0, 1.0)
            jump_penalty = np.zeros(frame_count - 1, dtype=float)
            jump_penalty[consecutive] = np.clip(all_jumps[consecutive] / scale - 1.0, 0.0, 3.0)
            hard_score[1:] += jump_penalty * point_weight

    coverage = {name: values["coverage_at_cutoff"] for name, values in metrics.items()}
    coverage_values = np.asarray(list(coverage.values()), dtype=float)
    lowest_bodypart = min(coverage, key=coverage.get) if coverage else ""
    coverage_summary = {
        "bodypart_count": len(coverage),
        "mean_coverage": float(np.mean(coverage_values)) if coverage_values.size else 0.0,
        "median_coverage": float(np.median(coverage_values)) if coverage_values.size else 0.0,
        "p10_coverage": float(np.percentile(coverage_values, 10)) if coverage_values.size else 0.0,
        "minimum_coverage": float(coverage.get(lowest_bodypart, 0.0)),
        "lowest_bodypart": lowest_bodypart,
    }

    ranked = np.argsort(-hard_score)[: min(hard_frame_count, frame_count)]
    hard_frames = [
        {"frame_index": int(index), "hardness_score": float(hard_score[index])}
        for index in ranked
    ]
    required_masks = list(trusted_masks.values())
    overall_trusted = (
        np.logical_and.reduce(required_masks)
        if required_masks else np.zeros(frame_count, dtype=bool)
    )
    overall_dropout_runs = run_lengths(~overall_trusted)
    overall_valid_runs = run_lengths(overall_trusted)
    overall_jump = (
        np.logical_or.reduce(list(jump_candidate_masks.values()))
        if jump_candidate_masks else np.zeros(frame_count, dtype=bool)
    )
    overall = {
        "total_frames": frame_count,
        "analyzable_frames": int(np.count_nonzero(overall_trusted)),
        "analyzable_rate": float(np.count_nonzero(overall_trusted) / frame_count) if frame_count else 0.0,
        "low_confidence_or_missing_frames": int(np.count_nonzero(~overall_trusted)),
        "dropout_segment_count": len(overall_dropout_runs),
        "longest_dropout_frames": max(overall_dropout_runs, default=0),
        "mean_dropout_frames": float(np.mean(overall_dropout_runs)) if overall_dropout_runs else 0.0,
        "jump_candidate_frames": int(np.count_nonzero(overall_jump)),
        "longest_valid_run_frames": max(overall_valid_runs, default=0),
    }
    return {
        "schema_version": 4,
        "pose_file": str(path.resolve()),
        "frame_count": frame_count,
        "pcutoff": cutoff,
        "coverage_summary": coverage_summary,
        "metrics": metrics,
        "overall": overall,
        "frame_rows": frame_rows,
        "hard_frames": hard_frames,
        "warning": QC_WARNING,
    }


def write_report(report: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "pose_qc_report.json"
    csv_path = output_dir / "hard_frames.csv"
    pd.DataFrame(report["hard_frames"]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary_path = output_dir / "tracking-qc-summary.csv"
    pd.DataFrame([{
        "pose_file": report.get("pose_file", ""),
        "pcutoff": report.get("pcutoff", ""),
        **report.get("overall", {}),
    }]).to_csv(summary_path, index=False, encoding="utf-8-sig")
    bodypart_path = output_dir / "tracking-qc-by-bodypart.csv"
    pd.DataFrame([
        {"bodypart": bodypart, **values}
        for bodypart, values in report.get("metrics", {}).items()
    ]).to_csv(bodypart_path, index=False, encoding="utf-8-sig")
    frames_path = output_dir / "tracking-qc-frames.csv"
    pd.DataFrame(report.get("frame_rows", [])).to_csv(
        frames_path, index=False, encoding="utf-8-sig"
    )
    report["artifacts"] = {
        "summary_csv": str(summary_path.resolve()),
        "bodypart_csv": str(bodypart_path.resolve()),
        "frames_csv": str(frames_path.resolve()),
        "hard_frames_csv": str(csv_path.resolve()),
    }
    # 프레임별 장문 데이터는 CSV에만 보관하여 UI가 읽는 JSON이 비대해지지 않게 한다.
    json_report = dict(report)
    json_report.pop("frame_rows", None)
    json_path.write_text(
        json.dumps(json_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return json_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Score DLC pose inference quality and rank hard frames.")
    parser.add_argument("pose_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--cutoff", type=float, default=0.6)
    parser.add_argument("--hard-frames", type=int, default=50)
    args = parser.parse_args()
    report = analyze_pose_csv(args.pose_csv.resolve(), args.cutoff, args.hard_frames)
    json_path, csv_path = write_report(report, args.output_dir.resolve())
    print(json.dumps({"coverage_summary": report["coverage_summary"], "report": str(json_path), "hard_frames": str(csv_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
