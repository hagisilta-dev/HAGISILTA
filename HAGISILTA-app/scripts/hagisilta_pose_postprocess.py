# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
"""Non-destructive, CPU-only pose filtering, recipes and analysis inputs.

The interior operation is DLC's scipy.signal.medfilt, independently for x/y.
Unlike its zero-padded boundaries, incomplete/non-finite windows retain the
original point. No model, likelihood, human label or tracking output is changed.
"""
from __future__ import annotations

from hagisilta_i18n import tr

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile
from uuid import uuid4

import cv2
import numpy as np
import pandas as pd
from scipy.signal import medfilt

from hagisilta_atomic_io import atomic_json, atomic_write_bytes


FILTER_SCHEMA = "hagisilta-pose-filter-test-v1"
COMPOSITION_SCHEMA = "hagisilta-pose-composition-v1"
WINDOWS = tuple(range(3, 32, 2))
RECIPE_SCHEMA = "hagisilta-pose-recipe-v1"
CHOICE_FILE = ".hagisilta-analysis-pose.json"
DEFAULT_RECIPE_NAME = "신체점 조합"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_stamp(path: Path) -> dict:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def source_key(video: Path, pose: Path) -> str:
    value = str(video.resolve()).casefold() + "\n" + str(pose.resolve()).casefold()
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def video_info(video: Path) -> dict:
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(tr('영상을 열 수 없습니다: {v0}', v0=video))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        return {
            "fps": fps if math.isfinite(fps) and fps > 0 else None,
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "reported_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    finally:
        capture.release()


def window_description(window: int, fps: float | None) -> str:
    half = window // 2
    if fps and math.isfinite(fps) and fps > 0:
        return tr('{v0:g} FPS · 앞뒤 {v1}프레임씩 · ±{v2:.3f}초', v0=fps, v1=half, v2=half / fps)
    return tr('FPS 확인 불가 · 앞뒤 {v0}프레임씩', v0=half)


def read_predictions(path: Path) -> pd.DataFrame:
    frame = (pd.read_hdf(path) if path.suffix.lower() in {".h5", ".hdf5"}
             else pd.read_csv(path, header=[0, 1, 2], index_col=0))
    validate_predictions(frame)
    return frame


def validate_predictions(frame: pd.DataFrame) -> None:
    if not isinstance(frame.columns, pd.MultiIndex) or frame.columns.nlevels != 3:
        raise ValueError(tr('단일 동물 DLC 추론 좌표(3단계 열 구조)가 필요합니다.'))
    if frame.empty or not frame.columns.is_unique:
        raise ValueError(tr('추론 좌표가 비어 있거나 중복된 열이 있습니다.'))
    if len(frame.columns.get_level_values(0).unique()) != 1:
        raise ValueError(tr('한 모델의 추론 좌표를 선택하세요.'))
    # Sparse labelled-image rows must never be treated as adjacent video frames.
    try:
        indices = np.asarray(frame.index, dtype=float)
    except (ValueError, TypeError):
        raise ValueError(tr('학습 라벨이 아닌 연속 영상의 추론 좌표가 필요합니다.')) from None
    if not np.array_equal(indices, np.arange(len(frame))):
        raise ValueError(tr('프레임 번호가 0부터 연속된 추론 좌표가 필요합니다.'))
    for key in dict.fromkeys(column[:-1] for column in frame.columns):
        if not all((*key, coord) in frame.columns for coord in ("x", "y", "likelihood")):
            raise ValueError(tr('신체점의 x/y/likelihood 열이 없습니다: {v0}', v0=key[-1]))


def median_predictions(frame: pd.DataFrame, window: int) -> tuple[pd.DataFrame, dict]:
    validate_predictions(frame)
    if window not in WINDOWS:
        raise ValueError(tr('필터 범위는 3~31 사이의 홀수 프레임 수로 지정하세요.'))
    if len(frame) < window:
        raise ValueError(tr('전체 {v0}프레임보다 필터 범위 {v1}프레임이 큽니다.', v0=len(frame), v1=window))
    result = frame.copy(deep=True)
    half = window // 2
    changes, eligible_count = [], 0
    for key in dict.fromkeys(column[:-1] for column in frame.columns):
        columns = [(*key, coord) for coord in ("x", "y")]
        xy = frame[columns].to_numpy(dtype=float)
        finite = np.isfinite(xy).all(axis=1)
        complete = np.convolve(finite.astype(int), np.ones(window, dtype=int), mode="same") == window
        complete[:half] = False
        complete[-half:] = False
        filtered = np.column_stack([medfilt(np.where(np.isfinite(xy[:, axis]), xy[:, axis], 0.0), window)
                                    for axis in (0, 1)])
        updated = xy.copy()
        updated[complete] = filtered[complete]
        result.loc[:, columns] = updated
        eligible_count += int(complete.sum())
        changes.extend(np.linalg.norm(updated[finite] - xy[finite], axis=1).tolist())
    delta = np.asarray(changes)
    summary = {
        "frames": len(frame), "points": len(changes), "eligible_points": eligible_count,
        "changed_points": int((delta > 1e-9).sum()),
        "median_change_px": float(np.median(delta)) if delta.size else None,
        "p95_change_px": float(np.percentile(delta, 95)) if delta.size else None,
        "max_change_px": float(delta.max()) if delta.size else None,
    }
    return result, summary


def create_filter_test(video: Path, pose: Path, output_root: Path, window: int) -> Path:
    video, pose = video.resolve(), pose.resolve()
    pose_stamp, video_stamp = file_stamp(pose), file_stamp(video)
    source_hash = sha256_file(pose)
    original = read_predictions(pose)
    filtered, summary = median_predictions(original, window)
    info = video_info(video)
    if file_stamp(pose) != pose_stamp or sha256_file(pose) != source_hash or file_stamp(video) != video_stamp:
        raise RuntimeError(tr('입력 파일이 처리 중 변경되었습니다. 추론이 끝난 뒤 다시 실행하세요.'))
    target = output_root.resolve() / source_key(video, pose)
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run = Path(tempfile.mkdtemp(prefix=f"median-{window:02d}-{stamp}-", dir=target))
    original_path, filtered_path = run / "original.h5", run / "median.h5"
    original.to_hdf(original_path, key="df_with_missing", mode="w", format="fixed")
    filtered.to_hdf(filtered_path, key="df_with_missing", mode="w", format="fixed")
    filtered.to_csv(run / "median.csv")
    pd.testing.assert_frame_equal(pd.read_hdf(original_path), original)
    pd.testing.assert_frame_equal(pd.read_hdf(filtered_path), filtered)
    if file_stamp(video) != video_stamp or file_stamp(pose) != pose_stamp or sha256_file(pose) != source_hash:
        raise RuntimeError(tr('입력 파일이 저장 중 변경되어 비교 결과를 확정하지 않았습니다.'))
    manifest = {
        "schema": FILTER_SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "median", "window": window, "policy": "preserve-incomplete-or-nonfinite-point-window",
        "source_video": str(video), "source_pose": str(pose), "source_pose_sha256": source_hash,
        "source_pose_stamp": pose_stamp, "video_stamp": video_stamp, "video_info": info,
        "original": "original.h5", "filtered": "median.h5", "csv": "median.csv",
        "original_sha256": sha256_file(original_path), "filtered_sha256": sha256_file(filtered_path),
        "summary": summary,
    }
    manifest_path = run / "filter-test.json"
    atomic_json(manifest_path, manifest)
    return manifest_path


def list_filter_tests(output_root: Path, video: Path, pose: Path) -> list[dict]:
    target = output_root / source_key(video, pose)
    rows = []
    for path in target.glob("*/filter-test.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if (not isinstance(record, dict) or record.get("schema") != FILTER_SCHEMA
                    or record.get("method") != "median" or record.get("window") not in WINDOWS):
                continue
            if source_key(Path(record["source_video"]), Path(record["source_pose"])) != target.name:
                continue
            datetime.fromisoformat(record["created_at"])
            if not isinstance(record["summary"], dict):
                continue
            for key in ("frames", "changed_points"):
                int(record["summary"][key])
            for key in ("original", "filtered"):
                member = (path.parent / record[key]).resolve()
                if member.parent != path.parent.resolve() or not member.is_file():
                    raise ValueError(tr('비교 좌표 파일 없음'))
            record["manifest"] = str(path)
            rows.append(record)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(rows, key=lambda row: row["created_at"], reverse=True)


def load_filter_test(path: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    path = path.resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") != FILTER_SCHEMA or record.get("method") != "median" or record.get("window") not in WINDOWS:
        raise ValueError(tr('지원하지 않는 후처리 비교 결과입니다.'))
    video = Path(record["source_video"])
    if file_stamp(video) != record["video_stamp"]:
        raise ValueError(tr('비교 결과 생성 후 영상 파일이 변경되었습니다. 새 비교 결과를 만드세요.'))
    data = []
    for key in ("original", "filtered"):
        member = (path.parent / record[key]).resolve()
        if member.parent != path.parent or sha256_file(member) != record[f"{key}_sha256"]:
            raise ValueError(tr('비교 좌표 파일이 변경되었거나 손상되었습니다.'))
        data.append(read_predictions(member))
    if not data[0].index.equals(data[1].index) or not data[0].columns.equals(data[1].columns):
        raise ValueError(tr('원본과 필터 결과의 프레임 또는 신체점 구성이 다릅니다.'))
    return record, data[0], data[1]


def load_comparison_family(path: Path) -> tuple[dict, pd.DataFrame, dict[int, dict]]:
    """Load one latest result per window, always retaining the selected run.

    A path alone does not identify a baseline: a later inference can replace
    it. Require the video stamp, source hash AND equal archived coordinates.
    Broken sibling results must not prevent opening a valid selected result.
    """
    path = path.resolve()
    selected, original, filtered = load_filter_test(path)
    variants = {selected["window"]: {**selected, "manifest": str(path), "pose": filtered}}
    candidates = list_filter_tests(path.parent.parent.parent,
                                   Path(selected["source_video"]), Path(selected["source_pose"]))
    for record in candidates:
        if (record["window"] in variants
                or record.get("source_pose_sha256") != selected["source_pose_sha256"]
                or record.get("video_stamp") != selected["video_stamp"]):
            continue
        try:
            validated, baseline, result = load_filter_test(Path(record["manifest"]))
            if not original.equals(baseline):
                continue
        except (OSError, ValueError, KeyError, TypeError):
            continue
        variants[record["window"]] = {**validated, "manifest": record["manifest"], "pose": result}
    return selected, original, dict(sorted(variants.items()))


def comparison_playlist(payloads: list[dict], output_root: Path, *,
                        selected_pose: Path, selected_manifest: Path | None = None) -> tuple[list[dict], int]:
    """Freeze the visible results on open; no filesystem work in context menus."""
    entries, selected_index, seen = [], 0, set()
    for row in payloads:
        video, pose = Path(row["video"]).resolve(), Path(row["pose"]).resolve()
        key = source_key(video, pose)
        if key in seen:
            continue
        seen.add(key)
        records = sorted(list_filter_tests(output_root, video, pose) + list_compositions(output_root, video, pose),
                         key=lambda row: row["created_at"], reverse=True)
        manifest = records[0]["manifest"] if records else ""
        if pose == selected_pose.resolve():
            selected_index = len(entries)
            if selected_manifest is not None:
                manifest = str(selected_manifest.resolve())
        composition = manifest if manifest and Path(manifest).name == "composition.json" else ""
        entries.append({"video": str(video), "pose": str(pose), "manifest": "" if composition else manifest,
                        "composition": composition,
                        "label": video.stem,
                        "result_label": str(row.get("result_label", "") or pose.stem)})
    if not entries:
        raise ValueError(tr('확인할 트래킹 결과가 없습니다.'))
    return entries, selected_index


def load_comparison_entry(entry: dict):
    """Return a validated family or the raw-only view for an unfiltered video."""
    if entry.get("composition"):
        path = Path(entry["composition"])
        record, original, _composed = load_composition(path)
        if source_key(Path(record["source_video"]), Path(record["source_pose"])) != source_key(
                Path(entry["video"]), Path(entry["pose"])):
            raise ValueError(tr('선택 영상과 보정 조합의 원본이 일치하지 않습니다.'))
        variants = {}
        for sibling in list_filter_tests(path.parent.parent.parent, Path(entry["video"]), Path(entry["pose"])):
            if (sibling["window"] in variants or sibling.get("source_pose_sha256") != record["source_pose_sha256"]
                    or sibling.get("video_stamp") != record["video_stamp"]):
                continue
            try:
                validated, baseline, result = load_filter_test(Path(sibling["manifest"]))
                if not original.equals(baseline):
                    continue
            except (OSError, ValueError, KeyError, TypeError):
                continue
            variants[sibling["window"]] = {**validated, "manifest": sibling["manifest"], "pose": result}
        return record, original, dict(sorted(variants.items())), path.parent / record["original"]
    manifest = entry.get("manifest")
    if manifest:
        record, original, variants = load_comparison_family(Path(manifest))
        if source_key(Path(record["source_video"]), Path(record["source_pose"])) != source_key(
                Path(entry["video"]), Path(entry["pose"])):
            raise ValueError(tr('선택 영상과 보정 결과의 원본이 일치하지 않습니다.'))
        pose_path = Path(manifest).resolve().parent / record["original"]
    else:
        pose_path = Path(entry["pose"]).resolve()
        original = read_predictions(pose_path)
        record = {"source_video": entry["video"], "source_pose": entry["pose"],
                  "video_info": video_info(Path(entry["video"]))}
        variants = {}
    return record, original, variants, pose_path


def create_filter_batch(plan: Path, report_path: Path, output_root: Path, window: int,
                        recipe: dict | None = None, recipe_name: str = "신체점 조합") -> dict:
    jobs = json.loads(plan.read_text(encoding="utf-8"))["jobs"]
    if not isinstance(jobs, list) or not jobs:
        raise ValueError(tr('보정할 트래킹 결과를 하나 이상 선택하세요.'))
    if window not in WINDOWS:
        raise ValueError(tr('필터 범위는 3~31 사이의 홀수 프레임 수로 지정하세요.'))
    unique = {}
    for job in jobs:
        unique.setdefault(source_key(Path(job["video"]), Path(job["pose"])), job)
    report = {"schema": "hagisilta-filter-batch-v1", "window": window, "complete": False,
              "recipe": recipe,
              "jobs": [{"video": str(job["video"]), "pose": str(job["pose"]), "status": "pending"}
                       for job in unique.values()]}

    def save_report():
        atomic_json(report_path, report)

    save_report()
    for index, job in enumerate(report["jobs"], 1):
        print(f"FILTER_BATCH={index}/{len(report['jobs'])}:{Path(job['video']).name}", flush=True)
        try:
            if recipe is None:
                manifest = create_filter_test(Path(job["video"]), Path(job["pose"]), output_root, window)
            else:
                manifest = create_composition(Path(job["video"]), Path(job["pose"]), output_root, recipe, recipe_name)
            job.update(status="complete", manifest=str(manifest))
            record = json.loads(manifest.read_text(encoding="utf-8"))
            description = window_description(window, record['video_info']['fps']) if recipe is None else recipe_display_name(recipe_name)
            print(f"FILTER_BATCH_SAVED={index}:{description}:{manifest}", flush=True)
        except Exception as error:
            job.update(status="failed", error=str(error))
            print(f"FILTER_BATCH_FAILED={index}:{error}", flush=True)
        save_report()  # Completed runs remain discoverable even after cancellation.
    report["complete"] = True
    save_report()
    return report


def recipe_display_name(name: str | None) -> str:
    """Localize only the built-in default; retain named user recipes verbatim."""
    return tr(DEFAULT_RECIPE_NAME) if not name or name == DEFAULT_RECIPE_NAME else str(name)


def validate_recipe(recipe: dict, bodyparts=None) -> dict[str, int]:
    if not isinstance(recipe, dict) or not recipe:
        raise ValueError(tr('신체점별 보정 조합이 비어 있습니다.'))
    if any(not isinstance(name, str) or not name or type(value) is not int or
           value not in (0, *WINDOWS) for name, value in recipe.items()):
        raise ValueError(tr('보정 조합은 신체점 이름과 보정 전(0) 또는 3~31 홀수 프레임 수여야 합니다.'))
    if bodyparts is not None and set(recipe) != set(bodyparts):
        missing, extra = set(bodyparts) - set(recipe), set(recipe) - set(bodyparts)
        raise ValueError(tr('신체점 구성 불일치 · 조합에 없음: {v0} · 영상에 없음: {v1}', v0=', '.join(sorted(missing)) or '-', v1=', '.join(sorted(extra)) or '-'))
    return dict(recipe)


def recipe_label(recipe: dict) -> str:
    return " · ".join(f"{name}: {tr('보정 전') if value == 0 else str(value) + 'F'}" for name, value in recipe.items())


def read_recipe(manifest: Path) -> tuple[dict, str]:
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if record.get("schema") not in (COMPOSITION_SCHEMA, RECIPE_SCHEMA):
        raise ValueError(tr('지원하지 않는 신체점 보정 조합입니다.'))
    return validate_recipe(record["recipe"]), str(record.get("name") or DEFAULT_RECIPE_NAME)


def save_recipe_preset(root: Path, recipe: dict, name: str) -> Path:
    """Store settings only: no source coordinates, filtered files or analysis choice."""
    record = {"schema": RECIPE_SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
              "name": name.strip()[:120] or DEFAULT_RECIPE_NAME, "recipe": validate_recipe(recipe)}
    manifest = root.resolve() / "recipes" / (uuid4().hex + ".json")
    atomic_json(manifest, record)
    return manifest


def list_recipe_presets(root: Path) -> list[dict]:
    # Old composition results remain usable as recipes without migration/writes.
    rows = list_compositions(root)
    for manifest in (root / "recipes").glob("*.json"):
        try:
            record = json.loads(manifest.read_text(encoding="utf-8"))
            if record.get("schema") != RECIPE_SCHEMA:
                continue
            validate_recipe(record["recipe"])
            datetime.fromisoformat(record["created_at"])
            rows.append({**record, "manifest": str(manifest.resolve())})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(rows, key=lambda row: row["created_at"], reverse=True)


def compose_predictions(original: pd.DataFrame, recipe: dict) -> tuple[pd.DataFrame, dict]:
    validate_predictions(original)
    parts = list(dict.fromkeys(original.columns.get_level_values(1)))
    recipe = validate_recipe(recipe, parts)
    result = original.copy(deep=True)
    per_part = {}
    # Filter each chosen point directly from the baseline; never filter a filter.
    for part in parts:
        columns = [column for column in original.columns if column[1] == part]
        if recipe[part]:
            filtered, summary = median_predictions(original[columns], recipe[part])
            result.loc[:, columns] = filtered
        else:
            summary = {"changed_points": 0, "p95_change_px": 0.0}
        per_part[part] = summary
    return result, {"frames": len(original), "changed_points": sum(row["changed_points"] for row in per_part.values()),
                    "per_bodypart": per_part, "p95_change_px": None}


def notify_compositions(root: Path) -> None:
    atomic_json(root / "composition-revision.json", {"changed_at": datetime.now(timezone.utc).isoformat()})


def create_composition(video: Path, pose: Path, root: Path, recipe: dict,
                       name: str = "신체점 조합", *, expected_baseline: pd.DataFrame | None = None) -> Path:
    video, pose = video.resolve(), pose.resolve()
    source_stamp, video_stamp = file_stamp(pose), file_stamp(video)
    source_hash = sha256_file(pose)
    original = read_predictions(pose)
    if expected_baseline is not None and not original.equals(expected_baseline):
        raise ValueError(tr('현재 원본 추론과 화면의 보정 전 좌표가 다릅니다. 최신 결과를 다시 열어 저장하세요.'))
    composed, summary = compose_predictions(original, recipe)
    info = video_info(video)
    target = root.resolve() / source_key(video, pose)
    target.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="composition-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-", dir=target))
    original.to_hdf(run / "original.h5", key="df_with_missing", mode="w", format="fixed")
    composed.to_hdf(run / "composed.h5", key="df_with_missing", mode="w", format="fixed")
    composed.to_csv(run / "composed.csv")
    pd.testing.assert_frame_equal(pd.read_hdf(run / "original.h5"), original)
    pd.testing.assert_frame_equal(pd.read_hdf(run / "composed.h5"), composed)
    if file_stamp(pose) != source_stamp or sha256_file(pose) != source_hash or file_stamp(video) != video_stamp:
        raise ValueError(tr('저장 중 입력이 변경되었습니다. 추론 완료 후 다시 저장하세요.'))
    record = {"schema": COMPOSITION_SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
              "name": name.strip()[:120] or DEFAULT_RECIPE_NAME, "recipe": validate_recipe(recipe),
              "source_video": str(video), "source_pose": str(pose), "source_pose_sha256": source_hash,
              "source_pose_stamp": source_stamp, "video_stamp": video_stamp, "video_info": info,
              "original": "original.h5", "filtered": "composed.h5", "csv": "composed.csv",
              "original_sha256": sha256_file(run / "original.h5"),
              "filtered_sha256": sha256_file(run / "composed.h5"), "summary": summary}
    manifest = run / "composition.json"
    atomic_json(manifest, record)
    notify_compositions(root)
    return manifest


def list_compositions(root: Path, video: Path | None = None, pose: Path | None = None) -> list[dict]:
    target = root / source_key(video, pose) if video is not None and pose is not None else root
    rows = []
    for manifest in target.glob("*/composition.json" if target != root else "*/*/composition.json"):
        try:
            record = json.loads(manifest.read_text(encoding="utf-8"))
            if record.get("schema") != COMPOSITION_SCHEMA:
                continue
            validate_recipe(record["recipe"])
            if source_key(Path(record["source_video"]), Path(record["source_pose"])) != manifest.parent.parent.name:
                continue
            datetime.fromisoformat(record["created_at"])
            for key in ("original", "filtered"):
                member = (manifest.parent / record[key]).resolve()
                if member.parent != manifest.parent.resolve() or not member.is_file():
                    raise ValueError(tr('조합 좌표 파일 없음'))
            rows.append({**record, "manifest": str(manifest.resolve())})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(rows, key=lambda row: row["created_at"], reverse=True)


def load_composition(manifest: Path, *, require_current: bool = False):
    manifest = manifest.resolve()
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if record.get("schema") == "hagisilta-pose-filter-test-v1":
        record, original, composed = load_filter_test(manifest)
        if require_current and sha256_file(Path(record["source_pose"])) != record["source_pose_sha256"]:
            raise ValueError(tr('보정 후 원본 추론이 변경되었습니다. 다시 보정하세요.'))
        record = {**record, "recipe": {part: int(record["window"]) for part in
                                      original.columns.get_level_values(1)}}
        return record, original, composed
    if record.get("schema") != COMPOSITION_SCHEMA:
        raise ValueError(tr('지원하지 않는 보정 조합입니다.'))
    video, source = Path(record["source_video"]), Path(record["source_pose"])
    if file_stamp(video) != record["video_stamp"]:
        raise ValueError(tr('보정 조합 생성 후 영상이 변경되었습니다. 다시 보정하세요.'))
    if require_current and sha256_file(source) != record["source_pose_sha256"]:
        raise ValueError(tr('보정 조합 생성 후 원본 추론이 변경되었습니다. 분석 입력을 다시 지정하세요.'))
    frames = []
    for key in ("original", "filtered"):
        member = (manifest.parent / record[key]).resolve()
        if member.parent != manifest.parent or sha256_file(member) != record[f"{key}_sha256"]:
            raise ValueError(tr('보정 조합 좌표 파일이 손상되었습니다.'))
        frames.append(read_predictions(member))
    original, composed = frames
    validate_recipe(record["recipe"], original.columns.get_level_values(1))
    if not original.columns.equals(composed.columns) or not original.index.equals(composed.index):
        raise ValueError(tr('보정 조합의 프레임·신체점 구성이 다릅니다.'))
    expected, _summary = compose_predictions(original, record["recipe"])
    if not expected.equals(composed):
        raise ValueError(tr('저장된 신체점 조합과 실제 좌표가 일치하지 않습니다.'))
    return record, original, composed


def analysis_choice(video: Path) -> dict | None:
    path = video.resolve().parent / CHOICE_FILE
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") != "hagisilta-analysis-pose-v1":
        raise ValueError(tr('분석 좌표 선택 파일을 확인하세요.'))
    if Path(record["source_video"]).resolve() != video.resolve():
        raise ValueError(tr('분석 좌표 선택과 ROI 영상이 다릅니다.'))
    if file_stamp(video) != record["video_stamp"]:
        raise ValueError(tr('분석 입력 지정 후 영상이 변경되었습니다. 다시 지정하세요.'))
    source = Path(record["source_pose"])
    if file_stamp(source) != record.get("source_pose_stamp") and sha256_file(source) != record["source_pose_sha256"]:
        raise ValueError(tr('분석 입력 지정 후 추론이 변경되었습니다. 다시 지정하세요.'))
    return record


def choose_analysis_pose(video: Path, pose: Path, manifest: Path | None = None) -> None:
    video, pose = video.resolve(), pose.resolve()
    if manifest:
        record, _original, _composed = load_composition(manifest, require_current=True)
        if source_key(video, pose) != source_key(Path(record["source_video"]), Path(record["source_pose"])):
            raise ValueError(tr('다른 영상 또는 다른 모델의 보정 결과는 지정할 수 없습니다.'))
    else:
        read_predictions(pose)
    choice = {"schema": "hagisilta-analysis-pose-v1", "source_video": str(video), "source_pose": str(pose),
              "video_stamp": file_stamp(video), "source_pose_sha256": sha256_file(pose),
              "source_pose_stamp": file_stamp(pose),
              "composition": str(manifest.resolve()) if manifest else None,
              "selected_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(video.parent / CHOICE_FILE, choice)


def load_analysis_pose(arena_dir: Path):
    """All three experiment readers use this. Stale selection fails closed."""
    choice_file = arena_dir / CHOICE_FILE
    if not choice_file.is_file():
        return None
    record = json.loads(choice_file.read_text(encoding="utf-8"))
    video, source = Path(record["source_video"]), Path(record["source_pose"])
    if video.resolve().parent != arena_dir.resolve():
        raise ValueError(tr('분석 좌표 선택의 ROI 폴더가 일치하지 않습니다.'))
    analysis_choice(video)
    if file_stamp(video) != record["video_stamp"] or sha256_file(source) != record["source_pose_sha256"]:
        raise ValueError(tr('선택한 분석 좌표의 원본 영상·추론이 변경되었습니다. 트래킹에서 분석 입력을 다시 지정하세요.'))
    if record.get("composition"):
        manifest = Path(record["composition"])
        selected, _original, frame = load_composition(manifest, require_current=True)
        if source_key(video, source) != source_key(Path(selected["source_video"]), Path(selected["source_pose"])):
            raise ValueError(tr('선택한 분석 좌표가 다른 ROI·모델의 결과입니다.'))
        path = manifest.parent / selected["filtered"]
        provenance = {"composition_manifest": str(manifest), "recipe": selected["recipe"],
                      "source_pose": str(source), "source_pose_sha256": record["source_pose_sha256"]}
    else:
        path, frame = source, read_predictions(source)
        provenance = {"composition_manifest": None, "source_pose": str(source),
                      "source_pose_sha256": record["source_pose_sha256"]}
    provenance["source_video"] = str(video.resolve())
    frame.attrs["hagisilta_analysis_pose"] = provenance
    print(f"ANALYSIS_POSE={path}:recipe={json.dumps(provenance.get('recipe', {}), ensure_ascii=False)}", flush=True)
    return frame, path


def selected_analysis_video(arena_dir: Path):
    path = arena_dir / CHOICE_FILE
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    video = Path(record["source_video"]).resolve()
    if video.parent != arena_dir.resolve():
        raise ValueError(tr('분석 좌표 선택의 ROI 영상이 다른 폴더에 있습니다.'))
    analysis_choice(video)
    return video


def apply_analysis_batch(targets: list[dict]) -> dict:
    """Validate every ROI before writing; roll back choices on a write failure."""
    prepared, seen = [], set()
    for target in targets:
        video, pose = Path(target["video"]).resolve(), Path(target["pose"]).resolve()
        if video.parent in seen:
            raise ValueError(tr('같은 ROI의 결과를 여러 개 선택했습니다. ROI당 한 결과를 선택하세요.'))
        seen.add(video.parent)
        manifest = Path(target["manifest"]).resolve() if target.get("manifest") else None
        if manifest:
            record, _original, _filtered = load_composition(manifest, require_current=True)
            if source_key(video, pose) != source_key(Path(record["source_video"]), Path(record["source_pose"])):
                raise ValueError(tr('보정본의 영상·모델이 다릅니다: {v0}', v0=video.name))
        else:
            read_predictions(pose)
        choice = {"schema": "hagisilta-analysis-pose-v1", "source_video": str(video), "source_pose": str(pose),
                  "video_stamp": file_stamp(video), "source_pose_stamp": file_stamp(pose),
                  "source_pose_sha256": sha256_file(pose), "composition": str(manifest) if manifest else None,
                  "selected_at": datetime.now(timezone.utc).isoformat()}
        path = video.parent / CHOICE_FILE
        prepared.append((path, choice, path.read_bytes() if path.is_file() else None))
    written = []
    try:
        for path, choice, previous in prepared:
            atomic_json(path, choice)
            written.append((path, previous))
    except Exception:
        for path, previous in reversed(written):
            if previous is None:
                path.unlink(missing_ok=True)
            else:
                atomic_write_bytes(path, previous)
        raise
    return {"complete": True, "count": len(prepared)}


def filter_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HAGISILTA pose median filter")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--pose", type=Path)
    parser.add_argument("--batch-plan", type=Path)
    parser.add_argument("--batch-report", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--recipe-manifest", type=Path)
    args = parser.parse_args(argv)
    recipe, recipe_name = None, DEFAULT_RECIPE_NAME
    if args.recipe_manifest:
        recipe, recipe_name = read_recipe(args.recipe_manifest)
    if args.batch_plan:
        if not args.batch_report or args.video or args.pose:
            parser.error(tr('--batch-plan에는 --batch-report만 함께 지정하세요.'))
        report = create_filter_batch(args.batch_plan, args.batch_report, args.output_root, args.window, recipe, recipe_name)
        complete = sum(job["status"] == "complete" for job in report["jobs"])
        print(f"FILTER_BATCH_COMPLETE={complete}/{len(report['jobs'])}", flush=True)
        return 0 if complete == len(report["jobs"]) else 1
    if not args.video or not args.pose:
        parser.error(tr('--video와 --pose 또는 --batch-plan이 필요합니다.'))
    if recipe is None:
        manifest = create_filter_test(args.video, args.pose, args.output_root, args.window)
    else:
        manifest = create_composition(args.video, args.pose, args.output_root, recipe, recipe_name)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    description = tr('중앙값 {v0}프레임 · {v1}', v0=args.window, v1=window_description(args.window, record['video_info']['fps'])) if recipe is None else recipe_display_name(recipe_name)
    print(description, flush=True)
    print(tr('원본 대비 좌표 변경량 (정확도 아님): {v0}', v0=json.dumps(record['summary'], ensure_ascii=False)), flush=True)
    print(f"FILTER_TEST_MANIFEST={manifest}", flush=True)
    return 0


def apply_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply saved analysis inputs to selected ROIs")
    parser.add_argument("--analysis-plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = apply_analysis_batch(json.loads(args.analysis_plan.read_text(encoding="utf-8"))["targets"])
    except Exception as error:
        atomic_json(args.report, {"complete": False, "error": str(error)})
        raise
    atomic_json(args.report, result)
    print(f"ANALYSIS_INPUTS_APPLIED={result['count']}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HAGISILTA pose postprocessing")
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("filter", add_help=False, help="Create filtered pose copies or bodypart recipes")
    modes.add_parser("apply", add_help=False, help="Apply saved analysis inputs to selected ROIs")
    args, remaining = parser.parse_known_args(argv)
    return filter_main(remaining) if args.mode == "filter" else apply_main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
