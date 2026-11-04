# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from hagisilta_i18n import tr

import argparse
import hashlib
import json
import math
import re
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ruamel.yaml import YAML

from hagisilta_dataset_split import temporal_group_map
from hagisilta_runtime import (
    edit_project_config, project_backup_root, project_dataset_metadata, project_state_root,
)


def video_sample_fingerprint(video: Path, frame_count: int) -> str:
    """Return a small content fingerprint without hashing the entire video."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return ""
    digest = hashlib.sha256()
    positions = sorted({0, max(0, frame_count // 2), max(0, frame_count - 1)})
    try:
        for position in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = capture.read()
            if not ok:
                return ""
            digest.update(position.to_bytes(8, "little", signed=False))
            digest.update(frame.tobytes())
    finally:
        capture.release()
    return digest.hexdigest()


def video_source_identity(video: Path) -> dict:
    """Describe the actual ROI/video content behind a review session."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(tr('영상을 열 수 없습니다: {v0}', v0=video))
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = round(float(capture.get(cv2.CAP_PROP_FPS) or 0.0), 6)
    finally:
        capture.release()

    identity = {
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "fps": fps,
        "sample_sha256": video_sample_fingerprint(video, frame_count),
    }
    per_video_metadata = video.parent / f"{video.stem}.roi-metadata.json"
    metadata_path = per_video_metadata if per_video_metadata.is_file() else video.parent / "roi-metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
        roi_identity = {
            key: metadata.get(key)
            for key in (
                "source_video",
                "source_offset_px",
                "crop_size_px",
                "corners_source_px",
                "arena_id",
            )
            if metadata.get(key) is not None
        }
        if roi_identity:
            identity["roi"] = roi_identity
    return identity


def same_review_source(previous: dict, current: dict) -> bool:
    """Allow re-reviewing the same ROI, but reject a reused stem for another ROI."""
    previous_roi = previous.get("roi") or {}
    current_roi = current.get("roi") or {}
    if previous_roi and current_roi:
        comparable = set(previous_roi) & set(current_roi)
        strong_keys = comparable & {
            "source_offset_px", "crop_size_px", "corners_source_px", "source_video"
        }
        if strong_keys:
            return all(previous_roi[key] == current_roi[key] for key in strong_keys)
    dimensions_match = all(
        previous.get(key) == current.get(key)
        for key in ("width", "height", "frame_count")
        if previous.get(key) is not None and current.get(key) is not None
    )
    previous_hash = str(previous.get("sample_sha256") or "")
    current_hash = str(current.get("sample_sha256") or "")
    return dimensions_match and bool(previous_hash and previous_hash == current_hash)


def load_pose(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".h5", ".hdf5"}:
        frame = pd.read_hdf(path)
    else:
        frame = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    if not isinstance(frame.columns, pd.MultiIndex) or frame.columns.nlevels < 3:
        raise ValueError(tr('DLC MultiIndex 좌표 파일이 아닙니다.'))
    return frame


def pose_value(frame: pd.DataFrame, row: int, bodypart: str, coord: str) -> float:
    bodypart_level = frame.columns.nlevels - 2
    coord_level = frame.columns.nlevels - 1
    for column in frame.columns:
        if str(column[bodypart_level]) == bodypart and str(column[coord_level]) == coord:
            value = pd.to_numeric(pd.Series([frame.iloc[row][column]]), errors="coerce").iloc[0]
            return float(value) if pd.notna(value) else math.nan
    return math.nan


def read_session(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("video") or not payload.get("pose"):
        raise ValueError(tr('세션에 영상 또는 DLC 좌표 경로가 없습니다.'))
    confirmed = set(map(int, payload.get("confirmed", []) or []))
    episode_candidates = {
        int(frame)
        for episode in payload.get("temporal_episodes", []) or []
        for frame in (episode.get("gt_frames", []) if isinstance(episode, dict) else [])
    }
    if not payload.get("edits") and not (confirmed & episode_candidates):
        raise ValueError(tr('세션에 사람이 수정하거나 검증한 시계열 GT 프레임이 없습니다.'))
    return payload


def write_image(path: Path, frame: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError(tr('PNG 인코딩 실패: {v0}', v0=path.name))
    encoded.tofile(path)


def update_config(config: Path, video: Path, width: int, height: int, backup_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup_root.mkdir(parents=True, exist_ok=True)
    backup = backup_root / f"config-before-video-review-import-{stamp}.yaml"
    shutil.copy2(config, backup)

    yaml = YAML()
    yaml.preserve_quotes = True
    with config.open("r", encoding="utf-8") as stream:
        payload = yaml.load(stream)
    video_sets = payload.setdefault("video_sets", {})
    video_sets[str(video)] = {"crop": f"0, {width}, 0, {height}"}
    # Keep the video registered so DLC can associate labeled-data/<video stem>,
    # but do not offer it as a fresh random-frame extraction source by default.
    video_metadata = project_dataset_metadata(payload, create=True).setdefault("videos", {})
    metadata = video_metadata.setdefault(str(video), {})
    metadata.update(
        {
            "origin": "video_review_gt",
            "source_video": str(video),
            "extract_enabled": False,
        }
    )
    # Use the same validated, atomic writer as training. Default ruamel width
    # can turn a long Windows path containing spaces into an invalid YAML key.
    edit_project_config(config, payload)
    return backup


def import_review(config: Path, session_path: Path) -> dict:
    config = config.resolve()
    project = config.parent
    session = read_session(session_path)
    video = Path(session["video"]).resolve()
    pose_path = Path(session["pose"]).resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    if not pose_path.is_file():
        raise FileNotFoundError(pose_path)

    yaml = YAML(typ="safe")
    config_payload = yaml.load(config.read_text(encoding="utf-8")) or {}
    scorer = str(config_payload.get("scorer") or "HAGISILTA")
    bodyparts = list(session.get("bodyparts") or config_payload.get("bodyparts") or [])
    if not bodyparts:
        raise ValueError(tr('신체점 목록을 찾을 수 없습니다.'))

    edits = {int(frame): values for frame, values in (session.get("edits", {}) or {}).items()}
    excluded = set(map(int, session.get("excluded", [])))
    confirmed = set(map(int, session.get("confirmed", [])))
    temporal_episodes = [
        episode for episode in (session.get("temporal_episodes", []) or [])
        if isinstance(episode, dict)
    ]
    episode_candidates = {
        int(frame)
        for episode in temporal_episodes
        for frame in (episode.get("gt_frames", []) or [])
    }
    verified_episode_frames = confirmed & episode_candidates
    selected_frames = sorted((set(edits) | verified_episode_frames) - excluded)
    pose = load_pose(pose_path)
    from hagisilta_review_baseline import validate_frame_baselines
    validate_frame_baselines(session, bodyparts, len(pose))
    selected_frames = [frame for frame in selected_frames if 0 <= frame < len(pose)]
    if not selected_frames:
        raise ValueError(tr('내보낼 수 있는 수정 프레임이 없습니다.'))

    # DLC associates a configured video with labeled-data/<exact video stem>.
    # A descriptive suffix here would make create_training_dataset skip it.
    dataset_name = video.stem
    target = project / "labeled-data" / dataset_name
    manifest_root = project_state_root(project, create=True) / "video-review-imports"
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_root / f"{dataset_name}.json"
    previous_manifest = {}
    if manifest_path.is_file():
        try:
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_manifest = {}

    current_identity = video_source_identity(video)
    previous_identity = dict(previous_manifest.get("source_identity") or {})
    if target.exists() and previous_manifest and not previous_identity:
        previous_video = Path(str(previous_manifest.get("source_video") or ""))
        if previous_video.is_file():
            previous_identity = video_source_identity(previous_video)
    if target.exists() and previous_identity and not same_review_source(previous_identity, current_identity):
        raise ValueError(
            tr('같은 이름에 서로 다른 ROI 결과가 이미 등록되어 있습니다.\n기존 데이터셋: {v0}\n기존 원본: {v1}\n새 원본: {v2}\n기존 데이터는 변경하지 않았습니다. 최신 버전에서 실제 영상 테스트를 다시 실행하면 날짜·시간 실행 ID가 자동으로 붙어 별도 데이터셋으로 생성됩니다.', v0=dataset_name, v1=previous_manifest.get('source_video', '-'), v2=video)
        )
    staging = target.with_name(f"{target.name}.importing")
    if staging.exists():
        # A terminated import must never block the next attempt or appear as a
        # real dataset. Keep the partial files for diagnosis, then rebuild the
        # staging directory from the last committed dataset.
        backup_root = project_backup_root(project, create=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        stale_backup = backup_root / f"stale-video-review-import-{dataset_name}-{stamp}"
        suffix = 1
        while stale_backup.exists():
            stale_backup = backup_root / f"stale-video-review-import-{dataset_name}-{stamp}-{suffix}"
            suffix += 1
        staging.rename(stale_backup)
    if target.exists():
        shutil.copytree(target, staging)
    else:
        staging.mkdir(parents=True)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(tr('영상을 열 수 없습니다: {v0}', v0=video))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(session.get("fps") or capture.get(cv2.CAP_PROP_FPS) or 0.0)

    rows: list[list[float]] = []
    indices: list[tuple[str, str, str]] = []
    try:
        for order, frame_index in enumerate(selected_frames, 1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(tr('프레임 읽기 실패: {v0}', v0=frame_index))
            filename = f"img{frame_index:06d}.png"
            write_image(staging / filename, image)

            values: list[float] = []
            frame_edits = edits.get(frame_index, {})
            for bodypart in bodyparts:
                baseline = (session.get("frame_baselines", {}).get(str(frame_index), {}).get("points", {}))
                edit = frame_edits.get(bodypart, baseline.get(bodypart))
                if edit is not None:
                    visible = bool(edit.get("visible", False))
                    x = float(edit.get("x", math.nan)) if visible else math.nan
                    y = float(edit.get("y", math.nan)) if visible else math.nan
                else:
                    x = pose_value(pose, frame_index, bodypart, "x")
                    y = pose_value(pose, frame_index, bodypart, "y")
                    # Model predictions can land a fraction of a pixel outside
                    # the image at an edge. They are reference coordinates for
                    # an otherwise edited frame, so clamp them to the nearest
                    # valid pixel instead of aborting the whole import.
                    if np.isfinite(x):
                        x = min(max(x, 0.0), float(width - 1))
                    if np.isfinite(y):
                        y = min(max(y, 0.0), float(height - 1))
                if edit is not None and np.isfinite(x) and not 0 <= x < width:
                    raise ValueError(tr('프레임 {v0} {v1} x 좌표가 영상 밖입니다: {v2}', v0=frame_index, v1=bodypart, v2=x))
                if edit is not None and np.isfinite(y) and not 0 <= y < height:
                    raise ValueError(tr('프레임 {v0} {v1} y 좌표가 영상 밖입니다: {v2}', v0=frame_index, v1=bodypart, v2=y))
                values.extend([x, y])
            rows.append(values)
            indices.append(("labeled-data", dataset_name, filename))
            if order % 100 == 0 or order == len(selected_frames):
                print(f"EXPORT_FRAMES={order}/{len(selected_frames)}", flush=True)
    finally:
        capture.release()

    columns = pd.MultiIndex.from_tuples(
        [(scorer, bodypart, coord) for bodypart in bodyparts for coord in ("x", "y")],
        names=["scorer", "bodyparts", "coords"],
    )
    index = pd.MultiIndex.from_tuples(indices)
    new_labels = pd.DataFrame(rows, index=index, columns=columns, dtype=float)
    csv_path = staging / f"CollectedData_{scorer}.csv"
    h5_path = staging / f"CollectedData_{scorer}.h5"
    existing_h5 = target / f"CollectedData_{scorer}.h5"
    if existing_h5.is_file():
        existing_labels = pd.read_hdf(existing_h5)
        existing_labels = existing_labels.loc[~existing_labels.index.isin(new_labels.index)]
        labels = pd.concat([existing_labels, new_labels]).sort_index()
    else:
        labels = new_labels.sort_index()
    labels.to_csv(csv_path)
    labels.to_hdf(h5_path, key="df_with_missing", mode="w")

    reloaded = pd.read_hdf(h5_path)
    image_count = len(list(staging.glob("img*.png")))
    missing_images = [Path(*index_value) for index_value in labels.index if not (project / Path(*index_value)).is_file()]
    # During staging, index paths still point at the final target. Check names
    # against staging as well before the atomic folder swap.
    missing_images = [path for path in missing_images if not (staging / path.name).is_file()]
    if len(reloaded) != len(labels) or image_count < len(labels) or missing_images:
        raise RuntimeError(tr('내보낸 이미지와 라벨 개수 검증에 실패했습니다.'))
    backup_root = project_backup_root(project, create=True)
    label_backup = None
    if target.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        label_backup = backup_root / f"labeled-data-{dataset_name}-before-review-{stamp}"
        target.rename(label_backup)
    try:
        staging.rename(target)
    except Exception:
        if label_backup and label_backup.exists() and not target.exists():
            label_backup.rename(target)
        raise
    csv_path = target / csv_path.name
    h5_path = target / h5_path.name

    backup = update_config(config, video, width, height, backup_root)
    current_frame_names = [f"img{frame:06d}.png" for frame in selected_frames]
    review_frame_names = sorted(
        set(map(str, previous_manifest.get("frame_names", []) or [])) | set(current_frame_names)
    )
    frame_groups = temporal_group_map(review_frame_names, fps)
    previous_frame_groups = {
        str(name): str(group)
        for name, group in (previous_manifest.get("frame_groups", {}) or {}).items()
        if str(name) in review_frame_names and str(group)
    }
    frame_groups.update(previous_frame_groups)
    current_explicit_episodes = []
    for episode in temporal_episodes:
        episode_id = str(episode.get("episode_id") or "")
        if not episode_id:
            continue
        imported_gt_frames = sorted({
            int(frame) for frame in (episode.get("gt_frames", []) or [])
            if int(frame) in selected_frames
        })
        start_frame = int(episode.get("start_frame", -1))
        end_frame = int(episode.get("end_frame", -1))
        imported_episode_frames = sorted({
            frame for frame in selected_frames if start_frame <= frame <= end_frame
        })
        if not imported_episode_frames:
            continue
        for frame in imported_episode_frames:
            frame_groups[f"img{frame:06d}.png"] = episode_id
        current_explicit_episodes.append({
            **episode,
            "episode_id": episode_id,
            "imported_gt_frames": imported_gt_frames,
            "imported_episode_frames": imported_episode_frames,
            "session": str(session_path.resolve()),
        })
    explicit_episode_records = {
        (
            str(episode.get("episode_id") or ""),
            int(episode.get("start_frame", -1)),
            int(episode.get("end_frame", -1)),
        ): episode
        for episode in (previous_manifest.get("explicit_temporal_episodes", []) or [])
        if isinstance(episode, dict)
    }
    for episode in current_explicit_episodes:
        explicit_episode_records[
            (
                str(episode["episode_id"]),
                int(episode.get("start_frame", -1)),
                int(episode.get("end_frame", -1)),
            )
        ] = episode
    temporal_groups: dict[str, list[str]] = {}
    for frame_name, group_id in frame_groups.items():
        temporal_groups.setdefault(group_id, []).append(frame_name)
    review_sessions = list(dict.fromkeys([
        *map(str, previous_manifest.get("review_sessions", []) or []),
        str(session_path.resolve()),
    ]))
    manifest = {
        "imported_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(config),
        "config_backup": str(backup),
        "session": str(session_path.resolve()),
        "source_video": str(video),
        "source_identity": current_identity,
        "source_pose": str(pose_path),
        "fps": fps,
        "dataset": dataset_name,
        "target": str(target),
        "frames": len(new_labels),
        "total_frames": len(labels),
        "frame_names": review_frame_names,
        "source_frame_indices": sorted({
            frame for name in review_frame_names
            if (match := re.search(r"(\d+)(?=\.[^.]+$)", name))
            for frame in [int(match.group(1))]
        }),
        "frame_groups": frame_groups,
        "temporal_groups": temporal_groups,
        "temporal_group_count": len(temporal_groups),
        "explicit_temporal_episodes": list(explicit_episode_records.values()),
        "explicit_temporal_episode_count": len(explicit_episode_records),
        "verified_episode_frames": sorted(verified_episode_frames - excluded),
        "review_sessions": review_sessions,
        "review_frame_baselines": {str(frame): session["frame_baselines"][str(frame)]
                                   for frame in selected_frames if str(frame) in session.get("frame_baselines", {})},
        "label_backup": str(label_backup) if label_backup else None,
        "bodyparts": bodyparts,
        "scorer": scorer,
        "policy": (
            "all human-edited non-excluded frames plus explicitly confirmed temporal-episode GT; "
            "unedited points retain reviewed per-frame baseline coordinates (raw or postprocessed)"
        ),
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_manifest.replace(manifest_path)

    return {**manifest, "csv": str(csv_path), "h5": str(h5_path), "manifest": str(manifest_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Import HAGISILTA video-review edits into a DLC project")
    parser.add_argument("config", type=Path)
    parser.add_argument("session", type=Path)
    args = parser.parse_args()
    result = import_review(args.config, args.session)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
