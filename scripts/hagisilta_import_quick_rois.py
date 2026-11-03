# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from hagisilta_i18n import tr

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from hagisilta_runtime import project_backup_root, project_dataset_metadata
from hagisilta_tracking_crop import TRACKING_CROP_CODEC, write_tracking_crop_video


def main() -> int:
    parser = argparse.ArgumentParser(description="Import HAGISILTA quick-inference ROI videos into a DLC dataset.")
    parser.add_argument("quick_output", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--default-frames", type=int, default=25)
    parser.add_argument("--source-video", type=Path)
    parser.add_argument("--roi-profile", type=Path)
    args = parser.parse_args()
    quick_output = args.quick_output.resolve()
    config = args.config.resolve()
    payload = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    project = Path(payload.get("project_path", config.parent))
    video_dir = project / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_sets = payload.setdefault("video_sets", {})
    dataset = project_dataset_metadata(payload, create=True)
    provenance = dataset.setdefault("videos", {})
    imported = []

    metadata_records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(quick_output.glob("arena_*/roi-metadata.json"))
    ]
    if not metadata_records:
        if not args.source_video or not args.roi_profile:
            raise FileNotFoundError(
                tr('ROI 추론 메타데이터가 없으며 원본 영상/ROI 프로필도 전달되지 않았습니다.')
            )
        source = args.source_video.resolve()
        profile_path = args.roi_profile.resolve()
        if not source.is_file():
            raise FileNotFoundError(tr('ROI 원본 영상이 없습니다: {v0}', v0=source))
        if not profile_path.is_file():
            raise FileNotFoundError(tr('ROI 프로필이 없습니다: {v0}', v0=profile_path))
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        if profile.get("profile_type") not in {
            "nort_arena_roi", "nort_camera_calibration", "hagisilta_zone_template"
        }:
            raise ValueError(tr('지원하는 HAGISILTA ROI/카메라 프로필이 아닙니다.'))
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(tr('원본 영상을 열 수 없습니다: {v0}', v0=source))
        actual_resolution = [
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ]
        capture.release()
        expected_resolution = [int(value) for value in profile.get("source_resolution_px", [])]
        if expected_resolution != actual_resolution:
            raise ValueError(
                tr('ROI 해상도 {v0}와 영상 해상도 {v1}가 다릅니다.', v0=expected_resolution, v1=actual_resolution)
            )
        for index, arena in enumerate(profile.get("arenas", []), start=1):
            corners = arena.get("mask_polygon_source_px") or arena.get("corners_source_px") or []
            polygon = np.asarray(corners, dtype=np.float32)
            if (
                polygon.ndim != 2 or polygon.shape[1:] != (2,)
                or len(polygon) < 3 or not np.isfinite(polygon).all()
                or abs(cv2.contourArea(polygon)) < 25
            ):
                raise ValueError(tr('추적 영역 {v0} ROI가 올바르지 않습니다.', v0=index))
            metadata_records.append({
                "arena_id": int(arena.get("id", index)),
                "source_video": str(source),
                "roi_profile": str(profile_path),
                "corners_source_px": corners,
            })

    for metadata in metadata_records:
        arena_id = int(metadata.get("arena_id", len(imported) + 1))
        source = Path(metadata["source_video"]).resolve()
        if not source.is_file():
            raise FileNotFoundError(tr('ROI 원본 영상이 없습니다: {v0}', v0=source))
        destination = video_dir / f"{source.stem}_roi02_arena_{arena_id:02d}.avi"
        preserved = Path(metadata.get("roi_video", "")) if metadata.get("roi_video") else None
        preserved_is_lossless = (
            bool(metadata.get("lossless"))
            and str(metadata.get("codec", "")).upper() == TRACKING_CROP_CODEC
        )
        if preserved and preserved.is_file() and preserved_is_lossless:
            shutil.copy2(preserved, destination)
            capture = cv2.VideoCapture(str(destination))
            info = {
                "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": float(capture.get(cv2.CAP_PROP_FPS)) or 10.0,
                "crop_size_px": [
                    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                ],
                "source_offset_px": metadata.get("source_offset_px", [0, 0]),
                "source_crop_size_px": metadata.get("source_crop_size_px", metadata.get("crop_size_px", [])),
                "padding_px": metadata.get("padding_px", {"left": 0, "top": 0, "right": 0, "bottom": 0}),
                "codec": TRACKING_CROP_CODEC,
                "container": "avi",
                "lossless": True,
                "mask_mode": metadata.get("mask_mode", "polygon_outside_median_background"),
            }
            capture.release()
        else:
            info = write_tracking_crop_video(
                source,
                metadata["corners_source_px"],
                destination,
                progress=lambda done, _total: print(f"ROI_EXPORT_FRAMES={done}", flush=True),
            )
        video_sets[str(destination)] = {
            "crop": f"0, {info['crop_size_px'][0]}, 0, {info['crop_size_px'][1]}"
        }
        source_metadata = dict(provenance.get(str(source), {}) or {})
        group_enabled = bool(source_metadata.get("dataset_group_enabled", False))
        provenance[str(destination)] = {
            "origin": "quick_roi_02",
            "source_video": str(source),
            "arena_id": arena_id,
            "roi_profile": metadata.get("roi_profile"),
            "corners_source_px": metadata.get("corners_source_px"),
            "frames_to_pick": int(args.default_frames),
            # ROI 등록과 라벨 프레임 추출은 별도 작업이다. 사용자가
            # 데이터셋 표에서 명시적으로 켜기 전에는 추출 대상으로 두지 않는다.
            # If frame extraction requested materialization, the source row has
            # already persisted dataset_group_enabled=True.  Carry that choice
            # into every generated Arena so the recursive extraction pass does
            # not silently omit the ROI clips.  A plain ROI registration still
            # defaults to disabled until the user enables the source group.
            "extract_enabled": group_enabled,
            "frames": int(info["frames"]),
            "fps": float(info["fps"]),
            "crop_size_px": info["crop_size_px"],
            "source_offset_px": info.get("source_offset_px"),
            "source_crop_size_px": info.get("source_crop_size_px"),
            "padding_px": info.get("padding_px"),
            "codec": info.get("codec"),
            "container": info.get("container"),
            "lossless": bool(info.get("lossless")),
            "mask_mode": info.get("mask_mode"),
        }
        imported.append(str(destination))
        print(f"IMPORTED_ARENA={arena_id}:{destination}", flush=True)

    if not imported:
        raise FileNotFoundError(tr('가져올 ROI 박스가 없습니다.'))
    backup_dir = project_backup_root(config.parent, create=True)
    backup = backup_dir / f"config-before-roi-import-{time.strftime('%Y%m%d-%H%M%S')}.yaml"
    shutil.copy2(config, backup)
    config.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(json.dumps({"imported": imported, "backup": str(backup)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
