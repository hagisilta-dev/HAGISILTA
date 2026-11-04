# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from hagisilta_i18n import tr

import argparse
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import deeplabcut
import numpy as np
from deeplabcut.modelzoo.video_inference import load_super_animal_config
from ruamel.yaml import YAML

from hagisilta_runtime import detect_device
from hagisilta_pose_model_catalog import TOPVIEWMOUSE_POSE_MODEL_IDS
from hagisilta_tracking_crop import write_tracking_crop_video
from hagisilta_labeled_video import infer_superanimal_with_lossless_video


def new_generation_id() -> str:
    """Return a sortable ID that is unique for a newly generated review run."""
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def generated_roi_stem(source_stem: str, arena_id: int, generation_id: str) -> str:
    return f"{source_stem}_run{generation_id}_arena_{arena_id:02d}_roi"


def emit_event(kind: str, **payload) -> None:
    print("HAGISILTA_EVENT " + json.dumps({"kind": kind, **payload}, ensure_ascii=False), flush=True)


def load_rois(profile: Path, frame_size: tuple[int, int]) -> list[dict]:
    payload = json.loads(profile.read_text(encoding="utf-8"))
    if payload.get("profile_type") not in {
        "nort_arena_roi", "nort_camera_calibration", "hagisilta_zone_template"
    }:
        raise ValueError(tr('HAGISILTA ROI/구역 템플릿이 아닙니다.'))
    expected = tuple(int(value) for value in payload["source_resolution_px"])
    if expected != frame_size:
        raise ValueError(
            tr('ROI 해상도 {v0}x{v1}와 영상 {v2}x{v3}가 다릅니다.', v0=expected[0], v1=expected[1], v2=frame_size[0], v3=frame_size[1])
        )
    rois = []
    for index, arena in enumerate(payload.get("arenas", []), start=1):
        corners = np.asarray(
            arena.get("mask_polygon_source_px") or arena.get("corners_source_px") or [],
            dtype=np.float32,
        )
        if corners.ndim != 2 or corners.shape[1:] != (2,) or len(corners) < 3 or abs(cv2.contourArea(corners)) < 25:
            raise ValueError(tr('영역 {v0} ROI가 올바르지 않습니다.', v0=index))
        rois.append({"id": int(arena.get("id", index)), "corners": corners})
    if not rois:
        raise ValueError(tr('ROI 템플릿에 추적 영역이 없습니다.'))
    return rois


def infer(video: Path, output: Path, args: argparse.Namespace) -> None:
    model_config = load_super_animal_config(
        "superanimal_topviewmouse", args.pose_model, args.detector
    )
    model_config["detector"]["model"]["box_score_thresh"] = float(args.detector_threshold)
    with tempfile.TemporaryDirectory(prefix="hagisilta-detector-config-") as config_dir:
        config_path = Path(config_dir) / "pytorch_config.yaml"
        with config_path.open("w", encoding="utf-8") as stream:
            YAML().dump(model_config, stream)
        infer_superanimal_with_lossless_video(
            videos=[str(video)],
            superanimal_name="superanimal_topviewmouse",
            model_name=args.pose_model,
            detector_name=args.detector,
            dest_folder=str(output),
            video_adapt=args.adapt,
            plot_trajectories=args.plot_trajectories,
            batch_size=args.batch_size,
            detector_batch_size=args.detector_batch_size,
            max_individuals=1,
            device=detect_device(),
            pcutoff=args.pcutoff,
            pseudo_threshold=args.pseudo_threshold,
            bbox_threshold=args.bbox_threshold,
            detector_epochs=args.detector_epochs,
            pose_epochs=args.pose_epochs,
            video_adapt_batch_size=args.adapt_batch_size,
            customized_model_config=str(config_path),
            create_labeled_video=not args.no_labeled_video,
            plot_bboxes=not args.no_bboxes,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run zero-shot TopViewMouse inference on one pilot video.")
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--adapt", action="store_true", help="Enable self-supervised video adaptation")
    parser.add_argument("--roi-profile", type=Path)
    parser.add_argument("--arena", type=int, help="Only infer one 1-based arena")
    parser.add_argument("--detector", default="fasterrcnn_resnet50_fpn_v2")
    parser.add_argument(
        "--pose-model",
        choices=TOPVIEWMOUSE_POSE_MODEL_IDS,
        default="hrnet_w32",
        help="Pose architecture with an official TopViewMouse checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--detector-batch-size", type=int, default=1)
    parser.add_argument("--detector-threshold", type=float, default=0.6)
    parser.add_argument("--pcutoff", type=float, default=0.1)
    parser.add_argument("--plot-trajectories", action="store_true")
    parser.add_argument("--no-labeled-video", action="store_true")
    parser.add_argument("--no-bboxes", action="store_true")
    parser.add_argument("--pseudo-threshold", type=float, default=0.1)
    parser.add_argument("--bbox-threshold", type=float, default=0.9)
    parser.add_argument("--detector-epochs", type=int, default=4)
    parser.add_argument("--pose-epochs", type=int, default=4)
    parser.add_argument("--adapt-batch-size", type=int, default=8)
    args = parser.parse_args()

    video = args.video.resolve()
    output = args.output.resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    output.mkdir(parents=True, exist_ok=True)

    if not args.roi_profile:
        frame_counter = cv2.VideoCapture(str(video))
        total_frames = int(frame_counter.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_counter.release()
        emit_event("job_start", total_arenas=1, total_frames=total_frames)
        emit_event("inference_start", arena=1, position=1, total_arenas=1)
        started = time.perf_counter()
        infer(video, output, args)
        emit_event("arena_done", arena=1, position=1, total_arenas=1, elapsed_seconds=time.perf_counter() - started)
        emit_event("job_done", total_arenas=1)
        return 0

    capture = cv2.VideoCapture(str(video))
    frame_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.release()
    rois = load_rois(args.roi_profile.resolve(), frame_size)
    if args.arena:
        rois = [roi for roi in rois if roi["id"] == args.arena]
        if not rois:
            raise ValueError(tr('ROI 템플릿에 박스 {v0}가 없습니다.', v0=args.arena))
    frame_counter = cv2.VideoCapture(str(video))
    total_frames = int(frame_counter.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_counter.release()
    emit_event("job_start", total_arenas=len(rois), total_frames=total_frames)
    generation_id = new_generation_id()
    with tempfile.TemporaryDirectory(prefix="nort-roi-") as temp_name:
        temp = Path(temp_name)
        for position, roi in enumerate(rois, start=1):
            arena_started = time.perf_counter()
            arena_id = roi["id"]
            arena_output = output / f"arena_{arena_id:02d}"
            arena_output.mkdir(parents=True, exist_ok=True)
            cropped = arena_output / f"{generated_roi_stem(video.stem, arena_id, generation_id)}.avi"
            print(f"ROI {position}/{len(rois)}: arena {arena_id} crop", flush=True)
            emit_event("crop_start", arena=arena_id, position=position, total_arenas=len(rois))
            metadata = write_tracking_crop_video(
                video,
                roi["corners"],
                cropped,
                lambda done, total, aid=arena_id, pos=position: emit_event(
                    "crop_progress", arena=aid, position=pos, total_arenas=len(rois), done=done, total=total
                ),
            )
            metadata.update({
                "arena_id": arena_id,
                "source_video": str(video),
                "roi_video": str(cropped),
                "roi_profile": str(args.roi_profile.resolve()),
                "corners_source_px": roi["corners"].tolist(),
                "generation_id": generation_id,
            })
            (arena_output / f"{cropped.stem}.roi-metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (arena_output / "roi-metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"ROI {position}/{len(rois)}: arena {arena_id} inference", flush=True)
            emit_event("inference_start", arena=arena_id, position=position, total_arenas=len(rois))
            infer(cropped, arena_output, args)
            emit_event(
                "arena_done", arena=arena_id, position=position, total_arenas=len(rois),
                elapsed_seconds=time.perf_counter() - arena_started,
            )
    print(f"ROI_INFERENCE_COMPLETE={len(rois)}", flush=True)
    emit_event("job_done", total_arenas=len(rois))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
