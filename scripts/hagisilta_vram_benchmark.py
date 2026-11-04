# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from hagisilta_i18n import tr

import argparse
import gc
import json
import tempfile
import time
from pathlib import Path

import cv2
import deeplabcut
import numpy as np
import torch

from hagisilta_runtime import detect_device
from hagisilta_pose_model_catalog import TOPVIEWMOUSE_POSE_MODEL_IDS
from hagisilta_tracking_crop import write_tracking_crop_video
from hagisilta_superanimal_inference import load_rois


CANDIDATES = ((1, 1), (4, 2), (8, 4), (16, 8))


def emit(kind: str, **payload) -> None:
    print("HAGISILTA_EVENT " + json.dumps({"kind": kind, **payload}, ensure_ascii=False), flush=True)


def largest_roi(rois: list[dict]) -> dict:
    def area(roi: dict) -> int:
        _x, _y, width, height = cv2.boundingRect(roi["corners"].astype("int32"))
        return int(width * height)

    return max(rois, key=area)


def run_candidate(
    video: Path,
    output: Path,
    detector: str,
    pose_model: str,
    pose_batch: int,
    detector_batch: int,
) -> dict:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = deeplabcut.video_inference_superanimal(
        videos=[str(video)],
        superanimal_name="superanimal_topviewmouse",
        model_name=pose_model,
        detector_name=detector,
        dest_folder=str(output),
        video_adapt=False,
        plot_trajectories=False,
        batch_size=pose_batch,
        detector_batch_size=detector_batch,
        pcutoff=0.3,
        max_individuals=1,
        device="cuda:0",
        plot_bboxes=False,
        create_labeled_video=False,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
    del result
    return {
        "pose_batch": pose_batch,
        "detector_batch": detector_batch,
        "peak_allocated_gb": round(peak_allocated, 3),
        "peak_reserved_gb": round(peak_reserved, 3),
        "free_before_gb": round(free_bytes / 1024**3, 3),
        "total_vram_gb": round(total_bytes / 1024**3, 3),
        "elapsed_seconds": round(elapsed, 2),
        "status": "ok",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Short empirical VRAM benchmark for HAGISILTA SuperAnimal inference")
    parser.add_argument("video", type=Path)
    parser.add_argument("roi_profile", type=Path)
    parser.add_argument("--detector", default="fasterrcnn_resnet50_fpn_v2")
    parser.add_argument(
        "--pose-model",
        choices=TOPVIEWMOUSE_POSE_MODEL_IDS,
        default="hrnet_w32",
    )
    parser.add_argument("--frames", type=int, default=96)
    args = parser.parse_args()
    if detect_device() != "cuda:0":
        raise RuntimeError(tr('VRAM 자동 테스트에는 NVIDIA CUDA GPU가 필요합니다.'))
    video = args.video.resolve()
    profile = args.roi_profile.resolve()
    capture = cv2.VideoCapture(str(video))
    frame_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    total_frames = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    capture.release()
    rois = load_rois(profile, frame_size)
    roi = largest_roi(rois)
    free_start, total_bytes = torch.cuda.mem_get_info()
    safe_limit_gb = free_start / 1024**3 * 0.75
    emit(
        "benchmark_start", arena=roi["id"], frames=args.frames,
        candidates=len(CANDIDATES), safe_limit_gb=round(safe_limit_gb, 2),
        total_vram_gb=round(total_bytes / 1024**3, 2),
    )
    results: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="nort-vram-benchmark-", ignore_cleanup_errors=True) as temp_name:
        temp = Path(temp_name)
        clip = temp / f"arena_{roi['id']:02d}_{args.frames}f.avi"
        indices = np.unique(
            np.linspace(0, total_frames - 1, min(args.frames, total_frames), dtype=np.int64)
        )
        clip_info = write_tracking_crop_video(
            video, roi["corners"], clip, frame_indices=indices
        )
        if int(clip_info["frames"]) < 8:
            raise RuntimeError(
                tr('VRAM 테스트용 프레임을 충분히 만들지 못했습니다: {v0}개', v0=clip_info['frames'])
            )
        emit("benchmark_clip_ready", arena=roi["id"], **clip_info)
        for position, (pose_batch, detector_batch) in enumerate(CANDIDATES, start=1):
            emit(
                "benchmark_candidate_start", position=position, candidates=len(CANDIDATES),
                pose_batch=pose_batch, detector_batch=detector_batch,
            )
            destination = temp / f"p{pose_batch}_d{detector_batch}"
            destination.mkdir(parents=True, exist_ok=True)
            try:
                result = run_candidate(
                    clip,
                    destination,
                    args.detector,
                    args.pose_model,
                    pose_batch,
                    detector_batch,
                )
            except Exception as error:
                message = str(error)
                if "need at least one array to stack" in message.lower():
                    result = {
                        "pose_batch": pose_batch, "detector_batch": detector_batch,
                        "status": "no_detections",
                        "error": tr('샘플 프레임에서 마우스를 검출하지 못했습니다.'),
                    }
                elif "out of memory" not in message.lower():
                    raise
                else:
                    result = {
                        "pose_batch": pose_batch, "detector_batch": detector_batch,
                        "status": "oom", "error": message,
                    }
                gc.collect()
                torch.cuda.empty_cache()
            results.append(result)
            emit("benchmark_candidate_result", position=position, candidates=len(CANDIDATES), **result)
            if result["status"] in {"oom", "no_detections"}:
                break
    safe = [
        result for result in results
        if result["status"] == "ok" and result["peak_reserved_gb"] <= safe_limit_gb
    ]
    recommendation = safe[-1] if safe else next((result for result in results if result["status"] == "ok"), None)
    if recommendation is None:
        if any(result["status"] == "no_detections" for result in results):
            raise RuntimeError(
                tr('VRAM 테스트 샘플에서 마우스를 검출하지 못했습니다. 마우스가 보이는 ROI인지 확인하거나 다른 시험 영상을 선택하세요.')
            )
        raise RuntimeError(tr('모든 Batch 후보에서 GPU 메모리 테스트가 실패했습니다.'))
    payload = {
        "recommended_pose": recommendation["pose_batch"],
        "recommended_detector": recommendation["detector_batch"],
        "recommended_peak_gb": recommendation["peak_reserved_gb"],
        "safe_limit_gb": round(safe_limit_gb, 2),
        "arena": roi["id"],
        "frames": args.frames,
        "results": results,
    }
    emit("benchmark_done", **payload)
    print("VRAM_BENCHMARK=" + json.dumps(payload, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
