# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from hagisilta_i18n import tr

import argparse
import json
import math
import os
import shutil
import threading
import time
from concurrent.futures import CancelledError, FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import cv2
import deeplabcut
import yaml
from deeplabcut.generate_training_dataset.metadata import TrainingDatasetMetadata
from deeplabcut.utils.auxiliaryfunctions import read_config

from hagisilta_runtime import detect_device, edit_project_config
from hagisilta_superanimal_inference import generated_roi_stem, load_rois, new_generation_id
from hagisilta_tracking_crop import write_tracking_crop_video, write_tracking_crop_videos
from hagisilta_source_identity import normalized_source_path, validate_unique_source_stems
from hagisilta_tracking_progress import TrackingProgress, inference_events, notify, path_key
from hagisilta_labeled_video import (
    create_lossless_labeled_videos, infer_superanimal_with_lossless_video, is_lossless_labeled_video,
)
from hagisilta_pose_model_catalog import (
    TOPVIEWMOUSE_POSE_MODEL_IDS,
    TOPVIEWMOUSE_PRETRAINED_DETECTOR_MODEL_IDS,
)
from hagisilta_temporal_refiner import (
    read_pose_table,
    refine_pose_table,
    temporal_checkpoint_path,
    write_pose_table,
    write_refinement_report,
)


MIB = 1024 * 1024
TEMPORAL_COMPARISON_DIRECTORY = "comparison-results"

TRAINED_INFERENCE_OVERRIDE_KEYS = {
    "detector_snapshot_index", "detector_batch_size", "dynamic", "ctd_conditions",
    "ctd_tracking", "top_down_dynamic", "use_shelve", "robust_nframes",
    "auto_track", "n_tracks", "animal_names", "calibrate", "identity_only",
    "cropping", "save_as_df", "show_gpu_memory", "inference_cfg",
}
SUPERANIMAL_INFERENCE_OVERRIDE_KEYS = {
    "scale_list", "cropping", "video_adapt", "plot_trajectories",
    "detector_batch_size", "adapt_iterations", "pseudo_threshold", "bbox_threshold",
    "detector_epochs", "pose_epochs", "max_individuals", "video_adapt_batch_size",
    "customized_pose_checkpoint", "customized_detector_checkpoint",
    "customized_model_config", "plot_bboxes", "fmpose_return_3d",
}


class VideoVramMonitor:
    """Sample CUDA memory and split one analyze_videos call by completed video."""

    def __init__(self, jobs: list[dict]) -> None:
        self.jobs = list(jobs)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._torch = None
        self._enabled = False
        self._index = 0
        self._peak_sample: dict[str, float] = {}
        self._file_state: list[dict[Path, int]] = [self._h5_state(job) for job in self.jobs]
        try:
            import torch

            if torch.cuda.is_available():
                self._torch = torch
                self._enabled = True
        except Exception:
            self._torch = None

    @staticmethod
    def _h5_state(job: dict) -> dict[Path, int]:
        state: dict[Path, int] = {}
        for path in Path(job["arena_output"]).glob("*.h5"):
            try:
                state[path.resolve()] = path.stat().st_mtime_ns
            except OSError:
                pass
        return state

    def _sample(self) -> dict[str, float]:
        if not self._enabled or self._torch is None:
            return {}
        try:
            free, total = self._torch.cuda.mem_get_info()
            return {
                "allocated": self._torch.cuda.memory_allocated() / MIB,
                "reserved": self._torch.cuda.memory_reserved() / MIB,
                "device": (total - free) / MIB,
            }
        except Exception:
            return {}

    @staticmethod
    def _format(sample: dict[str, float]) -> str:
        if not sample:
            return "unavailable"
        return (
            f"allocated={sample['allocated']:.0f}MiB:"
            f"reserved={sample['reserved']:.0f}MiB:"
            f"device={sample['device']:.0f}MiB"
        )

    def _video_label(self, index: int) -> str:
        job = self.jobs[index]
        return f"{job['source'].name}:arena {job['arena_id']}"

    def _print_start(self, index: int, sample: dict[str, float]) -> None:
        print(
            f"VRAM_VIDEO_START={index + 1}/{len(self.jobs)}:"
            f"{self._video_label(index)}:{self._format(sample)}",
            flush=True,
        )

    def _print_finish(self, index: int, end_sample: dict[str, float]) -> None:
        print(
            f"VRAM_VIDEO_PEAK={index + 1}/{len(self.jobs)}:"
            f"{self._video_label(index)}:{self._format(self._peak_sample)}",
            flush=True,
        )
        print(
            f"VRAM_VIDEO_END={index + 1}/{len(self.jobs)}:"
            f"{self._video_label(index)}:{self._format(end_sample)}",
            flush=True,
        )

    def _result_written(self, index: int) -> bool:
        before = self._file_state[index]
        current = self._h5_state(self.jobs[index])
        return bool(current) and any(
            path not in before or mtime > before[path] for path, mtime in current.items()
        )

    def _run(self) -> None:
        if not self.jobs:
            return
        sample = self._sample()
        self._peak_sample = dict(sample)
        self._print_start(0, sample)
        while not self._stop.wait(0.10):
            sample = self._sample()
            for key, value in sample.items():
                self._peak_sample[key] = max(self._peak_sample.get(key, value), value)
            while self._index < len(self.jobs) and self._result_written(self._index):
                self._print_finish(self._index, sample)
                self._index += 1
                if self._index < len(self.jobs):
                    self._peak_sample = dict(sample)
                    self._print_start(self._index, sample)

    def __enter__(self) -> "VideoVramMonitor":
        if self.jobs:
            self._thread = threading.Thread(target=self._run, name="vram-video-monitor", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        sample = self._sample()
        for key, value in sample.items():
            self._peak_sample[key] = max(self._peak_sample.get(key, value), value)
        while self._index < len(self.jobs):
            if not self._peak_sample:
                self._peak_sample = dict(sample)
            self._print_finish(self._index, sample)
            self._index += 1
            if self._index < len(self.jobs):
                self._print_start(self._index, sample)
                self._peak_sample = dict(sample)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained DLC model on project source-video ROIs.")
    parser.add_argument("config", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-mode", choices=("trained", "superanimal"), default="trained")
    parser.add_argument("--pose-model", choices=TOPVIEWMOUSE_POSE_MODEL_IDS, default="hrnet_w32")
    parser.add_argument(
        "--detector",
        choices=TOPVIEWMOUSE_PRETRAINED_DETECTOR_MODEL_IDS,
        default="fasterrcnn_resnet50_fpn_v2",
    )
    parser.add_argument("--shuffle", type=int, default=2)
    parser.add_argument(
        "--snapshot-index",
        default=None,
        help="DLC PyTorch snapshot selector: integer index or 'best'.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--labeled-video", action="store_true")
    parser.add_argument("--pcutoff", type=float, default=0.6)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dotsize", type=int, default=4)
    parser.add_argument("--execution-mode", choices=("queue", "stable"), default="queue")
    parser.add_argument("--evaluation-padding", choices=("model", "minimum", "training"), default="model")
    parser.add_argument("--roi-workers", type=int, default=4)
    parser.add_argument("--roi-validation", choices=("fast", "full"), default="fast")
    parser.add_argument("--pipeline-chunk-size", type=int, default=4)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--delete-roi-video", action="store_true")
    parser.add_argument("--inference-overrides-json", default="{}")
    parser.add_argument(
        "--temporal-refiner",
        choices=("auto", "on", "off"),
        default="auto",
        help="Apply the project temporal refiner after DLC inference.",
    )
    return parser.parse_args()


def parse_inference_overrides(raw: str, model_mode: str) -> dict:
    try:
        payload = json.loads(raw or "{}")
    except json.JSONDecodeError as error:
        raise ValueError(tr('PyTorch 추론 설정 JSON 오류: {v0}', v0=error)) from error
    if not isinstance(payload, dict):
        raise ValueError(tr('PyTorch 추론 설정은 JSON 객체여야 합니다.'))
    supported = (
        SUPERANIMAL_INFERENCE_OVERRIDE_KEYS
        if model_mode == "superanimal"
        else TRAINED_INFERENCE_OVERRIDE_KEYS
    )
    unsupported = sorted(set(payload) - supported)
    if unsupported:
        raise ValueError(
            tr('{v0} 추론에서 지원하지 않는 설정 키: ', v0=model_mode) + ", ".join(unsupported)
        )
    return payload


def snapshot_selector(value: str | None) -> int | str | None:
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        return text


def ensure_test_pose_config(config: Path, shuffle: int) -> Path | None:
    """Repair legacy portable bundles that omitted DLC's inference pose_cfg.yaml."""
    candidates = sorted(
        path
        for path in config.parent.glob(
            f"dlc-models-pytorch/iteration-*/*shuffle{int(shuffle)}/train/pytorch_config.yaml"
        )
        if path.is_file()
    )
    if not candidates:
        return None
    train_pose_config = candidates[0]
    test_pose_config = train_pose_config.parent.parent / "test" / "pose_cfg.yaml"
    if test_pose_config.is_file():
        return test_pose_config
    project_payload = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    pose_payload = yaml.safe_load(train_pose_config.read_text(encoding="utf-8")) or {}
    bodyparts = list(map(str, project_payload.get("bodyparts", []) or []))
    test_payload = {
        "dataset": str(config.parent),
        "dataset_type": "multi-animal-imgaug",
        "num_joints": len(bodyparts),
        "all_joints": [[index] for index in range(len(bodyparts))],
        "all_joints_names": bodyparts,
        "net_type": str(pose_payload.get("net_type", "hrnet_w32")),
        "global_scale": 1,
        "scoremap_dir": "test",
    }
    test_pose_config.parent.mkdir(parents=True, exist_ok=True)
    test_pose_config.write_text(
        yaml.safe_dump(test_payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"PORTABLE_MODEL_REPAIRED={test_pose_config}", flush=True)
    return test_pose_config


def resolve_training_set_index(config: Path, shuffle: int) -> int:
    """Return the TrainingFraction index that owns a saved DLC Shuffle.

    HAGISILTA keeps older models when a later dataset version changes the train
    fraction. DLC resolves a model from both ``shuffle`` and
    ``trainingsetindex``; always passing index 0 therefore makes an older model
    look missing even though its folder and metadata still exist.
    """
    fallback = 0
    try:
        metadata = TrainingDatasetMetadata.load(str(config), load_splits=False)
        matches = [
            item for item in metadata.shuffles
            if int(item.index) == int(shuffle)
        ]
        if not matches:
            print(
                f"INFERENCE_TRAINSET=shuffle{shuffle}:metadata-missing:index={fallback}",
                flush=True,
            )
            return fallback

        target_fraction = round(float(matches[-1].train_fraction), 2)
        payload = read_config(str(config))
        fractions = [
            round(float(value), 2)
            for value in (payload.get("TrainingFraction") or [])
        ]
        if not any(math.isclose(value, target_fraction, abs_tol=1e-9) for value in fractions):
            fractions.append(target_fraction)
            edit_project_config(config, {"TrainingFraction": fractions})
        resolved = next(
            index for index, value in enumerate(fractions)
            if math.isclose(value, target_fraction, abs_tol=1e-9)
        )
        print(
            f"INFERENCE_TRAINSET=shuffle{shuffle}:fraction={target_fraction:.2f}:index={resolved}",
            flush=True,
        )
        return resolved
    except Exception as error:
        print(
            f"INFERENCE_TRAINSET=shuffle{shuffle}:resolve-failed="
            f"{type(error).__name__}:{error}:index={fallback}",
            flush=True,
        )
        return fallback


def build_jobs(videos: list[dict], output: Path) -> list[dict]:
    validate_unique_source_stems(item["path"] for item in videos)
    jobs: list[dict] = []
    generation_id = new_generation_id()
    for video_index, item in enumerate(videos, start=1):
        source = Path(item["path"]).resolve()
        profile = Path(item["roi_profile"]).resolve()
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(tr('원본 영상을 열 수 없습니다: {v0}', v0=source))
        frame_size = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        capture.release()
        rois = load_rois(profile, frame_size)
        allowed = set(map(int, item.get("arenas", [])))
        rois = [roi for roi in rois if not allowed or int(roi["id"]) in allowed]
        source_output = output / source.stem
        for roi in rois:
            arena_id = int(roi["id"])
            arena_output = source_output / f"arena_{arena_id:02d}"
            group = str(item.get("group", ""))
            if group == "model_evaluation_pilot":
                cropped_stem = generated_roi_stem(source.stem, arena_id, generation_id)
            else:
                cropped_stem = f"{source.stem}_arena_{arena_id:02d}_roi"
            cropped = arena_output / f"{cropped_stem}.avi"
            jobs.append({
                "video_index": video_index,
                "video_total": len(videos),
                "source": source,
                "profile": profile,
                "roi": roi,
                "arena_id": arena_id,
                "arena_output": arena_output,
                "cropped": cropped,
                "group": group,
                "subject_id": item.get("subject_id", source.stem),
                "generation_id": generation_id if group == "model_evaluation_pilot" else "",
            })
    validate_crop_destinations(jobs)
    return jobs


def validate_crop_destinations(jobs: list[dict]) -> None:
    """Read-only guard before project, progress, or ROI files can be changed."""
    destinations: set[str] = set()
    for job in jobs:
        key = normalized_source_path(job["arena_output"])
        if key in destinations:
            raise ValueError(tr('ROI 결과 저장 위치가 중복됩니다: {v0}', v0=job['arena_output']))
        destinations.add(key)
        metadata_path = Path(job["arena_output"]) / "roi-metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        owner = metadata.get("source_video") if isinstance(metadata, dict) else None
        if owner and normalized_source_path(owner) != normalized_source_path(job["source"]):
            raise ValueError(
                tr('기존 ROI 결과가 다른 원본 영상의 결과입니다. 덮어쓰지 않도록 작업을 중단했습니다.\n- 기존 원본: {v0}\n- 현재 원본: {v1}\n- 저장 위치: {v2}', v0=owner, v1=job['source'], v2=job['arena_output'])
            )


def existing_results(job: dict, shuffle: int) -> tuple[list[Path], list[Path]]:
    output = job["arena_output"]
    h5 = [
        path for path in output.glob("*.h5")
        if "evaluation-results" not in path.name.lower()
        and f"shuffle{shuffle}" in path.name.lower()
        and not path.stem.lower().endswith("_spatial")
    ]
    labeled = [
        path for path in output.glob("*labeled*.avi")
        if f"shuffle{shuffle}" in path.name.lower() and is_lossless_labeled_video(path)
    ]
    return h5, labeled


def comparison_pose_path(pose_path: Path, variant: str) -> Path:
    comparison_root = pose_path.parent / TEMPORAL_COMPARISON_DIRECTORY
    return comparison_root / f"{pose_path.stem}__{variant}{pose_path.suffix}"


def preserve_pose_variant(
    pose_path: Path, variant: str, active_pose_path: Path | None = None
) -> Path:
    destination = comparison_pose_path(active_pose_path or pose_path, variant)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pose_path, destination)
    source_csv = pose_path.with_suffix(".csv")
    destination_csv = destination.with_suffix(".csv")
    if source_csv.is_file():
        shutil.copy2(source_csv, destination_csv)
    else:
        destination_csv.unlink(missing_ok=True)
    return destination


def prepare_mode_reanalysis(job: dict, args: argparse.Namespace) -> None:
    """Preserve the active variant, then remove DLC's active result cache.

    DLC 3.0's compatibility wrapper always supplies ``overwrite=False`` to the
    PyTorch API. Passing another overwrite value through ``torch_kwargs`` raises
    a duplicate-key TypeError, so a mode change must clear the selected scorer's
    generated cache before calling DLC again.
    """
    if not job.get("force_overwrite"):
        return
    candidates = existing_results(job, args.shuffle)[0]
    if not candidates:
        return
    pose_path = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    marker = pose_path.with_suffix(".temporal.json")
    if marker.is_file():
        preserve_pose_variant(pose_path, "temporal")
        legacy_spatial = pose_path.parent / "spatial-original" / pose_path.name
        if legacy_spatial.is_file():
            preserve_pose_variant(legacy_spatial, "spatial", pose_path)
    else:
        preserve_pose_variant(pose_path, "spatial")
    for generated in (
        pose_path,
        pose_path.with_suffix(".csv"),
        pose_path.with_name(f"{pose_path.stem}_full.pickle"),
        pose_path.with_name(f"{pose_path.stem}_meta.pickle"),
        marker,
    ):
        generated.unlink(missing_ok=True)


def temporal_mode_requested(config: Path, args: argparse.Namespace) -> bool:
    if args.model_mode != "trained" or args.temporal_refiner == "off":
        return False
    if args.temporal_refiner == "on":
        return True
    return temporal_checkpoint_path(config.parent, args.shuffle).is_file()


def result_matches_temporal_mode(
    config: Path, job: dict, args: argparse.Namespace
) -> bool:
    h5_files = existing_results(job, args.shuffle)[0]
    if not h5_files:
        return False
    if args.model_mode != "trained":
        return True
    latest_pose = max(h5_files, key=lambda path: (path.stat().st_mtime_ns, path.name))
    has_refined_result = latest_pose.with_suffix(".temporal.json").is_file()
    requested = temporal_mode_requested(config, args)
    if has_refined_result != requested:
        return False
    if requested:
        return (
            comparison_pose_path(latest_pose, "spatial").is_file()
            and comparison_pose_path(latest_pose, "temporal").is_file()
        )
    return True


def apply_temporal_refinement(config: Path, job: dict, args: argparse.Namespace) -> Path | None:
    if args.model_mode != "trained":
        return None
    if not temporal_mode_requested(config, args):
        candidates = existing_results(job, args.shuffle)[0]
        for pose_path in candidates:
            pose_path.with_suffix(".temporal.json").unlink(missing_ok=True)
            if getattr(args, "no_csv", False):
                pose_path.with_suffix(".csv").unlink(missing_ok=True)
        if candidates:
            latest_pose = max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))
            paired_spatial = comparison_pose_path(latest_pose, "spatial")
            paired_temporal = comparison_pose_path(latest_pose, "temporal")
            # An ON run creates a fair before/after pair from one DLC inference.
            # A later OFF-only run must not replace only the spatial half with a
            # different DLC run, otherwise the visible difference is not the TCN.
            if not paired_spatial.is_file() or not paired_temporal.is_file():
                preserve_pose_variant(latest_pose, "spatial")
        return None
    checkpoint = temporal_checkpoint_path(config.parent, args.shuffle)
    if not checkpoint.is_file():
        if args.temporal_refiner == "on":
            raise RuntimeError(
                tr('Shuffle {v0}의 시간 문맥 보정 모델이 없습니다: {v1}', v0=args.shuffle, v1=checkpoint)
            )
        print(f"TEMPORAL_REFINER=not-available:shuffle={args.shuffle}", flush=True)
        return None
    candidates = existing_results(job, args.shuffle)[0]
    if not candidates:
        raise RuntimeError(tr('시간 문맥 보정에 사용할 DLC 좌표가 없습니다: {v0}', v0=job['arena_output']))
    pose_path = max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))
    preserve_pose_variant(pose_path, "spatial")
    spatial_root = pose_path.parent / "spatial-original"
    spatial_root.mkdir(parents=True, exist_ok=True)
    spatial_h5 = spatial_root / pose_path.name
    if spatial_h5.exists():
        spatial_h5.unlink()
    pose_path.replace(spatial_h5)
    source_table = read_pose_table(spatial_h5)
    capture = cv2.VideoCapture(str(job["cropped"]))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    if fps <= 0 or width <= 0 or height <= 0:
        spatial_h5.replace(pose_path)
        raise RuntimeError(tr('시간 문맥 보정용 ROI 영상의 FPS 또는 크기를 확인할 수 없습니다.'))
    spatial_csv: Path | None = None
    csv_path = pose_path.with_suffix(".csv")
    try:
        refined, report = refine_pose_table(
            source_table,
            checkpoint,
            fps,
            width,
            height,
            detect_device() if args.device == "auto" else args.device,
            max(16, min(64, int(args.batch_size) * 2)),
            Path(job["cropped"]),
            Path(job["arena_output"]) / "roi-metadata.json",
        )
        write_pose_table(refined, pose_path)
        if csv_path.is_file():
            spatial_csv = spatial_root / csv_path.name
            if spatial_csv.exists():
                spatial_csv.unlink()
            csv_path.replace(spatial_csv)
            write_pose_table(refined, csv_path)
            report["spatial_csv"] = str(spatial_csv)
        report.update(
            {
                "spatial_pose": str(spatial_h5),
                "refined_pose": str(pose_path),
                "video": str(job["cropped"]),
            }
        )
        report_path = pose_path.with_suffix(".temporal.json")
        write_refinement_report(report, report_path)
        temporal_pose = preserve_pose_variant(pose_path, "temporal")
        write_refinement_report(report, temporal_pose.with_suffix(".temporal.json"))
        print(
            f"TEMPORAL_REFINER=applied:{job['source'].name}:arena={job['arena_id']}:"
            f"mean_shift_px={report['mean_shift_px']:.3f}",
            flush=True,
        )
        return pose_path
    except Exception:
        pose_path.unlink(missing_ok=True)
        if spatial_h5.exists():
            spatial_h5.replace(pose_path)
        if spatial_csv is not None and spatial_csv.exists():
            csv_path.unlink(missing_ok=True)
            spatial_csv.replace(csv_path)
        raise


def crop_stage_callback(job: dict, cancel_event=None):
    def crop_stage(stage, frames, total):
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError("ROI preparation cancelled")
        label = {"encoding": tr('생성'), "fast_validation": tr('빠른 검증')}.get(stage, tr('픽셀 검증'))
        print(
            f"ROI_CROP_PROGRESS={job['source'].name}:arena={job['arena_id']}:"
            + tr('{label} {frames}/{total} 프레임', label=label, frames=frames, total=total), flush=True,
        )
        notify(job, "cropping", crop_stage=stage, crop_frames=frames, crop_total=total)
    return crop_stage


def finish_crop(job: dict, metadata: dict, started: float) -> dict:
    output: Path = job["arena_output"]
    metadata.update({
        "source_video": str(job["source"]),
        "roi_profile": str(job["profile"]),
        "arena_id": job["arena_id"],
        "group": job["group"],
        "subject_id": job["subject_id"],
        "generation_id": job.get("generation_id", ""),
        "roi_video": str(job["cropped"]),
    })
    if "evaluation_padding" in job:
        metadata["evaluation_padding"] = job["evaluation_padding"]
    (output / f"{job['cropped'].stem}.roi-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "roi-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    job["crop_seconds"] = time.perf_counter() - started
    print(
        f"ROI_CROP_COMPLETE={job['source'].name}:arena={job['arena_id']}:"
        f"encoder={metadata.get('encoder_backend', 'FFV1')}:"
        f"{tr('생성')} {metadata.get('encoding_seconds', 0):.2f}s:"
        f"{tr('빠른 검증') if metadata.get('validation_mode') == 'fast' else tr('픽셀 검증')} "
        f"{metadata.get('validation_seconds', 0):.2f}s", flush=True,
    )
    notify(job, "ready")
    return job


def prepare_crop(job: dict) -> dict:
    notify(job, "cropping")
    started = time.perf_counter()
    job["arena_output"].mkdir(parents=True, exist_ok=True)
    metadata = write_tracking_crop_video(
        job["source"], job["roi"]["corners"], job["cropped"],
        stage_progress=crop_stage_callback(job),
        encoder_threads=job.get("crop_encoder_threads"),
        validation_mode=job.get("roi_validation", "fast"),
    )
    return finish_crop(job, metadata, started)


def prepare_crop_group(jobs: list[dict], cancel_event=None) -> list[dict]:
    """Prepare one source's ROI group using a shared decoded frame stream."""
    if not jobs:
        return []
    started = time.perf_counter()
    source_key = normalized_source_path(jobs[0]["source"])
    validation_mode = jobs[0].get("roi_validation", "fast")
    for job in jobs:
        if normalized_source_path(job["source"]) != source_key:
            raise ValueError(tr('ROI 묶음에는 같은 원본 영상만 사용할 수 있습니다.'))
        if job.get("roi_validation", "fast") != validation_mode:
            raise ValueError(tr('ROI 묶음의 검증 방식은 같아야 합니다.'))
    for job in jobs:
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError("ROI preparation cancelled")
        notify(job, "cropping")
        job["arena_output"].mkdir(parents=True, exist_ok=True)
    metadata = write_tracking_crop_videos(
        jobs[0]["source"],
        [{"corners": job["roi"]["corners"], "destination": job["cropped"],
          "stage_progress": crop_stage_callback(job, cancel_event)} for job in jobs],
        validation_mode=validation_mode,
        encoder_threads=jobs[0].get("crop_encoder_threads"),
    )
    if len(metadata) != len(jobs):
        raise RuntimeError(tr('ROI 묶음 생성 결과 수가 요청과 일치하지 않습니다.'))
    # A successful bulk return means the videos are already published. Do not
    # interrupt their provenance writes if cancellation arrived after the last
    # validation callback. Likewise, one metadata I/O error must not skip the
    # other published members; finish those and then propagate the first error.
    finished = []
    first_error = None
    for job, record in zip(jobs, metadata):
        try:
            finished.append(finish_crop(job, record, started))
        except Exception as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error
    return finished


def group_crop_jobs(jobs: list[dict], max_encoders: int) -> list[list[dict]]:
    """Stable source grouping; never open more than the encoder budget per group."""
    max_encoders = max(1, int(max_encoders))
    by_source: dict[str, list[dict]] = {}
    for job in jobs:
        by_source.setdefault(normalized_source_path(job["source"]), []).append(job)
    return [source_jobs[index:index + max_encoders]
            for source_jobs in by_source.values()
            for index in range(0, len(source_jobs), max_encoders)]


class CropEncoderSlots:
    """Acquire a whole group atomically; partial slot ownership can deadlock."""
    def __init__(self, capacity: int):
        self.capacity = max(1, int(capacity))
        self.available = self.capacity
        self.condition = threading.Condition()
        self.cancel_event = threading.Event()
        self.error: BaseException | None = None

    def acquire(self, count: int) -> None:
        if not 1 <= count <= self.capacity:
            raise ValueError("ROI encoder request exceeds its worker budget")
        with self.condition:
            self.condition.wait_for(lambda: self.cancel_event.is_set() or self.available >= count)
            if self.cancel_event.is_set():
                raise CancelledError("ROI preparation cancelled")
            self.available -= count

    def release(self, count: int) -> None:
        with self.condition:
            self.available += count
            self.condition.notify_all()

    def cancel(self, error: BaseException | None = None) -> None:
        with self.condition:
            if error is not None and not isinstance(error, CancelledError) and self.error is None:
                self.error = error
            self.cancel_event.set()
            self.condition.notify_all()


def prepared_crop_jobs(jobs: list[dict], max_encoders: int):
    """Bound grouped prefetch while inference consumes completed ROI jobs.

    At most ``max_encoders`` frames/crops are encoded concurrently, even when
    several sources share the pool. A bounded window of group futures keeps
    crop work running while the caller performs a GPU inference chunk.
    """
    groups = iter(group_crop_jobs(jobs, max_encoders))
    slots = CropEncoderSlots(max_encoders)
    pool = ThreadPoolExecutor(max_workers=slots.capacity, thread_name_prefix="roi-crop")
    pending = set()

    def prepare(group):
        slots.acquire(len(group))
        try:
            return prepare_crop_group(group, slots.cancel_event)
        except BaseException as error:
            slots.cancel(error)
            raise
        finally:
            slots.release(len(group))

    def prefetch():
        while len(pending) < 2 * slots.capacity:
            group = next(groups, None)
            if group is None:
                break
            pending.add(pool.submit(prepare, group))

    try:
        prefetch()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            pending.difference_update(done)
            if slots.error is not None:
                raise slots.error
            # Raise the original crop failure before a sibling cancellation.
            ordered = sorted(done, key=lambda future: isinstance(future.exception(), CancelledError))
            completed = [future.result() for future in ordered]
            prefetch()
            for group in completed:
                yield from group
    finally:
        slots.cancel()
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


def analyze_one(config: Path, job: dict, args: argparse.Namespace) -> None:
    inference_kwargs = {
        "batch_size": args.batch_size,
        "device": detect_device() if args.device == "auto" else args.device,
        "snapshot_index": snapshot_selector(args.snapshot_index),
    }
    inference_kwargs.update(args.inference_overrides)
    spec = getattr(args, "padding_spec", {"mode": "model"})
    if spec["mode"] != "model":
        from hagisilta_evaluation_padding import sized_spec, padding_transform
        capture = cv2.VideoCapture(str(job["cropped"]))
        try:
            spec = sized_spec(spec, int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        finally:
            capture.release()
        inference_kwargs["transform"] = padding_transform(spec)
        metadata_path = job["cropped"].with_name(job["cropped"].stem + ".roi-metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["evaluation_padding"] = spec
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"EVALUATION_PADDING={spec['mode']}:{spec['width']}x{spec['height']}:resize=false", flush=True)
    with VideoVramMonitor([job]), inference_events([job]):
        deeplabcut.analyze_videos(
            str(config), [str(job["cropped"])], shuffle=args.shuffle,
            trainingsetindex=args.trainingset_index,
            save_as_csv=not args.no_csv, destfolder=str(job["arena_output"]),
            **inference_kwargs,
        )


def analyze_superanimal_one(job: dict, args: argparse.Namespace) -> None:
    """Run DLC's bundled TopViewMouse model on one cropped ROI."""
    inference_kwargs = {
        "video_adapt": False,
        "plot_trajectories": False,
        "batch_size": args.batch_size,
        "detector_batch_size": 1,
        "max_individuals": 1,
        "device": detect_device() if args.device == "auto" else args.device,
        "pcutoff": args.pcutoff,
        "adapt_iterations": 1000,
        "pseudo_threshold": 0.1,
        "bbox_threshold": 0.9,
        "detector_epochs": 4,
        "pose_epochs": 4,
        "video_adapt_batch_size": 8,
        "create_labeled_video": args.labeled_video,
        "plot_bboxes": True,
    }
    inference_kwargs.update(args.inference_overrides)
    notify(job, "inference")
    progress = job.get("_tracking_progress")
    with VideoVramMonitor([job]):
        infer_superanimal_with_lossless_video(
            videos=[str(job["cropped"])],
            superanimal_name="superanimal_topviewmouse",
            model_name=args.pose_model,
            detector_name=args.detector,
            dest_folder=str(job["arena_output"]),
            progress_callback=progress.label_event if progress else None,
            **inference_kwargs,
        )


def analyze_many(config: Path, jobs: list[dict], args: argparse.Namespace) -> None:
    if not jobs:
        return
    inference_kwargs = {
        "batch_size": args.batch_size,
        "device": detect_device() if args.device == "auto" else args.device,
        "snapshot_index": snapshot_selector(args.snapshot_index),
    }
    inference_kwargs.update(args.inference_overrides)
    with VideoVramMonitor(jobs), inference_events(jobs):
        deeplabcut.analyze_videos(
            str(config), [str(job["cropped"]) for job in jobs],
            shuffle=args.shuffle, trainingsetindex=args.trainingset_index,
            save_as_csv=not args.no_csv, destfolder=None,
            **inference_kwargs,
        )


def label_one(config: Path, job: dict, args: argparse.Namespace) -> None:
    progress = job.get("_tracking_progress")
    create_lossless_labeled_videos(
        str(config), [str(job["cropped"])], shuffle=args.shuffle,
        trainingsetindex=args.trainingset_index,
        destfolder=str(job["arena_output"]), draw_skeleton=True, pcutoff=args.pcutoff,
        dotsize=args.dotsize, overwrite=True,
        snapshot_index=snapshot_selector(args.snapshot_index),
        progress_callback=progress.label_event if progress else None,
    )


def coordinates_finished(job, args):
    """Called after inference/refinement returns, including native cached results."""
    if not job.get("_tracking_progress"):
        return
    poses = existing_results(job, args.shuffle)[0]
    if args.model_mode == "superanimal":
        poses = list(job["arena_output"].glob("*.h5"))
    if not poses:
        raise RuntimeError(tr('추론 좌표가 저장되지 않았습니다: {v0}', v0=job['cropped'].name))
    pose = max(poses, key=lambda path: path.stat().st_mtime_ns)
    notify(job, "video_pending" if args.labeled_video else "done", pose=pose)


def run_stable(config: Path, jobs: list[dict], args: argparse.Namespace) -> int:
    completed, total = 0, len(jobs)
    for job in jobs:
        job["roi_validation"] = getattr(args, "roi_validation", "fast")
        started = time.perf_counter()
        existing_h5, existing_labeled = existing_results(job, args.shuffle)
        mode_matches = result_matches_temporal_mode(config, job, args)
        job["force_overwrite"] = bool(existing_h5) and not mode_matches
        if args.model_mode == "trained" and args.skip_completed and mode_matches:
            coordinates_finished(job, args)
            if args.labeled_video and not existing_labeled:
                if not job["cropped"].is_file():
                    prepare_crop(job)
                label_one(config, job, args)
            elif args.labeled_video and existing_labeled:
                notify(job, "done", labeled=max(existing_labeled, key=lambda path: path.stat().st_mtime_ns))
            if args.delete_roi_video and job["cropped"].is_file():
                job["cropped"].unlink()
            completed += 1
            print(
                f"BATCH_TRACK_DONE={completed}/{total}:{job['source'].name}:"
                f"arena {job['arena_id']}:skipped",
                flush=True,
            )
            continue
        print(
            f"BATCH_TRACK={completed + 1}/{total}:video {job['video_index']}/{job['video_total']}:"
            f"arena {job['arena_id']}:crop",
            flush=True,
        )
        prepare_mode_reanalysis(job, args)
        prepare_crop(job)
        print(f"BATCH_TRACK={completed + 1}/{total}:arena {job['arena_id']}:inference", flush=True)
        if args.model_mode == "superanimal":
            analyze_superanimal_one(job, args)
            if not args.labeled_video:
                coordinates_finished(job, args)
        else:
            analyze_one(config, job, args)
            apply_temporal_refinement(config, job, args)
            coordinates_finished(job, args)
        if args.model_mode == "trained" and args.labeled_video:
            label_one(config, job, args)
        if args.delete_roi_video and job["cropped"].is_file():
            job["cropped"].unlink()
        completed += 1
        print(
            f"BATCH_TRACK_DONE={completed}/{total}:{job['source'].name}:arena {job['arena_id']}:"
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return completed


def run_queue(config: Path, jobs: list[dict], args: argparse.Namespace) -> int:
    total = len(jobs)
    ready: list[dict] = []
    inference_jobs: list[dict] = []
    labeled_jobs: list[dict] = []
    skipped_jobs: list[dict] = []

    for job in jobs:
        existing_h5, existing_labeled = existing_results(job, args.shuffle)
        job["existing_h5"] = bool(existing_h5)
        job["existing_labeled"] = bool(existing_labeled)
        mode_matches = result_matches_temporal_mode(config, job, args)
        job["force_overwrite"] = bool(existing_h5) and not mode_matches
        if args.skip_completed and mode_matches:
            coordinates_finished(job, args)
            if args.labeled_video and not existing_labeled:
                ready.append(job)
                labeled_jobs.append(job)
            else:
                skipped_jobs.append(job)
                if args.labeled_video and existing_labeled:
                    notify(job, "done", labeled=max(existing_labeled, key=lambda path: path.stat().st_mtime_ns))
        else:
            ready.append(job)
            inference_jobs.append(job)
            if args.labeled_video:
                labeled_jobs.append(job)

    workers = max(1, min(int(args.roi_workers), 8, len(ready) or 1, os.cpu_count() or 1))
    for job in ready:
        # Share the CPU budget across ROI jobs instead of multiplying the full
        # per-video encoding pool by up to eight concurrent crops.
        job["crop_encoder_threads"] = max(1, min(4, (os.cpu_count() or 1) // workers))
        job["roi_validation"] = getattr(args, "roi_validation", "fast")
    for job in inference_jobs:
        prepare_mode_reanalysis(job, args)
    print(
        f"BATCH_QUEUE_PREPARE={len(ready)}:workers={workers}:total={total}",
        flush=True,
    )
    prepared = 0
    inferred = 0
    inference_ids = {id(job) for job in inference_jobs}
    inference_batch: list[dict] = []
    chunk_size = max(1, min(int(args.pipeline_chunk_size), len(inference_jobs) or 1))

    # Regenerating only the annotated export must not rewrite its source or
    # invalidate saved analysis/composition provenance.
    reusable = [job for job in ready if id(job) not in inference_ids and job["cropped"].is_file()]
    reusable_ids = {id(job) for job in reusable}
    required = [job for job in ready if id(job) not in reusable_ids]
    for job in reusable:
        job["crop_seconds"] = 0.0
    prepared = len(reusable)

    if ready:
        crop_stream = prepared_crop_jobs(required, workers)
        try:
            for job in crop_stream:
                prepared += 1
                print(
                    f"BATCH_QUEUE_CROP={prepared}/{len(ready)}:{job['source'].name}:"
                    f"arena {job['arena_id']}:{job.get('crop_seconds', 0):.1f}s",
                    flush=True,
                )
                if id(job) in inference_ids:
                    inference_batch.append(job)
                if len(inference_batch) >= chunk_size:
                    print(
                        f"BATCH_PIPELINE_INFERENCE={inferred + 1}-"
                        f"{inferred + len(inference_batch)}/{len(inference_jobs)}:"
                        f"chunk={len(inference_batch)}",
                        flush=True,
                    )
                    analyze_many(config, inference_batch, args)
                    for temporal_job in inference_batch:
                        apply_temporal_refinement(config, temporal_job, args)
                        coordinates_finished(temporal_job, args)
                    inferred += len(inference_batch)
                    inference_batch = []
        finally:
            # Close eagerly on inference errors/interrupts, cancelling queued
            # crop groups and stopping live encoders at their next callback.
            crop_stream.close()

        if inference_batch:
            print(
                f"BATCH_PIPELINE_INFERENCE={inferred + 1}-"
                f"{inferred + len(inference_batch)}/{len(inference_jobs)}:"
                f"chunk={len(inference_batch)}",
                flush=True,
            )
            analyze_many(config, inference_batch, args)
            for temporal_job in inference_batch:
                apply_temporal_refinement(config, temporal_job, args)
                coordinates_finished(temporal_job, args)
            inferred += len(inference_batch)

    if labeled_jobs:
        print(f"BATCH_QUEUE_LABELED_VIDEO={len(labeled_jobs)}", flush=True)
        progress = labeled_jobs[0].get("_tracking_progress")
        create_lossless_labeled_videos(
            str(config), [str(job["cropped"]) for job in labeled_jobs],
            shuffle=args.shuffle, trainingsetindex=args.trainingset_index, destfolder=None,
            draw_skeleton=True, pcutoff=args.pcutoff, dotsize=args.dotsize, overwrite=True,
            snapshot_index=snapshot_selector(args.snapshot_index),
            progress_callback=progress.label_event if progress else None,
        )

    completed = 0
    for job in jobs:
        status = "skipped" if job in skipped_jobs else "queue"
        if args.delete_roi_video and job["cropped"].is_file():
            job["cropped"].unlink()
        completed += 1
        print(
            f"BATCH_TRACK_DONE={completed}/{total}:{job['source'].name}:"
            f"arena {job['arena_id']}:{status}",
            flush=True,
        )
    return completed


def main() -> int:
    args = parse_args()
    try:
        args.inference_overrides = parse_inference_overrides(
            args.inference_overrides_json, args.model_mode
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    config, plan_path, output = args.config.resolve(), args.plan.resolve(), args.output.resolve()
    from hagisilta_evaluation_padding import padding_spec, validate_padding_run, validate_padding_overrides
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    videos = plan.get("videos", [])
    validate_unique_source_stems(item["path"] for item in videos)
    validate_padding_run(args.evaluation_padding, args.model_mode, args.execution_mode, videos)
    validate_padding_overrides(args.evaluation_padding, args.inference_overrides)
    args.padding_spec = padding_spec(config, args.shuffle, args.evaluation_padding)
    jobs = build_jobs(videos, output)
    if args.model_mode == "trained":
        args.trainingset_index = resolve_training_set_index(config, args.shuffle)
        ensure_test_pose_config(config, args.shuffle)
    else:
        args.trainingset_index = 0
    output.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        job["roi_validation"] = args.roi_validation
        if job.get("group") == "model_evaluation_pilot":
            job["evaluation_padding"] = args.padding_spec
    print(
        f"BATCH_TRACK_MODE={args.execution_mode}:model={args.model_mode}:jobs={len(jobs)}:roi_workers={args.roi_workers}",
        flush=True,
    )
    progress = TrackingProgress(jobs, config, args)
    try:
        completed = (
            run_stable(config, jobs, args)
            if args.model_mode == "superanimal" or args.execution_mode == "stable"
            else run_queue(config, jobs, args)
        )
    except BaseException:
        progress.finish_unfinished("error")
        raise
    print(f"BATCH_TRACK_COMPLETE={completed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
