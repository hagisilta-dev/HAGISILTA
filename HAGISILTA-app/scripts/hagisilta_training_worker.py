# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

# DLC/ruamel may otherwise open UTF-8 config.yaml with Windows CP949 when this
# worker is launched directly instead of through the HAGISILTA GUI.
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
if os.name == "nt" and not sys.flags.utf8_mode:
    os.execv(sys.executable, [sys.executable, *sys.argv])

import deeplabcut
import numpy as np
import torch
from PIL import Image

from hagisilta_runtime import (
    detect_device,
    edit_project_config,
    install_unicode_safe_opencv_reader,
    project_backup_root,
    project_state_root,
)
from hagisilta_label_review import temporarily_filter_excluded_labels
from deeplabcut.modelzoo import build_weight_init
from deeplabcut.core.engine import Engine
from deeplabcut.generate_training_dataset.metadata import TrainingDatasetMetadata
from deeplabcut.generate_training_dataset.trainingsetmanipulation import merge_annotateddatasets
from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils.auxiliaryfunctions import read_config
from hagisilta_dataset_split import indices_from_merged_data, load_split_plan
from hagisilta_pose_model_catalog import (
    HAGISILTA_BU_TRAINING_POSE_MODEL_IDS,
    require_hagisilta_training_pose_model,
)
from hagisilta_training_input import (
    inference_data_config, training_data_config, validate_input_overrides, whole_roi_training,
)


DEFAULT_SHUFFLE = 1

# Run at module import time so Windows DataLoader worker processes inherit the
# same Unicode-path behavior when ``dataloader_workers`` is greater than zero.
install_unicode_safe_opencv_reader()


def install_persistent_dataloader_workers(enabled: bool) -> bool:
    """Keep DLC training workers alive across epochs without patching DLC files."""
    if not enabled:
        return False
    from deeplabcut.pose_estimation_pytorch.apis import training as dlc_training

    current_factory = dlc_training.DataLoader
    if getattr(current_factory, "_hagisilta_persistent_workers", False):
        return True

    def persistent_dataloader(*args, **kwargs):
        # PyTorch rejects persistent_workers=True when num_workers is zero.
        # DLC's validation loader omits num_workers and therefore remains unchanged.
        if int(kwargs.get("num_workers", 0) or 0) > 0:
            kwargs.setdefault("persistent_workers", True)
        return current_factory(*args, **kwargs)

    persistent_dataloader._hagisilta_persistent_workers = True
    dlc_training.DataLoader = persistent_dataloader
    return True


def backup_path(config: Path, label: str, suffix: str) -> Path:
    root = project_backup_root(config.parent, create=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return root / f"{label}-{stamp}{suffix}"


def repair_training_metadata_if_needed(config: Path) -> Path | None:
    """Recover DLC metadata.yaml after an abrupt shutdown left it unreadable."""
    metadata_path = TrainingDatasetMetadata.path(read_config(str(config)))
    if not metadata_path.exists():
        return None
    try:
        TrainingDatasetMetadata.load(str(config), load_splits=True)
        return None
    except Exception as error:
        backup = backup_path(config, "corrupt-training-metadata", ".yaml")
        shutil.copy2(metadata_path, backup)
        rebuilt = TrainingDatasetMetadata.create(str(config))
        # This application creates DLC 3 PyTorch projects exclusively. DLC's
        # fallback reconstruction assumes legacy TensorFlow when metadata.yaml
        # is missing, so correct the recovered shuffle engines explicitly.
        rebuilt = replace(
            rebuilt,
            shuffles=tuple(replace(item, engine=Engine.PYTORCH) for item in rebuilt.shuffles),
        )
        rebuilt.save()
        TrainingDatasetMetadata.load(str(config), load_splits=True)
        print(f"RECOVERED_TRAINING_METADATA={metadata_path}", flush=True)
        print(f"CORRUPT_METADATA_BACKUP={backup}", flush=True)
        print(f"METADATA_RECOVERY_REASON={type(error).__name__}: {error}", flush=True)
        return backup


def repair_training_log_if_needed(config: Path, shuffle: int) -> list[Path]:
    """Quarantine NUL-filled CSV logs so DLC can create a clean logger file."""
    repaired: list[Path] = []
    pattern = f"**/*shuffle{shuffle}/train/learning_stats.csv"
    for log_path in config.parent.glob(f"dlc-models-pytorch/{pattern}"):
        try:
            raw = log_path.read_bytes()
            first_line = raw.splitlines()[0].decode("utf-8") if raw.splitlines() else ""
            valid = b"\x00" not in raw and "step" in first_line.split(",")
        except Exception:
            valid = False
        if valid:
            continue
        backup = backup_path(config, f"corrupt-learning-stats-shuffle{shuffle}", ".csv")
        shutil.copy2(log_path, backup)
        log_path.unlink()
        repaired.append(backup)
        print(f"CORRUPT_TRAINING_LOG_BACKUP={backup}", flush=True)
    return repaired


def flatten_pytorch_updates(updates: dict, prefix: str = "") -> dict:
    """Convert convenient nested JSON to DLC's dot-separated update paths."""
    flattened = {}
    for key, value in updates.items():
        key = str(key)
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_pytorch_updates(value, path))
        else:
            flattened[path] = value
    return flattened


def selected_device(device: str) -> str:
    return detect_device() if device == "auto" else device


def activate_training_fraction(config: Path, active_fraction: float) -> int:
    """Keep every generated Shuffle fraction while making the active one index 0."""
    active_fraction = round(float(active_fraction), 2)
    cfg = read_config(str(config))
    existing = [round(float(value), 2) for value in (cfg.get("TrainingFraction") or [])]
    fractions = [active_fraction]
    fractions.extend(value for value in existing if not np.isclose(value, active_fraction))
    edit_project_config(config, {"TrainingFraction": fractions})
    return 0


def evaluation_trainingset_index(config: Path, shuffle: int, fallback: int = 0) -> int:
    """Resolve the config TrainingFraction index that belongs to a saved Shuffle."""
    repair_training_metadata_if_needed(config)
    metadata = TrainingDatasetMetadata.load(str(config), load_splits=False)
    matches = [item for item in metadata.shuffles if int(item.index) == int(shuffle)]
    if not matches:
        return int(fallback)
    target_fraction = round(float(matches[-1].train_fraction), 2)
    cfg = read_config(str(config))
    fractions = [round(float(value), 2) for value in (cfg.get("TrainingFraction") or [])]
    if not any(np.isclose(value, target_fraction) for value in fractions):
        fractions.append(target_fraction)
        edit_project_config(config, {"TrainingFraction": fractions})
    resolved = next(
        index for index, value in enumerate(fractions)
        if np.isclose(value, target_fraction)
    )
    print(
        f"EVALUATION_TRAINSET=shuffle{shuffle} fraction={target_fraction:.2f} index={resolved}",
        flush=True,
    )
    return resolved


def prepare(
    config: Path,
    shuffle: int,
    train_fraction: float,
    superanimal: str,
    pose_model: str,
    detector_name: str,
    memory_replay: bool,
    split_plan: Path | None,
) -> None:
    repair_training_metadata_if_needed(config)
    repair_training_log_if_needed(config, shuffle)
    pose_model = require_hagisilta_training_pose_model(pose_model)
    weight_init = build_weight_init(
        cfg=str(config),
        super_animal=superanimal,
        model_name=pose_model,
        detector_name=detector_name,
        with_decoder=False,
        memory_replay=memory_replay,
    )
    with temporarily_filter_excluded_labels(config) as excluded_count:
        print(f"REVIEW_EXCLUDED_FRAMES={excluded_count}", flush=True)
        train_indices = None
        validation_indices = None
        if split_plan is not None:
            split_plan = split_plan.resolve()
            if not split_plan.is_file():
                raise FileNotFoundError(f"Dataset split plan not found: {split_plan}")
            plan = json.loads(split_plan.read_text(encoding="utf-8"))
            cfg = read_config(str(config))
            training_folder = Path(cfg["project_path"]) / auxiliaryfunctions.get_training_set_folder(cfg)
            training_folder.mkdir(parents=True, exist_ok=True)
            merged = merge_annotateddatasets(cfg, str(training_folder))
            train_indices, validation_indices, test_indices, missing = indices_from_merged_data(
                merged.index, plan
            )
            if missing:
                preview = ", ".join(missing[:5])
                raise ValueError(
                    f"Split plan does not cover {len(missing)} labeled frames. "
                    f"Refresh and validate the dataset split first: {preview}"
                )
            if not train_indices or not validation_indices:
                raise ValueError(
                    "The split plan requires at least one Train frame and one Validation frame."
                )
            effective_fraction = len(train_indices) / (len(train_indices) + len(validation_indices))
            # DLC identifies a generated training set with an integer-percent
            # folder name (for example ``trainset80``) and writes the matching
            # two-decimal value (0.8) to metadata.yaml.  Keeping a more precise
            # value such as 0.79902 in config.yaml makes train_network look for
            # a different, nonexistent shuffle even though the exact custom
            # train/validation indices were generated correctly.
            dlc_training_fraction = round(effective_fraction, 2)
            activate_training_fraction(config, dlc_training_fraction)
            split_archive = project_state_root(config.parent, create=True) / "training-splits"
            split_archive.mkdir(parents=True, exist_ok=True)
            archived_plan = split_archive / (
                f"shuffle-{shuffle:02d}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
            )
            shutil.copy2(split_plan, archived_plan)
            frozen_plan = split_archive / f"shuffle-{shuffle:02d}-split.json"
            shutil.copy2(split_plan, frozen_plan)
            print(f"DATASET_SPLIT_PLAN={split_plan}", flush=True)
            print(f"DATASET_SPLIT_ARCHIVE={archived_plan}", flush=True)
            print(f"DATASET_SPLIT_FOR_SHUFFLE={frozen_plan}", flush=True)
            print(f"DATASET_TRAIN_FRAMES={len(train_indices)}", flush=True)
            print(f"DATASET_VALIDATION_FRAMES={len(validation_indices)}", flush=True)
            print(f"DATASET_RESERVED_TEST_FRAMES={len(test_indices)}", flush=True)
        else:
            activate_training_fraction(config, train_fraction)
        deeplabcut.create_training_dataset(
            str(config),
            num_shuffles=1,
            Shuffles=[shuffle],
            userfeedback=False,
            net_type=pose_model,
            detector_type=detector_name,
            weight_init=weight_init,
            trainIndices=[train_indices] if train_indices is not None else None,
            testIndices=[validation_indices] if validation_indices is not None else None,
        )


def train(
    config: Path,
    epochs: int,
    batch_size: int,
    shuffle: int,
    detector_batch_size: int,
    pose_threshold: float,
    learning_rate: float,
    augmentation_strength: str,
    input_size: int,
    validation_interval: int,
    random_seed: int,
    max_snapshots: int,
    save_epochs: int,
    superanimal: str,
    pose_model: str,
    device: str,
    display_iters: int,
    detector_epochs: int,
    detector_save_epochs: int,
    dataloader_workers: int,
    persistent_workers: bool,
    pin_memory: bool,
    freeze_bn_stats: bool,
    pytorch_overrides: dict,
    base_snapshot: Path | None,
    requested_input_size: tuple[int, int] | None = None,
) -> None:
    # DLC 3.0 expects dot-separated paths here. Passing a nested
    # ``{"train_settings": {...}}`` replaces that complete section and removes
    # required defaults such as ``seed``, which prevents training from starting.
    effective_overrides = flatten_pytorch_updates(pytorch_overrides)
    validate_input_overrides(effective_overrides)
    effective_overrides["train_settings.dataloader_workers"] = dataloader_workers
    effective_overrides["train_settings.dataloader_pin_memory"] = pin_memory
    effective_overrides["runner.optimizer.params.lr"] = learning_rate
    effective_overrides["model.backbone.freeze_bn_stats"] = freeze_bn_stats
    # Replace the complete input sections so old crop/resize/collate settings
    # cannot silently survive in a prepared Shuffle. Network weights are unchanged.
    effective_overrides["data.train"] = training_data_config(augmentation_strength, requested_input_size)
    effective_overrides["data.inference"] = inference_data_config()
    if input_size:
        print(f"LEGACY_INPUT_SIZE_IGNORED={input_size}:policy=full-roi-pad-v1", flush=True)
    effective_overrides["runner.eval_interval"] = validation_interval
    effective_overrides["train_settings.seed"] = random_seed
    transfer_snapshot = None
    if base_snapshot is not None:
        base_snapshot = base_snapshot.resolve()
        if not base_snapshot.is_file():
            raise FileNotFoundError(f"Base model snapshot not found: {base_snapshot}")
        # A DLC resume snapshot also contains the old epoch, optimizer, and scheduler.
        # For a new Shuffle we only want its learned pose weights: the enlarged GT
        # dataset must start a fresh optimizer and run exactly the requested epochs.
        source = torch.load(base_snapshot, map_location="cpu", weights_only=True)
        if not isinstance(source, dict) or "model" not in source:
            raise ValueError(f"Invalid DLC pose snapshot: {base_snapshot}")
        runtime = project_state_root(config.parent, create=True) / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        transfer_snapshot = runtime / f"transfer-init-shuffle{shuffle}.pt"
        torch.save(
            {
                "model": source["model"],
                "metadata": {"epoch": 0, "source_snapshot": str(base_snapshot)},
            },
            transfer_snapshot,
        )
        print(f"BASE_SNAPSHOT={base_snapshot}", flush=True)
        print(f"TRANSFER_WEIGHTS={transfer_snapshot}", flush=True)
    persistent_enabled = install_persistent_dataloader_workers(
        persistent_workers and dataloader_workers > 0
    )
    print(
        f"DATALOADER_WORKERS={dataloader_workers} "
        f"PERSISTENT_WORKERS={persistent_enabled} PIN_MEMORY={pin_memory}",
        flush=True,
    )
    with whole_roi_training():
        deeplabcut.train_network(
            str(config),
            shuffle=shuffle,
            trainingsetindex=0,
            epochs=epochs,
            save_epochs=save_epochs or max(10, epochs // 5),
            max_snapshots_to_keep=max_snapshots,
            display_iters=display_iters,
            device=selected_device(device),
            batch_size=batch_size,
            detector_batch_size=detector_batch_size,
            detector_epochs=detector_epochs or None,
            detector_save_epochs=detector_save_epochs or None,
            pose_threshold=pose_threshold,
            pytorch_cfg_updates=effective_overrides,
            superanimal_name=superanimal,
            superanimal_transfer_learning=True,
            snapshot_path=str(transfer_snapshot) if transfer_snapshot else None,
        )


def evaluate(
    config: Path,
    shuffle: int,
    trainingset_index: int,
    plotting: bool,
    show_errors: bool,
    per_keypoint: bool,
    pcutoff: float,
    bodyparts: str,
    rescale: bool,
    snapshots: str,
    device: str,
) -> None:
    trainingset_index = evaluation_trainingset_index(
        config, shuffle, trainingset_index
    )
    comparison_bodyparts: str | list[str] = "all"
    if bodyparts.strip().lower() != "all":
        comparison_bodyparts = [part.strip() for part in bodyparts.split(",") if part.strip()]
    snapshots_to_evaluate = [item.strip() for item in snapshots.split(",") if item.strip()] or None
    deeplabcut.evaluate_network(
        str(config),
        shuffles=[shuffle],
        trainingsetindex=trainingset_index,
        plotting=plotting,
        show_errors=show_errors,
        comparison_bodyparts=comparison_bodyparts,
        rescale=rescale,
        per_keypoint_evaluation=per_keypoint,
        snapshots_to_evaluate=snapshots_to_evaluate,
        pcutoff=pcutoff,
        device=selected_device(device),
    )


def evaluate_reserved_test(
    config: Path,
    shuffle: int,
    split_plan: Path,
    pcutoff: float,
    device: str,
) -> Path:
    """Evaluate the final model on HAGISILTA's untouched third split."""
    split_plan = split_plan.resolve()
    plan = json.loads(split_plan.read_text(encoding="utf-8"))
    cfg = read_config(str(config))
    training_folder = Path(cfg["project_path"]) / auxiliaryfunctions.get_training_set_folder(cfg)
    # Use the exact same reviewed-label view as dataset preparation. Without
    # this filter, frames the user excluded during label review are merged back
    # in and incorrectly reported as absent from the frozen Test split.
    with temporarily_filter_excluded_labels(config) as excluded_count:
        print(f"TEST_REVIEW_EXCLUDED_FRAMES={excluded_count}", flush=True)
        merged = merge_annotateddatasets(cfg, str(training_folder))
    _train, _validation, test_indices, missing = indices_from_merged_data(merged.index, plan)
    if missing:
        # The UI saves the split plan atomically, but an evaluation worker can
        # remain alive while the user confirms a newer dataset version.  Read
        # the plan once more before treating the mismatch as corrupt data.
        refreshed_plan = json.loads(split_plan.read_text(encoding="utf-8"))
        if refreshed_plan != plan:
            plan = refreshed_plan
            _train, _validation, test_indices, missing = indices_from_merged_data(
                merged.index, plan
            )
            print("TEST_SPLIT_PLAN_REFRESHED=1", flush=True)
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"Final Test split does not cover {len(missing)} labeled frames: {preview}"
        )
    if not test_indices:
        raise ValueError("Final Test split has no frames.")

    project = Path(cfg["project_path"])
    test_images: list[Path] = []
    index_by_path: dict[str, int] = {}
    for position in test_indices:
        raw_index = merged.index[position]
        parts = tuple(map(str, raw_index)) if isinstance(raw_index, tuple) else (str(raw_index),)
        image_path = project.joinpath(*parts).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        test_images.append(image_path)
        index_by_path[str(image_path).casefold()] = position

    output = project_state_root(config.parent, create=True) / "test-evaluations" / f"shuffle-{shuffle:02d}"
    output.mkdir(parents=True, exist_ok=True)
    trainingset_index = evaluation_trainingset_index(config, shuffle, 0)
    predictions = deeplabcut.analyze_images(
        str(config),
        test_images,
        destfolder=output,
        shuffle=shuffle,
        trainingsetindex=trainingset_index,
        device=selected_device(device),
        save_as_csv=True,
        pcutoff=pcutoff,
    )
    if not predictions:
        raise RuntimeError("No predictions were produced for the final Test split.")

    column_names = list(merged.columns.names) if hasattr(merged.columns, "names") else []
    bodypart_level = column_names.index("bodyparts") if "bodyparts" in column_names else len(column_names) - 2
    coordinate_level = column_names.index("coords") if "coords" in column_names else len(column_names) - 1
    bodyparts = list(map(str, cfg.get("bodyparts", []) or []))
    columns: dict[str, dict[str, object]] = {part: {} for part in bodyparts}
    for column in merged.columns:
        part = str(column[bodypart_level])
        coordinate = str(column[coordinate_level]).lower()
        if part in columns and coordinate in {"x", "y"}:
            columns[part][coordinate] = column

    errors: list[float] = []
    normalized_errors: list[float] = []
    confident_errors: list[float] = []
    per_keypoint: dict[str, dict[str, list[float] | int]] = {
        part: {
            "errors": [], "normalized_errors": [], "confident_errors": [],
            "points": 0, "predicted_points": 0, "confident_points": 0,
            "pck_3": 0, "pck_5": 0, "pck_10": 0,
        }
        for part in bodyparts
    }
    frame_records: list[dict[str, object]] = []
    metric_ground_truth: dict[str, np.ndarray] = {}
    metric_predictions: dict[str, np.ndarray] = {}
    for prediction_path, payload in predictions.items():
        position = index_by_path.get(str(Path(prediction_path).resolve()).casefold())
        if position is None:
            continue
        pose = np.asarray(payload.get("bodyparts"), dtype=float)
        while pose.ndim > 2:
            pose = pose[0]
        if pose.ndim != 2:
            continue
        ground_truth = merged.iloc[position]
        try:
            with Image.open(prediction_path) as image:
                image_width, image_height = image.size
        except (OSError, ValueError):
            image_width = image_height = 0
        roi_diagonal = float(np.hypot(image_width, image_height))
        ground_truth_pose = np.zeros((1, len(bodyparts), 3), dtype=float)
        for bodypart_index, part in enumerate(bodyparts):
            if "x" not in columns[part] or "y" not in columns[part]:
                continue
            target_x = float(ground_truth[columns[part]["x"]])
            target_y = float(ground_truth[columns[part]["y"]])
            if not np.isfinite(target_x) or not np.isfinite(target_y):
                continue
            ground_truth_pose[0, bodypart_index] = (target_x, target_y, 2.0)
            per_keypoint[part]["points"] += 1
            predicted_x = predicted_y = likelihood = float("nan")
            if bodypart_index < len(pose):
                predicted_x = float(pose[bodypart_index, 0])
                predicted_y = float(pose[bodypart_index, 1])
                likelihood = float(pose[bodypart_index, 2]) if pose.shape[1] > 2 else 1.0
            predicted = bool(np.isfinite(predicted_x) and np.isfinite(predicted_y))
            confident = bool(predicted and np.isfinite(likelihood) and likelihood >= pcutoff)
            error = (
                float(np.hypot(predicted_x - target_x, predicted_y - target_y))
                if predicted else None
            )
            normalized_error = (
                float(error / roi_diagonal * 100.0)
                if error is not None and roi_diagonal > 0 else None
            )
            if predicted and error is not None:
                errors.append(error)
                per_keypoint[part]["errors"].append(error)
                per_keypoint[part]["predicted_points"] += 1
                if normalized_error is not None:
                    normalized_errors.append(normalized_error)
                    per_keypoint[part]["normalized_errors"].append(normalized_error)
                for threshold, key in ((3.0, "pck_3"), (5.0, "pck_5"), (10.0, "pck_10")):
                    if error <= threshold:
                        per_keypoint[part][key] += 1
            if confident and error is not None:
                confident_errors.append(error)
                per_keypoint[part]["confident_errors"].append(error)
                per_keypoint[part]["confident_points"] += 1
            frame_records.append({
                "frame": Path(prediction_path).name,
                "image_path": str(Path(prediction_path).resolve()),
                "bodypart": part,
                "image_width": image_width,
                "image_height": image_height,
                "roi_diagonal_px": roi_diagonal,
                "ground_truth_x": target_x,
                "ground_truth_y": target_y,
                "predicted_x": predicted_x if predicted else "",
                "predicted_y": predicted_y if predicted else "",
                "likelihood": likelihood if np.isfinite(likelihood) else "",
                "error_px": error if error is not None else "",
                "normalized_error_percent": normalized_error if normalized_error is not None else "",
                "confident": int(confident),
                "dropout": int(not confident),
                "within_3px": int(error is not None and error <= 3.0),
                "within_5px": int(error is not None and error <= 5.0),
                "within_10px": int(error is not None and error <= 10.0),
            })
        visible_keypoints = int(np.count_nonzero(ground_truth_pose[0, :, 2] > 0))
        if visible_keypoints >= 2 and pose.shape[1] >= 2:
            prediction_pose = np.zeros((1, len(bodyparts), 3), dtype=float)
            copied_keypoints = min(len(bodyparts), len(pose))
            copied_coordinates = min(3, pose.shape[1])
            prediction_pose[0, :copied_keypoints, :copied_coordinates] = pose[
                :copied_keypoints, :copied_coordinates
            ]
            metric_key = str(Path(prediction_path).resolve())
            metric_ground_truth[metric_key] = ground_truth_pose
            metric_predictions[metric_key] = prediction_pose
    if not errors:
        raise RuntimeError("Final Test split contains no comparable labeled keypoints.")

    def rmse(values: list[float]) -> float | None:
        return float(np.sqrt(np.mean(np.square(values)))) if values else None

    def median(values: list[float]) -> float | None:
        return float(np.median(values)) if values else None

    keypoint_report = {}
    for part, values in per_keypoint.items():
        points = int(values["points"])
        confident_points = int(values["confident_points"])
        keypoint_report[part] = {
            "points": points,
            "predicted_points": int(values["predicted_points"]),
            "confident_points": confident_points,
            "coverage_at_pcutoff": confident_points / points if points else 0.0,
            "dropout_rate": 1.0 - (confident_points / points if points else 0.0),
            "rmse_px": rmse(values["errors"]),
            "median_error_px": median(values["errors"]),
            "normalized_error_percent": median(values["normalized_errors"]),
            "pck_3px": int(values["pck_3"]) / points if points else 0.0,
            "pck_5px": int(values["pck_5"]) / points if points else 0.0,
            "pck_10px": int(values["pck_10"]) / points if points else 0.0,
            "confident_rmse_px": rmse(values["confident_errors"]),
        }
    pose_map = None
    pose_mar = None
    if metric_ground_truth:
        try:
            from deeplabcut.core.metrics import compute_metrics

            pose_metrics = compute_metrics(
                metric_ground_truth,
                metric_predictions,
                single_animal=True,
                pcutoff=pcutoff,
                compute_detection_rmse=False,
            )
            if np.isfinite(pose_metrics.get("mAP", float("nan"))):
                pose_map = float(pose_metrics["mAP"])
            if np.isfinite(pose_metrics.get("mAR", float("nan"))):
                pose_mar = float(pose_metrics["mAR"])
        except Exception as exc:
            print(f"FINAL_TEST_OKS_WARNING={exc}", flush=True)
    result = {
        "schema_version": 3,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "shuffle": shuffle,
        "dataset_version": int(plan.get("dataset_version", 1) or 1),
        "split_plan": str(split_plan.resolve()),
        "test_frames": len(test_images),
        "labeled_points": len(frame_records),
        "predicted_points": len(errors),
        "pcutoff": float(pcutoff),
        "coverage_at_pcutoff": sum(int(record["confident"]) for record in frame_records) / len(frame_records),
        "dropout_rate": 1.0 - (sum(int(record["confident"]) for record in frame_records) / len(frame_records)),
        "rmse_px": rmse(errors),
        "median_error_px": median(errors),
        "normalized_error_percent": median(normalized_errors),
        "pck_3px": sum(int(record["within_3px"]) for record in frame_records) / len(frame_records),
        "pck_5px": sum(int(record["within_5px"]) for record in frame_records) / len(frame_records),
        "pck_10px": sum(int(record["within_10px"]) for record in frame_records) / len(frame_records),
        "confident_rmse_px": rmse(confident_errors),
        "mAP": pose_map,
        "mAR": pose_mar,
        "per_keypoint": keypoint_report,
    }
    result_path = output / "final-test-results.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = output / "pose-validation-summary.csv"
    summary_fields = [
        "evaluated_at", "shuffle", "dataset_version", "test_frames", "labeled_points",
        "predicted_points",
        "pcutoff", "rmse_px", "median_error_px", "normalized_error_percent",
        "pck_3px", "pck_5px", "pck_10px", "coverage_at_pcutoff", "dropout_rate",
        "mAP", "mAR",
    ]
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerow({key: result.get(key) for key in summary_fields})
    bodypart_path = output / "pose-validation-by-bodypart.csv"
    bodypart_fields = [
        "bodypart", "points", "predicted_points", "confident_points",
        "rmse_px", "median_error_px", "normalized_error_percent",
        "pck_3px", "pck_5px", "pck_10px", "coverage_at_pcutoff", "dropout_rate",
        "confident_rmse_px",
    ]
    with bodypart_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=bodypart_fields)
        writer.writeheader()
        for part, values in keypoint_report.items():
            writer.writerow({"bodypart": part, **values})
    frames_path = output / "pose-validation-frames.csv"
    with frames_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_records[0]))
        writer.writeheader()
        writer.writerows(frame_records)
    result["artifacts"] = {
        "summary_csv": str(summary_path.resolve()),
        "bodypart_csv": str(bodypart_path.resolve()),
        "frames_csv": str(frames_path.resolve()),
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"FINAL_TEST_RESULTS={result_path}", flush=True)
    print(f"FINAL_TEST_FRAMES={len(test_images)}", flush=True)
    print(f"FINAL_TEST_RMSE={result['rmse_px']:.6f}", flush=True)
    if pose_map is not None:
        print(f"FINAL_TEST_MAP={pose_map:.6f}", flush=True)
    if pose_mar is not None:
        print(f"FINAL_TEST_MAR={pose_mar:.6f}", flush=True)
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare, train, or evaluate a HAGISILTA DLC project model.")
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "action",
        choices=["prepare", "train", "fit", "evaluate", "evaluate-test", "all"],
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shuffle", type=int, default=DEFAULT_SHUFFLE)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--detector-batch-size", type=int, default=1)
    parser.add_argument("--pose-threshold", type=float, default=0.1)
    parser.add_argument("--max-snapshots", type=int, default=3)
    parser.add_argument("--save-epochs", type=int, default=0)
    parser.add_argument("--superanimal", default="superanimal_topviewmouse")
    parser.add_argument(
        "--pose-model",
        choices=HAGISILTA_BU_TRAINING_POSE_MODEL_IDS,
        default="hrnet_w32",
    )
    parser.add_argument("--detector-name", default="fasterrcnn_resnet50_fpn_v2")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--memory-replay", action="store_true")
    parser.add_argument(
        "--split-plan",
        type=Path,
        default=None,
        help="Frozen HAGISILTA per-frame Train/Validation/Test plan. Test frames are reserved.",
    )
    parser.add_argument("--display-iters", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument(
        "--augmentation-strength",
        choices=["weak", "standard", "strong"],
        default="standard",
    )
    parser.add_argument("--input-size", type=int, default=0,
                        help="Legacy argument; whole-ROI training calculates its input size automatically.")
    parser.add_argument("--input-width", type=int, default=None,
                        help="Label-safe training crop width; no resize-to-fit. Omit both dimensions for full ROI.")
    parser.add_argument("--input-height", type=int, default=None,
                        help="Label-safe training crop height; validated against Train labels and augmentation.")
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--detector-epochs", type=int, default=0)
    parser.add_argument("--detector-save-epochs", type=int, default=0)
    parser.add_argument(
        "--dataloader-workers",
        type=int,
        default=4,
        help=(
            "PyTorch DataLoader child-process count. Zero disables child processes; "
            "OpenCV/NumPy/PyTorch may still use multiple CPU threads."
        ),
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep DataLoader child processes alive between training epochs.",
    )
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-bn-stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pytorch-overrides-json", default="{}")
    parser.add_argument(
        "--base-snapshot",
        type=Path,
        default=None,
        help="Initialize a new Shuffle from this trained DLC pose snapshot (weights only).",
    )
    parser.add_argument("--eval-trainingset-index", type=int, default=0)
    parser.add_argument("--eval-pcutoff", type=float, default=0.6)
    parser.add_argument("--eval-bodyparts", default="all")
    parser.add_argument("--eval-snapshots", default="")
    parser.add_argument("--no-eval-plotting", action="store_true")
    parser.add_argument("--no-eval-show-errors", action="store_true")
    parser.add_argument("--no-eval-per-keypoint", action="store_true")
    parser.add_argument("--eval-rescale", action="store_true")
    args = parser.parse_args()
    requested_input_size = None
    if args.input_width is not None or args.input_height is not None:
        from hagisilta_training_input import validate_canvas_size
        try:
            requested_input_size = validate_canvas_size((args.input_width, args.input_height))
        except ValueError as error:
            parser.error(str(error))
    config = args.config.resolve()
    if not config.is_file():
        raise FileNotFoundError(config)

    if args.action in {"prepare", "fit", "all"}:
        prepare(
            config,
            args.shuffle,
            args.train_fraction,
            args.superanimal,
            args.pose_model,
            args.detector_name,
            args.memory_replay,
            args.split_plan,
        )
    if args.action in {"train", "fit", "all"}:
        train(
            config,
            args.epochs,
            args.batch_size,
            args.shuffle,
            args.detector_batch_size,
            args.pose_threshold,
            args.learning_rate,
            args.augmentation_strength,
            args.input_size,
            args.validation_interval,
            args.random_seed,
            args.max_snapshots,
            args.save_epochs,
            args.superanimal,
            args.pose_model,
            args.device,
            args.display_iters,
            args.detector_epochs,
            args.detector_save_epochs,
            args.dataloader_workers,
            args.persistent_workers,
            args.pin_memory,
            args.freeze_bn_stats,
            json.loads(args.pytorch_overrides_json),
            args.base_snapshot,
            requested_input_size,
        )
    if args.action in {"evaluate", "all"}:
        evaluate(
            config,
            args.shuffle,
            args.eval_trainingset_index,
            not args.no_eval_plotting,
            not args.no_eval_show_errors,
            not args.no_eval_per_keypoint,
            args.eval_pcutoff,
            args.eval_bodyparts,
            args.eval_rescale,
            args.eval_snapshots,
            args.device,
        )
    if args.action in {"evaluate", "evaluate-test", "all"} and args.split_plan is not None:
        # Validation is tied to each Shuffle's training run. Re-evaluation compares
        # models only on the currently confirmed common Test split.
        evaluate_reserved_test(
            config,
            args.shuffle,
            args.split_plan.resolve(),
            args.eval_pcutoff,
            args.device,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
