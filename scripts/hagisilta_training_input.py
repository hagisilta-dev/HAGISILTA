# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
"""Whole-ROI or label-safe crop geometry; image/label files are never rewritten.

The lightweight size helpers are shared with the desktop UI. DLC/torch imports
are deliberately deferred to the training worker, never the UI process.
"""
from __future__ import annotations

from hagisilta_i18n import tr

import copy
import json
import math
from contextlib import contextmanager
from functools import lru_cache
from enum import Enum
from pathlib import Path

POLICY = "full-roi-pad-v1"
FIT_POLICY = "full-roi-fit-v2"
CROP_POLICY = "label-safe-crop-v1"
CROP_MARGIN = 32
PRESETS = {
    "weak": {"p": 0.35, "rotation": 15, "scaling": [0.8, 1.15], "noise": 6.0, "blur": False},
    "standard": {"p": 0.5, "rotation": 30, "scaling": [0.5, 1.25], "noise": 12.75, "blur": True},
    "strong": {"p": 0.7, "rotation": 45, "scaling": [0.5, 1.4], "noise": 20.0, "blur": True},
}


def _json_value(value):
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Unsupported input-layout value: {type(value).__name__}")


def ceil32(value: float) -> int:
    return max(32, math.ceil(value / 32) * 32)


def validate_canvas_size(size) -> tuple[int, int]:
    if len(size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 32 or v % 32 for v in size):
        raise ValueError(tr('가로·세로를 각각 32 이상의 32배수로 입력하세요.'))
    return tuple(size)


def input_layout(sizes: list[tuple[int, int]], strength: str, requested_size=None,
                 *, label_points=None, policy=None) -> dict:
    """Bound every rotated/scaled ROI or labelled extent, not one example.

For 0 <= angle <= r, the horizontal extent is w*cos(angle)+h*sin(angle).
Its maximum occurs at an endpoint or atan(h/w); the vertical bound swaps w,h.
The extra four pixels cover affine output rounding and interpolation support.
"""
    preset = PRESETS[strength]
    if not sizes or any(w <= 0 or h <= 0 for w, h in sizes):
        raise ValueError(tr('Train 이미지 크기를 확인할 수 없습니다.'))
    rotation = math.radians(preset["rotation"])

    def extent(w, h):
        angle = min(rotation, math.atan2(h, w))
        return max(w, preset["scaling"][1] * (w * math.cos(angle) + h * math.sin(angle)))

    width = ceil32(max(extent(w, h) for w, h in sizes) + 4)
    height = ceil32(max(extent(h, w) for w, h in sizes) + 4)
    minimum_width, minimum_height = width, height
    if requested_size is not None:
        width, height = validate_canvas_size(requested_size)
        if policy not in (FIT_POLICY, POLICY):
            if label_points is None or len(label_points) != len(sizes):
                raise ValueError(tr('크롭 크기를 확인하려면 모든 Train 이미지의 라벨이 필요합니다.'))
            spans = []
            for (native_w, native_h), points in zip(sizes, label_points):
                visible = [(float(x), float(y)) for x, y in points if math.isfinite(x) and math.isfinite(y)]
                if any(not (0 <= x < native_w and 0 <= y < native_h) for x, y in visible):
                    raise ValueError(tr('원본 이미지 밖에 있는 Train 라벨을 먼저 교정하세요.'))
                if visible:
                    xs, ys = zip(*visible)
                    spans.append((max(xs) - min(xs) + 1, max(ys) - min(ys) + 1))
            crop_min_w = ceil32(max((extent(w, h) for w, h in spans), default=0) + 2 * CROP_MARGIN + 4)
            crop_min_h = ceil32(max((extent(h, w) for w, h in spans), default=0) + 2 * CROP_MARGIN + 4)
            if width < crop_min_w or height < crop_min_h:
                raise ValueError(tr('라벨과 여유 공간 보존에 최소 {v0} × {v1} px가 필요합니다 (회전·확대 포함). 크롭을 늘리거나 증강 강도를 낮추세요. 자동 축소하지 않습니다.', v0=crop_min_w, v1=crop_min_h))
            return {
                "policy": CROP_POLICY, "strength": strength, "train_frames": len(sizes),
                "native_max_width": max(w for w, h in sizes), "native_max_height": max(h for w, h in sizes),
                "width": width, "height": height, "minimum_width": crop_min_w, "minimum_height": crop_min_h,
                "size_mode": "manual", "requested_size": list(requested_size),
                "rotation_degrees": preset["rotation"], "scale_range": list(preset["scaling"]),
                "padding_position": "center", "padding_border": "constant", "crop": True,
                "crop_margin": CROP_MARGIN, "resize_to_fit": False, "resize_scale": 1.0,
                "labelled_frames": len(spans),
            }
    fit_scale = min(1.0, width / minimum_width, height / minimum_height)
    # Resize the complete automatic canvas by ONE constant isotropic factor.
    # Fitting each random affine output separately would cancel its scale jitter.
    longest = max(minimum_width, minimum_height)
    fit_longest = max(1, math.floor(longest * fit_scale))
    fit_scale = fit_longest / longest
    resize = fit_scale < 1.0
    return {
        "policy": FIT_POLICY if resize else POLICY, "strength": strength, "train_frames": len(sizes),
        "native_max_width": max(w for w, h in sizes),
        "native_max_height": max(h for w, h in sizes),
        "width": width, "height": height,
        "minimum_width": minimum_width, "minimum_height": minimum_height,
        "size_mode": "manual" if requested_size is not None else "auto",
        "requested_size": list(requested_size) if requested_size is not None else None,
        "rotation_degrees": preset["rotation"], "scale_range": list(preset["scaling"]),
        "padding_position": "center", "padding_border": "constant",
        "crop": False, "resize_to_fit": resize,
        "resize_scale": fit_scale, "resize_longest_side": fit_longest,
    }


@lru_cache(maxsize=20000)
def _image_size(path: str, mtime_ns: int, byte_size: int) -> tuple[int, int]:
    from PIL import Image
    with Image.open(path) as image:
        return image.size


def image_sizes(paths) -> list[tuple[int, int]]:
    result = []
    for path in paths:
        path = Path(path)
        stamp = path.stat()  # Missing files must not silently shrink the canvas.
        result.append(_image_size(str(path.resolve()), stamp.st_mtime_ns, stamp.st_size))
    return result


def planned_train_images(project: Path, plan: dict) -> list[Path]:
    result = []
    root = (project / "labeled-data").resolve()
    for folder, bundle in (plan.get("bundles") or {}).items():
        if not bundle.get("enabled", True):
            continue
        for frame, role in (bundle.get("assignments") or {}).items():
            if role != "train":
                continue
            path = (root / folder / frame).resolve()
            if not path.is_relative_to(root):
                raise ValueError(tr('라벨 이미지 경로가 프로젝트 labeled-data 밖에 있습니다.'))
            result.append(path)
    return result


@lru_cache(maxsize=128)
def _label_rows(path: str, mtime_ns: int, byte_size: int, bodyparts: tuple) -> dict:
    """Read only the scorer's annotation table; never guesses between scorers."""
    import pandas as pd
    table = pd.read_hdf(path)
    if not isinstance(table.columns, pd.MultiIndex):
        raise ValueError(tr('라벨 열 형식을 확인하세요: {v0}', v0=Path(path).name))
    coord_level = table.columns.names.index("coords") if "coords" in table.columns.names else table.columns.nlevels - 1
    part_level = table.columns.names.index("bodyparts") if "bodyparts" in table.columns.names else coord_level - 1
    pairs = {}
    found_parts = set()
    for column in table.columns:
        if bodyparts and column[part_level] not in bodyparts:
            continue
        if column[coord_level] in ("x", "y"):
            found_parts.add(column[part_level])
            key = tuple(value for i, value in enumerate(column) if i != coord_level)
            pairs.setdefault(key, {})[column[coord_level]] = column
    if (not pairs or set(bodyparts) - found_parts or table.columns.has_duplicates
            or any(set(pair) != {"x", "y"} for pair in pairs.values())):
        raise ValueError(tr('신체점 x/y 라벨 열을 확인하세요: {v0}', v0=Path(path).name))
    result = {}
    for index, row in table.iterrows():
        value = index[-1] if isinstance(index, tuple) else index
        name = str(value).replace("\\", "/").rsplit("/", 1)[-1]
        if name in result:
            raise ValueError(tr('중복 라벨 행을 확인하세요: {v0}', v0=name))
        points = [(float(row[pair["x"]]), float(row[pair["y"]])) for pair in pairs.values()]
        if any((math.isfinite(x) != math.isfinite(y)) or math.isinf(x) or math.isinf(y) for x, y in points):
            raise ValueError(tr('불완전한 x/y 라벨을 확인하세요: {v0}', v0=name))
        result[name] = tuple((x, y) for x, y in points if math.isfinite(x) and math.isfinite(y))
    return result


def training_label_points(config: Path, paths) -> list:
    import yaml
    cfg = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    bodyparts = cfg.get("multianimalbodyparts") or cfg.get("bodyparts") or []
    bodyparts = tuple(bodyparts + (cfg.get("uniquebodyparts") or []))
    scorer = str(cfg.get("scorer") or "")
    tables = {}
    result = []
    for path in map(Path, paths):
        if path.parent not in tables:
            candidates = ([path.parent / f"CollectedData_{scorer}.h5"] if scorer
                          else list(path.parent.glob("CollectedData_*.h5")))
            if len(candidates) != 1 or not candidates[0].is_file():
                raise ValueError(tr('Train 라벨 파일을 하나로 확인할 수 없습니다: {v0}', v0=path.parent.name))
            label = candidates[0]
            stat = label.stat()
            tables[path.parent] = _label_rows(str(label.resolve()), stat.st_mtime_ns, stat.st_size, bodyparts)
        if path.name not in tables[path.parent]:
            raise ValueError(tr('Train 이미지의 라벨 행이 없습니다: {v0}/{v1}', v0=path.parent.name, v1=path.name))
        result.append(tables[path.parent][path.name])
    return result


def dataset_label_points(dataset) -> list:
    """Use actual worker Train/replay annotations, never Validation/Test labels."""
    import numpy as np
    result = []
    grouped = {}
    for annotation in dataset.annotations:
        values = np.asarray(annotation.get("keypoints", []), dtype=float).reshape(-1, 3)
        points = values[values[:, 2] > 0, :2]
        if not np.isfinite(points).all():
            raise ValueError(tr('표시된 Train 신체점에 유효하지 않은 좌표가 있습니다.'))
        grouped.setdefault(annotation["image_id"], []).extend(map(tuple, points))
    for item in dataset.images:
        result.append(grouped.get(item["id"], []))
    return result


def training_data_config(strength: str, requested_size=None) -> dict:
    preset = PRESETS[strength]
    if requested_size is not None:
        requested_size = list(validate_canvas_size(requested_size))
    return {
        "affine": {"p": preset["p"], "rotation": preset["rotation"],
                   "scaling": list(preset["scaling"]), "translation": 0, "fit_output": True},
        "gaussian_noise": preset["noise"], "motion_blur": preset["blur"],
        "normalize_images": True, "crop_sampling": None, "collate": None,
        "hagisilta_full_roi": {"policy": CROP_POLICY if requested_size else POLICY,
                               "strength": strength, "requested_size": requested_size},
    }


def inference_data_config() -> dict:
    # Saved on NEW Shuffles only. DLC's ordinary inference path understands all
    # these keys, including the inverse coordinate offset after center padding.
    return {"normalize_images": True, "auto_padding": {
        "pad_width_divisor": 32, "pad_height_divisor": 32,
        "position": "center", "border_mode": "constant",
    }}


def validate_input_overrides(flattened: dict) -> None:
    unsafe = [key for key in flattened if key == "data" or key.startswith("data.")]
    if unsafe:
        raise ValueError(
            tr('학습 입력·라벨 변환 설정은 자동 관리합니다. 전문가 JSON에서 다음 항목을 제거하세요: ') + ", ".join(sorted(unsafe))
        )


def build_full_roi_transform(layout: dict, *, photometric: bool = True):
    """A serializable Albumentations pipeline for Windows workers.

Unlike DLC 3.0's default builder, Affine expands its output canvas before
padding. New manual inputs crop around all visible labels without fit-resizing.
Automatic inputs retain the whole ROI. Historical fit-policy records remain
reproducible, but are no longer produced by the manual-input UI.
"""
    import albumentations as A
    import cv2
    preset = PRESETS[layout["strength"]]
    crop = layout.get("policy") == CROP_POLICY
    resize = layout.get("resize_to_fit", False)
    pad_width = layout["minimum_width"] if resize else layout["width"]
    pad_height = layout["minimum_height"] if resize else layout["height"]
    transforms = [
        A.Affine(p=preset["p"], scale=tuple(preset["scaling"]),
                 rotate=(-preset["rotation"], preset["rotation"]),
                 keep_ratio=True, fit_output=True, mode=cv2.BORDER_CONSTANT, cval=0),
        A.PadIfNeeded(min_width=pad_width, min_height=pad_height,
                      position="center", border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
    ]
    if crop:
        from hagisilta_training_crop import LabelSafeCrop
        # A fixed border makes the full margin feasible even for edge labels.
        transforms.insert(1, A.CropAndPad(px=layout["crop_margin"], keep_size=False,
                                        pad_mode=cv2.BORDER_CONSTANT, pad_cval=0, p=1.0))
        transforms.append(LabelSafeCrop(layout["width"], layout["height"], layout["crop_margin"]))
    if resize:
        transforms.extend([
            A.LongestMaxSize(max_size=layout["resize_longest_side"], interpolation=cv2.INTER_AREA, p=1.0),
            A.PadIfNeeded(min_width=layout["width"], min_height=layout["height"],
                          position="center", border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
        ])
    if photometric:
        if preset["blur"]:
            transforms.append(A.MotionBlur(blur_limit=(3, 7), p=0.5))
        transforms.append(A.GaussNoise(var_limit=(0, preset["noise"] ** 2), mean=0, per_channel=True, p=0.5))
        transforms.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return A.Compose(transforms,
        keypoint_params=A.KeypointParams("xy", remove_invisible=False, label_fields=["class_labels"]),
        bbox_params=A.BboxParams(format="coco", label_fields=["bbox_labels"]))


@contextmanager
def whole_roi_training():
    """Use DLC's supported transform argument without editing site-packages.

Resolve dimensions from the actual DLC Train split (also after memory replay),
before allocating a model. Store policy, geometry and the serialized pipeline
next to that new model so the exact input procedure can be reproduced.
"""
    import albumentations as A
    from deeplabcut.pose_estimation_pytorch.apis import training as api
    from deeplabcut.pose_estimation_pytorch.task import Task
    original = api.train

    def train_full_roi(loader, run_config, task, **kwargs):
        marker = run_config.get("data", {}).get("train", {}).get("hagisilta_full_roi")
        if not marker or task == Task.DETECT:
            return original(loader=loader, run_config=run_config, task=task, **kwargs)
        if marker.get("policy") not in (POLICY, FIT_POLICY, CROP_POLICY) or task != Task.BOTTOM_UP:
            raise ValueError(tr('지원하지 않는 학습 입력 설정입니다.'))
        if kwargs.get("transform") is not None:
            raise ValueError(tr('학습 입력의 이미지 변환을 중복 지정할 수 없습니다.'))
        dataset = loader.create_dataset(mode="train", task=task)
        paths = [Path(item["file_name"]) for item in dataset.images]
        sizes = image_sizes(paths)
        labels = dataset_label_points(dataset) if marker.get("policy") == CROP_POLICY else None
        layout = input_layout(sizes, marker["strength"], marker.get("requested_size"),
                              label_points=labels, policy=marker.get("policy"))
        transform = build_full_roi_transform(layout)
        train_cfg = training_data_config(marker["strength"], marker.get("requested_size"))
        train_cfg["hagisilta_full_roi"] = layout
        train_cfg["auto_padding"] = {
            "min_width": layout["width"], "min_height": layout["height"],
            "pad_width_divisor": None, "pad_height_divisor": None,
            "position": "center", "border_mode": "constant",
        }
        loader.update_model_cfg({"data.train": train_cfg, "data.inference": inference_data_config()})
        run_config = copy.deepcopy(loader.model_cfg)
        record = {**layout, "transform": A.to_dict(transform),
                  "train_images": [str(path) for path in paths]}
        target = loader.model_folder / "hagisilta-input-layout.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=_json_value), encoding="utf-8")
        temporary.replace(target)
        print(f"TRAIN_INPUT_POLICY={layout['policy']}", flush=True)
        print(f"TRAIN_INPUT_SIZE={layout['width']}x{layout['height']}:frames={len(paths)}", flush=True)
        print(f"TRAIN_INPUT_MODE={layout['size_mode']}:minimum={layout['minimum_width']}x{layout['minimum_height']}", flush=True)
        print(f"TRAIN_INPUT_RESIZE={layout['resize_scale']:.8f}:crop={str(layout['crop']).lower()}:"
              f"whole_roi={str(not layout['crop']).lower()}", flush=True)
        print(f"TRAIN_INPUT_LAYOUT={target}", flush=True)
        print("VALIDATION_PADDING=center:multiple=32:resize=false", flush=True)
        kwargs["transform"] = transform
        return original(loader=loader, run_config=run_config, task=task, **kwargs)

    api.train = train_full_roi
    try:
        yield
    finally:
        api.train = original
