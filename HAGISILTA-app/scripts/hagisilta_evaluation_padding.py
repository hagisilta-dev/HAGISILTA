# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
"""Per-run video-test padding, without changing trained model configuration."""
from __future__ import annotations

from hagisilta_i18n import tr

import hashlib
import json
from pathlib import Path

import yaml
from hagisilta_training_input import POLICY, FIT_POLICY, CROP_POLICY, ceil32, validate_canvas_size

MODES = ("model", "minimum", "training")


def padding_spec(config: Path, shuffle: int, mode: str) -> dict:
    if mode not in MODES:
        raise ValueError(tr('지원하지 않는 영상 테스트 패딩 방식입니다.'))
    if mode == "model":
        return {"mode": mode}
    project = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    if project.get("cropping", False):
        raise ValueError(tr("프로젝트에 별도 추론 크롭이 설정되어 있습니다. 패딩만 비교하려면 '모델 기본'으로 먼저 설정을 확인하세요."))
    iteration = int(project.get("iteration", 0))
    candidates = list(config.parent.glob(
        f"dlc-models-pytorch/iteration-{iteration}/*shuffle{int(shuffle)}/train/pytorch_config.yaml"))
    if len(candidates) != 1:
        raise ValueError(tr('Shuffle {v0}의 모델 설정을 하나로 확정할 수 없습니다.', v0=shuffle))
    model_path = candidates[0]
    raw = model_path.read_bytes()
    model = yaml.safe_load(raw) or {}
    if model.get("method") != "bu":
        raise ValueError(tr('영상 테스트 패딩 비교는 Bottom-up 모델에서만 지원합니다.'))
    data = model.get("data", {}).get("inference", {})
    unsupported = [key for key, value in data.items()
                   if value and key not in {"auto_padding", "normalize_images", "scale_to_unit_range"}]
    if unsupported:
        raise ValueError(tr("패딩 외의 영상 변환이 있는 모델입니다. '모델 기본'을 사용하세요: ") + ", ".join(unsupported))
    result = {"mode": mode, "model_config": str(model_path),
              "model_config_sha256": hashlib.sha256(raw).hexdigest(),
              "normalize_images": bool(data.get("normalize_images")),
              "scale_to_unit_range": bool(data.get("scale_to_unit_range")),
              "position": "center", "border": "constant", "resize": False, "crop": False}
    if mode == "training":
        record_path = model_path.parent / "hagisilta-input-layout.json"
        record = (json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file()
                  else model.get("data", {}).get("train", {}).get("hagisilta_full_roi", {}))
        if record.get("policy") == CROP_POLICY:
            raise ValueError(tr("이 모델은 라벨 주변을 크롭해 학습했습니다. 영상에는 정답 라벨이 없으므로 학습 크롭을 재현하지 않습니다. 전체 ROI 추적은 '모델 기본' 또는 '최소 패딩'을 선택하세요."))
        if record.get("policy") not in (POLICY, FIT_POLICY):
            raise ValueError(tr("이 모델에는 ROI 전체 보존 학습 크기 기록이 없습니다. '모델 기본' 또는 '최소 패딩'을 선택하세요."))
        if record.get("resize_to_fit") or record.get("policy") == FIT_POLICY:
            raise ValueError(tr("이 모델은 ROI 전체를 축소해 학습했습니다. '학습 패딩'은 여백만 비교하므로 축소 학습 입력을 재현할 수 없습니다. 원본 크기로 시험하려면 '모델 기본' 또는 '최소 패딩'을 선택하세요."))
        width, height = validate_canvas_size((record.get("width"), record.get("height")))
        result.update(width=width, height=height)
    return result


def sized_spec(spec: dict, width: int, height: int) -> dict:
    if width <= 0 or height <= 0:
        raise ValueError(tr('시험 ROI 영상 크기를 확인할 수 없습니다.'))
    result = dict(spec)
    if spec["mode"] == "minimum":
        result.update(width=ceil32(width), height=ceil32(height))
    if spec["mode"] != "model" and (width > result["width"] or height > result["height"]):
        raise ValueError(tr("시험 ROI {v0} × {v1}px가 학습 입력 {v2} × {v3}px보다 큽니다. 자동으로 자르거나 축소하지 않습니다. '최소 패딩' 또는 '모델 기본'을 선택하세요.", v0=width, v1=height, v2=result['width'], v3=result['height']))
    result.update(roi_width=width, roi_height=height)
    return result


def padding_transform(spec: dict):
    if spec["mode"] == "model":
        return None
    from deeplabcut.pose_estimation_pytorch.data.transforms import build_transforms
    return build_transforms({
        "normalize_images": spec["normalize_images"],
        "scale_to_unit_range": spec["scale_to_unit_range"],
        "auto_padding": {"min_width": spec["width"], "min_height": spec["height"],
                         "pad_width_divisor": None, "pad_height_divisor": None,
                         "position": "center", "border_mode": "constant", "border_value": 0},
    })


def validate_padding_run(mode, model_mode, execution_mode, videos):
    if mode != "model" and (model_mode != "trained" or execution_mode != "stable"
            or not videos or any(item.get("group") != "model_evaluation_pilot" for item in videos)):
        raise ValueError(tr('패딩 비교는 학습 모델의 대표 영상 테스트(순차 실행)에서만 사용할 수 있습니다.'))


def validate_padding_overrides(mode: str, overrides: dict) -> None:
    if mode == "model":
        return
    dynamic = overrides.get("dynamic", False)
    active_dynamic = dynamic[0] if isinstance(dynamic, (list, tuple)) and dynamic else bool(dynamic)
    if overrides.get("cropping") is not None or active_dynamic:
        raise ValueError(tr('패딩 비교에서는 추가 크롭을 함께 적용할 수 없습니다. 전문가 JSON의 cropping·dynamic 설정을 제거하세요.'))


def padding_label(spec: dict) -> str:
    title = {"model": tr('모델 기본'), "minimum": tr('최소 패딩'), "training": tr('학습 패딩')}.get(spec.get("mode"), "")
    if title and spec.get("width") and spec.get("height"):
        title += f" {spec['width']} × {spec['height']}"
    return title


def result_padding_label(video: Path) -> str:
    # Never read the directory-wide metadata: it belongs to the latest run.
    metadata = video.with_name(video.stem + ".roi-metadata.json")
    if not metadata.is_file():
        return ""
    try:
        return padding_label(json.loads(metadata.read_text(encoding="utf-8")).get("evaluation_padding", {}))
    except (OSError, ValueError, TypeError):
        return ""
