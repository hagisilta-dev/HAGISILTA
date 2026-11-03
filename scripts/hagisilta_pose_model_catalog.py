# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
"""Pose/detector choices verified against the bundled DeepLabCut runtime.

There are two named pose catalogs because the zero-shot and project-training
screens are separate workflows.  For the current HAGISILTA product scope they
intentionally contain the same two architectures:

* ``TOPVIEWMOUSE_PRETRAINED_POSE_MODELS`` is for zero-shot SuperAnimal
  inference and contains only architectures with an official
  ``superanimal_topviewmouse`` checkpoint.
* ``HAGISILTA_BU_TRAINING_POSE_MODELS`` is for HAGISILTA's low-label bottom-up
  project training and therefore also requires an official TopViewMouse
  checkpoint.

DeepLabCut registering an architecture is not sufficient reason to expose it
in HAGISILTA.  Architectures without an official TopViewMouse checkpoint would
change the product into a from-scratch training workflow with a substantially
larger annotation burden.
"""

from __future__ import annotations

from hagisilta_i18n import tr

from dataclasses import dataclass


def topviewmouse_bodyparts() -> list[str]:
    """Read the installed official output schema without importing DLC/GPU code."""
    from importlib.util import find_spec
    from pathlib import Path
    import yaml

    try:
        spec = find_spec("deeplabcut")
        roots = list(spec.submodule_search_locations or []) if spec else []
        if not roots:
            return []
        config = Path(roots[0]) / "modelzoo" / "project_configs" / "superanimal_topviewmouse.yaml"
        payload = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        values = payload.get("bodyparts", [])
        if not isinstance(values, (list, tuple)):
            return []
        return list(dict.fromkeys(str(part) for part in values if str(part).strip()))
    except (OSError, ValueError, TypeError, AttributeError, yaml.YAMLError):
        return []


@dataclass(frozen=True)
class PoseModelChoice:
    model_id: str
    display_name: str
    description: str
    topviewmouse_checkpoint: bool = False


TOPVIEWMOUSE_PRETRAINED_POSE_MODELS = (
    PoseModelChoice(
        "hrnet_w32",
        "HRNet-W32",
        "영상의 세부 위치 정보를 유지하면서 여러 해상도의 특징을 반복해서 결합해 "
        "신체점 위치를 예측하는 모델입니다.",
        True,
    ),
    PoseModelChoice(
        "resnet_50",
        "ResNet-50",
        "잔차 블록을 쌓아 영상의 형태와 패턴을 단계적으로 추출한 뒤 "
        "신체점 위치를 예측하는 모델입니다.",
        True,
    ),
)


HAGISILTA_BU_TRAINING_POSE_MODELS = TOPVIEWMOUSE_PRETRAINED_POSE_MODELS


TOPVIEWMOUSE_POSE_MODEL_IDS = tuple(
    model.model_id for model in TOPVIEWMOUSE_PRETRAINED_POSE_MODELS
)
TOPVIEWMOUSE_POSE_MODEL_ID_SET = frozenset(TOPVIEWMOUSE_POSE_MODEL_IDS)
HAGISILTA_BU_TRAINING_POSE_MODEL_IDS = tuple(
    model.model_id for model in HAGISILTA_BU_TRAINING_POSE_MODELS
)
HAGISILTA_BU_TRAINING_POSE_MODEL_ID_SET = frozenset(
    HAGISILTA_BU_TRAINING_POSE_MODEL_IDS
)

AVAILABLE_DETECTOR_MODEL_IDS = (
    "fasterrcnn_resnet50_fpn_v2",
    "fasterrcnn_mobilenet_v3_large_fpn",
    "ssdlite",
)
TOPVIEWMOUSE_PRETRAINED_DETECTOR_MODEL_IDS = (
    "fasterrcnn_resnet50_fpn_v2",
    "fasterrcnn_mobilenet_v3_large_fpn",
)


def _add_models(combo, models: tuple[PoseModelChoice, ...]) -> None:
    for model in models:
        combo.addItem(model.display_name, model.model_id)


def add_topviewmouse_pose_models(combo) -> None:
    """Populate zero-shot TopViewMouse pose choices."""
    _add_models(combo, TOPVIEWMOUSE_PRETRAINED_POSE_MODELS)


def add_hagisilta_training_pose_models(combo) -> None:
    """Populate the supported HAGISILTA bottom-up training choices."""
    _add_models(combo, HAGISILTA_BU_TRAINING_POSE_MODELS)


def _tooltip(title: str, models: tuple[PoseModelChoice, ...], footer: str = "") -> str:
    lines = [title]
    lines.extend(f"{model.display_name}: {tr(model.description)}" for model in models)
    if footer:
        lines.append(footer)
    return "\n".join(lines)


def topviewmouse_pose_model_tooltip() -> str:
    return _tooltip(
        tr('신체점 추론에 사용할 Pose 모델을 선택합니다.'),
        TOPVIEWMOUSE_PRETRAINED_POSE_MODELS,
    )


def hagisilta_training_pose_model_tooltip() -> str:
    return _tooltip(
        tr('신체점 학습에 사용할 Pose 모델을 선택합니다.'),
        HAGISILTA_BU_TRAINING_POSE_MODELS,
    )


def require_hagisilta_training_pose_model(model_id: str) -> str:
    normalized = str(model_id or "").strip().lower()
    if normalized not in HAGISILTA_BU_TRAINING_POSE_MODEL_ID_SET:
        supported = ", ".join(HAGISILTA_BU_TRAINING_POSE_MODEL_IDS)
        raise ValueError(
            tr('현재 HAGISILTA Bottom-up 학습 흐름에서 지원하지 않는 Pose 모델입니다: {v0!r}. 지원 모델: {v1}', v0=model_id, v1=supported)
        )
    return normalized
