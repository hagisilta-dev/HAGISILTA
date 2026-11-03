# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
"""Validated postprocessing drafts; per-frame snapshots keep review provenance."""

from hagisilta_i18n import tr
from pathlib import Path
import math

from hagisilta_pose_postprocess import load_composition, recipe_label, validate_recipe
from hagisilta_pose_postprocess import sha256_file


def review_source_label(source, *, localized=True):
    """Render structured provenance without changing stored or user-provided text."""
    if source.get("kind") == "raw":
        return tr('보정 전') if localized else "Unfiltered"
    if source.get("kind") == "postprocessed":
        if "window" in source:
            window = source["window"]
            return (tr('중앙값 {v0}프레임', v0=window) if localized else
                    f"Median {window} {'frame' if window == 1 else 'frames'}")
        if source.get("recipe"):
            try:
                recipe = validate_recipe(source["recipe"])
            except ValueError:
                return str(source.get("label", ""))
            # Through 0.3.190 median snapshots retained the generated label and
            # recipe, but omitted window. Equal per-bodypart windows alone do
            # not establish median provenance; require the exact legacy label.
            window = next(iter(recipe.values()))
            if window and all(value == window for value in recipe.values()) and source.get("label") in (
                    f"중앙값 {window}프레임", f"Median {window} frames"):
                return (tr('중앙값 {v0}프레임', v0=window) if localized else
                        f"Median {window} frames")
            if localized:
                return tr('신체점별 · ') + recipe_label(recipe)
            return "Per bodypart · " + " · ".join(
                f"{name}: {'Unfiltered' if value == 0 else str(value) + 'F'}"
                for name, value in recipe.items()
            )
    # Legacy/custom provenance with no structured description remains readable.
    return str(source.get("label", ""))


def load_review_draft(video, pose, manifest):
    manifest = Path(manifest).resolve()
    record, _original, draft = load_composition(manifest, require_current=True)
    if (Path(record["source_video"]).resolve() != Path(video).resolve() or
            Path(record["source_pose"]).resolve() != Path(pose).resolve()):
        raise ValueError(tr('선택한 영상·원본 추론과 후처리 저장본이 일치하지 않습니다.'))
    source = {
        "kind": "postprocessed",
        "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
        "source_pose": str(Path(pose).resolve()),
        "source_pose_sha256": record["source_pose_sha256"],
        "filtered_sha256": record["filtered_sha256"], "recipe": record["recipe"],
    }
    if "window" in record:
        source["window"] = record["window"]
    source["label"] = review_source_label(source, localized=False)
    return draft, source


def validate_frame_baselines(session, bodyparts, frame_count):
    """Never silently fill a damaged reviewed snapshot with raw predictions."""
    for frame, record in session.get("frame_baselines", {}).items():
        if not 0 <= int(frame) < frame_count or set(record["points"]) != set(bodyparts):
            raise ValueError(tr('교정 시작 좌표의 프레임·신체점 구성이 맞지 않습니다.'))
        for point in record["points"].values():
            for key in ("x", "y"):
                value = float(point[key])
                if point["visible"] and not math.isfinite(value):
                    raise ValueError(tr('저장된 교정 시작 좌표가 손상되었습니다.'))
