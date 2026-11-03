# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import random
import re
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path


PLAN_VERSION = 3
DEFAULT_RATIOS = {"train": 70, "validation": 15, "test": 15}
ROLE_LABELS = {"train": "Train", "validation": "Validation", "test": "Test"}
REVIEW_ORIGIN = "video_review_gt"
INITIAL_ORIGIN = "initial_kmeans"
SPLIT_MODE_OPTIMIZATION = "optimization"
SPLIT_MODE_INDEPENDENT = "independent"
DEFAULT_SPLIT_MODE = SPLIT_MODE_OPTIMIZATION
DEFAULT_GROUPING = "source_video"


def split_plan_path(project: Path) -> Path:
    current = project / ".hagisilta"
    legacy = project / ".nort"
    root = current if current.exists() or not legacy.exists() else legacy
    return root / "dataset-splits.json"


def load_split_plan(project: Path) -> dict:
    path = split_plan_path(project)
    if not path.is_file():
        return {
            "schema_version": PLAN_VERSION,
            "dataset_version": 1,
            "locked": False,
            "split_mode": DEFAULT_SPLIT_MODE,
            "grouping": DEFAULT_GROUPING,
            "ratios": dict(DEFAULT_RATIOS),
            "updated_at": "",
            "bundles": {},
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    payload.setdefault("schema_version", PLAN_VERSION)
    payload.setdefault("dataset_version", 1)
    payload.setdefault("locked", False)
    payload.setdefault("split_mode", DEFAULT_SPLIT_MODE)
    payload.setdefault("grouping", DEFAULT_GROUPING)
    payload.setdefault("ratios", dict(DEFAULT_RATIOS))
    payload.setdefault("updated_at", "")
    payload.setdefault("bundles", {})
    return payload


def split_composition_signature(plan: dict) -> str:
    """Identity of training membership and splits, independent of UI/lock state."""
    bundles = {}
    for folder, bundle in enabled_split_bundles(plan):
        assignments = bundle.get("assignments", {}) or {}
        meta = bundle.get("frame_meta", {}) or {}
        bundles[folder] = {
            "assignments": assignments,
            "frames": {
                name: {key: (meta.get(name, {}) or {}).get(key)
                       for key in ("source_frame", "origin", "temporal_group_id")}
                for name in assignments
            },
            "source": {key: bundle.get(key) for key in ("source_video", "arena", "subject_id")},
        }
    content = {
        "split_mode": plan.get("split_mode", DEFAULT_SPLIT_MODE),
        "grouping": plan.get("grouping", DEFAULT_GROUPING),
        "bundles": bundles,
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def save_split_plan(project: Path, payload: dict) -> Path:
    path = split_plan_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A corrupt existing plan must not silently become an empty/new revision.
    previous = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    updated = dict(payload)
    version = max(int(previous.get("dataset_version", 1)), int(updated.get("dataset_version", 1)))
    signature = split_composition_signature(updated)
    basis = previous.get("confirmed_composition", {}) or {}
    if not basis and previous and split_plan_is_effectively_locked(previous):
        basis = {"version": int(previous.get("dataset_version", 1)),
                 "signature": split_composition_signature(previous)}
    changed = bool(basis and int(basis["version"]) == version and basis["signature"] != signature)

    # Older builds could give different Shuffle datasets the same version.
    # Reconcile the editable plan once, leaving historical/active run files intact.
    highest = version
    collision = False
    if not previous.get("composition_tracking_version"):
        for frozen in sorted((path.parent / "training-splits").glob("shuffle-*-split.json")):
            record = json.loads(frozen.read_text(encoding="utf-8"))
            recorded_version = int(record.get("dataset_version", 1))
            highest = max(highest, recorded_version)
            if recorded_version == version and split_composition_signature(record) != signature:
                collision = True
    if changed or collision:
        version = highest + 1
        updated["previous_locked_version"] = int(previous.get("dataset_version", payload.get("dataset_version", 1)))
        updated["draft_reason"] = "학습 프레임 또는 분할 구성 변경"
    if previous and version > int(previous.get("dataset_version", 1)):
        history = path.parent / "dataset-split-history"
        history.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        shutil.copy2(path, history / f"split-v{int(previous.get('dataset_version', 1)):03d}-{stamp}.json")
    updated["dataset_version"] = version
    updated["composition_tracking_version"] = 1
    if split_plan_is_effectively_locked(updated):
        basis = {"version": version, "signature": signature}
    if basis:
        updated["confirmed_composition"] = basis
    updated["schema_version"] = PLAN_VERSION
    updated["updated_at"] = datetime.now().isoformat(timespec="seconds")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(updated, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    # Callers immediately render this same plan: expose the version actually saved.
    payload.update(updated)
    return path


def normalized_ratios(ratios: dict | None) -> dict[str, int]:
    values = dict(DEFAULT_RATIOS)
    if ratios:
        for role in values:
            try:
                values[role] = max(0, int(ratios.get(role, values[role])))
            except (TypeError, ValueError):
                pass
    total = sum(values.values())
    if total <= 0:
        return dict(DEFAULT_RATIOS)
    if total == 100:
        return values
    scaled = {role: int(round(value * 100 / total)) for role, value in values.items()}
    scaled["train"] += 100 - sum(scaled.values())
    return scaled


def block_assignments(
    frame_names: list[str], ratios: dict | None = None, seed: str = "", block_count: int = 20
) -> dict[str, str]:
    """Assign adjacent frames together while spreading blocks across all roles."""
    names = sorted(dict.fromkeys(map(str, frame_names)), key=natural_frame_key)
    if not names:
        return {}
    ratios = normalized_ratios(ratios)
    active = [role for role, value in ratios.items() if value > 0]
    if len(active) == 1:
        return {name: active[0] for name in names}

    count = min(max(len(active), int(block_count)), len(names))
    blocks: list[list[str]] = []
    for index in range(count):
        start = round(index * len(names) / count)
        end = round((index + 1) * len(names) / count)
        if end > start:
            blocks.append(names[start:end])

    order = list(range(len(blocks)))
    numeric_seed = int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16], 16)
    random.Random(numeric_seed).shuffle(order)
    targets = {
        role: round(len(names) * ratios[role] / 100)
        for role in ROLE_LABELS
    }
    targets["train"] += len(names) - sum(targets.values())
    assigned_counts = {role: 0 for role in ROLE_LABELS}
    result: dict[str, str] = {}
    for block_index in order:
        block = blocks[block_index]
        candidates = [role for role in ROLE_LABELS if ratios[role] > 0]
        role = max(
            candidates,
            key=lambda value: (targets[value] - assigned_counts[value], ratios[value]),
        )
        for name in block:
            result[name] = role
        assigned_counts[role] += len(block)
    return result


def frame_number(name: str) -> int | None:
    """Return the original video frame number encoded in a DLC image name."""
    match = re.search(r"(\d+)(?=\.[^.]+$)", Path(str(name)).name)
    return int(match.group(1)) if match else None


def temporal_group_map(
    frame_names: list[str],
    fps: float = 0.0,
    *,
    max_gap_seconds: float = 0.5,
    fallback_gap_frames: int = 10,
    prefix: str = "review",
) -> dict[str, str]:
    """Group sparse edits and bursts without treating a whole review as continuous.

    Each corrected frame is expanded by a short temporal guard. Overlapping guards
    form one group; isolated corrections therefore remain one-frame groups.
    """
    numbered = [(name, frame_number(name)) for name in dict.fromkeys(map(str, frame_names))]
    numbered = [(name, index) for name, index in numbered if index is not None]
    numbered.sort(key=lambda item: (item[1], natural_frame_key(item[0])))
    if not numbered:
        return {}
    try:
        gap = max(1, round(float(fps) * float(max_gap_seconds))) if float(fps) > 0 else 0
    except (TypeError, ValueError):
        gap = 0
    gap = gap or max(1, int(fallback_gap_frames))
    groups: list[list[tuple[str, int]]] = []
    for item in numbered:
        if not groups or item[1] - groups[-1][-1][1] > gap:
            groups.append([item])
        else:
            groups[-1].append(item)
    result: dict[str, str] = {}
    for items in groups:
        start, end = items[0][1], items[-1][1]
        group_id = f"{prefix}-{start:09d}-{end:09d}"
        for name, _index in items:
            result[name] = group_id
    return result


def grouped_assignments(
    frame_groups: dict[str, str], ratios: dict | None = None, seed: str = ""
) -> dict[str, str]:
    """Assign complete temporal/source groups while approximating target ratios."""
    if not frame_groups:
        return {}
    ratios = normalized_ratios(ratios)
    grouped: dict[str, list[str]] = {}
    for frame, group_id in frame_groups.items():
        grouped.setdefault(str(group_id), []).append(str(frame))
    order = sorted(grouped)
    numeric_seed = int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16], 16)
    random.Random(numeric_seed).shuffle(order)
    total = len(frame_groups)
    targets = {role: round(total * ratios[role] / 100) for role in ROLE_LABELS}
    targets["train"] += total - sum(targets.values())
    assigned_counts = {role: 0 for role in ROLE_LABELS}
    result: dict[str, str] = {}
    for group_id in order:
        frames = grouped[group_id]
        candidates = [role for role in ROLE_LABELS if ratios[role] > 0]
        role = max(
            candidates,
            key=lambda value: (targets[value] - assigned_counts[value], ratios[value]),
        )
        for frame in frames:
            result[frame] = role
        assigned_counts[role] += len(frames)
    return result


def source_group_id(bundle: dict, folder: str = "", grouping: str = DEFAULT_GROUPING) -> str:
    """Return a stable leakage barrier shared by every derivative of one recording.

    Older projects only stored cropped ROI filenames.  Strip the generated ROI/Arena
    suffixes so every crop from the same source recording stays in one split.
    """
    if grouping == "subject_id":
        subject = str(bundle.get("subject_id", "") or "").strip()
        if subject:
            return f"subject:{subject.casefold()}"
    explicit = str(bundle.get("original_source_video", "") or "").strip()
    source = Path(explicit or str(bundle.get("source_video", "") or folder)).stem
    source = re.sub(r"_roi\d*(?:_arena[_ -]?\d+.*)?$", "", source, flags=re.IGNORECASE)
    source = re.sub(r"_arena[_ -]?\d+(?:_roi(?:_q\d+)?)?.*$", "", source, flags=re.IGNORECASE)
    source = source.strip(" _-.") or str(bundle.get("subject_id", "") or folder)
    return f"video:{source.casefold()}"


def sync_independent_assignments(plan: dict, *, rebalance: bool = False, seed: str = "") -> dict[str, str]:
    """Keep complete source groups in exactly one split across all bundles."""
    grouping = str(plan.get("grouping", DEFAULT_GROUPING) or DEFAULT_GROUPING)
    ratios = normalized_ratios(plan.get("ratios"))
    bundles = {
        str(folder): bundle
        for folder, bundle in (plan.get("bundles", {}) or {}).items()
        if not bundle.get("missing")
    }
    grouped: dict[str, list[tuple[str, dict]]] = {}
    for folder, bundle in bundles.items():
        group_id = source_group_id(bundle, folder, grouping)
        bundle["split_group_id"] = group_id
        grouped.setdefault(group_id, []).append((folder, bundle))

    roles = {} if rebalance else {
        str(group_id): str(role)
        for group_id, role in (plan.get("group_roles", {}) or {}).items()
        if group_id in grouped and role in ROLE_LABELS
    }
    # Disabled rows keep a role so re-enabling them is deterministic, but they
    # must not distort the target ratios of the dataset that is actually used.
    weights = {
        group_id: sum(
            len(bundle.get("assignments", {}) or {})
            for _folder, bundle in entries
            if bundle.get("enabled", True)
        )
        for group_id, entries in grouped.items()
    }
    total = sum(weights.values())
    targets = {role: round(total * ratios[role] / 100) for role in ROLE_LABELS}
    targets["train"] += total - sum(targets.values())
    counts = {role: 0 for role in ROLE_LABELS}
    for group_id, role in roles.items():
        counts[role] += weights.get(group_id, 0)

    pending = [group_id for group_id in grouped if group_id not in roles]
    numeric_seed = int(hashlib.sha256((seed or "independent-split").encode("utf-8")).hexdigest()[:16], 16)
    random.Random(numeric_seed).shuffle(pending)
    pending.sort(key=lambda group_id: weights[group_id], reverse=True)
    for group_id in pending:
        candidates = [role for role in ROLE_LABELS if ratios[role] > 0]
        role = max(candidates, key=lambda value: (targets[value] - counts[value], ratios[value]))
        roles[group_id] = role
        counts[role] += weights[group_id]

    for group_id, entries in grouped.items():
        role = roles[group_id]
        for _folder, bundle in entries:
            bundle["group_role"] = role
            bundle["assignments"] = {
                frame_name: role for frame_name in (bundle.get("assignments", {}) or {})
            }
    plan["group_roles"] = roles
    plan["ratios"] = ratios
    return roles


def apply_split_mode(
    plan: dict,
    mode: str,
    *,
    grouping: str = DEFAULT_GROUPING,
    seed: str = "",
) -> None:
    """Explicitly switch between operational frame splitting and independent groups."""
    mode = mode if mode in {SPLIT_MODE_OPTIMIZATION, SPLIT_MODE_INDEPENDENT} else DEFAULT_SPLIT_MODE
    plan["split_mode"] = mode
    plan["grouping"] = grouping or DEFAULT_GROUPING
    plan["locked"] = False
    for folder, bundle in (plan.get("bundles", {}) or {}).items():
        bundle["locked"] = False
        bundle.pop("group_role", None)
        bundle.pop("split_group_id", None)
        if mode == SPLIT_MODE_OPTIMIZATION:
            rebuild_bundle_assignments(bundle, seed=str(folder))
    if mode == SPLIT_MODE_INDEPENDENT:
        plan["group_roles"] = {}
        sync_independent_assignments(plan, rebalance=True, seed=seed)
    else:
        plan.pop("group_roles", None)


def update_bundle_frames(
    bundle: dict,
    frame_names: list[str],
    *,
    review_frame_names: list[str] | None = None,
    review_fps: float = 0.0,
    review_session: str = "",
    review_group_map: dict[str, str] | None = None,
) -> dict[str, int | bool]:
    """Merge current labels without reshuffling an existing split.

    Newly imported human corrections default to Train in operational mode.  In
    independent mode they inherit the source group's frozen role.
    """
    names = sorted(dict.fromkeys(map(str, frame_names)), key=natural_frame_key)
    name_set = set(names)
    review_set = set(map(str, review_frame_names or [])) & name_set
    frame_meta = bundle.setdefault("frame_meta", {})
    old_assignments = dict(bundle.get("assignments", {}) or {})
    assignments = {name: role for name, role in old_assignments.items() if name in name_set}
    forced_role = str(bundle.get("group_role", "") or "")
    if forced_role not in ROLE_LABELS:
        forced_role = ""
    review_groups = temporal_group_map(list(review_set), review_fps)
    for name, group_id in (review_group_map or {}).items():
        name, group_id = str(name), str(group_id)
        if name in review_set and group_id:
            review_groups[name] = group_id

    for name in names:
        meta = dict(frame_meta.get(name, {}) or {})
        index = frame_number(name)
        meta.setdefault("source_frame", index)
        if name in review_set:
            meta["origin"] = REVIEW_ORIGIN
            meta["temporal_group_id"] = review_groups.get(name, f"review-{name}")
            if review_session:
                meta["review_session"] = review_session
        else:
            meta.setdefault("origin", INITIAL_ORIGIN)
            meta.setdefault("temporal_group_id", f"initial-{name}")
        frame_meta[name] = meta
    for missing in set(frame_meta) - name_set:
        frame_meta.pop(missing, None)

    new_names = [name for name in names if name not in assignments]
    new_review = [name for name in new_names if name in review_set]
    new_initial = [name for name in new_names if name not in review_set]
    for name in new_review:
        group_id = frame_meta[name].get("temporal_group_id")
        existing_roles = {
            assignments[member]
            for member, meta in frame_meta.items()
            if member in assignments and meta.get("temporal_group_id") == group_id
        }
        # A correction extending an already assigned burst follows that burst.
        # Otherwise new human corrections default to Train.
        assignments[name] = forced_role or (next(iter(existing_roles)) if len(existing_roles) == 1 else "train")
    if new_initial:
        initial_groups = {
            name: str(frame_meta[name].get("temporal_group_id") or f"initial-{name}")
            for name in new_initial
        }
        if forced_role:
            assignments.update({name: forced_role for name in new_initial})
        else:
            assignments.update(
                grouped_assignments(initial_groups, bundle.get("ratios"), seed=str(bundle.get("folder_seed", "")))
            )

    bundle["assignments"] = assignments
    bundle["frame_meta"] = frame_meta
    bundle["origin_counts"] = {
        INITIAL_ORIGIN: sum(meta.get("origin") != REVIEW_ORIGIN for meta in frame_meta.values()),
        REVIEW_ORIGIN: sum(meta.get("origin") == REVIEW_ORIGIN for meta in frame_meta.values()),
    }
    review_group_ids = {
        str(meta.get("temporal_group_id"))
        for meta in frame_meta.values()
        if meta.get("origin") == REVIEW_ORIGIN and meta.get("temporal_group_id")
    }
    review_singletons = 0
    for group_id in review_group_ids:
        if sum(
            meta.get("origin") == REVIEW_ORIGIN and meta.get("temporal_group_id") == group_id
            for meta in frame_meta.values()
        ) == 1:
            review_singletons += 1
    bundle["review_group_count"] = len(review_group_ids)
    bundle["review_singleton_count"] = review_singletons
    return {
        "changed": assignments != old_assignments or set(frame_meta) != set(names),
        "new_frames": len(new_names),
        "new_review_frames": len(new_review),
        "removed_frames": len(set(old_assignments) - name_set),
    }


def rebuild_bundle_assignments(bundle: dict, seed: str = "") -> dict[str, str]:
    """Explicitly redistribute a bundle while keeping each temporal group intact."""
    assignments = bundle.get("assignments", {}) or {}
    frame_meta = bundle.get("frame_meta", {}) or {}
    groups = {
        name: str((frame_meta.get(name, {}) or {}).get("temporal_group_id") or f"frame-{name}")
        for name in assignments
    }
    rebuilt = grouped_assignments(groups, bundle.get("ratios"), seed=seed)
    bundle["assignments"] = rebuilt
    return rebuilt


def natural_frame_key(name: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name))


def split_counts(assignments: dict[str, str]) -> dict[str, int]:
    counts = {role: 0 for role in ROLE_LABELS}
    for role in assignments.values():
        if role in counts:
            counts[role] += 1
    return counts


def format_split(bundle: dict) -> str:
    counts = split_counts(bundle.get("assignments", {}) or {})
    total = sum(counts.values())
    if total:
        return f"Train {counts['train']} · Val {counts['validation']} · Test {counts['test']}"
    ratios = normalized_ratios(bundle.get("ratios"))
    return f"Train {ratios['train']}% · Val {ratios['validation']}% · Test {ratios['test']}%"


def enabled_split_bundles(plan: dict) -> list[tuple[str, dict]]:
    """Return only bundles that will actually participate in training."""
    return [
        (str(folder), bundle)
        for folder, bundle in (plan.get("bundles", {}) or {}).items()
        if bundle.get("enabled", True) and not bundle.get("missing")
    ]


def split_plan_is_effectively_locked(plan: dict) -> bool:
    """Treat a global lock or locks on every enabled row as a frozen split."""
    enabled = enabled_split_bundles(plan)
    return bool(plan.get("locked") or (enabled and all(bundle.get("locked") for _, bundle in enabled)))


def unlocked_enabled_split_folders(plan: dict) -> list[str]:
    """List training bundles that are not protected by either lock scope."""
    if plan.get("locked"):
        return []
    return [folder for folder, bundle in enabled_split_bundles(plan) if not bundle.get("locked")]


def indices_from_merged_data(data_index, plan: dict) -> tuple[list[int], list[int], list[int], list[str]]:
    """Map DLC's merged MultiIndex to the frozen per-frame assignments."""
    bundles = plan.get("bundles", {}) or {}
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    missing: list[str] = []
    for position, raw_index in enumerate(data_index):
        parts = tuple(map(str, raw_index)) if isinstance(raw_index, tuple) else (str(raw_index),)
        folder = parts[-2] if len(parts) >= 2 else ""
        frame = Path(parts[-1]).name
        bundle = bundles.get(folder)
        if bundle and not bundle.get("enabled", True):
            continue
        if not bundle:
            missing.append(f"{folder}/{frame}")
            continue
        role = (bundle.get("assignments", {}) or {}).get(frame)
        if role == "train":
            train.append(position)
        elif role == "validation":
            validation.append(position)
        elif role == "test":
            test.append(position)
        else:
            missing.append(f"{folder}/{frame}")
    return train, validation, test, missing
