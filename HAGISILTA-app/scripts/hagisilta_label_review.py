# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from hagisilta_runtime import project_state_root


def review_path(config: Path) -> Path:
    return project_state_root(config.resolve().parent) / "labeling-review.json"


def load_review(config: Path) -> dict:
    path = review_path(config)
    if not path.is_file():
        return {"version": 1, "folders": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.setdefault("version", 1)
        payload.setdefault("folders", {})
        return payload
    except Exception:
        return {"version": 1, "folders": {}}


def save_review(config: Path, payload: dict) -> Path:
    path = review_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def excluded_frames(config: Path) -> dict[str, set[str]]:
    payload = load_review(config)
    result: dict[str, set[str]] = {}
    for folder, frames in (payload.get("folders", {}) or {}).items():
        excluded = {
            str(name) for name, state in (frames or {}).items()
            if (state or {}).get("decision") == "excluded"
        }
        if excluded:
            result[str(folder)] = excluded
    return result


def _row_filename(index_value) -> str:
    if isinstance(index_value, tuple):
        index_value = index_value[-1]
    return Path(str(index_value)).name


@contextmanager
def temporarily_filter_excluded_labels(config: Path):
    """Hide explicitly excluded rows while DLC builds its training dataset.

    Original label files are copied byte-for-byte and restored in ``finally``.
    The generated training dataset keeps the filtering, while the editable source
    labels remain untouched after preparation finishes.
    """

    config = config.resolve()
    exclusions = excluded_frames(config)
    if not exclusions:
        yield 0
        return

    backup_root = project_state_root(config.parent, create=True) / "runtime" / f"label-filter-{time.time_ns()}"
    backups: list[tuple[Path, Path]] = []
    removed = 0
    try:
        for folder_name, names in exclusions.items():
            folder = config.parent / "labeled-data" / folder_name
            if not folder.is_dir():
                continue
            for label_file in folder.glob("CollectedData_*.h5"):
                frame = pd.read_hdf(label_file)
                keep = [
                    _row_filename(index_value) not in names
                    for index_value in frame.index
                ]
                count = len(keep) - sum(keep)
                if not count:
                    continue
                relative = label_file.relative_to(config.parent)
                backup = backup_root / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(label_file, backup)
                backups.append((label_file, backup))
                frame.loc[keep].to_hdf(label_file, key="df_with_missing", mode="w")
                removed += count
            for label_file in folder.glob("CollectedData_*.csv"):
                relative = label_file.relative_to(config.parent)
                backup = backup_root / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(label_file, backup)
                backups.append((label_file, backup))
                try:
                    frame = pd.read_csv(label_file, header=[0, 1, 2], index_col=[0, 1, 2])
                    keep = [_row_filename(index_value) not in names for index_value in frame.index]
                    frame.loc[keep].to_csv(label_file)
                except Exception:
                    shutil.copy2(backup, label_file)
                    backups.pop()
        yield removed
    finally:
        for original, backup in reversed(backups):
            if backup.is_file():
                shutil.copy2(backup, original)
        if backup_root.is_dir():
            shutil.rmtree(backup_root, ignore_errors=True)
