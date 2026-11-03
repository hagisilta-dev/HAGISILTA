# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
"""Read analysis-input bodypart headers without importing DLC or loading poses.

This is a UI schema lookup, not the full integrity check performed by
load_analysis_pose at analysis execution. No input or configuration is written.
"""
from __future__ import annotations

from hagisilta_i18n import tr

from collections import OrderedDict
import csv
import json
from pathlib import Path
import threading

import pandas as pd


CHOICE_FILE = ".hagisilta-analysis-pose.json"
_CACHE_LIMIT = 256
_CACHE: OrderedDict[str, tuple[tuple, tuple, dict]] = OrderedDict()
_CACHE_LOCK = threading.RLock()


def _fingerprint(path: Path) -> tuple:
    try:
        stat = path.stat()
        return (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
                stat.st_ino, stat.st_mode)
    except OSError:
        return (str(path), None)


def _candidate_inventory(arena: Path) -> tuple:
    """Check names explicitly; Windows folder timestamps can lag new entries."""
    choice = arena / CHOICE_FILE
    hdf = tuple(sorted(arena.glob("*.h5")))
    csv_paths = tuple(sorted(path for path in arena.glob("*.csv") if "qc" not in path.name.lower()))
    return (choice.exists() or choice.is_symlink(), hdf, csv_paths)


def _track(path: Path, dependencies: dict) -> Path:
    path = path.expanduser().resolve()
    dependencies[str(path)] = _fingerprint(path)
    return path


def _read_json(path: Path, dependencies: dict) -> dict:
    _track(path, dependencies)
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(tr('JSON 객체가 아닙니다: {v0}', v0=path))
    return value


def _record_path(record: dict, name: str, dependencies: dict) -> Path:
    value = record.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(tr('분석 입력의 {v0} 경로가 없습니다.', v0=name))
    path = _track(Path(value), dependencies)
    if not path.is_file():
        raise ValueError(tr('분석 입력 파일이 없습니다: {v0}', v0=path))
    return path


def _stamp_matches(path: Path, stamp) -> bool:
    if not isinstance(stamp, dict):
        return False
    stat = path.stat()
    return stamp.get("size") == stat.st_size and stamp.get("mtime_ns") == stat.st_mtime_ns


def _hash_marker(record: dict, name: str) -> str:
    value = record.get(name)
    if not isinstance(value, str) or len(value) != 64 or any(
        letter not in "0123456789abcdefABCDEF" for letter in value
    ):
        raise ValueError(tr('분석 입력의 {v0} 무결성 정보가 올바르지 않습니다.', v0=name))
    return value.lower()


def _read_columns(path: Path) -> pd.MultiIndex:
    if path.suffix.lower() in {".h5", ".hdf5"}:
        # Both pandas fixed and table stores support a zero-row selection.
        columns = pd.read_hdf(path, start=0, stop=0).columns
    elif path.suffix.lower() == ".csv":
        # Read exactly the three DLC header rows, never the coordinate rows.
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            rows = [next(reader) for _ in range(3)]
        if not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError(tr('DLC CSV 헤더 열 수가 일치하지 않습니다.'))
        has_index = rows[2][0] not in {"x", "y", "likelihood"}
        names = [row[0] for row in rows] if has_index else ["scorer", "bodyparts", "coords"]
        arrays = [row[1:] if has_index else row for row in rows]
        columns = pd.MultiIndex.from_arrays(arrays, names=names)
    else:
        raise ValueError(tr('지원하지 않는 좌표 파일 형식: {v0}', v0=path.suffix))
    if not isinstance(columns, pd.MultiIndex) or columns.nlevels < 2 or not len(columns):
        raise ValueError(tr('DLC 신체점·좌표 헤더가 없습니다.'))
    if not columns.is_unique:
        raise ValueError(tr('DLC 좌표 헤더에 중복된 열이 있습니다.'))
    return columns


def _bodyparts(columns: pd.MultiIndex) -> list[str]:
    names = list(columns.names)
    body_level = names.index("bodyparts") if "bodyparts" in names else 1
    coord_level = names.index("coords") if "coords" in names else columns.nlevels - 1
    if body_level == coord_level:
        raise ValueError(tr('DLC 신체점과 좌표 헤더를 구분할 수 없습니다.'))
    points: dict[str, set[str]] = {}
    for column in columns:
        part, coord = str(column[body_level]), str(column[coord_level])
        if part.strip() and not part.startswith("Unnamed:"):
            points.setdefault(part, set()).add(coord)
    result = [part for part, coords in points.items() if {"x", "y", "likelihood"} <= coords]
    if not result:
        raise ValueError(tr('x/y/likelihood가 모두 있는 신체점이 없습니다.'))
    return result


def _selected_file(arena: Path, choice: Path, dependencies: dict) -> Path:
    record = _read_json(choice, dependencies)
    if record.get("schema") != "hagisilta-analysis-pose-v1":
        raise ValueError(tr('분석 좌표 선택 파일 형식이 올바르지 않습니다.'))
    video = _record_path(record, "source_video", dependencies)
    source = _record_path(record, "source_pose", dependencies)
    if video.parent != arena:
        raise ValueError(tr('분석 좌표 선택의 영상이 다른 ROI 폴더에 있습니다.'))
    if not _stamp_matches(video, record.get("video_stamp")):
        raise ValueError(tr('분석 입력 지정 후 영상이 변경되었습니다. 분석 입력을 다시 지정하세요.'))
    source_hash = _hash_marker(record, "source_pose_sha256")
    # Old choices may omit source_pose_stamp. Their content hash is checked by
    # the existing full analysis reader; newer choices can fail closed here.
    if record.get("source_pose_stamp") is not None and not _stamp_matches(source, record["source_pose_stamp"]):
        raise ValueError(tr('분석 입력 지정 후 추론 파일이 변경되었습니다. 분석 입력을 다시 지정하세요.'))
    if not record.get("composition"):
        return source

    manifest = _record_path(record, "composition", dependencies)
    selected = _read_json(manifest, dependencies)
    schema = selected.get("schema")
    if schema not in {"hagisilta-pose-composition-v1", "hagisilta-pose-filter-test-v1"}:
        raise ValueError(tr('지원하지 않는 후처리 보정본 형식입니다.'))
    selected_video = _record_path(selected, "source_video", dependencies)
    selected_source = _record_path(selected, "source_pose", dependencies)
    if selected_video != video or selected_source != source:
        raise ValueError(tr('선택한 보정본이 다른 ROI 영상 또는 추론 결과의 보정본입니다.'))
    if _hash_marker(selected, "source_pose_sha256") != source_hash:
        raise ValueError(tr('선택한 보정본의 원본 추론 정보가 분석 입력과 다릅니다.'))
    if not _stamp_matches(video, selected.get("video_stamp")):
        raise ValueError(tr('보정본 생성 후 영상이 변경되었습니다.'))
    if selected.get("source_pose_stamp") is not None and not _stamp_matches(source, selected["source_pose_stamp"]):
        raise ValueError(tr('보정본 생성 후 원본 추론 파일이 변경되었습니다.'))
    members = {}
    for key in ("original", "filtered"):
        relative = selected.get(key)
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError(tr('보정본의 {v0} 파일 경로가 없습니다.', v0=key))
        member = _track(manifest.parent / relative, dependencies)
        if member.parent != manifest.parent or not member.is_file():
            raise ValueError(tr('보정본 좌표 파일이 없거나 경로가 올바르지 않습니다: {v0}', v0=member))
        _hash_marker(selected, f"{key}_sha256")
        members[key] = member
    original_columns = _read_columns(members["original"])
    filtered_columns = _read_columns(members["filtered"])
    if not original_columns.equals(filtered_columns):
        raise ValueError(tr('보정 전·후 좌표의 신체점 구성이 다릅니다.'))
    if schema == "hagisilta-pose-composition-v1":
        recipe = selected.get("recipe")
        if (not isinstance(recipe, dict) or set(recipe) != set(_bodyparts(original_columns))
                or any(type(value) is not int or value not in (0, *range(3, 32, 2)) for value in recipe.values())):
            raise ValueError(tr('보정 조합의 신체점 설정이 좌표 헤더와 다릅니다.'))
    elif selected.get("method") != "median" or selected.get("window") not in range(3, 32, 2):
        raise ValueError(tr('후처리 보정본의 중앙값 설정이 올바르지 않습니다.'))
    return members["filtered"]


def _resolve_one(arena: Path) -> dict:
    arena = arena.expanduser().resolve()
    key = str(arena)
    try:
        inventory = _candidate_inventory(arena)
    except OSError:
        inventory = None
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if (cached and inventory is not None and cached[1] == inventory
                and all(_fingerprint(Path(stamp[0])) == stamp for stamp in cached[0])):
            _CACHE.move_to_end(key)
            return cached[2]
    dependencies: dict[str, tuple] = {}
    _track(arena, dependencies)
    choice = _track(arena / CHOICE_FILE, dependencies)
    result = {"bodyparts": [], "has_results": False, "issues": [], "files": [], "missing": []}
    try:
        if inventory is None:
            inventory = _candidate_inventory(arena)
        if inventory[0]:
            result["has_results"] = True
            selected = _selected_file(arena, choice, dependencies)
        else:
            candidates = inventory[1] or inventory[2]
            if not candidates:
                result["missing"] = [str(arena)]
                selected = None
            else:
                result["has_results"] = True
                selected = _track(candidates[0], dependencies)
        if selected is not None:
            result["files"] = [str(selected)]
            result["bodyparts"] = _bodyparts(_read_columns(selected))
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration, ImportError) as exc:
        result["issues"] = [tr('{v0}: 분석 신체점 확인 실패 · {v1}', v0=arena, v1=exc)]
        result["bodyparts"] = []
    except Exception as exc:
        # HDF/PyTables corruption has library-specific exception classes. Keep
        # that failure visible rather than falling back to a different result.
        result["issues"] = [tr('{v0}: 분석 좌표 헤더를 읽을 수 없습니다 · {v1}', v0=arena, v1=exc)]
        result["bodyparts"] = []
    # Only cache an inspection whose dependencies stayed stable while reading.
    stamps = tuple(dependencies.values())
    try:
        inventory_stable = inventory is not None and _candidate_inventory(arena) == inventory
    except OSError:
        inventory_stable = False
    if inventory_stable and all(_fingerprint(Path(stamp[0])) == stamp for stamp in stamps):
        with _CACHE_LOCK:
            _CACHE[key] = (stamps, inventory, result)
            _CACHE.move_to_end(key)
            while len(_CACHE) > _CACHE_LIMIT:
                _CACHE.popitem(last=False)
    return result


def resolve_analysis_bodyparts(arena_dirs) -> dict:
    """Return the exact-name intersection of the checked ROIs' actual inputs.

Missing (not yet tracked) ROIs are reported separately and do not erase known
names. A broken explicit choice or unreadable result fails the combined schema
closed. Full coordinate hashes and filtering mathematics remain execution-time
checks in hagisilta_pose_postprocess.load_analysis_pose.
    """
    arenas = list(dict.fromkeys(Path(path).expanduser().resolve() for path in arena_dirs))
    records = [_resolve_one(arena) for arena in arenas]
    schemas = [record["bodyparts"] for record in records if record["bodyparts"]]
    issues = [issue for record in records for issue in record["issues"]]
    common = ([name for name in schemas[0] if all(name in schema for schema in schemas[1:])]
              if schemas and not issues else [])
    return {
        "bodyparts": common,
        "has_results": any(record["has_results"] for record in records),
        "issues": issues,
        "files": [file for record in records for file in record["files"]],
        "missing": [arena for record in records for arena in record["missing"]],
        "mixed": bool(schemas and any(set(schema) != set(schemas[0]) for schema in schemas[1:])),
    }
