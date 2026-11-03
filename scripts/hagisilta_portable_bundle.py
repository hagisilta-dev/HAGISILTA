# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from hagisilta_i18n import tr

import argparse
import json
import hashlib
import os
import re
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import yaml

from hagisilta_runtime import project_output_root, project_state_root


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ITEMS = (
    "config.yaml",
    "labeled-data",
    "training-datasets",
    "dlc-models-pytorch",
    "dlc-models",
    "evaluation-results-pytorch",
    "evaluation-results",
)
VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv", ".wmv", ".mpg", ".mpeg", ".m4v"}
UI_CATEGORIES = ("project_inputs", "project_training_settings", "project_batch_settings",
                 "project_ymaze_settings", "project_nort_settings")


def path_token(text):
    return str(text).replace("\\", "/").rstrip("/").casefold()


def absolute_text(text):
    return isinstance(text, str) and (text.startswith("/") or bool(re.match(r"^[A-Za-z]:[/\\]", text)))


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from strings(key)
            yield from strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from strings(child)


def read_document(path):
    text = path.read_text(encoding="utf-8-sig")
    return json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)


def structured_files(project):
    for path in project.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".yaml", ".yml"} and not any(
            part in {".hagisilta-backups", ".hagisilta-trash", "__pycache__", "runtime"} for part in path.relative_to(project).parts
        ):
            yield path


def rewrite(value, mapping):
    if isinstance(value, dict):
        return {rewrite(key, mapping): rewrite(child, mapping) for key, child in value.items()}
    if isinstance(value, list):
        return [rewrite(child, mapping) for child in value]
    if isinstance(value, str):
        normalized = path_token(value)
        for old, new in sorted(mapping.items(), key=lambda item: -len(item[0])):
            prefix = path_token(old)
            if normalized == prefix or normalized.startswith(prefix + "/"):
                return str(new).replace("\\", "/") + value.replace("\\", "/")[len(prefix):]
    return value


def write_document(path, value):
    temporary = path.with_name(path.name + ".portable-pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) if path.suffix.lower() == ".json" else
                         yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def rebase_documents(project, mapping):
    for path in structured_files(project):
        try:
            before = read_document(path)
        except (ValueError, yaml.YAMLError, UnicodeError):
            continue
        after = rewrite(before, mapping)
        if before != after:
            write_document(path, after)


def reindex_filter_history(project):
    # History directory identities include absolute video/pose paths.
    from hagisilta_pose_postprocess import source_key
    root = project_state_root(project) / "pose-filter-tests"
    mapping = {}
    if not root.is_dir():
        return
    for directory in list(root.iterdir()):
        if not directory.is_dir():
            continue
        records = list(directory.glob("*/composition.json")) + list(directory.glob("*/filter-test.json"))
        if not records:
            continue
        record = read_document(records[0])
        target = root / source_key(Path(record["source_video"]), Path(record["source_pose"]))
        if target != directory:
            if target.exists():
                raise ValueError(tr('보정 이력 경로 충돌: ') + str(target))
            directory.rename(target)
            mapping[str(directory)] = str(target)
    rebase_documents(project, mapping)


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def repair_config(config: Path, video_root: Path | None = None, mapping: dict | None = None) -> list[str]:
    """Rebase all structured project references, not only DLC video_sets."""
    config = config.resolve()
    payload = read_document(config) or {}
    replacements = dict(mapping or {})
    old_root = payload.get("project_path")
    if old_root and path_token(old_root) != path_token(config.parent):
        replacements[str(old_root)] = str(config.parent)
    backup = config.with_name("config.yaml.pre-portable-backup")
    if not backup.exists():
        shutil.copy2(config, backup)
    rebase_documents(config.parent, replacements)
    payload = read_document(config) or {}
    payload["project_path"] = str(config.parent)
    write_document(config, payload)
    referenced = set()
    for path in structured_files(config.parent):
        try:
            referenced.update(text for text in strings(read_document(path))
                              if absolute_text(text) and Path(text).suffix.lower() in VIDEO_SUFFIXES)
        except (ValueError, yaml.YAMLError, UnicodeError):
            continue
    candidates = {}
    if video_root:
        for path in video_root.resolve().rglob("*"):
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                candidates.setdefault(path.name.casefold(), []).append(path)
    links = {}
    missing = []
    for text in sorted(referenced):
        if Path(text).is_file():
            continue
        matches = candidates.get(Path(text).name.casefold(), [])
        if len(matches) == 1:
            links[text] = str(matches[0])
        else:
            missing.append(text)
    rebase_documents(config.parent, links)
    reindex_filter_history(config.parent)
    return missing


def export_bundle(config: Path, output: Path, include_videos: bool = False, include_results: bool = False) -> Path:
    config, output = config.resolve(), output.resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    project = config.parent
    output.parent.mkdir(parents=True, exist_ok=True)
    items = [*CORE_ITEMS, project_state_root(project).name]
    if include_videos:
        items.append("videos")
    if include_results:
        items.extend(("analysis", "labeled-videos", project_output_root(project).name))
    files = {}
    for name in dict.fromkeys(items):
        source = project / name
        for child in ([source] if source.is_file() else source.rglob("*") if source.is_dir() else []):
            if child.is_file() and child.resolve() != output and "__pycache__" not in child.parts and not any(
                part in {"runtime", ".hagisilta-trash", ".hagisilta-backups"} for part in child.relative_to(project).parts
            ) and (include_videos or child.suffix.lower() not in VIDEO_SUFFIXES):
                files["project/" + child.relative_to(project).as_posix()] = child
    # Follow only project-referenced videos/settings. Never bundle the repository's
    # unrelated global camera profiles or other experiments.
    queue = [path for path in files.values() if path.suffix.lower() in {".yaml", ".yml", ".json"}]
    from hagisilta_runtime import load_machine_settings
    machine = load_machine_settings()
    ui_state = {category: value for category in UI_CATEGORIES
                for key, value in (machine.get(category, {}) or {}).items() if path_token(key) == path_token(config)}
    ui_member = "project/" + project_state_root(project).name + "/portable-ui.json"
    if ui_state:
        previous_ui = files.pop(ui_member, None)
        queue = [path for path in queue if path != previous_ui]
        queue.append(ui_state)
    scanned, references, external, missing = set(), set(), {}, []
    while queue:
        path = queue.pop()
        if isinstance(path, dict):
            values = path
        else:
            if path in scanned:
                continue
            scanned.add(path)
            try:
                values = read_document(path)
            except (ValueError, yaml.YAMLError, UnicodeError):
                continue
        for text in strings(values):
            if not absolute_text(text):
                continue
            candidate = Path(text)
            suffix = candidate.suffix.lower()
            if suffix not in VIDEO_SUFFIXES | {".json", ".yaml", ".yml"}:
                continue
            if suffix in VIDEO_SUFFIXES:
                references.add(text)
                if not include_videos:
                    continue
            if not candidate.is_file():
                if suffix in VIDEO_SUFFIXES and text not in missing:
                    missing.append(text)
                continue
            candidate = candidate.resolve()
            if _inside(candidate, project):
                name = "project/" + candidate.relative_to(project).as_posix()
            else:
                identity = hashlib.sha256(path_token(candidate).encode("utf-8")).hexdigest()[:16]
                category = "source-videos" if suffix in VIDEO_SUFFIXES else "settings"
                name = f"project/portable-assets/{category}/{identity}/{candidate.name}"
                external[str(candidate)] = name.removeprefix("project/")
            if name not in files:
                files[name] = candidate
                if suffix in {".json", ".yaml", ".yml"}:
                    queue.append(candidate)
    if include_videos and missing:
        raise ValueError(tr('포함할 원본 영상이 없습니다. 영상 경로를 먼저 복구하세요:\n') + "\n".join(missing[:20]))
    manifest = {"format": "hagisilta-portable-project", "version": 2,
                "created_utc": datetime.now(timezone.utc).isoformat(), "project_name": project.name,
                "source_project": str(project), "included_videos": include_videos, "included_results": include_results,
                "path_map": external, "video_references": sorted(references), "files": {}}
    # Publish only a complete verified ZIP; an earlier bundle is preserved on failure.
    fd, pending = tempfile.mkstemp(prefix=output.stem + "-", suffix=".pending", dir=output.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, path in sorted(files.items()):
                before = path.stat()
                archive.write(path, name)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError(tr('내보내기 중 변경된 파일: ') + str(path))
                manifest["files"][name] = {"size": before.st_size, "mtime_ns": before.st_mtime_ns}
            if ui_state:
                archive.writestr(ui_member, json.dumps(ui_state, ensure_ascii=False, indent=2))
            archive.writestr("hagisilta-bundle.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        with zipfile.ZipFile(pending) as archive:
            failed = archive.testzip()
            if failed:
                raise ValueError(tr('ZIP 검증 실패: ') + failed)
        Path(pending).replace(output)
    finally:
        Path(pending).unlink(missing_ok=True)
    return output


def import_bundle(bundle: Path, destination_root: Path) -> tuple[Path, list[str]]:
    bundle, destination_root = bundle.resolve(), destination_root.resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        manifest_name = next((name for name in ("hagisilta-bundle.json", "nort-bundle.json") if name in names), None)
        if manifest_name is None or "project/config.yaml" not in names:
            raise ValueError(tr('HAGISILTA 프로젝트 번들이 아닙니다.'))
        manifest = json.loads(archive.read(manifest_name))
        base_name = str(manifest.get("project_name") or bundle.stem)
        if base_name in {".", ".."} or any(char in base_name for char in '/\\:') or not base_name:
            raise ValueError(tr('안전하지 않은 프로젝트 이름'))
        target, suffix = destination_root / base_name, 2
        while target.exists():
            target = destination_root / f"{base_name}-{suffix}"
            suffix += 1
        with tempfile.TemporaryDirectory(prefix="hagisilta-import-", dir=destination_root) as temp_name:
            temp = Path(temp_name)
            for info in archive.infolist():
                name = info.filename.replace("\\", "/")
                parts = PurePosixPath(name)
                if parts.is_absolute() or ".." in parts.parts or ":" in name or not _inside(temp / name, temp):
                    raise ValueError(tr('안전하지 않은 ZIP 경로: {v0}', v0=info.filename))
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(tr('ZIP 심볼릭 링크는 지원하지 않습니다.'))
            failed = archive.testzip()
            if failed:
                raise ValueError(tr('ZIP 손상: ') + failed)
            archive.extractall(temp)
            for name, info in manifest.get("files", {}).items():
                path = temp / name
                if not _inside(path, temp) or not path.is_file() or path.stat().st_size != info["size"]:
                    raise ValueError(tr('번들 파일 검증 실패: ') + name)
                os.utime(path, ns=(int(info["mtime_ns"]), int(info["mtime_ns"])))
            # Resolve against the final location while still staging. No global
            # repository assets or existing projects are overwritten.
            staged = temp / "project"
            legacy_assets = temp / ("hagisilta-assets" if (temp / "hagisilta-assets").exists() else "nort-assets")
            legacy_links = {}
            if legacy_assets.is_dir():
                local_assets = staged / "portable-assets" / "legacy"
                shutil.copytree(legacy_assets, local_assets)
                for document in structured_files(staged):
                    try:
                        values = strings(read_document(document))
                        for text in values:
                            normalized = text.replace("\\", "/")
                            for category in ("camera_profiles", "setups"):
                                marker = "/analysis/" + category + "/"
                                if marker in normalized:
                                    relative = category + "/" + normalized.split(marker, 1)[1]
                                    if _inside(local_assets / relative, local_assets) and (local_assets / relative).is_file():
                                        legacy_links[text] = str(target / "portable-assets" / "legacy" / relative)
                    except (ValueError, yaml.YAMLError, UnicodeError):
                        continue
            old = manifest.get("source_project") or (read_document(staged / "config.yaml") or {}).get("project_path")
            mapping = {str(old): str(target)} if old else {}
            mapping.update(legacy_links)
            if not manifest.get("included_videos"):
                # On the same computer a video-less bundle can still use the
                # original files. On another computer they are reported for relinking.
                for video in manifest.get("video_references", []):
                    mapping[video] = video
            for source, relative in manifest.get("path_map", {}).items():
                destination = target / relative
                if not _inside(destination, target):
                    raise ValueError(tr('안전하지 않은 영상/설정 매핑'))
                mapping[source] = str(destination)
            rebase_documents(staged, mapping)
            shutil.move(str(staged), str(target))
    config = target / "config.yaml"
    missing = repair_config(config)
    return config, missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Export/import relocatable HAGISILTA projects")
    sub = parser.add_subparsers(dest="action", required=True)
    export = sub.add_parser("export")
    export.add_argument("config", type=Path)
    export.add_argument("output", type=Path)
    export.add_argument("--include-videos", action="store_true")
    export.add_argument("--include-results", action="store_true")
    load = sub.add_parser("import")
    load.add_argument("bundle", type=Path)
    load.add_argument("destination", type=Path)
    load.add_argument("--report", type=Path)
    repair = sub.add_parser("repair")
    repair.add_argument("config", type=Path)
    repair.add_argument("--video-root", type=Path)
    repair.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.action == "export":
        path = export_bundle(args.config, args.output, args.include_videos, args.include_results)
        print(f"BUNDLE={path}")
    elif args.action == "import":
        config, missing = import_bundle(args.bundle, args.destination)
        if args.report:
            write_document(args.report, {"config": str(config), "missing_videos": missing})
        print(f"CONFIG={config}")
        if missing:
            print("MISSING_VIDEOS=" + json.dumps(missing, ensure_ascii=False))
    else:
        missing = repair_config(args.config, args.video_root)
        if args.report:
            write_document(args.report, {"config": str(args.config.resolve()), "missing_videos": missing})
        print(f"CONFIG={args.config.resolve()}")
        if missing:
            print("MISSING_VIDEOS=" + json.dumps(missing, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
