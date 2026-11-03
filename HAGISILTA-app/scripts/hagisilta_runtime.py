# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
from pathlib import Path


APP_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local")) / "HAGISILTA"
SETTINGS_FILE = APP_DIR / "machine-settings.json"
SETTINGS_BACKUP_FILE = APP_DIR / "machine-settings.previous.json"
_settings_backup_created = False
SETTINGS_FILE_ENV = "HAGISILTA_SETTINGS_FILE"


def machine_settings_file() -> Path:
    """Resolve a complete read/write settings override for isolated UI checks.

    Resolve lazily so a test can isolate settings after importing UI modules.
    Child processes inherit the same override through their environment.
    """
    override = os.environ.get(SETTINGS_FILE_ENV, "").strip()
    return Path(override).expanduser().resolve() if override else SETTINGS_FILE


def default_work_root() -> Path:
    """Return the current Windows Documents-based HAGISILTA workspace."""
    if os.name == "nt":
        try:
            buffer = ctypes.create_unicode_buffer(32768)
            # CSIDL_PERSONAL resolves redirected/OneDrive Documents correctly.
            result = ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buffer)
            if result == 0 and buffer.value:
                return Path(buffer.value) / "HAGISILTA-work"
        except (AttributeError, OSError):
            pass
    return Path.home() / "Documents" / "HAGISILTA-work"


def project_output_root(project: Path, create: bool = False) -> Path:
    """Return the branded output folder while preserving legacy projects."""
    project = Path(project)
    current = project / "hagisilta-outputs"
    legacy = project / "nort-outputs"
    root = current if current.exists() or not legacy.exists() else legacy
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def project_state_root(project: Path, create: bool = False) -> Path:
    """Return the hidden project state folder with legacy fallback."""
    project = Path(project)
    current = project / ".hagisilta"
    legacy = project / ".nort"
    root = current if current.exists() or not legacy.exists() else legacy
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def project_backup_root(project: Path, create: bool = False) -> Path:
    project = Path(project)
    current = project / ".hagisilta-backups"
    legacy = project / ".nort-backups"
    root = current if current.exists() or not legacy.exists() else legacy
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def project_dataset_metadata(payload: dict, create: bool = False) -> dict:
    """Return the dataset metadata block for current and legacy projects."""
    if "hagisilta_dataset" in payload or "nort_dataset" not in payload:
        return (
            payload.setdefault("hagisilta_dataset", {})
            if create
            else (payload.get("hagisilta_dataset", {}) or {})
        )
    return (
        payload.setdefault("nort_dataset", {})
        if create
        else (payload.get("nort_dataset", {}) or {})
    )


def edit_project_config(config: str | Path, updates: dict) -> dict:
    """Update project YAML without wrapping long path keys or truncating the file."""
    import io
    import tempfile

    from ruamel.yaml import YAML

    config = Path(config)
    codec = YAML()
    payload = codec.load(config.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Project configuration must contain a mapping: {config}")
    payload.update(updates)
    # DLC edit_config uses write_plainconfig, whose default width can split a
    # Windows path at a space and emit an invalid multiline mapping key.
    codec.width = 1_000_000
    buffer = io.StringIO()
    codec.dump(payload, buffer)
    serialized = buffer.getvalue()
    if YAML().load(serialized) != payload:
        raise ValueError(f"Project configuration failed YAML round-trip validation: {config}")

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=config.parent,
            prefix=f".{config.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, config)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return payload


def install_unicode_safe_opencv_reader() -> None:
    """Patch cv2.imread for Windows paths containing non-ASCII characters.

    OpenCV's filename API can return ``None`` for otherwise valid files under
    paths such as OneDrive\\문서. ``np.fromfile`` plus ``cv2.imdecode`` uses the
    Unicode-aware Windows file API while preserving normal OpenCV decoding.
    The marker keeps repeated imports and spawned DataLoader workers safe.
    """
    import cv2
    import numpy as np

    if getattr(cv2.imread, "_hagisilta_unicode_safe", False):
        return
    native_imread = cv2.imread

    def unicode_imread(filename, flags=cv2.IMREAD_COLOR):
        filename_text = str(filename)
        image = native_imread(filename_text, flags) if filename_text.isascii() else None
        if image is not None:
            return image
        path = Path(filename)
        if not path.is_file():
            return None
        try:
            encoded = np.fromfile(path, dtype=np.uint8)
            return cv2.imdecode(encoded, flags)
        except (OSError, ValueError):
            return None

    unicode_imread._hagisilta_unicode_safe = True
    cv2.imread = unicode_imread


def load_machine_settings() -> dict:
    # Never silently revive NORT-Studio paths or projects. Legacy projects are
    # still supported when the user explicitly opens or imports them.
    settings_file = machine_settings_file()
    if not settings_file.exists():
        return {}
    try:
        settings = json.loads(settings_file.read_text(encoding="utf-8-sig"))
        return settings if isinstance(settings, dict) else {}
    except Exception:
        return {}


def save_machine_settings(settings: dict) -> None:
    global _settings_backup_created
    settings_file = machine_settings_file()
    settings_file.parent.mkdir(parents=True, exist_ok=True)
    backup_file = (
        settings_file.with_name(f"{settings_file.stem}.previous.json")
        if os.environ.get(SETTINGS_FILE_ENV, "").strip() else SETTINGS_BACKUP_FILE
    )
    if settings_file.is_file() and not _settings_backup_created:
        shutil.copy2(settings_file, backup_file)
        _settings_backup_created = True
    temporary = settings_file.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, settings_file)


def get_work_root(create: bool = False) -> Path:
    environment = os.environ.get("HAGISILTA_WORK_ROOT")
    if environment:
        root = Path(environment).expanduser()
    else:
        stored = load_machine_settings().get("work_root")
        if stored:
            root = Path(stored).expanduser()
        else:
            root = default_work_root()
    root = root.resolve()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def set_work_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    settings = load_machine_settings()
    settings["work_root"] = str(root)
    save_machine_settings(settings)
    return root


def detect_device() -> str:
    requested = os.environ.get("HAGISILTA_DEVICE", "auto").strip().lower()
    if os.environ.get("HAGISILTA_FORCE_CPU") == "1" or requested == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available() and requested in {"auto", "cuda"}:
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def hardware_summary() -> dict:
    device = detect_device()
    summary = {"device": device, "gpu": None}
    if device.startswith("cuda"):
        try:
            import torch

            summary["gpu"] = torch.cuda.get_device_name(0)
        except Exception:
            summary["gpu"] = "CUDA GPU"
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="HAGISILTA per-machine runtime settings")
    parser.add_argument("--set-work-root", type=Path)
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = set_work_root(args.set_work_root) if args.set_work_root else get_work_root(args.create)
    payload = {"work_root": str(root), "settings_file": str(SETTINGS_FILE), **hardware_summary()}
    print(json.dumps(payload, ensure_ascii=False) if args.json else str(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
