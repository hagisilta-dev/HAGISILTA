# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
"""Read-only source identity checks for legacy stem-based output directories."""
from __future__ import annotations

from hagisilta_i18n import tr

from collections.abc import Iterable
import os
from pathlib import Path
import unicodedata


def normalized_source_path(path: str | os.PathLike) -> str:
    """Compare source paths consistently, including paths not on disk yet."""
    resolved = Path(path).expanduser().resolve(strict=False)
    return unicodedata.normalize("NFC", os.path.normcase(str(resolved))).casefold()


def validate_unique_source_stems(paths: Iterable[str | os.PathLike]) -> list[Path]:
    """Reject ambiguous legacy output names before creating project artifacts.

    Repeated references to the same source are harmless here and are deduplicated.
    Different extensions and directories do not distinguish the output folder.
    Keep existing output names: rejecting a conflict is safer than renaming saved
    ROI, tracking, and composition references behind the user's back.
    """
    owners: dict[str, Path] = {}
    seen_paths: set[str] = set()
    unique: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve(strict=False)
        identity = normalized_source_path(path)
        if identity in seen_paths:
            continue
        stem = unicodedata.normalize("NFC", path.stem).casefold()
        previous = owners.get(stem)
        if previous is not None:
            raise ValueError(
                tr('영상 이름이 같아 ROI·분석 결과 저장 위치가 충돌합니다. 확장자를 제외한 영상 이름을 서로 다르게 바꾼 뒤 다시 추가해 주세요.\n- {v0}\n- {v1}', v0=previous, v1=path)
            )
        owners[stem] = path
        seen_paths.add(identity)
        unique.append(path)
    return unique
