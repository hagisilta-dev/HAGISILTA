# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
"""Atomic sidecar writes resilient to short-lived Windows sharing locks.

The old destination is never deleted or truncated to work around a lock.
Permanent permission failures and other I/O errors remain errors.
"""
from __future__ import annotations

from hagisilta_i18n import tr

import errno
import json
import os
from pathlib import Path
import sys
import tempfile
import time


REPLACE_RETRY_DELAYS = (0.05, 0.10, 0.20, 0.40, 0.80, 1.0, 1.0)


def replace_with_retry(pending: Path, target: Path) -> None:
    """Publish once complete, with at most 3.55 seconds of lock backoff."""
    for attempt in range(len(REPLACE_RETRY_DELAYS) + 1):
        try:
            os.replace(pending, target)
            return
        except OSError as error:
            retryable = (getattr(error, "winerror", None) in {5, 32, 33}
                         or error.errno in {errno.EACCES, errno.EPERM})
            if not retryable or attempt == len(REPLACE_RETRY_DELAYS):
                raise
            if attempt == 0:
                print(f"FILE_REPLACE_RETRY={target}:" + tr('일시적 파일 접근 충돌 · 재시도'), file=sys.stderr, flush=True)
            time.sleep(REPLACE_RETRY_DELAYS[attempt])


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        # A fresh name also avoids re-opening yesterday's locked .pending file.
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=path.name + ".",
                                         suffix=".pending", delete=False) as stream:
            pending = Path(stream.name)
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        replace_with_retry(pending, path)
    finally:
        if pending is not None:
            try:
                pending.unlink(missing_ok=True)
            except OSError:
                # Only this write's private temporary file may remain. Never
                # remove the published destination or mask the original error.
                pass


def atomic_json(path: Path, value: dict) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    atomic_write_bytes(path, encoded)
