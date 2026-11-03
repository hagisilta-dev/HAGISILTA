# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
"""Offline Korean/English UI messages, fixed for the lifetime of one launch.

Messages use their Korean source as the catalog key. Translate templates before
formatting them: ``tr("선택한 {count}개 행", count=count)``. Project names, paths,
model identifiers and other user data must be passed as values, never translated.
The module deliberately performs no settings I/O at import time.
"""
from __future__ import annotations

from collections import Counter
import json
import logging
from numbers import Number
import os
from pathlib import Path
import re
from string import Formatter

try:
    from PySide6.QtCore import QCoreApplication, QTranslator
except ImportError:
    # Headless calculation workers use the same messages without requiring Qt.
    QCoreApplication = None
    QTranslator = object

import hagisilta_runtime as runtime


LANGUAGE_ENV = "HAGISILTA_UI_LANGUAGE"
LANGUAGE_SETTING = "ui_language"
SUPPORTED_LANGUAGES = ("ko", "en")
CATALOG_DIRECTORY = Path(__file__).resolve().parents[1] / "assets" / "i18n"
_session_language: str | None = None
_english_catalog: dict[str, str] | None = None
_logger = logging.getLogger(__name__)
# Catalog-only notation: {count} [[count|video|videos]]. Resolve the trusted
# template before formatting, so names/paths containing similar text stay intact.
_PLURAL_FORM = re.compile(r"\[\[([A-Za-z_]\w*)\|([^\[\]|{}]+)\|([^\[\]|{}]+)\]\]")


def _valid_language(value: object) -> str | None:
    language = value.strip().lower() if isinstance(value, str) else ""
    return language if language in SUPPORTED_LANGUAGES else None


def get_language() -> str:
    """Read the preference once and freeze it for this process and its children."""
    global _session_language
    if _session_language is None:
        _session_language = _valid_language(os.environ.get(LANGUAGE_ENV))
        if _session_language is None:
            _session_language = _valid_language(
                runtime.load_machine_settings().get(LANGUAGE_SETTING)
            ) or "ko"
        os.environ[LANGUAGE_ENV] = _session_language
    return _session_language


def save_language(language: str) -> None:
    """Save the next launch's language while keeping every current window stable."""
    selected = _valid_language(language)
    if selected is None:
        raise ValueError(f"Unsupported UI language: {language!r}")
    # Freeze the current preference before changing the persisted preference.
    get_language()
    settings = runtime.load_machine_settings()
    settings[LANGUAGE_SETTING] = selected
    runtime.save_machine_settings(settings)


def _placeholders(message: str) -> Counter:
    """Validate complete Python format fields, including nested specifications."""
    fields: Counter = Counter()
    for _literal, field, specification, conversion in Formatter().parse(message):
        if field is not None:
            fields[(field, specification, conversion)] += 1
            if specification:
                fields.update(_placeholders(specification))
    return fields


def _valid_plural_forms(source: str, translated: str) -> bool:
    if "[[" not in translated and "]]" not in translated:
        return True
    try:
        count_fields = {field for field, _specification, _conversion in _placeholders(source)}
    except ValueError:
        return False
    if "%n" in source:
        count_fields.add("n")
    matches = list(_PLURAL_FORM.finditer(translated))
    remainder = _PLURAL_FORM.sub("", translated)
    return (
        "[[" not in remainder and "]]" not in remainder
        and all(match[1] in count_fields for match in matches)
    )


def _select_plural_forms(message: str, counts: dict[str, object]) -> str:
    def select(match: re.Match) -> str:
        count = counts.get(match[1])
        singular = isinstance(count, Number) and not isinstance(count, bool) and count == 1
        return match[2] if singular else match[3]

    return _PLURAL_FORM.sub(select, message)


def _load_english_catalog() -> dict[str, str]:
    global _english_catalog
    if _english_catalog is not None:
        return _english_catalog
    catalog: dict[str, str] = {}
    for path in sorted(CATALOG_DIRECTORY.glob("en*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(payload, dict):
                raise ValueError("the catalog must be a source-to-translation mapping")
        except (OSError, ValueError) as error:
            _logger.warning("Cannot load UI catalog %s: %s", path.name, error)
            continue
        for source, translated in payload.items():
            if not isinstance(source, str) or not isinstance(translated, str) or not translated:
                _logger.warning("Ignoring invalid UI message in %s", path.name)
                continue
            try:
                source_fields = _placeholders(source)
            except ValueError:
                # Literal JSON or braces are valid UI text when no formatting is
                # requested. Requiring exact braces avoids swallowing a field.
                matches = source.count("{") == translated.count("{") and source.count("}") == translated.count("}")
            else:
                try:
                    matches = source_fields == _placeholders(translated)
                except ValueError:
                    # A valid source template must never accept malformed format
                    # syntax from a translation, even if brace counts agree.
                    matches = False
            if not matches:
                _logger.warning("Ignoring UI message with changed placeholders in %s: %s", path.name, source)
                continue
            if not _valid_plural_forms(source, translated):
                _logger.warning("Ignoring UI message with invalid plural forms in %s: %s", path.name, source)
                continue
            if source in catalog and catalog[source] != translated:
                _logger.warning("Conflicting UI message in %s; keeping the first: %s", path.name, source)
                continue
            catalog[source] = translated
    _english_catalog = catalog
    return catalog


def translate(source: str) -> str:
    """Translate an exact source message, preserving unknown text verbatim."""
    return _translated_message(source, {})


def _translated_message(source: str, counts: dict[str, object]) -> str:
    if get_language() == "en":
        translated = _load_english_catalog().get(source)
        if translated is not None:
            return _select_plural_forms(translated, counts)
    return source


def tr(source: str, **formatargs: object) -> str:
    """Translate a complete template before inserting untouched user values."""
    message = _translated_message(source, formatargs)
    return message.format(**formatargs) if formatargs else message


class HagisiltaTranslator(QTranslator):
    """Expose the same static catalog to Qt's standard translation interface."""

    def isEmpty(self) -> bool:
        # QTranslator's native implementation sees no .qm data in this subclass.
        return False

    def translate(self, context, sourceText, disambiguation=None, n=-1) -> str | None:
        if not sourceText or get_language() != "en":
            return None
        translated = _load_english_catalog().get(sourceText)
        if translated is None:
            # PySide maps None to a null QString (not found). An empty Python
            # string is a valid empty translation and would hide the source.
            return None
        translated = _select_plural_forms(translated, {"n": n})
        return translated.replace("%n", str(n)) if n >= 0 else translated


def init_i18n(app: QCoreApplication) -> HagisiltaTranslator:
    """Call immediately after QApplication creation, before constructing widgets."""
    if QCoreApplication is None:
        raise RuntimeError("Qt is required to initialize GUI translations")
    get_language()
    installed = getattr(app, "_hagisilta_translator", None)
    if installed is not None:
        return installed
    translator = HagisiltaTranslator(app)
    app.installTranslator(translator)
    # Retain the Python subclass as well as Qt ownership for virtual callbacks.
    app._hagisilta_translator = translator
    return translator
