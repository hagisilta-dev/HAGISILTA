# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
"""Read-only, metadata-only filter provenance for result tables and summaries.

History discovery is explicit (table refresh / worker completion). Selection and
polling only stat known files and read changed small JSON records, never poses.
Full coordinate/hash validation remains in the analysis and comparison loaders.
"""

from hagisilta_i18n import tr
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

from hagisilta_pose_postprocess import CHOICE_FILE, COMPOSITION_SCHEMA, recipe_label, validate_recipe
from hagisilta_pose_postprocess import FILTER_SCHEMA, WINDOWS, file_stamp, source_key


@dataclass(frozen=True)
class FilterDisplay:
    title: str
    settings: str
    detail: str
    kind: str


def needs_check(error):
    return FilterDisplay(tr('확인 필요'), tr('확인 필요'), str(error), "invalid")


class FilterStatusReader:
    def __init__(self, root):
        self.root = Path(root)
        self._json_cache = {}
        self._history = {}

    def read_json(self, path):
        path = Path(path).resolve()
        stamp = file_stamp(path)
        cached = self._json_cache.get(path)
        if cached is None or cached[0] != stamp:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise ValueError(tr('설정 기록 형식이 올바르지 않습니다.'))
            self._json_cache[path] = stamp, record
        return self._json_cache[path][1]

    def refresh_history(self, video, pose):
        key = source_key(Path(video), Path(pose))
        directory = self.root / key
        paths = list(directory.glob("*/filter-test.json")) + list(directory.glob("*/composition.json"))
        rows = []
        for path in paths:
            try:
                created = datetime.fromisoformat(self.read_json(path)["created_at"]).timestamp()
            except (OSError, ValueError, KeyError, TypeError):
                # Do not silently present an older result as the newest when a
                # newly saved record cannot be read.
                try:
                    created = path.stat().st_mtime
                except OSError:
                    continue
            rows.append((created, str(path)))
        self._history[key] = [path for _created, path in sorted(rows, reverse=True)]

    def recent(self, video, pose, *, refresh=False):
        key = source_key(Path(video), Path(pose))
        if refresh or key not in self._history:
            self.refresh_history(video, pose)
        paths = self._history[key]
        if not paths:
            return FilterDisplay(tr('보정 없음'), tr('보정 없음'), tr('저장된 좌표 후처리 보정본이 없습니다.'), "none")
        result = self.describe(paths[0], video, pose)
        return FilterDisplay(result.title, result.settings,
                             tr('저장된 보정 {v0}개 · 가장 최근 저장본\n{v1}', v0=len(paths), v1=result.detail), result.kind)

    def describe(self, manifest, video, pose):
        try:
            manifest, video, pose = Path(manifest).resolve(), Path(video).resolve(), Path(pose).resolve()
            record = self.read_json(manifest)
            if record.get("schema") == COMPOSITION_SCHEMA:
                recipe = validate_recipe(record["recipe"])
                title = tr('신체점별 · ') + (record.get("name") or tr('이름 없는 조합'))
                settings = title + " · " + recipe_label(recipe)
            elif (record.get("schema") == FILTER_SCHEMA and record.get("method") == "median"
                  and record.get("window") in WINDOWS):
                title = settings = tr('중앙값 {v0}프레임', v0=record['window'])
            else:
                raise ValueError(tr('지원하지 않는 보정 결과 기록입니다.'))
            if (Path(record["source_video"]).resolve() != video or
                    Path(record["source_pose"]).resolve() != pose):
                raise ValueError(tr('보정본의 영상·모델이 현재 결과와 다릅니다.'))
            self.check_stamps(record, video, pose)
            for name in ("original", "filtered"):
                member = (manifest.parent / record[name]).resolve()
                if member.parent != manifest.parent or not member.is_file() or member.stat().st_size == 0:
                    raise ValueError(tr('보정본의 좌표 파일이 없거나 경로가 올바르지 않습니다.'))
            created = datetime.fromisoformat(record["created_at"]).astimezone().strftime("%Y-%m-%d %H:%M:%S")
            detail = tr('{v0}\n저장: {v1}\n추론: {v2}\n보정 기록: {v3}', v0=settings, v1=created, v2=pose.name, v3=manifest)
            return FilterDisplay(title, settings, detail, "filtered")
        except (OSError, ValueError, KeyError, TypeError) as error:
            return needs_check(tr('{v0}\n보정 기록: {v1}', v0=error, v1=manifest))

    @staticmethod
    def check_stamps(record, video, pose):
        if file_stamp(video) != record.get("video_stamp"):
            raise ValueError(tr('보정 기록 이후 영상이 변경되었거나 영상 기록 확인이 필요합니다.'))
        if file_stamp(pose) != record.get("source_pose_stamp"):
            raise ValueError(tr('원본 추론이 변경되었거나 이전 기록의 재확인이 필요합니다.'))

    def analysis(self, video, pose):
        video, pose = Path(video).resolve(), Path(pose).resolve()
        try:
            choice_path = video.parent / CHOICE_FILE
            if not choice_path.exists():
                return FilterDisplay(tr('보정 전'), tr('보정 전'), tr('분석 입력을 따로 지정하지 않아 보정 전 추론 좌표를 사용합니다.'), "raw")
            choice = self.read_json(choice_path)
            if choice.get("schema") != "hagisilta-analysis-pose-v1" or Path(choice["source_video"]).resolve() != video:
                raise ValueError(tr('분석 입력 기록의 형식 또는 영상 경로가 올바르지 않습니다.'))
            source_pose = Path(choice["source_pose"]).resolve()
            self.check_stamps(choice, video, source_pose)
            if choice.get("composition"):
                result = self.describe(choice["composition"], video, source_pose)
            else:
                result = FilterDisplay(tr('보정 전'), tr('보정 전'), tr('보정 전 추론: {v0}', v0=source_pose), "raw")
            if source_pose != pose and result.kind != "invalid":
                return FilterDisplay(tr('다른 모델 · ') + result.title, tr('다른 모델 · ') + result.settings,
                                     tr('이 행이 아닌 다른 추론 결과가 분석 입력으로 지정되어 있습니다.\n{v0}', v0=result.detail), "other")
            return result
        except (OSError, ValueError, KeyError, TypeError) as error:
            return needs_check(tr('{v0}\n분석 입력 기록: {v1}', v0=error, v1=video.parent / CHOICE_FILE))
