# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
"""ROI lifecycle events; status describes finished work, never mere file counts."""

from hagisilta_i18n import tr
from collections import Counter
from contextlib import contextmanager
import inspect
import json
import os
from pathlib import Path
import threading
import time
from unittest.mock import patch
import uuid

PREFIX = "HAGISILTA_TRACK_EVENT "
STATUS_FILE = "tracking-status.json"
PHASES = {
    "queued": "대기", "cropping": "ROI 준비 중", "ready": "추론 대기",
    "inference": "추론 중", "writing": "좌표 저장 중",
    "coordinates_ready": "좌표 저장 완료", "video_pending": "확인 영상 대기",
    "labeling": "확인 영상 생성 중", "done": "완료",
    "error": "오류", "interrupted": "중단", "not_run": "미실행",
}
TERMINAL = {"done", "error", "interrupted", "not_run"}
READABLE = {"coordinates_ready", "video_pending", "labeling", "done"}
ROI_READY_PHASES = {"ready", "inference", "writing", *READABLE}
CROP_STAGES = {
    "encoding": "ROI 생성", "fast_validation": "빠른 검사",
    "validation": "정밀 검사", "full_validation": "정밀 검사",
}
# Fixed order is independent of the latest event/worker to report progress.
SUMMARY_STAGES = (
    ("error", "오류"), ("interrupted", "중단"), ("not_run", "미실행"),
    ("queued", "대기"), ("cropping", "ROI 준비"), ("encoding", "ROI 생성"),
    ("fast_validation", "빠른 검사"), ("validation", "정밀 검사"),
    ("ready", "추론 대기"), ("inference", "추론"), ("writing", "좌표 저장"),
    ("coordinates_ready", "좌표 저장 완료"), ("video_pending", "확인 영상 대기"),
    ("labeling", "확인 영상 생성"), ("unknown", "상태 확인 필요"),
)


def path_key(path):
    return os.path.normcase(str(Path(path).resolve())).casefold()


def stamp(path):
    info = Path(path).stat()
    return [info.st_size, info.st_mtime_ns]


def pose_key(path):
    path = Path(path)
    if path.parent.name == "comparison-results" and path.stem.endswith("__spatial"):
        path = path.parent.parent / (path.stem[:-len("__spatial")] + path.suffix)
    return path_key(path)


def saved_status(directory, shuffle=None):
    """Read-only restart fallback. Running records are not live after restart."""
    try:
        record = json.loads((Path(directory) / STATUS_FILE).read_text(encoding="utf-8"))
        if shuffle is not None and record.get("shuffle") != shuffle:
            return None
        phase = record["phase"]
        if phase == "done":
            for name in ("pose", "labeled"):
                if name == "labeled" and not record.get("labeled_required"):
                    continue
                if not record.get(name) or stamp(record[name]) != record.get(name + "_stamp"):
                    return None
        elif phase not in TERMINAL:
            # Preserve the completed preparation milestone when an older saved
            # event is presented as interrupted after restarting the app.
            record.setdefault("roi_ready", _roi_is_ready(record))
            record["phase"] = "interrupted"
        return record
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _nonnegative_int(value):
    """Event counters are optional in older status records."""
    try:
        if isinstance(value, bool):
            return None
        number = int(value)
        return number if number >= 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def _progress_records(records):
    return [record for record in (records or ()) if isinstance(record, dict)]


def _roi_is_ready(record):
    milestone = record.get("roi_ready")
    if isinstance(milestone, bool):
        return milestone
    # Backward compatibility for old worker events. A currently regenerated
    # ROI can explicitly set False while retaining previously saved coordinates.
    return (record.get("phase") in ROI_READY_PHASES
            or bool(record.get("pose")) or bool(record.get("labeled")))


def pipeline_progress_counts(records):
    """Count independent milestones from the latest event for each ROI.

    No filesystem access: preparation, saved coordinates, optional check video,
    and complete pipeline work have different meanings. Missing initial events
    do not reduce the advertised run total or invent completed work.
    """
    records = _progress_records(records)
    return {
        "total": max([len(records)] + [_nonnegative_int(item.get("total")) or 0 for item in records]),
        "roi_ready": sum(_roi_is_ready(item) for item in records),
        "coordinates": sum(bool(item.get("pose")) or bool(item.get("labeled"))
                           or item.get("phase") in READABLE for item in records),
        "done": sum(item.get("phase") == "done" for item in records),
        "labeled_required": any(bool(item.get("labeled_required")) for item in records),
        "labeled": sum(bool(item.get("labeled"))
                       or (item.get("phase") == "done" and bool(item.get("labeled_required")))
                       for item in records),
    }


def phase_text(record):
    if record.get("phase") == "cropping":
        label = CROP_STAGES.get(record.get("crop_stage"))
        if label:
            label = tr(label)
        if label:
            frames = _nonnegative_int(record.get("crop_frames"))
            total = _nonnegative_int(record.get("crop_total"))
            if frames is not None and total:
                percent = min(100, frames * 100 // total)
                return tr('{v0} {v1:,}/{v2:,} 프레임 ({v3}%)', v0=label, v1=frames, v2=total, v3=percent)
            if frames is not None:
                return tr('{v0} {v1:,} 프레임', v0=label, v1=frames)
            return label + tr(' 중')
    return tr(PHASES.get(record.get("phase"), '좌표 있음'))


def _summary_stage(record):
    phase = record.get("phase")
    if phase == "cropping":
        stage = record.get("crop_stage")
        if stage == "full_validation":
            return "validation"
        return stage if stage in CROP_STAGES else "cropping"
    return phase if phase in PHASES else "unknown"


def _stage_counts(records):
    counts = Counter(_summary_stage(item) for item in records)
    return [f"{tr(label)} {counts[stage]}" for stage, label in SUMMARY_STAGES if counts[stage]]


def _arena_order(record):
    arena = record.get("arena_id", "?")
    number = _nonnegative_int(arena)
    return (0, number, str(arena)) if number is not None else (1, 0, str(arena))


def _source_name(record):
    # Pure string handling: live progress must never rescan/resolve video files.
    return str(record.get("source") or record.get("video") or "").replace("\\", "/").rstrip("/")


def _roi_detail(record):
    text = f"ROI {record.get('arena_id', '?')}: {phase_text(record)}"
    if record.get("phase") in {"error", "interrupted"} and record.get("error"):
        text += " · " + " ".join(str(record["error"]).split())
    return text


def source_summary(records):
    records = list(records)
    if not records:
        return tr('준비'), ""
    done = sum(item.get("phase") == "done" for item in records)
    if done == len(records):
        text = tr(PHASES["done"])
    elif len(records) == 1:
        text = phase_text(records[0])
    else:
        text = " · ".join(_stage_counts(records))
    if done and done != len(records):
        text += tr(' · {v0}/{v1} 완료', v0=done, v1=len(records))
    detail = "\n".join(_roi_detail(item) for item in sorted(records, key=_arena_order))
    return text, detail


def parallel_progress_summary(records):
    """Return a bounded stage summary and stable rows for every active ROI.

    Queued/ready/saved/finished rows stay in the counts so large queues cannot
    bury the workers currently encoding, checking, inferring or writing.
    This is a pure rendering helper; callers keep the latest record per ROI.
    """
    records = _progress_records(records)
    if not records:
        return tr('ROI 작업 대기'), tr('ROI 작업이 시작되면 진행 상태가 표시됩니다.')
    counts = pipeline_progress_counts(records)
    done, total = counts["done"], counts["total"]
    milestones = tr('ROI 준비 {v0}/{v1} · 좌표 저장 {v2}/{v3}', v0=counts['roi_ready'], v1=total, v2=counts['coordinates'], v3=total)
    labeled = (tr('확인 영상 {v0}/{v1}', v0=counts['labeled'], v1=total) if counts["labeled_required"]
               else tr('확인 영상 생성 안 함'))
    summary = milestones + "\n" + " · ".join(
        [tr('전체 완료 {v0}/{v1}', v0=done, v1=total), labeled] + _stage_counts(records)
    )
    visible_phases = {"cropping", "inference", "writing", "labeling", "error", "interrupted"}
    active = sorted((item for item in records if item.get("phase") in visible_phases),
                    key=lambda item: (_source_name(item).casefold(), _arena_order(item)))
    source_paths = {}
    for item in active:
        source = _source_name(item)
        source_paths.setdefault(source.rsplit("/", 1)[-1].casefold(), set()).add(source.casefold())
    lines = []
    for item in active:
        source = _source_name(item)
        name = source.rsplit("/", 1)[-1]
        if len(source_paths[name.casefold()]) > 1:
            name = source
        lines.append(f"{name or tr('영상 미지정')} · {_roi_detail(item)}")
    if not lines:
        if done == total:
            lines.append(tr('모든 ROI 작업이 완료되었습니다.'))
        elif len(records) == total and all(item.get("phase") in TERMINAL for item in records):
            lines.append(tr('실행되지 않은 ROI가 있습니다. 실행 로그를 확인하세요.'))
        else:
            lines.append(tr('다음 ROI 처리 단계를 기다리는 중입니다.'))
    return summary, "\n".join(lines)


def source_disk_summary(directory, roi_count, shuffle=None):
    records, coordinates = [], 0
    for arena in sorted(Path(directory).glob("arena_*")):
        if not arena.is_dir():
            continue
        poses = [p for p in arena.glob("*.h5") if p.stat().st_size > 0
                 and "evaluation-results" not in p.name and not p.stem.endswith("_spatial")]
        if shuffle is not None:
            import re
            poses = [p for p in poses if re.search(rf"shuffle{int(shuffle)}(?:_|\D|$)", p.stem)]
        coordinates += bool(poses)
        record = saved_status(arena, shuffle)
        if record:
            records.append(record)
    if len(records) == roi_count and roi_count:
        text, detail = source_summary(records)
    else:
        text = tr('좌표 있음 {v0}/{v1}', v0=coordinates, v1=roi_count) if coordinates else tr('준비')
        detail = tr('좌표 저장 여부입니다. 확인 영상 생성 완료와 구분합니다.')
    return text, detail, coordinates


def notify(job, phase, **fields):
    progress = job.get("_tracking_progress")
    if progress is not None:
        progress.emit(job, phase, **fields)


class TrackingProgress:
    def __init__(self, jobs, config, args):
        self.jobs = list(jobs)
        self.run_id = uuid.uuid4().hex
        self.lock = threading.RLock()
        self.records = {}
        self.config = str(Path(config).resolve())
        self.shuffle = args.shuffle
        self.required = bool(args.labeled_video)
        for job in self.jobs:
            job["_tracking_progress"] = self
            self.emit(job, "queued")

    def emit(self, job, phase, **fields):
        with self.lock:
            key = path_key(job["cropped"])
            record = dict(self.records.get(key, {}))
            previous_phase = record.get("phase")
            if phase == "queued":
                # A new attempt must not inherit prior output/milestone evidence.
                record = {}
            elif phase == "inference":
                # New inference replaces earlier coordinates; label-only reuse
                # never enters this phase and retains its saved coordinate count.
                for name in ("pose", "pose_stamp", "labeled", "labeled_stamp"):
                    record.pop(name, None)
            elif phase == "cropping" and previous_phase != "cropping":
                for name in ("crop_stage", "crop_frames", "crop_total"):
                    record.pop(name, None)
            record.update(run_id=self.run_id, config=self.config, shuffle=self.shuffle,
                          total=len(self.jobs), source=str(job["source"].resolve()),
                          video=str(job["cropped"].resolve()), arena_id=job["arena_id"],
                          labeled_required=self.required, phase=phase, updated=time.time())
            record.update({name: str(value) if isinstance(value, Path) else value for name, value in fields.items()})
            if phase in {"queued", "cropping"}:
                record["roi_ready"] = False
            elif phase in ROI_READY_PHASES or fields.get("pose") or fields.get("labeled"):
                record["roi_ready"] = True
            else:
                record.setdefault("roi_ready", _roi_is_ready(record))
            for name in ("pose", "labeled"):
                if record.get(name) and Path(record[name]).is_file():
                    record[name + "_stamp"] = stamp(record[name])
            self.records[key] = record
            directory = Path(job["arena_output"])
            pending = directory / (STATUS_FILE + ".tmp")
            try:
                directory.mkdir(parents=True, exist_ok=True)
                pending.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                pending.replace(directory / STATUS_FILE)
            except OSError as error:
                print(f"TRACK_STATUS_WARNING={error}", flush=True)
            # Start on a new line even when DLC's progress bar uses carriage returns.
            print("\n" + PREFIX + json.dumps(record, ensure_ascii=False), flush=True)

    def finish_unfinished(self, phase="interrupted"):
        for job in self.jobs:
            current = self.records[path_key(job["cropped"])]["phase"]
            if current not in TERMINAL:
                self.emit(job, "not_run" if current in {"queued", "ready"}
                          else "interrupted" if current in {"coordinates_ready", "video_pending"} else phase)

    def label_event(self, phase, video, pose, output=None):
        job = next((job for job in self.jobs if path_key(job["cropped"]) == path_key(video)), None)
        if job:
            self.emit(job, phase, pose=Path(pose), **({"labeled": Path(output)} if output else {}))


@contextmanager
def inference_events(jobs):
    """Observe native DLC boundaries without splitting its shared-model batch call."""
    if not any(job.get("_tracking_progress") for job in jobs):
        yield
        return
    from deeplabcut.pose_estimation_pytorch.apis import videos

    native_infer, native_save = videos.video_inference, videos.create_df_from_prediction
    by_video = {path_key(job["cropped"]): job for job in jobs}

    def infer(video, *args, **kwargs):
        key = path_key(getattr(video, "video_path", video))
        job = by_video.get(key)
        if job:
            notify(job, "inference")
        try:
            result = native_infer(video, *args, **kwargs)
        except BaseException as error:
            if job:
                notify(job, "error", error=str(error))
            raise
        if job:
            notify(job, "writing")
        return result

    def save(*args, **kwargs):
        values = inspect.signature(native_save).bind(*args, **kwargs).arguments
        pose = Path(values["output_path"]) / f"{values['output_prefix']}.h5"
        job = next((job for job in jobs if path_key(job["arena_output"]) == path_key(pose.parent)
                    and pose.stem.startswith(job["cropped"].stem)), None)
        result = native_save(*args, **kwargs)
        if job:
            notify(job, "coordinates_ready", pose=pose)
        return result

    with patch.object(videos, "video_inference", infer), patch.object(videos, "create_df_from_prediction", save):
        yield
