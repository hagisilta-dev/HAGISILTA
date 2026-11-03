# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from hagisilta_i18n import tr

from collections.abc import Callable, Iterable
from contextlib import ExitStack
from pathlib import Path
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

from hagisilta_atomic_io import replace_with_retry


TRACKING_CROP_CODEC = "FFV1"
TRACKING_CROP_CONTAINER = "avi"
TRACKING_CROP_DISK_RESERVE_BYTES = 64 * 1024 * 1024
TRACKING_CROP_DISK_CHECK_INTERVAL = 50
SOURCE_FRAME_SHORTAGE_MIN_TOLERANCE = 2
SOURCE_FRAME_SHORTAGE_RATIO = 0.001
TRACKING_CROP_FAST_SAMPLE_COUNT = 9


def _ffmpeg_executable() -> str | None:
    # imageio-ffmpeg is bundled in the DLC environment. No download/install here.
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except (ImportError, OSError, RuntimeError):
        return None


class _FFmpegCropWriter:
    backend = "ffmpeg-ffv1"

    def __init__(self, executable, destination, fps, size, threads):
        self.threads = threads
        self.errors = tempfile.TemporaryFile()
        width, height = size
        command = [
            executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-pixel_format", "bgr24",
            "-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0",
            "-an", "-c:v", "ffv1", "-level", "3", "-slices", "4", "-slicecrc", "1",
            "-threads", str(threads), "-pix_fmt", "bgr0", "-f", "avi", str(destination),
        ]
        try:
            self.process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=self.errors,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except BaseException:
            self.errors.close()
            raise

    def write(self, frame):
        # BufferedWriter copies/consumes the view before returning. The bounded OS
        # pipe overlaps encoding with the next crop without an unbounded frame queue.
        try:
            self.process.stdin.write(memoryview(frame).cast("B"))
        except OSError as error:
            raise RuntimeError(tr('무손실 ROI 압축 도구가 프레임 저장 중 종료되었습니다.')) from error

    def close(self):
        try:
            self.process.stdin.close()
            code = self.process.wait(timeout=60)
            if code:
                self.errors.seek(0)
                detail = self.errors.read(4096).decode("utf-8", errors="replace")
                raise RuntimeError(tr('무손실 ROI 압축에 실패했습니다 ({v0}): {v1}', v0=code, v1=detail))
        finally:
            self.abort()

    def abort(self):
        if self.process.poll() is None:
            self.process.kill()
        # Kill before closing a possibly buffered pipe so error cleanup cannot
        # wait on an encoder that is no longer consuming input.
        try:
            self.process.stdin.close()
        except OSError:
            pass
        self.process.wait()
        self.errors.close()


class _OpenCVCropWriter:
    backend = "opencv-ffv1"
    threads = None

    def __init__(self, destination, fps, size):
        self.writer = cv2.VideoWriter(
            str(destination), cv2.VideoWriter_fourcc(*TRACKING_CROP_CODEC), fps, size
        )
        if not self.writer.isOpened():
            self.writer.release()
            raise RuntimeError(tr('무손실 추적 크롭 영상을 만들 수 없습니다: {v0}', v0=destination))

    def write(self, frame):
        self.writer.write(frame)

    def close(self):
        self.writer.release()

    abort = close


def _open_crop_writer(destination, fps, size, threads):
    executable = _ffmpeg_executable()
    if executable:
        try:
            return _FFmpegCropWriter(executable, destination, fps, size, threads)
        except OSError:
            # Older/offline environments can still use their existing lossless
            # OpenCV encoder. Runtime encoding errors must not be hidden.
            pass
    return _OpenCVCropWriter(destination, fps, size)


def _tracking_crop_geometry(
    corners: np.ndarray, frame_size: tuple[int, int]
) -> dict:
    width, height = frame_size
    polygon = np.asarray(corners, dtype=np.float32)
    if (
        polygon.ndim != 2
        or polygon.shape[1:] != (2,)
        or len(polygon) < 3
        or not np.isfinite(polygon).all()
        or abs(cv2.contourArea(polygon)) < 25
    ):
        raise ValueError(tr('추적 크롭 영역이 올바르지 않습니다.'))

    x, y, crop_w, crop_h = cv2.boundingRect(polygon.astype(np.int32))
    x, y = max(0, x), max(0, y)
    crop_w, crop_h = min(crop_w, width - x), min(crop_h, height - y)
    if crop_w < 16 or crop_h < 16:
        raise ValueError(tr('추적 크롭 영역이 너무 작습니다.'))

    pad_right = 0
    pad_bottom = 0
    if crop_w % 2:
        if x + crop_w < width:
            crop_w += 1
        elif x > 0:
            x -= 1
            crop_w += 1
        else:
            pad_right = 1
    if crop_h % 2:
        if y + crop_h < height:
            crop_h += 1
        elif y > 0:
            y -= 1
            crop_h += 1
        else:
            pad_bottom = 1

    output_w = crop_w + pad_right
    output_h = crop_h + pad_bottom
    local = np.round(polygon - np.array([x, y], dtype=np.float32)).astype(np.int32)
    mask = np.zeros((output_h, output_w), dtype=np.uint8)
    cv2.fillPoly(mask, [local], 255)
    return {
        "source_offset_px": [x, y],
        "source_crop_size_px": [crop_w, crop_h],
        "crop_size_px": [output_w, output_h],
        "padding_px": {"left": 0, "top": 0, "right": pad_right, "bottom": pad_bottom},
        "mask": mask,
    }


def write_tracking_crop_video(
    video: Path,
    corners: np.ndarray | list[list[float]],
    destination: Path,
    progress: Callable[[int, int], None] | None = None,
    max_frames: int | None = None,
    frame_indices: Iterable[int] | None = None,
    *,
    stage_progress: Callable[[str, int, int], None] | None = None,
    encoder_threads: int | None = None,
    validation_mode: str = "full",
) -> dict:
    """Write one pixel-lossless crop; retain full verification for old callers."""
    if max_frames is not None and (int(max_frames) != max_frames or max_frames < 1):
        raise ValueError(tr('max_frames는 1 이상의 정수여야 합니다.'))
    selected_indices = None
    if frame_indices is not None:
        if max_frames is not None:
            raise ValueError(tr('max_frames와 frame_indices는 동시에 사용할 수 없습니다.'))
        selected_indices = sorted({int(index) for index in frame_indices if int(index) >= 0})
    return _write_tracking_crop_videos(
        video, [{"corners": corners, "destination": destination,
                 "progress": progress, "stage_progress": stage_progress}],
        validation_mode=validation_mode, encoder_threads=encoder_threads,
        max_frames=max_frames, selected_indices=selected_indices,
        count_unknown_source=(validation_mode == "full" or max_frames is not None),
    )[0]


def write_tracking_crop_videos(
    video: Path,
    crops: Iterable[dict],
    *,
    validation_mode: str = "fast",
    encoder_threads: int | None = None,
) -> list[dict]:
    """Decode a source once and fan each frame out to bounded lossless writers.

    Each crop provides ``corners``, ``destination`` and optional ``progress`` /
    ``stage_progress`` callbacks. Results follow input order. Every pending file
    is validated before publishing any member. A failed publication rolls the
    group back; if restoration is blocked, the error names preserved backups.
    ``encoder_threads`` is the per-writer budget, not a group-wide CPU budget.
    """
    return _write_tracking_crop_videos(
        video, list(crops), validation_mode=validation_mode,
        encoder_threads=encoder_threads,
    )


def _fast_sample_indices(frames: int) -> list[int]:
    # Odd sample count guarantees first, exact middle and last; integer spacing
    # is deterministic and no output-size-dependent validation loop is needed.
    if frames < 1:
        return []
    indices = [index * (frames - 1) // (TRACKING_CROP_FAST_SAMPLE_COUNT - 1)
               for index in range(TRACKING_CROP_FAST_SAMPLE_COUNT)]
    indices[TRACKING_CROP_FAST_SAMPLE_COUNT // 2] = frames // 2
    return sorted(set(indices))


def _publish_tracking_crop_group(states: list[dict]) -> None:
    """Commit all completed crops, restoring the old group on ordinary errors.

    Backups intentionally live outside auto-cleaned encoder work directories.
    If a lock also prevents rollback, the only surviving old copy must remain
    available for manual recovery after the caller's ExitStack has unwound.
    This is rollback for caught failures, not a power-loss-safe transaction.
    """
    if len(states) == 1:
        replace_with_retry(states[0]["pending"], states[0]["destination"])
        return
    records = []

    def remove_empty_backup_directories():
        for record in records:
            directory = record["backup_directory"]
            if directory is not None:
                try:
                    directory.rmdir()
                except OSError:
                    # Nonempty means a recoverable old copy is still present.
                    pass

    try:
        # Back up the entire prior group before publishing its first member.
        for state in states:
            destination = state["destination"]
            record = {"destination": destination, "pending": state["pending"],
                      "backup": None, "backup_directory": None, "published": False}
            records.append(record)
            if destination.exists() or destination.is_symlink():
                if not destination.is_file() and not destination.is_symlink():
                    raise ValueError(tr('ROI 저장 경로가 파일이 아닙니다: {v0}', v0=destination))
                directory = Path(tempfile.mkdtemp(
                    prefix=".hagisilta-crop-recovery-", dir=destination.parent))
                record["backup_directory"] = directory
                backup = directory / destination.name
                replace_with_retry(destination, backup)
                record["backup"] = backup
        for record in records:
            replace_with_retry(record["pending"], record["destination"])
            record["published"] = True
    except BaseException as error:
        recovery_failures = []
        for record in reversed(records):
            destination = record["destination"]
            backup = record["backup"]
            try:
                if backup is not None:
                    replace_with_retry(backup, destination)
                elif record["published"]:
                    # This destination did not exist before this operation.
                    destination.unlink()
            except BaseException as recovery_error:
                if backup is not None:
                    recovery_failures.append(
                        tr('기존 영상 복구 파일: {v0} → {v1} ({v2})', v0=backup, v1=destination, v2=recovery_error))
                else:
                    recovery_failures.append(
                        tr('원래 없던 새 출력 제거 필요: {v0} ({v1})', v0=destination, v1=recovery_error))
        remove_empty_backup_directories()
        if recovery_failures:
            raise RuntimeError(
                tr('그룹 ROI 저장 실패 후 일부 출력을 원래 상태로 복구하지 못했습니다. 아래 복구 파일은 자동 삭제하지 않고 보존했습니다. ')
                + " ; ".join(recovery_failures)
                + tr(' · 원래 저장 오류: {v0}', v0=error)
            ) from error
        raise
    else:
        # All final outputs are valid and installed. Failure to remove a now
        # redundant backup must not report the successful commit as a failure.
        for record in records:
            backup = record["backup"]
            if backup is not None:
                try:
                    backup.unlink()
                except OSError as error:
                    print(tr('ROI 이전 영상 백업 정리 실패(보존됨): {v0} · {v1}', v0=backup, v1=error),
                          file=sys.stderr, flush=True)
        remove_empty_backup_directories()


def _validate_tracking_crop(state: dict, frames: int, fps: float, mode: str) -> dict:
    started = time.perf_counter()
    sample_indices = _fast_sample_indices(frames) if mode == "fast" else []
    total = len(sample_indices) if mode == "fast" else frames
    stage = "fast_validation" if mode == "fast" else "validation"
    callback = state["stage_progress"]
    if callback:
        callback(stage, 0, total)
    validation = cv2.VideoCapture(str(state["pending"]))
    decoded_frames = 0
    decoded_size_valid = True
    sample_verified = True
    decoded_hash = hashlib.sha256()
    output_w, output_h = state["geometry"]["crop_size_px"]
    try:
        opened = validation.isOpened()
        codec_value = int(validation.get(cv2.CAP_PROP_FOURCC)) if opened else 0
        decoded_codec = "".join(chr((codec_value >> (8 * index)) & 255) for index in range(4))
        decoded_fps = float(validation.get(cv2.CAP_PROP_FPS)) if opened else 0.0
        reported_frames = int(validation.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
        decoded_size = (int(validation.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(validation.get(cv2.CAP_PROP_FRAME_HEIGHT))) if opened else (0, 0)
        metadata_valid = (opened and decoded_codec.upper() == TRACKING_CROP_CODEC
                          and np.isfinite(decoded_fps)
                          and abs(decoded_fps - fps) <= max(0.001, fps * 0.0001))
        if mode == "fast":
            # Header count is compared with the actual write count, never with
            # the less reliable source count. A last-frame seek detects an AVI
            # that advertises more frames than it can actually decode.
            metadata_valid = (metadata_valid and reported_frames == frames
                              and decoded_size == (output_w, output_h))
            if metadata_valid:
                hashes = state["frame_hashes"]
                for index in sample_indices:
                    if index and not validation.set(cv2.CAP_PROP_POS_FRAMES, index):
                        sample_verified = False
                        break
                    ok, decoded_frame = validation.read()
                    if not ok:
                        sample_verified = False
                        break
                    decoded_frames += 1
                    if decoded_frame.shape != (output_h, output_w, 3):
                        decoded_size_valid = False
                    if abs(validation.get(cv2.CAP_PROP_POS_FRAMES) - (index + 1)) > 0.5:
                        sample_verified = False
                    hashes.seek(index * hashlib.sha256().digest_size)
                    if hashlib.sha256(memoryview(decoded_frame)).digest() != hashes.read(32):
                        sample_verified = False
                    if callback:
                        callback(stage, decoded_frames, total)
            valid = (metadata_valid and decoded_frames == total
                     and decoded_size_valid and sample_verified)
            pixel_verified = False  # Sampling must never claim full-frame verification.
        else:
            while opened:
                ok, decoded_frame = validation.read()
                if not ok:
                    break
                if decoded_frame.shape != (output_h, output_w, 3):
                    decoded_size_valid = False
                decoded_hash.update(memoryview(decoded_frame))
                decoded_frames += 1
                if callback and (decoded_frames % 250 == 0 or decoded_frames == frames):
                    callback(stage, decoded_frames, frames)
            pixel_verified = decoded_hash.digest() == state["written_hash"].digest()
            valid = (metadata_valid and decoded_frames == frames
                     and decoded_size_valid and pixel_verified)
            sample_verified = pixel_verified
        if not valid:
            raise RuntimeError(
                tr('무손실 추적 크롭 영상 검증에 실패했습니다. 코덱 {v0} · 검증 프레임 {v1}/{v2} · 저장 프레임 {v3}/{v4} · FPS {v5}/{v6} · 픽셀 일치 {v7}', v0=decoded_codec or '-', v1=decoded_frames, v2=total, v3=reported_frames, v4=frames, v5=decoded_fps, v6=fps, v7=sample_verified if mode == 'fast' else pixel_verified)
            )
    finally:
        validation.release()
    return {
        "validation_mode": mode,
        "validated_frames": decoded_frames,
        "sample_indices": sample_indices,
        "sample_verified": sample_verified,
        "pixel_verified": pixel_verified,
        "validation_seconds": time.perf_counter() - started,
    }


def _write_tracking_crop_videos(
    video, crops, *, validation_mode, encoder_threads, max_frames=None,
    selected_indices=None, count_unknown_source=False,
):
    if validation_mode not in {"fast", "full"}:
        raise ValueError(tr('validation_mode는 fast 또는 full이어야 합니다.'))
    video = Path(video)
    resolved_video = video.resolve()
    destinations = [Path(item["destination"]) for item in crops]
    resolved_destinations = [path.resolve() for path in destinations]
    if resolved_video in resolved_destinations:
        raise ValueError(tr('원본 영상과 ROI 저장 경로는 달라야 합니다.'))
    if len(set(resolved_destinations)) != len(resolved_destinations):
        raise ValueError(tr('ROI 저장 경로는 서로 달라야 합니다.'))
    if not crops:
        return []
    threads = max(1, min(4, int(encoder_threads or os.cpu_count() or 1)))
    started = time.perf_counter()
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise RuntimeError(tr('원본 영상을 열 수 없습니다: {v0}', v0=video))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 10.0
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(tr('원본 영상의 FPS가 올바르지 않습니다.'))
        source_frames_reported = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        source_frames = source_frames_reported
        if source_frames == 0 and selected_indices is None and count_unknown_source:
            counter = cv2.VideoCapture(str(video))
            try:
                while counter.isOpened():
                    ok, _frame = counter.read()
                    if not ok:
                        break
                    source_frames += 1
            finally:
                counter.release()
        total_frames = (len(selected_indices) if selected_indices is not None else
                        min(source_frames, max_frames) if max_frames else source_frames)
        # ExitStack releases every writer and temporary directory even when a
        # later writer fails to start. Pixel buffers are one frame per ROI and
        # encoder pipes provide backpressure without a growing frame queue.
        with ExitStack() as resources:
            states = []
            for item, destination in zip(crops, destinations):
                geometry = _tracking_crop_geometry(
                    np.asarray(item["corners"], dtype=np.float32), (width, height))
                crop_w, crop_h = geometry["source_crop_size_px"]
                output_w, output_h = geometry["crop_size_px"]
                mask = geometry.pop("mask")
                destination.parent.mkdir(parents=True, exist_ok=True)
                free_bytes = shutil.disk_usage(destination.parent).free
                if free_bytes < TRACKING_CROP_DISK_RESERVE_BYTES:
                    raise RuntimeError(
                        tr('무손실 추적 크롭 영상을 만들 최소 디스크 공간이 부족합니다. 필요한 최소 여유 공간 {v0:.0f}MB · 현재 {v1:.0f}MB', v0=TRACKING_CROP_DISK_RESERVE_BYTES / 1024 ** 2, v1=free_bytes / 1024 ** 2)
                    )
                work = resources.enter_context(tempfile.TemporaryDirectory(
                    prefix=".hagisilta-crop-", dir=destination.parent))
                pending = Path(work) / "roi.avi"
                writer = _open_crop_writer(pending, fps, (output_w, output_h), threads)
                resources.callback(writer.abort)
                # Just 32 bytes per frame, spilling after 64 KiB. This lets
                # unknown/misreported source lengths still use exact first /
                # middle / last samples without storing video or re-decoding it.
                frame_hashes = (resources.enter_context(tempfile.SpooledTemporaryFile(
                    max_size=64 * 1024, dir=work)) if validation_mode == "fast" else None)
                states.append({
                    "destination": destination, "pending": pending, "writer": writer,
                    "geometry": geometry, "output": None,
                    "copy_mask": np.ascontiguousarray(mask[:crop_h, :crop_w]),
                    "written_hash": hashlib.sha256(), "frame_hashes": frame_hashes,
                    "progress": item.get("progress"), "stage_progress": item.get("stage_progress"),
                })
            frames = 0
            for state in states:
                if state["stage_progress"]:
                    state["stage_progress"]("encoding", 0, total_frames)
            while True:
                if selected_indices is not None:
                    if frames >= len(selected_indices):
                        break
                    capture.set(cv2.CAP_PROP_POS_FRAMES, selected_indices[frames])
                ok, frame = capture.read()
                if not ok:
                    break
                for state in states:
                    geometry = state["geometry"]
                    crop_w, crop_h = geometry["source_crop_size_px"]
                    output_w, output_h = geometry["crop_size_px"]
                    x, y = geometry["source_offset_px"]
                    source_crop = frame[y:y + crop_h, x:x + crop_w]
                    if state["output"] is None:
                        background = tuple(int(value) for value in np.median(source_crop.reshape(-1, 3), axis=0))
                        state["output"] = np.full((output_h, output_w, 3), background, dtype=np.uint8)
                    output = state["output"]
                    cv2.copyTo(source_crop, state["copy_mask"], output[:crop_h, :crop_w])
                    state["written_hash"].update(memoryview(output))
                    if state["frame_hashes"] is not None:
                        state["frame_hashes"].write(hashlib.sha256(memoryview(output)).digest())
                    state["writer"].write(output)
                frames += 1
                if frames % TRACKING_CROP_DISK_CHECK_INTERVAL == 0:
                    for parent in {destination.parent for destination in destinations}:
                        if shutil.disk_usage(parent).free < TRACKING_CROP_DISK_RESERVE_BYTES:
                            raise RuntimeError(
                                tr('무손실 추적 크롭 영상 생성 중 디스크 공간이 부족해 중단했습니다. {v0}개 프레임까지 생성했으며 미완성 출력 파일은 자동으로 삭제됩니다.', v0=frames)
                            )
                if frames == 1 or frames % 250 == 0 or frames == total_frames:
                    for state in states:
                        progress_total = max(frames, total_frames)
                        if state["progress"]:
                            state["progress"](frames, progress_total)
                        if state["stage_progress"]:
                            state["stage_progress"]("encoding", frames, progress_total)
                if selected_indices is None and max_frames and frames >= max_frames:
                    break
            for state in states:
                state["writer"].close()
            capture.release()
            encoding_seconds = time.perf_counter() - started
            if frames == 0:
                raise RuntimeError(tr('추적 크롭 영상에 저장된 프레임이 없습니다: {v0}', v0=video))
            frame_count_tolerance = (0 if selected_indices is not None else max(
                SOURCE_FRAME_SHORTAGE_MIN_TOLERANCE, int(total_frames * SOURCE_FRAME_SHORTAGE_RATIO)))
            if max(0, total_frames - frames) > frame_count_tolerance:
                raise RuntimeError(
                    tr('추적 크롭 영상의 프레임이 누락되었습니다: 예상 {v0}개 · 저장 {v1}개 · 허용 오차 {v2}개', v0=total_frames, v1=frames, v2=frame_count_tolerance)
                )
            if source_frames == 0 and selected_indices is None:
                source_frames = frames
            results = []
            for state in states:
                verification = _validate_tracking_crop(state, frames, fps, validation_mode)
                writer = state["writer"]
                results.append({
                    **state["geometry"],
                    "frames": frames,
                    "source_frames": source_frames,
                    "source_frames_reported": source_frames_reported,
                    "source_frame_count_difference": (
                        source_frames - frames
                        if selected_indices is None and max_frames is None else None),
                    "source_frame_count_tolerance": frame_count_tolerance,
                    "fps": fps,
                    "codec": TRACKING_CROP_CODEC,
                    "container": TRACKING_CROP_CONTAINER,
                    "lossless": True,
                    "mask_mode": "polygon_outside_median_background",
                    "encoder_backend": writer.backend,
                    "encoder_threads": writer.threads,
                    "sha256": state["written_hash"].hexdigest(),
                    "encoding_seconds": encoding_seconds,
                    **verification,
                })
            _publish_tracking_crop_group(states)
            return results
    finally:
        capture.release()
