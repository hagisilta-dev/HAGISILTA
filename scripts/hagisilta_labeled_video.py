# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
"""Lossless AVI output for DLC's existing point/skeleton/bbox renderer.

Adapters are scoped to a worker call, never written into the installed DLC.
DLC still chooses the scorer, snapshot, bodyparts, skeleton and colors.
Only its encoder/output path changes. Every decoded output pixel is verified.
"""

from hagisilta_i18n import tr
from contextlib import ExitStack, contextmanager
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from threading import RLock
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

CODEC = "FFV1"
_LOCK = RLock()


def source_video_candidates(directory):
    """Never feed an annotated export back into ROI setup or pose analysis."""
    for pattern in ("*_roi.avi", "*.avi", "*.mp4"):
        paths = sorted(path for path in Path(directory).glob(pattern)
                       if path.is_file() and "labeled" not in path.stem.lower())
        if paths:
            return paths
    return []


def _stamp(path):
    info = Path(path).stat()
    return {"size": info.st_size, "mtime_ns": info.st_mtime_ns}


def _codec(capture):
    value = int(capture.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((value >> (8 * index)) & 255) for index in range(4)).upper()


def is_lossless_labeled_video(path):
    path = Path(path)
    if path.suffix.lower() != ".avi" or "labeled" not in path.stem.lower():
        return False
    capture = cv2.VideoCapture(str(path))
    try:
        return capture.isOpened() and _codec(capture) == CODEC and capture.read()[0]
    finally:
        capture.release()


class _LosslessClip:
    """DLC VideoProcessor interface with checked writes and no YUV subsampling."""
    def __init__(self, fname, sname, *, rows, sw="", sh="", fps=None, codec=CODEC):
        self.capture = cv2.VideoCapture(str(fname))
        self.writer = None
        self.written = 0
        self.digest = hashlib.sha256()
        self.extra_source_frame = False
        self.nframes = rows  # Do not trust a container's approximate frame count.
        try:
            if not self.capture.isOpened() or rows <= 0:
                raise ValueError(tr('확인 영상의 원본 또는 좌표 프레임을 읽을 수 없습니다.'))
            self.w, self.h = (int(self.capture.get(prop)) for prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
            self.sw, self.sh = int(sw or self.w), int(sh or self.h)
            self.rate = float(fps or self.capture.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(self.rate) or self.rate <= 0 or min(self.sw, self.sh) <= 0:
                raise ValueError(tr('확인 영상의 FPS·해상도가 올바르지 않습니다.'))
            # OpenCV silently truncates odd sizes; pad at most one pixel instead.
            self.size = (self.sw + self.sw % 2, self.sh + self.sh % 2)
            self.writer = cv2.VideoWriter(str(sname), cv2.VideoWriter_fourcc(*CODEC), self.rate, self.size)
            if not self.writer.isOpened():
                raise RuntimeError(tr('FFV1 무손실 AVI 저장기를 열지 못했습니다.'))
        except Exception:
            self.close()
            raise

    def width(self):
        return self.w

    def height(self):
        return self.h

    def fps(self):
        return self.rate

    def load_frame(self):
        ok, bgr = self.capture.read()
        if not ok:
            raise RuntimeError(tr('확인 영상 원본 프레임 누락: {v0}/{v1}', v0=self.written, v1=self.nframes))
        return np.flip(bgr, 2)  # DLC draws in RGB.

    def save_frame(self, frame):
        if frame is None or frame.dtype != np.uint8 or frame.shape != (self.sh, self.sw, 3):
            raise ValueError(tr('확인 영상 프레임의 크기·색상 형식이 변경되었습니다.'))
        bgr = np.ascontiguousarray(np.flip(frame, 2))
        if self.size != (self.sw, self.sh):
            bgr = cv2.copyMakeBorder(bgr, 0, self.size[1] - self.sh, 0, self.size[0] - self.sw, cv2.BORDER_CONSTANT)
        self.digest.update(bgr.tobytes())
        self.writer.write(bgr)
        self.written += 1
        if self.written == self.nframes:
            self.extra_source_frame = self.capture.read()[0]

    def close(self):
        if self.writer is not None:
            self.writer.release()
        self.capture.release()


def _verify(path, clip):
    capture = cv2.VideoCapture(str(path))
    count, digest = 0, hashlib.sha256()
    try:
        if not capture.isOpened() or _codec(capture) != CODEC:
            raise RuntimeError(tr('확인 영상이 FFV1 코덱으로 저장되지 않았습니다.'))
        rate = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isclose(rate, clip.rate, rel_tol=1e-5, abs_tol=1e-5):
            raise RuntimeError(tr('확인 영상 FPS가 변경되었습니다.'))
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape != (clip.size[1], clip.size[0], 3):
                raise RuntimeError(tr('확인 영상 해상도가 변경되었습니다.'))
            digest.update(frame.tobytes())
            count += 1
        if count != clip.nframes or count != clip.written or clip.extra_source_frame:
            raise RuntimeError(tr('확인 영상·좌표 프레임 불일치: 좌표 {v0} · 저장 {v1}', v0=clip.nframes, v1=count))
        if digest.digest() != clip.digest.digest():
            raise RuntimeError(tr('확인 영상의 무손실 픽셀 검증에 실패했습니다.'))
    finally:
        capture.release()
    return {"codec": CODEC, "lossless": True, "frames": count, "fps": rate,
            "size_px": list(clip.size), "padding_right_bottom_px": [clip.size[0] - clip.sw, clip.size[1] - clip.sh],
            "decoded_pixels_sha256": digest.hexdigest(), "pixel_verified": True}


def render_lossless_video(renderer, *args, **kwargs):
    """Run DLC's renderer with explicit AVI output and atomically publish it."""
    from deeplabcut.utils import make_labeled_video as module

    bound = inspect.signature(renderer).bind(*args, **kwargs)
    values = bound.arguments
    video, pose = Path(values["video"]).resolve(), Path(values["h5file"]).resolve()
    requested = Path(values.get("output_path") or pose.with_name(pose.stem + "_labeled.avi")).resolve()
    output = requested.with_suffix(".avi")
    if output in (video, pose):
        raise ValueError(tr('확인 영상을 원본 영상·좌표에 덮어쓸 수 없습니다.'))
    before = [_stamp(video), _stamp(pose)]
    rows = len(pd.read_hdf(pose))
    output.parent.mkdir(parents=True, exist_ok=True)
    clips = []

    def factory(*clip_args, **clip_kwargs):
        clip = _LosslessClip(*clip_args, rows=rows, **clip_kwargs)
        clips.append(clip)
        return clip

    with tempfile.TemporaryDirectory(prefix=".hagisilta-labeled-", dir=output.parent) as temporary:
        pending = Path(temporary) / "pending.avi"
        values["output_path"], values["codec"] = str(pending), CODEC
        try:
            with patch.object(module, "vp", factory):
                renderer(*bound.args, **bound.kwargs)
        finally:
            for clip in clips:
                clip.close()
        if len(clips) != 1:
            raise RuntimeError(tr('DLC 확인 영상 저장 경로가 변경되었습니다.'))
        report = _verify(pending, clips[0])
        if before != [_stamp(video), _stamp(pose)]:
            raise RuntimeError(tr('확인 영상 생성 중 원본 영상·좌표가 변경되었습니다.'))
        report.update(source_video=str(video), source_pose=str(pose), source_stamps=before, output=str(output))
        metadata = Path(temporary) / "metadata.json"
        metadata.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        pending.replace(output)
        metadata.replace(output.with_suffix(".lossless.json"))
    print(f"LABELED_VIDEO={output}:codec=FFV1:frames={rows}:pixel_verified=true", flush=True)
    return output


@contextmanager
def lossless_labeled_video_output(progress_callback=None):
    """Cover trained, queued, and SuperAnimal output; always restore DLC hooks."""
    from deeplabcut.utils import make_labeled_video as module

    with _LOCK, ExitStack() as stack:
        native = module.create_video
        outputs, names = [], {}

        def render(*args, **kwargs):
            values = inspect.signature(native).bind(*args, **kwargs).arguments
            if progress_callback:
                progress_callback("labeling", values["video"], values["h5file"])
            try:
                result = render_lossless_video(native, *args, **kwargs)
            except BaseException:
                if progress_callback:
                    progress_callback("error", values["video"], values["h5file"])
                raise
            if kwargs.get("output_path"):
                names[str(kwargs["output_path"])] = str(result)
            outputs.append(result)
            if progress_callback:
                progress_callback("done", values["video"], values["h5file"], result)

        # DLC logs an MP4 path after rendering; keep its message truthful too.
        class OutputNames:
            def __init__(self, stream):
                self.stream = stream

            def write(self, text):
                for old, new in names.items():
                    text = text.replace(old, new)
                return self.stream.write(text)

            def __getattr__(self, name):
                return getattr(self.stream, name)

        stack.enter_context(patch.object(sys, "stdout", OutputNames(sys.stdout)))
        stack.enter_context(patch.object(module, "create_video", render))
        stack.enter_context(patch.object(module, "_create_labeled_video", render))
        # Avoid fork workers escaping scoped verification/result collection.
        stack.enter_context(patch.object(module, "get_start_method", return_value="spawn"))
        superanimal = importlib.import_module("deeplabcut.pose_estimation_pytorch.modelzoo.inference")
        stack.enter_context(patch.object(superanimal, "create_video", render))
        working_directory = Path.cwd()
        try:
            yield outputs
        finally:
            os.chdir(working_directory)


def create_lossless_labeled_videos(*args, progress_callback=None, **kwargs):
    import deeplabcut

    kwargs.update(codec=CODEC, overwrite=True, max_workers=1, fastmode=True)
    with lossless_labeled_video_output(progress_callback) as outputs:
        result = deeplabcut.create_labeled_video(*args, **kwargs)
        if not outputs or not result or any(value is not True for value in result):
            raise RuntimeError(tr('무손실 확인 영상 생성이 완료되지 않았습니다. 추론 결과 경로를 확인하세요.'))
    return outputs


def infer_superanimal_with_lossless_video(*args, progress_callback=None, **kwargs):
    import deeplabcut

    if not kwargs.get("create_labeled_video", True):
        return deeplabcut.video_inference_superanimal(*args, **kwargs)
    with lossless_labeled_video_output(progress_callback) as outputs:
        result = deeplabcut.video_inference_superanimal(*args, **kwargs)
        if not outputs:
            raise RuntimeError(tr('SuperAnimal 무손실 확인 영상이 생성되지 않았습니다.'))
    return result
