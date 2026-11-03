# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from hagisilta_i18n import init_i18n, tr

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from PySide6.QtCore import QPointF
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMessageBox

from hagisilta_video_review import APP_ICON, VideoReviewPilot
from hagisilta_window_chrome import install_wheel_value_guard


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class LabelFolder:
    name: str
    path: Path
    images: list[Path]
    frame: pd.DataFrame
    label_path: Path
    bodyparts: list[str]
    point_columns: dict[str, dict[str, object]]
    row_by_image: dict[str, int]
    dirty: bool = False

    def points(self, image_name: str) -> dict[str, QPointF]:
        row = self.frame.iloc[self.row_by_image[image_name]]
        result: dict[str, QPointF] = {}
        for part in self.bodyparts:
            columns = self.point_columns.get(part, {})
            if "x" not in columns or "y" not in columns:
                continue
            x_value, y_value = row[columns["x"]], row[columns["y"]]
            if pd.notna(x_value) and pd.notna(y_value):
                result[part] = QPointF(float(x_value), float(y_value))
        return result

    def set_points(self, image_name: str, points: dict[str, QPointF]) -> None:
        row_position = self.row_by_image[image_name]
        for part in self.bodyparts:
            columns = self.point_columns.get(part, {})
            if "x" not in columns or "y" not in columns:
                continue
            point = points.get(part)
            self.frame.iloc[row_position, self.frame.columns.get_loc(columns["x"])] = (
                point.x() if point is not None else np.nan
            )
            self.frame.iloc[row_position, self.frame.columns.get_loc(columns["y"])] = (
                point.y() if point is not None else np.nan
            )
        self.dirty = True


@dataclass
class ReviewAsset:
    labels: LabelFolder
    video: Path
    pose: Path
    output: Path
    image_by_frame: dict[int, str]


def _column_layout(frame: pd.DataFrame) -> tuple[list[str], dict[str, dict[str, object]]]:
    if not isinstance(frame.columns, pd.MultiIndex) or frame.columns.nlevels < 3:
        raise ValueError(tr('DLC 다중 열 라벨 형식이 아닙니다.'))
    names = list(frame.columns.names)
    bodypart_level = names.index("bodyparts") if "bodyparts" in names else frame.columns.nlevels - 2
    coordinate_level = names.index("coords") if "coords" in names else frame.columns.nlevels - 1
    bodyparts = list(dict.fromkeys(str(column[bodypart_level]) for column in frame.columns))
    columns: dict[str, dict[str, object]] = {part: {} for part in bodyparts}
    for column in frame.columns:
        coordinate = str(column[coordinate_level]).lower()
        if coordinate in {"x", "y"}:
            columns[str(column[bodypart_level])][coordinate] = column
    return bodyparts, columns


def load_label_folder(config: Path, folder_name: str, configured_bodyparts: list[str], scorer: str) -> LabelFolder:
    folder = config.parent / "labeled-data" / folder_name
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    images = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(tr('라벨 이미지가 없습니다: {p0}', p0=folder_name))
    label_files = sorted(folder.glob("CollectedData_*.h5"), key=lambda path: path.stat().st_mtime, reverse=True)
    if label_files:
        label_path = label_files[0]
        frame = pd.read_hdf(label_path)
    else:
        csv_files = sorted(folder.glob("CollectedData_*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
        if csv_files:
            frame = pd.read_csv(csv_files[0], header=[0, 1, 2], index_col=0)
            label_path = csv_files[0].with_suffix(".h5")
        else:
            if not configured_bodyparts:
                raise ValueError(tr('config.yaml에 신체점이 없습니다.'))
            columns = pd.MultiIndex.from_product(
                [[scorer], configured_bodyparts, ["x", "y"]],
                names=["scorer", "bodyparts", "coords"],
            )
            frame = pd.DataFrame(columns=columns, dtype=float)
            label_path = folder / f"CollectedData_{scorer}.h5"
    bodyparts, point_columns = _column_layout(frame)
    if configured_bodyparts and bodyparts != configured_bodyparts:
        raise ValueError(
            tr('{p0}의 라벨 신체점이 프로젝트 설정과 다릅니다.\n프로젝트: {p1}\n라벨: {p2}', p0=folder_name, p1=', '.join(configured_bodyparts), p2=', '.join(bodyparts))
        )
    existing_by_name: dict[str, int] = {}
    for position, index_value in enumerate(frame.index):
        raw = index_value[-1] if isinstance(index_value, tuple) else index_value
        existing_by_name[Path(str(raw).replace("\\", "/")).name] = position
    for image in images:
        if image.name not in existing_by_name:
            frame.loc[f"labeled-data/{folder_name}/{image.name}"] = np.nan
    image_names = {image.name for image in images}
    row_by_image: dict[str, int] = {}
    for position, index_value in enumerate(frame.index):
        raw = index_value[-1] if isinstance(index_value, tuple) else index_value
        name = Path(str(raw).replace("\\", "/")).name
        if name in image_names:
            row_by_image[name] = position
    return LabelFolder(folder_name, folder, images, frame, label_path, bodyparts, point_columns, row_by_image)


def save_label_folder(config: Path, data: LabelFolder) -> None:
    if not data.dirty and data.label_path.is_file():
        return
    backup_root = config.parent / ".hagisilta-backups" / "labels" / time.strftime("%Y%m%d-%H%M%S") / data.name
    for source in (data.label_path, data.label_path.with_suffix(".csv")):
        if source.is_file():
            backup_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, backup_root / source.name)
    csv_path = data.label_path.with_suffix(".csv")
    temporary_h5 = data.label_path.with_name(f".{data.label_path.name}.hagisilta-tmp")
    temporary_csv = csv_path.with_name(f".{csv_path.name}.hagisilta-tmp")
    try:
        data.frame.to_hdf(temporary_h5, key="df_with_missing", mode="w")
        data.frame.to_csv(temporary_csv)
        os.replace(temporary_h5, data.label_path)
        os.replace(temporary_csv, csv_path)
        data.dirty = False
    finally:
        temporary_h5.unlink(missing_ok=True)
        temporary_csv.unlink(missing_ok=True)


def read_label_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(tr('라벨 이미지를 읽을 수 없습니다: {p0}', p0=path))
    return image


def image_frame_index(path: Path) -> int:
    match = re.fullmatch(r"img(\d+)", path.stem, re.IGNORECASE)
    if not match:
        raise ValueError(tr('원본 프레임 번호를 알 수 없는 라벨 이미지입니다: {p0}', p0=path.name))
    return int(match.group(1))


def source_video_for_folder(config: Path, labels: LabelFolder) -> Path:
    payload = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    candidates = [
        Path(str(value))
        for value in (payload.get("video_sets", {}) or {})
        if Path(str(value)).stem.casefold() == labels.name.casefold() and Path(str(value)).is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            tr('{p0}의 원본/ROI 영상을 찾지 못했습니다.\n프로젝트 영상 경로를 복구한 뒤 다시 라벨링을 여세요.', p0=labels.name)
        )
    image = read_label_image(labels.images[0])
    expected_height, expected_width = image.shape[:2]
    maximum_frame = max(image_frame_index(path) for path in labels.images)
    for candidate in candidates:
        capture = cv2.VideoCapture(str(candidate))
        if not capture.isOpened():
            capture.release()
            continue
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if (width, height) == (expected_width, expected_height) and frame_count > maximum_frame:
            return candidate.resolve()
    raise ValueError(
        tr('{p0}의 라벨 이미지와 크기·프레임 번호가 맞는 원본/ROI 영상이 없습니다.', p0=labels.name)
    )


def create_review_asset(config: Path, labels: LabelFolder, destination: Path) -> ReviewAsset:
    destination.mkdir(parents=True, exist_ok=True)
    video = source_video_for_folder(config, labels)
    capture = cv2.VideoCapture(str(video))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
    capture.release()
    image_by_frame = {image_frame_index(path): path.name for path in labels.images}
    if not frame_count or max(image_by_frame) >= frame_count:
        raise ValueError(tr('추출 프레임 번호가 원본/ROI 영상 범위를 벗어납니다.'))

    columns = pd.MultiIndex.from_tuples(
        [
            ("HAGISILTA", part, coordinate)
            for part in labels.bodyparts
            for coordinate in ("x", "y", "likelihood")
        ],
        names=["scorer", "bodyparts", "coords"],
    )
    pose_frame = pd.DataFrame(np.nan, index=range(frame_count), columns=columns, dtype=float)
    for frame_index, image_name in image_by_frame.items():
        points = labels.points(image_name)
        values: list[float] = []
        for part in labels.bodyparts:
            point = points.get(part)
            if point is None:
                values.extend([math.nan, math.nan, math.nan])
            else:
                values.extend([point.x(), point.y(), 1.0])
        pose_frame.iloc[frame_index] = values
    pose = destination / "label-frames.h5"
    pose_frame.to_hdf(pose, key="df_with_missing", mode="w")
    output = destination / "session"
    output.mkdir(parents=True, exist_ok=True)
    (destination / "frame-map.json").write_text(
        json.dumps({"folder": labels.name, "frames": image_by_frame}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ReviewAsset(labels, video, pose, output, image_by_frame)


def import_review_session(config: Path, asset: ReviewAsset) -> int:
    session_path = asset.output / "review-session.json"
    if not session_path.is_file():
        return 0
    payload = json.loads(session_path.read_text(encoding="utf-8"))
    edits = payload.get("edits", {}) or {}
    changed_frames = 0
    for raw_index, bodypart_edits in edits.items():
        frame_index = int(raw_index)
        if frame_index not in asset.image_by_frame or not isinstance(bodypart_edits, dict):
            continue
        image_name = asset.image_by_frame[frame_index]
        points = asset.labels.points(image_name)
        for part, value in bodypart_edits.items():
            if part not in asset.labels.bodyparts or not isinstance(value, dict):
                continue
            x_value, y_value = value.get("x"), value.get("y")
            visible = bool(value.get("visible", True))
            if visible and pd.notna(x_value) and pd.notna(y_value):
                points[part] = QPointF(float(x_value), float(y_value))
            else:
                points.pop(part, None)
        asset.labels.set_points(image_name, points)
        changed_frames += 1
    if asset.labels.dirty:
        save_label_folder(config, asset.labels)
    return changed_frames


def target_folders(config: Path, requested: list[str]) -> list[str]:
    root = config.parent / "labeled-data"
    if requested:
        return list(dict.fromkeys(requested))
    if not root.is_dir():
        return []
    return [
        path.name
        for path in sorted(root.iterdir())
        if path.is_dir() and not path.name.endswith(("_labeled", ".importing"))
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Open extracted label frames in the HAGISILTA video correction window.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--folder", action="append", default=[])
    args = parser.parse_args()
    config = args.config.resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    payload = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    configured_bodyparts = [str(value) for value in payload.get("bodyparts", [])]
    scorer = str(payload.get("scorer") or "HAGISILTA")
    folders = target_folders(config, list(args.folder))
    app = QApplication(sys.argv)
    init_i18n(app)
    app.setApplicationName("HAGISILTA")
    if APP_ICON.is_file():
        app.setWindowIcon(QIcon(str(APP_ICON)))
    install_wheel_value_guard(app)

    runtime_parent = config.parent / ".hagisilta" / "runtime"
    runtime_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="initial-label-review-", dir=runtime_parent) as temporary:
        runtime = Path(temporary)
        try:
            assets = [
                create_review_asset(
                    config,
                    load_label_folder(config, folder, configured_bodyparts, scorer),
                    runtime / f"{index + 1:03d}",
                )
                for index, folder in enumerate(folders)
            ]
            if not assets:
                raise ValueError(tr('열 수 있는 라벨 프레임이 없습니다.'))
            playlist = [
                {
                    "video": str(asset.video),
                    "pose": str(asset.pose),
                    "output": str(asset.output),
                    "label": tr('{p0} · 라벨 프레임 {p1}개', p0=asset.labels.name, p1=len(asset.image_by_frame)),
                    "show_bboxes": False,
                    "label_frames": sorted(asset.image_by_frame),
                }
                for asset in assets
            ]
            first = assets[0]
            window = VideoReviewPilot(
                first.video,
                first.pose,
                first.output,
                scorer,
                playlist=playlist,
                playlist_index=0,
            )
            window.setWindowTitle(tr('HAGISILTA · 초기 프레임 라벨링'))
            window.video_meta.setText(tr('원본/ROI 영상 · {p0}', p0=first.video.name))
            window.pose_meta.setText(tr('라벨 프레임 {p0}개', p0=len(first.image_by_frame)))
            window.footer_status.setText(
                tr('원본 영상을 재생해 문맥을 확인하고 라벨 프레임에서 점을 수정합니다. 저장한 좌표는 학습 라벨에 반영됩니다.')
            )
            window.show()
            exit_code = app.exec()
            changed = sum(import_review_session(config, asset) for asset in assets)
            if changed:
                print(f"LABEL_FRAMES_UPDATED={changed}", flush=True)
            return exit_code
        except Exception as error:
            QMessageBox.critical(None, tr('초기 프레임 라벨링 실패'), str(error))
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
