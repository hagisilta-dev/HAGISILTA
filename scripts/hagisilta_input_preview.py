# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
"""Read-only preview of the training worker's real affine/padding pipeline."""

from hagisilta_i18n import tr
import copy
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtGui import QImage, QColor, QPen, QPolygonF, QBrush
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QFrame, QHBoxLayout,
    QLabel, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from hagisilta_training_input import CROP_POLICY, build_full_roi_transform
from hagisilta_video_review import VideoCanvas
from hagisilta_window_chrome import (
    HeaderContext,
    apply_hagisilta_window_identity, enable_full_window_controls,
    create_hagisilta_header, create_hagisilta_help_button,
    hagisilta_control_stylesheet, install_wheel_value_guard,
)


def preview_geometry(path: Path, layout: dict, angle: float, scale: float, labels=None) -> dict:
    with Image.open(path) as source:
        image = np.array(source.convert("RGB"))
    height, width = image.shape[:2]
    transform = build_full_roi_transform(layout, photometric=False)
    affine = transform.transforms[0]
    affine.p = 1.0
    affine.rotate = (angle, angle)
    affine.scale = {"x": (scale, scale), "y": (scale, scale)}
    if layout.get("policy") == CROP_POLICY:
        import albumentations as A
        if labels is None:
            raise ValueError(tr('라벨 주변 크롭 미리보기에는 해당 Train 이미지의 라벨이 필요합니다.'))
        # The same crop transform; choose a reproducible feasible center for preview.
        transform.transforms[-1].randomize = False
        replay = A.ReplayCompose(transform.transforms,
            keypoint_params=A.KeypointParams("xy", remove_invisible=False, label_fields=["class_labels"]),
            bbox_params=A.BboxParams(format="coco", label_fields=["bbox_labels"]))
        arguments = dict(image=image, keypoints=labels, class_labels=list(range(len(labels))), bboxes=[], bbox_labels=[])
        result = replay(**arguments)
        before_crop = copy.deepcopy(result["replay"])
        crop_record = before_crop["transforms"][-1]
        rectangle = crop_record["params"]
        crop_record["applied"] = False
        whole = A.ReplayCompose.replay(before_crop, **arguments)
        result.update(source_image=whole["image"], crop_box=tuple(rectangle[k] for k in ("x_min", "y_min", "x_max", "y_max")))
    else:
        result = transform(image=image, keypoints=[(0, 0), (width - 1, 0),
                           (width - 1, height - 1), (0, height - 1)],
                           class_labels=[0, 1, 2, 3], bboxes=[], bbox_labels=[])
    if result["image"].shape[:2] != (layout["height"], layout["width"]):
        raise ValueError(tr('현재 이미지·증강이 지정한 입력 크기를 벗어납니다. 크기를 다시 계산하세요.'))
    return {**result, "native_size": (width, height)}


class TrainingInputPreview(QDialog):
    def __init__(self, paths, layout, parent=None, *, labels=None):
        super().__init__(parent)
        self.paths, self.input_layout = list(paths), dict(layout)
        self.labels = labels
        self.is_crop = layout.get("policy") == CROP_POLICY
        self.setWindowTitle(tr('HAGISILTA · 학습 입력 미리보기'))
        enable_full_window_controls(self)
        apply_hagisilta_window_identity(self)
        self.resize(1280 if self.is_crop else 1040, 820)
        self.setMinimumSize(740, 580)
        self.setStyleSheet("QDialog {background:#23242a; color:#e4e5e8;} QLabel {color:#e4e5e8;}"
                          + hagisilta_control_stylesheet())
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        header, header_row = create_hagisilta_header(tr('학습 입력 미리보기'), "HAGISILTA · TRAINING INPUT")
        header_row.addWidget(create_hagisilta_help_button(self, tr('학습 입력 도움말'),
            tr('실제 Train 이미지에 학습과 같은 회전·배율·중앙 패딩을 적용합니다.\n라벨 주변 크롭은 모든 표시된 신체점과 여유 공간을 보존하며 크기 맞춤 축소를 하지 않습니다.\n왼쪽 파란 사각형은 증강 후 전체에서 자를 위치, 오른쪽은 실제 학습 입력과 라벨입니다. 학습에서는 이 조건을 만족하는 범위 안에서 크롭 위치가 달라집니다.\n검은 여백은 실제 패딩이며 표시선과 점만 미리보기용입니다. 라벨이 없는 신체 부위는 직접 확인하세요.\n밝기·노이즈·블러는 생략합니다. 기존 축소 모델의 기록을 열면 그 이전 변환을 재현합니다.\n입력 크기는 Train에만 적용하며 Validation·Test·추론은 기존 원본 크기 방식을 유지합니다.\n휠 확대·축소 · 가운데 버튼 드래그 이동\n이 창에서는 프로젝트 이미지·라벨·모델을 변경하지 않습니다.')))
        HeaderContext(header, tr('{v0} × {v1} px · 읽기 전용', v0=layout['width'], v1=layout['height']))
        root.addWidget(header)
        body = QFrame(objectName="detachedBody")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(18, 12, 18, 12)
        self.images = QComboBox()
        self.images.setMinimumContentsLength(30)
        self.images.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        for index, path in enumerate(self.paths):
            self.images.addItem(f"{index + 1}/{len(self.paths)} · {path.parent.name} / {path.name}")
        self.images.setToolTip(tr('현재 분할의 모든 Train 이미지 · Validation/Test 제외'))
        body_layout.addWidget(self.images)
        row = QHBoxLayout()
        self.angle = QDoubleSpinBox()
        self.angle.setRange(-layout["rotation_degrees"], layout["rotation_degrees"])
        self.angle.setSuffix(" °")
        self.scale = QDoubleSpinBox()
        self.scale.setRange(*layout["scale_range"])
        self.scale.setValue(1)
        self.scale.setSingleStep(0.05)
        self.scale.setSuffix(tr(' 배'))
        for label, field in ((tr('회전'), self.angle), (tr('증강 배율'), self.scale)):
            row.addWidget(QLabel(label))
            row.addWidget(field)
        extremes = QPushButton(tr('최대 회전·확대'), objectName="toolButton")
        extremes.clicked.connect(self.show_extreme)
        reset = QPushButton(tr('증강 초기화'), objectName="toolButton")
        reset.clicked.connect(self.reset_geometry)
        fit = QPushButton(tr('화면 맞춤'), objectName="toolButton")
        row.addWidget(extremes)
        row.addWidget(reset)
        row.addStretch(1)
        row.addWidget(fit)
        body_layout.addLayout(row)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        source_panel = QWidget()
        source_layout = QVBoxLayout(source_panel)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.addWidget(QLabel(tr('크롭 위치 · 증강 후 전체')))
        self.source_canvas = VideoCanvas(allow_point_placement=False)
        source_layout.addWidget(self.source_canvas, 1)
        splitter.addWidget(source_panel)
        output_panel = QWidget()
        output_layout = QVBoxLayout(output_panel)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(QLabel(tr('실제 학습 입력')))
        self.canvas = VideoCanvas(allow_point_placement=False)
        output_layout.addWidget(self.canvas, 1)
        splitter.addWidget(output_panel)
        source_panel.setVisible(self.is_crop)
        splitter.setSizes([550, 550])
        body_layout.addWidget(splitter, 1)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        body_layout.addWidget(self.summary)
        root.addWidget(body, 1)
        footer = QFrame(objectName="detachedFooter")
        footer_row = QHBoxLayout(footer)
        footer_row.setContentsMargins(18, 12, 18, 12)
        footer_row.addStretch(1)
        close = QPushButton(tr('닫기'), objectName="toolButton")
        close.setFixedWidth(88)
        close.clicked.connect(self.accept)
        footer_row.addWidget(close)
        root.addWidget(footer)
        # Enter in a numeric field must not activate the header's help button.
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
        fit.clicked.connect(self.canvas.fit_to_view)
        fit.clicked.connect(self.source_canvas.fit_to_view)
        self.boundary = None
        self.source_boundary = None
        self.label_items = []
        self.images.currentIndexChanged.connect(self.refresh)
        self.angle.valueChanged.connect(self.refresh)
        self.scale.valueChanged.connect(self.refresh)
        install_wheel_value_guard(self)
        self.refresh()

    def show_extreme(self):
        self.angle.setValue(self.input_layout["rotation_degrees"])
        self.scale.setValue(self.input_layout["scale_range"][1])

    def reset_geometry(self):
        self.angle.setValue(0)
        self.scale.setValue(1)

    def refresh(self, *_args):
        try:
            result = preview_geometry(self.paths[self.images.currentIndex()], self.input_layout,
                                      self.angle.value(), self.scale.value(),
                                      self.labels[self.images.currentIndex()] if self.labels is not None else None)
            image = np.ascontiguousarray(result["image"])
            height, width = image.shape[:2]
            self.canvas.set_frame(QImage(image.data, width, height, image.strides[0], QImage.Format_RGB888).copy())
            if self.boundary is not None:
                self.canvas.scene.removeItem(self.boundary)
            pen = QPen(QColor("#5b82e8"))
            pen.setCosmetic(True)
            for item in self.label_items:
                self.canvas.scene.removeItem(item)
            self.label_items = []
            if self.is_crop:
                whole = np.ascontiguousarray(result["source_image"])
                h, w = whole.shape[:2]
                self.source_canvas.set_frame(QImage(whole.data, w, h, whole.strides[0], QImage.Format_RGB888).copy())
                if self.source_boundary is not None:
                    self.source_canvas.scene.removeItem(self.source_boundary)
                x1, y1, x2, y2 = result["crop_box"]
                self.source_boundary = self.source_canvas.scene.addRect(x1, y1, x2-x1, y2-y1, pen)
                self.boundary = None
                for x, y in result["keypoints"]:
                    self.label_items.append(self.canvas.scene.addEllipse(float(x)-3, float(y)-3, 6, 6,
                                            pen, QBrush(QColor("#ffffff"))))
            else:
                self.boundary = self.canvas.scene.addPolygon(
                    QPolygonF([QPointF(float(x), float(y)) for x, y in result["keypoints"]]), pen)
            native_w, native_h = result["native_size"]
            fit = self.input_layout.get("resize_scale", 1.0)
            geometry = (tr(' · 라벨 주변 크롭 · {v0}개 점 보존 · 크기 맞춤 축소 없음', v0=len(result['keypoints'])) if self.is_crop
                        else (tr(' · 전체 축소 {v0:.1%} · 자르기 없음', v0=fit) if fit < 1 else tr(' · 크기 맞춤 축소 없음')))
            self.summary.setText(tr('원본 {v0} × {v1} px → 입력 {v2} × {v3} px', v0=native_w, v1=native_h, v2=width, v3=height)
                                 + geometry + tr(' · 증강 포함 배율 {v0:.1%}', v0=fit * self.scale.value())
                                 + (tr(' · 왼쪽 파란 테두리: 크롭 위치') if self.is_crop else tr(' · 파란 테두리: ROI 이미지 경계'))
                                 + tr(' · 검은 여백: 패딩'))
        except (OSError, ValueError) as error:
            self.summary.setText(tr('미리보기 불가 · {v0}', v0=error))
