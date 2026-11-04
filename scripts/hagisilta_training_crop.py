# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
"""Serializable, label-safe training crop. Imported only by preview/workers."""

from hagisilta_i18n import tr
import math
import random

import albumentations as A
from albumentations.augmentations.crops import functional as crops


class LabelSafeCrop(A.DualTransform):
    """Keep all finite annotation points plus context; never resize to fit.

    Crop origins are sampled from the feasible interval, not just centered on
    the animal. This avoids teaching the model that labels always sit centrally.
    Padding before this transform supplies context at original-image edges.
    """
    def __init__(self, width, height, margin=32, randomize=True, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.width, self.height = int(width), int(height)
        self.margin, self.randomize = int(margin), bool(randomize)

    @property
    def targets_as_params(self):
        return ["image", "keypoints"]

    def get_params_dependent_on_targets(self, params):
        image = params["image"]
        # Albumentations' internal xy representation is x,y,angle,scale,
        # optional extra channels, then class_labels. DLC supplies visibility
        # as the first extra channel. Preview xy pairs have no extra channel.
        points = [(float(k[0]), float(k[1])) for k in params["keypoints"]
                  if (len(k) < 6 or k[4] > 0) and math.isfinite(k[0]) and math.isfinite(k[1])]

        def origin(axis, size, image_size):
            low, high = 0, image_size - size
            if points:
                lower = min(point[axis] for point in points)
                upper = max(point[axis] for point in points)
                low = max(low, math.ceil(upper + self.margin + 1 - size))
                high = min(high, math.floor(lower - self.margin))
            if low > high:
                raise ValueError(tr('크롭 크기가 라벨과 여유 공간보다 작습니다. 신체점을 자르거나 축소하지 않습니다. 크기를 늘리세요.'))
            return random.randint(low, high) if self.randomize else (low + high) // 2

        x, y = origin(0, self.width, image.shape[1]), origin(1, self.height, image.shape[0])
        return {"x_min": x, "y_min": y, "x_max": x + self.width, "y_max": y + self.height}

    def apply(self, image, x_min=0, y_min=0, x_max=0, y_max=0, **params):
        return image[y_min:y_max, x_min:x_max].copy()

    def apply_to_bbox(self, bbox, **params):
        return crops.bbox_crop(bbox, *(params[key] for key in ("x_min", "y_min", "x_max", "y_max", "rows", "cols")))

    def apply_to_keypoint(self, keypoint, **params):
        return crops.crop_keypoint_by_coords(keypoint, (params["x_min"], params["y_min"],
                                                       params["x_max"], params["y_max"]))

    def get_transform_init_args_names(self):
        return "width", "height", "margin", "randomize"
