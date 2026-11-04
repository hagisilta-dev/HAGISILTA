# Pretrained model terms

English | [Korean](PRETRAINED_MODELS.ko.md)

HAGISILTA uses pretrained weights for pose inference and initialization of its
supported Bottom-up training workflow. The weights have their own licenses,
separate from the Apache-2.0 license for HAGISILTA's original files listed in
[LICENSE_SCOPE.json](LICENSE_SCOPE.json). Read the publisher's terms before
downloading or using a model.

## Supported SuperAnimal-TopViewMouse checkpoints

The following checkpoints are supplied by
[mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse).
The pose models `hrnet_w32` and `resnet_50` support both pretrained inference and
formal Bottom-up training with TopViewMouse initial weights in HAGISILTA.

| Model identifier | Checkpoint filename | Use in HAGISILTA |
| --- | --- | --- |
| `hrnet_w32` | `superanimal_topviewmouse_hrnet_w32.pt` | Pose inference and initial weights for formal Bottom-up training |
| `resnet_50` | `superanimal_topviewmouse_resnet_50.pt` | Pose inference and initial weights for formal Bottom-up training |
| `fasterrcnn_resnet50_fpn_v2` | `superanimal_topviewmouse_fasterrcnn_resnet50_fpn_v2.pt` | Pretrained detector |
| `fasterrcnn_mobilenet_v3_large_fpn` | `superanimal_topviewmouse_fasterrcnn_mobilenet_v3_large_fpn.pt` | Pretrained detector |

Use the publisher repository's model terms for these checkpoints.

## TopViewMouse use conditions

The publisher labels the license Modified MIT. It is a custom model license,
not the standard MIT license or SPDX `MIT`. This summary refers to the
[model card at revision d443257e4b1145f7439e0953c2015d60fcdc1185](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse/blob/d443257e4b1145f7439e0953c2015d60fcdc1185/README.md).

The model card specifies:

- Use for academic, non-commercial purposes under a non-transferable grant.
- Retention of the publisher's copyright and permission notices, including
  Copyright 2023 Mackenzie Mathis, Shaokai Ye and contributors.
- A restriction against deliberately harming animals.
- Warranty and liability disclaimers.
- Contact with Mackenzie W. Mathis and/or EPFL TTO for commercial licensing
  outside the grant.

The full model-card terms govern use of the weights. The publisher recommends
transfer learning, fine-tuning, and video adaptation. Before publishing,
sublicensing, or commercially using original or fine-tuned checkpoints, confirm
the required permission with the rights holder.

## Downloading weights

The app and developer ZIPs contain no model weights.
Required weights are downloaded separately from the publisher repository.
The publisher's conditions apply to these downloads as well as to weights used
for training initialization.

Downloaded model versions are not fixed and may change when the publisher
updates the repository. For reproducibility, keep the weight files used in your
study and record their source and applicable terms.

## Additional MobileNet/ImageNet weight

With DLC 3.0.0 and TorchVision 0.28.0, the supported MobileNet detector also uses
ImageNet initialization weights from
[`mobilenet_v3_large-8738ca79.pth`](https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth).

Consult the
[TorchVision pretrained model notice](https://github.com/pytorch/vision/blob/v0.28.0/README.md#pre-trained-model-license)
and confirm permitted use and redistribution of this weight with its publisher.
TorchVision's BSD source-code license and HAGISILTA's Apache-2.0 license do not
establish those permissions. The model and its training data have separate
terms.

## Research attribution and user data

In research reports, follow the publisher model card's SuperAnimal citation
instructions and the applicable software and method citations in the DeepLabCut
documentation. Citation and permission to use or redistribute weights are
separate requirements.

This document does not license user videos, labels, numerical CSV results, or
other user-owned data. Consider the applicable rights for each dataset,
output, and fine-tuned checkpoint separately. The license scope for application
artwork is described in [Application assets](LICENSE_GUIDE.md#application-assets).
