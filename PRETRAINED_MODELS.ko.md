# 사전학습 모델 이용 조건

[English](PRETRAINED_MODELS.md) | 한국어

HAGISILTA는 Pose 추론과 지원하는 Bottom-up 학습의 초기화에 사전학습 가중치를
사용합니다. 가중치에는 [LICENSE_SCOPE.json](LICENSE_SCOPE.json)에 명시된
HAGISILTA 자체 파일의 Apache-2.0과 별도의 라이선스가 적용됩니다. 모델을
다운로드하거나 사용하기 전에 게시자의 조건을 확인하세요.

## 지원하는 SuperAnimal-TopViewMouse 체크포인트

아래 체크포인트는
[mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse)에서
제공합니다. Pose 모델인 `hrnet_w32`와 `resnet_50`은 HAGISILTA에서 사전학습
추론과 TopViewMouse 초기 가중치를 이용한 정식 Bottom-up 학습을 모두 지원합니다.

| 모델 식별자 | 체크포인트 파일명 | HAGISILTA에서의 용도 |
| --- | --- | --- |
| `hrnet_w32` | `superanimal_topviewmouse_hrnet_w32.pt` | Pose 추론 및 정식 Bottom-up 학습의 초기 가중치 |
| `resnet_50` | `superanimal_topviewmouse_resnet_50.pt` | Pose 추론 및 정식 Bottom-up 학습의 초기 가중치 |
| `fasterrcnn_resnet50_fpn_v2` | `superanimal_topviewmouse_fasterrcnn_resnet50_fpn_v2.pt` | 사전학습 검출기 |
| `fasterrcnn_mobilenet_v3_large_fpn` | `superanimal_topviewmouse_fasterrcnn_mobilenet_v3_large_fpn.pt` | 사전학습 검출기 |

이 체크포인트의 이용 조건은 게시자 저장소에서 확인하세요.

## TopViewMouse 이용 조건

게시자의 라이선스 표기는 Modified MIT입니다. 표준 MIT 라이선스나 SPDX `MIT`와
다른 맞춤 모델 라이선스입니다. 아래 요약의 기준은
[revision d443257e4b1145f7439e0953c2015d60fcdc1185의 모델 카드](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse/blob/d443257e4b1145f7439e0953c2015d60fcdc1185/README.md)입니다.

모델 카드에는 다음 조건이 명시되어 있습니다.

- 학술 및 비상업 목적의 이용과 허가의 양도 불가.
- Copyright 2023 Mackenzie Mathis, Shaokai Ye and contributors를 포함한
  게시자의 저작권과 허가 고지 보존.
- 고의적인 동물 위해 제한.
- 보증과 책임에 대한 면책.
- 허가 범위 밖의 상업용 이용은 Mackenzie W. Mathis 또는 EPFL TTO에 문의.

가중치 이용에는 모델 카드의 전체 조건이 적용됩니다. 게시자는 전이학습, 보강학습과
영상 적응을 권장합니다. 원본이나 추가 학습한 체크포인트를 공개, 재허가하거나
상업적으로 이용하기 전에 필요한 권한을 권리자에게 확인하세요.

## 가중치 다운로드

앱 ZIP과 개발용 ZIP에는 모델 가중치가 없습니다. 필요한 가중치는
게시자 저장소에서 별도로 받습니다. 별도 다운로드와
학습 초기화에 사용하는 가중치에도 게시자의 조건이 적용됩니다.

다운로드되는 모델 버전은 고정되어 있지 않으며, 게시자가 저장소를 갱신하면
바뀔 수 있습니다. 연구를 재현할 수 있도록 사용한 가중치 파일을 보관하고 출처와
해당 이용 조건을 기록하세요.

## 추가 MobileNet/ImageNet 가중치

DLC 3.0.0과 TorchVision 0.28.0에서 지원하는 MobileNet 검출기는 ImageNet 초기
가중치 파일인
[`mobilenet_v3_large-8738ca79.pth`](https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth)를
추가로 사용합니다.

[TorchVision 사전학습 모델 고지](https://github.com/pytorch/vision/blob/v0.28.0/README.md#pre-trained-model-license)를
읽고 이 가중치의 이용 및 재배포 허용 범위를 게시자에게 확인하세요. TorchVision의
BSD 소스 코드 라이선스와 HAGISILTA의 Apache-2.0으로 해당 권한이 정해지지는
않습니다. 모델과 학습 데이터에는 별도의 조건이 있습니다.

## 연구 인용과 사용자 자료

논문 등 연구 보고서에서는 게시자 모델 카드의 SuperAnimal 인용 안내와
DeepLabCut 문서의 해당 소프트웨어 및 방법 인용 안내를 따르세요. 인용과 가중치의
이용 및 재배포 허가는 별개의 요건입니다.

이 문서는 사용자 영상, 라벨, 수치 CSV 결과와 그 밖의 사용자 소유 자료에
라이선스를 부여하지 않습니다. 데이터셋, 출력물과 추가 학습한
체크포인트에 적용되는 권리는 각각 확인하세요. 앱 이미지의 라이선스 범위는
[앱 리소스](LICENSE_GUIDE.ko.md#앱-리소스)에서 안내합니다.
