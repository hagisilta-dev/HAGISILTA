# 라이선스 안내

[English](LICENSE_GUIDE.md) | 한국어

HAGISILTA 자체 코드, 문서, 번역과 명시된 앱 리소스는 [Apache-2.0](LICENSE)으로
배포합니다. 전체 적용 파일 목록은 [LICENSE_SCOPE.json](LICENSE_SCOPE.json)에,
프로젝트 저작권과 출처 표시는 [NOTICE](NOTICE)에 있습니다. 외부 라이브러리,
실행 파일, 사전학습 가중치와 데이터셋은 각각의 라이선스를 유지합니다.
전체 조건은 해당 라이선스 원문에서 확인하세요.

## 배포판 구성

HAGISILTA 앱 ZIP에는 앱 소스, 런처, 설치 스크립트, 설정과 선별한 앱 리소스가
포함됩니다. Python, Miniforge, 외부 실행 패키지, CUDA 라이브러리, 사전학습
체크포인트는 포함하지 않습니다. 설치기는 원래 배포 서비스에서 실행 패키지를
가져오며 모델 가중치는 필요할 때 별도로 받습니다.

## 앱 리소스

프로젝트 저작권: Copyright 2026 Ok Seungwon and Song Jun-Hwi.

| 리소스 | 제작자와 적용 범위 | 라이선스 |
| --- | --- | --- |
| `assets/hagisilta-splash-artwork-v4.png` | 스플래시 이미지. 제작자: Song Jun-Hwi. | Apache-2.0 |
| `assets/hagisilta-app.ico` | 자체 제작한 HAGISILTA 앱 아이콘. | Apache-2.0 |
| `assets/check.svg`, `assets/spin-up.svg`, `assets/spin-down.svg` | 간단한 기하 도형으로 직접 작성한 조작 아이콘. | Apache-2.0 |
| `assets/i18n/*.json`, `assets/i18n/installer/*.json`의 자체 작성 문구 | HAGISILTA 앱 번역 카탈로그. 외부 명칭과 인용된 고지는 각각의 기존 지위를 유지합니다. | 자체 작성 앱 문구에 Apache-2.0 적용 |

리소스를 재배포할 때는 Apache-2.0에 따라 해당 저작권과 출처 표시를 보존하고,
수정한 파일에 변경 사항을 표시하세요. 전체 조건은 [LICENSE](LICENSE)에 있습니다.

Apache-2.0은 명시된 제한적 목적을 넘어서는 상표권을 부여하지 않습니다.
이미지 재배포가 프로젝트나 저자의 보증, 지지를 뜻하지는 않습니다.

### 폰트와 의존성 리소스

HAGISILTA 앱 ZIP에는 폰트 파일이 없습니다. 앱은 Segoe UI와 Malgun Gothic 등
시스템에 설치된 폰트를 요청합니다. 폰트 재배포에는 해당 공급자의 조건이 적용됩니다.
[Microsoft 폰트 FAQ](https://learn.microsoft.com/en-us/typography/fonts/font-faq)를
확인하세요.

설치되는 Matplotlib, JupyterLab 등 의존성 내부의 폰트, 아이콘에는 별도 고지가
있으므로 [구성요소별 라이선스](#구성요소별-라이선스)를 확인하세요.
외부 패키지의 리소스는 HAGISILTA의 Apache-2.0으로 재허가하지 않습니다.

### 사용자 자료와 모델 가중치

사용자 영상, 라벨, 데이터셋, 프로젝트 파일, 모델 체크포인트는 앱 배포판에
포함되지 않으며 이 문서의 라이선스 적용 대상도 아닙니다. HAGISILTA를 사용해도
해당 자료의 저작권과 공개 조건은 바뀌지 않습니다. 외부 사전학습 가중치는
[PRETRAINED_MODELS.ko.md](PRETRAINED_MODELS.ko.md)를 확인하세요.

## 구성요소별 라이선스

아래 표는 2026-11-03에 Windows 설치 환경에서 확인한 버전을 기준으로 합니다.
`environment.yml`에 버전 범위가 있고 Miniforge 다운로드가 `latest`를 사용하므로
설치 시점에 따라 버전이 달라질 수 있습니다. 설치하거나 재배포하는 버전의
라이선스 파일을 확인하세요. 패키지에 포함된 네이티브 라이브러리, 폰트와
모델 가중치는 Python 코드와 별도의 조건을 따를 수 있습니다.

| 구성요소와 버전 | 라이선스 / 별도 구성요소 | 공식 출처와 재배포 조건 |
| --- | --- | --- |
| DeepLabCut 3.0.0 | LGPL-3.0-or-later | [해당 버전의 라이선스](https://github.com/DeepLabCut/DeepLabCut/blob/v3.0.0/LICENSE). 라이선스, 저작자 표시와 고지를 보존하세요. 제공되는 `AUTHORS`, `NOTICE.yml`도 포함합니다. 라이브러리나 결합 저작물을 배포할 때 적용되는 LGPL 조건을 충족해야 합니다. 모델 가중치에는 별도 조건이 있습니다. |
| DLClibrary 0.0.12 | LGPL v3 원문 및 LGPLv3 분류자 | [라이선스](https://github.com/DeepLabCut/DLClibrary/blob/main/LICENSE). 라이선스와 `NOTICE.yml`을 보존하세요. 코드 라이선스는 이 라이브러리가 내려받는 모델 가중치에 적용되지 않습니다. |
| PySide6, PySide6 Essentials/Addons, Shiboken6 6.12.0 | 메타데이터에 LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only 제시. 별도 상업용 선택지 있음 | HAGISILTA는 LGPLv3 선택지가 있는 QtCore, QtGui, QtWidgets를 사용합니다. [Qt for Python 라이선스](https://doc.qt.io/qtforpython-6/licenses.html), [LGPL 의무](https://www.qt.io/development/open-source-lgpl-obligations). 해당 라이선스 원문, 모듈과 플러그인 라이선스, 외부 구성요소 고지를 포함하세요. 라이브러리를 교체하거나 수정하고 그 수정을 디버깅할 수 있도록 해야 합니다. 다른 Qt 모듈은 GPL 전용일 수 있습니다. |
| OpenCV Python / headless 4.11.0.86 | OpenCV 핵심 Apache-2.0, Python 빌드 스크립트 MIT, 포함된 FFmpeg LGPL2.1, Intel IPP 별도 조건 | [OpenCV 라이선스](https://opencv.org/license/), [Python wheel 프로젝트](https://github.com/opencv/opencv-python). 패키지 라이선스와 FFmpeg, Intel IPP 고지를 포함한 전체 `LICENSE-3RD-PARTY.txt`를 보존하세요. |
| imageio-ffmpeg 0.6.0 및 그 Windows FFmpeg 7.1 실행 파일 | Python 래퍼 BSD-2-Clause. 해당 실행 파일은 GPL-3.0-or-later | [래퍼 프로젝트](https://github.com/imageio/imageio-ffmpeg), [FFmpeg 라이선스](https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md). 이 실행 파일은 `--enable-gpl --enable-version3`로 빌드되었습니다. 해당 바이너리의 재배포에는 라이선스에서 정한 GPL 고지와 대응 소스 접근 제공이 필요합니다. OpenCV에 포함된 FFmpeg와는 다른 빌드입니다. |
| PyTorch 2.13.0+cu130, TorchVision 0.28.0+cu130 | PyTorch 최상위 BSD 계열. 추가 Apache/BSD/MIT/BSL 및 LLVM 예외 구성요소 있음. TorchVision BSD-3-Clause | [PyTorch 라이선스](https://github.com/pytorch/pytorch/blob/main/LICENSE), [TorchVision 라이선스](https://github.com/pytorch/vision/blob/v0.28.0/LICENSE). 제공된 라이선스 디렉터리 전체를 보존하세요. 사전학습 가중치와 GPU 바이너리에는 각각의 조건이 적용됩니다. |
| GPU wheel의 CUDA 13 runtime, cuDNN 9.20.0.48 | 별도의 NVIDIA SDK 조건 | [CUDA 13.0 EULA](https://docs.nvidia.com/cuda/archive/13.0.0/eula/index.html), [cuDNN EULA](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/eula.html). 허용된 파일을 사용하고 해당 파일의 조건에 맞춰 재배포해야 합니다. |
| NumPy 1.26.4, SciPy 1.17.1, pandas 2.3.3 | BSD 계열 프로젝트 라이선스와 추가 수치 라이브러리, 런타임 고지 | [NumPy 라이선스](https://numpy.org/doc/1.26/license.html), [SciPy 라이선스](https://github.com/scipy/scipy/blob/main/LICENSE.txt), [pandas 라이선스](https://github.com/pandas-dev/pandas/blob/main/LICENSE). 프로젝트 고지뿐 아니라 제공된 OpenBLAS/LAPACK/GCC-runtime 고지와 예외도 보존하세요. |
| Matplotlib 3.8.4 | PSF 기반 Matplotlib 라이선스, 별도 폰트, 구성요소 고지 | [라이선스](https://matplotlib.org/stable/project/license.html). 포함된 폰트 등 제공된 관련 고지를 모두 보존하고, Matplotlib 자체를 수정한다면 변경 사항을 설명하세요. |
| Shapely 2.2.0 | BSD-3-Clause, 포함된 GEOS LGPL2.1 | [Shapely](https://shapely.readthedocs.io/en/latest/), [GEOS](https://libgeos.org). `LICENSE_GEOS`를 보존하고 재배포하는 GEOS 바이너리에 적용되는 LGPL 조건을 충족해야 합니다. |
| blosc2 4.9.1 | BSD-3-Clause, 내부 TinyCC/libtcc LGPL-2.1-or-later, tinyexpr zlib, SLEEF BSL-1.0 | [miniexpr](https://github.com/Blosc/miniexpr), [minicc](https://github.com/Blosc/minicc). 포함된 고지와 함께 배포되는 libtcc 바이너리에 적용되는 소스 제공 조건을 보존하세요. |
| certifi 2026.6.17, fqdn 1.5.1, tqdm 4.69.0 | certifi/fqdn MPL-2.0, tqdm MPL-2.0 AND MIT | [certifi 라이선스](https://github.com/certifi/python-certifi/blob/master/LICENSE), [tqdm 라이선스](https://github.com/tqdm/tqdm/blob/master/LICENCE), [MPL 2.0](https://www.mozilla.org/en-US/MPL/2.0/). 적용 대상 파일의 고지와 해당 소스의 이용 가능성을 보존하세요. |
| Albumentations 1.4.3 | 이 버전은 MIT | [해당 버전의 라이선스](https://github.com/albumentations-team/albumentations/blob/1.4.3/LICENSE). MIT 고지를 보존하세요. 다른 릴리스에는 다른 조건이 적용될 수 있습니다. |
| JupyterLab 4.6.1 | BSD 계열, 포함된 JavaScript, 폰트, 아이콘에는 추가 조건 있음 | [JupyterLab 라이선스](https://github.com/jupyterlab/jupyterlab/blob/main/LICENSE). 제공된 `third-party-licenses.json`과 구성요소의 저작권 및 출처 표시를 보존하세요. [EPL-2.0](https://github.com/kieler/elkjs/blob/master/LICENSE.md)의 elkjs 0.9.3과 [CC BY 4.0 / OFL 1.1 / MIT](https://fontawesome.com/license/free)의 Font Awesome Free 5.15.4도 포함합니다. |
| h5py 3.16.0 | BSD-3-Clause 및 HDF5, 기타 포함 구성요소 고지 | [h5py 라이선스](https://github.com/h5py/h5py/blob/master/LICENSE). HDF5를 포함한 제공 라이선스 디렉터리 전체를 보존하세요. |
| ruamel.yaml 0.19.1 | MIT | [공식 프로젝트](https://pypi.org/project/ruamel.yaml/0.19.1/). 설치된 버전의 저작권과 허가 고지를 보존하세요. |
| filelock 3.15.4 | 이 버전은 Unlicense | [해당 버전의 프로젝트](https://pypi.org/project/filelock/3.15.4/). 제공된 라이선스를 보존하세요. 다른 릴리스에는 다른 조건이 적용될 수 있습니다. |
| Miniforge 설치기, Conda 26.3.2 기본 환경 | Miniforge BSD-3-Clause, 포함된 패키지는 독립적인 라이선스 사용 | [Miniforge 라이선스](https://github.com/conda-forge/miniforge/blob/main/LICENSE). 개별 패키지의 라이선스를 보존하고 해당 소스 제공 조건을 충족해야 합니다. 기본 패키지에는 GPL-2.0-or-later의 lzo, LGPL-2.1-only의 libiconv, LGPL-3.0-only의 frozendict가 포함됩니다. |
| Windows UCRT 10.0.26100.0, vc14_runtime/vcomp14 14.51.36231 | Microsoft SDK/Visual C++ runtime 조건 | [Microsoft 재배포 문서](https://learn.microsoft.com/en-us/cpp/windows/redistributing-visual-cpp-files). 재배포 자격과 포함할 파일에 적용되는 조건을 확인하세요. |

### FFmpeg 실행과 라이선스

HAGISILTA는 ROI 영상을 저장할 때 imageio-ffmpeg의 FFmpeg 실행 파일을 별도
프로세스로 호출해 원시 프레임을 전달합니다. HAGISILTA 자체 소스의 Apache-2.0과
FFmpeg 실행 파일의 라이선스는 각각 유지되며, 해당 바이너리를 재배포할 때는
적용되는 고지와 대응 소스 제공 조건을 따라야 합니다. GPL에서 프로그램의
결합 여부는 프로세스 분리뿐 아니라 통신 방식과 내용, 결합 관계도 함께
고려합니다. [GNU FAQ](https://www.gnu.org/licenses/gpl-faq.en.html#MereAggregation)를
확인하세요.

### 실행 환경을 함께 재배포하는 경우

외부 실행 패키지를 배포판에 포함하면 해당 라이선스, 저작권 고지와 그 밖의
필수 파일을 함께 제공해야 합니다. LGPL/GPL 구성요소와 결합 저작물에는
해당하는 소스 제공 조건과 라이브러리 교체 조건도 적용됩니다. 재배포하는
FFmpeg 바이너리에 필요한 빌드 정보와 대응 소스 접근도 제공하세요.

내부 라이브러리, 폰트와 플러그인을 포함한 각 패키지의 전체 고지를 사용하세요.
pycocotools나 webencodings를 함께 배포하려면 해당 버전의 완전한 고지를 확보해야
합니다. 위 표는 주요 구성요소를 다루며, 실행 환경을 묶어 배포할 때는 그 밖에
포함되는 패키지의 라이선스도 적용해야 합니다.

## 라이선스 원문

배포판에는 주요 외부 라이선스와 고지를 원문 그대로 포함합니다.
아래 표는 원문 출처를 안내하며 적용 대상은 위의 구성요소별 설명과
[PRETRAINED_MODELS.ko.md](PRETRAINED_MODELS.ko.md)에서 확인할 수 있습니다.

| 파일 | 원문 출처 |
| --- | --- |
| [`LICENSE`](LICENSE) | [Apache Software Foundation, Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt), 원문 그대로 |
| [`licenses/LGPL-3.0.txt`](licenses/LGPL-3.0.txt) | [FFmpeg n7.1, COPYING.LGPLv3](https://raw.githubusercontent.com/FFmpeg/FFmpeg/n7.1/COPYING.LGPLv3) |
| [`licenses/GPL-3.0.txt`](licenses/GPL-3.0.txt) | [FFmpeg n7.1, COPYING.GPLv3](https://raw.githubusercontent.com/FFmpeg/FFmpeg/n7.1/COPYING.GPLv3) |
| [`licenses/LGPL-2.1.txt`](licenses/LGPL-2.1.txt) | [FFmpeg n7.1, COPYING.LGPLv2.1](https://raw.githubusercontent.com/FFmpeg/FFmpeg/n7.1/COPYING.LGPLv2.1) |
| [`licenses/DeepLabCut-NOTICE.yml`](licenses/DeepLabCut-NOTICE.yml) | 설치 패키지: `deeplabcut==3.0.0`, `deeplabcut-3.0.0.dist-info/licenses/NOTICE.yml` |
| [`licenses/DLClibrary-NOTICE.yml`](licenses/DLClibrary-NOTICE.yml) | 설치 패키지: `dlclibrary==0.0.12`, `dlclibrary-0.0.12.dist-info/NOTICE.yml` |

위 패키지 경로는 HAGISILTA 실행 환경의 `Lib/site-packages`를 기준으로 합니다.
FFmpeg 태그는 라이선스 문서의 출처를 나타내며, 개별 FFmpeg 빌드에는 서로 다른
라이선스 조건이 적용될 수 있습니다. LGPL-3.0은 GPL-3.0에 추가 허용 조건을
적용하므로 두 원문을 함께 제공합니다.

파일별 출처와 SHA-256은
[`licenses/source-provenance.json`](licenses/source-provenance.json)에 있습니다.

각 의존성 패키지에 포함된 라이선스와 저작권 고지 파일을 보존하세요.
의존성을 재배포할 때는 해당 라이선스의 소스 제공 의무와 그 밖의 적용 조건도
준수해야 합니다. 이 참고 사본이 패키지의 고지 파일이나 의무를 대신하지는 않습니다.
