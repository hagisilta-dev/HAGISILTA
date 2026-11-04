# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from hagisilta_i18n import tr

import importlib
import platform
import sys


def package_version(module_name: str, display_name: str, failures: list[str]) -> None:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", tr('설치됨'))
        print(f"{display_name}: {version}")
    except Exception as exc:
        failures.append(tr('{v0} 불러오기 실패: {v1}', v0=display_name, v1=exc))


def main() -> int:
    failures: list[str] = []
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")

    for module_name, display_name in (
        ("deeplabcut", "DeepLabCut"),
        ("cv2", "OpenCV"),
        ("pandas", "pandas"),
        ("h5py", "h5py"),
        ("PySide6", "PySide6"),
        ("shapely", tr('Shapely (ROI 경계 분할)')),
    ):
        package_version(module_name, display_name, failures)

    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        print(f"PyTorch CUDA runtime: {torch.version.cuda or tr('없음')}")
        print(tr('CUDA 사용 가능: {v0}', v0=torch.cuda.is_available()))
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            value = torch.ones((64, 64), device="cuda").sum().item()
            print(tr('GPU 연산 테스트: {v0}', v0=tr('통과') if value == 4096.0 else tr('실패')))
        else:
            print(tr('GPU를 사용할 수 없어 CPU 모드로 실행됩니다. 학습과 추론은 매우 느릴 수 있습니다.'))
    except Exception as exc:
        failures.append(tr('PyTorch 점검 실패: {v0}', v0=exc))

    if failures:
        print(tr('\n문제:'))
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(tr('\nHAGISILTA 실행 환경 점검을 통과했습니다.'))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
