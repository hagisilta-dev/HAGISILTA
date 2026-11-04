# License guide

English | [Korean](LICENSE_GUIDE.ko.md)

HAGISILTA's original code, documentation, translations and listed application
assets are distributed under [Apache-2.0](LICENSE). The complete file inventory
is in [LICENSE_SCOPE.json](LICENSE_SCOPE.json), and project attribution is in
[NOTICE](NOTICE). External libraries, executables, pretrained weights and
datasets retain their own licenses. The original license texts provide the
full terms.

## Distribution contents

The HAGISILTA app ZIP contains application source, launchers, installation scripts,
configuration and selected application assets. It does not contain Python,
Miniforge, third-party runtime packages, CUDA libraries or pretrained checkpoints.
The installer obtains runtime packages from their upstream distribution services;
model weights are obtained separately when required.

## Application assets

Project copyright: Copyright 2026 Ok Seungwon and Song Jun-Hwi.

| Asset | Attribution and scope | License |
| --- | --- | --- |
| `assets/hagisilta-splash-artwork-v4.png` | Splash artwork by Song Jun-Hwi. | Apache-2.0 |
| `assets/hagisilta-app.ico` | Original HAGISILTA application icon. | Apache-2.0 |
| `assets/check.svg`, `assets/spin-up.svg`, `assets/spin-down.svg` | Original control icons made from simple geometric shapes. | Apache-2.0 |
| Original text in `assets/i18n/*.json`, `assets/i18n/installer/*.json` | HAGISILTA application translation catalogs; third-party names and quoted notices retain their own status. | Apache-2.0 for original application text |

When redistributing these assets, preserve the applicable copyright and
attribution notices and identify modified files as required by Apache-2.0.
The full terms are in [LICENSE](LICENSE).

Apache-2.0 does not grant trademark rights beyond its stated limited purposes.
Redistributing the artwork does not imply endorsement by the project or its
authors.

### Fonts and dependency assets

No font files are included in the HAGISILTA app ZIP. The application requests
system-installed fonts such as Segoe UI and Malgun Gothic. Font redistribution
is governed by the font supplier's terms; see the
[Microsoft font FAQ](https://learn.microsoft.com/en-us/typography/fonts/font-faq).

Fonts and icons within installed dependencies such as Matplotlib and JupyterLab
have separate notices; see [Component licenses](#component-licenses).
External package assets are not relicensed under HAGISILTA's Apache-2.0 license.

### User materials and model weights

User videos, labels, datasets, project files and model checkpoints are not
included in the app distribution or licensed by this document. Using HAGISILTA
does not change their copyright or publication conditions. For external
pretrained weights, see [PRETRAINED_MODELS.md](PRETRAINED_MODELS.md).

## Component licenses

The table describes versions verified in a Windows installation on 2026-11-03.
Versions can differ between installations because `environment.yml` includes
version ranges and the Miniforge download uses `latest`. Use the license files
for the versions you install or redistribute. Native libraries, fonts and
model weights included with a package may have separate terms from its Python code.

| Component and version | License / separate components | Official source and redistribution conditions |
| --- | --- | --- |
| DeepLabCut 3.0.0 | LGPL-3.0-or-later | [Versioned license](https://github.com/DeepLabCut/DeepLabCut/blob/v3.0.0/LICENSE). Retain its license, authorship and notices, including supplied `AUTHORS` and `NOTICE.yml` files; meet the applicable LGPL conditions if conveying the library or a combined work. Model weights have separate terms. |
| DLClibrary 0.0.12 | LGPL v3 text and LGPLv3 classifier | [License](https://github.com/DeepLabCut/DLClibrary/blob/main/LICENSE). Retain the license and `NOTICE.yml`; its code license does not cover the model weights it downloads. |
| PySide6, PySide6 Essentials/Addons, Shiboken6 6.12.0 | Metadata offers LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only; separate commercial option | HAGISILTA uses QtCore, QtGui and QtWidgets, which offer an LGPLv3 option. [Qt for Python licenses](https://doc.qt.io/qtforpython-6/licenses.html), [LGPL obligations](https://www.qt.io/development/open-source-lgpl-obligations). Include the applicable license texts, module/plugin licenses and third-party notices. Preserve the ability to replace or modify the library and debug those modifications. Other Qt modules may be GPL-only. |
| OpenCV Python / headless 4.11.0.86 | OpenCV core Apache-2.0; Python build scripts MIT; bundled FFmpeg LGPL2.1; Intel IPP separate terms | [OpenCV licensing](https://opencv.org/license/), [Python wheel project](https://github.com/opencv/opencv-python). Preserve the package license and full `LICENSE-3RD-PARTY.txt`, including FFmpeg and Intel IPP notices. |
| imageio-ffmpeg 0.6.0; its Windows FFmpeg 7.1 executable | Python wrapper BSD-2-Clause; listed executable GPL-3.0-or-later | [Wrapper project](https://github.com/imageio/imageio-ffmpeg), [FFmpeg licensing](https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md). This executable is built with `--enable-gpl --enable-version3`. Redistribution of that binary requires GPL notices and corresponding-source access as specified by its license. It is a different build from OpenCV's bundled FFmpeg. |
| PyTorch 2.13.0+cu130; TorchVision 0.28.0+cu130 | PyTorch top-level BSD-style; additional Apache/BSD/MIT/BSL and LLVM-exception components; TorchVision BSD-3-Clause | [PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE), [TorchVision license](https://github.com/pytorch/vision/blob/v0.28.0/LICENSE). Preserve the complete supplied license tree. Pretrained weights and GPU binaries have their own applicable terms. |
| CUDA 13 runtime; cuDNN 9.20.0.48 in the GPU wheel | Separate NVIDIA SDK terms | [CUDA 13.0 EULA](https://docs.nvidia.com/cuda/archive/13.0.0/eula/index.html), [cuDNN EULA](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/eula.html). Redistribution must use the permitted files and meet their matching terms. |
| NumPy 1.26.4; SciPy 1.17.1; pandas 2.3.3 | BSD-family project licenses; additional numerical-library/runtime notices | [NumPy license](https://numpy.org/doc/1.26/license.html), [SciPy license](https://github.com/scipy/scipy/blob/main/LICENSE.txt), [pandas license](https://github.com/pandas-dev/pandas/blob/main/LICENSE). Preserve the supplied OpenBLAS/LAPACK/GCC-runtime notices and exceptions as well as project notices. |
| Matplotlib 3.8.4 | PSF-based Matplotlib license; separate font/component notices | [License](https://matplotlib.org/stable/project/license.html). Preserve all relevant supplied notices, including bundled fonts; describe changes if modifying Matplotlib itself. |
| Shapely 2.2.0 | BSD-3-Clause; bundled GEOS LGPL2.1 | [Shapely](https://shapely.readthedocs.io/en/latest/), [GEOS](https://libgeos.org). Keep `LICENSE_GEOS` and satisfy the LGPL conditions applicable to any redistributed GEOS binary. |
| blosc2 4.9.1 | BSD-3-Clause; internal TinyCC/libtcc LGPL-2.1-or-later; tinyexpr zlib; SLEEF BSL-1.0 | [miniexpr](https://github.com/Blosc/miniexpr), [minicc](https://github.com/Blosc/minicc). Preserve bundled notices and the applicable source provisions for an included libtcc binary. |
| certifi 2026.6.17; fqdn 1.5.1; tqdm 4.69.0 | certifi/fqdn MPL-2.0; tqdm MPL-2.0 AND MIT | [certifi license](https://github.com/certifi/python-certifi/blob/master/LICENSE), [tqdm license](https://github.com/tqdm/tqdm/blob/master/LICENCE), [MPL 2.0](https://www.mozilla.org/en-US/MPL/2.0/). Preserve covered-file notices and applicable source availability. |
| Albumentations 1.4.3 | MIT for this version | [Versioned license](https://github.com/albumentations-team/albumentations/blob/1.4.3/LICENSE). Retain the MIT notice. Other releases may have different terms. |
| JupyterLab 4.6.1 | BSD-family; bundled JavaScript/fonts/icons have additional terms | [JupyterLab license](https://github.com/jupyterlab/jupyterlab/blob/main/LICENSE). Preserve the supplied `third-party-licenses.json` and component attribution, including elkjs 0.9.3 under [EPL-2.0](https://github.com/kieler/elkjs/blob/master/LICENSE.md) and Font Awesome Free 5.15.4 under [CC BY 4.0 / OFL 1.1 / MIT](https://fontawesome.com/license/free). |
| h5py 3.16.0 | BSD-3-Clause plus HDF5 and other bundled notices | [h5py license](https://github.com/h5py/h5py/blob/master/LICENSE). Retain the full supplied license directory, including HDF5. |
| ruamel.yaml 0.19.1 | MIT | [Official project](https://pypi.org/project/ruamel.yaml/0.19.1/). Retain the installed version's copyright and permission notice. |
| filelock 3.15.4 | Unlicense for this version | [Versioned project](https://pypi.org/project/filelock/3.15.4/). Retain the supplied license. Other releases may have different terms. |
| Miniforge installer; Conda 26.3.2 base environment | Miniforge BSD-3-Clause; included packages have independent licenses | [Miniforge license](https://github.com/conda-forge/miniforge/blob/main/LICENSE). Preserve the individual package licenses and meet their source-provision conditions. Base packages include lzo GPL-2.0-or-later, libiconv LGPL-2.1-only and frozendict LGPL-3.0-only. |
| Windows UCRT 10.0.26100.0; vc14_runtime/vcomp14 14.51.36231 | Microsoft SDK/Visual C++ runtime terms | [Microsoft redistribution documentation](https://learn.microsoft.com/en-us/cpp/windows/redistributing-visual-cpp-files). Check redistribution eligibility and the terms for the files being packaged. |

### FFmpeg execution and licensing

HAGISILTA encodes ROI videos by sending raw frames to the FFmpeg executable
supplied by imageio-ffmpeg through a separate process. HAGISILTA's original source retains
Apache-2.0 and the FFmpeg executable retains its own license; redistributing
that binary requires its applicable notices and corresponding-source access.
Whether interacting programs form a combined work under the GPL depends on
how they communicate and are combined, not solely on process separation; see
the [GNU FAQ](https://www.gnu.org/licenses/gpl-faq.en.html#MereAggregation).

### Redistributing runtime components

When including external runtime packages in a distribution, provide their
matching licenses, copyright notices and other required files. LGPL/GPL
components and combined works also require the applicable source-provision
and library-replacement arrangements. Keep the build information and
corresponding-source access required for the FFmpeg binary being redistributed.

Use the complete notices supplied with each package, including those for
embedded libraries, fonts and plugins. Before bundling pycocotools or
webencodings, obtain the complete notices for the exact versions being packaged.
The table covers principal components; a runtime bundle must also address the
licenses of its other included packages.

## License texts

The distribution includes selected upstream license texts and notices,
preserved verbatim. The table lists their sources; the component sections above
and [PRETRAINED_MODELS.md](PRETRAINED_MODELS.md) describe where they apply.

| File | Upstream source |
| --- | --- |
| [`LICENSE`](LICENSE) | [Apache Software Foundation, Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt), unmodified |
| [`licenses/LGPL-3.0.txt`](licenses/LGPL-3.0.txt) | [FFmpeg n7.1, COPYING.LGPLv3](https://raw.githubusercontent.com/FFmpeg/FFmpeg/n7.1/COPYING.LGPLv3) |
| [`licenses/GPL-3.0.txt`](licenses/GPL-3.0.txt) | [FFmpeg n7.1, COPYING.GPLv3](https://raw.githubusercontent.com/FFmpeg/FFmpeg/n7.1/COPYING.GPLv3) |
| [`licenses/LGPL-2.1.txt`](licenses/LGPL-2.1.txt) | [FFmpeg n7.1, COPYING.LGPLv2.1](https://raw.githubusercontent.com/FFmpeg/FFmpeg/n7.1/COPYING.LGPLv2.1) |
| [`licenses/DeepLabCut-NOTICE.yml`](licenses/DeepLabCut-NOTICE.yml) | Installed package: `deeplabcut==3.0.0`, `deeplabcut-3.0.0.dist-info/licenses/NOTICE.yml` |
| [`licenses/DLClibrary-NOTICE.yml`](licenses/DLClibrary-NOTICE.yml) | Installed package: `dlclibrary==0.0.12`, `dlclibrary-0.0.12.dist-info/NOTICE.yml` |

Package paths above are relative to `Lib/site-packages` in the HAGISILTA
application environment. The FFmpeg tag identifies the source of the license
documents; individual FFmpeg builds can have different license terms.
LGPL-3.0 incorporates GPL-3.0 with additional permissions, so both texts are
included.

File sources and SHA-256 digests are listed in
[`licenses/source-provenance.json`](licenses/source-provenance.json).

Keep each dependency package's own license and attribution files. When
redistributing dependencies, also meet the applicable source-provision and
other license conditions. These reference copies do not replace the package
files or obligations.
