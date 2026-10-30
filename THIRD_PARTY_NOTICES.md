# Third-party sources / 第三方来源

This repository reorganizes the author's existing DMCE research code. Original
third-party rights are retained. No repository-wide license is declared that
overrides the separate terms below. The dataset and ImageNet weights are obtained
from their original sources and are not included in this Git repository.

本仓库整理作者已有的 DMCE 研究代码，保留第三方来源及其权利；不以统一许可证覆盖下列组件的原有条款。数据集和 ImageNet 权重从原始来源获取，不随 Git 仓库分发。

## MFNet

`dmce/mfnet_backbone.py` was copied from the author's existing
`MFNet/model/MFNet.py`, originating in
[haqishen/MFNet-pytorch](https://github.com/haqishen/MFNet-pytorch).
The local upstream history ends at commit
`fff9e28ca47adb1491ea3c6c29df958da91da092`.
The channel adapter in `dmce/models.py` retains this architecture and fixes the
old channel integration. The public upstream repository did not expose a license
file when checked on 2026-10-31; this release does not assign a new license to that
upstream implementation.

MFNet 骨干来自原代码中的 `MFNet/model/MFNet.py`，上游为上述仓库。信道适配层修复了多尺度特征实际过信道的路径。核查时上游未提供许可证文件，本仓库不为该上游实现另行授予许可证。

Paper: Qishen Ha, Kohei Watanabe, Takumi Karasawa, Yoshitaka Ushiku, Tatsuya Harada,
“MFNet: Towards Real-Time Semantic Segmentation for Autonomous Vehicles with
Multi-Spectral Scenes,” IROS 2017.
[Official project and dataset](https://www.mi.t.u-tokyo.ac.jp/projects/mil_multispectral/index.html).

## Diffusion UNet and positional encoding

`dmce/unet.py` retains the UNet supplied with the original DMCE code. Its
architecture and helper blocks derive from
[Janspiry/Image-Super-Resolution-via-Iterative-Refinement](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/blob/master/model/sr3_modules/unet.py),
distributed under Apache License 2.0. The inherited DMCE adaptations include an
MLP conditioning variant and channel-grid handling. The positional encoding
acknowledges [lmnt-com/wavegrad](https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py),
also under Apache License 2.0, with the original notice:

> Copyright 2020 LMNT, Inc. All Rights Reserved.

The applicable [Apache License 2.0 text](licenses/Apache-2.0.txt) is included.
The refactoring keeps these adapted network blocks, while the schedule, training,
channel integration, and experiment entry points are implemented separately.

UNet 保留原 DMCE 代码中的网络块，其结构与辅助模块来自上述 SR3 实现；位置编码引用 WaveGrad。原代码对条件 MLP、信道网格等已有适配。两项上游均采用 Apache 2.0，完整许可证随仓库提供；本次重构将扩散日程、训练、信道接入与独立实验入口拆开实现。

## Libraries and data

PyTorch, torchvision, NumPy, Pillow, and matplotlib retain their respective
licenses. Please cite the MFNet dataset paper when using its data. Public download
URLs and Drive resource keys in the manifest identify publicly shared files;
they are not account credentials and do not change the dataset's terms.

依赖库遵循各自许可证；使用 MFNet 数据请引用其论文。下载清单中的公开文件链接及 resource key 仅用于定位官方共享文件，不是账号凭证，也不改变数据集的使用条款。
