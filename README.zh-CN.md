# DMCE — ICC 2024

[English](README.md) | [简体中文](README.zh-CN.md)

本仓库提供 **IEEE ICC 2024** 论文 **DMCE: Diffusion Model Channel Enhancer for Multi-User Semantic Communication Systems** 的研究代码，论文页码为 855–860。

作者：Youcheng Zeng、Xinxin He、Xu Chen、Haonan Tong、Zhaohui Yang、Yijun Guo、Jianjun Hao。

[论文](https://arxiv.org/abs/2401.16017) · [IEEE DOI](https://doi.org/10.1109/ICC51166.2024.10622730) · [引用信息](CITATION.cff)

本版本由作者原有代码整理而来，将依赖手工注释切换的实验拆成独立入口和配置。每个入口可以独立运行，不导入旧项目目录；原工作目录另行保留。传统 JPEG / LDPC / 4-QAM / DeepLabv3 数字传输链路不在本仓库的复现范围内。

**复现范围：**这是修复后的原实现，仍采用简化的共享逐元素衰落信道，没有完整实现论文的多用户 MIMO 混合。扩散训练使用 DDPM 噪声预测目标，采样采用确定性逐步更新。本实现的实测结果不能直接称为论文曲线的精确复现；比较数值前请阅读下文的实现差异。

## 独立实验入口

| 入口 | 降噪对象 | 论文 / 原代码对应 |
|---|---|---|
| `experiments/dmce.py` | 均衡前增强 CSI H | Fig. 3 主方法；`R50withDM.py` |
| `experiments/without_dmce.py` | 不降噪，用初始 CSI 均衡 | Fig. 3 without DMCE；`R50withoutDM.py` |
| `experiments/y_denoising.py` | 对接收信号 Y 降噪，学习干净 HX | Fig. 3 Y-oriented；`R50withDMHX.py` |
| `experiments/x_denoising.py` | 对均衡后的 X 估计降噪 | 原代码扩展，**不是 Fig. 3 的独立曲线**；`R50withDMX.py` |
| `experiments/mfnet_channel.py` | MFNet 多尺度特征过信道 | Fig. 3 MFNet with channel；`MFwithChannel.py` |

每组实验在 `configs/` 中有独立 JSON，共用默认值放在 `configs/base.json`，不需要再手工注释代码。

## 安装与公开 MFNet 数据

建议使用 Python 3.11 或 3.12，以及相互兼容的 PyTorch / torchvision。GPU 服务器优先复用已经能运行的 CUDA 环境，不要为了匹配旧版本号而将支持 Blackwell 的环境降级。

```bash
git clone https://github.com/ZYC99/DMCE.git
cd DMCE
python -m pip install -r requirements.txt
# 可选：结果汇总及科研绘图
python -m pip install -r requirements-analysis.txt
python scripts/download_mfnet.py --destination datasets/ir_seg_dataset
```

MFNet 数据公开可用：[官方项目页](https://www.mi.t.u-tokyo.ac.jp/projects/mil_multispectral/index.html)、[原始语义分割数据目录](https://drive.google.com/drive/folders/18BQFWRfhXzSuMloUmtiBRFrr6NSrf8Fw)。下载脚本使用公开链接，核对官方文件大小与 MD5，跳过已经验证的文件以支持续传，并支持 `HTTPS_PROXY`；不需要 Google 账号 token 或 cookie。清单共有 3,182 个文件：1,606 张图像、1,569 张标签、7 个文本文件，总计 739,201,315 字节。额外 37 张没有标签的图像不会被擅自加入数据划分。

```text
datasets/ir_seg_dataset/
  images/00001D.png      # RGB + 热红外，四个独立测量通道
  labels/00001D.png     # 整数语义类别标签
  train.txt
  val.txt
  test.txt
  test_day.txt
  test_night.txt
```

官方训练列表有 1,568 项（含翻转），对应 784 个原始场景；验证集 392 个，测试集 393 个，其中白天 205、夜间 188。三个原始场景划分互不重叠。如果没有预先生成 `_flip.png`，加载器会从原图和标签动态翻转。第四通道是 IR，不是透明度，不能丢弃；标签采用最近邻缩放。

## 单独运行实验

```bash
# DMCE 完整三阶段训练，默认每阶段 100 轮
python experiments/dmce.py --data-dir datasets/ir_seg_dataset --device cuda:0

# 以下入口也都可以单独运行
python experiments/without_dmce.py --data-dir datasets/ir_seg_dataset --device cuda:0
python experiments/y_denoising.py --data-dir datasets/ir_seg_dataset --device cuda:0
python experiments/x_denoising.py --data-dir datasets/ir_seg_dataset --device cuda:0
python experiments/mfnet_channel.py --data-dir datasets/ir_seg_dataset --device cuda:0
```

`--dry-run` 只打印最终配置，不导入 torch、不下载权重、不生成训练文件。`--stage smoke` 使用 64×64 合成输入及缩小的 UNet 检查前向和反向传播，不能用作精度实验。

```bash
python experiments/dmce.py --dry-run
python experiments/dmce.py --stage smoke --device cuda:0
```

配置文件内的相对路径以本仓库为基准；命令行相对路径以当前工作目录为基准。入口也可以从其他目录通过绝对路径调用。默认输出位于 `runs/<实验名>/`。

## 三阶段训练与共享初始化

| 阶段 | 更新模块 | 数据 / 权重依赖 |
|---|---|---|
| `stage1` | 干净信道下的语义与 JSCC 编解码器 | MFNet 图像和标签；ResNet50 使用 ImageNet V1 初始化 |
| `stage2` | 扩散噪声预测器 | 在线合成 H / HX / X，与图像训练独立 |
| `stage3` | 仅接收端解码器 | Stage 1 和对应 Stage 2 的最优权重；冻结编码器、其 BatchNorm 统计及扩散模型 |

Without-DMCE 和 MFNet 跳过 Stage 2。四个 ResNet 实验可以共享同一个干净 Stage 1，保证消融使用一致初始化；MFNet 使用自己的骨干与初始化。

```bash
python experiments/without_dmce.py --stage stage1 --device cuda:0
python experiments/dmce.py --stage all --device cuda:0 \
  --stage1-checkpoint runs/without_dmce/stage1/best.pt
python experiments/y_denoising.py --stage all --device cuda:0 \
  --stage1-checkpoint runs/without_dmce/stage1/best.pt
```

也可以逐阶段运行：

```bash
python experiments/dmce.py --stage stage2 --device cuda:0
python experiments/dmce.py --stage stage3 --device cuda:0 \
  --stage1-checkpoint runs/without_dmce/stage1/best.pt \
  --diffusion-checkpoint runs/dmce/stage2/best.pt
```

`--stage all` 收到相应初始化权重后会跳过 Stage 1 或 Stage 2。每个阶段保存 `config.json`、`metrics.jsonl`、`best.pt`、`last.pt`。语义阶段按验证集九类 mIoU 选择 best，扩散阶段按验证集噪声预测 MSE 选择 best。权重同时保存优化器和随机数状态；已有实验受覆盖保护。

```bash
python experiments/dmce.py --stage stage3 --device cuda:0 \
  --resume runs/dmce/stage3/last.pt --epochs 120
```

`--epochs` 表示目标总轮数，不是额外训练轮数。重新开始实验请换一个 `--output-dir`。旧权重需要显式导入并通过形状检查；新扩散权重还会核对训练目标与扩散配置。修复后的版本建议重新训练。

## 运行完整实验集

Linux 下的调度脚本执行 10 个训练阶段和 16 个评测任务：五种方法的完整测试集、昼间与夜间 SNR 扫描，以及 CSI NMSE。脚本复用共享 ResNet Stage 1，在多张 GPU 间按依赖安排任务，记录退出码，并保存不可变代码快照与哈希。它使用全部 batch 和配置中的轮数，没有隐藏的短测限制。

```bash
python scripts/run_suite.py \
  --data-dir datasets/ir_seg_dataset \
  --output-dir runs/paper_seed42 --gpus 0 1
```

长实验请放在 tmux 或其他持久进程管理器中。共享服务器应通过 `--gpu-lock-template '/shared/locks/gpu{gpu}.lock'` 接入已有的 GPU 锁约定。调度器还会检查 GPU 上是否已有计算进程，不会终止其他任务。状态和日志位于 `runs/paper_seed42/status.json` 与 `logs/`，数值输出位于其中的 `runs/` 子目录。

中断后使用相同路径、配置并添加 `--resume`。调度器检查原子保存的 checkpoint，决定哪些阶段需要继续训练；不要为同一输出目录启动第二个控制进程。

也可同时启动结果整理进程，让它等待训练完成后核对权重、生成图表、在空闲 GPU 上测量推理速度，并生成中英文结果报告。报告会核对指标、定性图和耗时测试使用同一组权重。共享服务器应为它指定与训练相同的 GPU 锁模板：

```bash
python scripts/finish_suite.py --suite-dir runs/paper_seed42 \
  --data-dir datasets/ir_seg_dataset --output-dir results/paper_seed42 \
  --gpus 0 1 --wait
```

## 评测、定性图与速度测量

```bash
# Fig. 3 对应任务：全部 11 个 SNR 点；也支持 --split test_day / test_night
python experiments/dmce.py --stage eval --device cuda:0 \
  --checkpoint runs/dmce/stage3/best.pt --snrs 0 2 4 6 8 10 12 14 16 18 20

# Fig. 4 对应任务：初始 CSI 与增强后 CSI 的 NMSE
python experiments/dmce.py --stage nmse --device cuda:0 \
  --checkpoint runs/dmce/stage2/best.pt

# 完整实验结束后，导出可分享的 JSON/CSV 与 PNG/PDF 曲线
python scripts/collect_results.py --suite-dir runs/paper_seed42 --output-dir results/paper_seed42

# Fig. 5 对应任务：固定选取官方昼间和夜间列表各前两张，SNR=4 dB
python scripts/export_qualitative.py --run-dir runs/paper_seed42/runs \
  --data-dir datasets/ir_seg_dataset --output-dir results/paper_seed42/qualitative --device cuda:0

# 全模型 FP32 推理：同步计时、预热、UNet 调用次数与显存峰值
python scripts/benchmark_inference.py --run-dir runs/paper_seed42/runs \
  --data-dir datasets/ir_seg_dataset --output results/paper_seed42/latency.json \
  --device cuda:0 --batch-size 1 --warmup 2 --repeats 10

# 根据实测数据和权重哈希生成中英文结果报告
python scripts/write_results_summary.py --results-dir results/paper_seed42
```

主指标 `miou` 使用固定九类分母，完全缺失的类别记 0；`miou_foreground` 另外报告八个前景类。两者都使用完整混淆矩阵，保留背景像素上的误检。JSON/CSV 数值为 [0,1] 内的比例，图中显示百分比。CSI NMSE 使用总误差功率除以总真实信道功率，再转为 dB。每个 SNR 评测开始时重置随机种子。

完整测试集、昼间与夜间分别执行评测，因此信道抽样和 batch 平均噪声功率可能不同，不能把昼夜混淆矩阵相加来还原完整测试矩阵。结果报告默认拒绝短测或自定义小规模实验；`--allow-bounded` 仅用于调试，会在两种语言的报告中醒目标明短测属性。

耗时不包含数据读取与权重加载，包含编码器、信道、降噪器和解码器。必须同时报告 batch size 与 UNet 调用次数：逆扩散起点依赖 SNR，`T=1000` 不表示每个 SNR 都执行 1,000 次网络计算。

**当前执行状态：**完整 100 轮实验正在运行，全部训练与评测结束后再补入最终数值。此前的短测见 [VALIDATION.md](VALIDATION.md)，不作为论文精度结果。

## 默认配置及其来源

| 参数 | 默认值 | 来源 |
|---|---|---|
| 图像 / 类别 | 480×640 / 9 类 | 论文 IV-A |
| 扩散日程 | T=1000；β 线性从 1e-4 到 0.02 | 论文 IV-A |
| Stage 1 / Stage 3 | 100 轮；SGD，lr=.01，每轮乘 .95 | 论文 IV-A |
| Stage 2 | 100 轮；Adam，lr=1e-4，线性衰减 | 轮数来自论文；优化器与学习率沿用旧脚本 |
| 语义 batch / workers / seed | 8 / 4 / 42 | 工程默认，论文未完整说明 |
| Stage 2 样本 | 每轮训练 20,000 / 验证 1,000，batch 64 | 基于旧脚本的在线合成流程 |
| 训练 SNR | 20 dB | 沿用旧微调脚本，论文未精确指定 |
| 测试 SNR | 0, 2, …, 20 dB | Fig. 3–4 |

H 训练同时使用实部与虚部，共享单通道 UNet；Y/HX 使用双通道；X 使用单通道。`--config overrides.json` 递归合并配置，未知参数会报错；骨干及降噪目标由入口固定。`--epochs`、`--batch-size`、`--workers`、`--seed`、`--train-snr`、`--snrs` 可覆盖常用值；扩散 batch size 单独使用 JSON 中的 `stage2.batch_size`。

## 修复内容与论文差异

1. 原实现是两路实数特征共用的逐元素衰落信道，填充为 70×70 网格，没有完整实现式 (4) 的 MIMO 混合、阵列响应及多用户检测。三条合成路径仍共享时延。距离改用浮点数以避免 int8 溢出；每个样本单独抽取 H，不再整个 batch 共用一次随机 H。噪声功率仍根据 batch 平均接收功率计算。
2. 修复 IR 分支误用 RGB 接收特征；MFNet 所有尺度及跳接均实际使用过信道后的特征。MFNet 与 ResNet 传输的特征数量不同，本对比没有统一信道使用次数预算。
3. 功率按样本归一化，并保护零范数。统一复高斯噪声为 E|Z|²=σ²，实部与虚部分别为 σ²/2，CSI 噪声同样遵守该约定。扩散日程补齐 ᾱ₀=1，统一 t/T 条件，正确缩放起始观测，并处理零噪声边界。
4. 原代码的确定性逆更新在代数上属于 η=0 的 DDIM 式相邻时间步更新，训练目标为 DDPM 噪声预测。它不是带随机后验采样的 ancestral DDPM，也尚未跳步。观测条件通过逆扩散起点进入，没有独立的观测或导频条件编码器。
5. Stage 2 的 X/HX 仍来自高斯合成数据，没有使用真实 JSCC 特征。X 扩展的均衡后噪声并非独立同分布高斯，使用标量方差是近似。解释结果时需要考虑这种分布不匹配。
6. Stage 3 同时冻结编码器参数和 BatchNorm 统计，并采用论文中的 SGD。RGB 与 IR 分别缩放，避免 RGBA 对 RGB 的隐式透明度预乘；标签使用最近邻。指标统计补回旧函数忽略的背景相关误检。

这些修复会改变与旧日志、旧权重的对应关系，需要重新训练；修复完成本身不证明已经恢复论文报告的数值增益。

## 后续改进方向

以下是研究计划，**不是本版本已经实现的提速或已经测得的收益**。

- **优先减少采样步数。**尝试使用重排时间步的 [DDIM](https://arxiv.org/abs/2010.02502)，对比 10 / 20 / 50 / 100 步，并从与观测噪声匹配的位置启动。保持 t/T 条件及端点缩放一致，同时记录九类 mIoU、CSI NMSE、UNet 调用次数与同步计时。当前更新已经具有 η=0 DDIM 的形式，仅修改名称不会加速；如需严格比较 DDPM 与 DDIM，还应单独实现随机 ancestral DDPM 基线。
- **比较高阶求解器与蒸馏。**可评估 [DPM-Solver](https://arxiv.org/abs/2206.00927) 等 ODE 求解器，再参考 [Consistency Models](https://proceedings.mlr.press/v202/song23a.html) 将已训练增强器蒸馏为一步或少步网络。这些方法在图像生成中的提速不能直接照搬为 CSI 增强的结论，需要重新标定观测噪声并检查低 SNR 稳定性。
- **补齐物理模型。**引入显式多用户 MIMO 矩阵混合、天线几何、用户独立信道、基于导频的 CSI 估计，以及 MMSE 或学习式检测。在讨论通信效率前，统一各基线的信道使用次数和发射功率预算。
- **学习真实语义特征分布。**缓存冻结后的 Stage 1 JSCC 特征，供 Y/X 扩散训练使用；联合建模实部与虚部，并显式输入导频、初始 CSI 和 SNR。与当前高斯合成训练、直接 CNN/UNet 降噪器进行对照。
- **提高效率和泛化能力。**测量轻量 UNet、混合精度、编译优化、按 SNR 自适应停止及蒸馏。测试训练范围外的 SNR、时延扩展、多普勒、CSI 误差和用户数量，避免只在训练信道分布内评测。
- **改善分割与统计可靠性。**研究面向少数类别的损失函数，报告昼夜及逐类 IoU，至少采用三个随机种子和配对信道随机抽样。单种子实验无法刻画不同训练运行之间的不确定性。

## 测试与来源

```bash
python -m unittest discover -s tests -v
python tests/validate_pipeline.py --device cuda:0 --output-dir validation/pipeline
```

测试覆盖入口隔离、实部/虚部与 IR 信号处理、归一化、扩散时间索引、四通道预处理、虚拟翻转、包含背景误检的 IoU、冻结 BatchNorm，以及权重兼容和续训。真实数据短测有明确标识，不能代替完整训练；详情见 [VALIDATION.md](VALIDATION.md)。

MFNet 骨干与适配后的扩散 UNet 保留上游归属，参见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 及附带 Apache 2.0 文本。Git 不包含数据集文件、私有服务器路径、访问凭证或训练权重。

## 引用

```bibtex
@inproceedings{zeng2024dmce,
  author = {Youcheng Zeng and Xinxin He and Xu Chen and Haonan Tong and
            Zhaohui Yang and Yijun Guo and Jianjun Hao},
  title = {DMCE: Diffusion Model Channel Enhancer for Multi-User Semantic Communication Systems},
  booktitle = {ICC 2024 - IEEE International Conference on Communications},
  year = {2024},
  pages = {855--860},
  doi = {10.1109/ICC51166.2024.10622730}
}
```
