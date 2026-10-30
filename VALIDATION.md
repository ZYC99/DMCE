# Validation record / 验证记录

Date / 日期: 2026-10-31 (Asia/Shanghai).

## Scope / 范围

The checks below establish that the independent entry points, stage dependencies,
checkpoint loading, and evaluation tools execute correctly. They do **not** establish
convergence or recover the paper's reported accuracy. The separate full-data,
100-epoch-per-stage run is now complete; its measured results are linked below.

以下检查确认独立入口、阶段衔接、权重加载和评测工具可以正常执行，**不证明模型收敛或恢复论文精度**。完整数据、每阶段 100 轮的正式实验已完成，实测结果单独记录于下方链接。

The original project directories were preserved. SHA-256 checks on 88 original
source/data-list files found no changes caused by the refactoring.

旧项目目录保留原样；对原有 88 个源码及数据列表文件的 SHA-256 核查未发现本次整理造成的改动。

## Completed full experiment / 已完成的正式实验

[English results](results/paper_seed42_20261031/README.md) · [中文结果](results/paper_seed42_20261031/README.zh-CN.md)

- All **26 scheduled jobs** exited successfully. All **10 training stages** contain epochs 1–100, with no batch limit: semantic training/validation used 1,568/392 entries each epoch; diffusion stages used 20,000/1,000 synthetic samples per epoch. Resolution was 480×640 and the diffusion schedule had T=1000.
- All **165 segmentation measurements** (5 methods × 3 splits × 11 SNRs) were recomputed from their 9×9 confusion matrices. Per-class IoU, nine-class/foreground mIoU and pixel accuracy matched within 1e−12. Pixel counts, ground-truth histograms and CSV/JSON values matched. Evaluation checkpoint paths and SHA-256 hashes matched the corresponding trained best checkpoints; their modification times preceded evaluation.
- Tensor comparisons confirmed frozen encoders and BatchNorm buffers in all five methods, updated decoder weights, and frozen diffusion state in all three denoising methods. All **11 training-module AST comparisons** with the immutable execution snapshot passed.
- The separately seeded CSI test completed **11 SNR points × 1,000 channels** with seed 200042 and the trained Stage 2 checkpoint. The initial NMSE curve agreed with −SNR within 0.0001 dB in this run.
- Fig. 5 exports use the first **3 day + 2 night** examples at **4 dB**, matching the paper's sample counts. All five comparison grids and all four curve figures were visually inspected. The qualitative, quantitative and latency outputs refer to the same selected checkpoints.
- Inference timing completed **20 method/SNR combinations × 10 repetitions**, after two warmups each, in FP32 at batch size 1. Synchronized timings, UNet calls, memory use and checkpoint hashes are retained. The complete result-report validation passed; the 14 regression tests also passed after the qualitative-export update.

26 个调度任务全部正常退出；10 个训练阶段均完成第 1–100 轮，没有 batch 截断。语义训练每轮使用 1,568 条训练数据和 392 条验证数据；扩散训练每轮使用 20,000/1,000 个合成训练/验证样本，分辨率为 480×640，扩散日程 T=1000。

165 个分割测量点均由 9×9 混淆矩阵独立重算，逐类 IoU、九类与前景 mIoU、像素准确率误差在 1e−12 内；像素计数、真实标签直方图、CSV/JSON 数值一致。评测权重路径与 SHA-256 对应各方法的最佳权重，文件更新时间早于评测启动。五种方法均通过编码器、BatchNorm、解码器和扩散冻结状态核对；11 个训练模块的 AST 与执行快照一致。

独立 CSI 测试使用种子 200042，在 11 个 SNR 下各评测 1,000 个信道；本次初始 NMSE 与 −SNR 的最大偏差小于 0.0001 dB。图 5 按论文数量固定选取前 3 张昼间、前 2 张夜间样本，SNR 为 4 dB；5 张对比图及 4 张曲线图均已目视检查。定量、定性和耗时结果使用相同的最佳权重。耗时覆盖 20 个方法/SNR 组合，每项预热 2 次、测量 10 次，精度为 FP32、batch size 为 1；完整报告核验及更新后的 14 项回归检查均通过。

These results describe one training seed and the corrected legacy channel model. They do not establish the paper's full MIMO curves, reported percentage gains, or multi-seed statistical reliability.

这些结果对应单个训练种子与修复后的原信道模型，不能据此宣称恢复论文完整 MIMO 曲线、其中的百分比增益或多种子统计可靠性。

## Environment / 环境

| Component / 组件 | Local CPU checks / 本地 | GPU checks / 服务器 |
|---|---|---|
| Python | 3.11.15 | 3.12.3 |
| PyTorch | 2.5.1 | 2.11.0a0+eb65b36914.nv26.02 |
| torchvision | 0.20.1 | 0.25.0a0+1e53952f.nv26.02.44259020 |
| NumPy / Pillow | 1.26.4 / 11.0.0 | 2.1.0 / 12.1.1 |
| Accelerator | CPU | NVIDIA RTX PRO 5000 72GB Blackwell |

The working CUDA environment was reused. ImageNet ResNet50 V1 initialization was
verified against the hash prefix in the official checkpoint filename.

复用服务器已有 CUDA 环境；ImageNet ResNet50 V1 文件已按官方文件名中的哈希前缀校验。

## Completed checks / 已完成检查

- **10 regression tests**, passing on CPU and CUDA-capable server environments:
  CLI/config isolation, independent working directories, IR signal use, per-example
  power normalization, diffusion timing/zero-noise behavior, four-channel and label
  resizing, virtual flips, background-aware IoU, BatchNorm freezing, checkpoint
  resume and rejection of mismatched diffusion targets/schedules.
- **16 synthetic integration subprocesses**, all passing: shared Stage 1, all five
  downstream pipelines, five SNR sweeps, CSI NMSE, and resuming all three stages.
  Inputs are 64×64 with a reduced diffusion network; temporary random checkpoints
  are removed by the test script.
- **12 bounded real-data subprocesses**, all passing: official four-channel PNGs,
  ImageNet initialization, 480×640 input, T=1000, full default UNet. Each stage uses
  one epoch and at most one batch of two examples; SNR evaluations cover 0 and 20 dB.
- **Freeze audit** on bounded real-data checkpoints: all 657 encoder state tensors
  in each ResNet method and all 238 MFNet encoder tensors remained unchanged during
  Stage 3, including BatchNorm buffers. All five decoders updated. Each Stage 3
  diffusion state exactly matched its Stage 2 initialization.
- **26 scheduler jobs**, all passing on a deliberately small fixture: all ten
  training stages, all/day/night evaluation for five methods, and CSI NMSE. This
  exercises dependency scheduling and shared GPU locks without claiming accuracy.
- **Post-processing checks**, passing on the bounded suite: complete-result
  collection, JSON/CSV and PNG/PDF plots, fixed day/night qualitative exports, and
  latency/UNet-call reporting. The collector correctly labels this as a custom or
  bounded run, not the full MFNet protocol.
- **Publication-report checks**, passing: English/Chinese tables are generated from
  the measured artifacts; default reporting rejects bounded output and mismatched
  latency checkpoint hashes. Qualitative grids were visually checked for readable
  captions and class legends. These previews remain bounded validation output.

上述检查分别覆盖 10 项回归、16 个合成数据集成子进程、12 个真实数据短测子进程、冻结状态逐张量核对、26 个调度任务，以及结果汇总、定性图和耗时统计。全部通过。短测只验证运行链路，不作为论文精度或速度结论。

The bounded real-data fixture uses disjoint official splits:

| Split / 划分 | Sample IDs / 样本 |
|---|---|
| Train | 00001D, 00002D |
| Validation | 00498D, 00499D |
| Test | 01402D, 01403D |

## Public dataset audit / 公开数据完整性

Source: [MFNet official project](https://www.mi.t.u-tokyo.ac.jp/projects/mil_multispectral/index.html)
and its linked original semantic-segmentation folder. All **3,182/3,182** manifest
files passed official byte-size and MD5 verification; zero failures. All PNG files
also passed decoding/integrity checks.

清单内 **3,182 / 3,182** 个文件均通过官方大小与 MD5 校验，失败数为 0；所有 PNG 也通过完整性检查。

| Property / 项目 | Verified value / 核查值 |
|---|---|
| Images / 图像 | 1,606, RGBA 640×480; fourth channel is IR |
| Labels / 标签 | 1,569, L 640×480 |
| Text files / 文本 | 7 |
| Total bytes / 总字节数 | 739,201,315 |
| Training list / 训练列表 | 1,568 entries including flips; 784 original scenes |
| Validation / 验证 | 392 original scenes |
| Test / 测试 | 393 = 205 day + 188 night |
| Original-scene overlap / 原始场景重叠 | None among train/validation/test |

The 37 unlabelled extra images are not inserted into any split. Public file IDs,
resource keys, sizes, and MD5s are stored in `configs/mfnet_download_manifest.json`;
there are no account credentials in this manifest.

额外 37 张未标注图片没有加入任何划分。公开文件清单只记录文件定位与校验信息，不包含账号凭证。

## Repeat checks / 复查

```bash
python -m unittest discover -s tests -v
python tests/validate_pipeline.py --device cuda:0 --output-dir validation/pipeline_repeat

# Provide a valid train/val/test fixture in this data directory.
# This remains a bounded check: one epoch, one batch per pass.
python tests/validate_real_data.py --device cuda:0 \
  --data-dir datasets/ir_seg_dataset --output-dir validation/real_data_repeat
```

Use a new output directory to retain earlier records. The test scripts write their
commands, exit codes, and elapsed times to local JSON and logs. Private absolute
paths and large debug checkpoints are excluded from the public Git repository.

使用新的输出目录以保留先前记录。测试脚本将实际命令、退出码和耗时写入本地 JSON 与日志；含私有绝对路径的原始日志和大型调试权重不提交到公开 Git 仓库。

## Synthetic CSI random-stream audit / 合成 CSI 随机流核对

A CUDA probe confirmed that reusing training seed 42 in NMSE evaluation repeated the first 64 synthetic training channels. The suite now uses training seed + 200000 for CSI testing; diffusion validation uses +100000. Full reports reject reused train/validation seeds and verify the CSI checkpoint hash. Training was unchanged; the full NMSE evaluation was repeated separately with seed 200042, preserving the original output. The published report uses the separately seeded evaluation.

CUDA 核对确认，NMSE 评测复用训练种子 42 会重复首批 64 个训练信道。调度器现改用训练种子 +200000 作为 CSI 测试种子，扩散验证使用 +100000。完整报告拒绝复用训练或验证种子，并核对 CSI 权重哈希。训练保持不变，已另以种子 200042 完成全量 NMSE 重测，保留原始输出；发布报告采用独立随机流的结果。

Four new regression checks passed: default/custom training seeds produce separate CSI test streams, reused seeds are rejected in full reports, and mismatched CSI checkpoint hashes are rejected. A bounded end-to-end run with test seed 200042 also generated both reports successfully.

新增 4 项回归检查通过：默认与自定义训练种子均产生独立 CSI 测试流；完整报告拒绝复用种子；CSI 权重哈希不符时拒绝汇总。另以测试种子 200042 完成一次短测及中英文报告生成验证。
