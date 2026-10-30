# Validation record / 验证记录

Date / 日期: 2026-10-31 (Asia/Shanghai).

## Scope / 范围

The checks below establish that the independent entry points, stage dependencies,
checkpoint loading, and evaluation tools execute correctly. They do **not** establish
convergence or recover the paper's reported accuracy. A separate full-data,
100-epoch-per-stage run is in progress and must be reported separately.

以下检查确认独立入口、阶段衔接、权重加载和评测工具可以正常执行，**不证明模型收敛或恢复论文精度**。完整数据、每阶段 100 轮的正式实验另行运行与记录。

The original project directories were preserved. SHA-256 checks on 88 original
source/data-list files found no changes caused by the refactoring.

旧项目目录保留原样；对原有 88 个源码及数据列表文件的 SHA-256 核查未发现本次整理造成的改动。

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
