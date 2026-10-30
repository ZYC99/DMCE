# DMCE — ICC 2024

[English](README.md) | [简体中文](README.zh-CN.md)

Research code for **DMCE: Diffusion Model Channel Enhancer for Multi-User Semantic Communication Systems**, published at **IEEE ICC 2024**, pp. 855–860.

Youcheng Zeng, Xinxin He, Xu Chen, Haonan Tong, Zhaohui Yang, Yijun Guo, Jianjun Hao.

[Paper](https://arxiv.org/abs/2401.16017) · [IEEE DOI](https://doi.org/10.1109/ICC51166.2024.10622730) · [Citation](CITATION.cff)

This release reorganizes the author's existing research code into independent, configurable experiments. Each entry point runs without importing the old project. The original working directories were preserved separately. Traditional JPEG/LDPC/4-QAM/DeepLabv3 transmission is outside this release's scope.

**Reproduction scope:** this is the corrected legacy implementation. It retains a simplified shared elementwise fading channel rather than the paper's full multi-user MIMO mixing model. The sampler is deterministic and adjacent-step, with DDPM noise-prediction training. Results from this implementation must not be presented as an exact reproduction of the published curves. See [implementation differences](#implementation-differences) before making numerical comparisons.

## Independent experiments

| Entry point | Denoising target | Paper / legacy mapping |
|---|---|---|
| `experiments/dmce.py` | CSI H, before equalization | DMCE, Fig. 3; `R50withDM.py` |
| `experiments/without_dmce.py` | None; equalize using initial CSI | Without DMCE, Fig. 3; `R50withoutDM.py` |
| `experiments/y_denoising.py` | Received Y, learning clean HX | Y-oriented, Fig. 3; `R50withDMHX.py` |
| `experiments/x_denoising.py` | Equalized X estimate | Legacy extension; **not a separate Fig. 3 curve**; `R50withDMX.py` |
| `experiments/mfnet_channel.py` | MFNet multiscale features through the channel | MFNet with channel, Fig. 3; `MFwithChannel.py` |

Each entry has its own JSON in `configs/`. Shared defaults live in `configs/base.json`; no manual commenting is needed.

## Setup and public MFNet data

Use Python 3.11 or 3.12 and a compatible PyTorch/torchvision pair. On a GPU server, reuse its working CUDA environment; do not replace a Blackwell-compatible build with an older wheel just to match a version number.

```bash
git clone https://github.com/ZYC99/DMCE.git
cd DMCE
python -m pip install -r requirements.txt
# Optional: scientific figures / result collection
python -m pip install -r requirements-analysis.txt
python scripts/download_mfnet.py --destination datasets/ir_seg_dataset
```

MFNet is public: [official project](https://www.mi.t.u-tokyo.ac.jp/projects/mil_multispectral/index.html), [original segmentation data](https://drive.google.com/drive/folders/18BQFWRfhXzSuMloUmtiBRFrr6NSrf8Fw). The downloader uses public links, verifies official file sizes and MD5 hashes, resumes by skipping verified files, and supports `HTTPS_PROXY`. It needs no Google account token or cookie. The manifest contains 3,182 files: 1,606 images, 1,569 labels, and 7 text files, totaling 739,201,315 bytes. The 37 extra images without labels are not added to the official splits.

```text
datasets/ir_seg_dataset/
  images/00001D.png      # RGB + thermal IR, four independent channels
  labels/00001D.png     # integer semantic class IDs
  train.txt
  val.txt
  test.txt
  test_day.txt
  test_night.txt
```

The official training list has 1,568 entries including flips, representing 784 original scenes; validation has 392 scenes and test has 393 (205 day, 188 night). These original-scene splits are disjoint. Missing physical `_flip.png` files are generated virtually during loading. Keep the fourth image channel: it is IR, not transparency. Labels use nearest-neighbor resizing.

## Run an experiment

```bash
# Full three-stage DMCE training, default 100 epochs per stage
python experiments/dmce.py --data-dir datasets/ir_seg_dataset --device cuda:0

# Each of these is independently runnable
python experiments/without_dmce.py --data-dir datasets/ir_seg_dataset --device cuda:0
python experiments/y_denoising.py --data-dir datasets/ir_seg_dataset --device cuda:0
python experiments/x_denoising.py --data-dir datasets/ir_seg_dataset --device cuda:0
python experiments/mfnet_channel.py --data-dir datasets/ir_seg_dataset --device cuda:0
```

`--dry-run` prints the resolved configuration without torch imports, downloads, or output files. `--stage smoke` uses synthetic 64×64 inputs and a small UNet to check forward/backward execution; it is not an accuracy experiment.

```bash
python experiments/dmce.py --dry-run
python experiments/dmce.py --stage smoke --device cuda:0
```

Configuration paths are relative to this repository; command-line paths are relative to the current working directory. Entry scripts also work when invoked by absolute path from another directory. Results default to `runs/<experiment>/`.

## Training protocol and shared initialization

| Stage | Updated modules | Data / checkpoint dependency |
|---|---|---|
| `stage1` | Semantic + JSCC encoders and decoders, clean channel | MFNet images and labels; ImageNet ResNet50 V1 initialization |
| `stage2` | Diffusion noise predictor only | Online synthetic H / HX / X; independent of image training |
| `stage3` | Receiver decoders only | Stage 1 and corresponding Stage 2 best checkpoints; encoders, their BatchNorm statistics, and diffusion remain frozen |

Without-DMCE and MFNet skip Stage 2. All four ResNet experiments can share the same clean Stage 1 checkpoint. MFNet uses its own backbone and initialization.

```bash
python experiments/without_dmce.py --stage stage1 --device cuda:0
python experiments/dmce.py --stage all --device cuda:0 \
  --stage1-checkpoint runs/without_dmce/stage1/best.pt
python experiments/y_denoising.py --stage all --device cuda:0 \
  --stage1-checkpoint runs/without_dmce/stage1/best.pt
```

For explicit stages:

```bash
python experiments/dmce.py --stage stage2 --device cuda:0
python experiments/dmce.py --stage stage3 --device cuda:0 \
  --stage1-checkpoint runs/without_dmce/stage1/best.pt \
  --diffusion-checkpoint runs/dmce/stage2/best.pt
```

`--stage all` skips Stage 1 or Stage 2 when its initialization checkpoint is supplied. Every stage saves `config.json`, `metrics.jsonl`, `best.pt`, and `last.pt`. Best selection uses validation nine-class mIoU for semantic stages and validation noise-prediction MSE for diffusion. Checkpoints preserve optimizer and RNG state. Existing runs are protected against accidental overwrite.

```bash
python experiments/dmce.py --stage stage3 --device cuda:0 \
  --resume runs/dmce/stage3/last.pt --epochs 120
```

`--epochs` is the total target epoch count, not the number of additional epochs. Use a new `--output-dir` for a fresh run. Legacy weights require explicit loading and shape checks; new diffusion checkpoints also validate the target and diffusion configuration. Retraining is recommended after the corrections below.

## Run the complete suite

On Linux, the suite scheduler runs 10 training stages and 16 evaluation jobs: all five methods, all/day/night segmentation sweeps, and CSI NMSE. It reuses one clean ResNet checkpoint, schedules dependencies across GPUs, records exit codes, and saves an immutable source snapshot and hashes. It uses every batch and configured epoch; it does not apply a hidden debug limit.

```bash
python scripts/run_suite.py \
  --data-dir datasets/ir_seg_dataset \
  --output-dir runs/paper_seed42 --gpus 0 1
```

Run this under tmux or another durable process manager. On a shared server, pass its established lock convention, for example `--gpu-lock-template '/shared/locks/gpu{gpu}.lock'`. The scheduler also checks whether the GPU already has compute processes. It does not terminate other workloads. State and logs live in `runs/paper_seed42/status.json` and `logs/`; numerical outputs are in its nested `runs/` directory.

Resume an interrupted suite with the same paths/configuration and `--resume`. The scheduler checks atomic checkpoints before deciding which stages need further training. Do not launch a second controller for the same output directory.

The optional post-processing controller can wait alongside training and then audit checkpoints, export figures, and benchmark on an idle GPU. Use the same GPU lock template as training on shared servers:

```bash
python scripts/finish_suite.py --suite-dir runs/paper_seed42 \
  --data-dir datasets/ir_seg_dataset --output-dir results/paper_seed42 \
  --gpus 0 1 --wait
```

## Evaluate, visualize, and measure speed

```bash
# Fig. 3 task: all 11 SNR points; also supports --split test_day / test_night
python experiments/dmce.py --stage eval --device cuda:0 \
  --checkpoint runs/dmce/stage3/best.pt --snrs 0 2 4 6 8 10 12 14 16 18 20

# Fig. 4 task: initial and enhanced CSI NMSE
python experiments/dmce.py --stage nmse --device cuda:0 \
  --checkpoint runs/dmce/stage2/best.pt

# After the complete suite, create portable JSON/CSV and PNG/PDF plots
python scripts/collect_results.py --suite-dir runs/paper_seed42 --output-dir results/paper_seed42

# Fig. 5 task: first two official day and night examples, fixed selection, 4 dB
python scripts/export_qualitative.py --run-dir runs/paper_seed42/runs \
  --data-dir datasets/ir_seg_dataset --output-dir results/paper_seed42/qualitative --device cuda:0

# Full-model FP32 inference; synchronized timing, warmup, UNet calls, peak memory
python scripts/benchmark_inference.py --run-dir runs/paper_seed42/runs \
  --data-dir datasets/ir_seg_dataset --output results/paper_seed42/latency.json \
  --device cuda:0 --batch-size 1 --warmup 2 --repeats 10
```

The main `miou` averages all nine classes with a fixed denominator; absent classes contribute zero. `miou_foreground` reports the eight foreground classes separately. Both use the full confusion matrix, including false positives on background. JSON/CSV values are fractions in [0,1]; plots show percentages. CSI NMSE is total squared error divided by total true channel power, then converted to dB. The evaluation seed resets at each SNR.

Latency excludes data loading and checkpoint loading and includes the encoder, channel, denoiser, and decoder. Report batch size and UNet evaluations alongside time: the SNR-dependent start index means `T=1000` does not imply 1,000 model calls at every SNR.

**Current execution status:** the full 100-epoch suite is running. Final numerical results will be added after all training and evaluation jobs finish. The earlier short tests are described in [VALIDATION.md](VALIDATION.md) and are not paper-level accuracy results.

## Defaults and configuration

| Setting | Default | Basis |
|---|---|---|
| Resolution / classes | 480×640 / 9 | Paper, Sec. IV-A |
| Diffusion schedule | T=1000; linear β from 1e-4 to 0.02 | Paper, Sec. IV-A |
| Stage 1 / Stage 3 | 100 epochs; SGD, lr=.01, factor .95 per epoch | Paper, Sec. IV-A |
| Stage 2 | 100 epochs; Adam, lr=1e-4, linear decay | Epoch count from paper; optimizer/rate from legacy scripts |
| Semantic batch / workers / seed | 8 / 4 / 42 | Engineering defaults, not fully specified in paper |
| Stage 2 samples | 20,000 train / 1,000 validation per epoch; batch 64 | Legacy-derived online synthetic protocol |
| Train SNR | 20 dB | Legacy fine-tuning scripts; not specified precisely in paper |
| Test SNR | 0, 2, …, 20 dB | Figs. 3–4 |

H training includes both real and imaginary components with a shared one-channel UNet. Y/HX uses two channels; X uses one. `--config overrides.json` merges JSON recursively and rejects unknown keys. Backbone and target are fixed by the entry point. `--epochs`, `--batch-size`, `--workers`, `--seed`, `--train-snr`, and `--snrs` override common settings; diffusion batch size is `stage2.batch_size` in JSON.

## Implementation differences

1. The inherited channel is shared elementwise fading for two real feature streams, padded onto a 70×70 grid. It omits full MIMO mixing, array responses, and multi-user detection in Eq. (4). The three synthetic paths retain a shared delay. Distances now use floating point to avoid int8 overflow; H is sampled per example instead of once for the entire batch. Noise power still follows batch-average received power.
2. The IR branch no longer accidentally reuses RGB received features. All MFNet scales now use the transmitted noisy features, including previously bypassed skip connections. MFNet and ResNet transmit different feature payloads, so this comparison does not enforce equal channel-use budgets.
3. Power normalization is per sample and protects zero norms. Complex Gaussian noise satisfies E|Z|²=σ², with σ²/2 per component, including CSI noise. The diffusion schedule includes ᾱ₀=1, uses consistent t/T conditioning, scales the starting observation correctly, and handles zero-noise starts.
4. The inherited deterministic reverse update is algebraically an η=0 DDIM-style adjacent-step update, trained with the DDPM noise objective. It is not an ancestral stochastic DDPM sampler, and it has no timestep skipping. Observation conditioning enters through the starting point; there is no separate observation/pilot conditioning encoder.
5. Stage 2 still uses Gaussian synthetic X/HX rather than stored JSCC features. The X extension's post-equalization noise is not i.i.d. Gaussian; its scalar variance is an approximation. This distribution mismatch must be considered when interpreting results.
6. Stage 3 freezes encoder BatchNorm statistics as well as parameters, and uses the paper's SGD protocol. RGB and IR are resized independently, avoiding RGBA alpha premultiplication; labels use nearest-neighbor interpolation. Metrics include the background-related errors omitted in the old helper.

These corrections change the experiment relative to old logs and weights. They require retraining and do not establish that the published numerical gains have been recovered.

## Future improvements

The following are research directions, **not implemented speedups or measured gains** in this release.

- **Prioritize fewer sampling steps.** Evaluate respaced [DDIM](https://arxiv.org/abs/2010.02502) with 10/20/50/100 steps, starting from the observation's matched noise level. Preserve t/T and endpoint scaling; measure nine-class mIoU, CSI NMSE, UNet calls, and synchronized latency together. Since the present update already has an η=0 DDIM-like form, simply renaming it “DDIM” does not accelerate it. A separately implemented ancestral DDPM would be needed for a clean DDPM-vs-DDIM comparison.
- **Evaluate higher-order solvers and distillation.** [DPM-Solver](https://arxiv.org/abs/2206.00927) offers a candidate fast ODE solver; [consistency models](https://proceedings.mlr.press/v202/song23a.html) motivate distilling a trained enhancer into one or a few steps. Their published image-generation speedups are not guarantees for CSI enhancement; recalibrate the observation noise and test low-SNR stability.
- **Close the physical-model gap.** Add explicit multi-user MIMO matrix mixing, antenna geometry, user-specific channels, pilot-based CSI estimates, and MMSE or learned detection. Match channel-use and transmit-power budgets across baselines before drawing communication-efficiency conclusions.
- **Train on the actual semantic-feature distribution.** Cache frozen Stage 1 JSCC features for Y/X training, model real/imaginary components jointly, and condition the enhancer on pilots, estimated CSI, and SNR. Compare against current Gaussian synthetic training and direct CNN/UNet denoisers.
- **Improve efficiency and robustness.** Benchmark smaller UNets, mixed precision, compilation, SNR-adaptive stopping, and distillation. Test unseen SNRs, delay spreads, mobility/Doppler, channel estimation errors, and user counts instead of evaluating only the training channel family.
- **Improve segmentation and statistical reliability.** Study imbalance-aware losses for rare classes, report day/night and per-class IoU, and repeat training with at least three seeds and paired channel draws. A single-seed result cannot quantify run-to-run uncertainty.

## Tests and provenance

```bash
python -m unittest discover -s tests -v
python tests/validate_pipeline.py --device cuda:0 --output-dir validation/pipeline
```

Tests cover independent entry points, real/imaginary signal handling, normalization, diffusion time indexing, four-channel preprocessing, virtual flips, background-aware IoU, frozen BatchNorm, and checkpoint compatibility/resume. The bounded real-data validator is explicitly labeled and never substitutes for full training. See [VALIDATION.md](VALIDATION.md).

The MFNet backbone and adapted diffusion UNet retain their upstream attribution. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the included Apache 2.0 text. Dataset files, private server paths, access credentials, and training checkpoints are excluded from Git.

## Citation

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
