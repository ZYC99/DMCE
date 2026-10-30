"""Render measured results into bilingual GitHub reports, with provenance checks.

This never fetches or substitutes published paper numbers. The default accepts
only the full MFNet protocol. --allow-bounded is solely for inspecting test output
and adds a prominent bounded-run label to both reports.
"""
import argparse
import json
import math
from pathlib import Path
import statistics

METHODS = {'dmce': 'DMCE (CSI)', 'without_dmce': 'Without DMCE', 'y_denoising': 'Y-oriented',
           'x_denoising': 'X-oriented (extension)', 'mfnet_channel': 'MFNet with channel'}


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(map(str, headers)) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]]) + '\n'


def read_and_check(directory, allow_bounded=False):
    report = json.loads((directory / 'results.json').read_text())
    latency = json.loads((directory / 'latency.json').read_text())
    qualitative = json.loads((directory / 'qualitative/manifest.json').read_text())
    full = report['protocol'] == 'full_mfnet_100_epochs'
    if not full and not allow_bounded:
        raise ValueError('Refusing to publish a bounded/custom run as full results; use --allow-bounded for debugging only')
    if not report['training_modules_match_current_ast'] or not all(report['training_modules_match_current_ast'].values()):
        raise ValueError('Training module logic differs from the source snapshot')
    for experiment in METHODS:
        expected = report['training'][f'{experiment}_stage3']['best_checkpoint_sha256']
        if qualitative['experiments'][experiment]['checkpoint_sha256'] != expected:
            raise ValueError(f'{experiment}: qualitative checkpoint does not match metrics')
        if latency['checkpoints'][experiment]['sha256'] != expected:
            raise ValueError(f'{experiment}: latency checkpoint does not match metrics')
    expected_pairs = {(experiment, snr) for experiment in METHODS for snr in (0, 4, 10, 20)}
    actual_pairs = {(row['experiment'], row['snr_db']) for row in latency['results']}
    if full and (actual_pairs != expected_pairs or len(latency['results']) != len(expected_pairs)):
        raise ValueError('Full report needs each method at latency SNRs 0, 4, 10, and 20 dB')
    for row in latency['results']:
        if len(row['batch_ms']) != latency['repeats'] or len(row['unet_calls_per_batch']) != latency['repeats']:
            raise ValueError('Incomplete latency repetitions')
        if not all(math.isfinite(value) and value > 0 for value in row['batch_ms']):
            raise ValueError('Invalid latency measurement')
    if full and (qualitative['snr_db'] != 4 or len(qualitative['samples']) != 4):
        raise ValueError('Full report needs two day and two night qualitative examples at 4 dB')
    for name in qualitative['samples']:
        if not (directory / 'qualitative' / f'{name}_comparison.png').is_file():
            raise FileNotFoundError(f'Missing qualitative comparison: {name}')
    return report, latency, qualitative, full


def render(report, latency, qualitative, full, chinese=False):
    def lang(english, translated):
        return translated if chinese else english

    lines = [lang('# Measured experiment results', '# 实测实验结果'),
             '[English](README.md) | [简体中文](README.zh-CN.md)',
             lang('This report is generated from the measured JSON/CSV artifacts in this directory. Values are from the corrected legacy implementation, not copied from the ICC 2024 paper.',
                  '本报告由当前目录中的实测 JSON/CSV 自动生成。数值来自修复后的原实现，没有抄录 ICC 2024 论文的结果。')]
    if not full:
        lines.append(lang('**BOUNDED TEST OUTPUT — not a full experiment or paper accuracy result.**',
                          '**有界短测输出——不是完整实验，也不是论文精度结果。**'))
    seed = report['training']['shared_stage1']['config']['seed']
    samples = {split: report['segmentation']['dmce'][split][0]['samples'] for split in ('test', 'test_day', 'test_night')}
    lines += [lang(f'Protocol: `{report["protocol"]}`. Seed: **{seed}**. Test scenes: **{samples["test"]}** '
                   f'(**{samples["test_day"]}** day, **{samples["test_night"]}** night). '
                   'Models are selected by validation performance; test scores do not select checkpoints.',
                   f'协议：`{report["protocol"]}`；种子：**{seed}**。测试场景共 **{samples["test"]}** 个，'
                   f'其中白天 **{samples["test_day"]}**、夜间 **{samples["test_night"]}**。根据验证集选择权重，不用测试分数挑选模型。'),
              lang('This is a single-seed run without confidence intervals. It retains the simplified shared elementwise channel, synthetic diffusion training data, and deterministic adjacent-step sampler. It does not establish the paper\'s full MIMO results or published percentage gains. The X-oriented method is a legacy extension.',
                   '本次为单种子实验，没有置信区间。实现保留简化的共享逐元素信道、合成扩散训练数据及确定性相邻时间步采样，因此不代表已恢复论文完整 MIMO 模型或其中的百分比增益。X-oriented 是原代码扩展。'),
              lang('All/day/night are separate evaluation passes with reset random seeds. Channel draws and batch-average noise power can therefore differ between passes; the day/night confusion matrices are not partitions of a single all-test pass.',
                   '完整测试集、昼间和夜间分别执行评测并重置随机种子，因此各次评测的信道抽样与 batch 平均噪声功率可能不同；昼夜混淆矩阵不是同一次完整测试推理的拆分。')]
    for split, heading in [('test', lang('All test scenes', '完整测试集')),
                           ('test_day', lang('Day scenes', '白天场景')),
                           ('test_night', lang('Night scenes', '夜间场景'))]:
        index = {experiment: {row['snr_db']: row for row in report['segmentation'][experiment][split]}
                 for experiment in METHODS}
        snrs = sorted(index['dmce'])
        rows = [[f'{snr:g}', *[f'{100 * index[method][snr]["miou"]:.2f}' for method in METHODS]] for snr in snrs]
        lines += [f'## {heading}', lang('Nine-class mIoU (%); the background class is included.', '九类 mIoU（%），包括背景类。'),
                  table(['SNR (dB)', *METHODS.values()], rows), f'![{heading}](miou_{split}.png)']
    all_test = {experiment: {row['snr_db']: row for row in report['segmentation'][experiment]['test']}
                for experiment in METHODS}
    if 0 in all_test['dmce']:
        dmce, baseline = 100 * all_test['dmce'][0]['miou'], 100 * all_test['without_dmce'][0]['miou']
        lines.append(lang(f'At 0 dB, DMCE measured {dmce:.2f}% versus {baseline:.2f}% without DMCE, '
                          f'a difference of **{dmce - baseline:+.2f} percentage points** in this run. '
                          'This is an absolute difference, not a relative percentage improvement.',
                          f'本次 0 dB 下，DMCE 为 {dmce:.2f}%，Without-DMCE 为 {baseline:.2f}%，'
                          f'差值为 **{dmce - baseline:+.2f} 个百分点**。这是绝对差值，不是相对提升百分比。'))
    if 4 in all_test['dmce']:
        lines += [lang('## Per-class IoU at 4 dB', '## 4 dB 下的逐类 IoU'),
                  table([lang('Class', '类别'), *METHODS.values()],
                        [[name, *[f'{100 * all_test[method][4]["iou_per_class"][index]:.2f}' for method in METHODS]]
                         for index, name in enumerate(qualitative['classes'])])]
    lines += [lang('## CSI NMSE', '## CSI NMSE'),
              lang('Total squared error / total channel power, expressed in dB. Lower is better.',
                   '总误差功率 / 总真实信道功率，转为 dB；越低越好。'),
              table(['SNR (dB)', lang('Initial CSI (dB)', '初始 CSI（dB）'), lang('Enhanced CSI (dB)', '增强 CSI（dB）')],
                    [[f'{row["snr_db"]:g}', f'{row["initial_nmse_db"]:.3f}', f'{row["enhanced_nmse_db"]:.3f}'] for row in report['csi_nmse']]),
              '![CSI NMSE](csi_nmse.png)',
              lang('## Inference latency', '## 推理耗时'),
              lang(f'Device: **{latency["device"]}**. Precision: **{latency["dtype"]}**. Batch size: **{latency["batch_size"]}**. '
                   f'Warmup: **{latency["warmup"]}**, measured repetitions: **{latency["repeats"]}** per method/SNR. '
                   'Synchronized full-model timing includes channel and denoising, excludes loading and I/O. One input scene contains RGB and IR.',
                   f'设备：**{latency["device"]}**；精度：**{latency["dtype"]}**；batch size：**{latency["batch_size"]}**。'
                   f'每种方法、每个 SNR 预热 **{latency["warmup"]}** 次，测量 **{latency["repeats"]}** 次。'
                   '同步计时覆盖全模型、信道及降噪，不含权重加载和数据读取；一个输入场景同时包含 RGB 与 IR。'),
              table([lang('Method', '方法'), 'SNR (dB)', lang('Median ms/batch', '每 batch 中位耗时（ms）'),
                     lang('Min–max ms', '最小–最大（ms）'), lang('Median UNet calls/batch', '每 batch UNet 调用中位数')],
                    [[METHODS[row['experiment']], f'{row["snr_db"]:g}', f'{row["median_batch_ms"]:.2f}',
                      f'{min(row["batch_ms"]):.2f}–{max(row["batch_ms"]):.2f}', f'{statistics.median(row["unet_calls_per_batch"]):g}']
                     for row in latency['results']]),
              lang('T=1000 is the training schedule length; actual reverse calls depend on the observation noise and starting index. '
                   'These measurements are the current baseline. Respaced DDIM, DPM-Solver, and distillation remain future work; no speedup from them is claimed here.',
                   'T=1000 是训练日程长度，实际逆扩散调用次数取决于观测噪声与起点。这些是当前基线的实测值；跳步 DDIM、DPM-Solver 与蒸馏仍为后续方向，此处没有宣称它们已经带来提速。'),
              lang('## Fixed qualitative examples', '## 固定选取的定性示例'),
              lang(f'SNR: **{qualitative["snr_db"]:g} dB**. Selection: first entries of the official day/night lists, '
                   'without choosing examples after seeing model outputs. Sample IDs and checkpoint hashes are in the manifest.',
                   f'SNR：**{qualitative["snr_db"]:g} dB**。按官方昼间、夜间列表的顺序选取，未根据模型输出挑选展示样本。样本 ID 和权重哈希记录在清单中。')]
    lines.extend(f'![{name}](qualitative/{name}_comparison.png)' for name in qualitative['samples'])
    lines += [lang('## Training completion and provenance', '## 训练完成情况与来源'),
              table([lang('Stage', '阶段'), lang('Completed epochs', '完成轮数'), lang('Best epoch', '最优轮次'),
                     lang('Validation metric', '验证指标'), lang('Process hours', '进程耗时（小时）')],
                    [[name, row['completed_epochs'], row['best_epoch'],
                      (f'noise MSE {row["best_validation_metric"]:.6f}' if name.endswith('stage2') else f'mIoU {100 * row["best_validation_metric"]:.2f}%'),
                      f'{row["process_wall_seconds"] / 3600:.3f}' if 'process_wall_seconds' in row else '—']
                     for name, row in report['training'].items()]),
              lang('Process hours are per-job wall time; parallel jobs must not be summed as end-to-end wall time. '
                   'If a job is resumed, this records its latest launch rather than all prior attempts. '
                   'All receiver stages passed the tensor-level audit: encoder parameters and BatchNorm buffers remained unchanged, decoder weights updated, and diffusion state stayed frozen.',
                   '进程耗时为各任务的墙钟时间，并行任务不能直接相加作为整套实验历时；若发生续训，记录的是最近一次启动的历时，不包括先前尝试。接收端微调通过逐张量核对：编码器参数与 BatchNorm 缓冲保持不变，解码器权重有更新，扩散状态保持冻结。'),
              table([lang('Method', '方法'), lang('Frozen encoder tensors', '保持不变的编码器张量'),
                     lang('Updated decoder tensors', '更新的解码器权重张量'), lang('Frozen diffusion tensors', '保持不变的扩散张量')],
                    [[METHODS[name], row['unchanged_encoder_tensors'], row['updated_decoder_weight_tensors'], row['unchanged_diffusion_tensors']]
                     for name, row in report['freeze_audit'].items()]),
              lang('Runtime versions: ', '运行库版本：') + ', '.join(f'{name} `{version}`' for name, version in report['environment']['libraries'].items()) + '.',
              lang('The raw report preserves resolved research configurations, all epoch records, checkpoint SHA-256 hashes, source hashes, confusion matrices, and per-class metrics. '
                   'All recorded training-module AST comparisons passed. Local paths and credentials are omitted.',
                   '原始报告保留实际研究配置、逐轮记录、权重 SHA-256、源码哈希、混淆矩阵与逐类指标。记录的训练模块 AST 对照全部通过；不包含私有路径或账号凭证。'),
              lang('## Machine-readable artifacts', '## 机器可读数据'),
              '- [results.json](results.json)\n- [segmentation.csv](segmentation.csv)\n- [csi_nmse.csv](csi_nmse.csv)\n- [latency.json](latency.json)\n- [qualitative manifest](qualitative/manifest.json)']
    return '\n\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--allow-bounded', action='store_true')
    args = parser.parse_args()
    values = read_and_check(args.results_dir, args.allow_bounded)
    for chinese, filename in [(False, 'README.md'), (True, 'README.zh-CN.md')]:
        (args.results_dir / filename).write_text(render(*values, chinese=chinese))
    print('Wrote bilingual measured-results reports')


if __name__ == '__main__':
    main()
