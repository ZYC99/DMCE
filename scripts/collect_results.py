"""Collect an entire run_suite output into portable tables and scientific plots.

Fails if expected evaluation files or completed training stages are missing.
Absolute data/checkpoint paths and host details are omitted from public artifacts.
The paper's reported numbers are never substituted for measured results.
"""
import argparse
import ast
import csv
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

METHODS = {'dmce': 'DMCE (CSI)', 'without_dmce': 'Without DMCE', 'y_denoising': 'Y-oriented',
           'x_denoising': 'X-oriented (extension)', 'mfnet_channel': 'MFNet with channel'}


def environment():
    import torch
    value = {'collected_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'python': platform.python_version(), 'system': platform.system(),
             'libraries': {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'numpy', 'Pillow', 'matplotlib')},
             'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
             'default_dtype': str(torch.get_default_dtype()), 'amp_used': False,
             'cudnn_benchmark': torch.backends.cudnn.benchmark,
             'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                check=True, text=True, capture_output=True)
        value['gpu_models'] = sorted(set(result.stdout.splitlines()))
    except (FileNotFoundError, subprocess.CalledProcessError):
        value['gpu_models'] = []
    return value


def checkpoint_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit_freezing(run_root):
    import torch
    shared = torch.load(run_root / 'without_dmce/stage1/best.pt', map_location='cpu', weights_only=True)['model']
    checks = {}
    for experiment in METHODS:
        initial = (torch.load(run_root / 'mfnet_channel/stage1/best.pt', map_location='cpu', weights_only=True)['model']
                   if experiment == 'mfnet_channel' else shared)
        state = torch.load(run_root / experiment / 'stage3/best.pt', map_location='cpu', weights_only=True)['model']
        prefixes = ('conv',) if experiment == 'mfnet_channel' else ('resnet_rgb.', 'resnet_ir.', 'channel_encoder.')
        frozen = [key for key in initial if key.startswith(prefixes)]
        if not frozen or any(not torch.equal(initial[key], state[key]) for key in frozen):
            raise ValueError(f'{experiment}: frozen encoder state changed during Stage 3')
        updated = [key for key in initial if not key.startswith(prefixes)
                   and key.endswith(('.weight', '.bias')) and not torch.equal(initial[key], state[key])]
        if not updated:
            raise ValueError(f'{experiment}: no decoder weight changed during Stage 3')
        diffusion = 0
        if experiment in ('dmce', 'y_denoising', 'x_denoising'):
            initial_dm = torch.load(run_root / experiment / 'stage2/best.pt', map_location='cpu', weights_only=True)['model']
            if any(not torch.equal(value, state['ddpm.' + key]) for key, value in initial_dm.items()):
                raise ValueError(f'{experiment}: diffusion state changed during Stage 3')
            diffusion = len(initial_dm)
            del initial_dm
        checks[experiment] = {'unchanged_encoder_tensors': len(frozen), 'updated_decoder_weight_tensors': len(updated),
                              'unchanged_diffusion_tensors': diffusion}
        del initial, state
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    suite, output = args.suite_dir, args.output_dir
    state = json.loads((suite / 'status.json').read_text())
    incomplete = [name for name, result in state['results'].items() if result['status'] != 'completed']
    if incomplete:
        raise ValueError(f'Suite is incomplete: {incomplete}')
    import torch
    output.mkdir(parents=True, exist_ok=True)
    measurements, training = {}, {}
    full_protocol = True
    for job in state['jobs']:
        if not job['stage'].startswith('stage'):
            continue
        stage_path = suite / 'runs' / job['experiment'] / job['stage']
        with (stage_path / 'metrics.jsonl').open() as stream:
            # Atomic checkpoints may lag a logged epoch if a process was interrupted.
            history = {record['epoch']: record for line in stream if (record := json.loads(line))}
        last = torch.load(stage_path / 'last.pt', map_location='cpu', weights_only=True)
        configured_epochs = last['config'][job['stage']]['epochs']
        if last['epoch'] != configured_epochs or len(history) != configured_epochs:
            raise ValueError(f'Training is incomplete: {job["name"]}')
        config = last['config']
        public_config = {key: value for key, value in config.items()
                         if key not in ('data_dir', 'output_dir', 'run_dir', 'device')}
        del last
        best_path = stage_path / 'best.pt'
        best = torch.load(best_path, map_location='cpu', weights_only=True)
        training[job['name']] = {'configured_epochs': configured_epochs, 'completed_epochs': max(history),
                                'best_epoch': best['epoch'], 'best_validation_metric': best['best'],
                                'best_checkpoint_sha256': checkpoint_hash(best_path),
                                'config': public_config, 'history': list(history.values())}
        execution = state['results'][job['name']]
        if 'started_at' in execution and 'finished_at' in execution:
            training[job['name']]['process_wall_seconds'] = (
                datetime.datetime.fromisoformat(execution['finished_at']) -
                datetime.datetime.fromisoformat(execution['started_at'])).total_seconds()
        del best
        arguments = json.loads((stage_path / 'config.json').read_text())['arguments']
        full_protocol &= configured_epochs == 100 and arguments['max_batches'] is None
        full_protocol &= config['image_size'] == [480, 640] and config['diffusion']['steps'] == 1000
        if job['stage'] in ('stage1', 'stage3'):
            full_protocol &= all(row['train']['samples'] == 1568 and row['validation']['samples'] == 392
                                 for row in history.values())
        else:
            full_protocol &= config['stage2']['samples'] == 20000 and config['stage2']['validation_samples'] == 1000
    rows = []
    for experiment in METHODS:
        measurements[experiment] = {}
        for split, expected_count in [('test', 393), ('test_day', 205), ('test_night', 188)]:
            path = suite / 'runs' / experiment / 'evaluation' / f'{split}.json'
            values = json.loads(path.read_text())['results']
            measurements[experiment][split] = values
            full_protocol &= [value['snr_db'] for value in values] == list(range(0, 21, 2))
            full_protocol &= all(value['samples'] == expected_count for value in values)
            rows.extend(dict(experiment=experiment, split=split, **value) for value in values)
    nmse = json.loads((suite / 'runs/dmce/evaluation/csi_nmse.json').read_text())['results']
    full_protocol &= [value['snr_db'] for value in nmse] == list(range(0, 21, 2))
    source_hashes = json.loads((suite / 'source_sha256.json').read_text())
    # Only research code and protocol configurations belong in the public record;
    # machine-specific configuration filenames are not needed for reproduction.
    source_hashes = {name: digest for name, digest in source_hashes.items()
                     if name.startswith(('dmce/', 'experiments/')) or name in {f'configs/{key}.json' for key in ('base', *METHODS)}}
    current = Path(__file__).resolve().parents[1] / 'dmce'
    ast_matches = {path.name: ast.dump(ast.parse(path.read_text())) == ast.dump(ast.parse((current / path.name).read_text()))
                   for path in (suite / 'source/dmce').glob('*.py')}
    report = {'protocol': 'full_mfnet_100_epochs' if full_protocol else 'custom_or_bounded_run',
              'scope': 'refactored legacy implementation; not an exact numerical reproduction of the paper',
              'training': training, 'segmentation': measurements, 'csi_nmse': nmse,
              'freeze_audit': audit_freezing(suite / 'runs'),
              'environment': environment(), 'training_modules_match_current_ast': ast_matches,
              'source_sha256': source_hashes}
    (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    with (output / 'segmentation.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['experiment', 'split', 'snr_db', 'samples', 'miou',
                                                    'miou_foreground', 'pixel_accuracy', 'loss'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    with (output / 'csi_nmse.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(nmse[0]))
        writer.writeheader()
        writer.writerows(nmse)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for split in ('test', 'test_day', 'test_night'):
        figure, axes = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        for experiment, title in METHODS.items():
            values = measurements[experiment][split]
            axes.plot([row['snr_db'] for row in values], [row['miou'] * 100 for row in values],
                      marker='o', linestyle='--' if experiment == 'x_denoising' else '-', label=title)
        axes.set(xlabel='SNR (dB)', ylabel='9-class mIoU (%)', title=f'MFNet {split}: measured refactored implementation')
        axes.grid(alpha=.25)
        axes.legend(fontsize=8)
        for suffix in ('png', 'pdf'):
            figure.savefig(output / f'miou_{split}.{suffix}', dpi=180)
        plt.close(figure)
    figure, axes = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for key, title in [('initial_nmse_db', 'Initial CSI'), ('enhanced_nmse_db', 'Enhanced CSI')]:
        axes.plot([row['snr_db'] for row in nmse], [row[key] for row in nmse], marker='o', label=title)
    axes.set(xlabel='SNR (dB)', ylabel='CSI NMSE (dB)', title='Measured CSI estimation error')
    axes.grid(alpha=.25)
    axes.legend()
    for suffix in ('png', 'pdf'):
        figure.savefig(output / f'csi_nmse.{suffix}', dpi=180)
    plt.close(figure)
    print(f'Collected {len(training)} training stages, {len(rows)} segmentation measurements, and {len(nmse)} NMSE points')


if __name__ == '__main__':
    main()
