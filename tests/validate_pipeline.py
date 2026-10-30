"""Bounded integration checks with synthetic files; never reports scientific accuracy.

Run from refactored/: python tests/validate_pipeline.py --device cuda:0 --output-dir /tmp/dmce-validation
Temporary synthetic checkpoints are deleted after each experiment; logs/results remain.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    def run(label, command):
        start = time.monotonic()
        with (args.output_dir / f'{label}.log').open('w') as log:
            result = subprocess.run([sys.executable, *command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2'))
        record = {'check': label, 'returncode': result.returncode, 'seconds': round(time.monotonic() - start, 2)}
        results.append(record)
        (args.output_dir / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
        print(record, flush=True)
        if result.returncode:
            raise RuntimeError(f'Failed: {label}; see {args.output_dir / (label + ".log")}')

    with tempfile.TemporaryDirectory(prefix='dmce-validation-') as temporary:
        temp = Path(temporary)
        data = temp / 'dataset'
        (data / 'images').mkdir(parents=True)
        (data / 'labels').mkdir()
        rng = np.random.default_rng(0)
        for index in range(4):
            Image.fromarray(rng.integers(0, 256, (64, 64, 4), dtype=np.uint8)).save(data / 'images' / f'{index}.png')
            Image.fromarray(rng.integers(0, 9, (64, 64), dtype=np.uint8)).save(data / 'labels' / f'{index}.png')
        for split in ('train', 'val', 'test', 'test_day', 'test_night'):
            (data / f'{split}.txt').write_text('0\n1\n' if split == 'train' else '2\n3\n')
        override = temp / 'quick.json'
        override.write_text(json.dumps({
            'image_size': [64, 64], 'batch_size': 2, 'workers': 0, 'pretrained': False,
            'stage1': {'epochs': 1}, 'stage2': {'epochs': 1, 'samples': 2, 'validation_samples': 2, 'batch_size': 2},
            'stage3': {'epochs': 1},
            'diffusion': {'steps': 4, 'inner_channels': 8, 'norm_groups': 8,
                          'channel_mults': [1, 2], 'attention_resolutions': [], 'res_blocks': 1}}))
        common = ['--config', str(override), '--data-dir', str(data), '--output-dir', str(temp / 'runs'),
                  '--device', args.device, '--max-batches', '1']
        run('shared_stage1', ['experiments/without_dmce.py', '--stage', 'stage1', *common])
        shared = temp / 'runs/without_dmce/stage1/best.pt'
        for experiment in ('dmce', 'y_denoising', 'x_denoising', 'without_dmce', 'mfnet_channel'):
            entry = f'experiments/{experiment}.py'
            if experiment == 'mfnet_channel':
                run(f'{experiment}_all', [entry, '--stage', 'all', *common])
            else:
                run(f'{experiment}_all', [entry, '--stage', 'all', '--stage1-checkpoint', str(shared), *common])
            run(f'{experiment}_eval', [entry, '--stage', 'eval', '--snrs', '0', '20', *common])
            output = temp / 'runs' / experiment
            if experiment == 'dmce':
                run('dmce_nmse', [entry, '--stage', 'nmse', '--snrs', '0', '20', *common])
            if experiment == 'mfnet_channel':
                run('resume_stage3', [entry, '--stage', 'stage3', '--epochs', '2',
                                     '--resume', str(output / 'stage3/last.pt'), *common])
            shutil.copytree(output / 'evaluation', args.output_dir / experiment, dirs_exist_ok=True)
            shutil.rmtree(output / 'stage3')
            if (output / 'stage2').exists():
                shutil.rmtree(output / 'stage2')
        # Also exercise stage2 resume without an image dataset or semantic model.
        run('diffusion_stage2', ['experiments/dmce.py', '--stage', 'stage2', *common])
        run('resume_stage2', ['experiments/dmce.py', '--stage', 'stage2', '--epochs', '2',
                             '--resume', str(temp / 'runs/dmce/stage2/last.pt'), *common])
        run('resume_stage1', ['experiments/without_dmce.py', '--stage', 'stage1', '--epochs', '2',
                             '--resume', str(temp / 'runs/without_dmce/stage1/last.pt'), *common])


if __name__ == '__main__':
    main()
