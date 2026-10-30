"""Bounded real-data GPU validation at the default resolution and UNet size.

Uses one training batch and one validation/test batch per stage; this is NOT a
convergence run or a reproduction of the paper's accuracy. Keeps all outputs.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.data_dir, args.output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = args.output_dir / 'bounded.json'
    config.write_text(json.dumps({'batch_size': 2, 'workers': 2,
                                 'stage2': {'samples': 2, 'validation_samples': 2, 'batch_size': 2}}, indent=2))
    common = ['--config', str(config), '--data-dir', str(args.data_dir), '--output-dir', str(args.output_dir / 'runs'),
              '--device', args.device, '--epochs', '1', '--max-batches', '1']
    results = []

    def run(label, arguments):
        command = [sys.executable, *arguments]
        started = time.monotonic()
        with (args.output_dir / f'{label}.log').open('w') as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2'))
        record = {'check': label, 'command': command, 'returncode': result.returncode,
                  'seconds': round(time.monotonic() - started, 2)}
        results.append(record)
        (args.output_dir / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
        print({key: value for key, value in record.items() if key != 'command'}, flush=True)
        if result.returncode:
            raise RuntimeError(f'{label} failed; inspect its log')

    run('shared_stage1', ['experiments/without_dmce.py', '--stage', 'stage1', *common])
    shared = args.output_dir / 'runs/without_dmce/stage1/best.pt'
    for name in ('dmce', 'y_denoising', 'x_denoising', 'without_dmce', 'mfnet_channel'):
        entry = f'experiments/{name}.py'
        initialization = [] if name == 'mfnet_channel' else ['--stage1-checkpoint', str(shared)]
        run(f'{name}_all', [entry, '--stage', 'all', *initialization, *common])
        run(f'{name}_eval', [entry, '--stage', 'eval', '--snrs', '0', '20', *common])
        if name == 'dmce':
            run('dmce_nmse', [entry, '--stage', 'nmse', '--snrs', '0', '20', *common])


if __name__ == '__main__':
    main()
