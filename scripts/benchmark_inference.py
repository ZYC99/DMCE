"""Measure full-model latency, peak CUDA memory, and noise-predictor calls.

Excludes checkpoint loading and disk I/O. Uses fixed real test images, FP32, warmup,
CUDA synchronization, and a stated batch size; no accuracy claim follows from this
latency microbenchmark. Report other jobs/clock variability separately.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmce import checkpoints
from dmce.config import EXPERIMENTS
from dmce.data import SegmentationDataset
from dmce.models import build_model
from dmce.runner import device_for, seed_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--experiments', nargs='+', choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument('--snrs', nargs='+', type=float, default=[0, 4, 10, 20])
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()
    if min(args.batch_size, args.repeats) < 1 or args.warmup < 0:
        parser.error('Invalid batch size, repeats, or warmup')
    device = device_for(args.device)
    cuda = device.type == 'cuda'
    results = {'python': platform.python_version(), 'torch': torch.__version__,
               'device': torch.cuda.get_device_name(device) if cuda else str(device), 'dtype': 'float32',
               'batch_size': args.batch_size, 'warmup': args.warmup, 'repeats': args.repeats,
               'timing_scope': 'resident input to logits; includes channel and denoising, excludes I/O and loading',
               'checkpoints': {}, 'results': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for experiment in args.experiments:
        checkpoint_path = args.run_dir / experiment / 'stage3/best.pt'
        payload = checkpoints.read(checkpoint_path)
        config = payload['config']
        if payload.get('stage') != 'stage3' or payload.get('experiment') != experiment:
            raise ValueError('Wrong checkpoint stage or experiment')
        model = build_model(config).to(device)
        model.load_state_dict(payload['model'], strict=True)
        model.eval()
        with checkpoint_path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        results['checkpoints'][experiment] = {'sha256': digest, 'epoch': payload['epoch'],
                                              'image_size': config['image_size']}
        del payload
        dataset = SegmentationDataset(args.data_dir, 'test', config['image_size'])
        if args.batch_size > len(dataset):
            raise ValueError('Batch size exceeds test set')
        images = torch.stack([dataset[i][0] for i in range(args.batch_size)]).to(device)
        results['checkpoints'][experiment]['sample_ids'] = dataset.names[:args.batch_size]
        calls = [0]

        def count_call(*_):
            calls[0] += 1

        hook = model.ddpm.eps_model.register_forward_hook(count_call) if model.ddpm is not None else None
        for snr in args.snrs:
            timings, evaluations = [], []
            if cuda:
                torch.cuda.reset_peak_memory_stats(device)
            with torch.inference_mode():
                for repeat in range(-args.warmup, args.repeats):
                    seed_all(config['seed'] + repeat + args.warmup)
                    calls[0] = 0
                    if cuda:
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    model(images, snr)
                    if cuda:
                        torch.cuda.synchronize(device)
                    elapsed = (time.perf_counter() - started) * 1000
                    if repeat >= 0:
                        timings.append(elapsed)
                        evaluations.append(calls[0])
            row = {'experiment': experiment, 'snr_db': snr, 'median_batch_ms': statistics.median(timings),
                   'mean_batch_ms': statistics.mean(timings), 'min_batch_ms': min(timings), 'max_batch_ms': max(timings),
                   'median_ms_per_image_amortized': statistics.median(timings) / args.batch_size,
                   'unet_calls_per_batch': evaluations, 'batch_ms': timings,
                   'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated(device) / 1024**2 if cuda else None}
            results['results'].append(row)
            args.output.write_text(json.dumps(results, indent=2) + '\n')
            print({key: value for key, value in row.items() if key not in ('unet_calls_per_batch', 'batch_ms')}, flush=True)
        if hook is not None:
            hook.remove()
        del model, images
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
