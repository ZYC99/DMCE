import csv
import gc
import itertools
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import checkpoints
from .channel import response, complex_noise
from .config import smoke_config
from .data import SegmentationDataset, Metrics, check_dataset
from .diffusion import Diffusion, synthetic_clean
from .models import build_model


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name != 'cpu' and not name.startswith('cuda'):
        raise ValueError('Use cpu or cuda[:index]; this simulator requires complex tensor operations')
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    return device


def path_for(config, stage):
    return Path(config['run_dir']) / stage / 'best.pt'


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def prepare_output(config, stage, args):
    output = Path(config['run_dir']) / stage
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'last.pt').exists() and not args.resume:
        raise FileExistsError(f'{output}/last.pt exists; use --resume or a new --output-dir')
    write_json(output / 'config.json', {'config': config, 'arguments': vars_as_json(args)})
    return output


def vars_as_json(args):
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def loaders(config):
    result = []
    for split in ('train', 'val'):
        data = SegmentationDataset(config['data_dir'], split, config['image_size'], augment=split == 'train')
        if split == 'train' and len(data) < 2:
            raise ValueError('Training with BatchNorm requires at least two examples')
        # Avoid a single-item final batch (ResNet's smallest spatial map can be 1x1).
        batch = min(config['batch_size'], len(data))
        result.append(DataLoader(data, batch_size=batch, shuffle=split == 'train',
                                 num_workers=config['workers'], pin_memory=False,
                                 drop_last=split == 'train' and len(data) % batch == 1))
    return result


def pass_segmentation(model, loader, device, config, clean, limit, optimizer=None):
    metrics = Metrics(config['classes'])
    total_loss, count = 0.0, 0
    iterator = itertools.islice(loader, limit) if limit else loader
    with torch.set_grad_enabled(optimizer is not None):
        for step, (images, labels, _) in enumerate(iterator, 1):
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images, config['train_snr'], clean=clean)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite segmentation loss')
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            metrics.update(logits.detach(), labels)
            total_loss += float(loss.detach()) * len(images)
            count += len(images)
            if step % 20 == 0:
                print(f'  batch={step} loss={float(loss.detach()):.5f}', flush=True)
    if not count:
        raise ValueError('Empty data loader')
    return dict(metrics.result(), loss=total_loss / count, samples=count)


def train_semantic(config, args, device, stage):
    # All ResNet variants share the exact clean stage1 architecture and RNG consumption.
    model_config = dict(config, target='none') if stage == 'stage1' else config
    model = build_model(model_config, pretrained=config['pretrained'] and stage == 'stage1' and not args.resume).to(device)
    if stage == 'stage3' and not args.resume:
        checkpoints.load_semantic(model, args.stage1_checkpoint or path_for(config, 'stage1'))
        if model.ddpm is not None:
            checkpoints.load_diffusion(model.ddpm, args.diffusion_checkpoint or path_for(config, 'stage2'))
    model.set_stage(stage)
    settings = config[stage]
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad),
                                lr=settings['lr'], momentum=settings['momentum'], weight_decay=settings['weight_decay'])
    first, best = (1, -1.0)
    if args.resume:
        first, best = checkpoints.resume(args.resume, model, optimizer, config, stage)
    if first > settings['epochs']:
        raise ValueError(f'Already completed epoch {first - 1}; --epochs must be larger to resume')
    output = prepare_output(config, stage, args)
    train_loader, val_loader = loaders(config)
    for epoch in range(first, settings['epochs'] + 1):
        started = time.monotonic()
        for group in optimizer.param_groups:
            group['lr'] = settings['lr'] * settings['lr_decay'] ** (epoch - 1)
        model.train_for_stage(stage)
        training = pass_segmentation(model, train_loader, device, config, stage == 'stage1', args.max_batches, optimizer)
        model.eval()
        # Fix validation draws across epochs without disturbing the training RNG stream.
        with torch.random.fork_rng(devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []):
            torch.manual_seed(config['seed'] + 100000)
            validation = pass_segmentation(model, val_loader, device, config, stage == 'stage1', args.max_batches)
        improved = validation['miou'] > best
        best = max(best, validation['miou'])
        record = {'epoch': epoch, 'train': training, 'validation': validation,
                  'elapsed_seconds': time.monotonic() - started}
        with (output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print(f'{config["experiment"]}/{stage} epoch={epoch} loss={training["loss"]:.5f} val_mIoU={validation["miou"]:.5f}', flush=True)
        checkpoints.save(output / 'last.pt', model, optimizer, config, stage, epoch, best)
        if improved:
            checkpoints.save(output / 'best.pt', model, optimizer, config, stage, epoch, best)


def diffusion_pass(model, config, device, samples, limit, optimizer=None):
    batch_size = config['stage2']['batch_size']
    total, count = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for step, offset in enumerate(range(0, samples, batch_size)):
            if limit is not None and step >= limit:
                break
            clean = synthetic_clean(config['target'], min(batch_size, samples - offset), device, config)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss = model(clean)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite diffusion loss')
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += float(loss.detach()) * len(clean)
            count += len(clean)
            if (step + 1) % 20 == 0:
                print(f'  diffusion batch={step + 1} loss={float(loss.detach()):.5f}', flush=True)
    return total / count


def train_diffusion(config, args, device):
    model = Diffusion(config['target'], config['diffusion']).to(device)
    settings = config['stage2']
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['lr'])
    first, best = (1, float('inf'))
    if args.resume:
        first, best = checkpoints.resume(args.resume, model, optimizer, config, 'stage2')
    if first > settings['epochs']:
        raise ValueError(f'Already completed epoch {first - 1}; --epochs must be larger to resume')
    output = prepare_output(config, 'stage2', args)
    for epoch in range(first, settings['epochs'] + 1):
        started = time.monotonic()
        optimizer.param_groups[0]['lr'] = settings['lr'] * (1 - (epoch - 1) / settings['epochs'])
        model.train()
        training = diffusion_pass(model, config, device, settings['samples'], args.max_batches, optimizer)
        model.eval()
        with torch.random.fork_rng(devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []):
            torch.manual_seed(config['seed'] + 100000)
            validation = diffusion_pass(model, config, device, settings['validation_samples'], args.max_batches)
        improved = validation < best
        best = min(best, validation)
        with (output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps({'epoch': epoch, 'train_mse': training, 'validation_mse': validation,
                                     'elapsed_seconds': time.monotonic() - started}) + '\n')
        print(f'{config["experiment"]}/stage2 epoch={epoch} train_MSE={training:.5f} val_MSE={validation:.5f}', flush=True)
        checkpoints.save(output / 'last.pt', model, optimizer, config, 'stage2', epoch, best)
        if improved:
            checkpoints.save(output / 'best.pt', model, optimizer, config, 'stage2', epoch, best)


def evaluate(config, args, device):
    model = build_model(config).to(device)
    checkpoint = args.checkpoint or path_for(config, 'stage3')
    checkpoints.load_full(model, checkpoint, config)
    model.eval()
    loader = DataLoader(SegmentationDataset(config['data_dir'], args.split, config['image_size']),
                        batch_size=config['batch_size'], num_workers=config['workers'], shuffle=False)
    rows = []
    for snr in config['snrs']:
        seed_all(config['seed'])
        current = dict(config, train_snr=snr)
        result = pass_segmentation(model, loader, device, current, False, args.max_batches)
        rows.append(dict(snr_db=snr, **result))
        print(f'SNR={snr:g} dB mIoU={result["miou"]:.5f} foreground={result["miou_foreground"]:.5f}', flush=True)
    output = Path(config['run_dir']) / 'evaluation'
    write_json(output / f'{args.split}.json', {'config': config, 'arguments': vars_as_json(args), 'checkpoint': str(checkpoint), 'results': rows})
    with (output / f'{args.split}.csv').open('w') as stream:
        fields = ['snr_db', 'miou', 'miou_foreground', 'pixel_accuracy', 'loss']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def evaluate_nmse(config, args, device):
    model = Diffusion('h', config['diffusion']).to(device)
    checkpoint = args.checkpoint or args.diffusion_checkpoint or path_for(config, 'stage2')
    checkpoints.load_diffusion(model, checkpoint)
    model.eval()
    settings, side = config['stage2'], config['diffusion']['side']
    rows = []
    with torch.no_grad():
        for snr in config['snrs']:
            seed_all(config['seed'])
            initial, enhanced, power = 0.0, 0.0, 0.0
            for step, offset in enumerate(range(0, settings['validation_samples'], settings['batch_size'])):
                if args.max_batches is not None and step >= args.max_batches:
                    break
                batch = min(settings['batch_size'], settings['validation_samples'] - offset)
                h = response(batch, side * side, device, config['channel']).reshape(batch, 1, side, side)
                variance = h.abs().square().mean() / (10 ** (snr / 10))
                noisy = h + complex_noise(h, variance)
                parts = model.denoise(torch.cat((noisy.real, noisy.imag)), variance / 2)
                recovered = torch.complex(parts[:batch], parts[batch:])
                initial += float((noisy - h).abs().square().sum())
                enhanced += float((recovered - h).abs().square().sum())
                power += float(h.abs().square().sum())
            row = {'snr_db': snr, 'initial_nmse_db': 10 * math.log10(max(initial / power, 1e-30)),
                   'enhanced_nmse_db': 10 * math.log10(max(enhanced / power, 1e-30))}
            rows.append(row)
            print(row, flush=True)
    output = Path(config['run_dir']) / 'evaluation'
    write_json(output / 'csi_nmse.json', {'config': config, 'checkpoint': str(checkpoint), 'results': rows})
    with (output / 'csi_nmse.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def smoke(config, device):
    config = smoke_config(config)
    model = build_model(config).to(device)
    model.set_stage('stage3')
    images = torch.rand(2, 4, 64, 64, device=device)
    labels = torch.randint(0, 9, (2, 64, 64), device=device)
    logits = model(images, snr=20)
    assert logits.shape == (2, 9, 64, 64)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    if model.ddpm is not None:
        model.ddpm.requires_grad_(True).train()
        diffusion_loss = model.ddpm(synthetic_clean(config['target'], 2, device, config))
        diffusion_loss.backward()
        assert torch.isfinite(diffusion_loss)
    checks = 'backward, frozen modules' + (', diffusion loss' if model.ddpm is not None else '')
    print(f'SMOKE PASS {config["experiment"]}: logits={tuple(logits.shape)}, {checks}', flush=True)


def execute(config, args):
    device = device_for(config['device'])
    seed_all(config['seed'])
    if args.stage == 'smoke':
        smoke(config, device)
        return
    if args.stage in ('all', 'stage1', 'stage3'):
        check_dataset(config['data_dir'], ('train', 'val'))
    elif args.stage == 'eval':
        check_dataset(config['data_dir'], (args.split,))
    for key in ('resume', 'checkpoint', 'stage1_checkpoint', 'diffusion_checkpoint'):
        path = getattr(args, key)
        if path and not path.is_file():
            raise FileNotFoundError(f'{key}: {path}')
    if args.stage == 'all':
        if not args.stage1_checkpoint:
            train_semantic(config, args, device, 'stage1')
            gc.collect()
        if config['target'] != 'none' and not args.diffusion_checkpoint:
            train_diffusion(config, args, device)
            gc.collect()
        train_semantic(config, args, device, 'stage3')
    elif args.stage in ('stage1', 'stage3'):
        train_semantic(config, args, device, args.stage)
    elif args.stage == 'stage2':
        train_diffusion(config, args, device)
    elif args.stage == 'eval':
        evaluate(config, args, device)
    elif args.stage == 'nmse':
        evaluate_nmse(config, args, device)
