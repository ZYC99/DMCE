"""Configuration and CLI parsing deliberately require only the standard library."""
import argparse
import copy
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ('dmce', 'without_dmce', 'y_denoising', 'x_denoising', 'mfnet_channel')


def merge(base, override):
    for key, value in override.items():
        if key not in base:
            raise ValueError(f'Unknown configuration key: {key}')
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f'{key} must be an object')
            merge(base[key], value)
        else:
            base[key] = value
    return base


def parse(experiment, argv=None):
    parser = argparse.ArgumentParser(description=f'Independent DMCE experiment: {experiment}')
    parser.add_argument('--stage', choices=('all', 'stage1', 'stage2', 'stage3', 'eval', 'nmse', 'smoke'), default='all')
    parser.add_argument('--config', type=Path, help='JSON overrides; unknown keys are rejected')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, help='Base directory; the experiment name is appended')
    parser.add_argument('--device', help='auto, cpu, or cuda[:index]')
    parser.add_argument('--epochs', type=int, help='Override all training-stage epoch counts')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--train-snr', type=float)
    parser.add_argument('--snrs', nargs='+', type=float)
    parser.add_argument('--split', choices=('test', 'test_day', 'test_night', 'val'), default='test')
    parser.add_argument('--stage1-checkpoint', type=Path, help='Reuse a clean semantic checkpoint')
    parser.add_argument('--diffusion-checkpoint', type=Path, help='Reuse a stage2 denoiser')
    parser.add_argument('--checkpoint', type=Path, help='Full model for eval; denoiser for nmse')
    parser.add_argument('--resume', type=Path, help='Resume one explicit training stage from last.pt')
    parser.add_argument('--max-batches', type=int, help='Bound each train/eval pass for debugging')
    parser.add_argument('--no-pretrained', action='store_true', help='Do not download ImageNet weights')
    parser.add_argument('--dry-run', action='store_true', help='Print resolved configuration without importing torch')
    args = parser.parse_args(argv)
    config = json.loads((ROOT / 'configs' / 'base.json').read_text())
    experiment_config = json.loads((ROOT / 'configs' / f'{experiment}.json').read_text())
    merge(config, experiment_config)
    if args.config:
        merge(config, json.loads(args.config.read_text()))
    for key in ('experiment', 'target', 'backbone'):
        if config[key] != experiment_config[key]:
            parser.error(f'{key} is fixed by this entry point')
    for key in ('data_dir', 'output_dir', 'device', 'batch_size', 'workers', 'seed', 'train_snr', 'snrs'):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    if args.epochs is not None:
        for stage in ('stage1', 'stage2', 'stage3'):
            config[stage]['epochs'] = args.epochs
    if args.no_pretrained:
        config['pretrained'] = False
    for key in ('data_dir', 'output_dir'):
        # Defaults/config paths are relative to this independent directory; CLI paths use cwd.
        path = Path(config[key]).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() if getattr(args, key) is not None else ROOT) / path
        config[key] = str(path.resolve())
    config['run_dir'] = str(Path(config['output_dir']) / experiment)
    if args.stage == 'stage2' and config['target'] == 'none':
        parser.error('This experiment has no diffusion stage')
    if args.stage == 'nmse' and config['target'] != 'h':
        parser.error('CSI NMSE evaluation is available only for the dmce entry point')
    if args.resume and args.stage not in ('stage1', 'stage2', 'stage3'):
        parser.error('--resume requires an explicit training stage')
    if args.checkpoint and args.stage not in ('eval', 'nmse'):
        parser.error('--checkpoint is only for eval/nmse')
    if args.diffusion_checkpoint and config['target'] == 'none':
        parser.error('This experiment has no diffusion checkpoint')
    validate(config)
    if args.max_batches is not None and args.max_batches < 1:
        parser.error('--max-batches must be positive')
    return args, config


def validate(config):
    for key in ('batch_size', 'classes'):
        if config[key] < 1:
            raise ValueError(f'{key} must be positive')
    if config['classes'] != 9 or config['workers'] < 0:
        raise ValueError('This dataset requires 9 classes and nonnegative workers')
    height, width = config['image_size']
    if min(height, width) < 32 or height % 32 or width % 32:
        raise ValueError('image_size must contain positive multiples of 32')
    diffusion = config['diffusion']
    if height // 32 * (width // 32) * 16 > diffusion['side'] ** 2:
        raise ValueError('Image feature count exceeds diffusion grid capacity')
    if not 0 < diffusion['beta_start'] < diffusion['beta_end'] < 1 or diffusion['steps'] < 1:
        raise ValueError('Invalid diffusion schedule')
    if diffusion['inner_channels'] % diffusion['norm_groups'] or diffusion['inner_channels'] % 2:
        raise ValueError('UNet channels must be even and divisible by norm_groups')
    for stage in ('stage1', 'stage2', 'stage3'):
        if config[stage]['epochs'] < 1 or config[stage]['lr'] <= 0:
            raise ValueError(f'Invalid {stage} epochs/lr')
    for key in ('samples', 'validation_samples', 'batch_size'):
        if config['stage2'][key] < 1:
            raise ValueError(f'stage2.{key} must be positive')
    if not config['snrs'] or not all(math.isfinite(x) for x in config['snrs'] + [config['train_snr']]):
        raise ValueError('SNRs must be finite and nonempty')
    if not 0 < config['channel']['distance_min'] < config['channel']['distance_max']:
        raise ValueError('Invalid channel distance interval')


def smoke_config(config):
    config = copy.deepcopy(config)
    config.update(image_size=[64, 64], batch_size=2, workers=0, pretrained=False)
    config['diffusion'].update(steps=4, inner_channels=8, norm_groups=8,
                               channel_mults=[1, 2], res_blocks=1, attention_resolutions=[])
    return config
