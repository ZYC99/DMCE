"""Explicit checkpoint loading; incompatible models never silently use random weights."""
import random
from pathlib import Path

import numpy as np
import torch


def read(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint does not exist: {path}')
    # Only tensors and simple metadata are written by these entry points.
    value = torch.load(path, map_location='cpu', weights_only=True)
    return value if 'model' in value else {'model': value}


def load_semantic(model, path):
    payload = read(path)
    if payload.get('stage', 'stage1') != 'stage1':
        raise ValueError('Semantic initialization requires a stage1 checkpoint')
    state = {key: value for key, value in payload['model'].items() if not key.startswith('ddpm.')}
    result = model.load_state_dict(state, strict=False)
    missing = [key for key in result.missing_keys if not key.startswith('ddpm.')]
    if missing or result.unexpected_keys:
        raise ValueError(f'Incompatible semantic checkpoint: missing={missing}, unexpected={result.unexpected_keys}')


def load_diffusion(model, path):
    payload = read(path)
    if 'target' in payload and payload['target'] != model.target:
        raise ValueError(f'Denoiser target mismatch: {payload["target"]} != {model.target}')
    state = payload['model']
    if any(key.startswith('ddpm.') for key in state):
        state = {key[5:]: value for key, value in state.items() if key.startswith('ddpm.')}
    if 'stage' in payload:
        if payload['stage'] != 'stage2':
            raise ValueError('Expected a stage2 denoiser checkpoint')
        if payload['config']['diffusion'] != model.config:
            raise ValueError('Diffusion configuration differs from checkpoint; use its original diffusion settings')
        model.load_state_dict(state, strict=True)
    else:
        # Legacy schedules had an off-by-one error. Load the predictor, retain the fixed schedule.
        predictor = {key[len('eps_model.'):]: value for key, value in state.items() if key.startswith('eps_model.')}
        if not predictor:
            raise ValueError('No eps_model weights found in the legacy diffusion checkpoint')
        model.eps_model.load_state_dict(predictor, strict=True)
        print('Loaded legacy eps_model weights; using corrected diffusion schedule. Verify/retrain before comparison.')


def load_full(model, path, config):
    payload = read(path)
    if payload.get('experiment', config['experiment']) != config['experiment']:
        raise ValueError('Evaluation checkpoint belongs to a different experiment')
    if payload.get('stage', 'stage3') != 'stage3':
        raise ValueError('Channel evaluation requires a stage3 model')
    if config['target'] != 'none' and 'config' in payload and payload['config']['diffusion'] != config['diffusion']:
        raise ValueError('Diffusion configuration differs from evaluation checkpoint')
    model.load_state_dict(payload['model'], strict=True)


def save(path, model, optimizer, config, stage, epoch, best):
    numpy_state = np.random.get_state()
    state = model.state_dict()
    if stage == 'stage1':
        state = {key: value for key, value in state.items() if not key.startswith('ddpm.')}
    payload = {
        'model': state, 'optimizer': optimizer.state_dict(), 'config': config,
        'experiment': config['experiment'], 'target': config['target'], 'stage': stage,
        'epoch': epoch, 'best': best, 'torch_rng': torch.get_rng_state(),
        'python_rng': random.getstate(),
        'numpy_rng': [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def resume(path, model, optimizer, config, stage):
    payload = read(path)
    if payload.get('experiment') != config['experiment'] or payload.get('stage') != stage:
        raise ValueError('Resume checkpoint experiment/stage mismatch')
    if stage != 'stage1' and config['target'] != 'none' and payload['config']['diffusion'] != config['diffusion']:
        raise ValueError('Cannot resume with different diffusion settings')
    if stage == 'stage1':
        load_semantic(model, path)
    else:
        model.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    torch.set_rng_state(payload['torch_rng'])
    random.setstate(payload['python_rng'])
    ns = payload['numpy_rng']
    np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), *ns[2:]))
    if torch.cuda.is_available() and payload['cuda_rng']:
        torch.cuda.set_rng_state_all(payload['cuda_rng'])
    return payload['epoch'] + 1, payload['best']
