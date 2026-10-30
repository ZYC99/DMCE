"""Run the complete non-LDPC experiment DAG on one or more Linux GPUs.

Each process owns a GPU advisory lock. Training uses every batch and the configured
epoch counts; this is not a smoke-test wrapper. An immutable source copy, commands,
logs, and restart state are kept alongside the results. Run under tmux/nohup for
long experiments. Shared servers should supply their existing lock convention.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def make_jobs(source, output, data, config):
    common = ['--config', str(config), '--data-dir', str(data), '--output-dir', str(output / 'runs'), '--device', 'cuda:0']
    jobs = []
    shared = output / 'runs/without_dmce/stage1/best.pt'

    def add(name, experiment, stage, dependencies=(), extra=()):
        jobs.append({'name': name, 'experiment': experiment, 'stage': stage, 'dependencies': list(dependencies),
                     'command': [sys.executable, '-u', str(source / 'experiments' / f'{experiment}.py'),
                                 '--stage', stage, *common, *map(str, extra)]})

    add('shared_stage1', 'without_dmce', 'stage1')
    add('mfnet_stage1', 'mfnet_channel', 'stage1')
    for experiment in ('dmce', 'y_denoising', 'x_denoising'):
        add(f'{experiment}_stage2', experiment, 'stage2')
    for experiment in ('without_dmce', 'dmce', 'y_denoising', 'x_denoising', 'mfnet_channel'):
        dependencies = ['mfnet_stage1' if experiment == 'mfnet_channel' else 'shared_stage1']
        extra = [] if experiment == 'mfnet_channel' else ['--stage1-checkpoint', shared]
        if experiment in ('dmce', 'y_denoising', 'x_denoising'):
            dependencies.append(f'{experiment}_stage2')
        add(f'{experiment}_stage3', experiment, 'stage3', dependencies, extra)
        for split in ('test', 'test_day', 'test_night'):
            add(f'{experiment}_{split}', experiment, 'eval', [f'{experiment}_stage3'], ['--split', split])
    add('dmce_nmse', 'dmce', 'nmse', ['dmce_stage2'])
    return jobs


def gpu_is_busy(gpu):
    result = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True)
    return any(line.strip().isdigit() for line in result.stdout.splitlines())


def acquire_gpu(gpu, template):
    path = Path(template.format(gpu=gpu))
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if gpu_is_busy(gpu):
            handle.close()
            return None
        return handle
    except BlockingIOError:
        handle.close()
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, help='Optional JSON override; default paper-sized configuration')
    parser.add_argument('--gpus', type=int, nargs='+', required=True)
    parser.add_argument('--gpu-lock-template', default='/tmp/dmce-gpu{gpu}.lock')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        parser.error('GPU indices must be unique and nonnegative')
    if '{gpu}' not in args.gpu_lock_template:
        parser.error('--gpu-lock-template must contain {gpu}')
    output, data = args.output_dir.resolve(), args.data_dir.resolve()
    source, config_path = output / 'source', output / 'override.json'
    jobs = make_jobs(source, output, data, config_path)
    override = json.loads(args.config.read_text()) if args.config else {}
    if args.dry_run:
        print(json.dumps({'override': override, 'jobs': jobs}, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    controller_lock = (output / 'controller.lock').open('a')
    try:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('Another suite controller already owns this output directory')
    state_path = output / 'status.json'
    if state_path.exists() and not args.resume:
        parser.error('Output already exists; use --resume or choose another directory')
    if args.resume:
        if not state_path.is_file() or not source.is_dir():
            parser.error('--resume requires an existing suite with its source snapshot')
        if json.loads(config_path.read_text()) != override:
            parser.error('Resuming requires the original configuration override')
        state = json.loads(state_path.read_text())
        if state['jobs'] != jobs:
            parser.error('Resume data path, Python executable, or job plan differs')
        for record in state['results'].values():
            if record['status'] != 'completed':
                record['status'] = 'pending'
    else:
        source.mkdir()
        for name in ('dmce', 'experiments', 'configs', 'scripts'):
            shutil.copytree(ROOT / name, source / name, ignore=shutil.ignore_patterns('__pycache__'))
        hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(source.rglob('*')) if p.is_file()}
        write_json(output / 'source_sha256.json', hashes)
        write_json(config_path, override)
        state = {'created_at': now(), 'jobs': jobs,
                 'results': {job['name']: {'status': 'pending'} for job in jobs}}
    (output / 'logs').mkdir(exist_ok=True)
    running = {}
    try:
        while True:
            for gpu, active in list(running.items()):
                process, lock, log, name = active
                code = process.poll()
                if code is None:
                    continue
                log.close()
                lock.close()
                state['results'][name].update(status='completed' if code == 0 else 'failed',
                                               returncode=code, finished_at=now())
                print(f'{now()} {name} finished: {code}', flush=True)
                del running[gpu]
            for job in jobs:
                record = state['results'][job['name']]
                if record['status'] != 'pending':
                    continue
                if any(state['results'][dep]['status'] in ('failed', 'blocked') for dep in job['dependencies']):
                    record.update(status='blocked', reason='A prerequisite failed')
                    continue
                if not all(state['results'][dep]['status'] == 'completed' for dep in job['dependencies']):
                    continue
                for gpu in args.gpus:
                    if gpu in running:
                        continue
                    lock = acquire_gpu(gpu, args.gpu_lock_template)
                    if lock is None:
                        continue
                    command = list(job['command'])
                    last = output / 'runs' / job['experiment'] / job['stage'] / 'last.pt'
                    if args.resume and job['stage'].startswith('stage') and last.is_file():
                        # Check the atomic checkpoint, not a possibly-ahead metrics log.
                        import torch
                        payload = torch.load(last, map_location='cpu', weights_only=True)
                        finished = payload['epoch'] >= payload['config'][job['stage']]['epochs']
                        del payload
                        if finished:
                            record.update(status='completed', recovered_from_checkpoint=True, finished_at=now())
                            lock.close()
                            break
                        command += ['--resume', str(last)]
                    log = (output / 'logs' / f'{job["name"]}.log').open('a')
                    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                                       MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
                    process = subprocess.Popen(command, cwd=source, env=environment,
                                               stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
                    record.update(status='running', gpu=gpu, pid=process.pid, started_at=now(), command=command)
                    running[gpu] = process, lock, log, job['name']
                    print(f'{now()} {job["name"]} started on GPU {gpu}: pid={process.pid}', flush=True)
                    break
            state.update(updated_at=now(), controller_pid=os.getpid())
            write_json(state_path, state)
            if not running and not any(r['status'] == 'pending' for r in state['results'].values()):
                failed = any(r['status'] != 'completed' for r in state['results'].values())
                print('SUITE FAILED' if failed else 'SUITE COMPLETED', flush=True)
                return int(failed)
            time.sleep(5)
    finally:
        # Children retain their GPU lock if the controller is interrupted. A
        # resumed controller waits for those processes before launching more work.
        controller_lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
