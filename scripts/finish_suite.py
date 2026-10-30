"""After a complete suite, audit checkpoints, draw figures, and benchmark inference.

With --wait this can run beside the training controller. It waits for successful
completion, then acquires an otherwise idle GPU using the same lock convention.
Runtime logs with local paths stay in the suite directory, not public results.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_suite import acquire_gpu, now, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', required=True)
    parser.add_argument('--gpu-lock-template', default='/tmp/dmce-gpu{gpu}.lock')
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--allow-bounded', action='store_true', help='Label custom/short runs as bounded in the final reports')
    args = parser.parse_args()
    if not args.gpus or min(args.gpus) < 0 or '{gpu}' not in args.gpu_lock_template:
        parser.error('Invalid GPU indices or lock template')
    suite, output, data = args.suite_dir.resolve(), args.output_dir.resolve(), args.data_dir.resolve()
    state_path, status_path = suite / 'status.json', suite / 'postprocess_status.json'
    record = {'pid': os.getpid(), 'started_at': now(), 'status': 'waiting_training', 'steps': []}
    # Separate controller lock also prevents duplicate post-processing processes.
    import fcntl
    process_lock = (suite / 'postprocess.lock').open('a')
    try:
        fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('A post-processing controller already owns this suite')
    try:
        while True:
            state = json.loads(state_path.read_text())
            statuses = [value['status'] for value in state['results'].values()]
            if any(status in ('failed', 'blocked') for status in statuses):
                raise RuntimeError('Training/evaluation failed; inspect suite logs before collecting results')
            if statuses and all(status == 'completed' for status in statuses):
                break
            if not args.wait:
                raise RuntimeError('Suite is incomplete; use --wait to wait for it')
            try:
                os.kill(state['controller_pid'], 0)
            except ProcessLookupError:
                # The controller can finish between our status read and liveness check.
                latest = json.loads(state_path.read_text())
                if latest['results'] and all(value['status'] == 'completed' for value in latest['results'].values()):
                    break
                raise RuntimeError('Suite controller stopped before completion; resume the suite first')
            record['updated_at'] = now()
            write_json(status_path, record)
            time.sleep(15)

        def run(name, arguments, gpu=None, gpu_lock=None):
            record.update(status=name, updated_at=now())
            write_json(status_path, record)
            environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
            if gpu is not None:
                environment['CUDA_VISIBLE_DEVICES'] = str(gpu)
            started = time.monotonic()
            with (suite / 'logs' / f'postprocess_{name}.log').open('w') as stream:
                completed = subprocess.run([sys.executable, '-u', *map(str, arguments)], cwd=ROOT, env=environment,
                                           stdout=stream, stderr=subprocess.STDOUT,
                                           pass_fds=() if gpu_lock is None else (gpu_lock.fileno(),))
            record['steps'].append({'name': name, 'returncode': completed.returncode,
                                    'seconds': time.monotonic() - started})
            write_json(status_path, record)
            completed.check_returncode()

        run('collect', [ROOT / 'scripts/collect_results.py', '--suite-dir', suite, '--output-dir', output])
        record.update(status='waiting_gpu', updated_at=now())
        write_json(status_path, record)
        selected_gpu, lock = None, None
        while lock is None:
            for gpu in args.gpus:
                lock = acquire_gpu(gpu, args.gpu_lock_template)
                if lock is not None:
                    selected_gpu = gpu
                    break
            if lock is None:
                time.sleep(15)
        try:
            common = ['--run-dir', suite / 'runs', '--data-dir', data, '--device', 'cuda:0']
            run('qualitative', [ROOT / 'scripts/export_qualitative.py', *common, '--output-dir', output / 'qualitative'], selected_gpu, lock)
            run('latency', [ROOT / 'scripts/benchmark_inference.py', *common, '--output', output / 'latency.json'], selected_gpu, lock)
        finally:
            lock.close()
        summary = [ROOT / 'scripts/write_results_summary.py', '--results-dir', output]
        if args.allow_bounded:
            summary.append('--allow-bounded')
        run('summary', summary)
        record.update(status='completed', finished_at=now(), gpu=selected_gpu)
        write_json(status_path, record)
        print('POSTPROCESS COMPLETED', flush=True)
    except Exception as error:
        record.update(status='failed', finished_at=now(), error=str(error))
        write_json(status_path, record)
        raise
    finally:
        process_lock.close()


if __name__ == '__main__':
    main()
