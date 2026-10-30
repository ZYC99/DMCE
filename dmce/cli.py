import json
import sys

from .config import parse


def main(experiment, argv=None):
    try:
        args, config = parse(experiment, argv)
        if args.dry_run:
            stages = ['stage1'] + (['stage2'] if config['target'] != 'none' else []) + ['stage3']
            print(json.dumps({'stage': args.stage, 'training_stages': stages, 'config': config}, indent=2, ensure_ascii=False))
            return
        from .runner import execute
        execute(config, args)
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(2) from error
