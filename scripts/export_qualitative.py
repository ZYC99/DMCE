"""Export a fixed, explicitly recorded day/night comparison at 4 dB (Fig. 5 task)."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dmce import checkpoints
from dmce.config import EXPERIMENTS
from dmce.data import SegmentationDataset
from dmce.models import build_model
from dmce.runner import device_for, seed_all

PALETTE = np.array([[0, 0, 0], [64, 0, 128], [64, 64, 0], [0, 128, 192], [0, 0, 192],
                    [128, 128, 0], [64, 64, 128], [192, 128, 128], [192, 64, 0]], dtype=np.uint8)
CLASSES = ['unlabelled', 'car', 'person', 'bike', 'curve', 'car stop', 'guardrail', 'color cone', 'bump']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='Directory containing dmce/, without_dmce/, ...')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--snr', type=float, default=4)
    parser.add_argument('--samples-per-split', type=int, default=2)
    parser.add_argument('--experiments', nargs='+', choices=EXPERIMENTS, default=list(EXPERIMENTS))
    args = parser.parse_args()
    if args.samples_per_split < 1:
        parser.error('--samples-per-split must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = device_for(args.device)
    records, panels = {}, {}
    expected_samples = None
    for experiment in args.experiments:
        path = args.run_dir / experiment / 'stage3/best.pt'
        payload = checkpoints.read(path)
        config = payload['config']
        model = build_model(config).to(device)
        if payload.get('stage') != 'stage3' or payload.get('experiment') != experiment:
            raise ValueError(f'Wrong experiment or stage: {path}')
        model.load_state_dict(payload['model'], strict=True)
        model.eval()
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        records[experiment] = {'epoch': payload['epoch'], 'best_validation_miou': payload['best'],
                               'checkpoint_sha256': digest}
        del payload
        samples = []
        for split in ('test_day', 'test_night'):
            dataset = SegmentationDataset(args.data_dir, split, config['image_size'])
            for index in range(min(args.samples_per_split, len(dataset))):
                tensor, label, name = dataset[index]
                samples.append(name)
                # Same channel/noise seed across ResNet methods for the same sample.
                seed_all(config['seed'] + index + (10000 if split == 'test_night' else 0))
                with torch.inference_mode():
                    prediction = model(tensor[None].to(device), args.snr).argmax(1)[0].cpu().numpy().astype(np.uint8)
                directory = args.output_dir / name
                directory.mkdir(exist_ok=True)
                rgb = Image.fromarray(np.round(tensor[:3].numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                infrared = Image.fromarray(np.round(tensor[3].numpy() * 255).astype(np.uint8)).convert('RGB')
                truth = Image.fromarray(PALETTE[label.numpy()])
                coloured = Image.fromarray(PALETTE[prediction])
                for title, picture in [('rgb', rgb), ('infrared', infrared), ('ground_truth', truth), (experiment, coloured)]:
                    picture.save(directory / f'{title}.png')
                Image.fromarray(prediction).save(directory / f'{experiment}_ids.png')
                if name not in panels:
                    panels[name] = [('RGB', rgb), ('IR', infrared), ('Ground truth', truth)]
                panels[name].append((experiment, coloured))
        if expected_samples is None:
            expected_samples = samples
        elif samples != expected_samples:
            raise ValueError('Experiments used inconsistent sample lists')
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    for name, columns in panels.items():
        width, height = columns[0][1].size
        canvas = Image.new('RGB', (width * len(columns), height + 32), 'white')
        drawing = ImageDraw.Draw(canvas)
        for index, (title, picture) in enumerate(columns):
            canvas.paste(picture, (index * width, 32))
            drawing.text((index * width + 8, 8), title, fill='black')
        canvas.save(args.output_dir / f'{name}_comparison.png')
    metadata = {'snr_db': args.snr, 'selection': 'first N entries of official test_day/test_night, without cherry-picking',
                'samples': expected_samples, 'experiments': records, 'classes': CLASSES, 'palette': PALETTE.tolist()}
    (args.output_dir / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Exported {len(panels)} comparisons to {args.output_dir}')


if __name__ == '__main__':
    main()
