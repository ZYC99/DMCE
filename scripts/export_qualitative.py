"""Export a fixed, explicitly recorded day/night comparison at 4 dB (Fig. 5 task)."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
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
TITLES = {'dmce': 'DMCE (CSI)', 'without_dmce': 'Without DMCE', 'y_denoising': 'Y-oriented',
          'x_denoising': 'X-oriented (extension)', 'mfnet_channel': 'MFNet with channel'}


def comparison_grid(name, panels, snr):
    width = 320
    source_width, source_height = panels[0][1].size
    height = round(source_height * width / source_width)
    columns, gap, heading, caption = min(4, len(panels)), 10, 44, 34
    rows = (len(panels) + columns - 1) // columns
    canvas_width = columns * (width + gap) + gap
    legend_rows = (len(CLASSES) + columns * 2 - 1) // (columns * 2)
    canvas_height = heading + rows * (height + caption + gap) + legend_rows * 28 + gap
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    drawing = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=20)
    legend_font = ImageFont.load_default(size=16)
    drawing.text((gap, 10), f'{name} | SNR {snr:g} dB', fill='black', font=font)
    for index, (title, picture) in enumerate(panels):
        x = gap + (index % columns) * (width + gap)
        y = heading + (index // columns) * (height + caption + gap)
        resampling = Image.Resampling.BILINEAR if title in ('RGB', 'IR') else Image.Resampling.NEAREST
        canvas.paste(picture.resize((width, height), resampling), (x, y + caption))
        drawing.text((x, y + 4), TITLES.get(title, title), fill='black', font=font)
    legend_y = heading + rows * (height + caption + gap)
    cell_width = (canvas_width - 2 * gap) // (columns * 2)
    for index, (title, colour) in enumerate(zip(CLASSES, PALETTE)):
        x = gap + index % (columns * 2) * cell_width
        y = legend_y + index // (columns * 2) * 28
        drawing.rectangle((x, y + 3, x + 15, y + 18), fill=tuple(map(int, colour)))
        drawing.text((x + 21, y + 2), title, fill='black', font=legend_font)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True, help='Directory containing dmce/, without_dmce/, ...')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--snr', type=float, default=4)
    parser.add_argument('--day-samples', type=int, default=3, help='First N official day examples (Fig. 5: 3)')
    parser.add_argument('--night-samples', type=int, default=2, help='First N official night examples (Fig. 5: 2)')
    parser.add_argument('--samples-per-split', type=int, help='Override both counts with the same N for custom exports')
    parser.add_argument('--experiments', nargs='+', choices=EXPERIMENTS, default=list(EXPERIMENTS))
    args = parser.parse_args()
    if args.samples_per_split is not None:
        args.day_samples = args.night_samples = args.samples_per_split
    if min(args.day_samples, args.night_samples) < 1:
        parser.error('Sample counts must be positive')
    counts = {'test_day': args.day_samples, 'test_night': args.night_samples}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = device_for(args.device)
    records, panels = {}, {}
    expected_samples = None
    expected_by_split = None
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
        samples_by_split = {'test_day': [], 'test_night': []}
        for split in ('test_day', 'test_night'):
            dataset = SegmentationDataset(args.data_dir, split, config['image_size'])
            for index in range(min(counts[split], len(dataset))):
                tensor, label, name = dataset[index]
                samples.append(name)
                samples_by_split[split].append(name)
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
            expected_by_split = samples_by_split
        elif samples != expected_samples or samples_by_split != expected_by_split:
            raise ValueError('Experiments used inconsistent sample lists')
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    for name, columns in panels.items():
        comparison_grid(name, columns, args.snr).save(args.output_dir / f'{name}_comparison.png')
    metadata = {'snr_db': args.snr, 'selection': 'first N entries of official test_day/test_night, without cherry-picking',
                'samples': expected_samples, 'samples_by_split': expected_by_split,
                'experiments': records, 'classes': CLASSES, 'palette': PALETTE.tolist(),
                'display': 'four-column grid, 320px panels; nearest-neighbor label display; native predictions saved separately'}
    (args.output_dir / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Exported {len(panels)} comparisons to {args.output_dir}')


if __name__ == '__main__':
    main()
