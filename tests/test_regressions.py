import copy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn

from dmce import checkpoints
from dmce.channel import transmit, normalize_power
from dmce.config import EXPERIMENTS, ROOT, parse, smoke_config
from dmce.data import Metrics, SegmentationDataset, check_dataset
from dmce.diffusion import Diffusion
from dmce.models import build_model


class ZeroPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.times = []

    def forward(self, x, t):
        self.times.append(t.clone())
        return torch.zeros_like(x)


class IdentityDenoiser:
    def denoise(self, x, variance):
        return x


class RegressionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.config = smoke_config(parse('dmce', ['--dry-run'])[1])

    def test_all_entry_points_work_from_another_directory(self):
        for name in EXPERIMENTS:
            result = subprocess.run([sys.executable, str(ROOT / 'experiments' / f'{name}.py'), '--dry-run'],
                                    cwd='/tmp', text=True, capture_output=True, check=True)
            self.assertEqual(json.loads(result.stdout)['config']['experiment'], name)

    def test_variants_do_not_share_outputs_or_accept_wrong_target(self):
        directories = [parse(name, [])[1]['run_dir'] for name in EXPERIMENTS]
        self.assertEqual(len(set(directories)), 5)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse('without_dmce', ['--stage', 'stage2'])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parse('y_denoising', ['--stage', 'nmse'])

    def test_ir_uses_its_own_received_signal(self):
        # Opposite inputs remain opposite through a nearly noiseless channel. The old X branch
        # mistakenly recovered both users from the RGB received signal.
        symbols = torch.stack((torch.ones(2, 64), -torch.ones(2, 64)), dim=1)
        channel = dict(self.config['channel'], diffusion_side=70)
        for target in ('none', 'h', 'y', 'x'):
            recovered, _ = transmit(symbols, 140, target, IdentityDenoiser(), channel)
            torch.testing.assert_close(recovered, symbols, atol=1e-5, rtol=1e-5)

    def test_power_normalization_is_per_sample(self):
        normalized = normalize_power(torch.tensor([[[1.0, 1.0]], [[7.0, 7.0]]]))
        torch.testing.assert_close(normalized.square().mean((1, 2)), torch.ones(2))
        self.assertTrue(torch.isfinite(normalize_power(torch.zeros(2, 1, 8))).all())

    def test_sampling_time_matches_training_and_zero_noise_is_identity(self):
        model = Diffusion('h', self.config['diffusion'])
        predictor = ZeroPredictor()
        model.eps_model = predictor
        x = torch.randn(2, 1, 8, 8)
        torch.testing.assert_close(model.denoise(x, 0), x)
        variance = (1 - model.alphabar_t[2]) / model.alphabar_t[2]
        torch.testing.assert_close(model.denoise(x, variance), x)
        self.assertAlmostEqual(float(predictor.times[0][0]), 2 / model.n_T)

    def test_iou_counts_background_false_positives(self):
        metrics = Metrics(2)
        labels = torch.tensor([[[0, 1, 1]]])
        predictions = torch.tensor([[[1, 1, 0]]])
        logits = torch.nn.functional.one_hot(predictions, 2).permute(0, 3, 1, 2).float()
        metrics.update(logits, labels)
        self.assertAlmostEqual(metrics.result()['iou_per_class'][1], 1 / 3)

    def test_fourth_image_channel_is_not_alpha_and_labels_remain_discrete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'images').mkdir()
            (root / 'labels').mkdir()
            (root / 'train.txt').write_text('sample\n')
            image = np.zeros((4, 4, 4), dtype=np.uint8)
            image[..., :3] = 100  # IR=0 must not erase RGB during resizing.
            label = np.zeros((4, 4), dtype=np.uint8)
            label[:, 2:] = 8
            Image.fromarray(image).save(root / 'images/sample.png')
            Image.fromarray(label).save(root / 'labels/sample.png')
            images, labels, _ = SegmentationDataset(root, 'train', (8, 8))[0]
            self.assertTrue(torch.allclose(images[:3], torch.full_like(images[:3], 100 / 255)))
            self.assertEqual(set(labels.unique().tolist()), {0, 8})
            (root / 'train.txt').write_text('sample_flip\n')
            check_dataset(root, ('train',))
            _, flipped_label, _ = SegmentationDataset(root, 'train', (8, 8))[0]
            torch.testing.assert_close(flipped_label, labels.flip(1))

    def test_frozen_encoder_batchnorm_stays_frozen(self):
        config = copy.deepcopy(self.config)
        config.update(backbone='mfnet', target='none')
        model = build_model(config)
        model.set_stage('stage3')
        running_mean = model.conv1_rgb.bn.running_mean.clone()
        model(torch.rand(2, 4, 64, 64), clean=True).mean().backward()
        torch.testing.assert_close(running_mean, model.conv1_rgb.bn.running_mean)
        self.assertIsNone(model.conv1_rgb.conv.weight.grad)
        self.assertIsNotNone(model.decode1.conv.weight.grad)

    def test_checkpoint_resume_restores_optimizer_epoch_and_rejects_wrong_experiment(self):
        config = copy.deepcopy(self.config)
        config.update(experiment='mfnet_channel', backbone='mfnet', target='none')
        model = build_model(config)
        model.set_stage('stage1')
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pt'
            checkpoints.save(path, model, optimizer, config, 'stage1', 3, 0.25)
            epoch, best = checkpoints.resume(path, model, optimizer, config, 'stage1')
            self.assertEqual((epoch, best), (4, 0.25))
            with self.assertRaises(ValueError):
                checkpoints.resume(path, model, optimizer, dict(config, experiment='dmce'), 'stage1')

    def test_denoiser_rejects_a_different_schedule_or_target(self):
        model = Diffusion('h', self.config['diffusion'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'denoiser.pt'
            torch.save({'model': model.state_dict(), 'stage': 'stage2',
                        'target': 'h', 'config': self.config}, path)
            checkpoints.load_diffusion(model, path)
            wrong_schedule = dict(self.config['diffusion'], beta_end=0.03)
            with self.assertRaises(ValueError):
                checkpoints.load_diffusion(Diffusion('h', wrong_schedule), path)
            with self.assertRaises(ValueError):
                checkpoints.load_diffusion(Diffusion('x', self.config['diffusion']), path)


if __name__ == '__main__':
    unittest.main()
