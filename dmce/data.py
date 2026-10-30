from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def image_path(root, folder, name):
    path = Path(root) / folder / f'{name}.png'
    if not path.is_file() and name.endswith('_flip'):
        return Path(root) / folder / f'{name[:-5]}.png', True
    return path, False


def check_dataset(root, splits):
    root = Path(root)
    for split in splits:
        listing = root / f'{split}.txt'
        if not listing.is_file():
            raise FileNotFoundError(f'Missing split list: {listing}')
        names = [name.strip() for name in listing.read_text().splitlines() if name.strip()]
        if not names:
            raise ValueError(f'Empty split: {listing}')
        missing = [str(image_path(root, folder, name)[0]) for name in names
                   for folder in ('images', 'labels') if not image_path(root, folder, name)[0].is_file()]
        if missing:
            raise FileNotFoundError(f'{split}: missing {len(missing)} image/label files; first: {missing[0]}')


class SegmentationDataset(Dataset):
    def __init__(self, root, split, size, augment=False):
        self.root = Path(root)
        self.names = [s.strip() for s in (self.root / f'{split}.txt').read_text().splitlines() if s.strip()]
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        image_file, image_flip = image_path(self.root, 'images', name)
        label_file, label_flip = image_path(self.root, 'labels', name)
        with Image.open(image_file) as source:
            image = np.asarray(source).copy()
        with Image.open(label_file) as source:
            label = np.asarray(source).copy()
        if image_flip:
            image = image[:, ::-1]
        if label_flip:
            label = label[:, ::-1]
        if image.ndim != 3 or image.shape[2] != 4 or label.ndim != 2:
            raise ValueError(f'{name}: expected RGB-IR [H,W,4] and class IDs [H,W]')
        if image.shape[:2] != label.shape:
            raise ValueError(f'{name}: image and label sizes differ')
        if self.augment:
            if np.random.random() < 0.5:
                image, label = image[:, ::-1], label[:, ::-1]
            height, width = label.shape
            top, left = np.random.randint(0, max(1, height // 10)), np.random.randint(0, max(1, width // 10))
            bottom = np.random.randint(max(top + 1, height - height // 10), height + 1)
            right = np.random.randint(max(left + 1, width - width // 10), width + 1)
            image, label = image[top:bottom, left:right], label[top:bottom, left:right]
        height, width = self.size
        # Four channels are independent measurements; RGBA resize would premultiply RGB by IR.
        image = np.stack([np.asarray(Image.fromarray(image[..., c]).resize(
            (width, height), Image.Resampling.BILINEAR)) for c in range(4)], axis=0).astype(np.float32) / 255
        label = np.asarray(Image.fromarray(label).resize((width, height), Image.Resampling.NEAREST)).astype(np.int64)
        label[label >= 9] = 0  # Original dataset convention.
        return torch.from_numpy(image), torch.from_numpy(label), name


class Metrics:
    def __init__(self, classes):
        self.confusion = torch.zeros(classes, classes, dtype=torch.int64)

    def update(self, logits, labels):
        classes = len(self.confusion)
        valid = (labels >= 0) & (labels < classes)
        encoded = labels[valid] * classes + logits.argmax(1)[valid]
        self.confusion += torch.bincount(encoded.cpu(), minlength=classes * classes).reshape(classes, classes)

    def result(self):
        matrix = self.confusion.double()
        intersection = matrix.diag()
        union = matrix.sum(0) + matrix.sum(1) - intersection
        iou = intersection / union.clamp_min(1)
        # Absent classes contribute 0, retaining the fixed nine-class denominator in Eq. (18).
        return {
            'pixel_accuracy': float(intersection.sum() / matrix.sum().clamp_min(1)),
            'miou': float(iou.mean()), 'miou_foreground': float(iou[1:].mean()),
            'iou_per_class': iou.tolist(), 'confusion_matrix': self.confusion.tolist(),
        }
