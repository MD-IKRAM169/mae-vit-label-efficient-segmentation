"""ADE20K and Cityscapes dataset loaders; no dataset redistribution.

ADE20K uses 0 as ignore, class IDs 1..150 mapped to 0..149.
Cityscapes gtFine labelIds mapped to 19 official train IDs.
"""
from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset

CITYSCAPES_LABEL_IDS = (7, 8, 11, 12, 13, 17, 19, 20, 21,
                        22, 23, 24, 25, 26, 27, 28, 31, 32, 33)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class SegmentationDataset(Dataset):
    def __init__(self, root, dataset: str, split="train", image_size=224,
                 images_only=False, augment=False):
        self.root = Path(root)
        self.dataset = dataset.lower()
        self.split = split
        self.image_size = image_size
        self.images_only = images_only
        self.augment = augment
        if self.dataset == "ade20k":
            d = {"train": "training", "val": "validation"}.get(split, split)
            image_dir = self.root / "images" / d
            self.image_files = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
            self.mask_files = [self.root / "annotations" / d / (p.stem + ".png")
                               for p in self.image_files]
        elif self.dataset == "cityscapes":
            image_dir = self.root / "leftImg8bit" / split
            self.image_files = sorted(image_dir.rglob("*_leftImg8bit.png"))
            self.mask_files = []
            for p in self.image_files:
                prefix = p.name.removesuffix("_leftImg8bit.png")
                mask_dir = self.root / "gtFine" / split / p.parent.name
                mapped = mask_dir / (prefix + "_gtFine_labelTrainIds.png")
                original = mask_dir / (prefix + "_gtFine_labelIds.png")
                self.mask_files.append(mapped if mapped.exists() else original)
        else:
            raise ValueError("dataset must be ade20k or cityscapes")
        if not self.image_files:
            raise FileNotFoundError(f"No images found under {image_dir}")
        if not images_only:
            missing = [str(p) for p in self.mask_files if not p.exists()]
            if missing:
                raise FileNotFoundError(f"Missing {len(missing)} mask(s), e.g. {missing[0]}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        with Image.open(self.image_files[index]) as im:
            image = im.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = None
        if not self.images_only:
            path = self.mask_files[index]
            with Image.open(path) as im:
                mask = im.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
            mask = np.asarray(mask, dtype=np.int64).copy()
            if self.dataset == "ade20k":
                mask = np.where((mask >= 1) & (mask <= 150), mask - 1, 255)
            elif path.name.endswith("_labelIds.png"):
                mapped = np.full(mask.shape, 255, dtype=np.int64)
                for idx, label_id in enumerate(CITYSCAPES_LABEL_IDS):
                    mapped[mask == label_id] = idx
                mask = mapped
            else:
                mask = np.where((mask >= 0) & (mask <= 18), mask, 255)
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if mask is not None:
                mask = np.fliplr(mask).copy()
        arr = np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy() / 255.0
        rgb = torch.from_numpy(arr)
        if self.images_only:
            return rgb
        return rgb, torch.from_numpy(mask.astype(np.int64, copy=False))


def label_subset(dataset, fraction=0.1, seed=42):
    """Fixed-seed image subset."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0,1]")
    if fraction == 1:
        return dataset
    count = max(1, round(len(dataset) * fraction))
    ids = sorted(np.random.default_rng(seed).choice(len(dataset), size=count, replace=False).tolist())
    return Subset(dataset, ids)
