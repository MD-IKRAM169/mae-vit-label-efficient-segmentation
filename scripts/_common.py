"""Shared command-line training utilities."""
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mae_vit_seg.data import SegmentationDataset
from mae_vit_seg.model import ModelConfig


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_config(args):
    return ModelConfig(
        image_size=args.image_size, patch_size=args.patch_size,
        embed_dim=args.embed_dim, depth=args.depth, heads=args.heads,
        decoder_dim=args.decoder_dim, decoder_depth=args.decoder_depth,
        decoder_heads=args.decoder_heads, num_classes=args.num_classes,
        mask_ratio=args.mask_ratio,
    )


def add_model_args(parser):
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--decoder-dim", type=int, default=128)
    parser.add_argument("--decoder-depth", type=int, default=2)
    parser.add_argument("--decoder-heads", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=None,
                        help="Defaults to 150 for ADE20K, 19 for Cityscapes")
    parser.add_argument("--mask-ratio", type=float, default=0.75)


def add_training_args(parser):
    parser.add_argument("--dataset", choices=["ade20k", "cityscapes"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    add_model_args(parser)


def setup(args):
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if args.num_classes is None:
        args.num_classes = 150 if args.dataset == "ade20k" else 19
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def data_loader(args, split, images_only=False, augment=False, subset=None):
    ds = SegmentationDataset(args.data_root, args.dataset, split=split,
                             image_size=args.image_size, images_only=images_only, augment=augment)
    if subset is not None:
        from mae_vit_seg.data import label_subset
        ds = label_subset(ds, fraction=subset, seed=args.seed)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=(split == "train"),
                      num_workers=args.num_workers, pin_memory=torch.cuda.is_available())


def csv_log(path, row):
    path = Path(path)
    fields = list(row.keys())
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path, state_dict, cfg, info):
    torch.save({"state_dict": state_dict, "config": cfg.to_dict(), "info": info}, path)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=True)
