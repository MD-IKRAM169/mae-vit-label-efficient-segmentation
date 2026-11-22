"""Compute validation mIoU from a saved reference segmenter checkpoint."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from _common import load_checkpoint
from mae_vit_seg.data import SegmentationDataset
from mae_vit_seg.metrics import ConfusionMatrix
from mae_vit_seg.model import ModelConfig, SemanticSegmenter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset", choices=["ade20k", "cityscapes"], required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)
    cfg = ModelConfig(**checkpoint["config"])
    expected = 150 if args.dataset == "ade20k" else 19
    if cfg.num_classes != expected:
        raise ValueError(f"Checkpoint has {cfg.num_classes} classes but {args.dataset} expects {expected}")
    model = SemanticSegmenter(cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    val = SegmentationDataset(args.data_root, args.dataset, args.split, cfg.image_size)
    loader = DataLoader(val, batch_size=args.batch_size, num_workers=args.num_workers)
    cm = ConfusionMatrix(cfg.num_classes)
    with torch.no_grad():
        for images, targets in loader:
            cm.update(model(images.to(device)), targets.to(device))
    print(f"{args.dataset}/{args.split} mean IoU: {100 * cm.mean_iou():.2f}%")
    print("Absent classes are excluded; pixels with ID 255 ignored.")


if __name__ == "__main__":
    main()
