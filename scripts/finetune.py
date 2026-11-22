"""Supervised head training followed by end-to-end segmentation fine-tuning."""
import argparse
import time
import torch
from _common import (add_training_args, csv_log, data_loader, load_checkpoint,
                     save_checkpoint, setup)
from mae_vit_seg.losses import cross_entropy_dice
from mae_vit_seg.metrics import ConfusionMatrix
from mae_vit_seg.model import ModelConfig, SemanticSegmenter


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    cm = ConfusionMatrix(num_classes)
    for images, masks in loader:
        logits = model(images.to(device))
        cm.update(logits, masks.to(device))
    return cm.mean_iou()


def train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = cross_entropy_dice(model(images), masks)
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * images.shape[0]
    return total / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_args(parser)
    parser.add_argument("--pretrained", type=str, required=True, help="Path to mae_last.pt")
    parser.add_argument("--head-epochs", type=int, default=2)
    parser.add_argument("--train-fraction", type=float, default=0.10)
    args = parser.parse_args()
    device = setup(args)
    if args.head_epochs < 0:
        raise ValueError("head-epochs cannot be negative")
    pretrained = load_checkpoint(args.pretrained, device)
    cfg = ModelConfig(**pretrained["config"])
    if cfg.num_classes != args.num_classes:
        raise ValueError(f"class-count mismatch: pretrained config={cfg.num_classes}; dataset={args.num_classes}")
    args.image_size = cfg.image_size
    model = SemanticSegmenter(cfg).to(device)
    model.encoder.load_state_dict({key.removeprefix("encoder."): value for key, value in
                                   pretrained["state_dict"].items() if key.startswith("encoder.")})
    train = data_loader(args, split="train", augment=True, subset=args.train_fraction)
    val = data_loader(args, split="val")
    best = float("-inf")
    step = 0
    for stage, epochs in [("head", args.head_epochs), ("finetune", args.epochs)]:
        if stage == "head":
            model.freeze_backbone()
            params = model.head.parameters()
        else:
            model.unfreeze_backbone()
            params = model.parameters()
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            loss = train_epoch(model, train, optimizer, device)
            miou = evaluate(model, val, device, cfg.num_classes)
            step += 1
            print(f"[{stage}] epoch {epoch}/{epochs}: loss={loss:.4f} val mIoU={miou:.4f} ({time.perf_counter()-started:.1f}s)", flush=True)
            csv_log(args.output_dir / "finetune_log.csv",
                    {"step": step, "stage": stage, "epoch": epoch,
                     "train_loss": loss, "val_miou": miou})
            info = {"stage": stage, "epoch": epoch, "dataset": args.dataset,
                    "train_fraction": args.train_fraction, "subset_seed": args.seed}
            save_checkpoint(args.output_dir / "segmenter_last.pt", model.state_dict(), cfg, info)
            if miou == miou and miou > best:
                best = miou
                save_checkpoint(args.output_dir / "segmenter_best.pt", model.state_dict(), cfg, info)
    print("Reference training complete. Results are not the historical benchmark.")


if __name__ == "__main__":
    main()
