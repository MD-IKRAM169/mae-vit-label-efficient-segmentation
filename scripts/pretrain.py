"""Self-supervised MAE pretraining."""
import argparse
import time
import torch
from _common import (add_training_args, csv_log, data_loader, model_config,
                     save_checkpoint, setup)
from mae_vit_seg.model import MaskedAutoencoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_args(parser)
    args = parser.parse_args()
    device = setup(args)
    cfg = model_config(args)
    model = MaskedAutoencoder(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train = data_loader(args, split="train", images_only=True, augment=True)
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        model.train()
        total = 0.0
        for images in train:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = model(images)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(images)
        mean_loss = total / len(train.dataset)
        print(f"[MAE] epoch {epoch}/{args.epochs}: masked MSE={mean_loss:.5f} ({time.perf_counter()-start:.1f}s)", flush=True)
        csv_log(args.output_dir / "pretrain_log.csv", {"epoch": epoch, "masked_mse": mean_loss})
        save_checkpoint(args.output_dir / "mae_last.pt", model.state_dict(), cfg,
                        {"stage": "pretrain", "epoch": epoch, "dataset": args.dataset})
    print(f"Saved checkpoint: {args.output_dir / 'mae_last.pt'}")


if __name__ == "__main__":
    main()
