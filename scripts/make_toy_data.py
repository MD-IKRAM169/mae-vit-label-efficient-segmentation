"""Generate a tiny synthetic ADE20K-shaped dataset for pipeline smoke tests only."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/toy_ade20k"))
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--val-count", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=64)
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    for split, count in [("training", args.train_count), ("validation", args.val_count)]:
        for i in range(count):
            image = Image.new("RGB", (args.image_size, args.image_size), (27, 39, 52))
            mask = Image.new("L", (args.image_size, args.image_size), 1)
            x, y = rng.integers(2, args.image_size // 3, size=2)
            w = args.image_size // 2
            color = tuple(int(v) for v in rng.integers(110, 256, size=3))
            ImageDraw.Draw(image).rectangle((int(x), int(y), int(x + w), int(y + w)), fill=color)
            ImageDraw.Draw(mask).rectangle((int(x), int(y), int(x + w), int(y + w)), fill=2)
            img_dir = args.output_dir / "images" / split
            ann_dir = args.output_dir / "annotations" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            ann_dir.mkdir(parents=True, exist_ok=True)
            image.save(img_dir / f"toy_{i:03}.jpg")
            mask.save(ann_dir / f"toy_{i:03}.png")
    print(f"Created synthetic example data in {args.output_dir} (not ADE20K benchmark data).")


if __name__ == "__main__":
    main()
