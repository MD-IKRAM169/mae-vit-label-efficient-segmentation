"""Save palette-colored semantic predictions from a trained reference checkpoint."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from _common import load_checkpoint
from mae_vit_seg.model import ModelConfig, SemanticSegmenter


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--image', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = load_checkpoint(args.checkpoint, device)
    cfg = ModelConfig(**ck['config'])
    net = SemanticSegmenter(cfg).to(device)
    net.load_state_dict(ck['state_dict'])
    net.eval()
    with Image.open(args.image) as im:
        orig = im.convert('RGB')
        original_size = orig.size
        resized = orig.resize((cfg.image_size, cfg.image_size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    with torch.no_grad():
        idx = net(torch.from_numpy(arr).to(device)).argmax(1)[0].cpu().numpy().astype(np.uint8)
    palette = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        palette[i] = ((i*67+29)%256, (i*137+47)%256, (i*197+83)%256)
    out = Image.fromarray(idx, mode='P')
    out.putpalette(palette.flatten().tolist())
    out = out.resize(original_size, Image.Resampling.NEAREST)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.convert('RGB').save(args.output)
    print(f'Saved class-ID visualization: {args.output} (palette illustrative)')


if __name__ == '__main__':
    main()
