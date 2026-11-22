# Architecture and experiment specification

## Three stages

1. **Self-supervised pretraining:** resize RGB inputs to 224×224, divide into 16×16 patches, randomly mask 75% of the 196 patches, encode visible patches with a ViT and train a decoder to reconstruct masked image patches using pixel mean-squared error.
2. **Segmentation head:** attach a transposed-convolution decoder to the ViT patch features for per-pixel class logits. Initially keep the encoder frozen.
3. **Fine-tuning:** unfreeze the encoder and optimize cross-entropy + soft Dice over the labeled subset.

Reported datasets: ADE20K (150 train classes) and Cityscapes (19 train classes). The reported experimental grid contains 10/50 epochs and batch sizes 4/8, 10% labeled training images, AdamW (learning rate 1e-4, weight decay 0.05), and an NVIDIA T4 with 16 GB VRAM.

## Reference-code design choices

This is a **new reference implementation**, not original code. The compact default model has embed dimension 192, depth 4, and fixed sin/cos position encodings. The segmentation decoder uses two transposed-convolution layers. These dimensions and details are implementation choices, **not established details of the historical trained backbone**. Pretraining and downstream checkpoints are tied by stored configuration, so the same encoder architecture is used.

Reference mIoU ignores target 255 and excludes classes with zero union; benchmark evaluation conventions should be verified against dataset-specific procedures before comparing to external papers. The code is intended for inspection, small experiments and reproducibility setup, not to claim the original results have been duplicated.
