# Reproducibility checklist

The code was reconstructed from the methodology for demonstration. To claim verified reproducibility of the 2025 results, the repository would need:

- Original unmodified source files or notebooks and specific dependency lockfiles.
- Original MAE and segmentation checkpoints.
- Complete per-run logs, seed records, and exact 10%-label sample lists.
- Data versions, train/validation splits and class-mapping documentation.
- The exact evaluation protocol for 150 ADE20K and 19 Cityscapes classes.
- Independent reruns with explicit hardware, run duration, and observed mIoU.
- Supervised controls with matched backbones and equal evaluation protocols.

The provided PyTorch unit tests establish core program behavior, not experimental accuracy. The public benchmark figures should remain described as *reported*, not independently verified.
