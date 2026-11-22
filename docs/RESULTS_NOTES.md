# Reported findings and provenance

This page transcribes findings from the supplied September 2025 FAU MAE-ViT research report **and the author's training screenshots**. These are reported historical values, not independent tests of the new reference code.

## mIoU (%)

| Dataset | 10e / b4 | 10e / b8 | 50e / b4 | 50e / b8 | OneFormer literature reference |
|---|---:|---:|---:|---:|---:|
| ADE20K | 1.74 | 2.00 | 26.00 | **28.50** | 57.70 |
| Cityscapes | 0.32 | 0.50 | 16.00 | **18.20** | 83.00 |

The experiments described for MAE-ViT use 10% of labeled images; the cited OneFormer values assume a different, full-supervision protocol. These comparisons **cannot** establish matched-model superiority or isolate the effect of MAE pretraining.

## Training evidence

- The ADE20K terminal screenshot shows a 10-epoch/batch-4 pretraining loss decreasing from 0.9426 to 0.0347, and fine-tuning loss from 4.7932 to 3.5839. After five additional end-to-end epochs it reports 1.74% average mIoU.
- The Cityscapes terminal screenshot shows a 10-epoch/batch-4 pretraining loss decreasing from 1.0707 to 0.0253, and fine-tuning loss from 2.8113 to 1.7486; the screen shows a **NaN at epoch 7** and subsequent end-to-end losses decreasing to 1.6567. It prints an average mIoU of 0.32%.
- The report's Tables 3–6 state the additional configurations at 10/50 epochs and batch 4/8. The table numbers are separated from screenshot observations in CSVs to preserve provenance.
- The source report describes image sizes 224×224, 75% masking, AdamW (LR 1e-4, WD 0.05), cross-entropy + Dice, NVIDIA T4, and 10% labeled training subsets.

## Data / methodological caveats

1. **Contradictory epochs:** Several table columns labeled as 10-epoch runs nonetheless contain measurements at epochs 25 and 50. Do not treat those as results of completed 10-epoch runs.
2. **External baseline:** OneFormer 57.7/83.0 are reference values, not a controlled same-backbone same-protocol baseline in this repo.
3. **Missing artifacts:** Original implementation, exact configurations, checkpoints, sample-ID splits, class-wise metrics, complete independent validation records and independent benchmark reproduction were not supplied.
4. **No established statistical certainty:** No repeated-seed or confidence-interval study is presented.
5. **Image provenance:** The provided 4-panel street segmentation image appears to include a model overlay for another segmentation architecture, so it is not labeled here as an output of this MAE-ViT implementation.
6. **The report caption's claim** that annotation reduction occurs without sacrificing quality is not supported by the reported lower mIoU relative to OneFormer.

The CSVs are designed to be auditable: \`reported_miou.csv\` and \`reported_pretraining_loss.csv\` carry report figures; \`screenshot_training_logs.csv\` records original terminal-screen values separately.

Recommended citation wording: **“The 2025 project report states 28.5% ADE20K and 18.2% Cityscapes mIoU for its 10%-label MAE-ViT study; these values have not been independently reproduced by the newer reference code in this repository.”**
