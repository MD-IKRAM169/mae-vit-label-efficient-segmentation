# Masked Autoencoder Vision Transformer for Label-Efficient Image Segmentation

Self-supervised learning project exploring whether Masked Autoencoder (MAE)
pretraining with Vision Transformers can reduce the amount of labeled data
required for semantic image segmentation.

This project was completed as a research project at
Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).

## Motivation

Semantic segmentation models normally require large quantities of manually
annotated data. Annotation can be expensive and time-consuming, particularly
in domains such as biomedical imaging and autonomous systems.

This project investigates whether self-supervised MAE pretraining can learn
useful visual representations from unlabeled images before fine-tuning for
segmentation with only a small labeled subset.

## Approach

The project follows a three-stage pipeline:

1. **MAE Pretraining**
   - Split images into patches
   - Randomly mask 75% of patches
   - Train a Vision Transformer encoder to learn representations through reconstruction

2. **Segmentation Head**
   - Attach a lightweight decoder to the pretrained ViT backbone
   - Upsample learned features to produce pixel-level predictions

3. **Fine-Tuning**
   - Unfreeze the backbone
   - Fine-tune the complete model using only 10% of labeled data

## Datasets

Experiments were performed using:

- ADE20K
- Cityscapes

The raw datasets are not redistributed in this repository.
Please obtain them from their official sources.

## Tech Stack

- Python
- PyTorch
- CUDA
- Vision Transformers (ViT)
- Masked Autoencoders (MAE)
- Semantic Segmentation
- Self-Supervised Learning

## Evaluation

The experiments investigate:

- Reconstruction loss during MAE pretraining
- Fine-tuning loss
- Mean Intersection over Union (mIoU)
- Impact of training duration
- Impact of batch size
- Performance under a 10% labeled-data setting

## Key Learning

The project demonstrates how self-supervised representation learning can be
combined with Vision Transformers for label-efficient semantic segmentation.

It also highlights practical trade-offs between annotation cost, computational
resources, training duration and segmentation performance.

## Project Documents

- Research report
- Project presentation

## Author

**Md Ikram Tareq**  
MSc Data Science  
Friedrich-Alexander-Universität Erlangen-Nürnberg
