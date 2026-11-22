"""Compact, from-scratch PyTorch MAE / ViT segmentation *reference* model.

Reconstructed from the research methodology; not the original experiment code.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 224
    patch_size: int = 16
    embed_dim: int = 192
    depth: int = 4
    heads: int = 3
    decoder_dim: int = 128
    decoder_depth: int = 2
    decoder_heads: int = 4
    num_classes: int = 150
    mask_ratio: float = 0.75

    def validate(self):
        assert self.image_size > 0 and self.image_size % self.patch_size == 0
        assert self.embed_dim % self.heads == 0
        assert self.decoder_dim % self.decoder_heads == 0
        assert self.depth > 0 and self.decoder_depth > 0
        assert 0.0 < self.mask_ratio < 1.0
        assert self.num_classes > 1

    def to_dict(self):
        return asdict(self)


def positional_encoding(num_patches: int, dim: int) -> torch.Tensor:
    """Standard fixed 1D sine/cosine positions, shape (1, N, D)."""
    if dim % 2:
        raise ValueError("embedding dimension must be even")
    position = torch.arange(num_patches, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    enc = torch.empty(num_patches, dim)
    enc[:, 0::2] = torch.sin(position * div)
    enc[:, 1::2] = torch.cos(position * div)
    return enc.unsqueeze(0)


def transformer(width: int, heads: int, layers: int) -> nn.TransformerEncoder:
    block = nn.TransformerEncoderLayer(
        d_model=width, nhead=heads, dim_feedforward=width * 4,
        dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)


class ViTEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.grid = cfg.image_size // cfg.patch_size
        n_patches = self.grid ** 2
        self.patch_embed = nn.Conv2d(3, cfg.embed_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.register_buffer("pos_embed", positional_encoding(n_patches, cfg.embed_dim), persistent=False)
        self.blocks = transformer(cfg.embed_dim, cfg.heads, cfg.depth)
        self.norm = nn.LayerNorm(cfg.embed_dim)

    def patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        if tuple(images.shape[1:]) != (3, self.cfg.image_size, self.cfg.image_size):
            raise ValueError(f"expected [B,3,{self.cfg.image_size},{self.cfg.image_size}], got {tuple(images.shape)}")
        return self.patch_embed(images).flatten(2).transpose(1, 2) + self.pos_embed

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.norm(self.blocks(self.patch_tokens(images)))

    def forward_visible(self, images: torch.Tensor, visible_indices: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_tokens(images)
        visible = torch.gather(tokens, 1, visible_indices.unsqueeze(-1).expand(-1, -1, tokens.size(-1)))
        return self.norm(self.blocks(visible))


class MaskedAutoencoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ViTEncoder(cfg)
        self.decoder_embed = nn.Linear(cfg.embed_dim, cfg.decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.register_buffer("decoder_pos", positional_encoding(self.encoder.grid ** 2, cfg.decoder_dim), persistent=False)
        self.decoder = transformer(cfg.decoder_dim, cfg.decoder_heads, cfg.decoder_depth)
        self.decoder_norm = nn.LayerNorm(cfg.decoder_dim)
        self.decoder_pred = nn.Linear(cfg.decoder_dim, cfg.patch_size * cfg.patch_size * 3)

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        p = self.cfg.patch_size
        b, c, h, w = images.shape
        if c != 3 or h != self.cfg.image_size or w != h:
            raise ValueError("unexpected image dimensions")
        return images.reshape(b, 3, h // p, p, w // p, p).permute(0, 2, 4, 3, 5, 1).reshape(b, -1, p * p * 3)

    def forward(self, images: torch.Tensor):
        """Returns masked-only MSE, reconstructed patch pixels, and mask (1=masked)."""
        b = images.size(0)
        n = self.encoder.grid ** 2
        keep = max(1, round(n * (1.0 - self.cfg.mask_ratio)))
        noise = torch.rand(b, n, device=images.device)
        shuffled = noise.argsort(dim=1)
        restore = shuffled.argsort(dim=1)
        visible_ids = shuffled[:, :keep]
        visible = self.encoder.forward_visible(images, visible_ids)
        decoded_visible = self.decoder_embed(visible)
        mask_tokens = self.mask_token.expand(b, n - keep, -1)
        shuffled_tokens = torch.cat([decoded_visible, mask_tokens], dim=1)
        all_tokens = torch.gather(shuffled_tokens, 1, restore.unsqueeze(-1).expand(-1, -1, self.cfg.decoder_dim))
        reconstructed = self.decoder_pred(self.decoder_norm(self.decoder(all_tokens + self.decoder_pos)))
        mask = torch.ones(b, n, device=images.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, 1, restore)
        mse = (reconstructed - self.patchify(images)).square().mean(dim=-1)
        loss = (mse * mask).sum() / mask.sum().clamp_min(1)
        return loss, reconstructed, mask


class SemanticSegmenter(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ViTEncoder(cfg)
        mid = max(32, cfg.embed_dim // 2)
        self.head = nn.Sequential(
            nn.ConvTranspose2d(cfg.embed_dim, mid, kernel_size=4, stride=4),
            nn.GELU(),
            nn.ConvTranspose2d(mid, mid // 2, kernel_size=4, stride=4),
            nn.GELU(),
            nn.Conv2d(mid // 2, cfg.num_classes, kernel_size=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b = images.size(0)
        features = self.encoder.forward_features(images)
        g = self.encoder.grid
        features = features.transpose(1, 2).reshape(b, self.cfg.embed_dim, g, g)
        logits = self.head(features)
        if logits.shape[-2:] != images.shape[-2:]:
            logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return logits

    def freeze_backbone(self):
        self.encoder.requires_grad_(False)

    def unfreeze_backbone(self):
        self.encoder.requires_grad_(True)
