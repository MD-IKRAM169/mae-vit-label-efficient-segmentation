"""CPU-only tests of reference code, not reproduction of historical scores."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from mae_vit_seg.data import SegmentationDataset, label_subset
from mae_vit_seg.losses import cross_entropy_dice
from mae_vit_seg.metrics import ConfusionMatrix
from mae_vit_seg.model import ModelConfig, MaskedAutoencoder, SemanticSegmenter


def tiny_config():
    return ModelConfig(image_size=32, patch_size=8, embed_dim=32, depth=1,
                       heads=4, decoder_dim=16, decoder_depth=1,
                       decoder_heads=4, num_classes=3)


def test_masked_autoencoder_forward_backward():
    torch.manual_seed(1)
    model = MaskedAutoencoder(tiny_config())
    x = torch.rand(2,3,32,32)
    loss,pred,mask = model(x)
    assert pred.shape == (2,16,192)
    assert mask.shape == (2,16)
    assert (mask==1).sum().item() == 24
    assert torch.isfinite(loss)
    loss.backward()
    assert model.encoder.patch_embed.weight.grad is not None


def test_segmenter_frozen_and_finetuned_gradients():
    m = SemanticSegmenter(tiny_config())
    assert m(torch.rand(2,3,32,32)).shape == (2,3,32,32)
    m.freeze_backbone()
    assert not any(p.requires_grad for p in m.encoder.parameters())
    m.unfreeze_backbone()
    assert all(p.requires_grad for p in m.encoder.parameters())
    x = torch.rand(2,3,32,32)
    y = torch.randint(0,3,(2,32,32))
    loss = cross_entropy_dice(m(x),y)
    loss.backward()
    assert torch.isfinite(loss)


def test_miou_with_ignore_index():
    cm = ConfusionMatrix(2)
    labels = torch.tensor([[[0,1],[255,1]]])
    predictions = torch.tensor([[[0,0],[1,1]]])
    cm.update(predictions,labels)
    assert abs(cm.mean_iou()-0.5) < 1e-7


def test_all_ignored_loss_is_defined():
    logits = torch.randn(1,2,2,2,requires_grad=True)
    loss = cross_entropy_dice(logits,torch.full((1,2,2),255))
    loss.backward()
    assert loss.item() == 0.0
    assert torch.all(logits.grad == 0)


def test_ade20k_ignore_and_label_mapping(tmp_path: Path):
    image_dir = tmp_path/'images'/'training'
    mask_dir = tmp_path/'annotations'/'training'
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((2,2,3),dtype=np.uint8)).save(image_dir/'test.jpg')
    Image.fromarray(np.array([[0,1],[2,150]],dtype=np.uint8)).save(mask_dir/'test.png')
    dataset = SegmentationDataset(tmp_path,'ade20k',image_size=2)
    _,mask = dataset[0]
    assert mask.tolist() == [[255,0],[1,149]]
    assert len(label_subset(dataset,0.1,seed=5)) == 1
