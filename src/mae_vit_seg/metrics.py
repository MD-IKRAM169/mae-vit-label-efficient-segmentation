"""Semantic segmentation metrics with ignored pixels and absent classes."""
import torch


class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        predicted = logits.argmax(1) if logits.ndim == 4 else logits
        valid = ((labels != self.ignore_index) & (labels >= 0)
                 & (labels < self.num_classes))
        true_values = labels[valid].to(device="cpu", dtype=torch.long)
        pred_values = predicted[valid].to(device="cpu", dtype=torch.long)
        counts = torch.bincount(true_values * self.num_classes + pred_values,
                                minlength=self.num_classes ** 2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    def iou_per_class(self):
        matrix = self.matrix.to(torch.float64)
        intersection = matrix.diag()
        union = matrix.sum(0) + matrix.sum(1) - intersection
        return torch.where(union > 0, intersection / union.clamp_min(1), float("nan"))

    def mean_iou(self) -> float:
        scores = self.iou_per_class()
        valid = ~torch.isnan(scores)
        return float(scores[valid].mean()) if valid.any() else float("nan")
