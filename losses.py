import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """
    InfoNCE-style consistency regularization.

    Positive samples are examples in the same mini-batch
    with the same relation label.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        device = embeddings.device
        batch_size = embeddings.size(0)

        if batch_size <= 1:
            return torch.tensor(0.0, device=device)

        embeddings = F.normalize(embeddings, p=2, dim=1)

        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature

        labels = labels.contiguous().view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).float().to(device)

        self_mask = torch.eye(batch_size, device=device)
        positive_mask = positive_mask - self_mask

        logits_mask = 1.0 - self_mask

        exp_similarity = torch.exp(similarity) * logits_mask

        numerator = (exp_similarity * positive_mask).sum(dim=1)
        denominator = exp_similarity.sum(dim=1).clamp(min=1e-12)

        valid_mask = positive_mask.sum(dim=1) > 0

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        loss = -torch.log((numerator + 1e-12) / denominator)

        return loss[valid_mask].mean()


class BERTKGRelLoss(nn.Module):
    def __init__(
        self,
        contrastive_lambda: float = 0.1,
        contrastive_temperature: float = 0.07,
    ):
        super().__init__()

        self.ce_loss = nn.CrossEntropyLoss()
        self.contrastive_loss = SupervisedContrastiveLoss(
            temperature=contrastive_temperature
        )
        self.contrastive_lambda = contrastive_lambda

    def forward(self, logits, embeddings, labels):
        classification_loss = self.ce_loss(logits, labels)
        consistency_loss = self.contrastive_loss(embeddings, labels)

        total_loss = classification_loss + self.contrastive_lambda * consistency_loss

        return {
            "loss": total_loss,
            "classification_loss": classification_loss.detach(),
            "contrastive_loss": consistency_loss.detach(),
        }