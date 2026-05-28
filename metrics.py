import numpy as np
import torch


def accuracy(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return float((preds == labels).mean())


def relation_ranking_metrics(logits, labels):
    """
    Computes MR, MRR, Hits@1, Hits@3, Hits@10
    for relation prediction.

    logits: numpy array [num_examples, num_relations]
    labels: numpy array [num_examples]
    """
    ranks = []

    for logit, true_label in zip(logits, labels):
        sorted_indices = np.argsort(-logit)
        rank = np.where(sorted_indices == true_label)[0][0] + 1
        ranks.append(rank)

    ranks = np.asarray(ranks)

    return {
        "MR": float(np.mean(ranks)),
        "MRR": float(np.mean(1.0 / ranks)),
        "Hits@1": float(np.mean(ranks <= 1)),
        "Hits@3": float(np.mean(ranks <= 3)),
        "Hits@10": float(np.mean(ranks <= 10)),
    }


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()

    all_logits = []
    all_labels = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            role_ids=batch["role_ids"],
            head_span_mask=batch["head_span_mask"],
            tail_span_mask=batch["tail_span_mask"],
        )

        logits = outputs["logits"]

        all_logits.append(logits.cpu())
        all_labels.append(batch["labels"].cpu())

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    preds = np.argmax(all_logits, axis=1)

    results = {
        "accuracy": accuracy(preds, all_labels),
    }

    results.update(relation_ranking_metrics(all_logits, all_labels))

    return results