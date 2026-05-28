import os
import argparse
import logging

import torch
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from config import BERTKGRelConfig
from dataloading import KGDataModule
from model import BERTKGRelModel
from losses import BERTKGRelLoss
from metrics import evaluate_model
from utils import (
    setup_logging,
    set_seed,
    ensure_dir,
    get_device,
    save_json,
)


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train BERT-KGRel")

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--bert_model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--num_neighbors", type=int, default=8)

    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=3)

    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--contrastive_lambda", type=float, default=0.1)
    parser.add_argument("--contrastive_temperature", type=float, default=0.07)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def build_config(args):
    config = BERTKGRelConfig()

    for key, value in vars(args).items():
        setattr(config, key, value)

    return config


def create_optimizer(model, config):
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    return torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
    )


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    criterion,
    device,
    config,
    epoch,
):
    model.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_con_loss = 0.0

    progress = tqdm(dataloader, desc=f"Epoch {epoch}")

    for step, batch in enumerate(progress):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            role_ids=batch["role_ids"],
            head_span_mask=batch["head_span_mask"],
            tail_span_mask=batch["tail_span_mask"],
        )

        loss_dict = criterion(
            logits=outputs["logits"],
            embeddings=outputs["embeddings"],
            labels=batch["labels"],
        )

        loss = loss_dict["loss"]

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.max_grad_norm,
        )

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_cls_loss += loss_dict["classification_loss"].item()
        total_con_loss += loss_dict["contrastive_loss"].item()

        progress.set_postfix(
            {
                "loss": total_loss / (step + 1),
                "ce": total_cls_loss / (step + 1),
                "con": total_con_loss / (step + 1),
            }
        )

    return {
        "train_loss": total_loss / len(dataloader),
        "train_classification_loss": total_cls_loss / len(dataloader),
        "train_contrastive_loss": total_con_loss / len(dataloader),
    }


def save_checkpoint(model, tokenizer, config, metrics, path):
    ensure_dir(path)

    model_to_save = model.module if hasattr(model, "module") else model

    model_to_save.bert.save_pretrained(path)
    tokenizer.save_pretrained(path)

    torch.save(
        model_to_save.state_dict(),
        os.path.join(path, "pytorch_model_full.bin"),
    )

    save_json(metrics, os.path.join(path, "metrics.json"))
    save_json(config.__dict__, os.path.join(path, "training_config.json"))


def main():
    setup_logging()

    args = parse_args()
    config = build_config(args)

    ensure_dir(config.output_dir)
    set_seed(config.seed)

    device = get_device(config.device)

    logger.info("Loading data...")
    data_module = KGDataModule(config)

    train_loader = data_module.get_dataloader("train", shuffle=True)
    dev_loader = data_module.get_dataloader("dev", shuffle=False)

    logger.info("Building model...")
    model = BERTKGRelModel(
        bert_model_name=config.bert_model_name,
        num_labels=len(data_module.relations),
        role_vocab_size=config.role_vocab_size,
        fusion_dim=config.fusion_dim,
        dropout=config.dropout,
    )

    model.resize_token_embeddings(len(data_module.tokenizer))
    model.to(device)

    criterion = BERTKGRelLoss(
        contrastive_lambda=config.contrastive_lambda,
        contrastive_temperature=config.contrastive_temperature,
    )

    optimizer = create_optimizer(model, config)

    total_training_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_training_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    best_mrr = -1.0
    best_metrics = None

    logger.info("Starting training...")

    for epoch in range(1, config.num_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            config=config,
            epoch=epoch,
        )

        dev_metrics = evaluate_model(model, dev_loader, device)

        epoch_metrics = {
            "epoch": epoch,
            **train_metrics,
            **{f"dev_{k}": v for k, v in dev_metrics.items()},
        }

        logger.info(epoch_metrics)

        if dev_metrics["MRR"] > best_mrr:
            best_mrr = dev_metrics["MRR"]
            best_metrics = epoch_metrics

            logger.info("New best checkpoint found.")
            save_checkpoint(
                model=model,
                tokenizer=data_module.tokenizer,
                config=config,
                metrics=best_metrics,
                path=os.path.join(config.output_dir, "best_model"),
            )

    save_json(best_metrics, os.path.join(config.output_dir, "best_metrics.json"))

    logger.info("Training complete.")
    logger.info(f"Best metrics: {best_metrics}")


if __name__ == "__main__":
    main()