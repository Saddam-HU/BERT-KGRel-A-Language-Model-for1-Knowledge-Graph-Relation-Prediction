import os
import argparse
import torch

from config import BERTKGRelConfig
from dataloading import KGDataModule
from model import BERTKGRelModel
from metrics import evaluate_model
from utils import setup_logging, get_device, load_json


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BERT-KGRel")

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test"])
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def main():
    setup_logging()

    args = parse_args()

    config_path = os.path.join(args.checkpoint_dir, "training_config.json")
    config_dict = load_json(config_path)

    config = BERTKGRelConfig()
    for key, value in config_dict.items():
        setattr(config, key, value)

    config.data_dir = args.data_dir
    config.device = args.device

    device = get_device(config.device)

    data_module = KGDataModule(config)

    model = BERTKGRelModel(
        bert_model_name=config.bert_model_name,
        num_labels=len(data_module.relations),
        role_vocab_size=config.role_vocab_size,
        fusion_dim=config.fusion_dim,
        dropout=config.dropout,
    )

    model.resize_token_embeddings(len(data_module.tokenizer))

    state_dict = torch.load(
        os.path.join(args.checkpoint_dir, "pytorch_model_full.bin"),
        map_location=device,
    )

    model.load_state_dict(state_dict)
    model.to(device)

    dataloader = data_module.get_dataloader(args.split, shuffle=False)

    results = evaluate_model(model, dataloader, device)

    print(f"\nResults on {args.split}:")
    for key, value in results.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()