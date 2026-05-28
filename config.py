from dataclasses import dataclass


@dataclass
class BERTKGRelConfig:
    # Data
    data_dir: str = "data/FB15k-237"
    output_dir: str = "outputs/bert_kgrel"

    # Encoder
    bert_model_name: str = "bert-base-uncased"
    max_seq_length: int = 128
    max_entity_tokens: int = 48
    max_context_tokens: int = 24
    num_neighbors: int = 8

    # Model
    dropout: float = 0.1
    role_vocab_size: int = 5
    fusion_dim: int = 768

    # Training
    train_batch_size: int = 32
    eval_batch_size: int = 32
    num_epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Loss
    contrastive_lambda: float = 0.1
    contrastive_temperature: float = 0.07

    # Runtime
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 0

    # Special tokens
    head_token: str = "[HEAD]"
    tail_token: str = "[TAIL]"
    ctx_token: str = "[CTX]"
    prompt_token: str = "[PROMPT]"

    prompt_text: str = "predict relation between head and tail"