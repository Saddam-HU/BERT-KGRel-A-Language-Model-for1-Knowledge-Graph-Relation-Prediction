import torch
import torch.nn as nn
from transformers import AutoModel


class BERTKGRelModel(nn.Module):
    """
    BERT-KGRel model.

    Implements:
    - BERT encoder
    - role embeddings
    - head/tail span pooling
    - cross-entity interaction vector
    - gated fusion
    - relation classification head
    """

    def __init__(
        self,
        bert_model_name: str,
        num_labels: int,
        role_vocab_size: int = 5,
        fusion_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(bert_model_name)

        hidden_size = self.bert.config.hidden_size

        self.role_embeddings = nn.Embedding(role_vocab_size, hidden_size)

        interaction_dim = hidden_size * 4
        gate_input_dim = hidden_size + interaction_dim

        self.cls_projection = nn.Linear(hidden_size, fusion_dim)
        self.interaction_projection = nn.Linear(interaction_dim, fusion_dim)
        self.gate = nn.Linear(gate_input_dim, fusion_dim)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fusion_dim, num_labels)

    def resize_token_embeddings(self, new_size: int):
        self.bert.resize_token_embeddings(new_size)

    @staticmethod
    def masked_mean_pool(sequence_output, mask):
        """
        Mean pooling over selected token spans.
        """
        mask = mask.unsqueeze(-1)
        masked_output = sequence_output * mask

        summed = masked_output.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)

        return summed / counts

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        role_ids,
        head_span_mask,
        tail_span_mask,
    ):
        word_embeddings = self.bert.embeddings.word_embeddings(input_ids)
        role_embeddings = self.role_embeddings(role_ids)

        inputs_embeds = word_embeddings + role_embeddings

        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state

        z_cls = sequence_output[:, 0, :]

        z_head = self.masked_mean_pool(sequence_output, head_span_mask)
        z_tail = self.masked_mean_pool(sequence_output, tail_span_mask)

        z_interaction = torch.cat(
            [
                z_head,
                z_tail,
                torch.abs(z_head - z_tail),
                z_head * z_tail,
            ],
            dim=-1,
        )

        gate_input = torch.cat([z_cls, z_interaction], dim=-1)

        g = torch.sigmoid(self.gate(gate_input))

        z_cls_proj = self.cls_projection(z_cls)
        z_int_proj = self.interaction_projection(z_interaction)

        z_fuse = g * z_cls_proj + (1.0 - g) * z_int_proj

        z_fuse = self.dropout(z_fuse)

        logits = self.classifier(z_fuse)

        return {
            "logits": logits,
            "embeddings": z_fuse,
        }