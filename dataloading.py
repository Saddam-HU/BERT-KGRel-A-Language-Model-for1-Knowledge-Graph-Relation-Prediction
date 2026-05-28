import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from utils import (
    read_tsv,
    read_lines,
    read_entity_text,
    build_graph_neighbors,
    format_relation_name,
)


ROLE_SPECIAL = 0
ROLE_HEAD = 1
ROLE_TAIL = 2
ROLE_CONTEXT = 3
ROLE_PROMPT = 4


@dataclass
class KGExample:
    guid: str
    head: str
    relation: str
    tail: str


class KGRelationDataset(Dataset):
    def __init__(
        self,
        examples: List[KGExample],
        rel2id: Dict[str, int],
        entity_text: Dict[str, str],
        graph_neighbors: Dict[str, List[Tuple[str, str]]],
        tokenizer,
        max_seq_length: int = 128,
        max_entity_tokens: int = 48,
        max_context_tokens: int = 24,
        num_neighbors: int = 8,
        prompt_text: str = "predict relation between head and tail",
        head_token: str = "[HEAD]",
        tail_token: str = "[TAIL]",
        ctx_token: str = "[CTX]",
        prompt_token: str = "[PROMPT]",
    ):
        self.examples = examples
        self.rel2id = rel2id
        self.entity_text = entity_text
        self.graph_neighbors = graph_neighbors
        self.tokenizer = tokenizer

        self.max_seq_length = max_seq_length
        self.max_entity_tokens = max_entity_tokens
        self.max_context_tokens = max_context_tokens
        self.num_neighbors = num_neighbors

        self.prompt_text = prompt_text
        self.head_token = head_token
        self.tail_token = tail_token
        self.ctx_token = ctx_token
        self.prompt_token = prompt_token

    def __len__(self):
        return len(self.examples)

    def textualize_entity(self, entity: str) -> str:
        """
        Implements entity textualization:
            x(e) = Name(e) ⊕ Desc(e) ⊕ Attr(e)

        Here entity2text.txt provides the description.
        If an entity has no description, the entity ID is used.
        """
        text = self.entity_text.get(entity, entity)
        return text.strip()

    def build_neighbor_context(self, entity: str) -> str:
        """
        Implements neighbor-aware contextualization:
            c(e) = concat(RelTok(r'), Name(e'))

        Top-K is approximated by preserving the first K unique relation-neighbor pairs.
        You can replace this with TF-IDF, degree-normalized, or frequency-based scoring.
        """
        neighbors = self.graph_neighbors.get(entity, [])

        selected = []
        seen = set()

        for relation, neighbor in neighbors:
            key = (relation, neighbor)
            if key in seen:
                continue

            seen.add(key)
            rel_text = format_relation_name(relation)
            neighbor_text = self.entity_text.get(neighbor, neighbor)
            neighbor_name = neighbor_text.split(".")[0]

            selected.append(f"{rel_text} {neighbor_name}")

            if len(selected) >= self.num_neighbors:
                break

        return " ; ".join(selected)

    def _tokenize_with_limit(self, text: str, max_tokens: int) -> List[str]:
        tokens = self.tokenizer.tokenize(text)
        return tokens[:max_tokens]

    def _build_feature(self, example: KGExample) -> Dict[str, torch.Tensor]:
        head_text = self.textualize_entity(example.head)
        tail_text = self.textualize_entity(example.tail)

        head_context = self.build_neighbor_context(example.head)
        tail_context = self.build_neighbor_context(example.tail)

        head_tokens = self._tokenize_with_limit(head_text, self.max_entity_tokens)
        tail_tokens = self._tokenize_with_limit(tail_text, self.max_entity_tokens)

        head_ctx_tokens = self._tokenize_with_limit(head_context, self.max_context_tokens)
        tail_ctx_tokens = self._tokenize_with_limit(tail_context, self.max_context_tokens)

        prompt_tokens = self.tokenizer.tokenize(self.prompt_text)

        tokens = []
        token_type_ids = []
        role_ids = []
        head_span_mask = []
        tail_span_mask = []

        def add(token_list, type_id, role_id, is_head=False, is_tail=False):
            tokens.extend(token_list)
            token_type_ids.extend([type_id] * len(token_list))
            role_ids.extend([role_id] * len(token_list))
            head_span_mask.extend([1 if is_head else 0] * len(token_list))
            tail_span_mask.extend([1 if is_tail else 0] * len(token_list))

        add([self.tokenizer.cls_token], 0, ROLE_SPECIAL)

        add([self.head_token], 0, ROLE_SPECIAL)
        add(head_tokens, 0, ROLE_HEAD, is_head=True)

        add([self.ctx_token], 0, ROLE_SPECIAL)
        add(head_ctx_tokens, 0, ROLE_CONTEXT)

        add([self.tokenizer.sep_token], 0, ROLE_SPECIAL)

        add([self.tail_token], 1, ROLE_SPECIAL)
        add(tail_tokens, 1, ROLE_TAIL, is_tail=True)

        add([self.ctx_token], 1, ROLE_SPECIAL)
        add(tail_ctx_tokens, 1, ROLE_CONTEXT)

        add([self.tokenizer.sep_token], 1, ROLE_SPECIAL)

        add([self.prompt_token], 0, ROLE_SPECIAL)
        add(prompt_tokens, 0, ROLE_PROMPT)

        add([self.tokenizer.sep_token], 0, ROLE_SPECIAL)

        if len(tokens) > self.max_seq_length:
            tokens = tokens[: self.max_seq_length]
            token_type_ids = token_type_ids[: self.max_seq_length]
            role_ids = role_ids[: self.max_seq_length]
            head_span_mask = head_span_mask[: self.max_seq_length]
            tail_span_mask = tail_span_mask[: self.max_seq_length]

            tokens[-1] = self.tokenizer.sep_token
            role_ids[-1] = ROLE_SPECIAL

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        pad_len = self.max_seq_length - len(input_ids)

        input_ids += [self.tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        token_type_ids += [0] * pad_len
        role_ids += [ROLE_SPECIAL] * pad_len
        head_span_mask += [0] * pad_len
        tail_span_mask += [0] * pad_len

        label_id = self.rel2id[example.relation]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "role_ids": torch.tensor(role_ids, dtype=torch.long),
            "head_span_mask": torch.tensor(head_span_mask, dtype=torch.float),
            "tail_span_mask": torch.tensor(tail_span_mask, dtype=torch.float),
            "labels": torch.tensor(label_id, dtype=torch.long),
        }

    def __getitem__(self, idx):
        return self._build_feature(self.examples[idx])


class KGDataModule:
    def __init__(self, config):
        self.config = config

        self.train_triples = read_tsv(os.path.join(config.data_dir, "train.tsv"))
        self.dev_triples = read_tsv(os.path.join(config.data_dir, "dev.tsv"))
        self.test_triples = read_tsv(os.path.join(config.data_dir, "test.tsv"))

        self.relations = read_lines(os.path.join(config.data_dir, "relations.txt"))
        self.entities = read_lines(os.path.join(config.data_dir, "entities.txt"))
        self.entity_text = read_entity_text(os.path.join(config.data_dir, "entity2text.txt"))

        self.rel2id = {rel: idx for idx, rel in enumerate(self.relations)}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        # Use train graph for neighborhood context to avoid test leakage.
        self.graph_neighbors = build_graph_neighbors(self.train_triples)

        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_model_name)

        special_tokens = {
            "additional_special_tokens": [
                config.head_token,
                config.tail_token,
                config.ctx_token,
                config.prompt_token,
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)

    def _make_examples(self, triples: List[List[str]], split: str) -> List[KGExample]:
        examples = []

        for idx, triple in enumerate(triples):
            h, r, t = triple
            if r not in self.rel2id:
                continue

            examples.append(
                KGExample(
                    guid=f"{split}-{idx}",
                    head=h,
                    relation=r,
                    tail=t,
                )
            )

        return examples

    def get_dataset(self, split: str) -> KGRelationDataset:
        if split == "train":
            examples = self._make_examples(self.train_triples, "train")
        elif split == "dev":
            examples = self._make_examples(self.dev_triples, "dev")
        elif split == "test":
            examples = self._make_examples(self.test_triples, "test")
        else:
            raise ValueError(f"Unknown split: {split}")

        return KGRelationDataset(
            examples=examples,
            rel2id=self.rel2id,
            entity_text=self.entity_text,
            graph_neighbors=self.graph_neighbors,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_seq_length,
            max_entity_tokens=self.config.max_entity_tokens,
            max_context_tokens=self.config.max_context_tokens,
            num_neighbors=self.config.num_neighbors,
            prompt_text=self.config.prompt_text,
            head_token=self.config.head_token,
            tail_token=self.config.tail_token,
            ctx_token=self.config.ctx_token,
            prompt_token=self.config.prompt_token,
        )

    def get_dataloader(self, split: str, shuffle: Optional[bool] = None) -> DataLoader:
        dataset = self.get_dataset(split)

        if shuffle is None:
            shuffle = split == "train"

        batch_size = (
            self.config.train_batch_size
            if split == "train"
            else self.config.eval_batch_size
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
        )