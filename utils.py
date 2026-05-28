import os
import csv
import json
import random
import logging
from typing import List, Dict, Tuple

import numpy as np
import torch


logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_tsv(path: str) -> List[List[str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 3:
                rows.append([row[0].strip(), row[1].strip(), row[2].strip()])
    return rows


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_entity_text(path: str) -> Dict[str, str]:
    entity_text = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", maxsplit=1)
            if len(parts) == 2:
                entity, text = parts
                entity_text[entity] = text
            elif len(parts) == 1:
                entity_text[parts[0]] = parts[0]

    return entity_text


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_label_maps(relations: List[str]):
    rel2id = {rel: idx for idx, rel in enumerate(relations)}
    id2rel = {idx: rel for rel, idx in rel2id.items()}
    return rel2id, id2rel


def get_device(device_name: str = "cuda") -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def format_relation_name(relation: str) -> str:
    """
    Converts KG relation IDs into readable text.
    Example:
        /people/person/place_of_birth -> people person place of birth
        born_in -> born in
    """
    relation = relation.replace("/", " ")
    relation = relation.replace("_", " ")
    relation = relation.replace(".", " ")
    return " ".join(relation.split())


def build_graph_neighbors(
    triples: List[List[str]],
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Builds an undirected one-hop neighborhood:
        N(e) = {(r, e') | (e,r,e') or (e',r,e)}
    """
    graph = {}

    for h, r, t in triples:
        graph.setdefault(h, []).append((r, t))
        graph.setdefault(t, []).append((r, h))

    return graph