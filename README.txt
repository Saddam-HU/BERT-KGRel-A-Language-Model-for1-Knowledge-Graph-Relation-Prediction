# 7. Methodology



## 7.1 Overview

BERT-KGRel is a transformer-based relation prediction framework designed for knowledge graph completion and relation classification tasks.





The framework integrates:

* Entity textualization
* Neighbor-aware contextualization
* Relation-aware prompting
* Role-aware embeddings
* Cross-entity interaction modeling
* Gated feature fusion
* Transformer-based encoding







The model predicts the probability distribution over candidate relations for a given entity pair $(h,t)$.

---

# 7.2 Entity Textualization

For each entity $e \in \mathcal{E}$, a textual representation is constructed:

```math id="2kz1s5"
\mathbf{x}(e) = \text{Name}(e) \oplus \text{Desc}(e) \oplus \text{Attr}(e)
```


where:

* $\text{Name}(e)$ denotes the entity name
* $\text{Desc}(e)$ denotes entity description text
* $\text{Attr}(e)$ denotes optional attribute information
* $\oplus$ denotes concatenation


Entity descriptions are loaded from `entity2text.txt`.

To control input length, entity texts are truncated:

```math id="yz3hpo"
\tilde{\mathbf{x}}(e) = \tau_L(\mathbf{x}(e))
```


where $\tau_L(\cdot)$ truncates the sequence to the maximum allowed token length.

---





# 7.3 Neighbor-Aware Contextualization

For each entity, one-hop graph neighborhoods are extracted from the training graph:

```math id="fjj7c4"
\mathcal{N}_K(e)
```

where $K$ denotes the maximum neighborhood size.  

Neighborhood context is constructed by concatenating neighboring relations and entities:

```math id="a0hlx6"
\mathbf{c}(e) = \operatorname{concat}(r', e')
```

Only neighborhoods from the training split are used to avoid information leakage.

---








# 7.4 Contextualized Entity Construction

The contextualized entity representation is formed as:

```math id="r9z7bl"
\mathbf{u}(e) =
\tilde{\mathbf{x}}(e)
\oplus [\texttt{CTX}]
\oplus \mathbf{c}(e)
```

where `[CTX]` is a special context token.

---






# 7.5 Relation-Aware Prompt Construction

A relation prediction prompt is constructed for each entity pair $(h,t)$:

```math id="1qg54x"
\mathbf{s}(h,t)
```

The prompt sequence contains:

* Head entity representation
* Tail entity representation
* Contextual neighborhood information
* Prompt tokens
* Special role tokens

Special tokens include:

* `[HEAD]`
* `[TAIL]`
* `[CTX]`
* `[PROMPT]`

---






# 7.6 Transformer Encoding

The input sequence is tokenized into:

```math id="f56pup"
\{w_1, w_2, \dots, w_L\}
```

Input embeddings are constructed using:

* Word embeddings
* Segment embeddings
* Role embeddings

The resulting hidden representation matrix is:

```math id="o5d9l5"
\mathbf{H}^{(0)} \in \mathbb{R}^{L \times d}
```

The sequence is encoded through $N$ transformer layers:

```math id="ihdc1m"
\mathbf{H}^{(N)}
```

using a pretrained BERT encoder.

---





# 7.7 Global and Span Representations

The global sequence representation is extracted from the CLS token:

```math id="o7j0m2"
\mathbf{z}_{\mathrm{cls}} = \mathbf{H}^{(N)}_1
```

Head and tail entity representations are computed using masked mean pooling:

```math id="jlhfwo"
\mathbf{z}_h, \mathbf{z}_t
```

over the corresponding token spans.

---





# 7.8 Cross-Entity Interaction Modeling

An interaction representation is constructed using:

```math id="byxw64"
\mathbf{z}_{\mathrm{int}} =
[
\mathbf{z}_h ;
\mathbf{z}_t ;
|\mathbf{z}_h - \mathbf{z}_t| ;
\mathbf{z}_h \odot \mathbf{z}_t
]
```

where:

* $|\cdot|$ denotes absolute difference
* $\odot$ denotes element-wise multiplication

---








# 7.9 Gated Fusion Mechanism

The model computes a gating vector:

```math id="m7o1r6"
\mathbf{g} = \sigma(\mathbf{W}_g[\mathbf{z}_{cls};\mathbf{z}_{int}] + b_g)
```

The fused representation is computed as:

```math id="0a6myk"
\mathbf{z}_{fuse}
=
\mathbf{g} \odot \mathbf{z}_{cls}
+
(1-\mathbf{g}) \odot \mathbf{z}_{int}
```

---






# 7.10 Relation Prediction

The final fused representation is passed through a classification layer:

```math id="w52f6r"
\mathbf{o}(h,t)
```

Relation probabilities are computed using softmax:

```math id="dl4x1k"
P(r'|h,t)
=
\operatorname{softmax}(\mathbf{o}(h,t))
```

---






# 7.11 Loss Function

For each triple $(h,r,t)$, the training objective minimizes the negative log-likelihood:

```math id="f4ocpz"
\ell(h,r,t)
=
-\log P(r|h,t)
```

The implementation additionally supports contrastive learning regularization.

---






# 7.12 Training Procedure

Training is performed using mini-batch optimization.

For each minibatch:

1. Entity texts are constructed
2. Neighborhood contexts are extracted
3. Contextualized entity sequences are formed
4. Relation-aware prompts are generated
5. Token embeddings are computed
6. Transformer encoding is applied
7. Interaction representations are computed
8. Gated fusion is performed
9. Relation probabilities are predicted
10. Classification loss is computed
11. Parameters are updated using AdamW optimizer

Learning rate scheduling uses linear warmup with decay.

---






# 7.13 Evaluation Methodology

The framework is evaluated using:

* Accuracy
* Mean Rank (MR)
* Mean Reciprocal Rank (MRR)
* Hits@1
* Hits@3
* Hits@10

The proposed framework is compared against multiple knowledge graph completion baselines.

An ablation study evaluates the contribution of:

* Neighbor-aware contextualization
* Role embeddings
* Interaction module
* Gated fusion
* Contrastive learning objective





Evaluation is conducted on benchmark datasets including:

* FB15k
* FB15k-237
* WN18
* WN18RR
* NELL995
