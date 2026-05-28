# Multimodal Late Interaction Retrieval

Comparing **single-vector cosine similarity (Baseline)** vs. **Late Interaction MaxSim (Proposed)** for image-text retrieval on Flickr30k, using [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) as the backbone.

The key idea: instead of pooling all token embeddings into one vector and computing a single dot product, Late Interaction retains all token-level embeddings and scores a query-document pair via MaxSim — the sum of each query token's maximum similarity to any document token. This is the ColBERT-style interaction adapted for vision-language models.

---

## Results on Flickr30k (1K evaluation pool)

Both methods use the same model, dataset, and evaluation code. Only the scoring strategy differs.

### Main Comparison

| Direction | Baseline (2048d) | Late Interaction (2048d) | Delta |
|-----------|-----------------|--------------------------|-------|
| T2I R@1   | 82.44%          | 81.62%                   | -0.82% |
| T2I R@5   | 96.52%          | 95.22%                   | -1.30% |
| T2I MRR   | 88.62%          | 87.76%                   | -0.86% |
| I2T R@1   | **94.10%**      | 81.30%                   | -12.80% |
| I2T R@5   | 99.50%          | 95.30%                   | -4.20% |
| I2T MRR   | **96.36%**      | 87.39%                   | -8.97% |

**Finding**: Baseline outperforms Late Interaction on this dataset, particularly for I2T. Qwen3-VL-Embedding is trained with a last-token pooling objective, so the single pooled vector captures semantics more effectively than unfiltered token-level MaxSim. Visual tokens (~300 per image) introduce noise when summed over in MaxSim.

### Dimension Ablation

| Dim  | Baseline T2I R@1 | LI T2I R@1 | Baseline I2T R@1 | LI I2T R@1 |
|------|-----------------|------------|-----------------|------------|
| 2048 | 82.44%          | 81.62%     | 94.10%          | 81.30%     |
| 1024 | 82.38%          | 81.46%     | 94.00%          | 80.10%     |
| 512  | 81.86%          | 81.30%     | 93.80%          | 80.20%     |
| 256  | 80.12%          | 81.28%     | 92.30%          | 79.60%     |

**Finding**: Late Interaction is significantly more robust to dimensionality reduction. T2I R@1 drops only **0.34%** from 2048→256 for LI, vs **2.32%** for Baseline. This advantage holds for I2T as well.

### Baseline on Full 31K Pool

For reference, Baseline evaluated on the full 31,014-image pool:

| Direction | R@1    | R@5    | R@10   | MRR    |
|-----------|--------|--------|--------|--------|
| T2I       | 48.53% | 70.83% | 78.48% | 58.81% |
| I2T       | 67.43% | 86.80% | 91.94% | 76.03% |

---

## Method

```
Backbone: Qwen3-VL-Embedding-2B  (LLM-based VL embedding, 2B params, BF16)

Baseline:
  Image / Text → model.model(**inputs).last_hidden_state
              → last non-padding token pooling  (right-padded, NOT [:,-1,:])
              → L2 normalize → single vector [2048]
              → score = dot(txt, img)

Late Interaction:
  Image / Text → model.model(**inputs).last_hidden_state
              → retain ALL token embeddings  [seq_len × 2048]
              → L2 normalize per token
              → score(q, d) = Σ_i  max_j  cos(q_i, d_j)   (padding masked to -inf)
```

---

## Setup

```bash
pip install transformers accelerate datasets qwen-vl-utils tqdm torch
```

GPU: tested on RTX 5070 12 GB. Peak VRAM ~5 GB for extraction (batch 8 images or 32 texts).

---

## Usage

### Baseline

```bash
python baseline.py          # Full Flickr30k (31K pool)
python baseline.py --smoke  # Quick smoke test on Flickr8k
```

### Late Interaction

```bash
# Step 1: extract token-level embeddings (~10 min, run once)
python late_interaction.py --extract

# Step 2: compute MaxSim scores and evaluate
python late_interaction.py
```

### Dimension Ablation

```bash
# Requires image_embs.pt + text_embs.pt (from baseline.py)
# Optionally uses token embeddings (from late_interaction.py --extract)
python ablation.py
```

---

## Repository Layout

```
├── baseline.py                  # Single-vector pipeline
├── late_interaction.py          # Late Interaction MaxSim pipeline
├── ablation.py                  # Dimension reduction ablation
├── results.json                 # Baseline 31K results
├── results_late_interaction.json  # Late Interaction 1K results
├── ablation_results.json        # Ablation results
└── Discussion_Summary.md        # Implementation notes and bug fixes
```

Generated at runtime (gitignored — large files):

```
├── image_embs.pt / text_embs.pt              # Baseline pooled embeddings (31K)
├── image_token_embs.pt / text_token_embs.pt  # Token-level embeddings (1K)
└── image_token_mask.pt / text_token_mask.pt  # Attention masks
```

---

## Key Implementation Notes

**Correct hidden state extraction** from `Qwen3VLForConditionalGeneration`:

```python
outputs = model.model(**inputs)       # call base model, NOT model(**inputs)
hidden  = outputs.last_hidden_state   # [B, seq_len, 2048]
# model(**inputs)[0] returns logits [B, seq_len, 151936] — wrong
```

**Pooling** (right-padded sequences):

```python
last_idx = inputs["attention_mask"].sum(dim=1) - 1
vec = hidden[torch.arange(B), last_idx, :]   # NOT hidden[:, -1, :]
```

**MaxSim padding mask**:

```python
sim_flat[:, doc_pad_flat] = float('-inf')   # exclude padding before max
```
