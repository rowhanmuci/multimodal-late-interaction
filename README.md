# Multimodal Late Interaction Retrieval

Comparing **single-vector cosine similarity (Baseline)** vs. **Late Interaction MaxSim (Proposed)** for image-text retrieval on Flickr30k, using [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) as the backbone.

The key idea: instead of pooling all token embeddings into one vector and computing a single dot product, Late Interaction retains all token-level embeddings and scores a query-document pair via MaxSim — the sum of each query token's maximum similarity to any document token. This is the ColBERT-style interaction adapted for vision-language models.

---

## Results on Flickr30k (1000 images × 5 captions)

### Baseline — Single-Vector Cosine Similarity

| Direction | R@1 | R@5 | R@10 | MRR |
|-----------|-----|-----|------|-----|
| T2I (Text → Image) | 48.53% | 70.83% | 78.48% | 58.81% |
| I2T (Image → Text) | 67.43% | 86.80% | 91.94% | 76.03% |

> Late Interaction results pending — run `late_interaction.py` to reproduce.

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

Both methods share the same model, dataset split, ground truth definition, and evaluation code — only the scoring strategy differs.

---

## Setup

```bash
pip install transformers accelerate datasets qwen-vl-utils tqdm torch
```

GPU: tested on RTX 5070 12 GB. Peak VRAM ~5 GB for extraction (batch 8 images or 32 texts).

The model and dataset are downloaded automatically on first run and cached by HuggingFace.

---

## Usage

### Baseline

```bash
# Full benchmark (Flickr30k, ~5 min first run, ~5 s with cache)
python baseline.py

# Quick smoke test (Flickr8k)
python baseline.py --smoke
```

Saves `image_embs.pt` [1000×2048], `text_embs.pt` [5000×2048], `results.json`.

### Late Interaction

```bash
# Step 1 — extract token-level embeddings (~10-15 min, run once)
python late_interaction.py --extract

# Step 2 — compute MaxSim scores and evaluate
python late_interaction.py
```

Saves `image_token_embs.pt`, `text_token_embs.pt`, `image_token_mask.pt`, `text_token_mask.pt` (all large, gitignored), `results_late_interaction.json`.

---

## Repository Layout

```
├── baseline.py                  # Single-vector pipeline
├── late_interaction.py          # Late Interaction MaxSim pipeline
├── results.json                 # Baseline evaluation results
└── Discussion_Summary.md        # Implementation notes and bug fixes
```

Generated at runtime (gitignored):

```
├── image_embs.pt / text_embs.pt              # Baseline pooled embeddings
├── image_token_embs.pt / text_token_embs.pt  # Token-level embeddings (Late Interaction)
├── image_token_mask.pt / text_token_mask.pt  # Attention masks
└── results_late_interaction.json
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
