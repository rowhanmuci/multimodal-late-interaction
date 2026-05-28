"""
ViDoRe benchmark evaluation: Baseline (EOS) vs Late Interaction (MaxSim) vs Hybrid.

Dataset: vidore/docvqa_test_subsampled (500 query-document pairs, 1-to-1 GT)
Metrics: Recall@1/5/10, NDCG@5 (primary ViDoRe metric)

Usage:
  # Extract embeddings + evaluate (first run, ~20-30 min)
  python vidore_eval.py

  # Re-evaluate only (embeddings cached)
  python vidore_eval.py --eval-only
"""

import os
import json
import math
import argparse
import torch
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from datasets import load_dataset
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID  = "Qwen/Qwen3-VL-Embedding-2B"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE     = torch.bfloat16

DATASET_ID = "vidore/docvqa_test_subsampled"

IMG_BATCH = 2    # large doc images (~1300 tokens each)
TXT_BATCH = 64

# Cache paths
CACHE_DIR       = "vidore_cache"
IMG_EOS_PATH    = f"{CACHE_DIR}/img_eos.pt"
TXT_EOS_PATH    = f"{CACHE_DIR}/txt_eos.pt"
IMG_TOK_PATH    = f"{CACHE_DIR}/img_tok.pt"
TXT_TOK_PATH    = f"{CACHE_DIR}/txt_tok.pt"
IMG_MASK_PATH   = f"{CACHE_DIR}/img_mask.pt"
TXT_MASK_PATH   = f"{CACHE_DIR}/txt_mask.pt"
RESULTS_PATH    = "results_vidore.json"

os.makedirs(CACHE_DIR, exist_ok=True)


# ── Model ──────────────────────────────────────────────────────────────────────
def load_model():
    print(f"[*] Loading model ({DTYPE}) to {DEVICE}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE,
    )
    model.eval()
    print(f"    VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return processor, model


# ── Dataset ────────────────────────────────────────────────────────────────────
def load_vidore():
    print(f"[*] Loading {DATASET_ID}...")
    ds = load_dataset(DATASET_ID, split="test")
    n  = len(ds)
    print(f"    {n} query-document pairs (1-to-1 GT).")
    queries = [ds[i]["query"] for i in range(n)]
    images  = [ds[i]["image"] for i in range(n)]
    gt      = list(range(n))   # query i → image i
    return queries, images, gt


# ── Extraction ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def _extract(model, processor, formatted_texts, images_list, batch_size):
    """Returns (eos_embs [N, 2048] f32, tok_embs [N, max_len, 2048] f16, masks [N, max_len] bool)."""
    all_eos, all_tok, all_mask = [], [], []

    for start in tqdm(range(0, len(formatted_texts), batch_size)):
        batch_texts  = formatted_texts[start : start + batch_size]
        batch_images = images_list[start : start + batch_size] if images_list else None

        inputs = processor(
            text=batch_texts, images=batch_images,
            return_tensors="pt", padding=True,
        ).to(DEVICE)

        hidden = model.model(**inputs).last_hidden_state  # [B, seq_len, 2048]

        # EOS: last non-padding token
        mask     = inputs["attention_mask"]
        last_idx = mask.sum(dim=1) - 1
        B        = hidden.shape[0]
        eos = F.normalize(hidden[torch.arange(B), last_idx].float(), p=2, dim=-1)

        # Token-level: L2-normalize per token
        tok = F.normalize(hidden.float(), p=2, dim=-1).half()

        all_eos.append(eos.cpu())
        all_tok.append(tok.cpu())
        all_mask.append(mask.bool().cpu())

    eos_embs = torch.cat(all_eos, dim=0)

    # Pad token tensors to global max_len
    max_len = max(t.shape[1] for t in all_tok)
    pad_tok, pad_mask = [], []
    for tok, msk in zip(all_tok, all_mask):
        B, L, D = tok.shape
        if L < max_len:
            tok = torch.cat([tok,  torch.zeros(B, max_len - L, D, dtype=tok.dtype)],  dim=1)
            msk = torch.cat([msk,  torch.zeros(B, max_len - L, dtype=torch.bool)],    dim=1)
        pad_tok.append(tok)
        pad_mask.append(msk)

    return eos_embs, torch.cat(pad_tok, dim=0), torch.cat(pad_mask, dim=0)


def extract_and_cache(processor, model, queries, images):
    print("[*] Formatting queries...")
    fmt_queries = []
    for q in queries:
        msg  = [{"role": "user", "content": q}]
        fmt_queries.append(processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False))

    print("[*] Formatting document images...")
    fmt_img_texts = []
    for img in images:
        msg = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text",  "text":  "Represent this document."},
        ]}]
        fmt_img_texts.append(processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False))

    print(f"[*] Extracting query embeddings (batch={TXT_BATCH})...")
    txt_eos, txt_tok, txt_mask = _extract(model, processor, fmt_queries, None, TXT_BATCH)

    print(f"[*] Extracting image embeddings (batch={IMG_BATCH})...")
    img_eos, img_tok, img_mask = _extract(model, processor, fmt_img_texts, images, IMG_BATCH)

    torch.save(txt_eos,  TXT_EOS_PATH)
    torch.save(img_eos,  IMG_EOS_PATH)
    torch.save(txt_tok,  TXT_TOK_PATH)
    torch.save(img_tok,  IMG_TOK_PATH)
    torch.save(txt_mask, TXT_MASK_PATH)
    torch.save(img_mask, IMG_MASK_PATH)
    print(f"[*] Saved: img_tok {img_tok.shape}, txt_tok {txt_tok.shape}")
    return txt_eos, img_eos, txt_tok, img_tok, txt_mask, img_mask


def load_cache():
    print("[*] Loading cached embeddings...")
    return (
        torch.load(TXT_EOS_PATH,  weights_only=True),
        torch.load(IMG_EOS_PATH,  weights_only=True),
        torch.load(TXT_TOK_PATH,  weights_only=True),
        torch.load(IMG_TOK_PATH,  weights_only=True),
        torch.load(TXT_MASK_PATH, weights_only=True),
        torch.load(IMG_MASK_PATH, weights_only=True),
    )


# ── MaxSim ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_maxsim(query_embs, query_masks, doc_embs, doc_masks):
    """query-by-query MaxSim. Returns [Q, D] float32."""
    Q, q_max, dim = query_embs.shape
    D, d_max, _   = doc_embs.shape

    score_matrix = torch.zeros(Q, D, dtype=torch.float32)
    doc_gpu  = doc_embs.to(DEVICE)
    dmsk_gpu = doc_masks.to(DEVICE)
    docs_flat    = doc_gpu.view(D * d_max, dim)
    doc_pad_flat = (~dmsk_gpu).view(D * d_max)

    for i in tqdm(range(Q), desc="MaxSim"):
        q_valid = query_embs[i][query_masks[i]].to(DEVICE)
        if q_valid.shape[0] == 0:
            continue
        sim = q_valid.float() @ docs_flat.T.float()   # [q_len, D*d_max]
        sim[:, doc_pad_flat] = float("-inf")
        sim = sim.view(q_valid.shape[0], D, d_max)
        # Mean over query tokens (length-normalized, avoids length bias)
        score_matrix[i] = sim.max(dim=-1).values.mean(dim=0).cpu()

    return score_matrix


# ── Metrics ────────────────────────────────────────────────────────────────────
def recall_at_k(scores, gt, k_list=(1, 5, 10)):
    sorted_idx = torch.argsort(scores, dim=1, descending=True)
    hits = {k: 0 for k in k_list}
    for i, g in enumerate(gt):
        for k in k_list:
            if g in sorted_idx[i, :k].tolist():
                hits[k] += 1
    n = len(gt)
    return {f"R@{k}": round(v / n * 100, 2) for k, v in hits.items()}


def ndcg_at_k(scores, gt, k=5):
    sorted_idx = torch.argsort(scores, dim=1, descending=True)
    ndcg = 0.0
    for i, g in enumerate(gt):
        top_k = sorted_idx[i, :k].tolist()
        if g in top_k:
            rank = top_k.index(g) + 1   # 1-indexed
            ndcg += 1.0 / math.log2(rank + 1)
    return round(ndcg / len(gt) * 100, 2)


def mean_reciprocal_rank(scores, gt):
    sorted_idx = torch.argsort(scores, dim=1, descending=True)
    mrr = 0.0
    for i, g in enumerate(gt):
        rank = sorted_idx[i].tolist().index(g) + 1
        mrr += 1.0 / rank
    return round(mrr / len(gt) * 100, 2)


def eval_scores(scores, gt, label):
    r = recall_at_k(scores, gt)
    ndcg = ndcg_at_k(scores, gt, k=5)
    mrr  = mean_reciprocal_rank(scores, gt)
    print(f"\n=== {label} ===")
    for k, v in r.items():
        print(f"  {k}: {v:.2f}%")
    print(f"  NDCG@5: {ndcg:.2f}%")
    print(f"  MRR:    {mrr:.2f}%")
    return {**r, "NDCG@5": ndcg, "MRR": mrr}


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip extraction, use cached embeddings")
    args = parser.parse_args()

    all_cached = all(os.path.exists(p) for p in
                     [TXT_EOS_PATH, IMG_EOS_PATH, TXT_TOK_PATH, IMG_TOK_PATH,
                      TXT_MASK_PATH, IMG_MASK_PATH])

    queries, images, gt = load_vidore()

    if not all_cached or not args.eval_only:
        processor, model = load_model()
        txt_eos, img_eos, txt_tok, img_tok, txt_mask, img_mask = \
            extract_and_cache(processor, model, queries, images)
        del processor, model
        torch.cuda.empty_cache()
        print("[*] Model unloaded.")
    else:
        txt_eos, img_eos, txt_tok, img_tok, txt_mask, img_mask = load_cache()

    results = {}

    # ── Baseline (EOS cosine) ──
    print("\n[*] Computing Baseline (EOS cosine)...")
    baseline_scores = txt_eos @ img_eos.T   # [N, N]
    results["baseline"] = eval_scores(baseline_scores, gt, "Baseline (EOS)")

    # ── Pure MaxSim ──
    print("\n[*] Computing Pure MaxSim (Late Interaction)...")
    maxsim_scores = compute_maxsim(txt_tok, txt_mask, img_tok, img_mask)
    results["maxsim"] = eval_scores(maxsim_scores, gt, "Pure MaxSim (LI)")

    # ── Hybrid α sweep ──
    print("\n[*] Computing Hybrid scores (EOS + MaxSim)...")
    for alpha in [0.3, 0.5, 0.7]:
        hybrid = (1 - alpha) * baseline_scores + alpha * maxsim_scores
        results[f"hybrid_a{int(alpha*10)}"] = eval_scores(
            hybrid, gt, f"Hybrid alpha={alpha}"
        )

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[*] Results saved to {RESULTS_PATH}")
