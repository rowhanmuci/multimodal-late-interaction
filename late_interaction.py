"""
Late Interaction (MaxSim) pipeline for Flickr30k retrieval.

Usage:
  # Step 1: Extract token-level embeddings (run once, ~5-10 min)
  python late_interaction.py --extract

  # Step 2: Compute Late Interaction scores and evaluate
  python late_interaction.py
"""

import os
import json
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

IMG_BATCH = 8    # smaller than baseline due to storing all token embeddings
TXT_BATCH = 32

IMG_TOK_PATH  = "image_token_embs.pt"   # [1000, max_img_len, 2048] fp16
TXT_TOK_PATH  = "text_token_embs.pt"    # [5000, max_txt_len, 2048] fp16
IMG_MASK_PATH = "image_token_mask.pt"   # [1000, max_img_len] bool
TXT_MASK_PATH = "text_token_mask.pt"    # [5000, max_txt_len] bool
RESULTS_PATH  = "results_late_interaction.json"


# ── Model ──────────────────────────────────────────────────────────────────────
def load_model():
    print(f"[*] Loading processor from {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print(f"[*] Loading model ({DTYPE}) to {DEVICE}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map=DEVICE,
    )
    model.eval()
    print(f"    VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return processor, model


# ── Dataset ────────────────────────────────────────────────────────────────────
N_EVAL = 1000   # standard 1K retrieval benchmark subset

def load_flickr30k():
    print("[*] Loading nlphuji/flickr30k test split (first 1000 images)...")
    ds = load_dataset("nlphuji/flickr30k", split="test")
    n  = min(len(ds), N_EVAL)
    print(f"    Using {n} images.")

    images   = [ds[i]["image"] for i in range(n)]
    captions = [ds[i]["caption"][j] for i in range(n) for j in range(5)]

    t2i_gt = [c // 5 for c in range(len(captions))]
    i2t_gt = [[5*i + j for j in range(5)] for i in range(n)]

    return images, captions, t2i_gt, i2t_gt


# ── Token-level extraction ─────────────────────────────────────────────────────
@torch.no_grad()
def _extract_tokens(model, processor, formatted_texts, images_list, batch_size):
    """
    Extract ALL token embeddings (not pooled). Returns padded tensor + mask.

    Returns:
        embs:  [N, max_len, 2048] float16
        masks: [N, max_len] bool  (True = valid token, False = padding)
    """
    all_embs  = []
    all_masks = []

    for start in tqdm(range(0, len(formatted_texts), batch_size)):
        batch_texts  = formatted_texts[start : start + batch_size]
        batch_images = images_list[start : start + batch_size] if images_list else None

        inputs = processor(
            text=batch_texts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
        ).to(DEVICE)

        outputs = model.model(**inputs)
        hidden  = outputs.last_hidden_state   # [B, seq_len, 2048]

        # L2-normalize each token embedding
        hidden = F.normalize(hidden.float(), p=2, dim=-1).half()  # keep fp16

        mask = inputs["attention_mask"].bool()  # [B, seq_len]

        all_embs.append(hidden.cpu())
        all_masks.append(mask.cpu())

    # Pad to same max_len across all batches
    max_len = max(e.shape[1] for e in all_embs)

    padded_embs  = []
    padded_masks = []
    for emb, msk in zip(all_embs, all_masks):
        B, L, D = emb.shape
        if L < max_len:
            pad_e = torch.zeros(B, max_len - L, D, dtype=emb.dtype)
            pad_m = torch.zeros(B, max_len - L, dtype=torch.bool)
            emb = torch.cat([emb, pad_e], dim=1)
            msk = torch.cat([msk, pad_m], dim=1)
        padded_embs.append(emb)
        padded_masks.append(msk)

    return torch.cat(padded_embs, dim=0), torch.cat(padded_masks, dim=0)


def extract_text_token_embeddings(processor, model, captions):
    print("[*] Formatting captions...")
    formatted = []
    for cap in captions:
        msg  = [{"role": "user", "content": cap}]
        text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        formatted.append(text)

    print(f"[*] Extracting text token embeddings ({len(captions)} captions, batch={TXT_BATCH})...")
    return _extract_tokens(model, processor, formatted, images_list=None, batch_size=TXT_BATCH)


def extract_image_token_embeddings(processor, model, images):
    print("[*] Formatting images...")
    formatted_texts = []
    for img in images:
        msg = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text",  "text":  "Represent this image."},
        ]}]
        text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        formatted_texts.append(text)

    print(f"[*] Extracting image token embeddings ({len(images)} images, batch={IMG_BATCH})...")
    return _extract_tokens(model, processor, formatted_texts, images_list=images, batch_size=IMG_BATCH)


def get_token_embeddings(processor, model, images, captions):
    cached = all(os.path.exists(p) for p in
                 [IMG_TOK_PATH, TXT_TOK_PATH, IMG_MASK_PATH, TXT_MASK_PATH])

    if cached:
        print("[*] Loading cached token embeddings...")
        img_embs  = torch.load(IMG_TOK_PATH,  weights_only=True)
        txt_embs  = torch.load(TXT_TOK_PATH,  weights_only=True)
        img_masks = torch.load(IMG_MASK_PATH, weights_only=True)
        txt_masks = torch.load(TXT_MASK_PATH, weights_only=True)
    else:
        txt_embs, txt_masks = extract_text_token_embeddings(processor, model, captions)
        img_embs, img_masks = extract_image_token_embeddings(processor, model, images)

        torch.save(img_embs,  IMG_TOK_PATH)
        torch.save(txt_embs,  TXT_TOK_PATH)
        torch.save(img_masks, IMG_MASK_PATH)
        torch.save(txt_masks, TXT_MASK_PATH)
        print(f"[*] Saved: img {img_embs.shape}, txt {txt_embs.shape}")

    return img_embs, img_masks, txt_embs, txt_masks


# ── MaxSim scorer ──────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_maxsim(query_embs, query_masks, doc_embs, doc_masks, desc="MaxSim"):
    """
    Late Interaction MaxSim scorer.

    query_embs:  [Q, q_max_len, dim] fp16
    query_masks: [Q, q_max_len]      bool
    doc_embs:    [D, d_max_len, dim] fp16
    doc_masks:   [D, d_max_len]      bool

    Returns score_matrix: [Q, D] float32
    """
    Q, q_max, dim = query_embs.shape
    D, d_max, _   = doc_embs.shape

    score_matrix = torch.zeros(Q, D, dtype=torch.float32)

    # Move docs to GPU once
    doc_embs_gpu  = doc_embs.to(DEVICE)                   # [D, d_max, dim]
    doc_masks_gpu = doc_masks.to(DEVICE)                   # [D, d_max]
    docs_flat     = doc_embs_gpu.view(D * d_max, dim)      # [D*d_max, dim]
    # Padding positions across all docs (flattened)
    doc_pad_flat  = (~doc_masks_gpu).view(D * d_max)       # [D*d_max]

    for i in tqdm(range(Q), desc=desc):
        q_valid = query_embs[i][query_masks[i]].to(DEVICE)   # [q_valid_len, dim]

        if q_valid.shape[0] == 0:
            continue

        # Similarity: [q_valid_len, D*d_max]
        sim_flat = (q_valid.float() @ docs_flat.T.float())

        # Mask doc padding positions → -inf before max
        sim_flat[:, doc_pad_flat] = float('-inf')

        # Reshape → [q_valid_len, D, d_max]
        sim = sim_flat.view(q_valid.shape[0], D, d_max)

        # Max over doc tokens → [q_valid_len, D]
        max_sim = sim.max(dim=-1).values

        # Sum over query tokens → [D]
        score_matrix[i] = max_sim.sum(dim=0).cpu()

    return score_matrix


# ── Metrics (same logic as baseline.py) ───────────────────────────────────────
def recall_at_k(scores, ground_truth_list, k_list=(1, 5, 10)):
    sorted_idx = torch.argsort(scores, dim=1, descending=True)
    results    = {k: 0.0 for k in k_list}

    for q, gt in enumerate(ground_truth_list):
        gt_set = {gt} if isinstance(gt, int) else set(gt)
        for k in k_list:
            if any(idx.item() in gt_set for idx in sorted_idx[q, :k]):
                results[k] += 1.0

    n = len(ground_truth_list)
    return {f"R@{k}": round(v / n * 100, 2) for k, v in results.items()}


def mean_reciprocal_rank(scores, ground_truth_list):
    sorted_idx = torch.argsort(scores, dim=1, descending=True)
    mrr = 0.0
    for q, gt in enumerate(ground_truth_list):
        gt_set = {gt} if isinstance(gt, int) else set(gt)
        for rank, idx in enumerate(sorted_idx[q].tolist(), start=1):
            if idx in gt_set:
                mrr += 1.0 / rank
                break
    return round(mrr / len(ground_truth_list) * 100, 2)


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate(img_embs, img_masks, txt_embs, txt_masks, t2i_gt, i2t_gt):
    results = {}

    print("\n[*] Computing T2I scores (text query -> image doc)...")
    t2i_scores = compute_maxsim(txt_embs, txt_masks, img_embs, img_masks, desc="T2I MaxSim")

    print("\n=== T2I (Text -> Image) ===")
    t2i_r   = recall_at_k(t2i_scores, t2i_gt)
    t2i_mrr = mean_reciprocal_rank(t2i_scores, t2i_gt)
    results["T2I"] = {**t2i_r, "MRR": t2i_mrr}
    for k, v in t2i_r.items():
        print(f"  {k}: {v:.2f}%")
    print(f"  MRR: {t2i_mrr:.2f}%")

    print("\n[*] Computing I2T scores (image query -> text doc)...")
    i2t_scores = compute_maxsim(img_embs, img_masks, txt_embs, txt_masks, desc="I2T MaxSim")

    print("\n=== I2T (Image -> Text) ===")
    i2t_r   = recall_at_k(i2t_scores, i2t_gt)
    i2t_mrr = mean_reciprocal_rank(i2t_scores, i2t_gt)
    results["I2T"] = {**i2t_r, "MRR": i2t_mrr}
    for k, v in i2t_r.items():
        print(f"  {k}: {v:.2f}%")
    print(f"  MRR: {i2t_mrr:.2f}%")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[*] Results saved to {RESULTS_PATH}")
    return results


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true",
                        help="Force re-extraction of token-level embeddings")
    args = parser.parse_args()

    if args.extract:
        for p in [IMG_TOK_PATH, TXT_TOK_PATH, IMG_MASK_PATH, TXT_MASK_PATH]:
            if os.path.exists(p):
                os.remove(p)
                print(f"[*] Removed {p}")

    # Ground truth is deterministic — no need to load the dataset for it
    t2i_gt = [c // 5 for c in range(5000)]
    i2t_gt = [[5 * i + j for j in range(5)] for i in range(1000)]

    cached = all(os.path.exists(p) for p in
                 [IMG_TOK_PATH, TXT_TOK_PATH, IMG_MASK_PATH, TXT_MASK_PATH])

    if cached:
        img_embs, img_masks, txt_embs, txt_masks = get_token_embeddings(
            None, None, None, None
        )
    else:
        processor, model = load_model()
        images, captions, _, _ = load_flickr30k()
        img_embs, img_masks, txt_embs, txt_masks = get_token_embeddings(
            processor, model, images, captions
        )
        del processor, model
        torch.cuda.empty_cache()
        print("[*] Model unloaded — VRAM freed for MaxSim.")

    evaluate(img_embs, img_masks, txt_embs, txt_masks, t2i_gt, i2t_gt)
