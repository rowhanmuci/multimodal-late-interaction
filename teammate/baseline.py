import os
import json
import torch
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from datasets import load_dataset
from tqdm import tqdm
import gc

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID   = "Qwen/Qwen3-VL-Embedding-2B"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE      = torch.bfloat16   # matches model config; RTX 5070 Blackwell native BF16

IMG_BATCH  = 4
TXT_BATCH  = 64
EMB_DIM    = 2048

IMG_EMB_PATH = "image_embs.pt"
TXT_EMB_PATH = "text_embs.pt"
RESULTS_PATH = "results.json"

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
def load_flickr(dataset_id="nlphuji/flickr30k", split="test"):
    """
    Returns dataset (lazy-loaded), captions (flat list of str), and ground truths.
    Each image has exactly 5 captions: captions[5*i + j] belongs to image i.
    Images are NOT pre-loaded; the dataset object supports lazy per-item access.
    """
    print(f"[*] Loading {dataset_id} ({split})...")
    ds = load_dataset(dataset_id, split=split)

    # nlphuji/flickr30k puts all 31k images in one "test" split,
    # but each row has a 'split' column with the Karpathy split label.
    if "split" in ds.column_names:
        full_size = len(ds)
        ds = ds.filter(lambda x: x["split"] == "test")
        print(f"    Filtered Karpathy test set: {len(ds)}/{full_size} images")

    n  = len(ds)
    print(f"    {n} images loaded.")

    print("[*] Pre-loading images into RAM...")
    images_list = [img.convert("RGB") for img in tqdm(ds["image"], desc="  Loading")]

    # Flickr30k: caption field is a list of 5 strings
    # Flickr8k:  caption_0 .. caption_4 are separate fields
    if "caption" in ds.column_names:
        captions = [ds[i]["caption"][j] for i in range(n) for j in range(5)]
    else:
        captions = [ds[i][f"caption_{j}"] for i in range(n) for j in range(5)]

    t2i_gt = [c // 5 for c in range(len(captions))]              # caption c → image c//5
    i2t_gt = [[5*i + j for j in range(5)] for i in range(n)]     # image i → 5 captions

    return images_list, captions, t2i_gt, i2t_gt


# ── Embedding extraction ────────────────────────────────────────────────────────
@torch.no_grad()
def _extract(model, processor, formatted_texts, images_list, batch_size):
    """
    Core extraction loop. Returns L2-normalised float32 embeddings on CPU.
    Uses last non-padding token pooling (right-padded sequences).
    """
    all_embs = []

    for start in tqdm(range(0, len(formatted_texts), batch_size)):
        batch_texts  = formatted_texts[start : start + batch_size]
        batch_images = images_list[start : start + batch_size] if images_list else None

        inputs = processor(
            text=batch_texts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
        ).to(DEVICE)

        # Use base model to get hidden states without computing the LM head
        # model.model is Qwen3VLModel; outputs[0] on the full model is logits (vocab_size=151936)
        outputs = model.model(**inputs)
        hidden  = outputs.last_hidden_state   # [B, seq_len, EMB_DIM]

        # Last non-padding token (right-padded → can't use [:, -1, :] in a batch)
        mask     = inputs["attention_mask"]           # [B, seq_len]
        last_idx = mask.sum(dim=1) - 1                # [B]
        B        = hidden.shape[0]
        embs     = hidden[torch.arange(B), last_idx]  # [B, EMB_DIM]

        embs = F.normalize(embs.float(), p=2, dim=-1)
        all_embs.append(embs.cpu())

    return torch.cat(all_embs, dim=0)


def extract_text_embeddings(processor, model, captions):
    print("[*] Formatting captions...")
    formatted = []
    for cap in captions:
        msg  = [{"role": "user", "content": cap}]
        text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        formatted.append(text)

    print(f"[*] Extracting text embeddings ({len(captions)} captions, batch={TXT_BATCH})...")
    return _extract(model, processor, formatted, images_list=None, batch_size=TXT_BATCH)


@torch.no_grad()
def extract_image_embeddings(processor, model, images_list):
    """Extract image embeddings."""
    n = len(images_list)
    print(f"[*] Extracting image embeddings ({n} images, batch={IMG_BATCH})...")

    all_embs = []
    for start in tqdm(range(0, n, IMG_BATCH)):
        end = min(start + IMG_BATCH, n)
        batch_images = images_list[start:end]

        formatted_texts = []
        for img in batch_images:
            msg = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text":  "Represent this image."},
            ]}]
            text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            formatted_texts.append(text)

        inputs = processor(
            text=formatted_texts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
        ).to(DEVICE)

        outputs = model.model(**inputs)
        hidden  = outputs.last_hidden_state   # [B, seq_len, EMB_DIM]

        mask     = inputs["attention_mask"]           # [B, seq_len]
        last_idx = mask.sum(dim=1) - 1                # [B]
        B        = hidden.shape[0]
        embs     = hidden[torch.arange(B), last_idx]  # [B, EMB_DIM]

        embs = F.normalize(embs.float(), p=2, dim=-1)
        all_embs.append(embs.cpu())

        # Free batch memory immediately
        del batch_images, formatted_texts, inputs, outputs, hidden, embs

    return torch.cat(all_embs, dim=0)


def get_embeddings(processor, model, images_list, captions):
    if os.path.exists(IMG_EMB_PATH) and os.path.exists(TXT_EMB_PATH):
        print(f"[*] Loading cached embeddings from {IMG_EMB_PATH} / {TXT_EMB_PATH}")
        img_embs = torch.load(IMG_EMB_PATH, weights_only=True)
        txt_embs = torch.load(TXT_EMB_PATH, weights_only=True)
    else:
        txt_embs = extract_text_embeddings(processor, model, captions)
        img_embs = extract_image_embeddings(processor, model, images_list)
        torch.save(img_embs, IMG_EMB_PATH)
        torch.save(txt_embs, TXT_EMB_PATH)
        print(f"[*] Saved embeddings: img {img_embs.shape}, txt {txt_embs.shape}")

    return img_embs, txt_embs


# ── Sanity checks ───────────────────────────────────────────────────────────────
def sanity_check(img_embs, txt_embs):
    n_img = img_embs.shape[0]
    assert img_embs.shape == (n_img, EMB_DIM),   f"img shape mismatch: {img_embs.shape}"
    assert txt_embs.shape == (n_img * 5, EMB_DIM), f"txt shape mismatch: {txt_embs.shape}"
    assert (img_embs.norm(dim=1) - 1.0).abs().max() < 1e-4, "Image embeddings not L2-normalised"
    assert (txt_embs.norm(dim=1) - 1.0).abs().max() < 1e-4, "Text embeddings not L2-normalised"

    # Mean cosine similarity between each image and its 1st caption (should be > 0.3)
    self_sim = (txt_embs[::5] * img_embs).sum(dim=1).mean().item()
    print(f"[*] Self-similarity (img<->1st caption): {self_sim:.4f}  (expected > 0.3)")
    if self_sim < 0.3:
        print("    WARNING: low self-similarity — check pooling / input format")


# ── Metrics (chunked, GPU-accelerated) ──────────────────────────────────────────
@torch.no_grad()
def _chunked_recall_mrr(query_embs, cand_embs, ground_truths,
                        k_list=(1, 5, 10), chunk_size=256):
    """
    Compute Recall@K and MRR on GPU in chunks.
    Never materialises the full [n_query × n_cand] score matrix.
    """
    max_k       = max(k_list)
    n_queries   = query_embs.shape[0]
    recall_hits = {k: 0 for k in k_list}
    mrr_sum     = 0.0

    for start in tqdm(range(0, n_queries, chunk_size), desc="  eval"):
        end     = min(start + chunk_size, n_queries)
        chunk_q = query_embs[start:end]            # [chunk, dim]
        scores  = chunk_q @ cand_embs.T            # [chunk, n_cand]  ← small

        # ── Recall@K via topk (avoids full argsort) ──
        topk_idx = torch.topk(scores, k=max_k, dim=1).indices.cpu().tolist()

        for i, q_idx in enumerate(range(start, end)):
            gt     = ground_truths[q_idx]
            gt_set = {gt} if isinstance(gt, int) else set(gt)

            for k in k_list:
                if any(idx in gt_set for idx in topk_idx[i][:k]):
                    recall_hits[k] += 1

            # ── MRR: rank = (# items scoring higher than best GT) + 1 ──
            gt_list       = [gt] if isinstance(gt, int) else gt
            best_gt_score = scores[i, gt_list].max()
            rank          = (scores[i] > best_gt_score).sum().item() + 1
            mrr_sum      += 1.0 / rank

    n      = len(ground_truths)
    recall = {f"R@{k}": round(v / n * 100, 2) for k, v in recall_hits.items()}
    mrr    = round(mrr_sum / n * 100, 2)
    return recall, mrr


# ── Evaluation (GPU-chunked) ────────────────────────────────────────────────────
def evaluate(img_embs, txt_embs, t2i_gt, i2t_gt):
    """
    Evaluate on GPU in chunks to avoid the ~20 GB score matrix.
    Embeddings (~1.5 GB total) are moved to VRAM; scores are computed
    in chunks of 256 queries at a time (~160 MB peak per chunk).
    """
    device = DEVICE
    print(f"\n[*] Moving embeddings to {device} for evaluation...")
    img_gpu = img_embs.to(device)
    txt_gpu = txt_embs.to(device)
    vram = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0
    print(f"    VRAM after load: {vram:.2f} GB")

    # Free CPU copies — they are no longer needed
    del img_embs, txt_embs
    gc.collect()

    results = {}

    print("\n=== T2I (Text -> Image) ===")
    t2i_r, t2i_mrr = _chunked_recall_mrr(txt_gpu, img_gpu, t2i_gt)
    results["T2I"] = {**t2i_r, "MRR": t2i_mrr}
    for k, v in t2i_r.items():
        print(f"  {k}: {v:.2f}%")
    print(f"  MRR: {t2i_mrr:.2f}%")

    print("\n=== I2T (Image -> Text) ===")
    i2t_r, i2t_mrr = _chunked_recall_mrr(img_gpu, txt_gpu, i2t_gt)
    results["I2T"] = {**i2t_r, "MRR": i2t_mrr}
    for k, v in i2t_r.items():
        print(f"  {k}: {v:.2f}%")
    print(f"  MRR: {i2t_mrr:.2f}%")

    # Free GPU embeddings
    del img_gpu, txt_gpu
    torch.cuda.empty_cache()

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[*] Results saved to {RESULTS_PATH}")
    return results


# ── Main ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="nlphuji/flickr30k",
                        help="HuggingFace dataset ID (use 'jxie/flickr8k' for quick smoke test)")
    parser.add_argument("--smoke", action="store_true",
                        help="Use Flickr8k for a quick pipeline check")
    args = parser.parse_args()

    if args.smoke:
        dataset_id = "jxie/flickr8k"
        img_cache  = "smoke_image_embs.pt"
        txt_cache  = "smoke_text_embs.pt"
        IMG_EMB_PATH = img_cache
        TXT_EMB_PATH = txt_cache
    else:
        dataset_id = args.dataset

    processor, model = load_model()
    images_list, captions, t2i_gt, i2t_gt = load_flickr(dataset_id)
    img_embs, txt_embs = get_embeddings(processor, model, images_list, captions)

    # ── Free heavy objects no longer needed ─────────────────────────────────
    del images_list, captions, processor, model
    gc.collect()
    torch.cuda.empty_cache()

    sanity_check(img_embs, txt_embs)
    evaluate(img_embs, txt_embs, t2i_gt, i2t_gt)
