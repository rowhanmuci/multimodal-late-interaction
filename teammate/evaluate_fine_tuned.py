"""
Evaluate the projection-only Late Interaction model.
Loads the base model (fully frozen) and the trained projection layer,
extracts 128-dim token-level embeddings, and computes pure MaxSim metrics.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from datasets import load_dataset
from tqdm import tqdm
import gc

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID   = "Qwen/Qwen3-VL-Embedding-2B"
PROJ_DIR   = "late_interaction_proj"              # output_dir from train_late_interaction.py
PROJ_PATH  = os.path.join(PROJ_DIR, "projection_layer.pt")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE      = torch.bfloat16
PROJ_DIM   = 128

IMG_BATCH  = 4
TXT_BATCH  = 32
RESULTS_PATH = "results_fine_tuned.json"

# ── Model Loading ──────────────────────────────────────────────────────────────
def load_fine_tuned():
    if not os.path.exists(PROJ_PATH):
        print(f"[!] Error: Missing projection weights at {PROJ_PATH}. Run training first.")
        import sys; sys.exit(1)

    print(f"[*] Loading processor from {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print(f"[*] Loading base model ({DTYPE}) to {DEVICE} (frozen)...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        attn_implementation="sdpa",
    )
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    print(f"[*] Loading projection layer weights from {PROJ_PATH}...")
    hidden_size = getattr(model.config, "text_config").hidden_size \
                  if hasattr(model.config, "text_config") else model.config.hidden_size
    projection = nn.Linear(hidden_size, PROJ_DIM, bias=False).to(DEVICE)
    projection.load_state_dict(torch.load(PROJ_PATH, map_location=DEVICE))
    projection.eval()

    print(f"    Model loading complete. VRAM allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return processor, model, projection

# ── Dataset Loading ────────────────────────────────────────────────────────────
def load_flickr30k():
    print("[*] Loading nlphuji/flickr30k test split...")
    ds = load_dataset("nlphuji/flickr30k", split="test")

    if "split" in ds.column_names:
        full_size = len(ds)
        ds = ds.filter(lambda x: x["split"] == "test")
        print(f"    Filtered Karpathy test set: {len(ds)}/{full_size} images")

    n = len(ds)
    print(f"    {n} images loaded.")

    print("[*] Pre-loading images into RAM...")
    images_list = [img.convert("RGB") for img in tqdm(ds["image"], desc="  Loading")]

    captions = [ds[i]["caption"][j] for i in range(n) for j in range(5)]
    
    t2i_gt = [c // 5 for c in range(len(captions))]
    i2t_gt = [[5*i + j for j in range(5)] for i in range(n)]
    
    return images_list, captions, t2i_gt, i2t_gt

# ── Feature Extraction ─────────────────────────────────────────────────────────
@torch.no_grad()
def extract_text_tokens(model, projection, processor, captions):
    print("[*] Formatting captions...")
    formatted = []
    for cap in captions:
        msg = [{"role": "user", "content": cap}]
        formatted.append(
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        )

    print(f"[*] Extracting 128-dim text token embeddings ({len(captions)} captions, batch={TXT_BATCH})...")
    all_tokens = []

    for start in tqdm(range(0, len(formatted), TXT_BATCH)):
        batch_texts = formatted[start : start + TXT_BATCH]
        inputs = processor(text=batch_texts, return_tensors="pt", padding=True).to(DEVICE)
        outputs = model.model(**inputs)         # frozen backbone, plain call
        hidden = outputs.last_hidden_state      # [B, seq_len, 2048]

        projected = projection(hidden.float())  # [B, seq_len, 128]
        projected = F.normalize(projected, p=2, dim=-1).half()
        mask = inputs["attention_mask"].bool()

        for i in range(projected.shape[0]):
            all_tokens.append(projected[i, mask[i]].cpu())

    return all_tokens

@torch.no_grad()
def extract_image_tokens(model, projection, processor, images_list):
    n = len(images_list)
    print(f"[*] Extracting 128-dim image token embeddings ({n} images, batch={IMG_BATCH})...")
    all_tokens = []

    for start in tqdm(range(0, n, IMG_BATCH)):
        batch_images = images_list[start : start + IMG_BATCH]
        formatted_texts = []
        for img in batch_images:
            msg = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text":  "Represent this image."},
            ]}]
            formatted_texts.append(
                processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            )

        inputs = processor(
            text=formatted_texts, images=batch_images,
            return_tensors="pt", padding=True,
        ).to(DEVICE)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(DTYPE)

        outputs = model.model(**inputs)         # frozen backbone, plain call
        hidden = outputs.last_hidden_state      # [B, seq_len, 2048]

        projected = projection(hidden.float())  # [B, seq_len, 128]
        projected = F.normalize(projected, p=2, dim=-1).half()
        mask = inputs["attention_mask"].bool()

        for i in range(projected.shape[0]):
            all_tokens.append(projected[i, mask[i]].cpu())

    return all_tokens

# ── MaxSim Evaluator ───────────────────────────────────────────────────────────
def pad_tensors(tensor_list):
    max_len = max(t.shape[0] for t in tensor_list)
    dim = tensor_list[0].shape[1]
    N = len(tensor_list)
    
    padded = torch.zeros((N, max_len, dim), dtype=tensor_list[0].dtype)
    mask = torch.zeros((N, max_len), dtype=torch.bool)
    
    for i, t in enumerate(tensor_list):
        seq = t.shape[0]
        padded[i, :seq, :] = t
        mask[i, :seq] = True
        
    return padded, mask

@torch.no_grad()
def evaluate_pure_maxsim(txt_list, img_list, chunk_size=100):
    n_txt = len(txt_list)
    n_img = len(img_list)

    print("[*] Computing 128-dim pure MaxSim score matrices...")
    scores_t2i = torch.zeros((n_txt, n_img), dtype=torch.float32)
    scores_i2t = torch.zeros((n_img, n_txt), dtype=torch.float32)

    print("    Padding images (candidates)...")
    img_padded, img_mask = pad_tensors(img_list)
    img_padded = img_padded.to(DEVICE)  # [n_img, max_img_len, 128]
    img_mask = img_mask.to(DEVICE)      # [n_img, max_img_len]
    
    img_lens = img_mask.sum(dim=1).view(1, -1).float()  # [1, n_img]

    for start in tqdm(range(0, n_txt, chunk_size), desc="  Computing MaxSim"):
        end = min(start + chunk_size, n_txt)
        batch_txt = txt_list[start:end]
        
        t_padded, t_mask = pad_tensors(batch_txt)
        t_padded = t_padded.to(DEVICE)  # [chunk, max_t_len, 128]
        t_mask = t_mask.to(DEVICE)      # [chunk, max_t_len]
        
        t_lens = t_mask.sum(dim=1).view(-1, 1).float()  # [chunk, 1]
        
        c_chunk_size = 100
        
        for c_start in range(0, n_img, c_chunk_size):
            c_end = min(c_start + c_chunk_size, n_img)
            c_chunk = img_padded[c_start:c_end]
            c_mask_chunk = img_mask[c_start:c_end]
            c_img_lens = img_lens[:, c_start:c_end].to(DEVICE)
            
            # cosine similarity: [chunk, c_chunk, max_t_len, max_img_len]
            sim = torch.einsum('qid,cjd->qcij', t_padded, c_chunk)  
            
            # --- T2I MaxSim ---
            sim_for_t2i = sim.clone()
            sim_for_t2i.masked_fill_(~c_mask_chunk.view(1, c_chunk.shape[0], 1, c_chunk.shape[1]), -100.0)
            max_sim_img = sim_for_t2i.max(dim=3).values  # [chunk, c_chunk, max_t_len]
            max_sim_img.masked_fill_(~t_mask.view(t_padded.shape[0], 1, t_padded.shape[1]), 0.0)
            norm_max_img = max_sim_img.sum(dim=2) / torch.clamp(t_lens, min=1.0)
            scores_t2i[start:end, c_start:c_end] = norm_max_img.cpu()
            
            # --- I2T MaxSim ---
            sim_for_i2t = sim.clone()
            sim_for_i2t.masked_fill_(~t_mask.view(t_padded.shape[0], 1, t_padded.shape[1], 1), -100.0)
            max_sim_txt = sim_for_i2t.max(dim=2).values  # [chunk, c_chunk, max_img_len]
            max_sim_txt.masked_fill_(~c_mask_chunk.view(1, c_chunk.shape[0], c_chunk.shape[1]), 0.0)
            norm_max_txt = max_sim_txt.sum(dim=2) / torch.clamp(c_img_lens, min=1.0)
            scores_i2t[c_start:c_end, start:end] = norm_max_txt.T.cpu()
            
            del sim, sim_for_t2i, max_sim_img, norm_max_img, sim_for_i2t, max_sim_txt, norm_max_txt
            
    return scores_t2i, scores_i2t

def calc_metrics(scores, ground_truths, k_list=(1, 5, 10)):
    max_k = max(k_list)
    n_queries = scores.shape[0]
    recall_hits = {k: 0 for k in k_list}
    mrr_sum = 0.0

    topk_idx = torch.topk(scores, k=max_k, dim=1).indices.tolist()

    for i in range(n_queries):
        gt = ground_truths[i]
        gt_set = {gt} if isinstance(gt, int) else set(gt)

        for k in k_list:
            if any(idx in gt_set for idx in topk_idx[i][:k]):
                recall_hits[k] += 1

        gt_list = [gt] if isinstance(gt, int) else gt
        best_gt_score = scores[i, gt_list].max()
        rank = (scores[i] > best_gt_score).sum().item() + 1
        mrr_sum += 1.0 / rank

    recall = {f"R@{k}": round(v / n_queries * 100, 2) for k, v in recall_hits.items()}
    mrr = round(mrr_sum / n_queries * 100, 2)
    return recall, mrr

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    processor, model, projection = load_fine_tuned()
    images_list, captions, t2i_gt, i2t_gt = load_flickr30k()
    
    txt_tokens = extract_text_tokens(model, projection, processor, captions)
    img_tokens = extract_image_tokens(model, projection, processor, images_list)
    
    # Free heavy model objects before calculation to save GPU memory
    del processor, model, projection
    gc.collect()
    torch.cuda.empty_cache()
    
    scores_t2i, scores_i2t = evaluate_pure_maxsim(txt_tokens, img_tokens)
    
    results = {}

    print("\n=== Fine-Tuned T2I (Text -> Image) ===")
    t2i_r, t2i_mrr = calc_metrics(scores_t2i, t2i_gt)
    results["T2I"] = {**t2i_r, "MRR": t2i_mrr}
    for k, v in t2i_r.items():
        print(f"  {k}: {v:.2f}%")
    print(f"  MRR: {t2i_mrr:.2f}%")

    print("\n=== Fine-Tuned I2T (Image -> Text) ===")
    i2t_r, i2t_mrr = calc_metrics(scores_i2t, i2t_gt)
    results["I2T"] = {**i2t_r, "MRR": i2t_mrr}
    for k, v in i2t_r.items():
        print(f"  {k}: {v:.2f}%")
    print(f"  MRR: {i2t_mrr:.2f}%")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[*] Fine-tuned evaluation results saved to {RESULTS_PATH}")
