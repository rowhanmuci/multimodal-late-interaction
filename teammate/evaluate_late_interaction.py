"""
Step 2: Evaluate Late Interaction
Combines the global semantic representation (last non-padding token / EOS) with 
the token-level MaxSim similarity (Late Interaction) to improve retrieval performance.
"""

import os
import json
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
import gc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_EMB_PATH = "image_token_embs.pt"
TXT_EMB_PATH = "text_token_embs.pt"
RESULTS_PATH = "results_late_interaction.json"


def load_gt(dataset_id="nlphuji/flickr30k", split="test"):
    print(f"[*] Loading ground truths for {dataset_id} ({split})...")
    ds = load_dataset(dataset_id, split=split)
    
    if "split" in ds.column_names:
        ds = ds.filter(lambda x: x["split"] == "test")
        
    n = len(ds)
    # Flickr30k: 5 captions per image
    if "caption" in ds.column_names:
        captions = [ds[i]["caption"][j] for i in range(n) for j in range(5)]
    else:
        captions = [ds[i][f"caption_{j}"] for i in range(n) for j in range(5)]

    t2i_gt = [c // 5 for c in range(len(captions))]              # caption c → image c//5
    i2t_gt = [[5*i + j for j in range(5)] for i in range(n)]     # image i → 5 captions
    return t2i_gt, i2t_gt


def pad_tensors(tensor_list):
    """Pads a list of 2D tensors [seq, dim] to a single 3D tensor [N, max_seq, dim] and a mask [N, max_seq]."""
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
def compute_hybrid_scores(txt_list, img_list, chunk_size=100):
    """
    Computes both T2I and I2T similarity matrices for EOS and MaxSim.
    """
    n_txt = len(txt_list)
    n_img = len(img_list)

    print("[*] Extracting and computing EOS token similarities...")
    # The last token in the unpadded sequence tensor is the EOS token representation
    txt_eos = torch.stack([t[-1] for t in txt_list]).to(DEVICE)
    img_eos = torch.stack([img[-1] for img in img_list]).to(DEVICE)
    
    # L2-normalize EOS vectors
    txt_eos = F.normalize(txt_eos.float(), p=2, dim=-1)
    img_eos = F.normalize(img_eos.float(), p=2, dim=-1)
    
    eos_sim_t2i = (txt_eos @ img_eos.T).cpu()  # [n_txt, n_img]
    eos_sim_i2t = eos_sim_t2i.T                # [n_img, n_txt]
    
    del txt_eos, img_eos
    gc.collect()
    torch.cuda.empty_cache()

    print("[*] Computing token-level MaxSim scores...")
    maxsim_t2i = torch.zeros((n_txt, n_img), dtype=torch.float32)
    maxsim_i2t = torch.zeros((n_img, n_txt), dtype=torch.float32)

    # Pad images once to speed up GPU computation
    print("    Padding images (candidates)...")
    img_padded, img_mask = pad_tensors(img_list)
    img_padded = img_padded.to(DEVICE)  # [n_img, max_img_len, dim]
    img_mask = img_mask.to(DEVICE)      # [n_img, max_img_len]
    
    img_lens = img_mask.sum(dim=1).view(1, -1).float()  # [1, n_img]

    for start in tqdm(range(0, n_txt, chunk_size), desc="  Computing MaxSim"):
        end = min(start + chunk_size, n_txt)
        batch_txt = txt_list[start:end]
        
        # Pad texts in this chunk
        t_padded, t_mask = pad_tensors(batch_txt)
        t_padded = t_padded.to(DEVICE)  # [chunk, max_t_len, dim]
        t_mask = t_mask.to(DEVICE)      # [chunk, max_t_len]
        
        t_lens = t_mask.sum(dim=1).view(-1, 1).float()  # [chunk, 1]
        
        c_chunk_size = 100  # Chunk images to avoid VRAM OOM
        
        for c_start in range(0, n_img, c_chunk_size):
            c_end = min(c_start + c_chunk_size, n_img)
            c_chunk = img_padded[c_start:c_end]
            c_mask_chunk = img_mask[c_start:c_end]
            c_img_lens = img_lens[:, c_start:c_end].to(DEVICE)
            
            # Compute cosine similarities: [chunk, c_chunk, max_t_len, max_img_len]
            sim = torch.einsum('qid,cjd->qcij', t_padded, c_chunk)  
            
            # --- T2I MaxSim ---
            # Mask out padded image patches (-100) before taking max
            sim_for_t2i = sim.clone()
            sim_for_t2i.masked_fill_(~c_mask_chunk.view(1, c_chunk.shape[0], 1, c_chunk.shape[1]), -100.0)
            max_sim_img = sim_for_t2i.max(dim=3).values  # [chunk, c_chunk, max_t_len]
            
            # Mask out padded text tokens (0.0) before sum
            max_sim_img.masked_fill_(~t_mask.view(t_padded.shape[0], 1, t_padded.shape[1]), 0.0)
            norm_max_img = max_sim_img.sum(dim=2) / torch.clamp(t_lens, min=1.0)  # [chunk, c_chunk]
            maxsim_t2i[start:end, c_start:c_end] = norm_max_img.cpu()
            
            # --- I2T MaxSim ---
            # Mask out padded text tokens (-100) before taking max
            sim_for_i2t = sim.clone()
            sim_for_i2t.masked_fill_(~t_mask.view(t_padded.shape[0], 1, t_padded.shape[1], 1), -100.0)
            max_sim_txt = sim_for_i2t.max(dim=2).values  # [chunk, c_chunk, max_img_len]
            
            # Mask out padded image patches (0.0) before sum
            max_sim_txt.masked_fill_(~c_mask_chunk.view(1, c_chunk.shape[0], c_chunk.shape[1]), 0.0)
            norm_max_txt = max_sim_txt.sum(dim=2) / torch.clamp(c_img_lens, min=1.0)  # [chunk, c_chunk]
            maxsim_i2t[c_start:c_end, start:end] = norm_max_txt.T.cpu()
            
            del sim, sim_for_t2i, max_sim_img, norm_max_img, sim_for_i2t, max_sim_txt, norm_max_txt
            
    return eos_sim_t2i, eos_sim_i2t, maxsim_t2i, maxsim_i2t


def calc_metrics(scores, ground_truths, k_list=(1, 5, 10)):
    max_k = max(k_list)
    n_queries = scores.shape[0]
    recall_hits = {k: 0 for k in k_list}
    mrr_sum = 0.0

    # scores: [n_queries, n_cand]
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


if __name__ == "__main__":
    if not os.path.exists(IMG_EMB_PATH) or not os.path.exists(TXT_EMB_PATH):
        print(f"[!] Error: Missing .pt files. Run `python extract_embeddings.py` first.")
        import sys; sys.exit(1)

    print(f"[*] Loading embeddings from disk...")
    img_tokens = torch.load(IMG_EMB_PATH, weights_only=True)
    txt_tokens = torch.load(TXT_EMB_PATH, weights_only=True)
    print(f"    Images: {len(img_tokens)}")
    print(f"    Texts:  {len(txt_tokens)}")

    t2i_gt, i2t_gt = load_gt()

    # Compute raw similarity matrices
    eos_t2i, eos_i2t, maxsim_t2i, maxsim_i2t = compute_hybrid_scores(txt_tokens, img_tokens)

    alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    all_results = {}

    print("\n" + "="*50)
    print(f"{'Alpha':<6} | {'T2I R@1':<8} {'T2I R@5':<8} {'T2I R@10':<8} {'T2I MRR':<8} | {'I2T R@1':<8} {'I2T R@5':<8} {'I2T R@10':<8} {'I2T MRR':<8}")
    print("-" * 90)

    for alpha in alphas:
        # Combine
        scores_t2i = (1.0 - alpha) * eos_t2i + alpha * maxsim_t2i
        scores_i2t = (1.0 - alpha) * eos_i2t + alpha * maxsim_i2t

        t2i_r, t2i_mrr = calc_metrics(scores_t2i, t2i_gt)
        i2t_r, i2t_mrr = calc_metrics(scores_i2t, i2t_gt)

        alpha_str = f"{alpha:.1f}"
        print(f"{alpha_str:<6} | {t2i_r['R@1']:<8.2f} {t2i_r['R@5']:<8.2f} {t2i_r['R@10']:<8.2f} {t2i_mrr:<8.2f} | {i2t_r['R@1']:<8.2f} {i2t_r['R@5']:<8.2f} {i2t_r['R@10']:<8.2f} {i2t_mrr:<8.2f}")

        all_results[alpha_str] = {
            "T2I": {**t2i_r, "MRR": t2i_mrr},
            "I2T": {**i2t_r, "MRR": i2t_mrr}
        }

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[*] Results saved to {RESULTS_PATH}")

