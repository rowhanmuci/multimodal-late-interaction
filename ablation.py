"""
Dimension ablation: Recall@K at 2048 / 1024 / 512 / 256 dimensions.

Requires:
  image_embs.pt, text_embs.pt          (from baseline.py)
Optional:
  image_token_embs.pt, text_token_embs.pt,
  image_token_mask.pt, text_token_mask.pt  (from late_interaction.py --extract)
"""

import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIMS   = [2048, 1024, 512, 256]

t2i_gt = [c // 5 for c in range(5000)]
i2t_gt = [[5 * i + j for j in range(5)] for i in range(1000)]


# ── Metrics ────────────────────────────────────────────────────────────────────
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


def eval_pair(t2i_scores, i2t_scores):
    return {
        "T2I": {**recall_at_k(t2i_scores, t2i_gt),
                "MRR": mean_reciprocal_rank(t2i_scores, t2i_gt)},
        "I2T": {**recall_at_k(i2t_scores, i2t_gt),
                "MRR": mean_reciprocal_rank(i2t_scores, i2t_gt)},
    }


# ── Baseline ablation ──────────────────────────────────────────────────────────
N_EVAL = 1000   # use same 1K subset as late_interaction.py for fair comparison

def run_baseline_ablation():
    print("[*] Loading baseline embeddings (first 1K images)...")
    img_full = torch.load("image_embs.pt", weights_only=True)[:N_EVAL]          # [1000, 2048] f32
    txt_full = torch.load("text_embs.pt",  weights_only=True)[:N_EVAL * 5]      # [5000, 2048] f32

    results = {}
    for dim in DIMS:
        img = F.normalize(img_full[:, :dim].float(), p=2, dim=-1)
        txt = F.normalize(txt_full[:, :dim].float(), p=2, dim=-1)
        results[str(dim)] = eval_pair(txt @ img.T, img @ txt.T)
        r = results[str(dim)]
        print(f"  {dim:4d}d  T2I R@1={r['T2I']['R@1']:.2f}%  I2T R@1={r['I2T']['R@1']:.2f}%")
    return results


# ── Late Interaction ablation ──────────────────────────────────────────────────
@torch.no_grad()
def _maxsim(query_embs, query_masks, doc_embs, doc_masks, desc):
    Q, q_max, dim = query_embs.shape
    D, d_max, _   = doc_embs.shape

    score_matrix = torch.zeros(Q, D, dtype=torch.float32)
    doc_gpu      = doc_embs.to(DEVICE)
    mask_gpu     = doc_masks.to(DEVICE)
    docs_flat    = doc_gpu.view(D * d_max, dim)
    pad_flat     = (~mask_gpu).view(D * d_max)

    for i in tqdm(range(Q), desc=desc, leave=False):
        q_valid = query_embs[i][query_masks[i]].to(DEVICE)
        if q_valid.shape[0] == 0:
            continue
        sim = q_valid.float() @ docs_flat.T.float()
        sim[:, pad_flat] = float("-inf")
        sim = sim.view(q_valid.shape[0], D, d_max)
        score_matrix[i] = sim.max(dim=-1).values.sum(dim=0).cpu()

    return score_matrix


def run_li_ablation():
    paths = ["image_token_embs.pt", "text_token_embs.pt",
             "image_token_mask.pt", "text_token_mask.pt"]
    if not all(os.path.exists(p) for p in paths):
        print("[!] Token embeddings not found — skipping LI ablation.")
        print("    Run `python late_interaction.py --extract` first.")
        return None

    print("[*] Loading token-level embeddings...")
    img_tok  = torch.load("image_token_embs.pt", weights_only=True)  # [1000, L_i, 2048] f16
    txt_tok  = torch.load("text_token_embs.pt",  weights_only=True)  # [5000, L_t, 2048] f16
    img_mask = torch.load("image_token_mask.pt", weights_only=True)
    txt_mask = torch.load("text_token_mask.pt",  weights_only=True)

    results = {}
    for dim in DIMS:
        # Slice last dimension, re-normalize per token
        img_d = F.normalize(img_tok[..., :dim].float(), p=2, dim=-1).half()
        txt_d = F.normalize(txt_tok[..., :dim].float(), p=2, dim=-1).half()

        t2i = _maxsim(txt_d, txt_mask, img_d, img_mask, f"T2I d={dim}")
        i2t = _maxsim(img_d, img_mask, txt_d, txt_mask, f"I2T d={dim}")

        results[str(dim)] = eval_pair(t2i, i2t)
        r = results[str(dim)]
        print(f"  {dim:4d}d  T2I R@1={r['T2I']['R@1']:.2f}%  I2T R@1={r['I2T']['R@1']:.2f}%")
    return results


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    out = {}

    print("\n=== Baseline Dimension Ablation ===")
    out["baseline"] = run_baseline_ablation()

    print("\n=== Late Interaction Dimension Ablation ===")
    li = run_li_ablation()
    if li:
        out["late_interaction"] = li

    with open("ablation_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n[*] Saved ablation_results.json")
