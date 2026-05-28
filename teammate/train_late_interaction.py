"""
Fine-tune a Linear projection layer for Late Interaction (ColBERT style).
The Qwen3-VL backbone is FULLY FROZEN. Only a small Linear(2048->128) layer is trained.
This avoids storing intermediate activations for 28 Transformer layers,
making training ~5-10x faster compared to LoRA-based approach.
"""

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from datasets import load_dataset
from tqdm import tqdm
import argparse

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# ── Projection Layer (the only thing we train) ─────────────────────────────────
class ProjectionLayer(nn.Module):
    """
    A single linear layer that maps Qwen3-VL's 2048-dim token embeddings to a
    compact 128-dim space optimized for MaxSim late interaction retrieval.

    The backbone is frozen and run under torch.no_grad(). Only this layer
    receives gradients, making training much faster and memory-efficient.
    """
    def __init__(self, hidden_size=2048, projection_dim=128):
        super().__init__()
        self.linear = nn.Linear(hidden_size, projection_dim, bias=False)
        # Orthogonal init for stable initial representations
        nn.init.orthogonal_(self.linear.weight)

    def forward(self, hidden_states):
        # hidden_states: [B, seq_len, 2048]
        projected = self.linear(hidden_states.float())   # [B, seq_len, proj_dim]
        return F.normalize(projected, p=2, dim=-1)       # L2 normalize per token

# ── Cached Embedding Dataset ───────────────────────────────────────────────────
class CachedEmbeddingDataset(Dataset):
    """
    Loads pre-computed 2048-dim token embeddings from disk (from cache_train_embeddings.py).
    Each item: (img_emb [seq_len, 2048], txt_emb [seq_len, 2048])
    For each image we randomly pick 1 of 5 captions (data augmentation).
    """
    def __init__(self, img_embs, txt_embs):
        # img_embs: List[Tensor[seq_len, 2048]], length = n_images
        # txt_embs: List[Tensor[seq_len, 2048]], length = n_images * 5
        self.img_embs = img_embs
        self.txt_embs = txt_embs   # flat: index i*5+j = image i, caption j
        self.n = len(img_embs)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img_emb = self.img_embs[idx]                        # [img_len, 2048]
        cap_idx = random.randint(0, 4)                      # random of 5 captions
        txt_emb = self.txt_embs[idx * 5 + cap_idx]         # [txt_len, 2048]
        return img_emb, txt_emb


def cached_collate(batch):
    """Pad a batch of (img_emb, txt_emb) pairs for DataLoader."""
    img_list, txt_list = zip(*batch)
    max_i = max(t.shape[0] for t in img_list)
    max_t = max(t.shape[0] for t in txt_list)
    dim = img_list[0].shape[1]
    B = len(img_list)

    img_pad  = torch.zeros(B, max_i, dim, dtype=img_list[0].dtype)
    img_mask = torch.zeros(B, max_i, dtype=torch.bool)
    txt_pad  = torch.zeros(B, max_t, dim, dtype=txt_list[0].dtype)
    txt_mask = torch.zeros(B, max_t, dtype=torch.bool)

    for i, (img, txt) in enumerate(zip(img_list, txt_list)):
        img_pad[i, :img.shape[0]]  = img;  img_mask[i, :img.shape[0]]  = True
        txt_pad[i, :txt.shape[0]]  = txt;  txt_mask[i, :txt.shape[0]]  = True

    return img_pad, img_mask, txt_pad, txt_mask

def compute_maxsim_loss(txt_embs, txt_mask, img_embs, img_mask, temperature=0.1):
    """
    Bidirectional InfoNCE loss using MaxSim late interaction scoring.
    txt_embs: [B, T, dim]    txt_mask: [B, T]
    img_embs: [B, I, dim]    img_mask: [B, I]
    """
    B = txt_embs.shape[0]

    # Pairwise cosine similarities: [B_query, B_candidate, T, I]
    sim = torch.einsum('qid,cjd->qcij', txt_embs.float(), img_embs.float())

    # --- Text-to-Image MaxSim ---
    # Mask padded image tokens before max
    sim_t2i = sim.masked_fill(~img_mask.view(1, B, 1, -1), -1e4)
    max_over_img = sim_t2i.max(dim=3).values          # [B, B, T]
    max_over_img.masked_fill_(~txt_mask.view(B, 1, -1), 0.0)
    txt_lens = txt_mask.float().sum(dim=1, keepdim=True)
    scores_t2i = max_over_img.sum(dim=2) / txt_lens.clamp(min=1.0)  # [B, B]

    # --- Image-to-Text MaxSim ---
    sim_i2t = sim.masked_fill(~txt_mask.view(B, 1, -1, 1), -1e4)
    max_over_txt = sim_i2t.max(dim=2).values          # [B, B, I]
    max_over_txt.masked_fill_(~img_mask.view(1, B, -1), 0.0)
    img_lens = img_mask.float().sum(dim=1)             # [B]
    scores_i2t = max_over_txt.sum(dim=2) / img_lens.view(1, B).clamp(min=1.0)  # [B, B]
    scores_i2t = scores_i2t.T                          # [B, B]

    targets = torch.arange(B, device=txt_embs.device)
    loss_t2i = F.cross_entropy(scores_t2i / temperature, targets)
    loss_i2t = F.cross_entropy(scores_i2t / temperature, targets)
    return (loss_t2i + loss_i2t) / 2.0

# ── Collate Function ───────────────────────────────────────────────────────────
def create_collate_fn(processor):
    def collate_fn(batch):
        images, captions = [], []
        for item in batch:
            images.append(item["image"].convert("RGB"))
            cap = item["caption"]
            captions.append(random.choice(cap) if isinstance(cap, list) else str(cap))

        # Format text queries
        formatted_caps = []
        for cap in captions:
            msg = [{"role": "user", "content": cap}]
            formatted_caps.append(
                processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            )

        # Format image queries
        formatted_imgs = []
        for img in images:
            msg = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text":  "Represent this image."},
            ]}]
            formatted_imgs.append(
                processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            )

        return {
            "images": images,
            "formatted_caps": formatted_caps,
            "formatted_imgs": formatted_imgs,
        }
    return collate_fn

# ── Extract Token Embeddings (backbone frozen, no grad) ────────────────────────
@torch.no_grad()
def extract_embeddings(model, processor, formatted_texts, images, device, dtype):
    """
    Run a frozen forward pass through Qwen3-VL to extract token-level hidden states.
    Returns (hidden_states [B, seq_len, 2048], attention_mask [B, seq_len]).
    """
    inputs = processor(
        text=formatted_texts,
        images=images if images is not None else None,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}
    # Cast pixel values to model dtype
    if "pixel_values" in inputs and inputs["pixel_values"] is not None:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

    outputs = model.model(**{k: v for k, v in inputs.items()
                             if k not in ("labels",)})
    hidden = outputs.last_hidden_state                 # [B, seq_len, 2048]
    mask   = inputs["attention_mask"].bool()           # [B, seq_len]
    return hidden.to(dtype), mask

# ── Training Script ─────────────────────────────────────────────────────────────
def train(args):
    # ── Fast path: train from pre-cached embeddings (no backbone needed) ──────
    if args.use_cache:
        cache_dir = args.use_cache
        img_path  = os.path.join(cache_dir, "train_img_embs.pt")
        txt_path  = os.path.join(cache_dir, "train_txt_embs.pt")
        if not os.path.exists(img_path) or not os.path.exists(txt_path):
            print(f"[!] Cache not found at {cache_dir}/. Run cache_train_embeddings.py first.")
            import sys; sys.exit(1)

        print(f"[*] Loading cached embeddings from {cache_dir}/...")
        img_embs = torch.load(img_path, map_location="cpu")
        txt_embs = torch.load(txt_path, map_location="cpu")
        print(f"    {len(img_embs)} image embeddings, {len(txt_embs)} text embeddings")
        print(f"    Hidden dim: {img_embs[0].shape[-1]}")

        hidden_size = img_embs[0].shape[-1]   # should be 2048
        print(f"[*] Initializing projection layer: Linear({hidden_size} → {args.proj_dim})")
        projection = ProjectionLayer(hidden_size=hidden_size, projection_dim=args.proj_dim).to(DEVICE)
        print(f"    Trainable parameters: {sum(p.numel() for p in projection.parameters()):,}")

        optimizer = torch.optim.AdamW(projection.parameters(), lr=args.lr, weight_decay=0.01)

        dataset = CachedEmbeddingDataset(img_embs, txt_embs)
        loader  = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=cached_collate,
            num_workers=0,
        )

        print(f"\n[*] Starting FAST training from cache (epochs={args.epochs})...")
        print(f"    {len(dataset)} training pairs | batch={args.batch_size} | temp={args.temp}")

        for epoch in range(args.epochs):
            projection.train()
            epoch_loss, steps_done = 0.0, 0
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
            for batch_idx, (img_pad, img_mask, txt_pad, txt_mask) in enumerate(pbar):
                if args.max_steps > 0 and batch_idx >= args.max_steps:
                    print(f"[*] Reached max_steps ({args.max_steps}).")
                    break

                img_pad  = img_pad.to(DEVICE)
                img_mask = img_mask.to(DEVICE)
                txt_pad  = txt_pad.to(DEVICE)
                txt_mask = txt_mask.to(DEVICE)

                # Projection only (no backbone)
                proj_img = projection(img_pad)   # [B, I, 128]
                proj_txt = projection(txt_pad)   # [B, T, 128]

                loss = compute_maxsim_loss(
                    proj_txt, txt_mask,
                    proj_img, img_mask,
                    temperature=args.temp,
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(projection.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                steps_done += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}",
                                   "avg":  f"{epoch_loss/steps_done:.4f}"})

            print(f"[*] Epoch {epoch+1} avg loss: {epoch_loss/max(steps_done,1):.4f}")

        os.makedirs(args.output_dir, exist_ok=True)
        proj_path = os.path.join(args.output_dir, "projection_layer.pt")
        torch.save(projection.state_dict(), proj_path)
        print(f"\n[*] Saved → {proj_path}")
        print("[*] Training completed!")
        return

    # ── Slow path: run backbone live (original approach) ─────────────────────

    print(f"[*] Loading processor and model from {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    # Limit image resolution to reduce token count per image.
    # Qwen3-VL uses one token per 28x28 patch, so:
    #   max_pixels = 128 * 28 * 28 = 100352  →  max ~128 tokens/image
    # This reduces Attention FLOPs by ~60x vs full resolution (~1000 tokens/image).
    processor.image_processor.min_pixels = args.min_pixels
    processor.image_processor.max_pixels = args.max_pixels
    print(f"    Image resolution limited: max_pixels={args.max_pixels} "
          f"(~{args.max_pixels // (28*28)} tokens/image max)")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        attn_implementation="sdpa",
    )

    # Freeze the entire backbone — no gradients needed here
    print("[*] Freezing backbone (0 backbone parameters will be updated)...")
    for param in model.parameters():
        param.requires_grad = False
    model.eval()  # Disables dropout in backbone

    # 1. Initialize the projection layer (the ONLY thing we train)
    hidden_size = getattr(model.config, "text_config").hidden_size \
                  if hasattr(model.config, "text_config") else model.config.hidden_size
    print(f"[*] Initializing projection layer: Linear({hidden_size} → {args.proj_dim})")
    projection = ProjectionLayer(hidden_size=hidden_size, projection_dim=args.proj_dim).to(DEVICE)
    print(f"    Trainable parameters: {sum(p.numel() for p in projection.parameters()):,}")

    # 2. Optimizer (only for projection layer)
    optimizer = torch.optim.AdamW(projection.parameters(), lr=args.lr, weight_decay=0.01)

    # 3. Load Dataset
    print(f"[*] Loading nlphuji/flickr30k dataset (logical split: {args.split})...")
    dataset = load_dataset("nlphuji/flickr30k", split="test")
    if "split" in dataset.column_names:
        dataset = dataset.filter(lambda x: x["split"] == args.split)

    if args.subset_size > 0:
        print(f"    Subsetting dataset to {args.subset_size} samples...")
        dataset = dataset.select(range(args.subset_size))

    collate_fn = create_collate_fn(processor)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # 4. Training Loop
    print(f"\n[*] Starting training (epochs={args.epochs}, max_steps={args.max_steps})...")
    print(f"    Effective batch size (negatives per query): {args.batch_size - 1}")
    print(f"    Temperature: {args.temp}")

    for epoch in range(args.epochs):
        projection.train()
        epoch_loss = 0.0
        steps_done = 0

        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            if args.max_steps > 0 and batch_idx >= args.max_steps:
                print(f"[*] Reached max_steps ({args.max_steps}). Stopping training.")
                break

            # --- Backbone forward (fully frozen, no grad, fast) ---
            txt_hidden, txt_mask = extract_embeddings(
                model, processor,
                batch["formatted_caps"], None, DEVICE, DTYPE
            )
            img_hidden, img_mask = extract_embeddings(
                model, processor,
                batch["formatted_imgs"], batch["images"], DEVICE, DTYPE
            )

            # --- Projection (trainable, computes gradients) ---
            projected_txt = projection(txt_hidden)   # [B, T, 128]
            projected_img = projection(img_hidden)   # [B, I, 128]

            # --- MaxSim InfoNCE Loss ---
            loss = compute_maxsim_loss(
                projected_txt, txt_mask,
                projected_img, img_mask,
                temperature=args.temp,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projection.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            steps_done += 1
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{epoch_loss / steps_done:.4f}"
            })

            del txt_hidden, img_hidden, projected_txt, projected_img, txt_mask, img_mask, loss

        print(f"[*] Epoch {epoch+1} finished. Average Loss: {epoch_loss / max(steps_done, 1):.4f}")

    # 5. Save Projection Layer Weights
    os.makedirs(args.output_dir, exist_ok=True)
    proj_path = os.path.join(args.output_dir, "projection_layer.pt")
    torch.save(projection.state_dict(), proj_path)
    print(f"\n[*] Saved projection layer weights to: {proj_path}")
    print("[*] Training completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",       type=int,   default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size",   type=int,   default=32,
                        help="Batch size (= number of negatives + 1)")
    parser.add_argument("--lr",           type=float, default=1e-3,
                        help="Learning rate for projection layer")
    parser.add_argument("--proj_dim",     type=int,   default=128,
                        help="Output dimension of projection layer")
    parser.add_argument("--temp",         type=float, default=0.1,
                        help="InfoNCE temperature")
    parser.add_argument("--split",        type=str,   default="train",
                        help="Dataset split (for live backbone mode)")
    parser.add_argument("--subset_size",  type=int,   default=-1,
                        help="Subset size (-1 = full dataset, live backbone mode only)")
    parser.add_argument("--max_steps",    type=int,   default=-1,
                        help="Max steps per epoch (-1 = full epoch)")
    parser.add_argument("--output_dir",   type=str,   default="late_interaction_proj",
                        help="Directory to save weights")
    # Image resolution control (live backbone mode only)
    parser.add_argument("--min_pixels",   type=int,   default=4*28*28)
    parser.add_argument("--max_pixels",   type=int,   default=128*28*28,
                        help="Max image pixels (128*28*28 ≈ 128 tokens/image)")
    # Fast cached mode
    parser.add_argument("--use_cache",    type=str,   default=None,
                        help="Path to embedding cache dir (from cache_train_embeddings.py). "
                             "If set, skips backbone entirely and trains projection layer only "
                             "from pre-computed embeddings — trains in minutes, not hours.")
    args = parser.parse_args()
    train(args)
