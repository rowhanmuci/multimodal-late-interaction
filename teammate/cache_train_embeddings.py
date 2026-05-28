"""
Phase 1 (of B approach): Extract and cache 2048-dim token embeddings for the
train split of Flickr30k. Run this ONCE; then train_late_interaction.py can
load the cached embeddings and train only the projection layer in minutes.

Storage estimate (at max_pixels=128*28*28 ≈ 128 tokens/image):
  Images : 31014 × ~128 × 2048 × 2 bytes ≈ 16 GB
  Texts  : 155070 × ~30  × 2048 × 2 bytes ≈  19 GB
  Total  : ~35 GB  (ensure enough disk space)

Run:
    python cache_train_embeddings.py
    # or to limit resolution further:
    python cache_train_embeddings.py --max_pixels 65536   # ~84 tokens/image, ~10 GB
"""

import os
import argparse
import torch
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from datasets import load_dataset
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.bfloat16


def load_model(max_pixels, min_pixels):
    print(f"[*] Loading processor from {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.image_processor.max_pixels = max_pixels
    processor.image_processor.min_pixels = min_pixels
    approx_tokens = max_pixels // (28 * 28)
    print(f"    Image resolution: max_pixels={max_pixels} (~{approx_tokens} tokens/image max)")

    print(f"[*] Loading model ({DTYPE}) to {DEVICE} (frozen)...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        attn_implementation="sdpa",
    )
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    print(f"    VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return processor, model


def load_train_split():
    print("[*] Loading nlphuji/flickr30k — filtering to train split...")
    ds = load_dataset("nlphuji/flickr30k", split="test")   # all data is in HF 'test'
    if "split" in ds.column_names:
        ds = ds.filter(lambda x: x["split"] == "train")
    n = len(ds)
    print(f"    Train split: {n} images, {n*5} captions")

    print("[*] Pre-loading images into RAM...")
    images   = [img.convert("RGB") for img in tqdm(ds["image"], desc="  Images")]
    # Store ALL 5 captions per image so training can randomly pick one per step
    captions = [[ds[i]["caption"][j] for j in range(5)] for i in range(n)]
    return images, captions  # captions is List[List[str]] (n × 5)


@torch.no_grad()
def extract_text_tokens(model, processor, captions_2d, batch_size):
    """
    captions_2d: List[List[str]]  shape [n_images, 5]
    Returns: flat list of n_images*5 tensors, each [seq_len, 2048]
    Index mapping: caption[i*5 + j] belongs to image[i], caption #j
    """
    flat_captions = [cap for caps in captions_2d for cap in caps]

    formatted = []
    for cap in flat_captions:
        msg  = [{"role": "user", "content": cap}]
        formatted.append(
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        )

    print(f"[*] Extracting text embeddings ({len(formatted)} captions, batch={batch_size})...")
    all_tokens = []
    for start in tqdm(range(0, len(formatted), batch_size)):
        batch = formatted[start : start + batch_size]
        inputs  = processor(text=batch, return_tensors="pt", padding=True).to(DEVICE)
        outputs = model.model(**inputs)
        # Keep raw 2048-dim (NOT L2-normalized here; projection layer will handle it)
        hidden  = outputs.last_hidden_state.half()   # [B, seq_len, 2048]
        mask    = inputs["attention_mask"].bool()
        for i in range(hidden.shape[0]):
            all_tokens.append(hidden[i, mask[i]].cpu())   # [valid_len, 2048]
    return all_tokens   # length = n_images * 5


@torch.no_grad()
def extract_image_tokens(model, processor, images, batch_size):
    print(f"[*] Extracting image embeddings ({len(images)} images, batch={batch_size})...")
    all_tokens = []
    for start in tqdm(range(0, len(images), batch_size)):
        batch_imgs = images[start : start + batch_size]
        texts = []
        for img in batch_imgs:
            msg = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text":  "Represent this image."},
            ]}]
            texts.append(
                processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            )
        inputs = processor(
            text=texts, images=batch_imgs,
            return_tensors="pt", padding=True,
        ).to(DEVICE)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(DTYPE)

        outputs = model.model(**inputs)
        hidden  = outputs.last_hidden_state.half()
        mask    = inputs["attention_mask"].bool()
        for i in range(hidden.shape[0]):
            all_tokens.append(hidden[i, mask[i]].cpu())
    return all_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_pixels",   type=int, default=128*28*28,
                        help="Max image pixels (128*28*28≈128 tokens/image)")
    parser.add_argument("--min_pixels",   type=int, default=4*28*28)
    parser.add_argument("--img_batch",    type=int, default=16,
                        help="Batch size for image extraction (larger = faster)")
    parser.add_argument("--txt_batch",    type=int, default=64,
                        help="Batch size for text extraction")
    parser.add_argument("--output_dir",   type=str, default="embedding_cache",
                        help="Directory to save cached embeddings")
    args = parser.parse_args()

    img_path = os.path.join(args.output_dir, "train_img_embs.pt")
    txt_path = os.path.join(args.output_dir, "train_txt_embs.pt")
    os.makedirs(args.output_dir, exist_ok=True)

    # Skip if already cached
    if os.path.exists(img_path) and os.path.exists(txt_path):
        print(f"[!] Cache already exists at {args.output_dir}/")
        print("    Delete the .pt files to re-extract.")
        import sys; sys.exit(0)

    processor, model = load_model(args.max_pixels, args.min_pixels)
    images, captions_2d = load_train_split()

    # Extract texts first (faster, no vision encoder)
    txt_tokens = extract_text_tokens(model, processor, captions_2d, args.txt_batch)
    print(f"[*] Saving {len(txt_tokens)} text embeddings → {txt_path}")
    torch.save(txt_tokens, txt_path)

    # Extract images
    img_tokens = extract_image_tokens(model, processor, images, args.img_batch)
    print(f"[*] Saving {len(img_tokens)} image embeddings → {img_path}")
    torch.save(img_tokens, img_path)

    print("\n[*] Done! Next step: train the projection layer with cached embeddings:")
    print("    python train_late_interaction.py --use_cache embedding_cache")
