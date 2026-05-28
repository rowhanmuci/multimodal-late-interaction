"""
Step 1: Extract token-level embeddings for Late Interaction retrieval.
Simplified version: Pre-loads images and saves unpadded lists of tensors directly, 
since 1,000 images easily fit in RAM.
"""

import os
import torch
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from datasets import load_dataset
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID  = "Qwen/Qwen3-VL-Embedding-2B"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE     = torch.bfloat16

IMG_BATCH = 4
TXT_BATCH = 32

IMG_EMB_PATH = "image_token_embs.pt"
TXT_EMB_PATH = "text_token_embs.pt"

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
def load_flickr30k():
    print("[*] Loading nlphuji/flickr30k test split...")
    ds = load_dataset("nlphuji/flickr30k", split="test")

    if "split" in ds.column_names:
        full_size = len(ds)
        ds = ds.filter(lambda x: x["split"] == "test")
        print(f"    Filtered Karpathy test set: {len(ds)}/{full_size} images")

    n  = len(ds)
    print(f"    {n} images loaded.")

    print("[*] Pre-loading images into RAM...")
    images_list = [img.convert("RGB") for img in tqdm(ds["image"], desc="  Loading")]

    captions = [ds[i]["caption"][j] for i in range(n) for j in range(5)]
    return images_list, captions


# ── Extraction ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_text_tokens(model, processor, captions):
    print("[*] Formatting captions...")
    formatted = []
    for cap in captions:
        msg  = [{"role": "user", "content": cap}]
        text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        formatted.append(text)

    print(f"[*] Extracting text token embeddings ({len(captions)} captions, batch={TXT_BATCH})...")
    all_tokens = []

    for start in tqdm(range(0, len(formatted), TXT_BATCH)):
        batch_texts = formatted[start : start + TXT_BATCH]

        inputs  = processor(text=batch_texts, return_tensors="pt", padding=True).to(DEVICE)
        outputs = model.model(**inputs)
        hidden  = F.normalize(outputs.last_hidden_state.float(), p=2, dim=-1).half()
        mask    = inputs["attention_mask"].bool()

        # Unpad each sequence and store as a separate tensor
        for i in range(hidden.shape[0]):
            valid_len = mask[i].sum().item()
            embs = hidden[i, mask[i]].cpu()  # [seq_len, dim]
            all_tokens.append(embs)

    return all_tokens


@torch.no_grad()
def extract_image_tokens(model, processor, images_list):
    n = len(images_list)
    print(f"[*] Extracting image token embeddings ({n} images, batch={IMG_BATCH})...")
    all_tokens = []

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

        inputs  = processor(
            text=formatted_texts, images=batch_images,
            return_tensors="pt", padding=True,
        ).to(DEVICE)
        outputs = model.model(**inputs)
        hidden  = F.normalize(outputs.last_hidden_state.float(), p=2, dim=-1).half()
        mask    = inputs["attention_mask"].bool()

        # Unpad each sequence
        for i in range(hidden.shape[0]):
            valid_len = mask[i].sum().item()
            embs = hidden[i, mask[i]].cpu()  # [seq_len, dim]
            all_tokens.append(embs)

    return all_tokens


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if os.path.exists(IMG_EMB_PATH) and os.path.exists(TXT_EMB_PATH):
        print(f"[!] Embeddings already extracted ({IMG_EMB_PATH}, {TXT_EMB_PATH}).")
        print("[!] To re-extract, delete these .pt files first.")
        print("[!] To evaluate: python evaluate_late_interaction.py")
        import sys; sys.exit(0)

    processor, model = load_model()
    images_list, captions = load_flickr30k()

    txt_tokens = extract_text_tokens(model, processor, captions)
    img_tokens = extract_image_tokens(model, processor, images_list)

    print(f"[*] Saving {len(txt_tokens)} text embeddings to {TXT_EMB_PATH}")
    torch.save(txt_tokens, TXT_EMB_PATH)

    print(f"[*] Saving {len(img_tokens)} image embeddings to {IMG_EMB_PATH}")
    torch.save(img_tokens, IMG_EMB_PATH)

    print("\n[*] Done! Next step: python evaluate_late_interaction.py")
