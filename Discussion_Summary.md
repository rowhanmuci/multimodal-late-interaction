# 專題討論摘要

## 一、提案可行性確認

整體方向可行，核心概念清楚：從 single-vector embedding 轉為 Late Interaction 的細粒度比對。

**已確認的條件：**
- Qwen3-VL-Embedding-2B 可從 HuggingFace 載入，VRAM 約 4.26GB（RTX 5070 12GB 充裕）
- 模型架構（LLM-based embedding）可提取 token-level hidden states
- Flickr30k 規模適中，Test Set 1000 張圖 × 5 captions，適合在有限資源內完成實驗

---

## 二、Qwen3-VL-Embedding 特徵提取原理（已實作確認）

模型架構流程如下：

```
Image → Visual Encoder → Visual Tokens (patch tokens, ~192 個/張)
Text  → Tokenizer      → Text Tokens
              ↓
 [Visual Tokens + Text Tokens] → Qwen3 LLM → Last Hidden States [B, seq_len, 2048]
                                                    ↓
                              取最後一個非 padding token → single vector [B, 2048]
```

### 正確的程式碼寫法（實作驗證過）

**模型載入**：以下兩種寫法均可，但取 hidden states 的方式不同：

```python
# 寫法 A：AutoModel（載入 Qwen3VLModel，outputs[0] 直接是 hidden states）
from transformers import AutoModel, AutoProcessor
model   = AutoModel.from_pretrained("Qwen/Qwen3-VL-Embedding-2B", dtype=torch.bfloat16, device_map="cuda")
outputs = model(**inputs)
hidden  = outputs[0]                     # [B, seq_len, 2048] ← hidden states

# 寫法 B：Qwen3VLForConditionalGeneration（outputs[0] 是 logits，需呼叫 base model）
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
model   = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-Embedding-2B", torch_dtype=torch.bfloat16, device_map="cuda")
outputs = model.model(**inputs)          # 呼叫 base model (Qwen3VLModel)
hidden  = outputs.last_hidden_state      # [B, seq_len, 2048] ← hidden states
```

> `Qwen3VLForConditionalGeneration(**inputs)[0]` 回傳的是 **logits**（151936 維），不是 hidden states，這是兩種寫法最關鍵的差異。

**Pooling（Baseline 用）**：模型是 right-padded，不能用 `[:, -1, :]`，要用 attention_mask：
```python
last_idx   = inputs["attention_mask"].sum(dim=1) - 1   # [B]
single_vec = hidden[torch.arange(B), last_idx, :]      # [B, 2048]
single_vec = F.normalize(single_vec.float(), p=2, dim=-1)
```

**Late Interaction 用法（不做 pooling）**：
```python
all_token_embeddings = hidden   # [B, seq_len, 2048]，保留所有 tokens
# 之後用 attention_mask 區分哪些是真實 token，哪些是 padding
```

### 輸入格式

**文字**（5000 captions）：
```python
msg  = [{"role": "user", "content": caption}]
text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
inputs = processor(text=text, return_tensors="pt", padding=True)
# batch_size = 64
```

**圖片**（1000 images，必須帶文字 prompt）：
```python
msg = [{"role": "user", "content": [
    {"type": "image", "image": pil_img},
    {"type": "text",  "text":  "Represent this image."},
]}]
text   = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
inputs = processor(text=text, images=[pil_img], return_tensors="pt", padding=True)
# batch_size = 16（每張圖 ~192 visual tokens）
```

---

## 三、實作指南（Project_Guideline.md）的問題與修正

### Bug 1（Critical）：MRL 降維的用法不正確

**錯誤寫法：**
```python
reduced_states = hidden_states[:, :, :target_dim]  # 截斷 token-level hidden states
```

**問題：** MRL 截斷特性只對最終 pooled embedding 有效，不是 token-level hidden states。
另外，pooling 本身也有錯（見 Bug 4）。

**正確做法（Baseline 用）：**
```python
last_idx = mask.sum(dim=1) - 1
pooled   = hidden[torch.arange(B), last_idx, :]   # 先用正確方式 pool
reduced  = pooled[:, :target_dim]                  # 再截斷
```

> **補充**：Nemotron ColEmbed V2 論文實際上用的是 linear projection layer 降維，不是 MRL slicing。MRL slicing 只有在模型訓練時明確加入 Matryoshka loss 才有意義。

---

### Bug 2（Critical）：Late Interaction 沒有 mask padding tokens

**問題位置：** `compute_late_interaction_maxsim` 的 MaxSim 計算

**問題：** 計算 token-to-token 相似度矩陣時，padding tokens 會參與 max 操作，污染分數。

**修正方向：** 在 MaxSim 前，將 padding 位置的相似度設為 `-inf`，讓 max 自動忽略：
```python
# sim_matrix shape: [Q_len, D_len]
# 把 doc 的 padding 位置設為 -inf
padding_mask = (doc_attention_mask == 0)   # [D_len]
sim_matrix[:, padding_mask] = float('-inf')
max_sim = sim_matrix.max(dim=1).values     # [Q_len]
```

---

### Bug 3（已確認修正方向）：圖片特徵缺乏文字 prompt

**問題：** 只傳圖片進模型，沒有文字 prompt，visual tokens 的語意空間可能未對齊文字空間。

**確認可行的做法：** 加固定 prompt `"Represent this image."`（已在 baseline.py 驗證，self-similarity = 0.67）

---

### Bug 4（新增，Critical）：Baseline pooling 用 `[:, -1, :]` 是錯的

**問題：** Qwen3-VL 的 tokenizer 是 **right-padded**，batch 中短句的最後一個位置是 padding token，不是真實 token。

**錯誤寫法：**
```python
single_vec = outputs.last_hidden_state[:, -1, :]   # 拿到 padding token 的 embedding
```

**正確寫法：**
```python
last_idx   = inputs["attention_mask"].sum(dim=1) - 1
single_vec = hidden[torch.arange(B), last_idx, :]
```

---

### 重要釐清：AutoModel vs. Qwen3VLForConditionalGeneration 的差異

實測確認兩種寫法的行為**完全不同**：

| Class | `outputs[0]` shape | 是什麼 |
|-------|-------------------|--------|
| `AutoModel` → 載入 `Qwen3VLModel` | `[B, seq_len, 2048]` | hidden states ✓ |
| `Qwen3VLForConditionalGeneration` | `[B, seq_len, 151936]` | logits（vocab size）✗ |

**結論**：
- 組員 guideline 用 `AutoModel` + `outputs[0]` 取 hidden states，這件事本身**沒有錯**
- 真正的問題是 MRL 截斷位置（Bug 1）和 padding pooling（Bug 4）
- 我們的 `baseline.py` 用 `Qwen3VLForConditionalGeneration` + `model.model(**inputs).last_hidden_state`，同樣正確，只是路線不同

兩種可行寫法：
```python
# 寫法 A（AutoModel）
model   = AutoModel.from_pretrained(...)          # 載入 Qwen3VLModel
outputs = model(**inputs)
hidden  = outputs[0]                              # [B, seq_len, 2048] ← hidden states

# 寫法 B（baseline.py 採用）
model   = Qwen3VLForConditionalGeneration.from_pretrained(...)
outputs = model.model(**inputs)                   # 呼叫 base model
hidden  = outputs.last_hidden_state               # [B, seq_len, 2048] ← hidden states
```

---

### 其他小問題（已確認狀態）

| 問題 | 狀態 |
|------|------|
| `AutoModel` vs 特定 class | **已釐清**：AutoModel 載入 `Qwen3VLModel`，`outputs[0]` = hidden states，可用 |
| MRR 未實作 | **已在 baseline.py 實作** |
| Flickr30k ground truth | **已確認**：T2I ground truth = `c // 5`；I2T ground truth = `[5*i .. 5*i+4]` |
| Flickr8k 欄位格式 | `caption_0` ~ `caption_4`，與 Flickr30k 的 `caption` list 不同 |

---

### 可以直接用的部分（原始 guideline）

- `compute_late_interaction_maxsim` 的矩陣 broadcasting 維度邏輯正確（加上 mask 後可用）
- 三階段 pipeline 結構清楚，方便分模組除錯
- FP16 / BF16 + 結果移到 CPU 的記憶體管理策略是對的

---

## 四、公平比較的原則（重要）

### 核心原則：控制變因

Baseline vs. Proposed 的比較中，**只有 aggregation / scoring 策略不同，其他全部必須一致**。

### 必須完全一致的部分

| 項目 | 原因 |
|------|------|
| 模型 backbone | 用不同模型等於比較模型能力，不是比較方法 |
| 測試集的圖片和文字 | 不同 split 數字沒有可比性 |
| 圖片前處理（resize, normalize） | 輸入不同會影響 embedding 品質 |
| Ground truth 的定義 | 例如 T2I 的正確答案怎麼算 |
| 評估指標的實作 | Recall@K 的計算邏輯要一模一樣 |

### 可以不同的部分（這才是在比較的東西）

- **Baseline：** 所有 token embeddings → last token pooling → 單一向量 → cosine similarity
- **Proposed：** 保留所有 token embeddings → Late Interaction MaxSim（不做 pooling）

---

## 五、程式架構（已實作）

```
baseline.py
  → model.model(**inputs).last_hidden_state
  → last non-padding token pooling
  → L2 normalize
  → 存 image_embs.pt（1000×2048）、text_embs.pt（5000×2048）
  → cosine similarity matrix
  → Recall@K, MRR

late_interaction.py（待實作）
  → 讀同一份 image_embs.pt / text_embs.pt（確保公平）
  → 但需要 token-level features（不是 pooled），需另外存
  → MaxSim with padding mask
  → Recall@K, MRR（同一套計算邏輯）
```

> **注意**：Late Interaction 需要的是 **token-level embeddings**（不是 pooled），因此需要另外存一份 `image_token_embs.pt`（1000 × seq_len × 2048）和 `text_token_embs.pt`（5000 × seq_len × 2048）。

---

## 六、實驗執行進度

| 步驟 | 狀態 | 結果 |
|------|------|------|
| 1. Pipeline smoke test（Flickr8k） | **完成** | T2I R@1=77.9%, I2T R@1=89.6% |
| 2. Baseline — Flickr30k | **完成** | T2I R@1=48.53%, I2T R@1=67.43% |
| 3. Proposed — Late Interaction | 待執行 | — |
| 4. 消融實驗（embedding 維度） | 待執行 | — |
| 5. 質性分析（bad case） | 待執行 | — |
