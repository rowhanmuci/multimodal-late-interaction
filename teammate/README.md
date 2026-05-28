# Flickr30k 多模態檢索與延遲交互（Late Interaction）優化套件

本套件針對 `Qwen/Qwen3-VL-Embedding-2B` 模型，在 Flickr30k 測試集（1,000 張圖像，5,000 句文字說明）上評估雙塔基線（Baseline）與各種延遲交互（Late Interaction / MaxSim）優化排序演算法。

---

## 系統與環境要求

| 項目 | 規格 |
| :--- | :--- |
| **Python** | 3.10+ |
| **PyTorch** | 2.x (支援 Native BF16) |
| **GPU** | Nvidia RTX 5080 / 5070 (VRAM 需求約 6-8 GB) |
| **快取需求** | 需下載 `Qwen/Qwen3-VL-Embedding-2B` 與 `nlphuji/flickr30k` 資料集 |

### 安裝相依套件

```bash
pip install transformers accelerate datasets qwen-vl-utils tqdm torch
```

---

## 專案檔案架構

```
teammate_package/
├── baseline.py                    # 雙塔基線評估（單向量 EOS 餘弦相似度）
├── extract_embeddings.py          # 提取並儲存 unpadded token-level 隱藏狀態
├── evaluate_late_interaction.py  # 基礎延遲交互評估（掃描 alpha 參數混合比例）
├── evaluate_fine_tuned.py         # 微調後投影層延遲交互評估
│
├── cache_train_embeddings.py      # 快取訓練集隱藏狀態（加速投影層微調）
├── train_late_interaction.py      # 微調投影層（以 InfoNCE + MaxSim 損失訓練）
│
├── report.md                      # 完整評估報告（包含詳細數據分析與創新的演算法變體）
├── README.md                      # 本文件
│
├── results.json                   # 基線評估結果
├── results_late_interaction.json  # 基礎 Late Interaction 掃描結果
└── results_fine_tuned.json        # 微調投影層評估結果
```

在執行 `extract_embeddings.py` 後，會自動在本地快取以下 token-level 表示檔案以加速後續評估：
* `image_token_embs.pt`：影像 token 表徵（List of Tensors）
* `text_token_embs.pt`：文本 token 表徵（List of Tensors）

---

## 執行與評估方式

### 1. 執行雙塔基線 (Baseline)

```bash
# 正式評估（Flickr30k，~5 分鐘首次，之後快取讀取約 5 秒）
python baseline.py

# 快速驗收煙霧測試（Flickr8k smoke test，~2 分鐘）
python baseline.py --smoke
```

### 2. 提取 Token-level 嵌入 (評估 Late Interaction 前的必要步驟)

```bash
# 提取 unpadded 影像及文本 token 隱藏狀態（執行一次，約 10 分鐘）
python extract_embeddings.py
```

### 3. 執行延遲交互評估

```bash
# 基礎評估（掃描不同的 alpha 混合權重）
python evaluate_late_interaction.py
```

### 4. 執行微調與評估投影層

```bash
# Step 1: 快取訓練集 embeddings（跳過 VLM 前向傳播，縮短微調時間）
python cache_train_embeddings.py

# Step 2: 以快取的表示快速訓練投影層（約 30-60 分鐘）
python train_late_interaction.py --use_cache embedding_cache

# Step 3: 評估微調後的投影層模型（128 維度）
python evaluate_fine_tuned.py
```

---

## 核心評估結果對比

下表整理了從雙塔基線到各類延遲交互優化變體在 Flickr30k 上的 Recall@1 與 MRR 指標（詳細 Recall@5, R@10 及各種進階/創新演算法變體見 [`report.md`](file:///c:/Users/bojyu/Downloads/teammate_package/report.md)）：

| 排序演算法與配置 | T2I R@1 | T2I MRR | I2T R@1 | I2T MRR | 核心改善機制 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **雙塔基線 (Pure EOS)** | 81.86% | 88.11% | 94.00% | 96.37% | 全局語義向量內積 |
| **純延遲交互 (Pure MaxSim)** | 80.68% | 87.29% | 78.40% | 85.78% | Token 級最大相似度均值 |
| **混合延遲交互 (Hybrid MaxSim, 0.5)** | 83.18% | 89.04% | 94.80% | 96.82% | 全局 (EOS) 與局部 (MaxSim) 融合 |
| **去模板切片混合 (Sliced Hybrid, 0.5)** | 83.66% | 89.37% | 94.30% | 96.47% | 切除 System/User 模板與結束標記 |
| **雙向切片純對齊 (Bi-directional Sliced Pure)** | 72.34% | 80.76% | 86.00% | 91.25% | 限制雙向 Token 均須匹配，大幅修正 I2T 偏差 |
| **IDF 加權切片混合 (IDF-Weighted Sliced Hybrid, 0.5)** | **84.12%** | **89.61%** | 94.30% | 96.47% | **引入詞頻 IDF 權重（T2I R@1 歷史新高）** |
| **Top-2 切片混合 (Top-2 Sliced Hybrid, 0.5)** | 83.86% | 89.43% | 94.50% | 96.57% | 平均 Top-2 相似標記，提升局部容錯率 |
| **Top-3 切片混合 (Top-3 Sliced Hybrid, 0.5)** | 83.52% | 89.25% | **94.60%** | **96.65%** | 平均 Top-3 相似標記（I2T R@1 創新高） |
| **投影層微調模型 (Fine-tuned LI, 128d)** | 75.52% | 83.40% | 82.70% | 88.11% | 128 維維度壓縮投影，適合輕量化與高速檢索部署 |

---

## 關鍵技術細節與實作

1. **多模態隱藏狀態安全提取**：
   我們透過直接呼叫 `model.model(**inputs)`（而非會計算 LM Head 輸出 Vocab Logits 的 `model(**inputs)`）來安全且高效地取得 `last_hidden_state` 嵌入向量（維度 2048）。
2. **正確的 Attention Pooling（基線用）**：
   由於影像/文字在 batch 推理時使用右側填充（Right-Padding），我們透過 `attention_mask` 計算出每個序列最後一個有效 Token 的索引來拿取 EOS 表示，避開 Padding 標記：
   ```python
   last_idx = inputs["attention_mask"].sum(dim=1) - 1
   embs = hidden[torch.arange(B), last_idx]
   ```
3. **去模板切片定位**：
   為了解除系統提示與包裝格式帶來的相似度偏置，我們對 Token 進行了精確的邊界切片：
   * 文本端核心：`tokens[14:-3]` (排除 `<|im_start|>system\nRepresent the user's input...` 等 14 個 prefix token 以及末尾 3 個 suffix token)
   * 影像端核心：`tokens[15:-8]` (僅保留 `<|image_pad|>` 對應的視覺特徵向量)

詳細技術細節、公式推導與微調細節，請參閱 [`report.md`](file:///c:/Users/bojyu/Downloads/teammate_package/report.md)。
