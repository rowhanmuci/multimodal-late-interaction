# 📄 Final Project Proposal  

## 📌 題目  
**基於輕量級 Late Interaction 的多模態檢索效能提升研究**

---

## 📌 研究動機與背景  
近年來，多模態模型（Multimodal Models）快速發展，例如 CLIP 與 Qwen3-VL-Embedding，能夠將文字與圖像映射至同一語意空間，使跨模態檢索（cross-modal retrieval）成為可能。

然而，現有方法多採用「單一向量表示（single-vector embedding）」來描述整體內容，這會導致細節資訊在壓縮過程中流失，進而影響檢索準確率。

為了解決此問題，近期研究如 Nemotron ColEmbed V2 提出「Late Interaction」機制，透過 token-level 或 patch-level 的細粒度比對，提升語意匹配能力。

---

## 📌 研究目標  
本專題旨在探討：

1. 單一向量表示與 Late Interaction 方法在多模態檢索上的差異  
2. 在不重新訓練大型模型的前提下，是否能透過輕量化改進提升檢索效能  
3. Late Interaction 對於不同查詢情境（簡單 vs 複雜語意）的影響  

---

## 📌 方法概述  

本研究將基於現有預訓練模型，設計以下兩種方法進行比較：

### 🔹 Baseline（基準方法）
- 使用 Qwen3-VL-Embedding 或 CLIP  
- 將圖像與文字各自轉為單一 embedding 向量  
- 使用 cosine similarity 進行檢索  

---

### 🔹 Proposed Method（改進方法）
- 保留文字的 token-level embeddings  
- 保留圖像的 patch-level embeddings（或中間層特徵）  
- 採用 Late Interaction 機制計算相似度，例如：
  - Max similarity pooling  
  - Average pooling  

👉 核心概念為：  
由「vector-to-vector」比較，轉為「token-to-token」細粒度匹配

---

## 📌 實驗設計  

### 📊 Dataset（擇一使用）
- MSCOCO（image-text pairing）  
- Flickr30k  

---

### 📏 評估指標
- Recall@K（K=1,5,10）  
- Mean Reciprocal Rank (MRR)  

---

### 📈 比較項目
1. Baseline（single embedding）  
2. Late Interaction 方法  
3. 不同 pooling 策略之影響  

---

## 📌 預期成果  
本研究預期：

- Late Interaction 方法在 Recall@K 指標上優於 baseline  
- 能觀察到在語意較複雜的查詢中，改進效果更為顯著  
- 證明即使不重新訓練模型，也能透過檢索策略提升性能  

---

## 📌 相關文獻  

1. CLIP  
   https://arxiv.org/abs/2103.00020 
2. Qwen3-VL-Embedding  
   https://arxiv.org/abs/2511.21631
3. ColBERT 
   https://arxiv.org/abs/2004.12832 

---

## 📌 專題貢獻  
本專題的貢獻包括：

- 提出一個**不需重新訓練模型的輕量化改進方法**  
- 實證 Late Interaction 在多模態檢索上的有效性  
- 分析不同表示方式對語意匹配的影響  

---

## 📌 補充說明（對應作業要求）  

1. 本題目屬於多模態機器學習與資訊檢索領域，符合課程範圍  
2. 將於期末報告中清楚說明：
   - embedding 與 Late Interaction 原理  
   - 系統流程與設計動機  
3. 將展示實驗結果（Recall@K 等指標）與實際檢索範例  