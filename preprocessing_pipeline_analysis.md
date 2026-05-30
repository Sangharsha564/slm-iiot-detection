# Preprocessing Pipeline Analysis
## SLM-Based Slow Rate DoS Detection — CIC IIoT 2025 Dataset
**Date:** May 2026 | **Author:** Sobit Thapa

---

## 1. Introduction

This document analyses preprocessing techniques from two source groups:
1. **Papers in our research folders** (Folders 1–4 + Supervisor Shared) — 46 papers reviewed
2. **Additional web-searched papers** (ArXiv, IEEE, Springer, 2024–2026)

The goal is to identify which techniques consistently improve detection performance and recommend the optimal preprocessing pipeline for our system — an SLM-based Slowloris detection system using the CIC IIoT 2025 dataset on resource-constrained IIoT devices.

---

## 2. What Our Papers Say About Preprocessing

### 2.1 Class Imbalance Handling

#### From Folder 1 (Slow-Rate Detection Papers)

**Paper 4 — Al-Shukaili et al. (Discover IoT, 2025)**
> "SMOTE applied on training set only. Wrapper-based RFE (50 features) achieved F1=95.45% on CIC-IDS2017. Key finding: SMOTE alone is insufficient without feature selection — combination is essential."

**Paper 8 — Ramirez-Martinez et al. (Computer Networks, 2025)**
> "Z-score normalization on 12 P4-extracted features. 80/20 train/test, 20 runs × 3 folds averaged. No oversampling — balanced via weighted loss. Achieved Slowloris F1=97.21% on CICIoT2023 (which had 41,437 Slowloris samples — far more than our 132)."

**Paper 6 — Bocu & Iavich (JNCA, 2024)**
> "Sliding window (30s) feature extraction. L2 normalization. No oversampling — 150M vs 150M balanced dataset naturally. Federated Learning across 10 nodes. Achieved 98.79% accuracy."

**Paper 2 — Gogoi & Ahmed (IEEE INDICON, 2022)**
> "Synthetic data generated mimicking CIC DoS 2017 distribution. Only 8 real attack traces — heavily dependent on synthetic generation. This is the weakest approach and explains their reliance on synthetic data."

**Paper 7 — Reed et al. (Future Internet, 2025)**
> "Only 2 features (packet length + inter-arrival time) selected by information gain. No normalization beyond feature selection. Dataset reduced by 99.8% (620,678 → 4,905 packets). Decision Tree achieved 95.9% accuracy. Key insight: for resource-constrained IoT, minimal features beat complex preprocessing."

#### From Folder 2 (LLM for IDS Papers)

**Paper F2-1 — Bui et al. (ACM Networking, 2024)**
> "No oversampling — used temporal adversarial train/test split. Fine-tuned BERT (110M) beats 7B models. Class imbalance handled through balanced sampling at inference time."

**Paper F2-3 — DoLLM (Li et al., arXiv 2024)**
> "9 flow features: frequency encoding for categorical, min-max normalization for numerical. Evaluated with 10:1 imbalance ratio (realistic). DoLLM improved XGBoost F1 by +6.6% in imbalanced condition — LLM more robust to imbalance than pure ML."

**Paper F2-4 — Mehavilla et al. (AI Review, 2026)**
> "z-score standardization + one-hot encoding → 194 dims via Zeek conn.log. No oversampling. XGBoost best performer (F1=0.9696). SHAP analysis: tunnel_parents, orig_bytes, proto, history, orig_ip_bytes = most discriminative."

#### From Supervisor Shared Papers

**DDoSBERT paper:**
> "Mutual Information feature selection outperforms Correlation and Univariate selection for DDoS feature selection — captures non-linear dependencies. Near-perfect accuracy on volumetric DDoS but this is trivially detectable by rate thresholds — slow-rate is harder."

**LLM-APTDS (Yang et al., FGCS, 2026):**
> "Complementary sample distribution for imbalanced data — LLM 1 trained on high-confidence positives (precision-focused), LLM 2 on balanced data (recall-focused). Dual-model fusion handles class imbalance WITHOUT oversampling. Novel approach for extreme imbalance."

#### From Folder 3 (Detect + Explain Papers)

**Paper F3-3 — From Flows to Words (arXiv, 2025):**
> "Boolean domain flags prepended to prompts: asymmetry_high, pkt_rate_high, ttl_anomaly, tcp_timer_anomaly, rare_service_state, short_burst. These flags dramatically improve LLM detection quality from near-random to F1=0.783. Key insight: hand-crafted domain flags are more powerful than raw numeric features for LLMs."

**Paper F3-2 — HuntGPT (Ali & Kostakos, 2023):**
> "SHAP + LIME provide the feature importance bridge between ML detection and LLM explanation. Standard KDD99 preprocessing, no novel techniques. Architecture insight is the contribution."

---

### 2.2 Feature Scaling/Normalization

| Paper | Technique Used | Why |
|---|---|---|
| Al-Shukaili et al. (F1-P4) | MinMaxScaler | Scale to [0,1] for DNN input |
| Ramirez-Martinez et al. (F1-P8) | Z-score (mean/std normalization) | Save from training, apply at inference |
| DoLLM (F2-P3) | Min-max normalization (9 features) | Simple, numerical features only |
| Mehavilla et al. (F2-P4) | Z-score + one-hot encoding | Zeek conn.log mixed features |
| Reed et al. (F1-P7) | None needed | 2 features only, DT doesn't require scaling |
| Bocu & Iavich (F1-P6) | L2 normalization | Federated learning regularization |

**Key observation:** Papers using tree-based models (XGBoost, RF, DT) often skip scaling entirely as it is not needed. Papers using neural networks and LLMs universally apply some form of normalization. No paper from our folders uses RobustScaler explicitly — but several use z-score which is sensitive to outliers.

---

### 2.3 Feature Selection

| Paper | Method | Features Selected | Result |
|---|---|---|---|
| Al-Shukaili et al. (F1-P4) | Wrapper RFE (RF) | 50 features | F1=95.45% — best in paper |
| Al-Shukaili et al. (F1-P4) | Filter SelectKBest + MI | 20 features | F1=94.12% — slightly worse |
| Rios et al. (F1-P9, FRE) | Information gain | 2 features | F1=99.18% — best result |
| Ramirez-Martinez et al. (F1-P8) | P4-computable features | 12 features | F1=97.21% |
| Reed et al. (F1-P7) | Information gain ranking | 2 features | 95.9% accuracy |
| DDoSBERT (Supervisor) | Mutual Information | Top features | Best MI > correlation > univariate |
| Mehavilla et al. (F2-P4) | SHAP on XGBoost | Top 5 features identified | tunnel_parents most important |

**Critical finding from our papers:**
- The FRE paper (Rios et al.) achieved **F1=99.18% with only 2 features** (Shannon entropy + distinct source ports) on their custom Slowloris dataset
- This is the highest Slowloris detection result in all our papers and used the fewest features
- This strongly suggests that for Slowloris specifically, a small number of highly discriminative features outperforms using all available features

---

### 2.4 Input Representation for LLMs/SLMs

| Paper | Format Used | Performance |
|---|---|---|
| F3-1 (Houssel et al.) | Key-value pairs: `"L4_DST_PORT: 80"` | Near-random (zero-shot), slight improvement with ORPO/KTO |
| F3-3 (From Flows to Words) | Compact NL + Boolean flags | F1=0.783 (best zero-shot LLM result) |
| F2-3 (DoLLM) | 9-feature vector → MLP embedding → LLM token | F1=0.938 (imbalanced) |
| F2-10 (TrafficLLM) | Traffic-specific tokenization layer | Best multi-task result |
| SS (DDoSBERT) | Feature name + value pairs, NL format | Near-perfect on volumetric DDoS |
| F2-1 (Bui et al.) | Raw payload bytes + 5-tuple | Fine-tuned BERT beats 7B models |

**Key finding:** Key-value pairs consistently outperform natural language sentences for LLM input. Boolean domain flags (From Flows to Words) are the highest-performing zero-shot approach. Specialized tokenization (TrafficLLM, DoLLM) outperforms generic verbalization.

---

## 3. Additional Research (Web Search, 2024–2026)

### 3.1 Advanced Oversampling for Rare Attacks

**CTGAN for Rare IoT Attacks (arXiv, 2025)**
- CTGAN generates high-fidelity synthetic rare attack samples without changing statistical properties
- Maintained detection accuracy above 98% while achieving materially higher recall for rare attack categories
- Combined CTGAN + SMOTEENN two-stage approach outperforms SMOTE alone for rare classes
- **Directly applicable to our 132 Slowloris samples**

**Borderline-SMOTE vs ADASYN vs SMOTE (MDPI Electronics, 2025)**
- Five methods compared: ROS, SMOTE, SMOTE-Tomek, Borderline-SMOTE, ADASYN
- For rare minority classes in IDS: ADASYN > Borderline-SMOTE > standard SMOTE
- ADASYN-based ensemble with LightGBM: 99.85% detection accuracy
- Standard SMOTE functions as recall amplifier but creates synthetic samples far from decision boundary

**SMOTE-IPF (Springer, 2025)**
- SMOTE + Iterative Partitioning Filter removes noisy synthetic samples
- Cleaner synthetic data than standard SMOTE
- 2-3% F1 improvement over standard SMOTE on highly imbalanced datasets

### 3.2 Feature Scaling Comparison

**Effects of Feature Selection and Normalization on NIDS (ScienceDirect, 2024)**
- Log transformation before scaling significantly improves performance on skewed features
- Network traffic features (packet counts, byte counts) follow power-law distributions
- Log1p + z-score normalization outperforms z-score alone by 2-4% on highly skewed features
- RobustScaler + log transformation is optimal combination for network traffic

### 3.3 Combined Feature Selection

**Top-K Feature Selection for IoT IDS (MDPI, 2025)**
- XGBoost + Mutual Information combined selection outperforms either alone
- Top 20 combined features achieves 0.98 accuracy — comparable to all-feature baseline with less noise
- Evaluation across Top-10, Top-15, Top-20, All features: Top-20 is optimal point

**Comprehensive Feature Selection (SCITEPRESS, 2025)**
- Chi-square + SMOTE + XGBoost: 0.98 accuracy with only 20 features
- Wrapper-based RFE > Filter methods but 10× computational cost
- For resource-constrained devices: Filter (MI + XGBoost importance) is the best trade-off

---

## 4. Comparative Analysis: Our Current Pipeline vs Literature

| Step | Our Current Approach | Literature Best Practice | Gap |
|---|---|---|---|
| **Class imbalance** | Standard SMOTE (training only) | CTGAN for extreme minority (132 samples) | Large — standard SMOTE creates unrealistic Slowloris samples |
| **Feature scaling** | RobustScaler only | Log1p transformation THEN RobustScaler | Small but impactful — skewed features not compressed |
| **Feature selection** | All 70 features (XGBoost), top 13 (SLM) | Combined XGBoost + MI, top 20 | Minor — we use only one selection method |
| **Label design** | 6 classes (MitM in "other") | Separate MitM (1,471 samples) and malware (1,444) | Moderate — heterogeneous class hurts boundaries |
| **SLM input format** | Natural language sentences | Key-value pairs + Boolean domain flags | Significant — sentences waste tokens, flags add signal |
| **Correlation removal** | Not done | Remove features with >0.95 correlation | Minor |
| **Train/test strategy** | Stratified 80/20 | Stratified 80/20 + cross-validation | Minor |
| **Domain flags** | Not implemented | Boolean attack-specific flags (F3-3) | High impact for SLM performance |

---

## 5. Recommended Preprocessing Pipeline

Based on all 46 folder papers and additional web search papers, the following pipeline is recommended for maximum performance on resource-constrained IIoT devices.

---

### STEP 1 — Data Loading and Cleaning
**What:** Load CSV, drop 24 non-feature columns (timestamps, IDs, MAC/IP string columns, binary_label, flow_duration_sec)

**Why:** Non-numeric columns cannot be used by any model. Zero-variance features (flow_duration_sec = 10 for all rows) add no information.

**Reference:** Universal across all papers. Reed et al. (F1-P7) found aggressive column dropping essential for IoT deployment.

---

### STEP 2 — Redesign to 8-Class Labels
**What:** Use label2 for categories. Pull Slowloris out of dos using label3. Result:
- 0 = benign (13,680), 1 = slowloris (132), 2 = dos_other (~3,148), 3 = ddos_other (3,234), 4 = recon (6,005), 5 = mitm (1,471), 6 = malware (1,444), 7 = other (916)

**Why:** Our current class 5 ("other") contains MitM, malware, web, and bruteforce — four fundamentally different attack types with different network signatures. Mixed classes confuse the classifier. With 1,471 MitM and 1,444 malware samples, both justify dedicated classes.

**Reference:** Multi-class IDS papers (MDPI Sensors, 2025) consistently separate attack categories when sufficient samples exist. Mehavilla et al. (F2-P4) show cleaner class boundaries improve F1 across all classes.

---

### STEP 3 — Remove Highly Correlated Features
**What:** Calculate Pearson correlation matrix. Drop one feature from each pair with correlation > 0.95.

**Why:** Highly correlated features are redundant — they add noise without information. Expected to remove 5-8 features from the 70, leaving ~62-65.

**Reference:** Al-Shukaili et al. (F1-P4) found wrapper RFE effectively removes correlated features. Comprehensive feature selection paper (SCITEPRESS, 2025) identifies correlation removal as first filter step.

---

### STEP 4 — Log1p Transformation for Skewed Features
**What:** Apply log1p (= log(x+1)) to features with high skewness (skewness > 2). Specifically: all packet count features, byte count features, and TCP flag count features which range from 0 to 1,488,000.

**Why:** Network traffic features follow power-law distributions. packet_all_count has mean=99,590 but median=56 — extreme right skew. Log transformation compresses this to a manageable range, making the distribution closer to normal and improving both tree-based and neural network model performance.

**Reference:** ScienceDirect (2024) — log transformation before scaling improves F1 by 2-4% on network traffic. Bocu & Iavich (F1-P6) use L2 normalization implying awareness of scale issues.

**Implementation note:** Use log1p not log to handle zero values (log(0) is undefined, log1p(0) = 0).

---

### STEP 5 — RobustScaler
**What:** Fit RobustScaler on training set only. Apply to both train and test.

**Why:** Even after log transformation, some extreme outliers remain. RobustScaler uses median and interquartile range instead of mean and standard deviation, making it resistant to remaining outliers. This is especially important for network traffic where legitimate anomalous flows (not attacks) create statistical outliers.

**Reference:** Standard practice for network traffic data. Al-Shukaili et al. (F1-P4) use MinMaxScaler; Ramirez-Martinez et al. (F1-P8) use z-score. RobustScaler is the most robust choice for our highly skewed dataset.

---

### STEP 6 — Combined Feature Selection (Top 20)
**What:** Run two selection methods independently:
1. XGBoost feature importance (gain-based) — rank all features
2. Mutual Information — rank all features independently

Take union of top 20 from each method. Final set: 20-25 most discriminative features confirmed by two independent methods.

**Why:** Using only one method can miss features that are important in a non-linear way (XGBoost gain can miss features that interact with others). Mutual information captures non-linear dependencies that XGBoost importance sometimes misses.

**Reference:**
- DDoSBERT (Supervisor Shared): "Mutual Information consistently outperforms Correlation and Univariate selection"
- Al-Shukaili et al. (F1-P4): Wrapper RFE (50 features) > Filter SelectKBest (20 features) but too slow for IoT
- Top-K Feature Selection (MDPI, 2025): Combined XGBoost + MI, top 20 is optimal for IoT IDS
- Rios et al. (F1-P9): Only 2 features needed for Slowloris detection using information gain — confirms MI is powerful for this task

---

### STEP 7 — Stratified 80/20 Train/Test Split
**What:** Stratified split maintaining class proportions in both sets. Test set is locked — never touched during preprocessing or training.

**Why:** Stratified split ensures rare classes (Slowloris: 132 total → ~26 in test) appear in both sets. Non-stratified split risks zero Slowloris in test set.

**Reference:** Universal across all papers. Ramirez-Martinez et al. (F1-P8) use 20 runs × 3 folds for statistical robustness — worth adding for final paper.

---

### STEP 8 — CTGAN Oversampling for Slowloris (Training Set Only)
**What:** Train a CTGAN model on the 106 Slowloris training samples. Generate synthetic samples to reach 1,000 Slowloris training samples. Apply Borderline-SMOTE to balance remaining minority classes (dos_other, malware, other, web, bruteforce) in training set.

**Why:** Standard SMOTE creates synthetic Slowloris samples by linear interpolation between real samples. With only 106 real samples, most synthetic samples are in a small region of feature space and do not capture the full diversity of Slowloris behavior. CTGAN uses a conditional GAN to generate samples from the learned data distribution — producing more realistic and diverse synthetic Slowloris flows.

**Reference:**
- CTGAN paper (arXiv, 2025): "CTGAN maintained detection accuracy above 98% while yielding materially higher recall for rare attack categories compared to SMOTE"
- CTGAN-IDS (MDPI Sensors, 2023): Two-stage CTGAN + SMOTEENN outperforms SMOTE alone for IoT rare attacks
- Gogoi & Ahmed (F1-P2): Synthetic data generation critical when real attack traces are few

**Important:** Apply CTGAN and oversampling to training set ONLY. Test set retains original distribution.

---

### STEP 9 — Compute Class Weights (for SLM training)
**What:** Compute sklearn class_weight="balanced" weights for all 8 classes. Slowloris weight will be ~37-40x.

**Why:** SLM uses class-weighted CrossEntropyLoss instead of SMOTE. Class weights handle imbalance at the loss function level without generating synthetic text descriptions (which would be unreliable).

**Reference:** LLM-APTDS (Supervisor Shared): dual-model approach handles imbalance without oversampling. F2-1 (Bui et al.): balanced sampling at inference time is valid alternative.

---

### STEP 10 — Boolean Domain Flags (for SLM input only)
**What:** Compute 6 binary domain flags per flow from the original unscaled features:

```python
flag_high_psh      = psh_count > threshold_psh           # Slowloris signature
flag_ip_flags_2    = ip_flags_min == 2                   # Unique Slowloris indicator  
flag_slow_interval = time_delta_avg < threshold_slow      # Slow-rate behavior
flag_high_packets  = packets_all_count > threshold_vol    # High volume
flag_low_payload   = payload_length_avg < threshold_pay   # Small payload (Slowloris)
flag_high_syn      = tcp_flags_syn_count > threshold_syn  # SYN flood signature
```

**Why:** From Flows to Words (F3-3) showed Boolean flags increase LLM detection from near-random to F1=0.783. Flags make implicit numeric patterns explicit as binary signals the LLM can reason about directly.

**Reference:**
- F3-3 (From Flows to Words, arXiv 2025): Boolean domain flags are the single biggest improvement for prompt-only LLM IDS
- Our own XGBoost analysis: ip_flags_min=2 ratio 8.2x; psh_count ratio 500x — these are natural threshold candidates

---

### STEP 11 — Save All Artifacts
**What:** Save the following to dataset/preprocessed/:
- `X_train.npy`, `X_test.npy` — selected features, scaled
- `y_train.npy`, `y_test.npy` — 8-class labels
- `X_train_resampled.npy` — after CTGAN + Borderline-SMOTE
- `y_train_resampled.npy`
- `scaler.pkl` — fitted RobustScaler (for inference)
- `feature_names.txt` — final 20-25 selected feature names
- `label_map.json` — 8-class mapping
- `class_weights.pkl` — for SLM training
- `domain_flags_thresholds.json` — thresholds for Boolean flags at inference

**Why:** All artifacts needed for both training and inference must be saved at preprocessing time. The scaler and feature list in particular are critical — without them, new data cannot be correctly preprocessed at inference.

**Reference:** Ramirez-Martinez et al. (F1-P8): "Feature normalization using mean/STD saved from training" — explicit about saving preprocessing artifacts.

---

### STEP 12 — Key-Value Verbalization (for SLM input)
**What:** Replace natural language sentences with compact key-value format:
```
ip_flags_min:2 packets_total:3241K ttl_min:64 window_max:65535 psh_count:812 
tcp_flags_std:6.68 time_delta_avg:0.0009 payload_avg:23.2 mss_max:1460 
macs_dst:7 HIGH_PSH:1 SLOW_INTERVAL:1 IP_FLAGS_2:1
```

**Why:** Natural language sentences waste tokens on filler words ("The flow had... packets with... and..."). Key-value format gives more information per token. Boolean flags appear at the end as explicit signal. Papers consistently show key-value > natural language for numeric tabular data.

**Reference:**
- F3-1 (Houssel et al.): Key-value pairs `"L4_DST_PORT: 80"` used as standard format
- F3-3 (From Flows to Words): Compact NL + Boolean flags achieves best zero-shot LLM result
- Supervisor Shared (DDoSBERT): "Feature name + value pairs" format
- Web search (LLM-Based Cyberattack Detection, MDPI, 2025): Key-value format standard for LLM IDS

---

## 6. Pipeline Summary

```
Raw CSV (30,030 rows × 97 columns)
         ↓
STEP 1:  Drop 24 non-feature columns → 73 columns
         ↓
STEP 2:  Redesign labels → 8 classes
         ↓
STEP 3:  Remove correlated features (>0.95) → ~62-65 features
         ↓
STEP 4:  Log1p transform on skewed features
         ↓
STEP 5:  RobustScaler (fit on train, apply to both)
         ↓
STEP 6:  Combined feature selection (XGBoost + MI, top 20)
         ↓
STEP 7:  Stratified 80/20 split
         ↓
STEP 8:  CTGAN oversampling for Slowloris (train only)
         + Borderline-SMOTE for other minorities
         ↓
STEP 9:  Compute class weights (for SLM)
         ↓
STEP 10: Compute Boolean domain flags (for SLM)
         ↓
STEP 11: Save all artifacts
         ↓
STEP 12: Key-value verbalization (for SLM training)
```

**For XGBoost:** Uses Steps 1-9 output (CTGAN-balanced, all 20 selected features)
**For Encoder SLM:** Uses Steps 1-7 output (original distribution) + Steps 9-12 (class weights, flags, key-value)
**For Decoder SLM:** Uses Steps 1-12 output (key-value format with Boolean flags as instruction input)

---

## 7. Resource-Constrained Deployment Considerations

| Component | Training cost | Inference cost | Notes |
|---|---|---|---|
| CTGAN | One-time ~10-30 min | Zero | Never runs at inference |
| Borderline-SMOTE | One-time ~1 min | Zero | Never runs at inference |
| Log1p + RobustScaler | One-time | Milliseconds | One line of code per flow |
| Feature selection (20 features) | One-time | Minimal | Reduces inference compute |
| Boolean flags | One-time threshold calc | Microseconds | 6 comparisons per flow |
| Key-value verbalization | — | Milliseconds | String formatting |

The entire inference pipeline for a new flow requires: log1p transformation → scale with saved scaler → select 20 features → compute 6 Boolean flags → format as key-value string. This runs in under 5 milliseconds on any hardware including Raspberry Pi-class IIoT devices.

---

## 8. Expected Performance Improvement

Based on literature reports for each individual technique:

| Improvement | Expected gain (F1 Slowloris) | Source |
|---|---|---|
| 8-class label redesign | +2-4% | Multi-class IDS papers (MDPI, 2025) |
| Log1p + RobustScaler | +2-3% | ScienceDirect (2024) |
| CTGAN oversampling | +10-15% | CTGAN paper (arXiv, 2025) |
| Combined XGBoost + MI selection | +1-2% | MDPI Top-K (2025) |
| Key-value format (SLM) | +5-8% | F3-3, F3-1 papers |
| Boolean domain flags (SLM) | +8-12% | F3-3 (From Flows to Words) |
| **Total expected (XGBoost)** | **~0.93-0.96** | (from current 0.91) |
| **Total expected (SLM)** | **~0.68-0.78** | (from current 0.53) |

---

## 9. References

### From Our Research Folders

1. Al-Shukaili, A., Kiah, M.L., & Ahmedy, I. (2025). Optimizing feature selection for LDDoS detection. *Discover Internet of Things*. [SMOTE + RFE + DNN pipeline]

2. Ramirez-Martinez, D., Pérez-Díaz, J., & Yungaicela-Naula, N. (2025). P4-Assisted Slowloris DDoS Attack Detection in IoT Environments. *Computer Networks*, Elsevier. [Z-score normalization, 12 P4 features, F1=97.21%]

3. Rios, J., Inácio, P., Magoni, D., & Freire, M. (2024). Detection of Slowloris Attacks using Machine Learning Algorithms. *ACM SAC 2024*. [2-feature information gain, F1=99.18% — highest Slowloris result]

4. Reed, B., Dooley, A., & Kouadri Mostefaoui, G. (2025). Minimal Overhead Modelling of Slow DoS Attack Detection for Resource-Constrained IoT Networks. *Future Internet*, MDPI. [2-feature LPC strategy, 99.8% data reduction for IoT]

5. Bui, T., Boffa, M., et al. (2024). A Systematic Comparison of Large Language Models Performance for Intrusion Detection. *ACM Networking*. [Fine-tuned BERT (110M) beats 7B; domain-specific fine-tuning]

6. Li, X., Zhang, Y., et al. (2024). DoLLM: Detecting Low-Rate DDoS Attack via Large Language Model. *arXiv*. [Min-max normalization, 9 features, +6.6% over XGBoost on imbalanced data]

7. Mehavilla, J., Rodríguez, M., García, S., & Alesanco, A. (2026). Evaluating Large Language Models Effectiveness for Flow-Based Intrusion Detection. *AI Review*. [Z-score + one-hot, SHAP features, XGBoost best baseline]

8. Houssel, P., Singh, S., Layeghy, S., & Portmann, M. (2024). Towards Explainable Network Intrusion Detection using LLMs. *IEEE BDCAT 2024*. [Key-value format; hallucination analysis; 7000× latency gap]

9. Rehman, U., Shah, M., et al. (2025). From Flows to Words: Can Zero-/Few-Shot LLMs Detect Network Intrusions? *arXiv*. [Boolean domain flags + key-value format; F1=0.783 best zero-shot]

10. Yang, L., et al. (2026). LLM-APTDS: High-precision APT detection for imbalanced data. *Future Generation Computer Systems*. [Dual-model fusion for imbalance without oversampling]

11. Bocu, R., & Iavich, M. (2024). Enhanced detection of low-rate DDoS attack patterns. *JNCA, Elsevier*. [L2 normalization; FL distributed training; 98.79% accuracy]

12. Diaf, A., et al. (2024). Beyond Detection: Leveraging LLMs for Cyber Attack Prediction in IoT Networks. *DCOSS-IoT 2024*. [Tranalyzer 26 features; min-max normalization; multi-stage SLM pipeline]

### From Additional Web Search

13. A Conditional Tabular GAN-Enhanced IDS for Rare Attacks in IoT Networks. *arXiv*, 2025. DOI: 10.48550/arXiv.2502.06031. [CTGAN for rare attack oversampling; 98%+ recall maintained]

14. Addressing Class Imbalance in Intrusion Detection: Comprehensive Evaluation. *MDPI Electronics*, 2025. [ROS vs SMOTE vs Borderline-SMOTE vs ADASYN comparison; ADASYN best for rare classes]

15. Top-K Feature Selection for IoT IDS: Contributions of XGBoost, LightGBM, RF. *MDPI Future Internet*, 2025. [Combined feature selection; Top-20 optimal for IoT]

16. Effects of Feature Selection and Normalization on NIDS. *ScienceDirect*, 2024. [Log transformation + z-score improves F1 by 2-4% on skewed features]

17. Optimizing Feature Selection for LDDoS Detection. *Discover IoT, Springer*, 2025. [Chi-square + SMOTE + XGBoost; 0.98 accuracy with 20 features]

18. Privacy-Preserving Synthetic Data Generation for IoT-Sensor IDS Using CTGAN. *PMC/NCBI*, 2024. [CTGAN for synthetic minority data in IoT sensor networks]

19. Multi-Class IDS for DDoS Attacks in IoT Networks Using Deep Learning and Transformers. *PMC/NCBI*, 2025. [Multi-class design; transformer + CNN-BiLSTM hybrid architecture]

20. SMOTE-IPF: Optimized IDS for Big Data with SMOTE-IPF Data Balancing. *Springer SN Computer Science*, 2025. [SMOTE + Iterative Partitioning Filter; 2-3% F1 improvement over standard SMOTE]
