# Research Notes — Supervisor Shared Folder
**Project:** Small Language Models for Slow-Rate DoS Attack Detection and Explanation  
**Folder:** `supervisor shared/` — 7 papers  
**Date compiled:** 2026-05-25  

---

## Paper SS-1: Towards Explainable Network Intrusion Detection using Large Language Models

**Full citation:** Houssel, P. et al. (2024). "Towards Explainable Network Intrusion Detection using Large Language Models." *IEEE BDCAT 2024*. University of Queensland, Australia.

### Problem / Motivation
Existing ML-based NIDS achieve high accuracy but remain black boxes — they cannot explain *why* a flow is malicious. The paper investigates whether LLMs can serve a dual role as both classifiers and explainers for network intrusion detection, using raw NetFlow data as input.

### System Architecture
- **Input format:** NetFlow records converted to natural-language text prompts (feature name + value pairs)
- **Models tested:**
  - GPT-4o (via API, closed-source)
  - LLama3-8B (open-source, local)
- **Paradigms evaluated:** Zero-shot, fine-tuning via KTO (Kahneman-Tversky Optimisation) and ORPO (Odds Ratio Preference Optimisation)
- **Dataset:** CIC-IDS dataset (multi-class network intrusion)
- **Evaluation:** Precision, Recall, F1; also qualitative hallucination assessment

### Key Technical Details
- **Zero-shot performance:** Both models performed near random — LLama3 ~48–50% precision, GPT-4o ~50–53%. The models lack cybersecurity-specific knowledge for direct flow classification.
- **Fine-tuning:** KTO/ORPO provided marginal improvement; best observed ~55–59% F1. Fine-tuning on NetFlow text is non-trivial because the feature space is numerical and lacks natural-language semantics.
- **Inference latency gap:**
  - LLama3-8B: ~14,000 µs per sample
  - Random Forest: ~2.03 µs per sample
  - **Ratio: ~7,000× slower** — a fundamental deployment obstacle for real-time IDS
- **Hallucination analysis:** LLMs produce fluent, confident explanations but embed factual errors — incorrect geographic locations (e.g., attributing traffic to wrong country), wrong protocol numbers, invented attack names. This is especially dangerous for a security analyst who may trust the LLM's authoritative tone.
- **Recommended future direction:** RAG (Retrieval-Augmented Generation) with function-calling to ground explanations in verified threat intelligence databases; separate ML classifier for detection + LLM for post-hoc explanation.

### Experimental Results
| Paradigm | Model | Precision |
|---|---|---|
| Zero-shot | LLama3-8B | ~48–50% |
| Zero-shot | GPT-4o | ~50–53% |
| Fine-tuned (KTO/ORPO) | LLama3-8B | ~55–59% |

Latency: LLama3 = 14,000 µs; RF = 2.03 µs (~7,000× gap)

### Limitations
- No slow-rate attack types tested (Slowloris, RUDY, etc.)
- Fine-tuning barely moves the needle beyond zero-shot for classification
- Latency makes any deployment infeasible without hardware acceleration
- Hallucinations in explanations undermine trust — no ground-truth grounding

### SLM Relevance to Project
**★★★★★ — Directly central**  
This paper provides the most direct empirical evidence supporting the project's core thesis: LLMs are too slow and too unreliable for direct classification of network traffic. The ~7,000× latency gap is a concrete benchmark to cite. The recommendation for hybrid architecture (ML classifier + LLM explainer) is precisely the design this project should investigate. The hallucination finding reinforces why RAG or structured prompting is needed. Direct citation expected in nearly every section of the thesis.

---

## Paper SS-2: Evaluating Small Language Models for Intrusion Detection on Automotive Embedded Platforms

**Full citation:** Salah, M. et al. (2025). "Evaluating Small Language Models for Intrusion Detection on Automotive Embedded Platforms." *RACS '25*. Montclair State University, USA.

### Problem / Motivation
Modern vehicles contain ECUs connected via CAN bus, making them vulnerable to cyber attacks. Traditional IDS methods (DNN, SVM) require powerful hardware. This paper asks: can compact SLMs (tens of millions of parameters) run IDS inference directly on automotive-grade embedded hardware in real time?

### System Architecture
- **Input format:** CAN bus logs converted to structured text — raw CAN frames become strings like `"ID_123 11 22 33 44 55 66 77 88"` (ID + 8 data bytes in hex). A sliding window groups multiple frames per inference step.
- **Models tested:**
  - **MiniLM** (22M parameters, 86MB on disk)
  - **DistilBERT** (66M parameters, 256MB)
  - **TinyBERT** (14M parameters, 56MB)
- **Dataset:** CAN-MIRGU — real-world CAN bus data collected under actual driving conditions; attack types: DoS, fuzzing, replay, spoofing attacks
- **Target hardware:** NXP i.MX 8M Plus (ARM Cortex-A53 quad-core @ 1.8 GHz, 6GB LPDDR4 RAM) — representative automotive-grade embedded platform
- **Reference hardware:** Desktop (AMD Ryzen 5 7530U CPU)

### Key Technical Details
- **CAN-to-text transformation:** No semantic enrichment — raw hex values are treated as text tokens. The tokeniser must learn statistical patterns in hex sequences, which it does effectively.
- **Window-based classification:** Multiple CAN frames aggregated per inference — the window size trades latency for context; the optimal size is empirically chosen.
- **~14× slowdown** observed uniformly across all three models when moving from desktop to embedded hardware — a reliable rule-of-thumb for embedded SLM deployment planning.
- **Memory footprint:** TinyBERT at 56MB easily fits on-chip; DistilBERT at 256MB is borderline and caused the worst latency.
- All models were fine-tuned via standard BERT sequence classification head (binary: attack / benign).

### Experimental Results
| Model | Params | Disk Size | F1 Score | Accuracy | Latency (Embedded) |
|---|---|---|---|---|---|
| MiniLM | 22M | 86MB | **0.883** | **92.4%** | 336 ms/window |
| TinyBERT | 14M | 56MB | 0.847 | ~90% | **157 ms/window** |
| DistilBERT | 66M | 256MB | 0.831 | ~88% | 1207 ms/window |

Key finding: DistilBERT's larger memory footprint causes extreme latency on embedded (1.2 s/window = ~0.8 FPS) — effectively impractical. TinyBERT achieves ~6.3 FPS (157 ms) — borderline real-time for CAN bus (which runs at up to 1 Mbit/s). MiniLM offers best accuracy–latency balance.

### Limitations
- CAN bus domain differs significantly from TCP/IP network traffic relevant to this project's slow-rate DoS focus
- No explanation/XAI component — pure classification
- Real-time threshold for CAN (1 ms inter-frame gap) is still not met by any model
- Dataset contains DoS attacks but the CAN variant (frame flooding), not slow-rate application-layer DoS

### SLM Relevance to Project
**★★★★★ — Directly central**  
This is the only paper in the entire corpus that provides empirical measurements of SLM inference speed on real embedded hardware. The 14× slowdown figure, the sub-100MB footprint constraint, and the three-way accuracy–latency–size trade-off are directly transferable to the network IDS embedded deployment context. The finding that even a 14M-parameter model (TinyBERT) can achieve F1=0.847 is strong justification for the SLM approach. This paper should be cited heavily in any hardware feasibility analysis.

---

## Paper SS-3: Lightweight Fine-Tuning of LLMs for Explainable Intrusion Detection in SDN

**Full citation:** Lodh, S. et al. (2025). "Lightweight Fine-Tuning of LLMs for Explainable Intrusion Detection in SDN." *WiMob 2025*.

### Problem / Motivation
Software-Defined Networking (SDN) centralises network control, making the SDN controller a high-value target. The paper investigates whether small-to-medium LLMs (2–7B parameters) fine-tuned via 4-bit QLoRA can serve as accurate IDS for SDN traffic, and whether SHAP can provide XAI explanations for these models.

### System Architecture
- **Models tested:**
  - GPT-Neo 2.7B
  - Phi-2 (~2.7B, Microsoft)
  - LLaMA2-7B (Meta)
- **Fine-tuning:** 4-bit QLoRA (Quantized LoRA) via BitsAndBytes library; LoRA rank=8, alpha=32, targeting attention layers
- **Input prompt format:** Tabular network features converted to text: `"'Feature Name' is 'Value'"` — one sentence per feature, all concatenated into a single prompt
- **Dataset:** InSDN dataset — 342,889 records, 84 features, 7 attack classes (DoS, DDoS, Probe, R2L, U2R, Web Attack, Normal)
- **XAI:** SHAP applied to both the LLMs (text inputs) and a baseline XGBoost classifier (tabular inputs)
- **Energy tracking:** CodeCarbon library used to estimate CO2 emissions per training run

### Key Technical Details
- **QLoRA efficiency:** 4-bit quantisation reduces VRAM from ~14GB (full LLaMA2-7B) to ~4GB, enabling fine-tuning on consumer GPUs. LoRA adds only ~1% additional trainable parameters.
- **Convergence:** All three models reach F1 = 1.00 at 350,000 training samples — full InSDN dataset. At smaller sample sizes, performance varies.
- **Energy consumption comparison:**
  - LLaMA2-7B: 0.304 kg CO2 at 25,000 samples (most expensive)
  - Phi-2: 0.173 kg CO2 consistently across sample sizes (most efficient per parameter)
  - GPT-Neo: intermediate
- **SHAP incompatibility finding (critical):** SHAP applied to LLM text inputs produces token-level attributions. Because tabular features are tokenised (e.g., `"0 + + id + le + min"` for a feature value), SHAP outputs scores for each sub-token rather than the original feature. The result is semantically meaningless — fragmented strings with no correspondence to network features. Example output: `"0 + + id + le + min + is + 0"` with scores on individual characters. This is fundamentally unintelligible to a network analyst.
- **SHAP on XGBoost (comparison):** On tabular XGBoost, SHAP produces clean feature-level attributions (e.g., Pkt_Len_Max=0.42, Init_Bwd_Win_Byts=0.31) — interpretable and directly actionable.

### Experimental Results
| Model | F1 (full dataset) | Approx. CO2 (25k samples) |
|---|---|---|
| GPT-Neo 2.7B | 1.00 | ~0.22 kg |
| Phi-2 ~2.7B | 1.00 | 0.173 kg |
| LLaMA2-7B | 1.00 | 0.304 kg |

Note: F1=1.00 on InSDN is consistent with prior work — InSDN is relatively easy (clear inter-class boundaries). The key finding is not the accuracy but the XAI incompatibility.

### Limitations
- InSDN may be too simple (F1=1.00 is suspicious; likely train/test overlap or easy dataset)
- No slow-rate attack types in InSDN
- SHAP analysis definitively shows LLM text-input XAI is broken — but the paper offers no alternative
- High energy consumption limits deployment at edge
- No latency/throughput numbers given for inference

### SLM Relevance to Project
**★★★★★ — Directly central (XAI design)**  
The SHAP incompatibility finding is one of the most important technical insights in the entire corpus. It definitively rules out applying SHAP directly to LLM text inputs for feature-level explanation — which is a design trap the project must avoid. The correct architecture must either (a) use SHAP on a separate tabular classifier and feed results to the LLM as structured text, or (b) design feature-attribution prompts that bypass tokenisation artifacts. Phi-2's energy efficiency also makes it a candidate SLM for the project's model selection. QLoRA fine-tuning is directly applicable as the parameter-efficient training method.

---

## Paper SS-4: Revolutionizing Cyber Threat Detection With Large Language Models — SecurityBERT

**Full citation:** Ferrag, M.A. et al. (2024). "Revolutionizing Cyber Threat Detection With Large Language Models: A Privacy-Preserving BERT-Based Lightweight Model for IoT/IIoT Devices." *IEEE Access*, Vol. 12.

### Problem / Motivation
IoT and IIoT devices are highly constrained (limited RAM/CPU) but face diverse cyber threats. Standard BERT (110M parameters) is too large for IoT deployment. Additionally, network traffic features may contain sensitive user information — existing approaches send raw features to cloud classifiers, raising privacy concerns. The paper addresses both constraints simultaneously.

### System Architecture
**SecurityBERT** — a custom compact BERT variant:
- **Architecture:** 15 transformer layers, 11M parameters total, 16.7MB model size (vs. BERT-base: 110M params, ~440MB)
- **Privacy-Preserving Fixed-Length Encoding (PPFLE):**
  - Each feature is hashed: `H(column_name + "$" + value)` — using a cryptographic hash function
  - Output: fixed-length binary/hex string per feature — no raw values exposed
  - Features are concatenated into a fixed-length "sentence" fed to the BERT tokeniser
  - Provides k-anonymity-style privacy: raw values cannot be recovered from the hash
- **Tokeniser:** ByteLevelBPE (Byte-Level Byte-Pair Encoding) with vocabulary size 5,000 — purpose-built for the hash-encoded input space, not natural language
- **Dataset:** Edge-IIoTset — 14 attack types (DoS, DDoS, Man-in-the-Middle, SQL Injection, XSS, etc.) + Normal traffic; 2M+ records
- **Deployment target:** CPU-only commodity hardware (and benchmarked on NVIDIA A100 GPU)

### Key Technical Details
- **PPFLE innovation:** Combines privacy with representation. The hash encoding ensures all features are the same length (no padding needed for variable-length feature values) and the BERT model never sees raw network data.
- **Tokeniser vocabulary (5,000):** Much smaller than standard BERT (30,522) — reduces model complexity and speeds inference. The vocab is trained on hashed network features, not text.
- **Training:** Standard masked language model pre-training followed by supervised fine-tuning for classification; 14+1 classes.
- **Comparison baselines:** CNN-LSTM, DNN, GAN-Transformer, standard BERT — SecurityBERT outperforms all.

### Experimental Results
| Model | Params | Accuracy | Inference (CPU avg) |
|---|---|---|---|
| **SecurityBERT** | **11M** | **98.2%** | **~0.15s** |
| CNN-LSTM | — | 97.14% | — |
| DNN | — | 94.67% | — |
| GAN-Transformer | — | 94.55% | — |
| Standard BERT | 110M | ~97% | ~1.2s |

GPU inference: 0.016s on NVIDIA A100 (SecurityBERT).  
Edge-IIoTset is the primary benchmark — 98.2% is the highest reported accuracy on this dataset at the time of publication.

### Limitations
- PPFLE hashing makes features opaque — XAI becomes harder (you cannot directly attribute decisions to original feature names without the mapping)
- 14 attack types in Edge-IIoTset do not include slow-rate DoS (Slowloris/RUDY/Slowread)
- 0.15s CPU inference may still be too slow for line-rate detection on high-speed links
- The privacy guarantee depends on hash collision resistance — not formally proven

### SLM Relevance to Project
**★★★★☆ — Highly relevant**  
SecurityBERT is the clearest existence proof that a BERT-based model compressed to 11M parameters can achieve state-of-the-art accuracy on IoT IDS benchmarks. The 10× reduction from BERT-base to 11M parameters while maintaining 98.2% accuracy is a strong justification for SLM-based IDS. PPFLE is a novel contribution for privacy-sensitive deployments. The primary gap is the absence of slow-rate DoS attacks from the evaluation — the project would need to replicate this approach on a dataset containing Slowloris/RUDY-type traffic. The architecture is directly inspirational for the project's model design.

---

## Paper SS-5: DDoSBERT — Transformers for DDoS Detection

**Full citation:** Le, T.T.H. et al. (2025). "DDoSBERT: A Transformer-Based Approach for DDoS Attack Detection Using Text Classification." *Computer Networks*, Elsevier.

### Problem / Motivation
DDoS detection has relied on traditional ML (Random Forest, SVM) applied to tabular network features. The paper proposes reframing network flow classification as a *text classification problem* — converting feature vectors to text strings and applying DistilBERT. This enables direct transfer of NLP pre-training to network security without designing custom neural architectures.

### System Architecture
- **Core approach:** Convert tabular flow features → string text → DistilBERT text classifier
- **Feature selection methods compared:**
  - Correlation-based (Pearson/Spearman correlation with label)
  - Mutual Information (MI) — measures non-linear feature-label dependence
  - Univariate statistical tests (ANOVA F-score)
  - Each method selects top-K features; paper evaluates sensitivity to K
- **DistilBERT variants fine-tuned (three models):**
  - Model 1: `distilbert-base-uncased` (standard 66M-param model)
  - Model 2: `prunebert-base-uncased-6-finepruned-w-distil-mnli` (pruned, smaller)
  - Model 3: `distilbert-base-uncased-finetuned-sst-2-english` (**best performer** — pre-fine-tuned on sentiment, then adapted)
- **Datasets (5 evaluated):**
  - APA-DDoS
  - DDoS Attack SDN
  - CRCDDoS2022
  - CICDDoS2019 (standard benchmark)
  - BCCC-2024 (most recent, hardest)
- **Task:** Multi-class DDoS attack type classification (binary + multi-class)

### Key Technical Details
- **Text conversion:** Numeric feature values concatenated in order, separated by spaces — similar to SS-3 (Lodh et al.) but without feature name labelling. Example: `"0.12 1500 443 6 0 1 ..."` where each token is a feature value.
- **DistilBERT suitability:** 66M parameters, 6 layers (vs. BERT-base 12 layers) — ~40% fewer parameters, ~60% faster inference, retains ~97% of BERT performance on NLP tasks. For network classification, this footprint is still not "small" by embedded standards.
- **Pre-fine-tuning advantage (Model 3):** The SST-2 sentiment-pre-tuned model outperforms the base model — suggesting that even domain-irrelevant intermediate fine-tuning improves the model's adaptability to the classification task.
- **Feature selection impact:** Mutual Information consistently outperforms Correlation and Univariate selection for DDoS feature selection — mutual information captures non-linear dependencies relevant to attack patterns.

### Experimental Results
| Dataset | Model 3 Accuracy | Notes |
|---|---|---|
| APA-DDoS | **100%** | Trivially separable dataset |
| DDoS Attack SDN | **100%** | Trivially separable |
| CRCDDoS2022 | **100%** | — |
| CICDDoS2019 | **~99.5%** | Standard benchmark |
| BCCC-2024 | **~98%** | Most realistic/challenging |

Mutual Information feature selection generally outperforms alternatives across datasets. Model 3 > Model 2 > Model 1 consistently.

### Limitations
- No slow-rate DoS attacks (Slowloris, RUDY, Slowread) — all evaluated attacks are volumetric DDoS
- 100% accuracy on three datasets suggests data leakage or trivially separable benchmarks
- No embedded hardware evaluation — inference latency not reported
- No XAI component — pure classification
- DistilBERT at 66M parameters is not truly lightweight for edge deployment
- Text encoding of numeric features may lose precision (floating point → string → token)

### SLM Relevance to Project
**★★★☆☆ — Moderately relevant**  
DDoSBERT validates the text-classification paradigm for network IDS and demonstrates that DistilBERT can achieve high accuracy on DDoS datasets. The first-text-based-transformer claim is a useful framing citation. However, the absence of slow-rate attacks is a significant gap — volumetric DDoS is trivially detectable by rate-thresholds, whereas slow-rate DoS requires semantic/temporal understanding. The mutual information feature selection methodology is directly applicable to the project's feature engineering phase. The paper's near-perfect accuracy numbers should be treated with scepticism.

---

## Paper SS-6: IDS-Agent — An LLM Agent for Intrusion Detection Systems

**Full citation:** Anonymous (Under review, ICLR 2025). "IDS-Agent: An LLM Agent for Intrusion Detection."

### Problem / Motivation
Traditional IDS are static classifiers — they classify individual flows but cannot reason across time, integrate external knowledge, or adapt their detection behaviour. This paper proposes the first LLM agent architecture for IDS, enabling dynamic reasoning, memory-augmented decision making, and sensitivity-tunable detection.

### System Architecture
**IDS-Agent** — a ReAct-style (Reasoning + Action) LLM agent:

**Action Space (sequential pipeline):**
1. **Data Extraction:** Extract relevant features from raw traffic
2. **Preprocessing:** Normalisation, encoding, feature transformation
3. **Classification:** Invoke one of 6 ML models — Random Forest (RF), K-Nearest Neighbour (KNN), Logistic Regression (LR), Decision Tree (DT), Multi-Layer Perceptron (MLP), Support Vector Classifier (SVC)
4. **Knowledge Retrieval:** Query external knowledge base (50 blog posts + 50 academic papers on IoT attacks)
5. **Long Memory Retrieval:** Access past session logs via similarity+recency-weighted retrieval
6. **Aggregation:** Combine ML classification scores + knowledge context + memory → final decision + explanation

**Memory System:**
- **Short-term memory:** Current session context (conversation history, recent decisions)
- **Long-term memory:** Cross-session persistence; retrieved via cosine similarity + recency weighting

**Knowledge Base:**
- 50 cybersecurity blogs + 50 academic papers on IoT attacks
- Enables RAG-style grounding of explanations

**LLM backbone:** GPT-4o (via API)

**Dataset:** ACI-IoT'23 and CIC-IoT'23 — IoT intrusion datasets including **Slowloris attacks** explicitly in ACI-IoT'23.

### Key Technical Details
- **Prompt-tunable sensitivity:** Agent's detection threshold can be adjusted via natural-language prompt instruction ("aggressive", "balanced", "conservative") — enabling security operators to tune false positive / false negative trade-off without retraining.
- **Zero-day detection mechanism:** Confidence-based uncertainty — if all 6 ML classifiers return low confidence scores (none exceeding threshold), the agent flags as zero-day / unknown attack. Recall=0.61 vs. ACGAN=0.41, RealNVP=0.47.
- **Inference latency:** Average 8.65 seconds per instance (including GPT-4o API call + ML classification + retrieval). Completely impractical for real-time IDS but demonstrates the architecture concept.
- **Multi-model voting:** Using 6 diverse classifiers provides robust ensemble decisions — the agent synthesises their outputs rather than relying on any single model.
- **Slowloris relevance:** ACI-IoT'23 includes Slowloris attacks — IDS-Agent successfully classifies them. This is the only paper in the corpus that directly tests an LLM-based system on Slowloris.

### Experimental Results
| Dataset | F1 Score | Notes |
|---|---|---|
| ACI-IoT'23 | **0.975** | Includes Slowloris |
| CIC-IoT'23 | **0.750** | More diverse, harder |

Zero-day detection recall: 0.61 (vs. 0.41 for ACGAN, 0.47 for RealNVP)  
Inference: 8.65s/instance (GPT-4o)

### Limitations
- 8.65s/instance inference is 4,000,000× slower than needed for real-time network IDS
- Depends on GPT-4o API — not deployable offline or on edge hardware
- Zero-day detection recall of 0.61 still misses nearly 40% of novel attacks
- Knowledge base is static — requires manual curation and updating
- CIC-IoT'23 F1=0.750 is moderate — the agent struggles with class diversity

### SLM Relevance to Project
**★★★★★ — Directly central (Architecture)**  
IDS-Agent is the clearest template for the project's proposed architecture. The separation of ML classification (fast, tabular) from LLM reasoning (slow, contextual) is the hybrid design pattern this project should adopt. The fact that Slowloris explicitly appears in the dataset — and is successfully detected — provides direct evidentiary support for extending this approach to slow-rate DoS. The project's SLM adaptation would replace GPT-4o with a small fine-tuned model to achieve deployable inference speed. The prompt-tunable sensitivity mechanism is directly adoptable for the project's explanation module. Highly likely to be a core citation.

---

## Paper SS-7: LLM-APTDS — APT Detection with LLMs and Strong Interpretability

**Full citation:** Yang, L., Ye, A., Liu, Y., Lu, W., & Huang, C. (2026). "LLM-APTDS: A high-precision advanced persistent threat detection system for imbalanced data based on large language models with strong interpretability." *Future Generation Computer Systems*, 178, 108315. Fujian Normal University, China.

### Problem / Motivation
Advanced Persistent Threats (APTs) involve multi-stage attack chains (reconnaissance → infiltration → lateral movement → data exfiltration) that evade traditional rule-based IDS. Existing provenance-graph-based detection methods either (a) rely on manually curated rules that cannot adapt to novel attacks, or (b) use ML black-boxes that provide no explainability. The paper proposes LLM-APTDS, combining LLMs' semantic reasoning with provenance graph analysis to achieve both high detection accuracy and interpretable attack reports.

### System Architecture
**LLM-APTDS** — two-stage pipeline:

**Stage 1: Malicious Entity Detection**
- **Input:** System event logs (processes, files, sockets — OS audit logs)
- **Preprocessing:**
  - Data cleaning: invalid data correction, missing value imputation, low-importance attribute removal
  - Data standardisation: unstructured logs → JSON key-value pairs; timestamp normalisation (ISO 8601); Unicode normalisation
- **Prompt engineering:** Few-shot prompts with task description + examples + constrained output space (Benign/Malicious only)
- **Dual-model collaborative detection:**
  - LLM 1 (Qwen-2.5-32B): fine-tuned with high-confidence positive (malicious) samples — emphasises precision
  - LLM 2 (DeepSeek-R1-14B): fine-tuned with balanced samples — improves boundary sensitivity
  - **Multi-Score (MS) fusion:** `MS = W₁·P(LLM₁) + W₂·P(LLM₂) + W₃·RF + W₄·e^{-λt}` where RF is request frequency and `e^{-λt}` is a time decay factor
  - Weights optimised via Bayesian optimisation; dynamic recalibration via sliding-window precision monitoring
  - Classification threshold T applied to MS → Benign / Malicious
- **Fine-tuning method:** LoRA (rank, alpha not specified) — AdamW optimiser; Qwen-2.5-32B: LR=1e-5, 3 epochs; DeepSeek-R1-14B: LR=3e-5, 4 epochs

**Stage 2: APT Process Explanation**
- **K-NN Graph Reconstruction:** Starting from detected malicious entities, build attack provenance subgraph
  - Adjacency matrix weighted by interaction count and temporal proximity: `A[i][j] = C_ij / (1 + |τ_i - τ_j|)`
  - Breadth-first search with depth-decay threshold `Θ = 0.7^depth × T`
- **ATT&CK Tactical Mapping:** Fine-tuned LLM 3 maps malicious subgraph to MITRE ATT&CK tactics/techniques (14 tactics: Initial Access, Persistence, Lateral Movement, Exfiltration, etc.)
- **Cyclic Enhancement Analysis:**
  - Round 1: LLM maps subgraph → tactic/technique set T = {t₁,...,tₙ}
  - Round 2: Chain of thought reasoning R = {r₁,...,rₙ} stored in long-term memory; Bi-directional GNN validates path consistency via cosine similarity
  - If path fails validation → rebuild strategy chain → iterate
  - Round 3: LLM generates structured attack report: technical identification + tactical correlation + defensive recommendations (tiered: urgent / high-priority / advisory)

**Dataset:** DARPA TC-E3 — Transparent Computing Engagement 3; three subsets:
- THEIA: 1,598,647 benign + 25,319 malicious events
- CADETS: 1,614,189 benign + 12,846 malicious events
- Trace: 3,220,594 benign + 68,082 malicious events

**Baselines compared:** LogGPT, THREATRACE, MAGIC, KAIROS, FLASH

### Key Technical Details
- **LLM model selection rationale:** DeepSeek-R1 chosen over Llama-3.2 and Qwen-2.5 for superior long-sequence dependency modeling — critical for multi-stage APT chains. Qwen-2.5 chosen for its cybersecurity-specific pre-training corpus.
- **Imbalanced data handling:** Complementary sample distribution — LLM 1 trained on high-confidence positives (precision-focused), LLM 2 on balanced data (recall-focused). Dual-model fusion handles class imbalance without oversampling.
- **Prompt ablation:** Example injection improves F1 by +2pp; full prompt engineering improves F1 by +3.5pp vs. no prompt.
- **Dual-model vs. single-model:** Collaborative framework improves F1 by up to +6.16% and recall by +5.4% over single-model baseline.
- **Incremental learning:** Model supports adding new attack types without full retraining — practically important for APT detection where attack patterns evolve.
- **Interpretability scoring:** Human evaluation by 5 cybersecurity engineers using two metrics:
  - Usefulness (0-20): technical effectiveness + tactical correlation + defensive utility
  - Readability (0-10): presentation structure + explanation clarity + report completeness

### Experimental Results

**Detection performance (LLM-APTDS vs. baselines):**

| Dataset | Precision | Recall | F1 |
|---|---|---|---|
| THEIA | **98.60%** | **99.82%** | **99.20%** |
| CADETS | **98.49%** | **99.77%** | **99.12%** |
| Trace | **97.65%** | **99.69%** | **98.65%** |

Compared to LogGPT: +2.9% F1, +2.4% Recall average.  
Compared to KAIROS (best graph-based baseline): slightly lower precision on Trace, but better F1 overall.

**Interpretability evaluation:**
| Metric | Mean Score | HIP (% exceeding threshold) |
|---|---|---|
| Usefulness (max 20) | 16.43 | 89.2% |
| Readability (max 10) | 7.27 | 87.4% |

**Performance overhead:**
| Dataset | Training Time | Detection Latency | Explanation Time |
|---|---|---|---|
| THEIA | 63h | 16h for 487K samples (12ms/sample) | 10 min (1 attack) |
| CADETS | 66h | 17h for 488K samples | 55 min (3 attacks) |
| Trace | 115h | 32h for 987K samples | 30 min (2 attacks) |

Memory: 8.6–12.4 GB during training; 3.2–4.1 GB during detection.  
Inference latency: **12ms per log event** — competitive with traditional rule-based detectors (which operate in microseconds for network flows but tens of milliseconds for log correlation).

### Limitations
- **Large models:** Qwen-2.5-32B and DeepSeek-R1-14B are too large for edge/embedded deployment — requires high-memory server infrastructure
- **Domain mismatch:** DARPA TC-E3 is system audit logs (OS-level process/file/socket events) — not network traffic features. Results do not transfer directly to network IDS
- **No slow-rate DoS:** APT attacks are multi-stage, long-dwell threats — fundamentally different from slow-rate DoS (which is a network-layer, application-layer attack on connection resources)
- **Training time:** 63–115h training is acceptable as a one-time cost but limits rapid adaptation
- **Explanation latency:** 10–55 minutes per attack chain is suitable for post-incident forensic analysis, not real-time alerting
- **Human evaluation subjectivity:** HIP criterion thresholds are manually defined; inter-rater reliability not reported
- **Proprietary data:** "Data availability: confidential" — reproducibility limited

### SLM Relevance to Project
**★★★☆☆ — Indirectly relevant (Explanation architecture)**  
LLM-APTDS is not directly relevant to slow-rate DoS detection or SLM deployment. The models are too large (32B+), the domain is OS logs (not network flows), and the attack type is APT (not DoS). However, the cyclic enhancement analysis framework — especially the three-layer explanation structure (technical identification → tactical correlation → defensive recommendations) — is a strong design template for the project's explanation module. The integration of MITRE ATT&CK as a structured knowledge base for grounding LLM explanations (preventing hallucination) is directly applicable. The Bi-GNN path validation concept is an interesting mechanism for ensuring causal consistency in explanations. Cite primarily in the "explanation architecture" section of the thesis.

---

## Cross-Paper Summary

### Papers by Relevance to Project

| Paper | Title (Short) | Core Contribution | Relevance |
|---|---|---|---|
| SS-1 | Houssel et al. (LLM for NIDS) | LLMs fail at direct classification; 7,000× latency gap; hallucinations; hybrid architecture needed | ★★★★★ |
| SS-2 | Salah et al. (SLMs on embedded) | TinyBERT/MiniLM on ARM embedded; 14× slowdown; sub-100MB feasibility | ★★★★★ |
| SS-3 | Lodh et al. (QLoRA + SHAP) | SHAP incompatible with LLM text inputs; Phi-2 energy efficiency; QLoRA method | ★★★★★ |
| SS-6 | IDS-Agent | First LLM agent IDS; Slowloris in dataset; hybrid ML+LLM+RAG architecture | ★★★★★ |
| SS-4 | SecurityBERT | 11M-param BERT achieves SOTA IoT IDS; PPFLE privacy encoding | ★★★★☆ |
| SS-5 | DDoSBERT | Text-classification paradigm for DDoS; MI feature selection | ★★★☆☆ |
| SS-7 | LLM-APTDS | Cyclic explanation with ATT&CK; dual-model collaboration; forensic-grade reports | ★★★☆☆ |

### Key Convergent Findings Across Supervisor-Shared Papers

**1. The Hybrid Architecture Consensus**  
Papers SS-1, SS-3, and SS-6 converge on the same conclusion: LLMs should not be used as end-to-end classifiers for network traffic. The correct architecture separates fast ML classification (tabular, sub-millisecond) from LLM-based reasoning/explanation (contextual, slower). This hybrid design is the project's strongest architectural foundation.

**2. Feasibility of SLMs on Constrained Hardware**  
SS-2 (TinyBERT 14M at 157ms on ARM Cortex-A53) and SS-4 (SecurityBERT 11M at 0.15s on commodity CPU) together demonstrate that sub-20M parameter models can deliver useful IDS performance on non-GPU hardware. The 14× embedded-to-desktop latency ratio from SS-2 provides a concrete planning factor for hardware feasibility analysis.

**3. The SHAP Problem for LLM Explanation**  
SS-3 definitively shows that SHAP cannot be applied to tokenised LLM text inputs for feature-level attribution. The project must use SHAP (or LIME/integrated gradients) on the tabular ML component, then translate those feature importances into natural-language explanations via the SLM — not apply attribution methods to the SLM's text inputs directly.

**4. Slow-Rate DoS Remains Unaddressed**  
None of the 7 papers directly evaluate slow-rate DoS detection via SLMs. SS-6 (IDS-Agent) is closest — Slowloris appears in ACI-IoT'23, and the agent achieves F1=0.975. This confirms the literature gap that justifies the project, and confirms that the ACI-IoT'23 dataset is the most appropriate benchmark to use.

**5. Explanation Quality Requires Structured Grounding**  
SS-1 (hallucinations), SS-7 (ATT&CK grounding), and SS-6 (RAG + knowledge retrieval) all point to the same need: LLM explanations must be grounded in verified external knowledge to prevent fabrication. For the project, this means the SLM explanation module needs either (a) RAG from a known attack knowledge base, or (b) constrained output templates aligned to attack taxonomy labels.

### Research Gaps Confirmed by This Folder

- **No SLM specifically designed for slow-rate DoS:** Despite SS-2's hardware feasibility work and SS-4's compact BERT work, no paper applies a sub-20M parameter model to Slowloris/RUDY/Slowread detection
- **No explainable slow-rate DoS IDS at edge:** The combination of slow-rate attack type + explanation + embedded deployment represents a clear research gap
- **Latency targets not met for slow-rate DoS:** SS-2's 157ms (TinyBERT) is reasonable for CAN bus (~1 kHz frame rate) but may need to be faster for application-layer DoS detection on web servers
- **SHAP-LLM incompatibility not solved:** SS-3 identifies the problem but proposes no solution — the project can contribute a novel feature-attribution-to-natural-language pipeline as an original contribution

---

*End of research_notes_supervisor_shared.md*
