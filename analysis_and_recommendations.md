# Research Synthesis & Architectural Recommendations
## SLM-Based Slow-Rate DoS Detection and Explanation for Resource-Constrained Devices
**Dataset:** CIC IIoT Dataset 2025  
**Architecture:** Encoder-based SLM (detection) + Decoder-based SLM (reasoning/explanation)  
**Compiled:** 2026-05-25  
**Based on:** 46 papers across 5 research folders

---

## Preamble: The Central Finding Across All Papers

Before addressing each question individually, one convergent conclusion emerges from nearly every paper in the corpus, and it must frame everything that follows:

> **LLMs and SLMs should never be used as standalone, end-to-end network classifiers for real-time traffic detection. The correct architecture separates fast ML/encoder detection from slower decoder-based reasoning. Every paper that tests this separation confirms it outperforms both pure-ML (on explanation) and pure-LLM (on accuracy and speed).**

This finding is supported by SS-1 (7,000× latency gap between LLaMA3-8B and Random Forest), F2-4 (LLMs 4 orders of magnitude slower than XGBoost), F3-1 (LLM precision ~50% — worse than random for zero-shot flow classification), and F4-3 (explicit endorsement of hybrid architecture). Your proposed design — encoder for detection, decoder for explanation — is architecturally correct and well-supported by the literature.

---

## 1. Best Lightweight Detection Methodology

### Recommended Approach: Two-Stage Hybrid — ML Baseline + Encoder SLM

The optimal detection pipeline is a two-stage architecture where a lightweight classical ML classifier acts as a first-pass triage, and a compact encoder-based SLM (BERT-class) provides the classification decision that feeds the explanation stage. Neither stage alone is sufficient.

**Stage 1 — ML baseline (always-on, real-time):**  
A Random Forest or XGBoost classifier trained on extracted flow features serves as the primary real-time classifier. Multiple papers confirm RF achieves 96–99.5% accuracy on slow-rate DoS tasks with sub-millisecond inference. Paper F1-8 (P4+RF) achieves 98.28% on Slowloris with 1.73s testing time. Paper F1-9 (FRE) achieves 99.52% with XGB and 99.80% with the FRE hybrid. Paper F1-7 (Reed et al.) achieves 95.9% with just 2 features (packet length + IAT) and a decision tree on IoT hardware. These results confirm that for the actual detection decision, classical ML is superior in accuracy and speed simultaneously.

**Stage 2 — Encoder SLM (for confident classification + embedding):**  
The encoder SLM (SecurityBERT-style compact BERT) takes the verbalized flow features and produces a classification embedding that also feeds the decoder's explanation stage. Paper SS-4 (SecurityBERT) demonstrates that an 11M-parameter BERT achieves 98.2% accuracy on IoT IDS — outperforming CNN-LSTM, DNN, and GAN-Transformer — at 0.15s inference on a commodity CPU. Paper SS-2 (Salah et al.) demonstrates TinyBERT (14M) at 157ms on ARM Cortex-A53 embedded hardware. Paper F2-1 (Bui et al.) definitively shows a fine-tuned BERT-class model outperforms even Mistral-7B by ~5% on classification tasks — larger models give no benefit for this task.

**Key methodology finding from Paper F1-4 (Al-Shukaili et al.):** For slow-rate DoS specifically, temporal modeling matters. A plain feedforward DNN underperforms when temporal patterns in incomplete requests, connection durations, and periodic keepalives are not captured. This suggests the encoder SLM should operate on a window of recent flows, not just a single flow in isolation.

**What to avoid:** Do not attempt to use the decoder SLM (7B-class model) for real-time detection decisions. The ~14,000µs LLM inference latency (SS-1) versus 2.03µs for Random Forest makes real-time deployment impossible. Do not use transformer autocoders (BERT-class) for packet-level analysis — they require flow-level aggregation first (F4-1 survey confirms single-packet analysis is fundamentally insufficient).

**For slow-rate DoS specifically**, the detection signature emerges over seconds to minutes. A minimum observation window of 15–30 seconds is recommended (consistent with Paper F1-3 slowTrack's W=15s window and Paper F1-6 Bocu et al.'s 30s intervals). The encoder SLM should receive a window-aggregated feature vector, not individual packet attributes.

---

## 2. Most Suitable Preprocessing Pipeline

### Step-by-Step Recommended Pipeline for CIC IIoT Dataset 2025

**Step 1 — Traffic capture and flow generation:**  
Use **Zeek** (not CICFlowMeter) to generate flow-level statistics. Paper F2-4 (Mehavilla et al., 2026) explicitly uses Zeek and shows that features including `proto`, `duration`, `orig_bytes`, `resp_bytes`, `conn_state`, `history`, `orig_pkts`, `orig_ip_bytes`, `resp_pkts`, `resp_ip_bytes` are sufficient for high-accuracy classification. Zeek produces the `conn.log` file natively. Alternatively, CICFlowMeter generates the 83-feature CIC-IDS standard feature set. For slow-rate DoS, Zeek is preferable because it captures `conn_state` and `history` fields that encode connection state transitions — critical for Slowloris (never-completed headers) and RUDY (never-completed POST body).

**Step 2 — Data cleaning:**  
- Replace infinite values with NaN, then apply mean imputation per feature (F1-4)  
- Remove duplicate rows and any flows shorter than 1 second (too short to be slow-rate DoS)
- Drop identifier columns (source IP, destination IP, source port) that would cause overfitting or data leakage; keep protocol, destination port, and connection state as categorical
- Handle class imbalance by applying SMOTE (Synthetic Minority Oversampling Technique) exclusively on the training split, never on validation or test (F1-4 clearly specifies this constraint)

**Step 3 — Feature normalisation:**  
- Apply **MinMax scaling** ([0, 1]) for BERT-class encoder inputs — consistent with SS-4 (SecurityBERT) and F3-7 (LSTM with MinMax)
- Apply **Z-score standardisation** for classical ML classifiers (XGBoost, RF) — consistent with F3-3 (From Flows to Words)
- Fit scalers on training data only; apply same fitted scaler to validation and test sets

**Step 4 — Temporal windowing (critical for slow-rate DoS):**  
Aggregate flows into time windows of W=15–30 seconds. Within each window, compute per-IP or per-flow statistics:
- Count of incomplete requests (Slowloris: incomplete GET; RUDY: incomplete POST)
- Mean and variance of flow duration
- Mean and variance of inter-packet interval
- TCP window size statistics (min, max, mean — Slowread signature: consistently near-zero)
- Ratio of HTTP 2xx to 4xx responses
- Active TCP socket count on server port
- Total bytes sent vs. Content-Length declared ratio

This approach is directly validated by Paper F1-3 (slowTrack's 8 behavioral parameters at W=15s, F1=99.89%) and Paper F1-6 (FLD-LRDDoS with 30s DPSF windows, 98.79% accuracy).

**Step 5 — Feature verbalization for SLM input:**  
Convert the cleaned, normalised feature vector to a text string for SLM input using one of two validated formats:

*Format A — Space-separated (simple, from Zeek conn.log, F2-4):*  
`"tcp - 0.398803 0 0 REJ T T 0 Sr 1 60 1 40"`

*Format B — Key-value pairs (more interpretable, from F3-1 and F3-4):*  
`"FLOW_DURATION: 127.5s TCP_WINDOW_MIN: 0 HEADER_COMPLETE: False PKT_RATE: 0.03 BYTE_RATE: 12.4 CONN_STATE: S1"`

Format B is strongly recommended for slow-rate DoS because it preserves feature semantics for the downstream explanation stage, and Paper F3-4 (eX-NIDS) demonstrates that feature-named inputs dramatically reduce hallucination in the decoder. Format A is acceptable for the encoder-only classification task where speed is prioritised.

**Step 6 — Boolean domain flag injection (for decoder prompts only):**  
Following Paper F3-3 (From Flows to Words), prepend boolean flags derived from domain knowledge before the feature values in the decoder prompt:
- `INCOMPLETE_HEADER: True` (if HTTP header was never completed)
- `SLOW_RATE: True` (if packet rate < 1 pkt/s sustained over 30s)
- `WINDOW_EXHAUSTION: True` (if TCP window size < 64 bytes)
- `LONG_CONNECTION: True` (if flow duration > 60s with minimal data)
- `PARTIAL_BODY: True` (if Content-Length >> actual body bytes transferred)

These flags directly encode slow-rate DoS attack semantics into the prompt and prevent the decoder from having to infer them from raw numbers.

---

## 3. Best Feature Selection/Extraction for Resource-Constrained Environments

### Primary Recommendation: Mutual Information + Domain-Guided Manual Selection

**From the evidence:**  
Paper SS-5 (DDoSBERT) shows Mutual Information (MI) consistently outperforms Pearson correlation and univariate statistical tests for network feature selection, because MI captures non-linear dependencies between features and attack labels — particularly important for slow-rate DoS where the attack signal is encoded in subtle temporal patterns rather than obvious value extremes.

Paper F1-4 (Al-Shukaili et al.) shows Wrapper-based RFE (Recursive Feature Elimination with RF) achieves better results than filter-based SelectKBest, but at significantly higher computational cost. For resource-constrained deployment, **MI filtering is the practical choice for initial selection**, followed by expert review to ensure slow-rate-specific features are retained even if MI ranks them lower on a balanced dataset.

**Recommended feature set for slow-rate DoS on CIC IIoT 2025 (12–15 features):**

*Temporal and rate features (most discriminative for slow-rate DoS):*
1. `flow_duration` — abnormally long for all slow-rate types
2. `pkt_rate` (pkts/s) — extremely low (<1 for Slowloris/RUDY)
3. `byte_rate` (bytes/s) — near-zero for slow-rate attacks
4. `iat_mean` (inter-arrival time, mean) — large and irregular for slow attacks
5. `iat_std` (inter-arrival time, std deviation) — high variance for Slowloris keepalive pattern

*TCP/connection state features:*
6. `tcp_window_size_min` — near-zero specifically for Slowread (discriminates from Slowloris)
7. `fin_flag_count` — low for slow attacks (connections never cleanly closed)
8. `syn_flag_count` — per F3-6 (on-device) and F1-8 (P4)
9. `conn_state` / `tcp_state` — S1/S2 (established but incomplete) vs. SF (successful)

*Volume and payload features:*
10. `orig_bytes` / `total_bytes` — very low for slow-rate attacks
11. `header_completeness` (ratio: headers sent / expected headers) — Slowloris specific
12. `content_length_ratio` (actual body / declared Content-Length) — RUDY specific
13. `protocol` (TCP=6) — categorical, encoded
14. `dst_port` (80/443 for HTTP/HTTPS DoS) — important for IoT context

*Optional but valuable (if budget allows):*
15. `entropy_src_ports` (Shannon entropy of source ports in time window) — from Paper F1-9

**What Paper F1-7 (Reed et al.) demonstrates about minimal features:**  
Even with just 2 features — `packet_length` (IG=0.84) and `inter-arrival time` (IG=0.55) — a decision tree achieves 95.9% accuracy on slow-rate HTTP DoS in an IoT network, with the dataset reduced from 396MB to 0.52MB (99.8% reduction). This is the absolute minimum viable feature set for genuinely ultra-constrained devices such as Raspberry Pi-class sensors with no preprocessing budget. For the SLM pipeline however, 12–15 features are appropriate.

**What to avoid:**  
Do not use all 83 CICFlowMeter features. Papers F1-4 and F2-4 both confirm diminishing returns beyond 20–50 features, and the verbalization length of 83 features would exceed the context window of compact SLMs. Do not use source and destination IP addresses as features — they cause identity-based overfitting and do not generalise across deployment environments.

---

## 4. Most Efficient Encoder SLM Architecture for Detection

### Recommended: SecurityBERT-style Compact BERT (11–22M parameters)

The encoder SLM's job is to receive verbalized flow features and produce (a) a classification label (attack type vs. benign) and (b) a contextual embedding that can be passed to the decoder for explanation. The encoder must be fast enough for near-real-time operation on constrained hardware.

**Top candidate: SecurityBERT architecture (SS-4 — Ferrag et al., 2024)**  
- 15 transformer layers, 11M parameters, 16.7MB model size
- Custom ByteLevelBPE tokenizer with vocabulary size 5,000 (suited to network data)
- PPFLE encoding: `H(column_name + "$" + value)` — fixed-length hashed representation per feature
- Result: 98.2% accuracy on Edge-IIoTset (highest reported)
- Inference: 0.15s on commodity CPU, 0.016s on GPU
- The PPFLE encoding eliminates variable-length verbalization problems and provides a fixed-length token sequence regardless of feature value magnitude

**Second candidate: TinyBERT (14M, 56MB) — best for embedded hardware (SS-2)**  
- F1=0.847 on CAN-MIRGU dataset on ARM Cortex-A53 at 157ms/window (~6.3 FPS)
- The 14× slowdown rule (desktop to embedded) gives 157ms embedded inference from ~11ms desktop
- 56MB fits comfortably in 6GB LPDDR4; even manageable in 2GB configurations
- Training via BERT task-specific knowledge distillation from a larger DistilBERT teacher

**Third candidate: MiniLM-L6 (22M, 86MB) — best F1 balance (SS-2)**  
- F1=0.883, Accuracy=92.4%, 336ms/window on embedded ARM hardware
- Higher accuracy than TinyBERT with reasonable memory overhead
- Good choice if 300–400ms latency is acceptable for the deployment context

**Architecture modifications for slow-rate DoS:**

The standard BERT sequence classification architecture requires one modification for temporal slow-rate detection. Rather than classifying a single flow record, the encoder should operate on a **window of N recent flows** (N=3–5, consistent with F3-7 optimal window_size=3). Each flow in the window is verbalized as a token sequence, flows are separated by a `[SEP]` token, and the final `[CLS]` embedding is used for classification. This window-level encoding directly captures the temporal evolution of slow-rate attack patterns.

Alternative: Use a simpler architecture from Paper F3-6 (Chatzimiltis et al.) — a 32-unit LSTM with (window=3, features=14) input achieving F1=0.98 at 0.03ms inference on Intel i7 CPU. The LSTM is far faster than BERT but produces less interpretable embeddings. For the dual SLM architecture, the BERT encoder is preferred because its `[CLS]` embedding is directly usable as a rich contextual representation for decoder prompting.

**Input format for encoder:**  
Use Format B key-value pairs (from Section 2, Step 5) for human-readable token alignment, but limit to the 12–15 selected features. This keeps the input sequence under 256 tokens — within any compact BERT's context window — enabling batch processing.

**Fine-tuning approach:**  
Fine-tune from a publicly available checkpoint. Priority order:
1. SecurityBERT (if pre-trained checkpoint is released) — cybersecurity-domain pre-training advantage
2. DistilBERT-base-uncased → distil to TinyBERT size via structured pruning
3. BERT-base-uncased → fine-tune with classification head replacement

From Paper F2-9 (SecureBERT + Llama-2 paper): domain-specific pre-training reduces required fine-tuning examples and improves convergence speed significantly. Starting from a security-pretrained checkpoint is preferable to starting from general BERT-base.

---

## 5. Best Lightweight Decoder SLM for Reasoning and Explanation

### Recommended Architecture by Hardware Tier

The decoder SLM receives a structured prompt containing the encoder's classification decision, SHAP/LIME feature importance scores, flow feature values, and domain context. It generates a human-readable explanation, attack type identification, and recommended response action.

**Tier 1 — Moderately constrained (8–16GB RAM, GPU not required):**  
**Phi-2 (2.7B parameters)** — Paper SS-3 (Lodh et al.) shows Phi-2 is the most energy-efficient model at 0.173 kg CO2 per training run (vs. 0.304 kg for LLaMA2-7B). Phi-2 was Microsoft's smallest competitive reasoning model and achieves strong performance on structured text tasks relative to its size. Fine-tuned via QLoRA (4-bit NF4), Phi-2 reaches F1=1.00 on InSDN at 350K samples. For explanation quality at this parameter count, the 16-factor SSRP framework (Section 6) is essential to compensate for the model's limited reasoning depth.

**Tier 2 — Highly constrained (4–8GB RAM, ARM-class edge device):**  
**LLaMA 3.2-3B** — Paper F3-6 (Rethinking On-Device) shows LLaMA3.2-3B achieves macro-F1=0.75 with few-shot+CoT on 6-class IoT DDoS detection. Critically, 3B+ parameters are the minimum for reliable TCP subtype discrimination — distinguishing Slowloris (incomplete headers) from RUDY (slow POST body) from Slowread (tiny TCP window) requires at least 3B. For explanation-only (not classification), quality is acceptable at 3B with proper prompting.

**Tier 3 — Ultra-constrained (1–4GB RAM, Raspberry Pi-class):**  
**Qwen 2.5-3B or TinyLlama-1.1B** with grammar-constrained decoding (GBNF, from F3-3). However, note the critical finding from Paper F3-3: Qwen 2.5-7B with domain flags achieves macro-F1=0.783 on UNSW-NB15, but TinyLlama-1.1B with the same flags achieves only 0.31 — a 60% relative drop. For genuinely 1B-class models, explanation quality degrades severely. The minimum useful decoder for security explanations is 2–3B parameters.

**For production recommendation:**  
**Gemma3-4B** (from F3-6) achieves the best balance — macro-F1=0.85 with few-shot+CoT on IoT DDoS — at 4B parameters. After 4-bit quantisation this occupies approximately 2.5GB RAM. Running via llama.cpp on ARM Cortex-A55 or similar is feasible at 1–3 second per explanation, which is acceptable for async post-detection explanation (the explanation is generated after the detection decision, not blocking it).

**What Paper F3-6 demonstrates conclusively:**  
Chain-of-thought prompting alone (without few-shot exemplars) is WORSE than no prompting for all 4B-and-below models. Gemma3-4B with CoT-only achieves F1=0.05 — catastrophic collapse — while the same model with few-shot achieves F1=0.85. This confirms that the decoder must receive concrete exemplars of slow-rate attack explanations, not abstract reasoning instructions. This is the single most practically important design constraint for the decoder component.

---

## 6. How Both Models Should Interact Efficiently

### Recommended: Async Two-Stage Pipeline with Structured Intermediate Representation

**Architecture overview:**
```
[Network Traffic]
        ↓
[Feature Extraction — Zeek/CICFlowMeter]
        ↓
[Preprocessing — Normalise, Window, Verbalize]
        ↓
[Stage 1: Fast ML Baseline — XGBoost/RF]  ← always-on, synchronous
        ↓ (if alert raised)
[Stage 2: Encoder SLM — TinyBERT/MiniLM]  ← synchronous, near-RT
        ↓ (classification + [CLS] embedding)
[XAI Bridge — SHAP on encoder or ML model]  ← synchronous, <1s
        ↓ (ranked feature importance JSON)
[Structured Intermediate Representation (SIR)]  ← combines all
        ↓
[Stage 3: Decoder SLM — Gemma3-4B/Phi-2]  ← asynchronous, non-RT
        ↓
[Human-Readable Explanation + Mitigation Recommendation]
```

**The Structured Intermediate Representation (SIR)** is the critical interface between encoder and decoder. This concept is validated by Paper F4-4 (LLM for Explain paper) which shows that how ML outputs are presented to the LLM matters as much as which LLM is used. The SIR should be a JSON-structured text block containing:

```json
{
  "detection_label": "Slowloris",
  "confidence": 0.97,
  "top_features": [
    {"name": "flow_duration", "value": "127.5s", "importance": 0.42, "direction": "high"},
    {"name": "pkt_rate", "value": "0.03 pkt/s", "importance": 0.38, "direction": "low"},
    {"name": "header_completeness", "value": "0.12", "importance": 0.31, "direction": "low"},
    {"name": "tcp_window_min", "value": "0", "importance": 0.11, "direction": "low"}
  ],
  "flow_window_summary": "Flow observed over 15s window: 3 packets, 245 bytes, connection state S1 (established, no data)",
  "source_context": {"dst_port": 80, "protocol": "TCP", "conn_state": "S1"},
  "domain_flags": {
    "incomplete_header": true,
    "slow_rate": true,
    "window_exhaustion": false
  }
}
```

This SIR is then injected into the decoder prompt alongside the domain knowledge context (attack descriptions, normal behavior norms, protocol definitions). Paper F3-4 (eX-NIDS) demonstrates this augmented-prompt approach improves explanation correctness from ~36% to 80%+ and brings Feature Consistency to 100%.

**Timing separation:**  
The encoder and ML classifier run synchronously and produce the detection decision within milliseconds. The XAI computation (SHAP on the ML model or encoder) runs within 1–2 seconds. The decoder runs asynchronously in a separate process or thread, not blocking the detection pipeline. From Paper F3-7 (Chatzimiltis et al.): "rApp is deliberately Non-RT because LLM inference latency exceeds Near-RT RIC timing budget; rApp is advisory, not control-path." This architectural principle applies directly — the decoder explanation is an advisory output for the security analyst, not an inline gate for traffic blocking decisions.

**Memory efficiency:**  
Both models should not be loaded simultaneously in RAM if memory is very constrained. Use a producer-consumer queue: the encoder produces SIR objects, places them in a queue, and the decoder consumes them in batch when resources allow. Batch inference on the decoder reduces per-sample latency by 2–4× (from Paper F2-1: BERT batch=1 gives 25fps; batch=20 gives 181fps — 7× improvement).

---

## 7. Best Training and Fine-Tuning Strategy

### Encoder SLM Training

**Dataset split:** Stratified 80/20 train/test split with a further 10% of training held out as validation for threshold calibration. SMOTE applied on training split only.

**Training protocol:**
- Start from SecurityBERT or DistilBERT-base-uncased checkpoint
- Replace classification head with a new linear layer matching the number of classes (Slowloris, RUDY, Slowread, Benign, and any other attack types in CIC IIoT 2025)
- Fine-tune all encoder layers (not just the head) — from Paper F2-1, fine-tuning all layers outperforms head-only fine-tuning by ~3–5%
- Optimizer: AdamW with weight decay 0.01
- Learning rate: 2e-5 with linear warmup (10% of steps) and linear decay
- Batch size: 32–64 (maximise within GPU VRAM budget)
- Epochs: 5–10 with early stopping (patience=3 on validation F1)
- Loss: Cross-entropy with class weight inverse proportional to frequency (handles class imbalance alongside SMOTE)

**Preventing catastrophic forgetting (Paper F3-7 insight):**  
When fine-tuning on CIC IIoT 2025 data, include 30% past data (e.g., from CIC-DoS2017 or CICIoT2023) in the training mix. Without this, per-day F1 on day 4 collapses to 0.36; with 30% past data ratio, all days maintain F1 ≥ 0.96.

**Threshold calibration (Paper F3-3 insight):**  
Train on 80% of data; calibrate the classification threshold (τ*) on the 10% validation slice by maximising macro-F1 across all attack classes; freeze τ* for final test evaluation. Do not use the default 0.5 threshold — it is rarely optimal for imbalanced security datasets.

### Decoder SLM Training (QLoRA)

**Model selection:** Gemma3-4B (preferred for explanation quality) or Phi-2 (preferred for energy efficiency). Use the instruct-tuned variant, not the base model — instruction tuning greatly improves zero-shot instruction following (from Paper F2-11).

**Fine-tuning method: QLoRA with 4-bit NF4 quantisation**  
Validated by Papers SS-3, F2-1, F2-5, F2-11, F2-12 across multiple security tasks:
- Base model: 4-bit NF4 quantisation via bitsandbytes (frozen during training)
- LoRA adapters: inserted into Q, K, V, and O projection matrices of all attention layers
- LoRA rank: r=8 (start here; increase to r=16 if validation loss plateaus)
- LoRA alpha: α=32 (scaling factor; keep at 4× rank)
- LoRA dropout: 0.05
- Optimizer: paged AdamW 32-bit
- Learning rate: 2e-4 with cosine schedule
- Batch size: 4 (micro) × 8 (gradient accumulation) = effective batch 32
- Epochs: 3–5 (Paper SS-3 finds convergence at 3–4 epochs across all models)

**Training data format — Instruction tuning:**  
Create a dataset of (instruction, input, response) triplets where:
- `instruction`: "You are a cybersecurity analyst. Based on the following network flow detection result and feature importance scores, explain in clear language why this traffic was classified as a slow-rate DoS attack, which specific attack subtype it is, and what mitigation steps should be taken."
- `input`: The SIR JSON block (from Section 6) + protocol knowledge context + normal behavior norms
- `response`: A structured explanation following the Obs→Evidence→Conclusion→Mitigation schema

**Generating training data:** Use a teacher LLM (GPT-4, DeepSeek-R1, or Claude) to generate labelled explanations for 1,000–5,000 SIR examples from your dataset. This is the offline knowledge distillation approach from Paper F3-6 (Rethinking On-Device), and it is highly efficient — the teacher generates the CoT-labelled knowledge base once, the student SLM learns from it.

**Few-shot exemplar library:**  
After fine-tuning, create a library of 3–5 high-quality exemplar explanations per attack type (Slowloris, RUDY, Slowread). These are retrieved at inference time using an XGBoost retriever (not BERT embeddings — Paper F3-6 conclusively shows XGBoost retriever outperforms BERT embeddings for numeric network feature similarity). Inject the 1–2 most similar exemplars into the decoder prompt at inference time. This dramatically improves explanation quality for edge cases.

---

## 8. Optimisation Methods for Resource-Constrained Deployment

### Encoder Optimisation

**INT8 Post-Training Quantisation:**  
Apply static INT8 quantisation to the fine-tuned encoder SLM using ONNX Runtime or PyTorch's quantisation toolkit. For BERT-class models, INT8 quantisation reduces model size by ~4× with less than 1% accuracy loss. The 11M SecurityBERT at 16.7MB becomes approximately 4.2MB in INT8 — suitable for microcontroller-class deployments.

**ONNX Export:**  
Export the fine-tuned encoder to ONNX format for cross-platform deployment. ONNX Runtime is hardware-agnostic and achieves 2–4× speedup over native PyTorch inference on CPU-only edge devices. From Paper SS-2: the NXP i.MX 8M Plus (ARM Cortex-A53) achieves ~157ms for TinyBERT in standard PyTorch; ONNX Runtime would reduce this to ~40–80ms.

**Structured Pruning:**  
Apply attention head pruning (remove 30–50% of attention heads with lowest importance scores) followed by fine-tuning recovery. From Paper F2-7 (Lightweight LLMs paper): 75% memory reduction at ~2–5% accuracy drop is achievable with knowledge distillation. For IoT deployment where 56MB TinyBERT is still borderline, structured pruning + distillation can reach sub-10MB models.

**Knowledge Distillation (Encoder):**  
Train a smaller student encoder (e.g., 3-layer BERT with 64 hidden dimensions, approximately 5M parameters) using the 15-layer SecurityBERT as teacher. The student learns to match both the teacher's logits (soft targets) and intermediate layer representations. This produces a model significantly smaller than TinyBERT with comparable accuracy on the narrow slow-rate DoS classification task.

### Decoder Optimisation

**4-bit GGUF Quantisation via llama.cpp:**  
After QLoRA fine-tuning, merge the LoRA adapters into the base model, then export to GGUF format and apply Q4_K_M quantisation. This is the most practical approach for CPU-only edge inference:
- Gemma3-4B: ~2.5GB in Q4_K_M (from ~8GB in FP16)
- Phi-2 2.7B: ~1.7GB in Q4_K_M
- LLaMA3.2-3B: ~2.0GB in Q4_K_M
All three fit within 4GB RAM budgets with operating system overhead included.

**Speculative Decoding:**  
Pair the decoder with a 2–3× smaller draft model for speculative decoding. The draft model generates tokens quickly; the full model validates them in parallel. On CPU-only inference, speculative decoding provides 2–3× throughput improvement for explanation generation tasks (where token acceptance rate is typically 70–80%).

**Inference via Ollama:**  
For the decoder, run via Ollama (which wraps llama.cpp) for easy model management and REST API access. Paper F3-6 explicitly uses Ollama for all on-device LLM evaluation. This eliminates the need to write custom inference code and provides built-in quantisation, context management, and concurrency support.

**Batch Explanation Processing:**  
Queue multiple SIR objects and process them in batches during lower-utilisation periods (night-time or scheduled maintenance windows). For slow-rate DoS, where attacks unfold over minutes, the 1–3s explanation generation delay is well within acceptable bounds — the detection happens in real time, the explanation is delivered asynchronously.

**Summary of recommended optimisation stack:**

| Component | Model | Technique | Final Size | Inference Speed |
|---|---|---|---|---|
| Encoder SLM | TinyBERT-14M or SecurityBERT-11M | INT8 ONNX | ~4–14MB | 40–80ms on ARM |
| Decoder SLM | Gemma3-4B or Phi-2 | QLoRA → Q4_K_M GGUF | ~1.7–2.5GB | 1–3s/explanation |
| ML Baseline | XGBoost or RF | Scikit-learn serialised | <5MB | <1ms |
| XAI Bridge | SHAP (Tree SHAP for XGBoost) | Native Python | Negligible | <100ms |

---

## 9. Suitable Evaluation Metrics

### Detection Metrics (Encoder + ML Baseline)

**Primary:**
- **Macro-F1** — unweighted average across all attack classes; most important for imbalanced datasets where minority slow-rate classes matter equally to majority benign class
- **Per-class F1** — separate F1 for Slowloris, RUDY, Slowread, Benign (and any other CIC IIoT 2025 classes); essential to show model does not sacrifice rare attack detection
- **Precision and Recall per class** — precision-recall tradeoff is security-critical: high recall = low false negatives (don't miss attacks); high precision = low false positives (don't alarm operators unnecessarily)

**Secondary:**
- **AUC-ROC** — area under the receiver operating characteristic curve; threshold-independent performance measure
- **False Positive Rate (FPR)** — false alarm rate; critical for operator trust
- **False Negative Rate (FNR)** — missed attack rate; critical for security coverage

**Resource metrics:**
- **Inference latency** (ms/sample, both single-sample and batched)
- **Peak RAM usage** (MB, during inference)
- **Model disk size** (MB, post-quantisation)
- **Energy consumption** (mWh or kg CO2 per training run, using CodeCarbon library — directly from Paper SS-3)
- **Throughput** (samples/second at batch inference)

### Explanation Quality Metrics (Decoder SLM)

This is the most under-standardised area in the literature. Two frameworks from the papers provide concrete metrics:

**Framework 1 — eX-NIDS Three-Metric Evaluation (F3-4, Houssel et al.):**
- **Correctness** (0–100%): Does the explanation correctly identify the attack type and its mechanism? Scored by domain expert annotation.
- **Feature Consistency** (0–100%): Does the explanation reference only features present in the input SIR? Automated check — any feature name in the explanation text must appear in the SIR feature list.
- **Factual Consistency** (0–100%): Are protocol numbers, port assignments, and TCP flag descriptions factually accurate? Checked against a ground-truth protocol dictionary.

**Framework 2 — SSRP Four-Metric Evaluation (F3-9, Zhou et al.) with strong inter-rater reliability (κ > 0.80):**
- **Evidence Grounding Accuracy**: Are flow feature values explicitly cited in the reasoning (e.g., "pkt_rate of 0.03 pkt/s indicates low-rate behaviour")?
- **Reasoning Faithfulness**: Are all conclusions traceable to provided features — no hallucinated context?
- **Structure Compliance**: Does the explanation follow the Obs→Evidence→Conclusion→Mitigation schema?
- **Attack Taxonomy Alignment**: Is the attack type correctly named and described per the standard taxonomy (Slowloris, RUDY, Slowread — from Papers F1-10, F1-11)?

**Recommended combined evaluation protocol:**
1. Sample 50–100 explanation outputs from the decoder (mix of attack types)
2. Score each on all 7 metrics above (3 from eX-NIDS + 4 from SSRP)
3. Compute inter-rater reliability (Cohen's κ) using two independent evaluators
4. Report mean per metric and High Interpretability Percentage (HIP) — proportion of explanations exceeding satisfactory threshold (from SS-7)
5. Supplement with automated scores: ROUGE-L against teacher-generated reference explanations; BERTScore for semantic similarity

**Additional metrics for the CIC IIoT 2025 context:**
- **Zero-day generalisation**: train on Slowloris/RUDY, test on Slowread (withheld from training) — measure F1 degradation
- **Adversarial robustness**: test against randomised User-Agent Slowloris (from F1-3 slowTrack robustness testing), SOCKS5-proxied attacks, and socket count variations

---

## 10. Recommended Tools, Frameworks, and Libraries

### Traffic Capture and Feature Extraction
- **Zeek IDS** (https://zeek.org) — preferred for generating structured conn.log with connection state history; outputs directly usable for SLM verbalization
- **CICFlowMeter** — alternative if 83-feature CIC standard format is required for comparability with prior work on CIC IIoT 2025
- **Scapy** (Python) — for custom packet-level feature extraction if needed; used in Papers F1-3 and F1-8
- **tshark / Wireshark** — for raw pcap capture and protocol-level filtering
- **Tranalyzer** (Paper F2-2) — alternative to CICFlowMeter with 71-feature output, better at flow boundary detection

### SLM Training and Inference
- **HuggingFace Transformers** (https://huggingface.co/transformers) — universal backbone for encoder SLM (BERT-class models, SecurityBERT, DistilBERT, TinyBERT)
- **HuggingFace PEFT** (https://github.com/huggingface/peft) — LoRA/QLoRA implementation; used in Papers F2-1, F2-5, F2-11, F2-12, SS-3
- **bitsandbytes** — 4-bit NF4 quantisation for QLoRA; essential for decoder fine-tuning on consumer GPU (single 24GB GPU sufficient for 7B models)
- **llama.cpp** + **GGUF format** — CPU-optimised quantised inference for decoder deployment; the de facto standard for edge LLM inference (Papers F3-3, F3-6)
- **Ollama** (https://ollama.com) — wraps llama.cpp with REST API; simplest deployment interface for decoder (used in Paper F3-6)
- **ONNX Runtime** — cross-platform accelerated inference for encoder SLM; 2–4× speedup over PyTorch CPU inference

### XAI and Feature Attribution
- **SHAP** (https://shap.readthedocs.io) — Tree SHAP for XGBoost/RF (fast, exact); Kernel SHAP for neural models (approximate); validated in Papers F2-3, F2-4, F3-2, F3-7
- **LIME** (https://github.com/marcotcr/lime) — local surrogate explanation; particularly useful for instance-level explanations; used in Papers F3-2, F3-7
- **Captum** (PyTorch) — Integrated Gradients for encoder SLM attribution; validated in Paper F2-1 for BERT-class models

### ML Baseline and Classical Methods
- **scikit-learn** — RF, XGBoost, SVM, decision trees, SMOTE (via imbalanced-learn), cross-validation, SHAP integration
- **XGBoost** (standalone) — for the primary ML baseline classifier
- **imbalanced-learn** — SMOTE and ADASYN for class imbalance handling

### Experiment Tracking and Optimisation
- **Weights & Biases (wandb)** — experiment tracking, hyperparameter search, model comparison (used in Papers F2-11, F2-12)
- **CodeCarbon** — CO2/energy consumption tracking during training (directly used in Paper SS-3; highly recommended for arguing SLM efficiency)
- **Optuna** or **Ray Tune** — hyperparameter optimisation for QLoRA configuration (rank, alpha, learning rate)

### RAG and Knowledge Retrieval (optional but recommended)
- **ChromaDB** — lightweight local vector database; used in Papers F2-1, F3-8 (IDS-Agent)
- **FAISS** — Facebook AI Similarity Search; faster for larger knowledge bases
- **LangChain** — RAG orchestration framework if knowledge base retrieval is integrated

### Evaluation and Metrics
- **py-readability-metrics** — Flesch Reading Ease, Gunning Fog for explanation readability (used in F3-7)
- **evaluate** (HuggingFace) — ROUGE, BERTScore, exact match for automated explanation evaluation
- **scipy.stats** — Friedman test for statistical significance across methods (from SS-7 LLM-APTDS)

---

## 11. Best Practical Implementation Workflow for Low-Resource Devices

### Phase 1 — Preparation (Weeks 1–2)

**1a. Dataset preparation:**  
Download CIC IIoT Dataset 2025. Install Zeek and run it over the dataset PCAP files to generate conn.log. Alternatively, use CICFlowMeter for the standard 83-feature CSV output. Label the flows by matching timestamps and source IPs to the ground-truth attack log (provided with CIC datasets). Verify label distribution — slow-rate DoS samples are likely a small minority.

**1b. Exploratory data analysis:**  
Compute per-feature statistics (mean, std, min, max) separately for each attack class. Plot distributions for the key features (flow_duration, pkt_rate, tcp_window_min, header_completeness). Confirm which features show clear class separation — this validates your feature selection before any model training. Use SHAP on a quickly-trained XGBoost model to get preliminary feature importance rankings.

**1c. Feature engineering:**  
Apply the pipeline from Section 2: clean, normalise, build temporal windows (W=15–30s), compute behavioral parameters (incomplete requests, TCP socket count, response rate). Save the processed dataset in both tabular (CSV for ML) and verbalized text (JSONL for SLM) formats.

### Phase 2 — Baseline Establishment (Weeks 3–4)

**2a. ML baseline:**  
Train XGBoost and Random Forest on the tabular feature set. Tune hyperparameters via 5-fold cross-validation. Report per-class F1, macro-F1, inference latency (µs/sample), and model size. This establishes the accuracy floor for the encoder SLM to match or exceed, and the speed ceiling for the decoder to work within.

**2b. Feature selection validation:**  
Apply Mutual Information feature selection. Compare 12-feature vs. 20-feature vs. 50-feature subsets. Confirm the recommended 12–15 features retain >95% of the full-feature accuracy — this justifies the compact SLM input.

**2c. Simple explainability baseline:**  
Apply SHAP (Tree SHAP) to the best ML baseline. Examine the generated feature importance plots for a sample of Slowloris, RUDY, and Slowread detections. These SHAP outputs are the "ground truth" for evaluating whether the decoder SLM's explanations correctly identify the same discriminative features.

### Phase 3 — Encoder SLM Training (Weeks 5–7)

**3a. Data preparation for encoder:**  
Verbalize the 12–15 selected features using Format B (key-value pairs). Create train/val/test splits (80/10/10) with stratification. Apply SMOTE on training split. Verify that validation and test splits have the original class distribution (no oversampling).

**3b. Encoder fine-tuning:**  
Start from TinyBERT or MiniLM checkpoint. Replace classification head. Fine-tune with AdamW, lr=2e-5, 5-epoch max with early stopping. Evaluate on validation set every epoch. Save best checkpoint (highest validation macro-F1).

**3c. Quantisation for deployment:**  
Export best encoder checkpoint to ONNX. Apply dynamic INT8 quantisation. Benchmark latency on target hardware (ARM Cortex-A53 or equivalent). Confirm the size reduction (from ~50–80MB FP32 to ~15–25MB INT8) and acceptable accuracy retention (<1% drop).

### Phase 4 — Decoder SLM Training (Weeks 8–11)

**4a. Teacher-generated explanation corpus:**  
Select 500–1000 representative flows from the labelled dataset (covering Slowloris, RUDY, Slowread, and benign). Generate the SIR JSON for each. Use a teacher LLM (GPT-4, Claude, or DeepSeek-R1) to generate high-quality explanations following the Obs→Evidence→Conclusion→Mitigation schema and aligned with the 16-factor SSRP framework. Human-review 10% of these explanations for quality control.

**4b. QLoRA fine-tuning:**  
Fine-tune Gemma3-4B-Instruct (or Phi-2) using QLoRA on the instruction dataset. Training on a single 24GB GPU (RTX 3090/4090) takes approximately 2–4 hours for 1000 examples at 5 epochs. Monitor training loss and validation ROUGE-L score. Save best checkpoint.

**4c. GGUF export and quantisation:**  
Merge LoRA adapters into base model weights. Export to GGUF format using llama.cpp's `convert.py`. Apply Q4_K_M quantisation. Verify the final model size (~1.7–2.5GB) fits within the target device's RAM budget.

**4d. Few-shot exemplar library:**  
Select 3 high-quality explanations per attack type from the teacher-generated corpus. Store them in the knowledge base. Train an XGBoost retriever on the SIR feature vectors to match incoming flows to the most similar exemplar. Configure the decoder prompt to include the top-1 most similar exemplar at inference time.

### Phase 5 — Integration and End-to-End Evaluation (Weeks 12–14)

**5a. Pipeline integration:**  
Connect all components: Zeek → feature engineering → ML baseline → encoder → SHAP → SIR → decoder queue. Implement the async decoder processing (queue-based, non-blocking for detection pipeline). Test the end-to-end latency from flow arrival to explanation output.

**5b. Comprehensive evaluation:**  
Run the full evaluation suite:
- Detection metrics (Section 9) on the held-out test set
- Explanation quality metrics (eX-NIDS + SSRP frameworks) on 50–100 sampled outputs
- Resource metrics on target hardware (RAM, latency, energy via CodeCarbon)
- Zero-day generalisation: train without Slowread, test on Slowread flows
- Robustness: test against Slowloris with randomised User-Agents and SOCKS5 proxies

**5c. Hardware deployment:**  
Deploy encoder (INT8 ONNX) and decoder (Q4_K_M GGUF via Ollama) on target embedded hardware. Repeat latency and RAM measurements in actual deployment conditions (not simulated). Compare to Paper SS-2's 14× slowdown rule for calibration.

---

## What to Adopt — Strongest Evidence

**Hybrid ML + Encoder + Decoder architecture** — confirmed independently by Papers F2-4, F2-6, F3-2, F3-7, F4-3, SS-1, and SS-6. Every paper that tests pure-LLM vs. hybrid converges on hybrid superiority.

**QLoRA (4-bit NF4, rank=8–16) for decoder fine-tuning** — confirmed by Papers SS-3, F2-1, F2-5, F2-11, F2-12. Achieves >90% of full fine-tuning quality at <10% parameter update cost. Single-GPU feasibility is validated.

**Few-shot exemplars + CoT traces for decoder prompting** — confirmed by Paper F3-6 as dramatically superior to CoT alone (F1=0.85 vs. F1=0.05 for Gemma3-4B). The exemplars must be retrieved by XGBoost similarity, not BERT embeddings.

**SHAP on the ML/encoder output as the XAI bridge** — confirmed by Papers F3-2, F3-7, F4-4. SHAP provides the feature-level attribution that grounds the decoder's explanation in concrete evidence. Prevents hallucination about which features caused the alert.

**eX-NIDS augmented prompting** — inject protocol definitions, feature descriptions, and domain knowledge into decoder prompt. Paper F3-4 shows Feature Consistency reaching 100% with augmentation vs. ~90% without. For slow-rate DoS, inject: Slowloris mechanism description, RUDY Content-Length exploitation description, Slowread TCP window mechanism, normal connection duration norms.

**16-factor SSRP framework** — apply all 16 factors from Paper F3-9 to the decoder system prompt and user prompt. The 40% reasoning quality improvement in 2–4B models directly compensates for the limited reasoning capacity of the small decoder SLM.

**Past data ratio (0.3) to prevent catastrophic forgetting** — from Paper F3-7. Without it, encoder performance degrades across temporal data splits. Mix 30% prior-distribution data with new CIC IIoT 2025 training data.

**Temporal windowing (W=15–30s)** — from Papers F1-3, F1-6, F4-1. Single-flow analysis is fundamentally insufficient for slow-rate DoS. The attack signature requires observation across time. This is the most important algorithmic insight from Folder 1.

---

## What to Avoid — Clear Negative Evidence

**Using any LLM/SLM as a direct real-time classifier** — Paper SS-1 quantifies the 7,000× latency gap. Paper F2-4 shows LLMs are 4 orders of magnitude slower than XGBoost. Paper F3-1 shows zero-shot LLM precision ~50% (random). These findings are consistent and conclusive.

**Applying SHAP directly to LLM/SLM text inputs** — Paper SS-3 definitively demonstrates that SHAP on tokenised text inputs produces fragment-level attributions (e.g., "0 + + id + le + min") that are semantically meaningless. SHAP must be applied to the tabular ML model or the structured feature vector, not to the LLM's text input.

**Chain-of-Thought prompting without exemplars** — Paper F3-6 shows CoT-alone collapses performance catastrophically for ≤4B models on numeric network data (F1=0.05 for Gemma3-4B). Abstract reasoning instructions without concrete examples are harmful, not helpful, for small decoders.

**All 83 CICFlowMeter features as SLM input** — verbalization of 83 features exceeds compact SLM context windows and adds noise. Papers F1-7 (2 features suffice), F3-6 (9 features), and F1-8 (12 P4-computable features) all confirm high accuracy is achievable with minimal feature sets. The encoder SLM's context should not exceed 256 tokens.

**DistilBERT (66M) for embedded deployment** — Paper SS-2 shows DistilBERT at 1,207ms/window on ARM Cortex-A53 — effectively impractical for near-real-time IoT IDS. Use TinyBERT (157ms) or MiniLM (336ms) instead.

**Fine-tuning with ORPO/KTO for classification** — Paper F3-1 shows ORPO and KTO fine-tuning of LLaMA3-8B on flow classification yields only marginal improvement over zero-shot (~5%), far below RF's F1=92%. These preference-optimisation methods are designed for RLHF alignment, not discriminative classification. Use standard supervised cross-entropy fine-tuning for the encoder.

**Using source/destination IP addresses as features** — causes identity-based overfitting to the specific testbed. Not generalisable across deployment environments. Explicitly excluded in Papers F2-4, F3-3, F3-8.

**RUDY and Slowread detection without packet-level local features** — Paper F3-5 (ShieldGPT) demonstrates that adding the first 5 packets' TCP window size, timestamp delta, and TCP flags dramatically separates these subtypes. Flow-level statistics alone are insufficient to distinguish RUDY (tiny body content-length ratio) from Slowread (tiny TCP window). Capture at least the first N packet-level details per flow.

---

## Trade-Offs to Explicitly Acknowledge

**Accuracy vs. inference speed (encoder):** SecurityBERT (11M) achieves 98.2% accuracy at 150ms CPU inference. TinyBERT (14M) achieves 84.7% F1 at 157ms embedded. There is a minor accuracy loss (~1.5%) for a comparable speed gain. For IoT deployments where every millisecond counts, TinyBERT is the right choice. For server-edge deployments with more compute, MiniLM or SecurityBERT is better.

**Explanation depth vs. decoder size:** 3B models produce adequate explanations with heavy prompting (SSRP framework), but 7B models produce richer, more nuanced explanations with less prompt engineering. For truly constrained devices (≤4GB RAM), accept a 3B decoder with maximal prompting. For edge servers with 8GB RAM, a 7B decoder (4-bit quantised to ~4.5GB) is achievable and preferred.

**Real-time detection vs. explanation latency:** The detection decision (ML + encoder) is synchronous and fast (<200ms total). The explanation (decoder) is asynchronous and slow (1–3s). This is not a problem architecturally — it matches how security operations work (instant alert → delayed analysis). The system must be designed to keep these paths separate so explanation latency never delays traffic blocking decisions.

**Privacy vs. interpretability (PPFLE):** SecurityBERT's PPFLE hashing provides privacy by making raw feature values irrecoverable from the model input. However, it also makes the SIR (which uses hashed tokens instead of readable values) less interpretable for the decoder. If privacy is a hard requirement (e.g., for patient data or sensitive IoT data), use PPFLE for the encoder but pass the original feature values (or their descriptions) to the decoder in the SIR. These are separate pipelines and do not need to use the same representation.

**Known attack types vs. zero-day generalisation:** Fine-tuned models are significantly better on seen attack types but degrade on unseen variants. Paper F2-3 (DoLLM) shows a frozen LLM backbone improves zero-day F1 by +33.3% over XGBoost. For the encoder, partial unfreezing (fine-tune only top 6 of 15 layers) may preserve generalisation better than full fine-tuning. For the decoder, the few-shot exemplar + XGBoost retriever approach naturally handles near-distribution shifts by providing the closest known example.

---

## Research Gaps and Original Contribution Opportunities

### Gap 1 — No SLM specifically evaluated on slow-rate HTTP DoS
All 46 reviewed papers test LLMs/SLMs on volumetric DDoS, general network intrusion, or automotive CAN attacks. **No paper applies a compact encoder-based SLM specifically to Slowloris, RUDY, and Slowread attack flows.** This is the core novelty of your project. CIC IIoT 2025 is a new dataset with no prior SLM work — strengthening the originality claim.

### Gap 2 — No dual SLM pipeline (encoder for detection, decoder for explanation) on constrained devices
The hybrid ML+LLM architecture exists in Papers F3-2, F3-5, F3-7, and SS-6, but always with large cloud-hosted LLMs (GPT-3.5, GPT-4, GPT-4o). No paper tests an end-to-end pipeline where BOTH the encoder and decoder are resource-constrained SLMs running locally on edge hardware. This is a genuine hardware novelty that directly addresses the deployment gap identified in Paper F4-7 (When LLMs Meet Cybersecurity: only 8% of surveyed LLM security papers use models <8B; slow-rate DoS attacks are completely absent from the LLM IDS literature).

### Gap 3 — SHAP-to-SLM explanation interface not formalised
Paper SS-3 identifies that SHAP is incompatible with LLM text inputs. Papers F3-2 and F3-7 use SHAP to generate XAI outputs that are fed to the LLM, but do not formalise the structured intermediate representation format. Your project can contribute a concrete, reusable SIR specification (like the JSON format in Section 6) as a transferable contribution to the field.

### Gap 4 — Slow-rate DoS explanation quality not evaluated
Paper F3-5 (ShieldGPT) provides qualitative GPT-4 explanation outputs for slow-rate DoS types and shows they are accurate, but does not evaluate them quantitatively. Your project can apply the combined eX-NIDS + SSRP evaluation framework to produce the first quantitative explanation quality benchmark for slow-rate DoS-specific SLM explanations.

### Gap 5 — CIC IIoT Dataset 2025 is unexplored
To our knowledge, no published paper yet uses the CIC IIoT Dataset 2025 for SLM-based IDS research. Being among the first to publish results on this dataset provides a timing advantage and ensures strong topicality.

### Gap 6 — Feature attribution bridge for embedded SLM explanation
No paper tests whether SHAP-based feature attributions computed on a tabular ML model can reliably ground a <7B decoder SLM's explanation in a resource-constrained, real-time pipeline. This specific integration — SHAP output → SIR → quantised decoder — is untested and constitutes an original engineering contribution.

---

## Final Architecture Summary

```
CIC IIoT Dataset 2025 (PCAP / flow logs)
        ↓
Zeek/CICFlowMeter — generate flow features
        ↓
Data cleaning + SMOTE (train only) + MinMax normalisation
        ↓
Feature selection (MI + domain expertise → 12–15 features)
        ↓
Feature verbalization → Format B key-value text
        ↓
    ┌───────────────────────────────────────┐
    │           DETECTION PIPELINE          │
    │                                       │
    │  XGBoost/RF (tabular, ~1ms)           │
    │         ↓ alert + confidence          │
    │  TinyBERT/MiniLM INT8 ONNX (~80ms)   │
    │         ↓ classification + [CLS]      │
    │  SHAP (Tree SHAP, <100ms)            │
    │         ↓ feature importance JSON     │
    └──────────────┬────────────────────────┘
                   ↓
    Structured Intermediate Representation (SIR)
    {label, confidence, top_features, flags, context}
                   ↓ (async queue)
    ┌───────────────────────────────────────┐
    │         EXPLANATION PIPELINE          │
    │                                       │
    │  XGBoost retriever → fetch exemplar  │
    │         ↓ 1 exemplar + CoT trace     │
    │  Domain context injection             │
    │  (protocol defs, attack descriptions) │
    │         ↓ augmented prompt (SSRP F1-F16)│
    │  Gemma3-4B Q4_K_M via Ollama (~2s)  │
    │         ↓                             │
    │  Structured explanation               │
    │  Obs → Evidence → Conclusion         │
    │  → Mitigation recommendation         │
    └───────────────────────────────────────┘
                   ↓
    Security analyst dashboard / alert log
```

**Target performance (based on literature evidence):**
- Detection accuracy (encoder): ~92–98% macro-F1 on slow-rate DoS classes
- Inference latency (encoder + ML): <200ms on ARM Cortex-A53
- Decoder model size: ~2.0–2.5GB (Gemma3-4B Q4_K_M)
- Explanation latency: 1–3s (async, non-blocking)
- Explanation correctness: >75% (based on eX-NIDS GPT-4 baseline; ~60–70% expected for 4B fine-tuned model with SSRP)
- Feature consistency: >95% (enforced by SIR construction)
- Total RAM footprint: <4GB (encoder + decoder loaded together)

---

---

## 12. Stage-by-Stage Tools, Frameworks, and Libraries Reference

This section maps every recommended tool directly to its pipeline stage, so it can serve as a practical setup checklist. Stages follow the order of the implementation workflow in Section 11.

---

### Stage 1 — Traffic Capture and Flow Generation

| Tool | Role | Why / Paper Evidence |
|---|---|---|
| **Zeek IDS** | Generates structured `conn.log` from raw PCAP; captures `conn_state`, `history`, `orig_bytes`, `resp_bytes`, flow duration, packet counts | Preferred over CICFlowMeter for slow-rate DoS because `conn_state` (S1/S2 = established but incomplete) and `history` (e.g., `Sr` = SYN-only) directly encode Slowloris and RUDY signatures. Explicitly used in F2-4 (Mehavilla et al.) |
| **CICFlowMeter** | Generates 83-feature CSV in standard CIC format from PCAP | Alternative if comparability with prior CIC-format benchmarks is required. Generates `Flow Duration`, `Avg Packet Length`, `Fwd IAT Mean`, `SYN Flag Count`, etc. Used across F1-4, F1-8, SS-5 |
| **Tranalyzer** | 71-feature alternative to CICFlowMeter with better flow boundary detection | Mentioned in F2-2 (TrafficLLM); better at detecting flow termination events relevant to Slowread (connection never properly closed) |
| **tshark / Wireshark** | Raw PCAP capture; protocol-level filtering of HTTP/TCP traffic | Used for capture in F1-3 (slowTrack), F1-8 (P4+RF). Useful for filtering only port 80/443 traffic before flow generation to reduce dataset size |
| **Scapy** (Python) | Custom packet-level feature extraction; first-N packet attributes | Used in F1-3 and F1-8. Required for extracting first-5-packet TCP window size and timestamp delta (critical for separating Slowread from Slowloris per F3-5 ShieldGPT) |

**Install:** `sudo apt install zeek wireshark tshark` · `pip install scapy`

---

### Stage 2 — Data Cleaning and Preprocessing

| Tool | Role | Why / Paper Evidence |
|---|---|---|
| **pandas** | DataFrame-based data loading, cleaning, and transformation | Standard; used in all surveyed implementations. For replacing `inf` with `NaN`, dropping duplicates, removing flows < 1 second |
| **NumPy** | Numerical operations; vectorised feature computation | Required for mean imputation, normalisation computation, and temporal window aggregation |
| **imbalanced-learn** (`imblearn`) | SMOTE (Synthetic Minority Oversampling Technique) and ADASYN for handling class imbalance | Applied to training split only — validated in F1-4 (Al-Shukaili et al.) and F1-9 (FRE). Slow-rate DoS samples are a small minority in IIoT traffic; SMOTE generates synthetic minority samples without leaking test distribution |
| **scikit-learn** `preprocessing` | `MinMaxScaler` for encoder SLM input; `StandardScaler` for ML classifiers | MinMaxScaler validated in SS-4 (SecurityBERT) and F3-7 (LSTM). StandardScaler validated in F3-3 (From Flows to Words). Scalers must be fit on training split only |
| **scikit-learn** `model_selection` | `StratifiedShuffleSplit`, `train_test_split` with stratify parameter | Stratified splitting ensures all attack classes appear in train/val/test in correct proportions. Required when slow-rate classes are small minority |

**Install:** `pip install pandas numpy imbalanced-learn scikit-learn`

---

### Stage 3 — Feature Selection

| Tool | Role | Why / Paper Evidence |
|---|---|---|
| **scikit-learn** `feature_selection` | `mutual_info_classif` for Mutual Information scoring; `SelectKBest` for automated top-K selection | MI validated as superior to Pearson correlation and univariate tests in SS-5 (DDoSBERT). `mutual_info_classif` handles non-linear dependencies — critical for slow-rate DoS where attack signal is encoded in temporal patterns |
| **XGBoost** `feature_importances_` | SHAP-based feature importance from a quickly-trained XGBoost model | Provides a fast, preliminary feature ranking before any SLM training. Useful in Phase 1b (EDA). Validated across F1-4, F2-4, F3-2 |
| **SHAP** (`shap.TreeExplainer`) | Tree SHAP for computing exact Shapley values on XGBoost/RF; visualise mean absolute SHAP as feature importance bar chart | Provides the most theoretically grounded feature ranking. Paper F2-4 (Mehavilla et al.) uses Tree SHAP on Zeek features to confirm which flow-level attributes drive classification. Run on the baseline XGBoost to validate your MI-selected feature set |
| **matplotlib / seaborn** | Feature distribution plots, class separation visualisation, SHAP summary plots | Used for the EDA step in Phase 1b to verify features show visible class separation before model training |

**Install:** `pip install scikit-learn xgboost shap matplotlib seaborn`

---

### Stage 4 — Feature Verbalization and Prompt Construction

| Tool | Role | Why / Paper Evidence |
|---|---|---|
| **Python** `string.format` / f-strings | Construct Format B key-value text strings from feature dictionaries | Directly validated in F3-1 (eX-NIDS), F3-4, and F3-3 (From Flows to Words). Format: `"FLOW_DURATION: 127.5s PKT_RATE: 0.03 CONN_STATE: S1 ..."` |
| **json** (stdlib) | Serialise the Structured Intermediate Representation (SIR) into a JSON block for decoder prompt injection | The SIR format (Section 6) is a JSON object — Python's stdlib `json.dumps(sir_dict, indent=2)` produces the formatted block. Validated structurally by F4-4 (LLM for Explain) and F3-4 (eX-NIDS) |
| **Jinja2** | Template-based prompt construction for the decoder; inject SIR, exemplar, domain knowledge, and SSRP framework variables into a structured prompt template | Cleaner than string concatenation for the 16-factor SSRP prompt (F3-9). Allows versioned prompt templates to be maintained independently of code |
| **HuggingFace `tokenizers`** | Tokenise verbalized feature strings before encoder SLM input; verify token count does not exceed 256 | Required to confirm the selected 12–15 features verbalised in Format B fit within TinyBERT/MiniLM's 256-token context window. Use `tokenizer.encode(text, return_tensors='pt')` and check `input_ids.shape[1]` |

**Install:** `pip install transformers tokenizers jinja2`

---

### Stage 5 — ML Baseline Training

| Tool | Role | Why / Paper Evidence |
|---|---|---|
| **XGBoost** | Primary ML baseline classifier; gradient-boosted decision trees on tabular features | Best accuracy baseline: F1-9 (FRE) achieves 99.52% with XGBoost alone. SS-1 confirms XGBoost inference at ~2µs/sample — the speed anchor for the entire pipeline |
| **scikit-learn** `RandomForestClassifier` | Alternative/complementary ML baseline; ensemble of decision trees | F1-8 (P4+RF) achieves 98.28% on Slowloris. F1-7 (Reed et al.) achieves 95.9% with 2 features. Provides a SHAP-compatible model (Tree SHAP) for the XAI bridge |
| **scikit-learn** `GridSearchCV` / **Optuna** | Hyperparameter tuning for XGBoost/RF (max_depth, n_estimators, learning_rate) | 5-fold stratified cross-validation. Optuna provides Bayesian optimisation — more efficient than grid search for XGBoost's larger hyperparameter space |
| **scikit-learn** `classification_report` | Per-class precision, recall, F1; macro-averaged F1 | Generates the full metrics table (precision, recall, F1 per class + macro/weighted averages). The primary evaluation output for Section 9 detection metrics |
| **joblib** | Serialise trained XGBoost/RF model to disk | `joblib.dump(model, 'xgb_baseline.pkl')`. Lightweight serialisation for edge deployment; model files are typically <5MB |

**Install:** `pip install xgboost scikit-learn optuna joblib`

---

### Stage 6 — Encoder SLM Fine-Tuning

| Tool | Role | Why / Paper Evidence |
|---|---|---|
| **HuggingFace `transformers`** | Load pre-trained encoder SLM checkpoints (TinyBERT, MiniLM, SecurityBERT, DistilBERT); attach classification head; fine-tune with `Trainer` API | Universal backbone for all BERT-class models. Fine-tuning all encoder layers (not just head) confirmed superior in F2-1 (Bui et al.) by ~3–5%. `AutoModelForSequenceClassification` with `num_labels` set to your class count |
| **HuggingFace `datasets`** | Load and preprocess training data as `Dataset` objects; apply tokenisation via `map()`; efficient batching for `Trainer` | Required for the HuggingFace `Trainer` API. Handles text→token conversion, attention masks, and label encoding in one pipeline |
| **HuggingFace `Trainer`** + `TrainingArguments` | Manage the full training loop: AdamW optimizer, lr schedule, early stopping, gradient accumulation, checkpoint saving | `TrainingArguments(output_dir, learning_rate=2e-5, num_train_epochs=5, per_device_train_batch_size=32, load_best_model_at_end=True, metric_for_best_model='f1')`. Handles the 30% past-data-ratio mixing (F3-7) by combining two `Dataset` objects |
| **`EarlyStoppingCallback`** | Stop training when validation F1 stops improving (patience=3 epochs) | Prevents overfitting on the relatively small CIC IIoT 2025 slow-rate DoS subset. Validated as best practice in F2-1, F2-11 |
| **Captum** | Integrated Gradients attribution for encoder SLM; token-level importance for understanding which verbalized features the encoder attends to | Validated specifically for BERT-class models in F2-1. Use during development to verify the encoder is attending to meaningful features (flow_duration, pkt_rate) rather than spurious tokens |
| **Weights & Biases (`wandb`)** | Experiment tracking: log training loss, validation F1, learning rate, and all hyperparameters per run | Used in F2-11 and F2-12. Enables side-by-side comparison of TinyBERT vs. MiniLM vs. SecurityBERT runs. `wandb.init(project="slm-ids")` at the start of each run |

**Install:** `pip install transformers datasets evaluate wandb captum`

---

### Stage 7 — Encoder SLM Optimisation and Quantisation

| Tool | Role | Why / Paper Evidence |
|---|---|---|
| **ONNX** + **`optimum`** (HuggingFace) | Export fine-tuned encoder to ONNX format | `optimum-cli export onnx --model ./best_encoder --task sequence-classification ./encoder_onnx/`. ONNX enables cross-platform deployment without PyTorch dependency on edge device |
| **ONNX Runtime** (`onnxruntime`) | Run INT8 quantised encoder on CPU; 2–4× speedup over PyTorch CPU inference | SS-2 (Salah et al.) projects ~40–80ms for TinyBERT INT8 ONNX on ARM Cortex-A53, down from ~157ms in PyTorch. Use `onnxruntime.InferenceSession` for inference |
| **`onnxruntime.quantization`** | Apply static INT8 post-training quantisation to the exported ONNX model | `quantize_dynamic(model_input, model_output, weight_type=QuantType.QInt8)`. Reduces SecurityBERT from ~16.7MB to ~4.2MB with <1% accuracy loss |
| **PyTorch** `torch.quantization` | Alternative: dynamic INT8 quantisation in PyTorch before ONNX export; `torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)` | Simpler than ONNX pipeline for initial testing. Performance gains are slightly lower than ONNX Runtime but faster to set up |
| **`torch.nn.utils.prune`** | Structured attention head pruning — remove 30–50% of attention heads with lowest importance | From F2-7 (Lightweight LLMs): 75% memory reduction at ~2–5% accuracy drop achievable. Apply `prune.ln_structured` to attention weight matrices, then fine-tune recovery pass |

**Install:** `pip install onnx onnxruntime optimum[onnxruntime]`

---

### Stage 8 — XAI Bridge (SHAP/LIME)

| Tool | Role | Why / Paper Evidence |
|---|---|---|
| **SHAP** `shap.TreeExplainer` | Fast, exact Shapley values for XGBoost/RF baseline — the primary XAI method | F3-2 (HuntGPT), F3-7 (Chatzimiltis), F4-4 (LLM for Explain). Tree SHAP runs in <100ms for a single sample on a trained XGBoost. Produces exact feature attributions without approximation |
| **SHAP** `shap.KernelExplainer` | Approximate Shapley values for encoder SLM classification head (applied to tabular feature vector, NOT to token inputs) | Use only on the tabular feature input to the encoder, not on tokenised text. SS-3 (Lodh et al.) proves SHAP on tokenised text fragments is meaningless — apply Kernel SHAP to the structured feature vector only |
| **LIME** (`lime.lime_tabular`) | Local surrogate linear model explaining individual predictions from XGBoost or encoder | F3-2, F3-7. LIME provides a complementary local explanation to SHAP. Useful when SHAP values need to be double-checked with an independent method. `LimeTabularExplainer` with `feature_names` set to your selected feature list |
| **Custom SIR builder** (Python dict → JSON) | Combine SHAP top-K feature importances + encoder label + confidence + domain flags into the SIR JSON block | No external library — pure Python. The SIR format is defined in Section 6. Rank SHAP values by absolute magnitude, take top 4–6 features, and package with metadata |

**Important:** Never apply SHAP or LIME directly to the decoder SLM's text input — this is explicitly invalidated by SS-3 (Lodh et al.), which shows SHAP produces meaningless token-fragment attributions ("0 + + id + le + min") on tokenised network data.

**Install:** `pip install shap lime`

---

### Stage 9 — Decoder SLM Fine-Tuning (QLoRA)

| Tool | Role | Why / Paper Evidence |
|---|---|---|
| **HuggingFace `transformers`** | Load instruct-tuned base model (Gemma3-4B-Instruct, Phi-2, LLaMA3.2-3B-Instruct) | Use `AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")`. The instruct-tuned variant is essential — confirmed in F2-11 to dramatically improve zero-shot instruction following |
| **`bitsandbytes`** | 4-bit NF4 quantisation of the frozen base model during QLoRA training | `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)`. Validated in SS-3, F2-1, F2-5, F2-11, F2-12. Makes fine-tuning a 4B model possible on a single 24GB GPU |
| **HuggingFace `peft`** | LoRA adapter insertion and management | `LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")`. Only ~1–2% of parameters are trained, keeping GPU memory manageable. Validated across all 5 QLoRA papers |
| **`trl`** (Transformer Reinforcement Learning) | `SFTTrainer` for supervised fine-tuning on instruction dataset | Wraps HuggingFace `Trainer` with built-in support for the `(instruction, input, response)` dataset format. `SFTTrainer(model, train_dataset, peft_config, dataset_text_field="text", ...)`. Used in F2-11 and F2-12 |
| **Optuna** | Hyperparameter search for QLoRA: rank (r=4/8/16), alpha (16/32/64), learning rate (1e-4, 2e-4, 5e-4) | 3 trials are usually sufficient to identify the optimal rank/alpha combination. SS-3 (Lodh et al.) finds convergence at 3–4 epochs across all decoder models tested |
| **`datasets`** + **`json`** | Load teacher-generated (instruction, input, response) JSONL dataset; convert SIR JSON to prompt format | Teacher explanations generated offline by GPT-4/Claude on 500–5,000 SIR examples. Store as JSONL: `{"instruction": "...", "input": "...", "output": "..."}` per line |
| **`accelerate`** | Multi-GPU training coordination; CPU offloading for models too large for single GPU | `accelerate launch train.py`. Required if training on multi-GPU setup; optional for single 24GB GPU + Gemma3-4B |

**Install:** `pip install transformers peft bitsandbytes trl accelerate datasets optuna`

---

### Stage 10 — Decoder SLM Optimisation and Quantisation

| Tool | Role | Why / Paper Evidence |
|---|---|---|
| **`peft`** `merge_and_unload()` | Merge trained LoRA adapter weights back into the base model; produce a standard HuggingFace model | Required before GGUF export. `model = model.merge_and_unload()`. Produces a full-weight model that llama.cpp can convert |
| **llama.cpp** | Convert merged HuggingFace model to GGUF format and apply Q4_K_M quantisation | `python convert.py merged_model/ --outfile decoder.gguf` then `./quantize decoder.gguf decoder_q4km.gguf q4_k_m`. Q4_K_M: Gemma3-4B → ~2.5GB, Phi-2 → ~1.7GB. CPU-optimised inference — validated in F3-3 and F3-6 |
| **Ollama** | Serve the quantised GGUF model via REST API (`POST /api/generate`); handles context management, concurrency, and model loading/unloading | Used explicitly in F3-6 (Rethinking On-Device). Create a `Modelfile` pointing to the GGUF file. `ollama create slm-ids -f Modelfile`. REST API eliminates need for custom inference code in the pipeline |
| **llama.cpp** speculative decoding | Draft-model-assisted token prediction for 2–3× throughput improvement on CPU | Pair Gemma3-4B Q4_K_M with Gemma3-1B Q4_K_M as draft model. Token acceptance rate ~70–80% for structured explanation outputs. Configured via `llama-server --model decoder_q4km.gguf --draft-model draft_q4km.gguf` |
| **ONNX Runtime** (encoder, already mentioned) | Run INT8 encoder in the same pipeline as Ollama decoder, keeping both within 4GB total RAM budget | Encoder ONNX: ~4–14MB. Decoder GGUF: ~1.7–2.5GB. Combined: well within 4GB hardware budget. Load encoder once at startup; Ollama keeps decoder in memory between explanation requests |

**Install:** `pip install peft` · Build llama.cpp from source (`cmake -B build && cmake --build build --config Release`) · `curl -fsSL https://ollama.com/install.sh | sh`

---

### Stage 11 — Few-Shot Exemplar Retrieval

| Tool | Role | Why / Paper Evidence |
|---|---|---|
| **XGBoost** (retriever mode) | Train a similarity scorer on SIR feature vectors; retrieve the most similar exemplar from the knowledge base at inference time | F3-6 (Rethinking On-Device) conclusively shows XGBoost retriever outperforms BERT embedding similarity for numeric network feature matching. The retriever is trained to rank exemplars by feature-space proximity |
| **FAISS** | Vector index for fast approximate nearest-neighbour search over exemplar feature vectors if knowledge base grows large (>1,000 exemplars) | F3-8 (IDS-Agent). For small exemplar libraries (3–5 per class = 15–25 total), a simple XGBoost similarity score over the full library is faster and more accurate than approximate ANN |
| **ChromaDB** | Lightweight local vector database for storing exemplar SIR vectors and associated explanation texts; supports metadata filtering | Used in F2-1, F3-8. Simpler setup than FAISS for small deployments. `chromadb.Client()` with a local persistent store |
| **Python dict / JSON file** | Ultra-lightweight exemplar store for ≤50 exemplars — just a JSON file loaded into memory at startup | Sufficient for the 3–5 exemplars per attack class (15–25 total) recommended in Section 7. No vector database infrastructure needed for this scale |

**Install:** `pip install faiss-cpu chromadb` (or use plain JSON for minimal deployments)

---

### Stage 12 — Deployment on Resource-Constrained Device

| Tool | Role | Why / Paper Evidence |
|---|---|---|
| **ONNX Runtime** (`onnxruntime`) | Run INT8 quantised encoder on ARM CPU; achieves ~40–80ms inference | Validated as the best CPU-only inference runtime for BERT-class models. SS-2 (Salah et al.) benchmarks on NXP i.MX 8M Plus ARM Cortex-A53 |
| **Ollama** | Serve quantised decoder GGUF via localhost REST API; manages model lifecycle and context | The production-ready serving layer for the decoder on edge devices. REST interface allows easy integration with any language or monitoring system |
| **`asyncio`** / **`queue.Queue`** | Implement the async producer-consumer pipeline separating synchronous detection from async explanation | Detection (ML + encoder) runs in the main thread, producing SIR objects. A background worker thread consumes SIR objects and calls Ollama. Prevents explanation latency from blocking detection decisions |
| **`fastapi`** / **`flask`** | Lightweight REST API to expose detection + explanation results to analyst dashboard or SIEM integration | Optional. Provides a simple HTTP endpoint (`POST /detect`) that accepts a flow feature dict and returns detection label + queued explanation |
| **`psutil`** | Monitor RAM and CPU usage during deployment; trigger alerts if memory exceeds budget | Validate that encoder + decoder together stay under 4GB RAM on target hardware. `psutil.virtual_memory().used` sampled every 30s |
| **CodeCarbon** | Measure energy consumption (kWh) and CO2 emissions during inference and training | Directly used in SS-3 (Lodh et al.) to compare model efficiency. `from codecarbon import EmissionsTracker; tracker = EmissionsTracker(); tracker.start(); ...; tracker.stop()`. Required for the energy efficiency argument in the thesis |

**Install:** `pip install onnxruntime fastapi flask psutil codecarbon` · Ollama installed natively

---

### Stage 13 — Reasoning and Explanation Generation

| Tool | Role | Why / Paper Evidence |
|---|---|---|
| **Ollama REST API** (`requests` or `httpx`) | Send the augmented decoder prompt (SIR + exemplar + domain context + SSRP framework) to the locally running Gemma3-4B/Phi-2 GGUF model | `POST http://localhost:11434/api/generate` with `{"model": "slm-ids", "prompt": full_prompt, "stream": false}`. Response contains the structured explanation text |
| **Jinja2** | Render the final decoder prompt from template, injecting SIR JSON, retrieved exemplar, domain knowledge, and all 16 SSRP factors | Keeps the 16-factor SSRP prompt (F3-9) maintainable as a versioned template file separate from code. Critical for ablation studies (which SSRP factors contribute most?) |
| **GBNF grammar** (llama.cpp grammar files) | Constrain decoder output to a valid JSON structure: `{"observation": "...", "evidence": [...], "conclusion": "...", "mitigation": "..."}` | From F3-3 (From Flows to Words). Grammar-constrained decoding eliminates malformed JSON outputs entirely. Define the grammar in a `.gbnf` file and pass it to llama.cpp/Ollama via the `grammar` parameter |
| **`langchain`** | Optional: orchestrate multi-step reasoning chain (RAG retrieval → prompt construction → decoder invocation → output parsing) | Used in F3-8 (IDS-Agent) for the ReAct agent loop. For a simpler single-step pipeline, plain Python with Jinja2 + Ollama is sufficient and lighter |

**Install:** `pip install requests httpx jinja2 langchain` (langchain optional)

---

### Stage 14 — Experiment Tracking and Evaluation

| Tool | Role | Why / Paper Evidence |
|---|---|---|
| **Weights & Biases (`wandb`)** | Log all training runs (encoder and decoder); track hyperparameters, loss curves, F1 scores; compare model variants side-by-side | Used in F2-11 and F2-12. Essential for comparing TinyBERT vs. MiniLM vs. SecurityBERT fine-tuning runs and for QLoRA rank/alpha ablations |
| **HuggingFace `evaluate`** | Compute ROUGE-L and BERTScore against teacher-generated reference explanations | `evaluate.load("rouge")`, `evaluate.load("bertscore")`. Automated proxy metrics for explanation quality. Supplement with human annotation using the eX-NIDS + SSRP framework |
| **`scipy.stats`** | Statistical significance testing (Friedman test, Wilcoxon signed-rank) for comparing model variants | SS-7 (LLM-APTDS) uses Friedman test to establish that performance differences are statistically significant. Required for rigorous thesis comparison of TinyBERT vs. MiniLM or Phi-2 vs. Gemma3-4B |
| **`sklearn.metrics`** | `classification_report`, `roc_auc_score`, `confusion_matrix`, `f1_score` with `average='macro'` | Full detection metrics suite. `roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')` for AUC-ROC |
| **`matplotlib` / `seaborn`** | Confusion matrices, ROC curves, SHAP summary plots, per-epoch F1 curves | Visualise model performance for thesis figures. `seaborn.heatmap` for confusion matrices; `shap.summary_plot` for feature importance |
| **CodeCarbon** | Track energy consumption during inference on target hardware for efficiency comparison between model variants | SS-3 (Lodh et al.) reports Phi-2 as most energy-efficient at 0.173 kg CO2/training run. Use CodeCarbon to produce equivalent figures for CIC IIoT 2025 experiments |
| **`py-readability-metrics`** | Flesch Reading Ease and Gunning Fog scores for explanation text readability | F3-7 (Chatzimiltis et al.). Measures whether generated explanations are understandable by security analysts without deep ML expertise |

**Install:** `pip install wandb evaluate scipy scikit-learn matplotlib seaborn codecarbon py-readability-metrics`

---

### Stage 15 — Human Evaluation of Explanations

| Tool / Method | Role | Why / Paper Evidence |
|---|---|---|
| **Manual annotation spreadsheet** | Two independent evaluators score each explanation on all 7 metrics (eX-NIDS 3 + SSRP 4) | F3-4 (eX-NIDS), F3-9 (SSRP). Sample 50–100 explanation outputs; each evaluator scores independently; compute inter-rater Cohen's κ |
| **`sklearn.metrics.cohen_kappa_score`** | Compute Cohen's κ between two evaluators' annotation vectors | κ > 0.80 = strong agreement (target from F3-9, SSRP paper). If κ < 0.60, refine the annotation rubric and re-evaluate |
| **Ground-truth feature dictionary** | A JSON file mapping each network feature to its expected direction for each attack type (e.g., `flow_duration` is HIGH for all slow-rate types) | Required for automated Factual Consistency scoring (eX-NIDS metric 3). Any explanation that claims `flow_duration` is LOW for Slowloris fails this check automatically |
| **Reference explanation library** | 3 gold-standard explanations per attack type (generated by teacher LLM and human-reviewed) | Used for ROUGE-L and BERTScore automated evaluation. Represents the "ideal" explanation output that the fine-tuned decoder should approximate |

---

### Consolidated Environment Setup

The following commands set up the full research environment from scratch:

```bash
# 1. Core data science stack
pip install pandas numpy scikit-learn imbalanced-learn xgboost joblib

# 2. Encoder SLM training
pip install transformers datasets evaluate tokenizers peft accelerate wandb captum

# 3. Decoder SLM fine-tuning (QLoRA)
pip install bitsandbytes trl optuna

# 4. XAI and feature attribution
pip install shap lime

# 5. ONNX optimisation (encoder)
pip install onnx onnxruntime optimum[onnxruntime]

# 6. Explanation generation and prompt orchestration
pip install requests httpx jinja2 langchain

# 7. Vector store for exemplar retrieval (optional)
pip install faiss-cpu chromadb

# 8. Evaluation and monitoring
pip install scipy matplotlib seaborn codecarbon py-readability-metrics psutil

# 9. Visualisation (SHAP plots, confusion matrices)
pip install matplotlib seaborn

# 10. llama.cpp (build from source for CPU-optimised decoder inference)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build --config Release

# 11. Ollama (decoder serving)
curl -fsSL https://ollama.com/install.sh | sh

# 12. Zeek (flow feature extraction — system package)
# Ubuntu/Debian: sudo apt install zeek
# Or: https://zeek.org/get-zeek/

# 13. CICFlowMeter (alternative flow extraction — Java-based)
# Download from: https://github.com/CanadianInstituteForCybersecurity/CICFlowMeter
```

**Recommended Python version:** 3.10 or 3.11 (bitsandbytes and some ONNX tools have version constraints)  
**GPU for training:** Single NVIDIA GPU with ≥16GB VRAM for encoder fine-tuning; ≥24GB VRAM for QLoRA decoder fine-tuning (RTX 3090/4090 or A10G). Inference (deployment) is CPU-only.  
**Target deployment hardware:** ARM Cortex-A53/A55 (e.g., Raspberry Pi 4/5, NXP i.MX 8M Plus, NVIDIA Jetson Nano) with ≥4GB LPDDR4 RAM.

---

*End of analysis_and_recommendations.md*  
*Based on 46 reviewed papers: 11 (slow-rate detection) + 12 (LLM for IDS) + 9 (detect+explain) + 7 (background) + 7 (supervisor shared)*
