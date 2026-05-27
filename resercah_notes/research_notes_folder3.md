# Research Notes — Folder 3: Detect + Explain
**Status**: ALL 9 PAPERS COMPLETE.
**Folder path**: `/Users/sangharshathapa/Desktop/slm/3. Detect + Explain/`
**Files** (9 total):
1. `Towards_Explainable_Network_Intrusion_Detection_using_Large_Language_Models.pdf`
2. `huntgpt.pdf`
3. `From Flows to Words.pdf`
4. `exnid.pdf`
5. `sheildgpt.pdf`
6. `Interpretable Anomaly-Based DDoS Detection in AI-RAN with XAI and LLMs.pdf`
7. `Rethinking On-Device.pdf`
8. `llm agent.pdf`
9. `Strengthening Human-Centric Chain-of-Thought Reasoning Integrity in LLMs.pdf`

---

## PAPER F3-1 — Towards Explainable Network Intrusion Detection using Large Language Models

**File**: `Towards_Explainable_Network_Intrusion_Detection_using_Large_Language_Models.pdf`
**Authors**: Houssel, Singh, Layeghy, Portmann (University of Queensland, Australia)
**Venue**: 2024 IEEE/ACM International Conference on Big Data Computing, Applications and Technologies (BDCAT)

### Detection Method
- Feasibility study of LLMs (GPT-4, LLaMA3-8B) as NIDS classifiers, primarily motivated by explainability
- Input format: NetFlow entries as **key-value text pairs** (e.g., `"L4_DST_PORT: 80"`) — NOT feature verbalization, NOT raw payload
- Zero-shot baseline: LLM instructed to output "1" (malicious) or "0" (benign)
- Fine-tuning: **ORPO** (Odds Ratio Preference Optimisation) and **KTO** (Kahneman-Tversky Optimisation) applied to LLaMA3-8B-Instruct
- Temperature=0.1 for deterministic binary output

### Dataset
- **NF-UNSW-NB15-v2** and **NF-CSE-CIC-IDS2018-v2** (standardized NetFlow datasets by Sarhan et al.)
- NOT IoT domain datasets — general network traffic
- 95%/5% train-test split; 10-fold cross-validation on test set
- Small test set used due to sheer cost of LLM inference

### Preprocessing & Feature Extraction
- NetFlow fields formatted as text key-value pairs (no verbalization, no natural language descriptions)
- No feature engineering beyond the standardized NetFlow feature set

### Models / Architecture
- **LLaMA3-8B-Instruct** (open source): zero-shot + two fine-tuning approaches
- **GPT-4-0613** (OpenAI): zero-shot only (fine-tuning cost prohibitive via API)
- Fine-tuning approaches:
  - **ORPO**: requires preference data (correct label = accepted, inverse = rejected)
  - **KTO**: does NOT require paired data; uses (prompt, response, binary relevance label)
- Baselines for comparison: Random Forest (F1=92.17%), LSTM (F1=92.82%), DANN (F1=97.81%)

### Training / Evaluation
- Hardware: NVIDIA RTX3090, 3.50 GHz CPU
- Metrics: Macro Average Precision and Recall (unweighted per class)

### Key Results
| Model | Dataset | Precision | Recall |
|-------|---------|-----------|--------|
| LLaMA3-8B (zero-shot) | NF-UNSW-NB15-v2 | 48.56% | 40.56% |
| GPT-4 (zero-shot) | NF-UNSW-NB15-v2 | 50.85% | 53.47% |
| LLaMA3-8B (zero-shot) | NF-CSE-CIC-IDS2018-v2 | 49.66% | 49.73% |
| GPT-4 (zero-shot) | NF-CSE-CIC-IDS2018-v2 | 50.25% | 51.02% |
| LLaMA3 KTO (50k samples) | NF-UNSW-NB15-v2 | 55.18% | 51.67% |
| LLaMA3 ORPO (50k samples) | NF-UNSW-NB15-v2 | 52.00% | 43.50% |

- **LLaMA3 zero-shot: WORSE than random (50%)** — expected
- **GPT-4 zero-shot: marginally above random** — not useful
- **Fine-tuning (ORPO/KTO): slight improvement, but still near random after 50k samples**
- **KTO slightly better than ORPO** for this task
- Comparison: RF achieves F1=92.17% on same dataset — LLMs are not competitive

### Inference Latency
| Method | Inference Time (µs) | Parameters |
|--------|--------------------|-----------:|
| LLaMA3-8B | 14,000 | 8.03B |
| Random Forest | 2.03 | 867K nodes |
| Decision Tree | 1.53 | — |
| LSTM | 25.02 | 56,555 |
| DANN | 28.40 | 67,208 |

**LLMs are ~7,000× slower than Random Forest**

### Explainability Analysis
- LLMs provide detailed, contextually relevant explanations for flow classifications
- **Strengths**: correctly identifies traffic types (DNS, HTTP) from features; convincing narrative
- **Critical weaknesses (hallucinations)**:
  - Misidentifies geolocation (claims China when it's Japan)
  - Confuses protocol numbers (protocol 139 ≠ NetBIOS; it's Host Identity Protocol)
  - Incorrectly assumes port 0 = malicious (valid in ICMP and other protocols)
  - Misidentifies Layer 7 protocol numbers (7 ≠ HTTP; it's IPP)
  - Fails to explain SYN flood attack (straightforward for signature-based detectors)
- LLMs treat individual suspicious features as independent clues rather than correlating them to attack signatures

### Proposed Future Direction
- **RAG + CTI source**: ground LLM in up-to-date threat intelligence to reduce hallucinations
- **Function calling**: enable LLM to issue active commands (firewall rule modifications, SIEM actions) in response to detected threats
- **Expand RAG** beyond CTI to include: endpoints, OS info, SBOM, firewall rules

### Tools / Frameworks
- Hugging Face Transformers (LLaMA3-8B-Instruct)
- OpenAI API (GPT-4)
- ORPO and KTO fine-tuning (NOT LoRA/QLoRA)
- NetFlow datasets from Sarhan et al. (standardized NF-v2 datasets)

### Advantages
- First thorough investigation of LLM adaptivity to NetFlow domain
- Tests both instruction-following LLMs (not just classifiers with replaced head)
- Identifies explainability as the genuine value proposition for LLMs in NIDS
- Quantifies inference time gap concretely (7,000×)

### Limitations
- ORPO/KTO fine-tuning does not converge well for binary classification on NetFlow features
- LLMs are impractical as standalone NIDS (near-random classification, 7000× slower)
- Hallucination is a systematic problem for security-critical explanations
- Does not test LoRA/QLoRA (may have helped classification)

### SLM Relevance
⭐⭐⭐⭐ **HIGH** — Critical finding: LLMs CANNOT replace ML for flow classification even with fine-tuning. The RAG+CTI+function-calling future direction is the exact architecture to aim for in SLM slow-rate DoS system. Hallucination analysis is important: explains why human-in-the-loop validation is necessary. The 7,000× latency gap reinforces the hybrid ML+SLM design.

---

## PAPER F3-2 — HuntGPT: Integrating ML-Based Anomaly Detection and Explainable AI with LLMs

**File**: `huntgpt.pdf`
**Authors**: Tarek Ali, Panos Kostakos (University of Oulu, Center for Ubiquitous Computing, Finland)
**Venue**: arXiv:2309.16021, September 2023

### Detection Method
- **Integrated threat hunting dashboard** combining ML detection + XAI explanation + LLM conversational agent
- NOT a new detection algorithm — a prototype system integrating existing components
- Pipeline: Random Forest detects anomaly → SHAP/LIME explains → GPT-3.5-Turbo translates to natural language → analyst interacts via chat

### Dataset
- **KDD99** (KDD Cup 1999 intrusion detection dataset) — widely used but older benchmark
- Network anomaly types: DoS, Probe, R2L, U2R, Normal

### Preprocessing & Feature Extraction
- Standard KDD99 features (41 features: basic, content, time-based, host-based)
- Model trained on KDD99 offline; predictions made on incoming packets
- No novel feature engineering

### Models / Architecture
- **3-layer system**:
  1. **Analytics Engine** (server): RF classifier (KDD99-trained) + SHAP explainer + LIME explainer → Elasticsearch + AWS S3 storage
  2. **Data Storage**: Elasticsearch (detected-packets and original-packets indices) + AWS S3 (SHAP/LIME plots)
  3. **IDS Dashboard** (Gradio UI): OpenAI connector + anomaly data fetcher → GPT-3.5-turbo chat agent
- **Random Forest** (ML model): trained on KDD99; main anomaly detector
- **SHAP**: global feature importance; identifies top features contributing to prediction class
- **LIME (tabular)**: local explanation per prediction; generates per-instance feature importance bar chart
- **GPT-3.5-turbo**: receives detected packet data + SHAP/LIME factors → generates natural language explanation + answers analyst follow-up questions
- Users can download a full incident report (PDF with all graphs + data)

### Training / Evaluation
- RF trained on full KDD99 dataset offline
- Technical accuracy of GPT-3.5: tested on cybersecurity certification exams
  - CISM Practice Exam (40 questions): **82.5% success rate**
  - ISACA CISM Practice Quiz (10 questions): **80% success rate**
  - ISACA Cybersecurity Fundamentals Quiz (25 questions): **72% success rate**
- Response readability: evaluated with 6 formulas (Flesch-Kincaid, Dale-Chall, etc.)
  - Consistently rated at **Graduate level** (grade 15-17) — complex but generally comprehensible for college-educated users

### Tools / Frameworks
- **Gradio** (UI dashboard framework)
- **Elasticsearch** (document storage + querying)
- **AWS S3** (plot image storage)
- **OpenAI API** (GPT-3.5-turbo)
- **SHAP** (global feature importance)
- **LIME** (local instance explanation)
- scikit-learn (Random Forest)
- Python

### Key Findings
- GPT-3.5 has strong cybersecurity domain knowledge (72-82.5% on CISM exams)
- Conversational agent effectively generates actionable responses and promotes user engagement
- Integrated XAI+LLM dramatically improves explanation accessibility vs. raw SHAP plots alone
- Generated explanations are graduate-level complexity — suitable for trained analysts, may be too complex for non-specialists
- LLM provides mitigation recommendations (e.g., firewalls, rate limiting) when asked

### Advantages
- First complete prototype of ML+XAI+LLM dashboard for IDS
- Modular design: each layer can be developed/maintained independently
- Practical: generates downloadable incident reports
- Interactive: analysts can ask follow-up questions in natural language
- GPT-3.5 demonstrates real cybersecurity knowledge (not just pattern matching)

### Limitations
- KDD99 is a dated dataset (1999); not representative of modern attacks
- RF on KDD99 doesn't include slow-rate HTTP attacks
- GPT-3.5 API required — latency and cost for production deployment
- No real-time detection capability yet (offline ML + human-triggered LLM)
- Response complexity (graduate level) may not serve non-technical stakeholders

### SLM Relevance
⭐⭐⭐⭐⭐ **VERY HIGH** — This is the CLOSEST prior work to the proposed SLM slow-rate DoS system architecture. The ML(detection) + SHAP(XAI) + LLM(explanation+Q&A) pipeline is exactly the design to replicate. Key lessons: SHAP provides the bridge between ML detection and LLM explanation; conversational interface is valued by analysts; GPT-3.5 domain knowledge is strong enough for cybersecurity Q&A. Replace KDD99+RF with slow-rate DoS dataset+ML; replace GPT-3.5 with a local SLM; add slow-rate DoS specific knowledge to explanations.

---

## PAPER F3-3 — From Flows to Words: Can Zero-/Few-Shot LLMs Detect Network Intrusions?

**File**: `From Flows to Words.pdf`
**Authors**: Rehman, Shah, Anwar, Islam (Future Data Minds Research Lab, Australia)
**Venue**: arXiv:2510.17883, October 2025 (preprint)

### Detection Method
- **Prompt-only** (NO fine-tuning, no gradient updates) LLM evaluation for IDS
- Core question: Can instruction-tuned LLMs detect intrusions using only prompt engineering?
- Four design innovations:
  1. **Flow-to-text protocol**: compact natural-language record from NetFlow features
  2. **Boolean domain flags**: interpretable inductive biases prepended to prompts
  3. **Grammar-constrained decoding** (GBNF): forces structured JSON output `{"prediction": "attack|benign", "p_attack": 0.0-1.0}`
  4. **Single threshold calibration** (τ*): maximize F1 on dev slice → freeze for test
- Three prompting modes: zero-shot / instruction-guided / few-shot

### Dataset
- **UNSW-NB15** (official CSVs: 175,341 train + 82,332 test rows)
- Binary label: benign (0) vs. attack (1)
- 39 numeric + 3 categorical features (proto, service, state) → 194 dims after encoding
- LLM evaluation on balanced subsets: N ∈ {200, 1000, 2000} drawn from test set

### Preprocessing & Feature Extraction
- **ML/DL**: z-score standardization + one-hot encoding → 194 dims
- **LLM**: flow-to-text serialization with rounded numeric cues
  - Includes: duration, packet-rate, byte-ratio, sttl/dttl, tcprtt, synack, ackdat, ct_state_ttl, proto, service, state
- **Boolean flags** (prepended to prompt):
  - `asymmetry_high`: (sbytes+1)/(dbytes+1) > threshold OR (spkts+1)/(dpkts+1) > threshold
  - `pkt_rate_high`: (spkts+dpkts)/max(10⁻⁶, dur) > threshold
  - `ttl_anomaly`: TTL values outside normal range
  - `tcp_timer_anomaly`: implausible synack/ackdat timing
  - `rare_service_state`: unusual service or state combination
  - `short_burst`: high packet rate in very short duration

### Models / Architecture
- **LLMs (prompt-only)**:
  - TinyLlama-1.1B, Qwen 2.5-3B, Qwen 2.5-7B-Instruct, Mistral-7B-Instruct
  - Served via **llama.cpp**, GGUF Q4_K_M quantization, NVIDIA T4 (16GB)
  - Context ≈ 1024 tokens, batch ≈ 1024, temperature=0, top-p=1 (fully deterministic)
  - GBNF grammar constrains output to exactly one JSON object
- **ML baselines**: LR, SVM (linear), Random Forest, XGBoost/LightGBM, MLP
- Calibration: τ* chosen from dev slice to maximize F1; then frozen

### Key Results

**ML/DL baselines (full test set):**
| Model | Accuracy | Precision(+) | Recall(+) | F1(+) |
|-------|----------|-------------|-----------|-------|
| XGBoost (balanced) | **0.9528** | 0.9407 | 0.9530 | **0.9465** |
| SVM (balanced) | 0.9327 | 0.9248 | 0.9196 | 0.9221 |
| RF (final) | 0.8711 | 0.8178 | 0.9854 | 0.8938 |

**LLM results (balanced subsets):**
| Model | Type | N | Accuracy | F1(+) | Macro-F1 |
|-------|------|---|----------|-------|---------|
| Mistral-7B | Zero-shot | small | 0.09 | — | 0.0826 |
| Qwen 2.5-3B | Zero-shot | — | 0.455 | — | 0.3127 |
| TinyLlama-1.1B | Guided | — | 0.455 | — | 0.3127 |
| **Qwen 2.5-7B + flags + calibrated** | Guided+Flags | **200** | **0.785** | **0.802** | **0.783** |
| Qwen 3B calibrated | Few-shot+Flags | 1000 | 0.696 | 0.682 | — |
| Qwen calibrated (recall-heavy) | Recall-opt | 2000 | 0.559 | 0.694 | — |

**Scaling sensitivity**: Performance degrades as N increases (200→1000→2000)

### Tools / Frameworks
- scikit-learn (LR, SVM, RF, MLP)
- LightGBM, XGBoost (boosted trees)
- llama.cpp (LLM serving, GGUF quantization)
- GBNF grammar (structured output enforcement)
- NVIDIA T4 GPU

### Key Findings
1. **Zero-shot LLMs are unreliable**: frequently collapse to single class (F1=0 for attack class), especially smaller models
2. **Flags + grammar + calibration are decisive**: structured prompts with domain flags dramatically improve detection quality
3. **Calibration is necessary**: without τ*, class collapse prevents useful precision-recall tradeoff
4. **Scaling sensitivity**: as evaluation set grows beyond N=200, performance degrades — LLMs struggle with numeric telemetry at scale
5. **Best case (N=200, 7B + flags + calibrated)**: macro-F1=0.783 — competitive with mid-tier ML but far below XGBoost
6. **Latency**: LLMs 1-2 orders of magnitude slower than tabular baselines even on T4
7. **Key advantage of LLM approach**: NO gradient training needed; human-readable prompts + flags can serve as auditable policy documentation
8. **Recommended hybrid**: fast tabular model screens → routes borderline/novel flows to LLM for secondary judgment/textual rationalization

### Advantages
- No fine-tuning required — rapid iteration via prompt edits
- Produces human-readable artifacts that can be audited as policy
- Grammar-constrained output enables reliable downstream processing
- Calibration provides tunable precision-recall tradeoff without retraining

### Limitations
- Performance degrades with scale — not suitable for high-volume real-time IDS
- Requires GPU for inference (tabular baselines only need CPU)
- Sensitivity to prompt design — fragile if flow distribution shifts
- Boolean flags require manual expert design

### SLM Relevance
⭐⭐⭐⭐ **HIGH** — Directly demonstrates what works for prompt-only SLM IDS: domain flags + grammar-constrained output + threshold calibration. The flow-to-text serialization methodology is directly applicable to slow-rate DoS flows. The boolean flag design (asymmetry, burst rate, timing anomalies) maps naturally to slow HTTP attack patterns (low rate, long connection duration, incomplete requests). The hybrid recommendation (tabular triage → LLM explanation) validates the proposed SLM experiment architecture.

---

## PAPER F3-4 — eX-NIDS: Augmented-Prompt Explainer for Network Intrusion Detection Systems

**File**: `exnid.pdf`
**Authors**: Houssel, Singh, Layeghy, Portmann (University of Queensland, Australia)
**Venue**: Computers & Electrical Engineering, 2026 (journal follow-up to F3-1 BDCAT 2024)
**GitHub**: https://github.com/Paulpey13/eX-NIDS

### Detection Method
- NOT a new detector — evaluation of LLM-generated explanations for NIDS decisions
- **Augmented-Prompt Explainer** (eX-NIDS): enriches the LLM prompt with three knowledge sources before asking for explanation
- Compares **Basic-Prompt Explainer** vs. **eX-NIDS Augmented-Prompt Explainer**
- Three augmentation layers:
  1. **NetFlow Specification**: field definitions, units, value ranges (e.g., what L4_PROTO=6 means)
  2. **IP-Specific Knowledge**: threat intelligence context, geolocation lookups, connection history for source/destination IPs
  3. **Protocol-Specific Knowledge**: Layer 7 app name mappings (L7_PROTO=7 → IPP, not HTTP), Layer 4 mappings
- NOT using RAG — augmentation uses structured database lookups (traditional DB, not vector search)

### Dataset
- Same as F3-1: **NF-UNSW-NB15-v2** and **NF-CSE-CIC-IDS2018-v2**
- Evaluation set: 50 random NetFlow samples (mix of benign and attacks)
- Human annotation: ground-truth explanations created by domain experts for quantitative scoring

### Preprocessing & Feature Extraction
- NetFlow features as key-value text pairs (same as F3-1)
- IP-specific context retrieved from external threat intel / geolocation DB
- Protocol numbers resolved via L7/L4 mapping tables

### Models / Architecture
- **LLaMA3-70B** (local deployment): large open-source model
- **GPT-4** (OpenAI API): commercial frontier model
- Both models tested with Basic-Prompt and Augmented-Prompt (eX-NIDS)
- Evaluation framework: 3 human-defined metrics scored per explanation

### Training / Evaluation
- **No fine-tuning** — pure prompting evaluation
- Explanation quality scored by human annotators on 3 axes:
  1. **Correctness**: does the explanation correctly identify the attack type and reasoning? (0-100%)
  2. **Feature Consistency**: does the explanation only reference features that are actually in the NetFlow record? (0-100%)
  3. **Factual Consistency**: are protocol/port/service descriptions factually accurate? (0-100%)

### Key Results — Explanation Quality

**Basic-Prompt Explainer:**
| Model | Correctness | Feature Consistency | Factual Consistency | Average |
|-------|-------------|--------------------|--------------------|---------|
| LLaMA3-70B | ~16% | ~90% | ~70% | ~58.67% |
| GPT-4 | ~60% | ~95% | ~80% | ~78.33% |

**eX-NIDS Augmented-Prompt Explainer:**
| Model | Correctness | Feature Consistency | Factual Consistency | Average |
|-------|-------------|--------------------|--------------------|---------|
| LLaMA3-70B | **36%** | **100%** | **90%** | **75.33%** |
| GPT-4 | **80%** | **100%** | **92%** | **90.66%** |

- **GPT-4 eX-NIDS: +20% improvement** in average score over Basic-Prompt (78.33% → 90.66%)
- **LLaMA3-70B eX-NIDS: +16.66% improvement** (58.67% → 75.33%)
- Feature Consistency reaches 100% with augmentation (LLM stops hallucinating non-existent features)
- Factual Consistency improves dramatically: protocol/service confusion eliminated by lookup injection

### Inference Latency & Cost
| Model | Mode | Latency | Cost |
|-------|------|---------|------|
| LLaMA3-70B | Local (eX-NIDS) | 4–6 s / explanation | Hardware only |
| GPT-4 | API (Basic) | ~2–3 s / explanation | ~$3.50–5.00 / 1000 |
| GPT-4 | API (eX-NIDS) | ~3–4 s / explanation | **$5.75–10.37 / 1000** |

- Augmented prompt is longer → more tokens → higher API cost
- Local LLaMA3-70B is cost-free but slower and less accurate

### Common Hallucination Errors (pre-augmentation)
- **TCP flag misinterpretation**: e.g., treating FIN+ACK as aggressive scanning rather than normal teardown
- **Wrong time unit conversion**: ms → minutes (off by factor of 60,000)
- **Wrong data rate units**: confusing Bps with bps (off by factor of 8)
- **Protocol number confusion**: layer 7 protocol IDs misidentified (7 ≠ HTTP)
- **IP geolocation errors**: wrong country attributed to IP addresses
- **Port 0 assumption**: incorrectly flagged as malicious

### Tools / Frameworks
- LLaMA3-70B via local inference (hardware not specified)
- OpenAI API (GPT-4)
- Custom DB lookups for threat intel + protocol mappings (NOT vector RAG)
- Human annotation evaluation framework

### Advantages
- First quantitative evaluation framework for LLM-generated NIDS explanations
- Shows augmented prompting is sufficient for factual consistency (no fine-tuning needed)
- Identifies specific, fixable hallucination categories
- GPT-4 achieves 90.66% average — close to expert quality
- Feature consistency of 100% achievable through structured augmentation

### Limitations
- 70B parameter model still only achieves 36% correctness — gap to GPT-4 (80%) is large
- GPT-4 API cost ($10/1000 queries) is non-trivial for production
- Evaluation set is small (50 samples) — may not generalize
- No automated evaluation — requires human annotation (expensive to scale)
- Does not test smaller models (7B, 13B) — gap between 70B and GPT-4 may be bridgeable

### SLM Relevance
⭐⭐⭐⭐⭐ **VERY HIGH** — Directly relevant to SLM explanation quality. Key lesson: augmented prompts with protocol/service/IP context are ESSENTIAL for accurate explanations — raw NetFlow → hallucinations. For slow-rate DoS, inject: HTTP behavior norms (expected packet rate, connection duration), Slowloris/RUDY attack signatures, TCP window semantics. The 3-metric evaluation framework (correctness, feature consistency, factual consistency) should be adopted as the evaluation protocol for the SLM explanation component. Also: smaller models (7B) were NOT tested — gap from 70B to GPT-4 suggests that for slow-rate DoS with highly specialized context, a fine-tuned 7B SLM may approach GPT-4 quality with domain injection.

---

## PAPER F3-5 — ShieldGPT: An LLM-based Framework for DDoS Mitigation

**File**: `sheildgpt.pdf`
**Authors**: Wang, Zhao, Liu, Zhang, Shi, Liu, Lin (Tsinghua University, Zhongguancun Laboratory, Huawei Technologies)
**Venue**: APNet 2024 (ACM SIGCOMM Asia-Pacific Networking Conference)

### Detection Method
- **3-component system**: Detection (YaTC) + Explanation (GPT-4) + Mitigation (CLI command generation)
- **YaTC** (Yet Another Traffic Classifier): masked autoencoder transformer for traffic representation
  - Trained on **CIC-DoS2017** dataset
  - Input: traffic representation = Global features (flow stats) + Local features (first 5 packets)
  - Output: attack type label
- **GPT-4** receives detected label + traffic features + domain knowledge → natural language explanation + mitigation commands
- End-to-end flow: YaTC detects → label + feature summary sent to GPT-4 → explanation + device-specific CLI commands generated

### Dataset
- **CIC-DoS2017** (CICIDS 2017 DoS subset): 7 application-layer DoS attack types
  - Slowloris, RUDY (R-U-Dead-Yet), Slowbody, Slowheaders, Slowread
  - GoldenEye, Hulk (volumetric variants also included)
- Dataset from Canadian Institute for Cybersecurity (publicly available)

### Preprocessing & Feature Extraction
- **Traffic Representation Schema**:
  - **Global features** (flow-level): flow duration, total bytes, total packets, packet rate, byte rate, inter-arrival time stats
  - **Local features** (per-packet for first 5 packets): packet size, timestamp delta, TCP flags, TCP window size, payload content (first N bytes)
- Local features capture fine-grained behavioral signatures that distinguish slow-rate attack subtypes
- No explicit feature verbalization in preprocessing — features encoded numerically for YaTC; serialized for GPT-4 prompt

### Models / Architecture
- **YaTC detector** (masked autoencoder transformer):
  - Pretrained on large unlabeled traffic corpus (self-supervised)
  - Fine-tuned on CIC-DoS2017
  - F1 ≥ 95% on all attack types
- **GPT-4** (via API): explanation + mitigation generator
  - Receives: attack label + flow feature summary + domain knowledge injection
  - Domain knowledge: attack descriptions + target device specification (Cisco IOS / Snort / iptables)
  - Generates: explanation paragraph + device-specific CLI mitigation commands

### Training / Evaluation
- YaTC fine-tuned on CIC-DoS2017
- GPT-4 prompted (no fine-tuning)
- Evaluation: YaTC detection F1 per attack class; qualitative analysis of GPT-4 explanations/commands

### Key Results — YaTC Detection F1 (CIC-DoS2017)
| Attack Type | F1 Score |
|-------------|---------|
| Slowloris | **0.996** |
| RUDY | 0.983 |
| Slowbody | 0.964 |
| Slowheaders | 0.993 |
| Slowread | **0.996** |
| GoldenEye | ≥0.95 |
| Hulk | ≥0.95 |

**All slow-rate attack types: F1 ≥ 0.964** — state-of-the-art detection accuracy

### GPT-4 Explanation Quality (Qualitative)
**Slowloris explanation**: GPT-4 correctly identifies:
- "Low byte rate + excessively long flow completion time + sequential '\r\n' in HTTP header fragments"
- "Attacker intentionally sends partial HTTP headers to hold connections open"

**Slowbody (RUDY) explanation**: GPT-4 correctly identifies:
- "Large Content-Length declaration (4096 bytes) but actual payload is very small"
- "Low Packet Rate (0.644 pps) — deliberate slow data submission"
- "Incomplete POST body — Content-Length never fulfilled"

**Slowheaders explanation**: GPT-4 correctly identifies:
- "Incomplete HTTP header fields sent at very low rate"
- "Connection maintained alive without completing header exchange"

**Slowread explanation**: GPT-4 correctly identifies:
- "Abnormally small TCP receive window advertised to server"
- "Server forced to send data slowly or stall"
- "Flow duration extends disproportionately relative to data volume"

### Generated CLI Mitigation Commands (examples)
**Cisco IOS** (for Slowloris):
```
ip http timeout-policy idle 15 life 30 requests 1000
```
**Snort** (for RUDY/Slowbody):
```
alert tcp any any -> $HTTP_SERVERS 80 (msg:"Slowbody Attack"; content:"Content-Length"; http_header; threshold: type both, track by_src, count 5, seconds 60; sid:100003;)
```
**iptables** (for Slowread):
```
iptables -A INPUT -p tcp --dport 80 -m connlimit --connlimit-above 10 -j DROP
```

### Domain Knowledge Injection Format
- **Attack description**: natural language description of attack mechanism (Slowloris, RUDY, etc.)
- **Device specification**: target network device type and OS (Cisco IOS, Snort rules, iptables)
- **Context window**: flow feature summary + first 5 packet details serialized as structured text

### Tools / Frameworks
- YaTC (masked autoencoder transformer) — custom implementation
- OpenAI API (GPT-4)
- CIC-DoS2017 dataset (Canadian Institute for Cybersecurity)
- Python, PyTorch

### Advantages
- **Only paper tested on ALL 5 slow-rate DoS types from CIC-DoS2017** with explicit F1 per class
- YaTC achieves state-of-the-art detection (F1≥0.964 for all slow-rate types)
- GPT-4 correctly identifies distinguishing features of EACH slow-rate attack subtype
- Generates device-specific actionable CLI commands — beyond explanation into response
- Local feature representation (first 5 packets) captures early attack detection signals

### Limitations
- GPT-4 API required — not deployable locally without API access/cost
- YaTC is a complex transformer (not interpretable by itself — requires LLM for explanation)
- No quantitative evaluation of explanation quality (qualitative only)
- Mitigation commands are generated but not automatically deployed or tested
- CIC-DoS2017 is a lab dataset — real-world generalization not validated

### SLM Relevance
⭐⭐⭐⭐⭐ **CRITICAL** — This is the MOST DIRECTLY RELEVANT paper to the proposed SLM slow-rate DoS system. Key takeaways:
1. **CIC-DoS2017 is the right dataset** — contains all 5 slow-rate attack types with verified F1 scores
2. **Traffic representation**: Global flow stats + first 5 packet details (size, timestamp, TCP flags, window size, payload) is the optimal feature schema for slow-rate detection
3. **GPT-4 explanation quality**: proves that with domain knowledge injection, LLM can correctly interpret slow-rate DoS patterns — replace GPT-4 with fine-tuned SLM
4. **Mitigation command generation**: future extension for the SLM system
5. **Benchmark target**: F1≥0.964 for Slowbody (hardest) sets the detection performance bar

---

## PAPER F3-6 — Rethinking On-Device LLM Inference for Edge DDoS Detection

**File**: `Rethinking On-Device.pdf`
**Authors**: Pan, Huang, Zhao, Chou, Guo (San Francisco State University / Stanford University / UT Dallas / Virginia Tech)
**Venue**: arXiv:2601.XXXXX, January 2026 (preprint)

### Detection Method
- **On-Device LLM (ODLLM)** for IoT edge DDoS detection — no cloud dependency
- Core question: can tiny LLMs (1B–4B) perform real-time DDoS detection on edge devices?
- **2-stage pipeline**:
  1. **Offline**: Teacher LLM (large model) generates CoT-labeled knowledge base (KB) from labeled training examples
  2. **Online**: XGBoost retriever finds K most similar exemplars from KB → exemplar prompts + CoT reasoning → ODLLM classifies
- Key finding: **Few-shot RAG (analogical mapping) >> Chain-of-Thought (abstract reasoning)** for structured numeric network data

### Dataset
- **CICIoT2023** (CICIOT 2023 — IoT traffic with DDoS attacks)
- 6 traffic classes: Benign + 5 DDoS subtypes
- Features: 9 features selected (proto, packet_rate, IAT, avg_payload_length, 5 TCP flag counts)

### Preprocessing & Feature Extraction
- **9-feature vector**: 
  1. `proto` (TCP/UDP/ICMP protocol number)
  2. `packet_rate` (packets per second)
  3. `IAT` (inter-arrival time, mean)
  4. `avg_payload_length` (bytes)
  5–9. TCP flag counts: SYN, FIN, RST, PSH, ACK (per flow)
- Features serialized to natural language text for LLM input
- XGBoost retriever uses same 9 features to find similar flows in KB

### Models / Architecture
- **ODLLMs tested** (via Ollama, NVIDIA RTX 4090):
  - LLaMA3.2-1B, LLaMA3.2-3B
  - Gemma3-1B, Gemma3-4B
- **Teacher LLM** (KB generation): larger model (not specified — GPT-4 class implied)
- **Retriever**: XGBoost (beats BERT-based BGE embeddings for numeric network features)
- **4 inference strategies**:
  1. **No KB** (zero-shot): LLM classifies from feature text alone
  2. **CoT only**: LLM given chain-of-thought reasoning template, no examples
  3. **Few-shot only**: LLM given K exemplars, no CoT reasoning
  4. **Few-shot + CoT**: exemplars WITH CoT reasoning traces (best strategy)

### Training / Evaluation
- Teacher KB generation: offline, one-time cost
- ODLLM inference: online, on-device
- Metric: Macro-F1 (6-class)
- Hardware: NVIDIA RTX 4090 (evaluation); edge deployment target: IoT gateway/router

### Key Results — Macro-F1 by Model × Strategy

| Model | No KB | CoT only | Few-shot only | Few-shot + CoT |
|-------|-------|----------|---------------|----------------|
| LLaMA3.2-1B | 0.05 | 0.12 | 0.35 | 0.50 |
| LLaMA3.2-3B | 0.20 | 0.25 | 0.60 | **0.75** |
| Gemma3-1B | 0.10 | 0.08 | 0.30 | 0.45 |
| Gemma3-4B | 0.51 | **0.05** | 0.70 | **0.85** |

**Critical observations**:
- **CoT alone WORSE than No KB** for most models (Gemma3-4B: CoT=0.05 vs No KB=0.51 — catastrophic degradation)
- **Few-shot dramatically better than CoT**: exemplar analogy > abstract reasoning for numeric data
- **3B+ parameter threshold**: reliable TCP subtype discrimination requires ≥3B params
- **Best overall**: Gemma3-4B one-shot: F1=0.85; LLaMA3.2-3B few-shot+CoT: F1=0.75

### Retriever Comparison: XGBoost vs. BERT-based BGE
| Retriever | LLaMA3.2-3B F1 | Gemma3-4B F1 |
|-----------|----------------|--------------|
| BGE (BERT embeddings) | 0.55 | 0.72 |
| **XGBoost** | **0.75** | **0.85** |

- **XGBoost retriever significantly outperforms semantic embeddings** for network feature similarity
- Reason: BGE embeddings capture semantic text similarity, not numeric feature proximity — wrong similarity metric for network flows
- XGBoost tree splits capture feature value thresholds that are directly relevant to traffic classification

### Why CoT Fails for Network Traffic
- CoT prompts ask the LLM to reason abstractly ("if packet_rate > X then...")
- LLMs lack the calibrated numeric intuition for network traffic thresholds
- Exemplar-based few-shot provides concrete analogical grounding: "this flow is similar to THIS confirmed DDoS flow"
- CoT can actively mislead the model by triggering incorrect reasoning chains

### Tools / Frameworks
- Ollama (local LLM inference framework)
- LLaMA3.2-1B/3B, Gemma3-1B/4B (open-source models)
- XGBoost (retriever for KB exemplar matching)
- BGE embeddings (BERT-based, tested as retriever — underperforms XGBoost)
- NVIDIA RTX 4090 (evaluation hardware)

### Advantages
- No cloud dependency — fully on-device inference for privacy-sensitive IoT
- Teacher KB generation is offline and one-time cost
- XGBoost as retriever is lightweight and interpretable
- Identifies the correct prompting strategy for numeric network data (few-shot > CoT)
- 4B model achieves F1=0.85 — competitive with some ML baselines

### Limitations
- 4B model minimum for reliable performance — may be too large for constrained edge devices
- Teacher KB generation requires large LLM (initial investment)
- Performance still below tabular ML (XGBoost alone typically F1>0.90 on same data)
- Tested on CICIoT2023 — not validated on slow-rate HTTP attacks (CIC-DoS2017)
- RTX 4090 used for evaluation — not an actual edge device

### SLM Relevance
⭐⭐⭐⭐⭐ **VERY HIGH** — Critical design lessons for SLM slow-rate DoS system:
1. **Use few-shot exemplars, NOT CoT alone** — for numeric network features, analogical reasoning works, abstract reasoning fails
2. **Use XGBoost as retriever** for KB exemplar selection (not BERT embeddings) — numeric feature proximity matters
3. **3B minimum** for reliable subtype discrimination (Slowloris vs. RUDY vs. Slowread — subtle TCP differences)
4. **2-stage pipeline**: Teacher LLM generates labeled explanations offline → SLM uses exemplars online
5. **CoT+few-shot can be combined**: teacher provides CoT traces as part of exemplar labels for best results
6. The 9-feature vector (proto + rate + IAT + payload + TCP flags) maps directly to slow-rate DoS features

---

## Paper F3-7 — Interpretable Anomaly-Based DDoS Detection in AI-RAN with XAI and LLMs

**File**: `Interpretable Anomaly-Based DDoS Detection in AI-RAN with XAI and LLMs.pdf`
**Authors**: Sotiris Chatzimiltis, Mohammad Shojafar, Mahdi Boloursaz Mashhadi, Rahim Tafazolli — University of Surrey (5G/6GIC), UK
**Venue**: arXiv:2507.21193v1 [cs.CR] (July 2025)
**Domain**: 5G/6G Open RAN security; DDoS detection via UE radio telemetry

### Problem & Motivation
Existing IDS work in Open RAN either does detection or explainability, not both integrated with automated LLM reasoning. Additionally, most work ignores the strict near-real-time constraints of Near-RT RIC xApp deployment. This paper bridges that gap with an LSTM + LIME/SHAP + LLM pipeline deployed within the Open RAN controller hierarchy.

### System Architecture
Two-component framework within O-RAN:
- **xApp: IDS** (Near-RT RIC) — receives KPMs per UE via E2 interface → MinMax-normalises → forms 3-timestep windows → LSTM binary classifier → forwards prediction + input to rApp
- **rApp: XAI + LLM** (Non-RT RIC) — applies LIME and SHAP to explain the LSTM prediction → LLM converts technical attribution scores into human-readable natural language with mitigation recommendations
- Offline model weights are periodically retrained in Non-RT RIC and pushed to xApp (Step 0)
- rApp is deliberately Non-RT (not xApp) because LLM inference latency exceeds Near-RT RIC timing budget; rApp is advisory, not control-path

### Dataset: NCSRD (National Centre of Scientific Research "Demokritos")
- Real-world 5G testbed, 3GPP-compliant, 3 cells, 9 UEs, Amarisoft hardware
- 686,009 KPM reports total: **674,553 benign** (98.3%), **11,456 malicious** (1.7%)
- 5 DDoS subtypes: SYN Flood (1,402), ICMP Flood (3,756), UDP Fragmentation (1,402), DNS Flood (1,399), GTP-U Flood (3,497)
- KPMs sampled every 5 seconds per UE; 14 features used after preprocessing

### Feature Set: 14 KPMs
`epre`, `pusch_snr`, `p_ue`, `ul_mcs`, `cqi`, `ul_bitrate`, `dl_mcs`, `dl_retx`, `ul_tx`, `dl_tx`, `ul_retx`, `dl_bitrate`, `dl_err`, `ul_err`

**EDA top discriminators (% diff attack vs. normal)**:
| Feature | Diff (%) |
|---|---|
| ul_err | +2,906% |
| dl_err | +546% |
| dl_bitrate | −96% |
| ul_retx | +89% |
| dl_tx | −51% |
| ul_tx | +28% |

Attack signature: uplink errors spike massively, downlink bitrate collapses, uplink retransmissions surge.

### Detection Model: LSTM
- Architecture: Input shape (3, 14) → 32-unit LSTM → Dense(1, sigmoid)
- Training: Adam optimizer, binary cross-entropy loss, batch=64, early stopping (patience=3), 80-20 split, MinMax normalisation
- **Window size = 3, past data ratio = 0.3** (optimal after grid search)
- Past data ratio mitigates **catastrophic forgetting**: without it, Day 4 F1=0.36; with ratio=0.3, all days F1 ≥ 0.96
- Inference: **0.03 ms/sample** (~36K FLOPs) on Intel i7-10700 CPU — suitable for near-real-time deployment

**Performance (window=3, ratio=0.3)**:
| Model | F1 | FPR (%) | FNR (%) |
|---|---|---|---|
| kNN [Christopoulou] | 1.00 | 0.02 | 0.15 |
| XGBoost [Christopoulou] | 1.00 | 0.16 | 2.54 |
| CNN [Xylouris] | 0.93 | N/A | N/A |
| LSTM [Xylouris] | 0.90 | N/A | N/A |
| **Proposed LSTM** | **0.98** | **0.05** | **6.31** |

Note: kNN achieves F1=1.00 but classifies each instance independently (no temporal modeling) and is expensive at inference; proposed LSTM captures temporal evolution and runs at 0.03 ms.

**Catastrophic forgetting ablation** (per-day F1):
| Ratio | Day 1 | Day 2 | Day 3 | Day 4 |
|---|---|---|---|---|
| 0.0 | 0.69 | 0.64 | 0.70 | 0.36 |
| 0.3 | 0.99 | 0.96 | 0.99 | 0.98 |

### XAI Methods
- **LIME**: Constructs local surrogate sparse linear model by perturbing input; outputs human-readable feature rules (e.g., "ul_bitrate_t0 > 0.22 contributes +0.097 to anomaly prediction")
- **SHAP (Kernel SHAP)**: Game-theoretic Shapley values; computationally approximated via weighted linear regression on feature subsets; provides both local (per-instance heatmap) and global (mean absolute) importance
- Global SHAP top features: `dl_tx_T0`, `ul_tx_T2`, `ul_retx_T1` — most discriminative features cluster at T1 and T2 (later timesteps carry more signal)
- Local SHAP for TP: strong red (attack) contributions from ul_bitrate, ul_tx, ul_retx, dl_tx; for TN: strong blue (normal) from dl_tx

### LLM Interpretability Module
**Prompt structure** (both zero-shot and few-shot):
1. General Feature Statistics table (normal vs. attack means/std for all 14 features)
2. LSTM Input Sequence (3×14 matrix)
3. Model Output (binary: normal/anomalous)
4. Local Explanation Tables (LIME contributions + SHAP heatmap)
5. Global SHAP Feature Importance table
6. Task Instructions: produce readable summary + assess misclassification likelihood + suggest mitigations

**LLMs evaluated**:
- GPT-4-Turbo (~1.7T params)
- DeepSeek-V3-R1 (~671B)
- Mistral-Large-Instruct-2411 (~123B)
- Gemini-2.0-Flash

**Readability metrics** (py-readability-metrics library):
- Flesch Reading Ease (higher = easier)
- Gunning Fog Index (U.S. grade level)

**Key readability findings** (Table VII):
- Enabling reasoning mode consistently improved clarity — GPT-4-Turbo went from Flesch 24.67 (no-shot) to 60.39 (with reasoning mode)
- Few-shot prompting did NOT universally improve readability; for OpenAI it made outputs more complex
- DeepSeek with reasoning: Flesch improved, Fog stayed at college level — reasoning stabilises both models
- Mistral and Gemini showed moderate gains under few-shot

**Zero-shot output example** (TP instance): Structured 5-bullet anomaly summary, misclassification likelihood ~15-25%, 5 mitigation steps
**Few-shot output example**: More detailed, with explicit LIME/SHAP evidence cited per bullet, sub-nested reasoning

### Preprocessing
- Remove singular/irrelevant columns (IP addresses etc.)
- Drop NaN rows
- Select continuous-transmission periods
- MinMax normalisation → sequences of shape (window_size=3, 14)

### Tools & Frameworks
- Python (pdfminer, py-readability-metrics)
- LIME library, Kernel SHAP
- GPT/DeepSeek/Mistral/Gemini APIs
- Open RAN: xApp + rApp within Near-RT RIC / Non-RT RIC / SMO

### Advantages
- End-to-end pipeline: detection → XAI → natural language explanation + mitigations
- 0.03ms inference — compatible with Near-RT RIC latency budget
- Temporal dependency capture via LSTM (competitor kNN is instance-independent)
- Past data ratio elegantly mitigates catastrophic forgetting without replay buffers
- Both LIME and SHAP provide complementary local explanations (rule-based vs. score-based)
- Future work explicitly names "low-rate or slow DDoS" as extension target

### Limitations
- kNN achieves F1=1.00 while proposed LSTM achieves 0.98 — some accuracy trade-off for temporal modeling
- FNR of 6.31% — meaningful in security contexts
- LLM inference is Non-RT only (latency too high for xApp)
- Evaluated on NCSRD (5G flooding DDoS) — not HTTP slow-rate attacks (Slowloris/RUDY)
- LLM readability improvements are inconsistent across prompting strategies and model families

### SLM Relevance
⭐⭐⭐⭐⭐ **VERY HIGH** — Direct architectural template for the SLM-based system:
1. **LSTM as detector** with window_size=3 and past data ratio to prevent catastrophic forgetting — directly adoptable
2. **LIME+SHAP as XAI bridge** between LSTM and SLM explanation stage — validated on 14 radio features; analogous to HTTP flow features
3. **Two-stage split**: detector runs fast (0.03ms, Near-RT); explainer (SLM) runs slower (Non-RT) — realistic deployment model
4. **Prompt structure** (feature stats + LSTM input + XAI output + instructions) is a concrete template for SLM prompting
5. The 14 KPM features map conceptually to slow-rate HTTP features: ul_err→error rate, dl_bitrate→throughput, ul_retx→retransmission count
6. **Future work explicitly targets slow-rate DDoS** — this paper is a direct predecessor

---

## Paper F3-8 — IDS-Agent: An LLM Agent for Explainable Intrusion Detection in IoT Networks

**File**: `llm agent.pdf`
**Authors**: Anonymous (under double-blind review at ICLR 2025)
**Domain**: IoT network intrusion detection; LLM agents; zero-day attack detection

### Problem & Motivation
ML-based IDSs lack explanation and cannot handle zero-day attacks (fixed output label space). LLM-only approaches (e.g., vanilla GPT-4 in-context learning) fail on complex, diverse datasets. This paper proposes the **first LLM agent for IDS**, combining iterative reasoning with a specialized toolbox and memory system.

### System Architecture: IDS-Agent
Inspired by ReAct (Reasoning + Acting) paradigm. For each input network traffic instance + user request, IDS-Agent iterates over three steps until producing a final JSON answer:

1. **Reasoning**: Core LLM generates a thought `r_i = LLM(s_i)` where `s_i` = short-term memory (all prior reasoning + actions + observations in current session)
2. **Action generation**: LLM generates a structured JSON action `a_i = LLM(r_i, s_i)` specifying tool name + parameters
3. **Observation update**: Tool executes and returns observation `o_i` in plain text, appended to short-term memory

**Termination**: When observation contains a "Final Answer:" JSON block with predicted labels + explanation.

### Action Space & Toolbox
| Action | Description |
|---|---|
| Data Extraction | Extracts flow record by line number or flow ID from CSV |
| Preprocessing | Feature scaling, encoding, F-test feature selection, standardization |
| Classification | Runs any of 6 ML models; returns top-3 labels + confidence scores |
| Knowledge Retrieval | Google/Wikipedia API for external knowledge; ChromaDB RAG for internal KB |
| Long Memory Retrieval | Retrieves top-k relevant past sessions from long-term memory base |
| Aggregation | Core LLM integrates all classifier results + knowledge + memory into final structured decision |

**6 ML classifiers**: Random Forest, K-Nearest Neighbors (KNN), Logistic Regression (LR), Decision Tree (DT), Multi-Layer Perceptron (MLP), Support Vector Classifier (SVC) — all pretrained on 10% of training data; output top-3 label predictions with confidence scores.

**Knowledge base**: 50 online blogs + 50 research papers on IoT attacks → chunked (1000 tokens, 200-token overlap) → embedded with OpenAI encoder → stored in ChromaDB vector DB.

### Memory & Knowledge Base
**Short-term Memory (STM)**: Current session reasoning trace R=[r_1,...,r_n], actions A=[a_1,...,a_n], observations O=[o_1,...,o_n]. Tracks iterative state; ensures consistency within session.

**Long-term Memory (LTM)**: Stores correct past decisions φ = {t, x, R, A, O, ŷ} (only sessions validated as correct). Retrieval uses weighted scoring:
```
argmax_j [λ1 · recency(t, t_j) + λ2 · cosim(E(Õ), E(O_j))]
```
where recency r(t,t_j) = 1 − (t−t_j)/max_k(t−t_k). **Best: λ1=λ2=0.5**, k=5.

**External Knowledge**: Same ChromaDB vector DB queried by LLM-generated queries; retrieved chunks compressed by LLM before use in observation update.

### Datasets
**ACI-IoT'23**: IoT traffic with multiple attack types including Slowloris (DoS), ICMP/SYN/UDP Flood, Reconnaissance (Host Discovery, OS Scan, Ping Sweep, Port Scan), Dictionary Attacks. Test: 200 benign + 20 per attack category.

**CIC-IoT'23**: 33 attack types (24 known + 9 unknown/zero-day). Includes DDoS-SlowLoris, DDoS-HTTP_Flood. Test: 100 benign + 10 per attack type.

### Results

**ACI-IoT'23** (GPT-4o core LLM):
| Metric | GPT-4o baseline | RF | Majority Vote | IDS-Agent (GPT-4o) |
|---|---|---|---|---|
| Binary Accuracy | 0.721 | 0.890 | 0.960 | **0.965** |
| FAR | 0.497 | 0.060 | 0.020 | **0.030** |
| Multi-class F1 | 0.682 | 0.750 | 0.962 | **0.975** |
| Recall | 0.754 | 0.760 | 0.961 | **0.972** |

**Slowloris F1 = 1.00** on ACI-IoT'23 (Table 6)

**CIC-IoT'23** (GPT-4o):
| Metric | IDS-Agent (GPT-4o) |
|---|---|
| Binary Accuracy | 0.904 |
| FAR | 0.030 |
| Multi-class F1 | **0.750** |

**DDoS-SlowLoris F1 = 0.82** on CIC-IoT'23 (Table 7)

**Zero-day attack detection** (9 unseen attack types from CIC-IoT'23):
| Method | Avg Recall |
|---|---|
| ACGAN | 0.41 |
| RealNVP | 0.47 |
| **IDS-Agent** | **0.61** |

Threshold: classify as "Unknown" if >2 classifiers confidence <0.7. Vuln Scan and SQL Injection had highest recall due to distinct OOD features.

### Ablation Study
| Module | In-Distribution Recall | Zero-Day Recall |
|---|---|---|
| Full IDS-Agent | 0.733 | 0.610 |
| Without KRM (Knowledge Retrieval) | 0.710 | **0.420** |
| Without LMM (Long Memory) | 0.702 | **0.560** |

KRM more important for zero-day; LMM important for temporally-evolving attacks.

**λ sensitivity** (Table 9): λ1=λ2=0.5 gives best accuracy=98.0%, precision=98.2%, recall=97.2%. Biasing toward recency (λ1=0.9) or similarity (λ2=0.9) both degrade performance.

### Detection Sensitivity
Adjustable via system prompt instruction:
| Sensitivity | Attack Recall | Benign Recall | Attack F1 |
|---|---|---|---|
| Aggressive | 0.97 | 0.90 | 0.97 |
| Balanced | 0.95 | 0.96 | 0.96 |
| Conservative | 0.85 | 0.98 | 0.87 |

No retraining required — sensitivity adjustable at inference time via prompt.

### Inference Time
| Method | Avg time/instance |
|---|---|
| GPT-4 in-context (baseline) | 3.36s |
| IDS-Agent (GPT-4o) | **8.65s** |

Additional time due to knowledge retrieval + aggregation; acceptable for real-world IDS.

### Case Studies
- Correctly identifies MITM-ArpSpoofing when 3/6 classifiers vote for it (others vote Benign/Recon-PortScan) — LLM reasons that "MITM appears in 3 classifiers with significant confidence"
- Correctly identifies reconnaissance when 3/6 vote Benign — LLM reasons "Host Discovery + OS Scan both belong to reconnaissance activities"
- These demonstrate that the LLM aggregator goes beyond majority voting by using semantic understanding of attack relationships

### Tools & Frameworks
- Python: scikit-learn (6 ML classifiers), ChromaDB, OpenAI API
- Google/Wikipedia APIs for external knowledge
- ReAct-style agent pipeline
- Structured JSON output format

### Advantages
- First LLM agent for IDS — combines ML classifiers, RAG, memory, and LLM reasoning
- Zero-day detection without retraining (61% recall vs. 41% for ACGAN)
- Adjustable sensitivity via prompt — no expert intervention needed
- Produces structured explanation alongside detection result
- Slowloris F1=1.00 on ACI-IoT'23 — relevant to slow-rate DoS target
- Extensible toolbox — new classifiers can be added without LLM fine-tuning

### Limitations
- 8.65s inference — not suitable for real-time detection (packet-level); OK for flow-level batch
- Relies on GPT-4o (cloud, costly) as core LLM — local SLM untested
- Zero-day recall of 0.61 still leaves 39% undetected
- CIC-IoT'23 performance (F1=0.75) lower than ACI-IoT'23 (F1=0.975) — harder dataset
- Benign recall drops from 0.91→0.86 with zero-day prompt (more false positives)

### SLM Relevance
⭐⭐⭐⭐⭐ **VERY HIGH** — Closest existing work to the proposed SLM-based system:
1. **Agent architecture is directly adaptable**: Replace GPT-4o core LLM with local SLM; ML classifiers remain unchanged
2. **Slowloris F1=1.00** confirms the toolbox approach works perfectly for slow-rate DoS detection
3. **Sensitivity adjustment via prompt** = key SLM design feature (no retraining for different deployment contexts)
4. **Zero-day capability** via low-confidence detection — relevant for new slow-rate variants (e.g., SlowDroid, RUDY variants)
5. **KRM+LMM ablation** quantifies exactly how much each memory type contributes — guideline for SLM resource budgeting
6. **LTM retrieval formula** (λ1=λ2=0.5) is a concrete implementation recipe for SLM memory

---

## Paper F3-9 — Strengthening Human-Centric Chain-of-Thought Reasoning Integrity in LLMs via a Structured Prompt Framework

**File**: `Strengthening Human-Centric Chain-of-Thought Reasoning Integrity in LLMs.pdf`
**Authors**: Jiling Zhou, Aisvarya Adeseye, Seppo Virtanen, Antti Hakkala, Jouni Isoaho — University of Turku, Finland
**Venue**: arXiv preprint, 2025 (funded by EU Horizon Marie Skłodowska-Curie grant 101177564 — HAIF)
**Domain**: LLM prompt engineering; cybersecurity reasoning; CoT quality evaluation

### Problem & Motivation
Chain-of-Thought (CoT) prompting improves LLM reasoning but without structural constraints can introduce hallucinations, reasoning drift, logical leaps, and unverifiable conclusions — especially harmful in security-sensitive contexts. Model scaling and fine-tuning are costly alternatives. This paper proposes a **lightweight prompt engineering framework** of 16 factors to enforce reasoning integrity without model modification.

Three properties of **reasoning integrity** (from human-centred perspective):
1. **Procedural transparency** — reasoning steps are visible and auditable
2. **Evidence alignment** — conclusions trace to observable data features
3. **Workflow consistency** — reasoning follows security analyst decision logic

### The Structured Prompt Framework: 16 Factors in 4 Dimensions

**Dimension 1: Context & Scope Control (F1–F5)**
| ID | Factor | Prompt Level | Purpose |
|---|---|---|---|
| F1 | Role specification (cybersecurity analyst / SOC expert) | S | Defines expert reasoning perspective |
| F2 | Explicit task scope constraints | S | Prevents task drift beyond dataset |
| F3 | Dataset grounding (DDoS dataset only) | S | Restricts reasoning to provided data |
| F4 | Avoid unstated assumptions | S | Reduces unsupported inference |
| F5 | Negative instruction (no inference beyond data) | S | Blocks external knowledge leakage |

**Dimension 2: Evidence Grounding & Traceability (F6–F8)**
| ID | Factor | Prompt Level | Purpose |
|---|---|---|---|
| F6 | Evidence citation requirement | U | Enforces feature → inference mapping |
| F7 | Feature-level anchoring | U | Grounds reasoning in measurable signals |
| F8 | Anomaly justification requirement | U | Requires explicit anomaly explanation |

**Dimension 3: Reasoning Structure & Cognitive Control (F9–F12)**
| ID | Factor | Prompt Level | Purpose |
|---|---|---|---|
| F9 | Output schema enforcement (Obs → Ev → Concl) | S | Standardises reasoning structure |
| F10 | Confidence calibration instruction | S | Encourages uncertainty acknowledgment |
| F11 | Reasoning depth control | S | Prevents overextended reasoning chains |
| F12 | Step-by-step reasoning requirement | S | Ensures transparent logical progression |

**Dimension 4: Security-Specific Analytical Constraints (F13–F16)**
| ID | Factor | Prompt Level | Purpose |
|---|---|---|---|
| F13 | Attack taxonomy alignment | U | Maps reasoning to volumetric/protocol/application layers |
| F14 | Signal-to-noise prioritization | U | Focuses on relevant features |
| F15 | Final answer verification (self-consistency check) | S | Validates final output reliability |
| F16 | Temporal reasoning constraints | U | Enforces trend-based anomaly logic |

Note: S = System prompt (global behavior, structure, calibration); U = User prompt (feature grounding, taxonomy alignment)

### 3 Prompt Types Evaluated
1. **Free CoT**: Step-by-step reasoning with minimal structural constraints (baseline)
2. **Evidence-Locked CoT**: Adds consistency constraints — conclusions must be grounded in given data; reduces hallucinations
3. **Structured Security Reasoning Prompt (SSRP)**: Full SSRP — follows actual security analysis workflow (threat detection → risk analysis → action recommendation); ensures output aligns with human expert decision logic

### Dataset
**DDoS SDN Kaggle dataset** (Kazin 2021): 400 rows selected, 23 columns (3 categorical + 20 numerical features), binary label (0=normal, 1=DDoS attack). Reflects realistic SDN network traffic.

### Models Evaluated
Gemma-2B, Gemma-3B (≈12B equivalent), Gemma-7B (≈27B), Llama-3.2-3B, Llama-3.1-8B, Llama-3.3-70B, Qwen3-4B, Qwen3-8B (≈32B), GPT-OSS-20B, ChatGPT-5.1

Two prompt variants per model: manually-designed prompts (M) and ChatGPT-generated prompts (C)

### Results

**Classification Accuracy Gains** (Table 2, With FW vs. Without FW):
- Gemma-2B: 69.8% → 72.6-75.0% (M/C) — **+4-5%**
- Llama-3B: 73.1% → 76.1% — **+4%**
- Qwen3-8B: 84.6% → 86.1% — **+2%**
- Llama-70B: 91.0% → 92.4% — **+1.3%**
- ChatGPT: highest baseline; framework still adds ~1%

Accuracy improvements: **1-5%** across all models; larger improvements in smaller models.

**Human-Evaluated Reasoning Quality Gains** (Table 3, 5 metrics):
- Gemma-2B Evidence: 0.72 → 1.05 (M) or 1.02 (C) — **~40-46% improvement**
- Llama-3B Evidence: 0.80 → 1.12 (M) or 0.98 (C) — **~37%**
- Larger models (70B): gains ~7-10% — diminishing returns
- **Key finding**: Reasoning quality gains far exceed accuracy gains — structured prompting primarily enhances *explainability* not *classification*

**Inter-rater agreement** (Cohen's κ):
| Dimension | κ |
|---|---|
| Evidence Grounding | 0.87 |
| Faithfulness | 0.84 |
| Structure Compliance | 0.89 |
| Taxonomy Alignment | 0.82 |

All κ > 0.80 = strong agreement → human evaluation is reliable and reproducible.

**Pareto analysis** (Figure 5): Framework consistently shifts all models to upper-right region (better accuracy AND better reasoning). Improvements are predominantly *vertical* (reasoning) rather than *horizontal* (accuracy), confirming explanation quality is the primary beneficiary.

**Ablation findings**: Removing evidence-grounding factors (F6-F8) causes largest drops in Faithfulness and Evidence scores. Structural controls (F9, F12) primarily affect Structure scores. Removing dataset scope constraints (F3, F5) causes reasoning drift and hallucination about traffic patterns.

**Critical operational insight**: A model may maintain high detection accuracy while producing weak, unverifiable reasoning — accuracy alone is insufficient to evaluate IDS trustworthiness.

### Evaluation Metrics
**Classification**: Accuracy, Precision, Recall, F1-score
**Reasoning quality** (manually scored by 2 independent researchers, average of scores):
- Evidence Grounding Accuracy — features explicitly cited
- Reasoning Faithfulness — no hallucinated claims
- Reasoning Structure Compliance — follows Obs→Ev→Concl schema
- Attack Taxonomy Alignment — DDoS types correctly named and described

### Tools & Frameworks
- Local LLM deployment (Ollama / Hugging Face Transformers)
- ChatGPT API for ChatGPT-5.1 and prompt generation
- Manual human evaluation with rubric; Cohen's κ computed
- Python for dataset handling and metric computation

### Advantages
- Lightweight — no fine-tuning, no additional training, no architecture change
- Model-agnostic — validated across 10 different models from 4 families
- Significant gains in smaller models (40% reasoning improvement in 2B-4B range)
- Provides a concrete, 16-factor checklist implementable in any SLM deployment
- Human-centred evaluation methodology with strong inter-rater reliability

### Limitations
- Evaluated on a single dataset (DDoS SDN, 400 rows) — limited generalizability
- ChatGPT-generated prompts outperform manual prompts — human prompt design is suboptimal
- Gains diminish with model size — large models (70B+) benefit less; minimum useful parameter count may exist for security tasks
- Local LLMs still significantly lag cloud models (Llama-70B approaches but doesn't match ChatGPT-5.1)
- No evaluation on slow-rate HTTP attacks (Slowloris, RUDY) — SDN DDoS dataset only

### SLM Relevance
⭐⭐⭐⭐⭐ **VERY HIGH** — Critical design guidance for SLM explanation module:
1. **16-factor framework is a direct implementation checklist** for SLM system prompts — apply F1-F16 verbatim to the SLM explanation stage
2. **40% reasoning gain in small models** (2B-4B) directly addresses the risk of using small local SLMs for explanation
3. **Evidence grounding (F6-F8) is the most critical dimension** — SLM must be explicitly constrained to cite features (e.g., ul_err, dl_bitrate) not hallucinate about general DDoS patterns
4. **Obs→Ev→Concl output schema (F9)** provides a structured explanation format suitable for operator dashboards
5. **Temporal reasoning constraints (F16)** directly applicable to time-series network features (e.g., "ul_retx increased 3× over the 5-minute window")
6. **Dataset grounding (F3)** prevents SLM from importing irrelevant domain knowledge when reasoning about the specific KPMs fed to it

---
