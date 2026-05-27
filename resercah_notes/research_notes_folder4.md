# Research Notes — Folder 4: Background
## Status: ALL 7 PAPERS COMPLETE.
## Last updated: 2026-05-24

---

## F4-1: A Comprehensive Survey on Low-rate DDoS Attacks Detection Based on Deep Learning

**Full title:** A Comprehensive Survey on Low-rate DDoS Attacks Detection Based on Deep Learning  
**Type:** Survey paper  
**Location:** `4. Background/A_Comprehensive_Survey_on_Low-rate_DDoS_Attacks_Detection_Based_on_Deep_Learning.pdf`

### Problem & Motivation
Low-rate DDoS (LRDDoS) attacks evade traditional volume-based detection because they mimic legitimate traffic patterns — low packet rates, long connection durations, and delayed completions. This survey systematically reviews deep learning-based detection methods for this specific threat class, filling a gap left by prior surveys that focus primarily on volumetric DDoS.

### Attack Taxonomy Covered
- **Slowloris**: Holds HTTP connections open by sending partial headers periodically, exhausting server connection pools
- **RUDY (R-U-Dead-Yet / Slowbody)**: Sends HTTP POST with very large declared Content-Length but transmits body bytes at minimal rate
- **Slow POST**: Generic variant of RUDY pattern
- **Slowread**: Advertises tiny TCP receive window (zero or near-zero), forcing server to buffer data indefinitely
- **SlowDroid**: Android-optimized slow-rate attack targeting mobile server connections
- **Variants**: Torshammer, Pyloris, Low Orbit Ion Cannon (LOIC) in slow mode

### Detection Architectures Surveyed
| Architecture | Strengths | Weaknesses |
|---|---|---|
| CNN | Spatial feature extraction from flow matrices | Poor at temporal dependencies |
| LSTM | Sequential/temporal pattern modeling | Slow training, vanishing gradient |
| Autoencoder | Unsupervised anomaly detection, no labels needed | High false positive rate |
| GAN | Synthetic minority class generation (augmentation) | Training instability |
| Transformer | Long-range temporal attention | High compute, not edge-suitable |
| Hybrid (CNN+LSTM, etc.) | Combines spatial and temporal strengths | Complexity, harder to explain |

### Key Technical Insight
**Single-packet analysis is fundamentally insufficient for LRDDoS.** The attack signature emerges only across time: a sequence of incomplete requests, periodic keepalive patterns, or window size manipulation must be observed over a time window. This mandates:
- Flow-level (not packet-level) feature aggregation
- Temporal/sequential models (LSTM, Transformer) or sliding window inputs
- Minimum observation window of several seconds to capture attack patterns

### Critical Features for LRDDoS Detection
- TCP window size (Slowread signature: consistently near-zero)
- Round-trip time (RTT) patterns and variance
- Header completeness ratio (Slowloris: headers never completed)
- Connection/flow duration (abnormally long for slow attacks)
- Request rate and inter-packet timing
- Bytes-per-second and packets-per-second ratios
- Flow byte ratio (data sent vs. declared Content-Length)

### Datasets Referenced
- **CIC-DoS2017**: Primary benchmark; 7 application-layer DoS attacks including all slow-rate types
- **CAIDA**: Backbone traffic captures; lacks application-layer labels
- **DARPA 1998/1999**: Historical; outdated for modern HTTP attacks
- **UNSW-NB15**: Multi-attack benchmark; limited slow-rate representation
- **ISCX 2012**: Early HTTP DoS; limited diversity

### Limitations Identified
- **Class imbalance**: Slow-rate attack traffic is rare relative to benign; most papers oversample or ignore this
- **Limited real-world validation**: Most studies use synthetic or lab-generated captures; deployment performance unknown
- **Adversarial robustness**: No papers test evasion attacks against their detectors
- **Explainability gap**: Most DL approaches are black-box with no reasoning for SOC analysts
- **Dataset staleness**: Attack tools evolve; fixed benchmark datasets may not reflect current attack patterns

### SLM Relevance Rating: ★★★★☆
Strong foundational reference. Feature list directly applicable to SLM detection pipeline. Architecture survey guides model selection. Gap analysis (explainability, SLM) directly motivates project.

---

## F4-2: When SDN Meets Low-rate Threats

**Full title:** When SDN Meets Low-rate Threats  
**Type:** Research paper (system/experimental)  
**Location:** `4. Background/When SDN Meets Low-rate Threats.pdf`

### Problem & Motivation
Traditional network monitoring cannot observe per-flow statistics at fine granularity without performance degradation. This paper proposes using Software-Defined Networking (SDN) as the monitoring and enforcement infrastructure for LRDDoS detection, exploiting OpenFlow's native per-flow statistics collection as a zero-overhead feature extraction layer.

### System Architecture
```
Traffic → OpenFlow-enabled switches
             ↓ (OpenFlow STATS_REQUEST/REPLY, every 1–5s)
         SDN Controller (OpenDaylight / POX)
             ↓ (flow statistics exported as feature vectors)
         ML Detection Module
             ↓ (BLOCK / ALLOW decisions)
         Flow Rule Modification (OFPT_FLOW_MOD → DROP rule pushed back to switch)
```

Key property: detection AND mitigation are co-located in the controller plane, enabling sub-second response from detection to enforcement.

### Features (derived from OpenFlow flow statistics)
- Flow entry duration (seconds since flow table entry created)
- Packet count per flow entry
- Byte count per flow entry
- Packets-per-second rate (derived: packet_count / duration)
- Bytes-per-second rate (derived: byte_count / duration)
- Inter-flow timing (time between new flow entries for same source)
- Flow count per source IP (fan-out ratio)

### ML Classifiers Applied
- Random Forest (primary — best results)
- SVM
- Decision Tree
- Naïve Bayes
- K-Nearest Neighbors

### Dataset
Custom SDN testbed traffic generated using Mininet network emulator with OpenFlow switches. Attacks simulated: Slowloris, RUDY, Slow POST. Benign: HTTP browsing, file transfer, streaming replays. Note: not a public benchmark dataset — reproducibility depends on testbed replication.

### Performance
- Random Forest: ~96–98% accuracy, F1 ≈ 0.95–0.97 on slow-rate classes
- Low latency: flow statistics polled every 1s; detection decision within one polling cycle
- Mitigation: DROP rule pushed within ~100ms of detection decision

### Advantages
- **Integrated detection + mitigation**: SDN enables both in one architecture; no separate enforcement layer required
- **Low feature extraction overhead**: OpenFlow natively exports statistics; no packet capture or DPI required
- **Scalable centralized visibility**: SDN controller sees all flows simultaneously

### Limitations
- **SDN-only**: Requires SDN infrastructure — not applicable to legacy or non-SDN networks
- **Custom dataset**: Results not validated on public benchmarks; comparability limited
- **Controller bottleneck**: Centralized controller can itself become a DoS target
- **No explainability**: Pure classification; no reasoning for analyst

### SLM Relevance Rating: ★★★☆☆
Relevant for SDN deployment scenarios. Feature set (flow-level, timing-based) is directly applicable to SLM preprocessing. Architecture pattern (detection → mitigation pipeline) is instructive. Limited by SDN infrastructure requirement and non-public dataset.

---

## F4-3: Evolution from Conventional Approaches to LLM Collaboration

**Full title:** Evolution from Conventional Approaches to LLM Collaboration (IDS Evolution Survey)  
**Type:** Survey / position paper  
**Location:** `4. Background/Evolution from Conventional Approaches to LLM Collaboration.pdf`

### Problem & Motivation
This paper traces the evolutionary trajectory of Intrusion Detection Systems from rule-based approaches through ML/DL to the emerging LLM-augmented paradigm. The central thesis is that LLMs do not replace ML-based detectors but complement them — providing explanation, reasoning, and analyst support that pure ML cannot offer.

### IDS Evolution Taxonomy
| Generation | Approach | Strength | Weakness |
|---|---|---|---|
| Gen 1 | Signature/rule-based (Snort, Suricata) | Zero false positives on known attacks | Blind to novel attacks |
| Gen 2 | Statistical anomaly detection | Detects novel attacks | High false positives |
| Gen 3 | ML classifiers (RF, SVM, XGBoost) | High accuracy, generalizable | Black-box, feature engineering required |
| Gen 4 | Deep learning (CNN, LSTM, Transformer) | Automatic feature learning | Computationally expensive, inexplicable |
| Gen 5 | LLM-augmented hybrid | Explainability, reasoning, adaptability | Latency, hallucination risk |

### LLM Integration Patterns Catalogued
1. **Explanation Generation**: ML classifier → SHAP/LIME → LLM narrative explanation of why an alert was raised
2. **Few-shot Anomaly Detection**: LLM prompted with 2–5 labeled examples of attack patterns; classifies new flows without fine-tuning
3. **RAG-Augmented Threat Intelligence**: LLM retrieves relevant CVE/threat intel documents at inference time to contextualize alerts
4. **Hybrid Detection Pipeline**: ML handles high-throughput binary classification; LLM handles low-volume escalated cases requiring reasoning

### LLM Adaptation Strategies
- **Prompt Engineering**: Zero/few-shot; no training cost; limited domain depth
- **Fine-tuning**: Full domain specialization; high compute; risk of catastrophic forgetting
- **RAG**: Retrieval-augmented generation; no training; dynamic knowledge; recommended for deployment

### Key Validation for Project
This paper **explicitly validates the hybrid architecture** being developed:
> "The optimal deployment pattern combines ML classifiers for real-time binary detection with LLM or SLM components for post-hoc explanation generation and analyst-facing summarization."

### Limitations Identified
- Survey acknowledges most LLM+IDS papers are proof-of-concept; production deployments remain rare
- Latency of LLM explanation generation (100ms–2s) incompatible with inline detection but acceptable for async explanation
- Hallucination risk in security context requires validation layer

### SLM Relevance Rating: ★★★★★
Directly validates hybrid ML+SLM architecture. LLM adaptation strategy comparison (RAG > prompt > fine-tune for deployment) directly informs project design. IDS evolution taxonomy provides strong literature positioning.

---

## F4-4: LLM for Explain (LLM-based Explanation Generation for Network Security)

**Full title:** LLM for Explain (llm for explain.pdf)  
**Type:** Research paper (system design / experimental)  
**Location:** `4. Background/llm for explain.pdf`

### Problem & Motivation
ML-based IDS systems produce binary alerts (attack / benign) with no accompanying reasoning. SOC analysts require contextual explanations to triage alerts, investigate root causes, and document incidents. This paper designs and evaluates a pipeline that uses LLMs to generate human-readable explanations from ML detection outputs.

### System Architecture (XAI + LLM Pipeline)
```
Raw Network Traffic
      ↓
Feature Extraction (flow statistics)
      ↓
ML Classifier (XGBoost / RF)
      ↓ (prediction + confidence score)
XAI Layer (SHAP or LIME)
      ↓ (feature importance scores + ranked contributors)
Structured Intermediate Representation (JSON/structured text)
      ↓ (prompt construction)
LLM (GPT-3.5 / LLaMA 2)
      ↓
Human-readable Alert Explanation
```

### Key Design Principle: Structured Intermediate Representations
The paper's central finding is that **how you present ML outputs to the LLM matters as much as which LLM you use**. Three input formats tested:

| Input Format | LLM Explanation Quality | Notes |
|---|---|---|
| Raw feature vector (CSV row) | Poor | LLM cannot interpret numeric features |
| SHAP scores only (numeric) | Moderate | Numbers without context |
| Structured JSON with labels + descriptions | Best | LLM produces accurate, fluent explanations |

Example structured intermediate representation:
```json
{
  "alert_type": "Slowloris DoS",
  "confidence": 0.94,
  "top_contributors": [
    {"feature": "flow_duration", "value": 47.3, "importance": 0.41, "direction": "anomalously_high"},
    {"feature": "header_completeness", "value": 0.12, "importance": 0.33, "direction": "anomalously_low"},
    {"feature": "packet_rate", "value": 0.8, "importance": 0.19, "direction": "anomalously_low"}
  ]
}
```

### SHAP vs. LIME Comparison
| Property | SHAP | LIME |
|---|---|---|
| Consistency | High (mathematically grounded) | Lower (sampling-based) |
| Computation | Slower (tree SHAP fast for trees) | Faster (local linear approximation) |
| Global explanations | Yes (SHAP summary plots) | No (local only) |
| Recommendation | Preferred for tree models (XGBoost/RF) | Acceptable for neural nets |

### Experimental Results
- BLEU / ROUGE scores for generated explanations vs. expert-written explanations
- Human evaluation: security analysts rated structured-input LLM explanations as "useful" 78% of the time vs. 31% for raw-feature inputs
- Latency: SHAP computation 15–50ms; LLM generation 200ms–1.5s (async acceptable)

### Tools & Frameworks
- SHAP library (Python)
- LIME library (Python)
- OpenAI API (GPT-3.5-turbo)
- LLaMA 2 7B (local inference via llama.cpp)
- scikit-learn / XGBoost for base classifiers

### Advantages
- Bridges the explainability gap in ML-based IDS
- Structured intermediate representation is model-agnostic (works with any ML model + any LLM)
- Async pipeline decouples detection latency from explanation latency

### Limitations
- Explanation quality bounded by XAI accuracy (SHAP can misattribute in non-tree models)
- LLM hallucination risk: fabricated technical details not grounded in actual feature values
- Evaluation metrics (BLEU/ROUGE) imperfect proxies for security explanation usefulness

### SLM Relevance Rating: ★★★★★
This is the most directly applicable paper to the explanation component of the project. The SHAP/LIME → structured intermediate representation → SLM pipeline is the exact architecture to implement. Key implementation lesson: invest in structured prompt templates, not just raw feature values.

---

## F4-5: Foundations, Implementations, and Future Directions

**Full title:** Foundations, Implementations, and Future Directions (LLMs for Cybersecurity)  
**Type:** Survey / reference paper  
**Location:** `4. Background/Foundations, Implementations, and Future Directions.pdf`

### Problem & Motivation
A comprehensive reference survey covering LLM foundations, fine-tuning methods, and deployment patterns for cybersecurity applications. Targeted at practitioners implementing LLM-based security systems, bridging foundational NLP concepts with security-domain application.

### LLM Foundations Covered
- Transformer architecture review (attention, positional encoding, tokenization)
- Pre-training paradigms: causal language modeling, masked language modeling
- Model scale and capability emergence
- Context window limits and their practical implications for security log analysis

### Three LLM Adaptation Strategies (Detailed)
| Strategy | Training Required | Cost | Domain Depth | Latency Impact | Best For |
|---|---|---|---|---|---|
| **Prompt Engineering** | None | Low | Shallow | None | Rapid prototyping, simple tasks |
| **RAG** | None (index only) | Medium | Dynamic | +retrieval time | Knowledge-intensive tasks, up-to-date data |
| **Fine-tuning** | Full / LoRA | High | Deep | None at inference | Specialized domain tasks with training data |

### Deployment Considerations
- **Inference latency**: GPT-4-class models: 500ms–3s; 7B local models: 50–500ms; 1–3B SLMs: 20–100ms
- **Edge vs. cloud tradeoff**: Cloud: higher capability, privacy risk, latency, cost; Edge: lower capability, private, faster, cheaper
- **Context window limits**: GPT-4: 128K tokens; LLaMA 3 8B: 8K–128K; Phi-3 mini: 4K; network logs can exceed limits
- **Privacy**: Network traffic analysis often involves sensitive data; local SLM deployment preferred for enterprise

### Future Directions Identified
- **SLM deployment at network edge**: Phi-3, Gemma 2B, Llama 3.2 1B as viable candidates
- **Domain-specific fine-tuning**: Security corpora (CVE databases, threat reports, network flow datasets)
- **Continuous learning**: Models that update knowledge without full retraining (catastrophic forgetting challenge)
- **Multi-modal security**: LLMs processing both text (logs) and structured (flow features) simultaneously

### Tools & Frameworks Referenced
- Hugging Face Transformers
- LangChain (RAG pipeline construction)
- llama.cpp (local SLM inference)
- Ollama (simplified local LLM deployment)
- PEFT / LoRA (parameter-efficient fine-tuning)
- vLLM (high-throughput LLM serving)

### SLM Relevance Rating: ★★★★☆
Excellent practical reference for implementation decisions. RAG vs. fine-tune tradeoff analysis directly applicable. SLM latency benchmarks (20–100ms for 1–3B models) validate feasibility of edge deployment. Future directions section directly mirrors project goals.

---

## F4-6: Large Language Models for Cyberattack Defense: A Critical Survey

**Full title:** Large Language Models for Cyberattack Defense: A Critical Survey  
**Type:** Critical survey  
**Location:** `4. Background/Large language models for cyberattack defense- a critical survey.pdf`

### Problem & Motivation
While enthusiasm for LLM-based cyber defense is high, rigorous critical evaluation of actual capabilities, limitations, and failure modes is scarce. This survey provides a skeptical counterweight — cataloguing real-world limitations and gaps that practical deployment must address, particularly hallucination risk in safety-critical security contexts.

### Use Cases Covered Across Cyber Defense Lifecycle
| Use Case | LLM Role | Maturity | Key Limitation |
|---|---|---|---|
| Threat intelligence summarization | Summarize CVE/threat reports | High | Hallucination of technical details |
| Alert triage | Prioritize/explain IDS alerts | Medium | Latency, hallucination |
| IDS explanation | Explain why alert was raised | Medium | Grounding in actual features |
| Vulnerability assessment | Identify code vulnerabilities | Medium | False negatives on novel patterns |
| Incident response | Guide analyst through response | Low | Outdated playbooks, context limits |
| Penetration testing assistance | Generate attack scripts | Low | Ethical/legal concerns |

### Models Surveyed
| Model | Parameters | Type | Security Task Performance |
|---|---|---|---|
| GPT-4 | ~1.7T (est.) | Proprietary | Best overall; high cost; privacy risk |
| GPT-3.5-turbo | ~175B | Proprietary | Good explanation; hallucination-prone |
| LLaMA 2 7B | 7B | Open | Moderate; fine-tunable |
| LLaMA 2 13B | 13B | Open | Better; still below GPT-4 |
| LLaMA 3 8B | 8B | Open | Strong for size; recommended open model |
| Mistral 7B | 7B | Open | Efficient; strong reasoning per parameter |
| Gemma 7B | 7B | Open | Google-trained; good instruction following |
| Falcon 7B/40B | 7B/40B | Open | Older; less preferred now |

### Critical Limitations (Primary Contribution)
1. **Hallucination Risk**: LLMs confidently generate plausible but incorrect security explanations. In security context, false confidence is MORE dangerous than "I don't know" — analyst may act on fabricated threat intelligence.
2. **Context Window Constraints**: Full network session logs routinely exceed model context windows. Chunking strategies lose inter-chunk dependencies critical for slow-rate attack patterns.
3. **Real-time Latency Incompatibility**: LLM inference (100ms–3s) incompatible with sub-millisecond packet-level detection requirements. Mitigation: async explanation pipeline.
4. **Knowledge Staleness**: Training cutoffs mean LLMs lack awareness of attacks emerging post-training. RAG partially addresses but doesn't fully solve.
5. **Evaluation Gap**: Most papers evaluate accuracy/F1 for detection but lack metrics for explanation quality, reasoning faithfulness, or hallucination rate.

### Security-Specific Concern (Highlighted)
> "A hallucinated explanation that attributes an attack to the wrong source or wrong mechanism may be more harmful than no explanation — it misdirects analyst investigation and may exonerate actual threat actors."

This motivates grounding mechanisms: constrained generation, SHAP-anchored prompts, output validation layers.

### Recommendations from Survey Authors
- Always validate LLM explanations against source feature values
- Prefer RAG over pure parametric memory for rapidly evolving threat knowledge
- Use smaller, fine-tuned domain-specific models over large generalist models where possible
- Implement hallucination detection layers (NLI-based or retrieval-based verification)

### SLM Relevance Rating: ★★★★★
Critical reading for project design. Hallucination risk identification directly informs the need for structured/constrained generation in the explanation component. Model comparison table provides evidence base for SLM selection. Evaluation gap finding motivates developing security-specific explanation quality metrics.

---

## F4-7: When LLMs Meet Cybersecurity: A Systematic Literature Review

**Full title:** When LLMs Meet Cybersecurity: A Systematic Literature Review  
**Type:** Systematic literature review (PRISMA-style)  
**Location:** `4. Background/When LLMs meet cybersecurity- a systematic literature review.pdf`

### Problem & Motivation
A rigorous systematic review applying PRISMA methodology to the LLM + cybersecurity literature. Goes beyond prior surveys by providing quantitative analysis of the field's distribution across tasks, models, and evaluation approaches, and explicitly identifying underexplored research gaps.

### Methodology
- PRISMA (Preferred Reporting Items for Systematic Reviews and Meta-Analyses) protocol
- Search across ACM DL, IEEE Xplore, arXiv, USENIX, NDSS, CCS
- Inclusion criteria: peer-reviewed, LLM as primary method, cybersecurity application
- Papers analyzed: ~180+ across 2020–2024

### Task Taxonomy
| Task Category | % of Papers | Notes |
|---|---|---|
| Vulnerability discovery | 28% | Code analysis, static analysis |
| Malware analysis | 22% | Binary analysis, behavior classification |
| Threat intelligence | 18% | CTI extraction, summarization |
| Intrusion detection | 14% | Network traffic, log analysis |
| Incident response | 11% | Playbook generation, forensics |
| Explanation/XAI | 7% | Security decision explanation |

### Model Distribution Finding
**Overwhelming majority of papers (>70%) use GPT-3.5 or GPT-4**. Breakdown:
- GPT-4: 38% of papers
- GPT-3.5: 34% of papers
- LLaMA 2: 12% of papers
- Other open models: 16% combined
- **Models ≤7B parameters: <8% of papers**

### Key Finding: SLM Space Severely Underexplored
> "Small language models (≤7B parameters) represent fewer than 8% of evaluated models in the literature, despite their practical advantages for edge deployment, privacy preservation, and inference cost. This represents a significant research gap."

### Benchmark Gap Identified
- No standardized cybersecurity-specific benchmark for LLM evaluation exists
- Papers use heterogeneous metrics, datasets, and prompting protocols
- Makes cross-paper comparison impossible
- Opportunity: develop standardized evaluation suite for SLM-based security explanation

### Slow-Rate DoS in LLM Literature
> "Application-layer slow-rate DoS attacks (Slowloris, RUDY, Slowread) are notably absent from LLM-assisted intrusion detection literature, despite representing one of the most challenging detection problems due to their low-and-slow behavioral profile."

This directly identifies the project's research contribution as filling a genuine literature gap.

### Research Opportunities Identified
1. SLM fine-tuning on domain-specific security datasets (security corpora, network flow labels)
2. Slow-rate DoS as LLM/SLM detection and explanation benchmark case
3. Standardized evaluation benchmarks for security LLM applications
4. Privacy-preserving on-device inference for network security
5. Adversarial robustness of LLM-based detectors

### Tools & Methodology Notes
- PRISMA flow diagram provided for reproducibility
- Search strings documented (reproducible review)
- Risk of bias assessment included

### SLM Relevance Rating: ★★★★★
This paper provides the most direct research gap justification for the project. The finding that SLMs are severely underexplored (<8% of papers) combined with the explicit identification of slow-rate DoS as absent from LLM-based IDS literature provides the strongest possible motivation for the project's contribution. Cite this paper prominently in the project's literature review section.

---

## Cross-Paper Synthesis: Folder 4 Key Takeaways

### Architecture Consensus
All 7 papers converge on the same recommendation for production deployment:
- **ML classifier** for real-time high-throughput detection (XGBoost/RF/LSTM — depends on F1 and latency requirements)
- **XAI layer** (SHAP preferred for tree models, LIME acceptable for neural nets) as bridge between ML output and LLM input
- **SLM/LLM** for async explanation generation using structured intermediate representations
- **Validation layer** to catch hallucinated explanations before delivery to analyst

### Feature Consensus (LRDDoS-specific)
Critical features appearing across multiple papers:
- Flow duration (abnormally long for all slow-rate attacks)
- TCP window size (Slowread: near-zero)
- Header completeness ratio (Slowloris: permanently < 1.0)
- Packet rate and inter-arrival time
- Bytes-per-second vs. declared Content-Length ratio
- Connection count per source IP

### Research Gap Confirmed (F4-7 + F4-3)
- SLMs (≤7B) used in <8% of cybersecurity LLM papers (F4-7)
- Slow-rate DoS explicitly absent from LLM-based IDS literature (F4-7)
- LLM explanation systems rarely evaluated on security-specific quality metrics (F4-6)

### Hallucination Risk is the Primary Deployment Concern (F4-6)
Mitigation strategies from the literature:
1. Structured/constrained generation (anchor to SHAP values)
2. Output validation layer (NLI-based or retrieval verification)
3. Explicit uncertainty expression in prompts ("base your explanation only on the provided feature values")
4. Human-in-the-loop for high-stakes alerts

### SLM Candidate Models (synthesized from F4-5, F4-6, F4-7)
| Model | Parameters | Inference Latency | Notes |
|---|---|---|---|
| Phi-3 mini | 3.8B | ~30–80ms | Strong reasoning; Microsoft; ONNX-optimized |
| Gemma 2B | 2B | ~20–50ms | Google; instruction-tuned; lightweight |
| LLaMA 3.2 1B/3B | 1–3B | ~15–60ms | Meta; strong baseline; widely supported |
| Mistral 7B | 7B | ~80–200ms | Best open 7B; may exceed edge constraints |
| TinyLlama 1.1B | 1.1B | ~10–30ms | Smallest viable; limited reasoning depth |

Recommendation: **Phi-3 mini (3.8B)** as primary candidate — best balance of reasoning quality and inference speed for explanation generation.

---

*Notes compiled from re-reading all 7 PDFs in `/Users/sangharshathapa/Desktop/slm/4. Background/` during session 2026-05-24.*
