# Deep-Dive Architecture Explanation
## SLM-Based Slow-Rate DoS Attack Detection and Explanation for Resource-Constrained IoT Devices

**Project context:** Encoder-based Small Language Model (SLM) for real-time detection + Decoder-based SLM for reasoning and explanation, deployed on resource-constrained IoT-edge devices using the CIC IIoT Dataset 2025.

---

## Table of Contents

1. [Why This Architecture Exists — The Problem It Solves](#1-why-this-architecture-exists)
2. [Overall System Workflow — End to End](#2-overall-system-workflow)
3. [Preprocessing Pipeline](#3-preprocessing-pipeline)
4. [Feature Engineering and Selection](#4-feature-engineering-and-selection)
5. [Input Representation and Tokenization](#5-input-representation-and-tokenization)
6. [Encoder SLM — Structure and How Detection Happens](#6-encoder-slm)
7. [The XAI Bridge — SHAP and the Attribution Layer](#7-the-xai-bridge)
8. [The Structured Intermediate Representation (SIR)](#8-the-structured-intermediate-representation)
9. [Decoder SLM — Structure and How Reasoning Is Generated](#9-decoder-slm)
10. [How the Encoder and Decoder Communicate](#10-how-encoder-and-decoder-communicate)
11. [Training Pipeline](#11-training-pipeline)
12. [Inference Workflow](#12-inference-workflow)
13. [Optimization for Resource-Constrained Devices](#13-optimization-for-resource-constrained-devices)
14. [Memory and Computational Considerations](#14-memory-and-computational-considerations)
15. [Role of Every Tool, Framework, and Component](#15-role-of-every-tool-framework-and-component)

---

## 1. Why This Architecture Exists — The Problem It Solves

Before explaining the architecture itself, it is essential to understand why it is designed the way it is. The architecture is a direct response to three hard constraints that no single-model approach can satisfy simultaneously.

### The Three Constraints

**Constraint 1: Speed.** IoT network intrusion detection must be near-real-time. A slow-rate DoS attack like Slowloris or RUDY continuously occupies server connections over minutes. If the detection system takes more than a few hundred milliseconds per decision, it cannot act before significant damage is done. Large language models are fundamentally incompatible with this requirement. A LLaMA3-8B model takes approximately 14,000 microseconds (14ms) per inference call on a modern GPU — and on ARM-class embedded hardware (the 14× slowdown rule from the literature), that becomes roughly 200ms just for the model call, before any preprocessing or postprocessing. An XGBoost classifier on the same features takes approximately 2 microseconds. The latency gap is roughly 7,000× (confirmed experimentally in the literature). No LLM can be a real-time classifier on edge hardware.

**Constraint 2: Accuracy with Explainability.** A detection-only system that produces a binary "attack / not attack" label is insufficient for a security operations context. When an IoT intrusion detection system raises an alert, a human analyst needs to understand what caused the alert, which specific attack subtype it is (Slowloris vs. RUDY vs. Slowread), and what to do about it. Classical ML models (XGBoost, Random Forest) are highly accurate but produce decisions that are not naturally human-readable. Pure text-based LLMs, on the other hand, produce verbose outputs but — crucially — achieve only ~50% precision on zero-shot network traffic classification (equivalent to random guessing), because they have no inherent knowledge of how to parse raw numeric network feature vectors.

**Constraint 3: Resource constraints.** The target deployment is IoT edge hardware — devices such as Raspberry Pi 4, NXP i.MX 8M Plus, or NVIDIA Jetson Nano — with 4–8GB RAM, no dedicated GPU, and ARM-class CPUs running at 1–2 GHz. These devices cannot load a 7B-parameter model in full precision. They cannot run PyTorch natively at acceptable speeds. They require quantized, optimized model artifacts specifically built for CPU inference.

### The Architecture's Answer

The system satisfies all three constraints by **separating responsibilities across three cooperating components**:

1. A **classical ML classifier** (XGBoost or Random Forest) handles the real-time detection decision. It is always running, takes microseconds, and is very accurate on flow-level features.
2. An **encoder-based SLM** (compact BERT-class, 11–22M parameters) takes verbalized flow features and produces a richer classification with contextual embeddings. It runs synchronously in near-real-time (~80ms on ARM with INT8 quantization). Its output also drives the XAI attribution layer.
3. A **decoder-based SLM** (4B parameter autoregressive model) receives a structured representation of the detection result and produces a human-readable explanation. It runs **asynchronously** — that is, it does not block the detection pipeline — and takes 1–3 seconds per explanation on CPU.

This three-stage design is consistent across seven papers in the literature that test hybrid ML+encoder+decoder architectures, and every one of them outperforms both pure-ML (which cannot explain) and pure-LLM (which cannot classify accurately or quickly) approaches.

---

## 2. Overall System Workflow — End to End

The following is the complete data flow through the system, from raw network traffic to a human-readable security explanation. Each stage is explained in detail in subsequent sections.

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAW NETWORK TRAFFIC                         │
│              (PCAP files / live interface capture)              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 STAGE A: FLOW GENERATION                        │
│  Zeek IDS processes raw packets → structured conn.log           │
│  Each row = one network flow (connection record)                │
│  Fields: proto, duration, orig_bytes, resp_bytes,               │
│          conn_state, history, orig_pkts, resp_pkts, ...         │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 STAGE B: PREPROCESSING                          │
│  1. Parse conn.log → pandas DataFrame                           │
│  2. Replace inf → NaN → mean imputation                         │
│  3. Drop duplicates, remove flows < 1s duration                 │
│  4. Temporal windowing: group flows into W=15–30s windows       │
│  5. Compute behavioral parameters per window per source IP      │
│  6. MinMaxScaler (encoder input) / StandardScaler (ML input)    │
│  7. SMOTE (training split only) for class balance               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE C: FEATURE SELECTION (offline)               │
│  Mutual Information scoring → rank all features                 │
│  Domain expert review → retain slow-rate-specific features      │
│  Result: 12–15 features retained from original 83              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE D: FEATURE VERBALIZATION                     │
│  Two parallel representations created from same features:       │
│  (a) Tabular vector → ML classifier input                       │
│  (b) Format B key-value text string → SLM encoder input         │
└──────────┬──────────────────────────────────────────────────────┘
           │
           ├────────────────────────────────────────────────┐
           │                                                │
           ▼                                                ▼
┌──────────────────────────┐              ┌─────────────────────────────┐
│  STAGE E1: ML BASELINE   │              │  STAGE E2: ENCODER SLM      │
│  XGBoost / Random Forest │              │  TinyBERT / MiniLM / Securi-│
│  ~1–2µs inference        │              │  tyBERT (11–22M parameters) │
│  Outputs: label +        │              │  ~80ms INT8 ONNX on ARM     │
│  confidence score        │              │  Outputs: label + [CLS]     │
│                          │              │  embedding vector           │
└──────────┬───────────────┘              └──────────────┬──────────────┘
           │                                             │
           └─────────────────────┬───────────────────────┘
                                 │ (both confirm attack)
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STAGE F: XAI BRIDGE                            │
│  SHAP TreeExplainer on ML model output                          │
│  → computes Shapley values per feature for this sample          │
│  → ranks features by absolute importance                        │
│  → records direction (value above/below benign baseline)        │
│  Runtime: <100ms                                                │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│          STAGE G: STRUCTURED INTERMEDIATE REPRESENTATION        │
│  Combines: detection label + confidence + ranked SHAP features  │
│  + domain flags (INCOMPLETE_HEADER, SLOW_RATE, etc.)            │
│  + flow window summary + source context                         │
│  Format: JSON object → serialized as string                     │
│  This is the exact input the decoder will receive               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              │ ← pushed into async queue (non-blocking)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               STAGE H: DECODER SLM (asynchronous)              │
│  Background worker picks up SIR from queue                      │
│  XGBoost retriever fetches closest exemplar from library        │
│  Jinja2 constructs augmented prompt:                            │
│    - SSRP 16-factor system prompt                               │
│    - Domain knowledge (attack descriptions, protocol defs)      │
│    - Retrieved exemplar (Obs→Evidence→Conclusion→Mitigation)    │
│    - SIR JSON block (the actual alert data)                      │
│  Gemma3-4B Q4_K_M via Ollama generates structured explanation   │
│  GBNF grammar constrains output to valid JSON schema            │
│  Runtime: 1–3 seconds                                           │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FINAL OUTPUT                                │
│  {                                                              │
│    "observation": "Flow from 192.168.1.44:54321 to             │
│      server:80 maintained an open TCP connection for           │
│      127.5 seconds with only 3 packets totalling 245 bytes.",   │
│    "evidence": [                                                │
│      "flow_duration = 127.5s (expected < 5s for benign HTTP)", │
│      "pkt_rate = 0.03 pkt/s (Slowloris keepalive pattern)",    │
│      "header_completeness = 0.12 (header never sent fully)"    │
│    ],                                                           │
│    "conclusion": "Slowloris DoS attack — HTTP connection       │
│      exhaustion via incomplete header transmission.",           │
│    "mitigation": "1. Drop TCP connections with duration >30s   │
│      and pkt_rate < 0.1 pkt/s. 2. Enable server-side timeout  │
│      for incomplete HTTP headers (recommended: 10s). ..."       │
│  }                                                              │
│                                                                 │
│  → Security analyst dashboard / SIEM alert log                 │
└─────────────────────────────────────────────────────────────────┘
```

The critical architectural principle is that **Stages E–G are synchronous and fast** (total ≈ 80–200ms), while **Stage H is asynchronous and does not block the detection decision**. The system blocks a suspicious connection immediately when the ML+encoder pipeline raises an alert. The explanation is generated concurrently and delivered to the analyst within 1–3 seconds.

---

## 3. Preprocessing Pipeline

The preprocessing pipeline transforms raw network traffic into structured, normalized, and verbalised inputs suitable for both classical ML classifiers and the encoder SLM. This is one of the most critical stages because slow-rate DoS attacks produce extremely subtle signatures that only become visible after careful temporal aggregation — they look like normal traffic at the individual-packet level.

### 3.1 Flow Generation with Zeek

Raw network traffic arrives as PCAP files (packet capture) or a live network interface. The first processing step is converting individual packets into **flow records** — summaries of entire network conversations between two endpoints.

Zeek IDS is used for this purpose. Zeek processes the PCAP and writes a `conn.log` file where each row represents one complete TCP/UDP/ICMP connection. The fields most relevant to slow-rate DoS detection are:

| Zeek Field | Meaning | Slow-Rate DoS Relevance |
|---|---|---|
| `ts` | Connection start timestamp | Required for temporal windowing |
| `proto` | Protocol (tcp/udp/icmp) | Slow-rate DoS is always TCP |
| `duration` | Total connection duration in seconds | **Abnormally long** for all slow-rate types (60–600s) |
| `orig_bytes` | Bytes sent by originator (attacker) | **Very low** — attacker sends minimal data |
| `resp_bytes` | Bytes sent by responder (server) | Also low — server is waiting |
| `orig_pkts` | Packets sent by originator | **Very few** — 1–3 packets per 15–30s window |
| `orig_ip_bytes` | Total IP-layer bytes from originator | Includes headers; still very low |
| `conn_state` | TCP connection state at termination | **S1/S2 for Slowloris** (established, never completed) |
| `history` | Character-encoded packet history | Encodes SYN/data/FIN sequence — reveals incomplete handshakes |
| `id.resp_p` | Destination port | Port 80/443 = HTTP/HTTPS DoS target |

Zeek is preferred over CICFlowMeter for this project because `conn_state` and `history` are native Zeek fields that directly encode connection state transitions. A Slowloris connection has `conn_state = S1` (SYN seen, SYN-ACK seen, data never completed) and `history = Sh` (SYN from originator only, no data). These fields require no additional computation — Zeek derives them directly from the TCP state machine as it processes packets.

CICFlowMeter is the alternative if comparability with prior CIC-format datasets is required. It generates 83 features including `Flow Duration`, `Fwd Packet Length Mean`, `Fwd IAT Mean`, `SYN Flag Count`, etc. Its disadvantage is that it does not natively produce `conn_state` or `history` equivalents, so these must be derived manually from TCP flag counts.

### 3.2 Parsing and Initial Cleaning

Zeek's `conn.log` is tab-separated with a header. The first cleaning steps are:

```
conn.log → pandas DataFrame
↓
Replace infinite values with NaN
↓
Mean imputation: for each feature, fill NaN with mean of that feature on training set
(mean computed on training data only; applied to val/test using training mean)
↓
Remove duplicate flow records (same 5-tuple + timestamp within 1ms)
↓
Remove flows with duration < 1 second
(legitimate slow-rate DoS flows last tens of seconds to minutes;
sub-second flows are either benign or volumetric, not slow-rate)
↓
Remove IP address columns (src_ip, dst_ip, src_port removed as features
— they cause identity-based overfitting to the specific testbed network)
↓
Retain: dst_port and proto as categorical identifiers
```

The removal of flows shorter than 1 second is particularly important. Slowloris attacks maintain connections for 30–600 seconds. RUDY attacks maintain connections for the duration of a slow POST upload (minutes). Slowread attacks sustain connections indefinitely by reading response data at near-zero speed. Any flow record shorter than 1 second cannot be any of these attack types, so filtering them reduces dataset noise and speeds up all subsequent processing.

### 3.3 Temporal Windowing — The Most Important Preprocessing Step

This is the most critical preprocessing step for slow-rate DoS detection, and it is the step most often omitted in general-purpose IDS literature. **A single flow record does not contain enough information to distinguish Slowloris from normal HTTP traffic.** A single Slowloris connection looks like a slow but legitimate request. The attack signature only emerges when you observe many such connections accumulating simultaneously toward the same server.

Temporal windowing groups flow records into time slices of W=15–30 seconds and computes aggregated statistics per source IP (or per server destination port) within each window:

```
For each source IP in each W=15s window:
  - count_flows: total number of active TCP flows to server
  - count_incomplete: flows with conn_state ∈ {S1, S2, SR, RSTR}
  - mean_duration: average flow duration within window
  - std_duration: standard deviation of flow duration
  - total_orig_bytes: total bytes sent by this IP in window
  - mean_pkt_rate: mean packets per second across all flows
  - min_tcp_window: minimum TCP window size advertised
    (near-zero specifically for Slowread — TCP window advertisement
    controls how much data the sender can send before acknowledgment)
  - header_completeness_ratio: for HTTP flows, ratio of
    (flows with complete HTTP header) / (total flows)
    (Slowloris keeps headers incomplete intentionally)
  - content_length_ratio: for POST flows, ratio of
    (actual bytes received in body) / (Content-Length declared)
    (RUDY sends a large Content-Length but transmits body 1 byte/s)
```

Why W=15s? Paper F1-3 (slowTrack) validates W=15s empirically and achieves F1=99.89% with 8 behavioral parameters computed in this window. Paper F1-6 (FLD-LRDDoS) uses W=30s and achieves 98.79%. The window must be long enough for the slow-rate attack pattern to accumulate (Slowloris keepalive interval is typically 5–15s) but short enough to detect the attack before the server is overwhelmed.

### 3.4 Handling Class Imbalance with SMOTE

Slow-rate DoS attacks represent a small fraction of total IoT network traffic. In a realistic deployment, 95%+ of traffic is benign. A naive classifier trained on this distribution would learn to predict "benign" for everything and still achieve 95% accuracy — useless for security purposes.

SMOTE (Synthetic Minority Oversampling Technique) addresses this by generating synthetic examples of the minority class (slow-rate attack flows) in the feature space. **SMOTE is applied exclusively to the training split.** The validation and test splits retain their original class distribution. This is a strict constraint — applying SMOTE to val/test would produce optimistically biased evaluation metrics.

The SMOTE procedure: for each minority-class sample, find its k nearest neighbors (k=5 by default) in the feature space, randomly pick one neighbor, and create a new synthetic sample at a random point on the line segment between the original sample and that neighbor. This creates plausible but non-duplicated minority class examples.

### 3.5 Normalisation

Two different normalisation strategies are applied depending on the downstream model:

**MinMax scaling [0, 1] for encoder SLM input:**
Feature values are mapped to [0, 1]: `x_scaled = (x - x_min) / (x_max - x_min)`. SecurityBERT (SS-4) and the LSTM model in F3-7 both use MinMax. The reason is that BERT-class models expect input token embeddings in a bounded, consistent range. Extremely large or negative values after verbalization create tokens that fall outside the model's learned embedding space.

**Z-score standardisation for ML classifiers (XGBoost, RF):**
Values are centered: `x_scaled = (x - mean) / std`. XGBoost and RF are technically invariant to monotonic feature transformations (they use rank-based splits), but standardisation improves SHAP value interpretability and is validated in F3-3.

Both scalers are **fit on training data only** and applied using the saved fit parameters to validation and test data. Using test statistics to fit the scaler would be data leakage.

---

## 4. Feature Engineering and Selection

### 4.1 Why Feature Selection Matters for SLMs

Compact SLMs like TinyBERT and MiniLM have a maximum context window of 256 tokens. If you verbalize all 83 CICFlowMeter features into a text string, the resulting text is approximately 500–700 tokens — exceeding the context window. The model truncates the input, silently losing features. Beyond this hard constraint, there is also a signal-to-noise problem: many CICFlowMeter features are highly correlated with each other and add no discriminative value for slow-rate DoS specifically.

Feature selection reduces 83 (or more) raw features to 12–15 highly discriminative, non-redundant features that fit within 256 tokens when verbalized.

### 4.2 Mutual Information Feature Selection

Mutual Information (MI) measures how much knowing the value of a feature reduces uncertainty about the class label. Unlike Pearson correlation, MI captures non-linear dependencies — crucial for slow-rate DoS features like `conn_state` (categorical) and `header_completeness` (which has a non-linear relationship with attack probability).

For each feature f and class label y:
```
MI(f; y) = Σ Σ P(f=a, y=b) × log[ P(f=a, y=b) / (P(f=a) × P(y=b)) ]
```

Features with high MI share a lot of information with the attack label. Paper SS-5 (DDoSBERT) confirms MI outperforms Pearson correlation and univariate statistical tests for network traffic feature selection.

### 4.3 The 12–15 Selected Features and Why Each Matters

After MI ranking and domain expert review, the following features are retained. Each one is explained in terms of why it is discriminative for slow-rate DoS:

**1. `flow_duration`**
All three slow-rate attack subtypes maintain connections for abnormally long periods. A legitimate HTTP request completes in 0.1–5 seconds. A Slowloris connection lasts 60–600+ seconds. This is the single strongest discriminating feature.

**2. `pkt_rate` (packets per second)**
Slow-rate attacks transmit at below 1 packet per second — often 1 packet every 10–30 seconds (just enough to keep the connection alive). Benign traffic operates at 10–1000+ packets per second. This feature distinguishes slow-rate attacks from volumetric DoS (which has extremely high packet rates) and from benign traffic.

**3. `byte_rate` (bytes per second)**
Derived from `orig_bytes / flow_duration`. Slow-rate attacks transfer minimal data — often 20–100 bytes total over 60+ seconds, giving byte rates near zero. Benign HTTP requests transfer kilobytes in under 5 seconds.

**4. `iat_mean` (inter-arrival time, mean)**
The mean time between consecutive packets in the flow. For Slowloris, the inter-arrival time is deliberately long (8–15 seconds between keepalive packets). For benign traffic, IAT is typically milliseconds. High `iat_mean` is a direct signature of slow-rate attacks.

**5. `iat_std` (inter-arrival time, standard deviation)**
Slowloris keepalive timing has a characteristic irregularity — keepalives are sent with slight timing jitter to evade detection. This produces a high standard deviation in inter-arrival times. Random jitter attacks specifically vary `iat_mean` to evade static threshold detectors, but the temporal aggregation approach captures the variance pattern.

**6. `tcp_window_size_min`**
This is the **primary discriminating feature for Slowread**, which is distinct from Slowloris and RUDY. In a Slowread attack, the attacker deliberately sets its TCP receive window advertisement to a very small value (sometimes 0 or 1 byte). This tells the server "I can only receive X bytes before I need to acknowledge" — forcing the server to send data in tiny chunks and hold the connection open. Legitimate clients have TCP window sizes of 65,535 bytes (or larger with window scaling). A minimum TCP window size near 0 is the Slowread signature.

**7. `fin_flag_count`**
In normal TCP, connections are closed cleanly with FIN packets. Slow-rate attacks maintain connections without closing them — the attacker never sends a FIN. Low `fin_flag_count` (relative to the number of established connections in the window) indicates connections that are being held open deliberately.

**8. `syn_flag_count`**
The ratio of SYN packets (connection initiation) to completed handshakes. A large number of SYNs with few corresponding completions (SYN-ACK received, data exchanged, FIN sent) indicates multiple connection attempts where the attacker keeps sockets open without completing requests.

**9. `conn_state`**
Zeek's connection state field encodes the TCP state machine history. The critical values for slow-rate DoS:
- `S1`: SYN and SYN-ACK seen, but no data sent (connection established, nothing transmitted — Slowloris signature early in an attack)
- `S2`: SYN, SYN-ACK, and some data seen, but no FIN (connection open with partial data — RUDY signature)
- `SF`: normal close (SYN, SYN-ACK, data, FIN seen — benign)
- `REJ`: connection rejected — indicates server has run out of connection slots (late-stage DoS)
Encoded as a one-hot categorical feature.

**10. `orig_bytes`**
Total bytes sent by the originator (attacker). Slow-rate attack clients send very little data — just enough to maintain the connection. Benign clients send full HTTP requests (hundreds to thousands of bytes). Very low `orig_bytes` combined with long duration is the compound signature of slow-rate attacks.

**11. `header_completeness`**
A derived feature: for HTTP flows, the ratio of headers received to headers expected. Slowloris specifically exploits the HTTP standard's requirement for a complete header before the server can respond. The attacker sends headers one line at a time, occasionally appending a new header line every 10–15 seconds, ensuring the request is never complete. `header_completeness = 0.1` means only 10% of the expected headers have been received. This is the **primary discriminating feature for Slowloris** versus RUDY (which sends a complete header but an incomplete body).

**12. `content_length_ratio`**
For HTTP POST flows: the ratio of actual body bytes received to the `Content-Length` value declared in the HTTP header. RUDY (R-U-Dead-Yet / Slowbody) sends a POST request with a large `Content-Length` (e.g., 1,000,000 bytes) but transmits the body at approximately 1 byte per second. `content_length_ratio = 0.001` after 60 seconds means the attacker declared a 1MB body but has only sent 1KB. This is the **primary discriminating feature for RUDY**.

**13. `protocol`**
TCP (6) vs. UDP (17) vs. ICMP (1). Slow-rate HTTP DoS is always TCP. While this feature alone is not discriminative (most traffic is TCP), it is important as a filter and as part of the verbalized feature string for the decoder's reasoning.

**14. `dst_port`**
Destination port. Slow-rate DoS targets web servers — ports 80 (HTTP) and 443 (HTTPS). An IoT device receiving a slow-rate attack on port 80 is behaving as a web server being targeted. This is important for the decoder's contextual reasoning about the attack's purpose.

**15. `entropy_src_ports`** (optional)
Shannon entropy of source ports in the time window. A single attacker using many source ports (port-hopping) to evade per-connection thresholds produces high entropy. Normal clients use a small number of source ports. This feature helps detect distributed slow-rate attacks.

### 4.4 Domain Expert Override

MI ranking is automated, but domain expert review is essential. Some slow-rate-specific features (e.g., `header_completeness`, `content_length_ratio`) may rank lower on MI if the training dataset has a specific class distribution where these features are not as statistically frequent. However, these features are definitionally important for distinguishing attack subtypes. The final feature set is therefore a combination of top-MI features and expert-retained features.

---

## 5. Input Representation and Tokenization

The 12–15 selected features must be converted into a format that the encoder SLM can process. This is called **feature verbalization** — transforming structured tabular data into a text string.

### 5.1 Why Verbalization Is Necessary

BERT-class encoder models are trained on text. They process sequences of tokens (subword units) and produce contextualized embeddings. They cannot directly receive a NumPy array of floating-point numbers. Verbalization bridges this gap: each feature is represented as a text token that the model can process through its embedding layer and self-attention mechanism.

### 5.2 Format A vs. Format B

**Format A — Space-separated (from Zeek conn.log directly):**
```
tcp - 0.398803 0 0 REJ T T 0 Sr 1 60 1 40
```
This is compact and fast to generate. The encoder learns to associate token positions with features. However, it is semantically opaque — there is no indication of what each value represents. If the feature vector length changes (adding or removing a feature), all positional associations break.

**Format B — Key-value pairs (recommended):**
```
FLOW_DURATION: 127.5s PKT_RATE: 0.03pkt/s BYTE_RATE: 1.9B/s IAT_MEAN: 42.5s 
IAT_STD: 11.2s TCP_WINDOW_MIN: 0 FIN_FLAGS: 0 SYN_FLAGS: 1 CONN_STATE: S1 
ORIG_BYTES: 245B HEADER_COMPLETE: 0.12 CONTENT_LEN_RATIO: 0.003 
PROTOCOL: TCP DST_PORT: 80 INCOMPLETE_HEADER: True SLOW_RATE: True
```

Format B is strongly preferred because:
- Feature names are preserved in the token sequence, so the encoder learns feature-name → value associations directly from the text
- The downstream decoder SLM can read the same Format B string and understand which value corresponds to which feature — directly reducing hallucination
- Adding or removing features does not invalidate positional associations
- The decoder's SHAP explanation ("pkt_rate of 0.03 pkt/s indicates...") can reference the exact name that appears in the SIR, ensuring Feature Consistency

### 5.3 Tokenization

Tokenization converts the text string into a sequence of integer IDs (tokens) that the model's embedding layer can process.

**For SecurityBERT:** A custom ByteLevelBPE tokenizer with vocabulary size 5,000 is used. This tokenizer is trained specifically on network data (flow feature strings), so it knows how to split tokens like "0.03pkt/s" and "CONN_STATE:S1" efficiently. The vocabulary is much smaller than standard BERT's 30,522 — reflecting that network feature text has a much smaller vocabulary than natural language. The PPFLE (Privacy-Preserving Fixed-Length Encoding) variant hashes each feature value: `H(column_name + "$" + value)` → a fixed-length hash token. This produces a completely fixed-length token sequence regardless of value magnitude, and prevents raw feature values from appearing in model inputs.

**For TinyBERT and MiniLM:** The standard WordPiece tokenizer with 30,522 vocabulary is used. This produces slightly longer token sequences than ByteLevelBPE for network data (because the general vocabulary has fewer network-specific tokens), but is simpler to use with pre-trained checkpoints.

The tokenization procedure:
```python
tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
encoding = tokenizer(
    verbalized_text,
    padding="max_length",
    truncation=True,
    max_length=256,
    return_tensors="pt"
)
# Returns:
# input_ids: tensor of token IDs (shape: [1, 256])
# attention_mask: 1 for real tokens, 0 for padding (shape: [1, 256])
# token_type_ids: all zeros for single-sequence input (shape: [1, 256])
```

The `max_length=256` parameter is critical. With 12–15 features in Format B, the typical verbalized string produces 80–160 tokens — comfortably within the 256-token limit. This budget must be verified explicitly during data preparation.

### 5.4 Window-Level Encoding for Temporal Patterns

A single flow's verbalized features do not capture the temporal evolution of a slow-rate attack. The encoder is therefore given a **window of N=3–5 consecutive flow records** per inference call. Each flow is verbalized independently and the flows are concatenated with `[SEP]` tokens:

```
[CLS] FLOW_DURATION: 127.5s PKT_RATE: 0.03 ... [SEP] FLOW_DURATION: 141.2s PKT_RATE: 0.02 ... [SEP] FLOW_DURATION: 115.8s PKT_RATE: 0.04 ... [SEP]
```

The `[CLS]` token at the beginning captures the aggregated representation of the entire window. In BERT's architecture, the `[CLS]` token's final hidden state is trained to represent the overall meaning of the input — making it the natural input to the classification head.

This window-level encoding directly captures the temporal accumulation of slow-rate connections: three flows in a row, each with long duration, low packet rate, and incomplete headers, is a much stronger signal than any single flow alone.

### 5.5 Boolean Domain Flags

Before the feature values in the decoder prompt (not the encoder input), five boolean flags are prepended. These flags are derived from domain knowledge and explicitly encode the semantic interpretation of the feature values:

- `INCOMPLETE_HEADER: True/False` — HTTP header was never completed (Slowloris)
- `SLOW_RATE: True/False` — packet rate < 1 pkt/s sustained over 30s
- `WINDOW_EXHAUSTION: True/False` — TCP window size < 64 bytes (Slowread)
- `LONG_CONNECTION: True/False` — flow duration > 60s with minimal data
- `PARTIAL_BODY: True/False` — actual body bytes << declared Content-Length (RUDY)

These flags are for the **decoder only** — they help the language model reason about the attack without having to infer the semantic interpretation from raw numbers. A decoder SLM that sees `INCOMPLETE_HEADER: True` knows immediately to reason about Slowloris; it does not need to compute what `header_completeness = 0.12` means.

---

## 6. Encoder SLM — Structure and How Detection Happens

### 6.1 What a BERT-Class Encoder Is

BERT (Bidirectional Encoder Representations from Transformers) is a neural network architecture based on the Transformer encoder. "Encoder" in this context means the model reads the entire input sequence simultaneously and produces a contextualized representation of each token — as opposed to a decoder (like GPT), which generates tokens one at a time left-to-right.

The BERT architecture consists of:
1. **Token Embedding Layer:** Converts each token ID into a dense vector of dimension d_model (typically 128–768 depending on model size). Adds positional embeddings (which encode the position of each token in the sequence) and segment embeddings (which distinguish the first and second sequences in a pair).
2. **Stacked Transformer Encoder Layers:** Each layer contains a Multi-Head Self-Attention sublayer and a Feed-Forward Network (FFN) sublayer, connected by residual connections and layer normalisation.
3. **Classification Head:** A linear layer on top of the `[CLS]` token's final hidden state, mapping from d_model dimensions to the number of classes (attack subtypes + benign).

### 6.2 Multi-Head Self-Attention — How the Model Reads Features

Self-attention is the mechanism that allows the model to relate each feature token to every other feature token in the input. This is what makes it possible for the model to learn that `CONN_STATE: S1` combined with `FLOW_DURATION: 127.5s` and `PKT_RATE: 0.03` collectively indicate Slowloris, even though none of these features individually is definitive.

In self-attention, each token produces three vectors: a Query (Q), a Key (K), and a Value (V), each of dimension d_k = d_model / num_heads. The attention score between token i and token j is:

```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
```

The `softmax(Q × K^T / √d_k)` term produces a probability distribution over all tokens in the sequence for each query token. High attention weight between two tokens means "these two features are highly relevant to each other for understanding this input." The model learns during fine-tuning that the `FLOW_DURATION` token should attend strongly to the `PKT_RATE` and `CONN_STATE` tokens when processing slow-rate DoS flows.

"Multi-head" means this computation is done h times in parallel (h=4 for TinyBERT, h=12 for full BERT-base), each with different learned Q, K, V projection matrices. Different heads can learn to capture different feature relationships simultaneously — one head may specialize in duration-rate relationships, another in flag-state relationships.

### 6.3 The Three Encoder SLM Candidates

**SecurityBERT (11M parameters, 16.7MB FP32):**
Custom architecture with 15 transformer layers and the ByteLevelBPE tokenizer trained on network data. The PPFLE encoding produces fixed-length hashed token sequences per feature, eliminating variable-length verbalization issues. Achieves 98.2% accuracy on Edge-IIoTset. This is the highest-accuracy option but requires access to the SecurityBERT checkpoint or replication of its pre-training procedure.

| Component | SecurityBERT | TinyBERT | MiniLM-L6 |
|---|---|---|---|
| Parameters | 11M | 14M | 22M |
| Layers | 15 | 4 | 6 |
| Hidden dim (d_model) | 128 | 312 | 384 |
| Attention heads | 4 | 12 | 12 |
| FP32 size | 16.7MB | 56MB | 86MB |
| INT8 ONNX size | ~4.2MB | ~14MB | ~22MB |
| Embedded ARM latency | ~40ms | ~157ms | ~336ms |
| Reported F1 | 98.2% | 84.7% | 88.3% |

**TinyBERT (14M parameters, 56MB FP32):**
A 4-layer, 312-hidden-dimension BERT distilled from BERT-base. TinyBERT's small number of layers (4 vs. BERT-base's 12) dramatically reduces both parameter count and inference time. Achieves F1=0.847 on ARM Cortex-A53 embedded hardware at 157ms per inference window — the fastest BERT-class option validated on embedded hardware. This is the recommended choice for the most constrained hardware (≤4GB RAM, ARM-class CPU).

**MiniLM-L6 (22M parameters, 86MB FP32):**
A 6-layer, 384-hidden-dimension BERT distilled using attention transfer. Achieves F1=0.883 at 336ms on the same embedded hardware. Better accuracy than TinyBERT at roughly 2× the latency cost. Recommended for slightly less constrained deployments (8GB RAM) where higher detection accuracy is prioritized.

### 6.4 The Classification Head

On top of the encoder's final layer, a classification head is attached:

```
[CLS] token final hidden state (shape: [d_model])
    ↓
Dropout (p=0.1) — regularisation
    ↓
Linear layer: d_model → num_classes
    (e.g., 312 → 4 for TinyBERT with 4 classes: Slowloris, RUDY, Slowread, Benign)
    ↓
Logits (shape: [num_classes])
    ↓
Softmax → class probabilities
    ↓
argmax → predicted class label + confidence score
```

The classification head is randomly initialized and trained from scratch during fine-tuning. The encoder layers are initialized from the pre-trained checkpoint and fine-tuned (all layers, not just the head — full fine-tuning consistently outperforms head-only by 3–5%).

### 6.5 How Detection Happens — The Decision Process

When a new flow window arrives:

1. The verbalized text string is tokenized → `input_ids`, `attention_mask` tensors
2. The encoder processes the tokens through all transformer layers
3. The `[CLS]` token's final hidden state is extracted (shape: [d_model])
4. The classification head maps this to logits
5. Softmax converts logits to class probabilities
6. The class with the highest probability is the predicted label
7. The raw confidence score (the maximum softmax probability) is recorded

If confidence ≥ threshold τ* (calibrated on validation set, typically 0.8–0.95 for attack classes), the detection is treated as a confirmed alert. If confidence < τ*, the sample is flagged as uncertain and may trigger additional analysis without immediate blocking.

The `[CLS]` embedding (the d_model-dimensional vector before the classification head) is also saved. This is the encoder's **contextual representation of the flow window** — a rich, high-dimensional encoding that captures the patterns the model found in the feature sequence. This embedding is passed to the decoder pipeline as part of the SIR, where it can optionally be used to retrieve similar exemplars from the knowledge base.

### 6.6 The Role of INT8 ONNX in Deployment

In development, the encoder runs in PyTorch FP32 — full 32-bit floating point precision. In deployment on edge hardware, it runs as an INT8 ONNX model.

ONNX (Open Neural Network Exchange) is a standardized model format that is hardware-agnostic. ONNX Runtime is a C++ execution engine optimized for CPU inference. The conversion pipeline is:

```
Fine-tuned PyTorch model (FP32)
    ↓ torch.onnx.export() or optimum-cli
ONNX model (FP32)
    ↓ onnxruntime.quantization.quantize_dynamic()
ONNX model (INT8)
```

INT8 quantization replaces 32-bit floating point weights and activations with 8-bit integers. This reduces model size by ~4× and speeds up inference by 2–4× on CPUs (which have hardware INT8 multiplication instructions that are faster than FP32). The accuracy loss is less than 1% for BERT-class models, making it an excellent trade-off.

---

## 7. The XAI Bridge — SHAP and the Attribution Layer

### 7.1 Why XAI Is Needed Between Detection and Explanation

The encoder SLM produces a detection label and a confidence score. These two values tell us **what** the model decided, but not **why**. The decoder SLM needs to generate a human-readable explanation of why the traffic was classified as an attack — citing specific features that contributed to the decision. Without knowing which features drove the classification, the decoder would have to guess, leading to hallucinated explanations that sound plausible but are not grounded in the actual detection evidence.

SHAP (SHapley Additive exPlanations) solves this by computing, for each feature, how much it contributed to the model's prediction — in a mathematically rigorous, game-theory-grounded way.

### 7.2 Shapley Values — The Mathematical Foundation

A Shapley value quantifies the average marginal contribution of a feature to the prediction, across all possible orderings of features. Intuitively, it answers: "If I add this feature to the set of features the model is using, how much does the prediction change on average?"

For a feature i, its Shapley value φ_i is:
```
φ_i = Σ (|S|! × (|F| - |S| - 1)! / |F|!) × [v(S ∪ {i}) - v(S)]
      S ⊆ F\{i}
```

Where F is the full set of features, S is a subset not containing i, and v(S) is the model's prediction using only the features in S. The Shapley value is the weighted average of the marginal contribution of feature i across all possible subsets.

For an XGBoost model, `shap.TreeExplainer` computes exact Shapley values in polynomial time using the tree structure — this is extremely fast (<100ms) compared to the exponential naive computation.

### 7.3 What SHAP Produces for a Single Detection

For a specific slow-rate DoS detection, SHAP produces a vector of Shapley values, one per feature:

```
Feature: flow_duration    | SHAP value: +0.42  | Direction: high  | Contribution: pushes toward "Slowloris"
Feature: pkt_rate         | SHAP value: +0.38  | Direction: low   | Contribution: pushes toward "Slowloris"
Feature: header_complete  | SHAP value: +0.31  | Direction: low   | Contribution: pushes toward "Slowloris"
Feature: tcp_window_min   | SHAP value: +0.11  | Direction: low   | Contribution: pushes toward "Slowloris"
Feature: orig_bytes       | SHAP value: +0.09  | Direction: low   | Contribution: pushes toward "Slowloris"
Feature: conn_state       | SHAP value: +0.08  | Direction: S1    | Contribution: pushes toward "Slowloris"
Feature: dst_port         | SHAP value: +0.01  | Direction: 80    | Contribution: slight push toward "Slowloris"
Feature: protocol         | SHAP value: +0.00  | Direction: tcp   | Contribution: negligible
```

The "direction" indicates whether the feature value is high or low relative to the average across all training samples. A positive SHAP value means the feature pushed the prediction toward the detected class; a negative value means it pushed against.

### 7.4 The Critical Constraint: SHAP Must Not Be Applied to LLM Token Inputs

Paper SS-3 (Lodh et al.) demonstrates conclusively that applying SHAP directly to the tokenised text inputs of an LLM produces semantically meaningless results. When a text string like `"FLOW_DURATION: 127.5s"` is tokenized, it becomes something like `[23, 4901, 82, 127, 15, 1234]` — six separate tokens. SHAP attributes importance to individual tokens, producing fragments like `"0"`, `"+ +"`, `"id"`, `"le"`, `"min"` that correspond to sub-word pieces of the original feature values. These attributions are uninterpretable.

The correct approach is to apply SHAP to the **tabular feature vector** feeding the XGBoost/RF model (or to the structured feature input of the encoder before verbalization). SHAP produces feature-level (not token-level) attributions, which are then incorporated into the SIR JSON and passed to the decoder as readable text.

### 7.5 Connecting SHAP Output to the Decoder

The SHAP values are sorted by absolute magnitude, and the top-4 to 6 features are selected as the most influential for this particular detection. Each entry in the SIR includes:
- Feature name (string, matching Format B key)
- Feature value (the actual measured value, with units)
- SHAP importance score (normalized to [0, 1] for readability)
- Direction (whether the value is "high" or "low" relative to benign average)

This information grounds every claim the decoder makes in specific, evidence-backed feature values — directly matching the decoder's explanation to the detection model's actual reasoning.

---

## 8. The Structured Intermediate Representation (SIR)

### 8.1 The Purpose of the SIR

The SIR is the carefully designed information bridge between the detection pipeline and the explanation pipeline. It is not just "pass everything to the LLM" — it is a structured, compact, semantically rich JSON object that contains exactly what the decoder needs to generate an accurate, grounded explanation.

Paper F4-4 (LLM for Explain paper) demonstrates that how ML detection outputs are presented to the LLM matters as much as which LLM is used. A well-structured presentation leads to accurate, feature-grounded explanations. An unstructured dump of numbers leads to hallucination. The SIR is the operationalization of this finding.

### 8.2 Full SIR Structure

```json
{
  "detection_label": "Slowloris",
  "confidence": 0.97,
  "ml_label": "Slowloris",
  "ml_confidence": 0.94,
  "encoder_label": "Slowloris",
  "encoder_confidence": 0.97,
  "top_features": [
    {
      "name": "flow_duration",
      "value": "127.5s",
      "importance": 0.42,
      "direction": "high",
      "benign_baseline": "2.3s",
      "interpretation": "connection held open 55x longer than benign average"
    },
    {
      "name": "pkt_rate",
      "value": "0.03 pkt/s",
      "importance": 0.38,
      "direction": "low",
      "benign_baseline": "47.2 pkt/s",
      "interpretation": "1572x fewer packets per second than benign average"
    },
    {
      "name": "header_completeness",
      "value": "0.12",
      "importance": 0.31,
      "direction": "low",
      "benign_baseline": "1.00",
      "interpretation": "only 12% of expected HTTP headers transmitted"
    },
    {
      "name": "tcp_window_min",
      "value": "0 bytes",
      "importance": 0.11,
      "direction": "low",
      "benign_baseline": "65535 bytes",
      "interpretation": "TCP receive window nearly zero — connection held deliberately stalled"
    }
  ],
  "flow_window_summary": {
    "window_duration_s": 15,
    "total_flows_in_window": 3,
    "flows_from_src_ip": 3,
    "total_bytes": 245,
    "total_packets": 9,
    "conn_states_seen": ["S1", "S1", "S2"]
  },
  "source_context": {
    "dst_port": 80,
    "protocol": "TCP",
    "conn_state": "S1"
  },
  "domain_flags": {
    "incomplete_header": true,
    "slow_rate": true,
    "window_exhaustion": false,
    "long_connection": true,
    "partial_body": false
  },
  "timestamp": "2025-03-15T14:23:07Z"
}
```

### 8.3 How the SIR Is Constructed

The SIR is assembled in Python by a small "SIR Builder" module that runs synchronously after the encoder and SHAP have completed:

```
encoder output → label, confidence, [CLS] embedding
SHAP output → top-K feature importance scores + directions
flow window data → raw feature values, timestamp, context
domain flag logic → five boolean conditionals applied to feature values

SIR Builder combines all four and serializes to JSON
Total construction time: <5ms
```

The `interpretation` field for each feature is generated using a lookup table of human-readable descriptions for each feature × direction combination. This is not LLM-generated at construction time — it is templated text that translates the raw comparison (feature value vs. benign baseline) into a natural language phrase. This ensures the interpretation is always factually accurate, regardless of what the decoder later generates.

---

## 9. Decoder SLM — Structure and How Reasoning Is Generated

### 9.1 What a Decoder SLM Is

A decoder SLM (like Gemma3-4B, Phi-2, or LLaMA 3.2-3B) is an autoregressive language model based on the Transformer decoder architecture. Unlike BERT (which reads the whole sequence simultaneously), a decoder generates text **one token at a time, left to right**, conditioning each new token on all previously generated tokens plus the input prompt.

The decoder architecture consists of:
1. **Token Embedding Layer:** Maps input token IDs to dense vectors
2. **Stacked Transformer Decoder Layers:** Each layer contains a Causal (Masked) Self-Attention sublayer (each token can only attend to preceding tokens — not future tokens) and a Feed-Forward Network sublayer
3. **Language Modelling Head:** A linear layer mapping from d_model to vocabulary size (typically 32,000–130,000 tokens), followed by softmax to produce a probability distribution over the next token

### 9.2 The Three Decoder Candidates

**Gemma3-4B (Google, 4 billion parameters):**
The recommended choice. 4B parameters give sufficient reasoning depth for distinguishing Slowloris from RUDY from Slowread and generating detailed mitigation recommendations. In Q4_K_M GGUF quantization, occupies ~2.5GB RAM. Achieves macro-F1=0.85 with few-shot+CoT on IoT DDoS classification tasks. Best explanation quality among sub-5B models.

**Phi-2 (Microsoft, 2.7B parameters):**
Most energy-efficient option (0.173 kg CO2 per training run vs. 0.304 kg for LLaMA2-7B, per SS-3). Phi-2 uses a "textbook quality" pre-training dataset that gives it disproportionately strong reasoning capability relative to its size. In Q4_K_M: ~1.7GB RAM. Best choice for very constrained deployments (2–4GB RAM) where energy efficiency is critical.

**LLaMA 3.2-3B (Meta, 3 billion parameters):**
Achieves macro-F1=0.75 with few-shot+CoT on 6-class IoT DDoS. The minimum parameter count for reliably distinguishing all three slow-rate subtypes (Slowloris, RUDY, Slowread) from each other. In Q4_K_M: ~2.0GB RAM.

### 9.3 The SSRP Framework — How the Decoder Is Prompted

The 16-factor Structured Security Reasoning Prompt (SSRP) framework (from F3-9, Zhou et al.) provides a structured approach to constructing the decoder prompt that consistently improves explanation quality by ~40% in 2–4B models. The 16 factors are organized across the system prompt and user prompt:

**System prompt factors (persistent, loaded once):**
- F1: Role definition — "You are a network security analyst specializing in slow-rate HTTP DoS attack detection on IoT networks."
- F2: Output format specification — "Always structure responses as: Observation, Evidence, Conclusion, Mitigation."
- F3: Domain knowledge — Slowloris mechanism description (incomplete HTTP header exploitation)
- F4: Domain knowledge — RUDY mechanism description (incomplete POST body, large Content-Length)
- F5: Domain knowledge — Slowread mechanism description (TCP window advertisement manipulation)
- F6: Normal behavior norms — "A legitimate HTTP connection completes within 0.1–5 seconds..."
- F7: Protocol definitions — TCP state machine states (SYN, SYN-ACK, FIN), HTTP header structure
- F8: Feature definitions — Each of the 12–15 features with plain-language description and normal ranges
- F9: Constraint — "Only cite features present in the provided detection data. Do not invent feature values."
- F10: Constraint — "Do not speculate about attacker identity or intent beyond what the traffic patterns indicate."

**User prompt factors (per-inference, regenerated each call):**
- F11: SIR JSON block — the actual detection data for this alert
- F12: Retrieved exemplar — one complete worked example of a similar attack
- F13: CoT trace — the exemplar's reasoning chain showing Obs→Evidence→Conclusion→Mitigation
- F14: Task instruction — "Based on the detection result above, generate a security explanation following the format in the exemplar."
- F15: Output schema — JSON schema for GBNF grammar constraint
- F16: Few-shot cue — "Now analyze the detection data and provide your explanation:"

### 9.4 The Critical Role of Few-Shot Exemplars

Paper F3-6 (Rethinking On-Device) provides the most important practical finding about decoder prompting: **Chain-of-Thought prompting alone (without exemplars) is catastrophically bad for ≤4B models on numeric data.** Gemma3-4B with CoT-only achieves F1=0.05 (worse than random), while Gemma3-4B with few-shot achieves F1=0.85.

The reason is that abstract reasoning instructions ("think step by step") require the model to apply them to unfamiliar numeric inputs in a novel domain. Small models cannot generalize these abstract instructions to network traffic data. What they can do is follow a concrete pattern shown in an exemplar. An exemplar provides an exact template: "When you see these kinds of feature values with these kinds of SHAP attributions, produce this kind of explanation."

A Slowloris exemplar looks like:
```
EXEMPLAR (Slowloris attack):
Detection: Slowloris, confidence 0.96
Key features: flow_duration=142s, pkt_rate=0.02, header_completeness=0.08

Observation: A TCP connection from client to server port 80 was maintained for 142 seconds 
with only 6 packets totalling 312 bytes transmitted.

Evidence:
- flow_duration of 142 seconds exceeds the benign average of 2.3 seconds by 61×, 
  indicating the connection was deliberately kept open far longer than necessary 
  for any legitimate HTTP request.
- pkt_rate of 0.02 pkt/s (one packet every 50 seconds) matches the Slowloris keepalive 
  pattern, where an incomplete HTTP header line is sent periodically to prevent 
  server-side timeout.
- header_completeness of 0.08 confirms the HTTP request header was never completed — 
  only 8% of the expected headers were transmitted, consistent with the Slowloris 
  technique of deliberately withholding the final CRLF that would complete the header.

Conclusion: This traffic is a Slowloris HTTP connection-exhaustion DoS attack. The 
attacker is maintaining a TCP connection to the server while sending incomplete HTTP 
headers, preventing the server from freeing the connection resource. With sufficient 
concurrent connections, this exhausts the server's connection pool.

Mitigation:
1. Enforce a server-side HTTP header timeout: close any connection that has not 
   submitted a complete HTTP header within 10 seconds.
2. Limit maximum concurrent connections per source IP to 10.
3. Apply rate limiting: drop connections from source IPs with pkt_rate < 0.1 pkt/s 
   sustained over 30 seconds.
```

This exemplar shows the decoder exactly what format, depth, and style of explanation is expected. The decoder's task becomes pattern completion rather than open-ended generation.

### 9.5 Exemplar Retrieval with XGBoost

At inference time, the system selects the most relevant exemplar from the library of 3–5 exemplars per attack type. Retrieval is done using an XGBoost model trained as a similarity scorer on the SIR feature vectors — not BERT embeddings. This is a validated finding from F3-6: XGBoost retrieval outperforms BERT embedding cosine similarity for numeric network feature matching.

The reason: BERT embeddings are designed for semantic similarity in natural language. Two flow records with numerically similar feature values (e.g., both with pkt_rate ≈ 0.03) may produce very different BERT embeddings if their verbalized text strings have different token orderings. XGBoost trained on the tabular feature vectors directly measures similarity in the feature space — which is exactly what's needed for finding a "this flow looks like that exemplar" match.

### 9.6 GBNF Grammar-Constrained Decoding

Autoregressive decoders can generate any text token at each step. Without constraints, the decoder may generate malformed JSON (missing brackets, incorrect field names) or produce output that doesn't follow the Obs→Evidence→Conclusion→Mitigation schema. This would require error handling and re-generation, adding latency.

GBNF (Grammar Based Normal Form) allows llama.cpp to constrain the decoder's output to a specific grammar — ensuring every generated token is valid according to the specified schema:

```gbnf
root   ::= "{" ws "\"observation\":" ws string "," ws
              "\"evidence\":" ws "[" ws string ("," ws string)* ws "]" "," ws
              "\"conclusion\":" ws string "," ws
              "\"mitigation\":" ws "[" ws string ("," ws string)* ws "]"
           ws "}"
string ::= "\"" [^"]* "\""
ws     ::= [ \t\n]*
```

At each generation step, llama.cpp evaluates which tokens are grammatically valid at the current position and masks the logits of all invalid tokens to -infinity before softmax. This guarantees valid JSON output without any post-processing, at no accuracy cost.

### 9.7 The Output — What the Decoder Generates

The decoder produces a JSON object with four fields:

- **observation:** One to two sentences describing what was observed in the traffic without interpretation. Pure measurement: "A TCP connection was maintained for 127.5 seconds with 3 packets and 245 bytes transmitted."
- **evidence:** A list of three to five specific feature-value statements with their significance relative to baseline. Each statement cites the exact feature name and value from the SIR.
- **conclusion:** One to two sentences identifying the specific attack subtype and mechanism, grounded in the evidence.
- **mitigation:** A numbered list of three to five specific, actionable steps for the network administrator or IoT device operator.

---

## 10. How the Encoder and Decoder Communicate

### 10.1 The Fundamental Timing Problem

The encoder runs in ~80ms. The decoder runs in 1–3 seconds. If the system waited for the decoder to finish before proceeding with detection, the effective detection rate would be limited to roughly 0.3–1 alerts per second — completely inadequate for a network with potentially thousands of flows per second. The communication architecture must therefore **completely decouple** the detection timeline from the explanation timeline.

### 10.2 The Producer-Consumer Queue

The synchronous detection pipeline (ML + encoder + SHAP + SIR builder) is the **producer**. The asynchronous decoder pipeline is the **consumer**. They communicate via a thread-safe queue:

```python
import queue
import threading

sir_queue = queue.Queue(maxsize=100)  # bounded to prevent memory overflow

# PRODUCER: runs in main thread, synchronous
def detection_pipeline(flow_window):
    ml_label, ml_conf = xgboost_model.predict(tabular_features)
    enc_label, enc_conf, cls_embedding = encoder_model.predict(verbalized_text)
    shap_values = shap_explainer.shap_values(tabular_features)
    sir = build_sir(ml_label, enc_label, enc_conf, shap_values, flow_window)
    
    if enc_conf >= threshold and enc_label != "Benign":
        sir_queue.put_nowait(sir)  # non-blocking; raises Full if queue is full
    
    return enc_label, enc_conf  # detection result returned immediately

# CONSUMER: runs in background thread, asynchronous
def explanation_worker():
    while True:
        sir = sir_queue.get(block=True)  # waits for new SIR
        exemplar = retriever.get_closest(sir["top_features"])
        prompt = jinja2_template.render(sir=sir, exemplar=exemplar)
        explanation = ollama_client.generate(model="slm-ids", prompt=prompt)
        store_explanation(sir["timestamp"], explanation)
        sir_queue.task_done()

# Start background worker
worker_thread = threading.Thread(target=explanation_worker, daemon=True)
worker_thread.start()
```

This pattern means:
- The detection decision is returned to the caller (network monitoring system) in ~80–200ms, regardless of how long the decoder takes
- The decoder processes SIR objects from the queue as fast as it can (1–3s each)
- If SIRs arrive faster than the decoder can process them (e.g., during a burst attack), the queue buffers up to 100 objects before dropping or alerting
- The explanation appears in the analyst dashboard 1–3 seconds after the alert

### 10.3 The [CLS] Embedding as Optional Semantic Bridge

Beyond the SIR JSON, the encoder's `[CLS]` embedding can optionally be passed to the decoder system. Since the decoder receives text (not tensors), the embedding is not directly usable. However, it can be used in two ways:
1. As the feature vector for XGBoost exemplar retrieval — instead of the raw tabular features, use the encoder's learned representation as the retrieval query
2. Stored in ChromaDB alongside the SIR for future retrieval across sessions

### 10.4 Memory Isolation Between Encoder and Decoder

On devices with very limited RAM (4–6GB), loading both models simultaneously may be infeasible — the encoder ONNX takes ~14MB, but the decoder GGUF takes ~2.5GB. The queue-based design enables a resource management strategy:
1. The encoder is loaded once at startup and kept resident in memory (small enough at 14MB INT8)
2. The decoder is loaded on demand by Ollama when the first SIR enters the queue, and can be configured to unload after a period of inactivity (`keep_alive: "5m"` in Ollama's Modelfile)
3. This means both models are never simultaneously loaded during the detection-only periods, conserving RAM for the operating system and other processes

---

## 11. Training Pipeline

Training happens in two completely independent phases — the encoder is trained first, then the decoder. They do not train jointly.

### 11.1 Encoder Training Pipeline

**Data flow:**
```
CIC IIoT Dataset 2025 (labeled PCAP/CSV)
    ↓ Zeek flow extraction
    ↓ Preprocessing (cleaning, windowing, normalization, SMOTE on train only)
    ↓ Feature selection (MI → 12–15 features)
    ↓ Feature verbalization (Format B key-value strings)
    ↓ Stratified split: 80% train / 10% validation / 10% test

Train split:
    ↓ HuggingFace Dataset tokenization
    ↓ TinyBERT/MiniLM pre-trained checkpoint loaded
    ↓ Classification head attached (random init)
    ↓ AdamW optimizer (lr=2e-5, weight_decay=0.01)
    ↓ Linear warmup: first 10% of steps
    ↓ Cross-entropy loss (class weights = inverse class frequency)
    ↓ For each epoch:
        ↓ Shuffle training data
        ↓ Forward pass: input_ids → encoder → [CLS] → head → logits
        ↓ Compute cross-entropy loss
        ↓ Backward pass: compute gradients
        ↓ Gradient clip to norm 1.0 (prevents gradient explosion)
        ↓ AdamW step (update all parameters)
        ↓ Every 100 steps: evaluate on validation set → log macro-F1
    ↓ EarlyStopping: if val macro-F1 doesn't improve for 3 epochs, stop
    ↓ Save best checkpoint (highest val macro-F1)
    
Test evaluation:
    ↓ Load best checkpoint
    ↓ Calibrate classification threshold τ* on validation set
        (find τ* that maximises macro-F1 across all classes)
    ↓ Apply τ* to test set predictions
    ↓ Report: per-class F1, macro-F1, precision, recall, AUC-ROC, FPR, FNR
```

**Preventing catastrophic forgetting:**
30% of the training data is sampled from a prior CIC-format IDS dataset (e.g., CIC-DoS2017 or CICIoT2023). Without this, when fine-tuning on the CIC IIoT 2025 data alone, the model forgets its general intrusion detection knowledge and overfits to the specific temporal patterns in the new dataset. With 30% past data, per-day test F1 remains above 0.96 across all evaluation days.

**The threshold calibration step:**
After training, the encoder's softmax output is a probability score. The default threshold (classify as attack if probability ≥ 0.5) is rarely optimal for imbalanced security datasets. Instead, a sweep is run on the validation set: for τ ∈ {0.3, 0.35, ..., 0.95}, compute macro-F1 at each threshold. The τ* that maximises macro-F1 is saved and used at test time and deployment. This consistently improves macro-F1 by 2–5% over the default 0.5 threshold.

### 11.2 Decoder Training Pipeline (QLoRA)

The decoder is not trained to classify — it is trained to generate explanations. The training data is a set of (instruction, input, response) triplets where `input` is a SIR JSON and `response` is a high-quality explanation.

**Step 1 — Generating the training corpus offline:**
A teacher LLM (GPT-4, Claude, or DeepSeek-R1) is used to generate the response field for each training example. The procedure:
1. Take 500–5,000 representative SIR examples from the labeled dataset
2. For each SIR, compose the full input prompt (SSRP system prompt + SIR JSON)
3. Send to teacher LLM
4. Teacher generates a high-quality Obs→Evidence→Conclusion→Mitigation explanation
5. Human review of 10% sample for quality control
6. Store as JSONL: `{"instruction": "...", "input": "<SIR_JSON>", "output": "<explanation>"}`

This is **offline knowledge distillation**: the expensive teacher runs once during dataset creation, and the cheap student learns from its outputs. The student (Gemma3-4B/Phi-2) never calls the teacher at inference time.

**Step 2 — QLoRA fine-tuning:**
```
Gemma3-4B-Instruct base weights (frozen, 4-bit NF4 quantized via bitsandbytes)
    ↓
LoRA adapters inserted into Q, K, V, O projection matrices of all attention layers
  (rank r=8, alpha=32, dropout=0.05 — only these ~40M adapter parameters are trained)
    ↓
Training dataset loaded and tokenized in instruction format:
  <|system|>You are a network security analyst...<|end|>
  <|user|><SIR_JSON><|end|>
  <|assistant|><explanation_JSON><|end|>
    ↓
SFTTrainer (Supervised Fine-Tuning Trainer from trl):
  - paged AdamW 32-bit optimizer (lr=2e-4, cosine schedule)
  - Micro batch size 4, gradient accumulation steps 8 → effective batch 32
  - Max sequence length 2048 tokens
  - Loss computed only on assistant (response) tokens — not on system/user tokens
  - 3–5 epochs with early stopping on validation loss
    ↓
Save best LoRA adapter checkpoint
```

**Why QLoRA works:**
The base model (Gemma3-4B at 4-bit NF4) has its weights quantized and frozen. Quantization reduces the model's memory footprint from ~16GB (FP32) to ~2.5GB (4-bit). LoRA (Low-Rank Adaptation) inserts two small matrices A and B into each attention projection: the original weight W is replaced by W + B×A where B has shape [d_out, r] and A has shape [r, d_in] with r=8 (the rank). Only A and B are trained (in full FP32 precision), contributing roughly 1–2% of the total parameter count. This makes fine-tuning feasible on a single 24GB GPU — the adapter parameters fit in GPU memory even though the full base model would not.

**Step 3 — Merging and quantizing for deployment:**
```
Load base model + trained LoRA adapters
    ↓ model.merge_and_unload() — mathematically merge B×A into W for each layer
Full-weight model (equivalent to fine-tuned but without adapter overhead)
    ↓ llama.cpp convert.py
GGUF format (FP16)
    ↓ ./quantize model.gguf model_q4km.gguf q4_k_m
GGUF format (Q4_K_M) — ~2.5GB for Gemma3-4B
```

---

## 12. Inference Workflow

### 12.1 Startup Sequence

When the system starts on the edge device:
1. Load XGBoost/RF model from disk (`xgb_model.pkl`, <5MB) → held in RAM
2. Load encoder ONNX session (`encoder_int8.onnx`, ~14MB) → held in RAM
3. Load SHAP explainer (uses the XGBoost model, no additional memory) → ready
4. Load exemplar library (JSON file, <1MB) → held in RAM
5. Load XGBoost exemplar retriever (`retriever.pkl`, <1MB) → held in RAM
6. Load Jinja2 prompt templates (text files, <100KB) → held in RAM
7. Start Ollama in background with Gemma3-4B Q4_K_M → model loaded on first request (~2.5GB when loaded)
8. Start explanation worker thread, listening on sir_queue
9. Begin processing incoming flow windows

### 12.2 Per-Flow-Window Inference (Synchronous, ~80–200ms total)

```
New flow window arrives (W=15s of network flows)
    ↓ [~5ms] Feature extraction + normalization
    
Extract 12–15 features from the window aggregate
Apply MinMaxScaler (saved from training)
Construct boolean domain flags

    ↓ [~1ms] ML baseline

Pass tabular feature vector to XGBoost model
Receive: label, confidence score

If XGBoost says Benign with high confidence → skip encoder, no alert

    ↓ [~2ms] Verbalization

Format B key-value string construction
Tokenization (AutoTokenizer)
Pack into input_ids + attention_mask tensors (shape [1, 256])

    ↓ [~80ms] Encoder SLM inference (INT8 ONNX)

onnxruntime.InferenceSession.run(
    output_names=["logits"],
    input_feed={"input_ids": ..., "attention_mask": ...}
)
Receive: logits (shape [1, num_classes])
softmax → probabilities
argmax → predicted class
Save [CLS] embedding for retrieval

    ↓ [~30ms] SHAP attribution

shap.TreeExplainer(xgb_model).shap_values(tabular_features)
Sort by absolute importance
Select top-5 features with direction labels

    ↓ [<5ms] SIR construction

Combine: enc_label, enc_conf, top_features, domain_flags, flow_summary
json.dumps(sir_dict) → SIR string

    ↓ [<1ms] Queue push

sir_queue.put_nowait(sir)  # non-blocking

→ Return (label, confidence) to caller — DETECTION COMPLETE
   Total: ~120–200ms on ARM Cortex-A53
```

### 12.3 Explanation Generation (Asynchronous, 1–3s)

```
Worker thread picks up SIR from queue (blocking wait)

    ↓ [<10ms] Exemplar retrieval

sir_features = extract_feature_vector(sir["top_features"])
exemplar_idx = xgb_retriever.predict([sir_features])
exemplar = exemplar_library[sir["detection_label"]][exemplar_idx]

    ↓ [<5ms] Prompt construction (Jinja2)

rendered_prompt = prompt_template.render(
    sir=sir,
    exemplar=exemplar,
    attack_descriptions=ATTACK_DESCRIPTIONS,
    normal_baselines=NORMAL_BASELINES
)

    ↓ [1–3s] Decoder inference (Ollama REST API)

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "slm-ids-gemma3-4b",
        "prompt": rendered_prompt,
        "grammar": GBNF_GRAMMAR,
        "stream": False,
        "options": {"temperature": 0.1, "top_p": 0.9, "num_ctx": 2048}
    }
)
explanation = json.loads(response.json()["response"])

    ↓ [<5ms] Storage + dispatch

store_explanation(alert_id=sir["timestamp"], explanation=explanation)
dispatch_to_dashboard(sir, explanation)
```

Low temperature (0.1) is used for decoder inference — this makes the output more deterministic and factual, avoiding creative or speculative language in security explanations. High temperature would increase the risk of hallucination.

---

## 13. Optimization for Resource-Constrained Devices

### 13.1 Encoder Optimization Stack

**Step 1 — INT8 Post-Training Quantization:**
Quantization reduces the bit-width of model weights from 32-bit floating point to 8-bit integers. For weights, this is done by finding a scale factor s and zero point z such that `float_value ≈ s × (int8_value - z)`. The scale and zero point are computed per tensor (dynamic quantization) or per channel (static quantization, more accurate).

Effect on encoder: TinyBERT shrinks from 56MB (FP32) to ~14MB (INT8). Inference speedup: 2–4× on ARM CPUs due to hardware INT8 multiply-accumulate instructions. Accuracy loss: <1% on classification tasks.

**Step 2 — ONNX Export:**
The fine-tuned PyTorch model is exported to ONNX format using HuggingFace's `optimum` library. ONNX Runtime's execution providers (CPU EP, ARM NEON EP) apply additional operator-level optimizations including:
- Graph fusion: combining separate operations (LayerNorm, GeLU, Add) into single optimized kernels
- Memory layout optimization: restructuring tensor data layouts for cache-efficient access
- Operator-level parallelism: running independent attention heads in parallel on multi-core ARM CPUs

**Step 3 — Structured Pruning (optional for ultra-constrained devices):**
Attention head pruning identifies and removes attention heads whose outputs are consistently close to zero (low importance heads). In BERT-class models, studies show 30–50% of attention heads can be removed with less than 2% accuracy loss. After pruning, the model undergoes a short fine-tuning recovery pass. This reduces FLOPs and inference time proportionally — removing 6 of 12 attention heads roughly halves the attention computation cost.

**Step 4 — Knowledge Distillation (offline, optional):**
Train a smaller student encoder (3-layer BERT, 64-dimensional hidden state, ~5M parameters) using the fine-tuned SecurityBERT or TinyBERT as teacher. The student learns to match both the teacher's logits (soft cross-entropy loss) and intermediate layer attention distributions (attention transfer loss). The resulting student model may be 3–5× smaller than TinyBERT while retaining 95%+ of its performance on the specific slow-rate DoS task — because the task is narrow enough that a very small model can specialize effectively.

### 13.2 Decoder Optimization Stack

**Q4_K_M Quantization:**
llama.cpp's Q4_K_M quantization scheme represents each weight using 4 bits on average, with mixed precision — some critically important weight groups (K-matrices in attention) are stored at higher precision, while less critical weights use lower precision. "K_M" means "K-quant, medium" — a balanced setting between Q4_K_S (small, less accurate) and Q4_K_L (large, more accurate).

The effect: Gemma3-4B goes from ~16GB (FP32) → ~8GB (FP16) → ~4.5GB (8-bit) → ~2.5GB (Q4_K_M). At 2.5GB, the model fits in 4GB RAM with operating system overhead (~500MB) and encoder (~14MB), leaving ~1GB headroom.

**KV-Cache Management:**
During autoregressive decoding, the model caches the key and value tensors from each transformer layer for all previously generated tokens (the KV-cache). This avoids recomputing attention over the entire prompt on each generation step. For a 2048-token context and 4B model, the KV-cache is approximately 200–400MB. llama.cpp manages this automatically, but the context length should be set to the minimum necessary (`num_ctx: 2048`) to keep KV-cache RAM bounded.

**Speculative Decoding:**
A small "draft model" (e.g., Gemma3-1B Q4_K_M at ~0.7GB) generates candidate tokens rapidly. The full 4B model then validates these candidate tokens in parallel (it processes all candidates in one forward pass rather than N sequential passes). If the candidate is accepted (probability above a threshold), it is kept; otherwise, the 4B model's own prediction is used. For structured explanation outputs (which have relatively high token predictability due to the JSON schema and domain vocabulary), acceptance rates of 70–80% are achievable, giving 2–3× throughput improvement.

**Ollama Configuration Tuning:**
```
OLLAMA_NUM_PARALLEL=1    # Only one inference at a time (edge device, no concurrency needed)
OLLAMA_MAX_LOADED_MODELS=1  # Only keep one model in RAM at a time
OLLAMA_KEEP_ALIVE=5m    # Unload model after 5 minutes of inactivity
OLLAMA_NUM_THREAD=4     # Use 4 CPU threads (matches ARM Cortex-A55 quad-core)
```

---

## 14. Memory and Computational Considerations

### 14.1 Complete RAM Budget

The following table shows the RAM consumption of every system component at inference time on a 4GB device:

| Component | Technology | Memory (loaded) |
|---|---|---|
| Operating system (Linux/RTOS) | System | ~500MB |
| Python interpreter + imports | Runtime | ~150MB |
| XGBoost model | scikit-learn/joblib | ~3MB |
| SHAP explainer | Python object | ~10MB |
| Encoder SLM | INT8 ONNX | ~14MB |
| Exemplar library + retriever | JSON + XGBoost | ~5MB |
| Jinja2 templates | Text | <1MB |
| Feature preprocessing pipeline | NumPy/scikit-learn | ~20MB |
| sir_queue + SIR objects | Python dict | <5MB |
| **Subtotal (without decoder)** | | **~708MB** |
| Decoder SLM (Gemma3-4B Q4_K_M) | GGUF via Ollama | ~2,500MB |
| KV-cache (2048 token context) | GGUF | ~350MB |
| Ollama process overhead | C++ runtime | ~150MB |
| **Total (with decoder loaded)** | | **~3,708MB** |

The total of ~3.7GB fits within a 4GB RAM budget with ~300MB margin. On a 6GB device, there is comfortable headroom.

**Management strategy:** The decoder's 2.5GB is the dominant cost. Since the decoder is asynchronous and non-critical-path, Ollama can be configured to unload it from RAM after 5 minutes of inactivity (`keep_alive: 5m`). During periods with no alerts (which is most of the time in a well-protected network), the full 4GB is available to the operating system and other processes. The decoder is reloaded (taking ~3–5 seconds) when the first SIR arrives after an inactivity period.

### 14.2 Computational Budget

**ARM Cortex-A53/A55 performance characteristics:**
- 4 cores at 1.5–2.0 GHz
- 2 INT8 MACs per clock cycle per core (hardware acceleration)
- No dedicated neural accelerator (unlike Jetson Nano which has a 128-core Maxwell GPU)
- LPDDR4 memory bandwidth: ~25–30 GB/s

**Encoder inference computational load:**
TinyBERT (4 layers, 312 hidden dim, 12 heads, 256 tokens):
- Attention: 2 × 4 × (256 × 312 + 312 × 312 + 256 × 312) × 12 ≈ 1.5B MACs
- FFN: 2 × 4 × (312 × 1248 + 1248 × 312) ≈ 3.1B MACs
- Total ≈ 4.6B MACs per inference
- At 2 cores × 1.5 GHz × 2 INT8 MACs/cycle ≈ 6 GMACs/s effective throughput
- Estimated time: 4.6G / 6G ≈ 0.77s (FP32) → ~0.2s (INT8 with ONNX optimizations)
- Actual measured: ~157ms (Paper SS-2, ARM Cortex-A53, PyTorch) → ~80ms (ONNX INT8)

**Decoder inference computational load:**
Gemma3-4B (32 layers, 2048 hidden dim, Q4_K_M):
- Each token generation: ~10B MACs (rough estimate for 4B model)
- For a 300-token explanation: 3T MACs total
- CPU throughput: ~2–4 tokens/second on ARM Cortex-A55
- Time for 300 tokens: ~75–150 seconds WITHOUT speculative decoding
- With llama.cpp optimizations (NEON intrinsics, quantized matmul): 3–6 tokens/second
- With speculative decoding (1B draft + 4B full, 75% acceptance): ~8–12 tokens/second
- Time for 300 tokens: ~25–100 seconds

**Important clarification:** The 1–3 second decoder estimate from the literature is measured on more powerful edge hardware (NVIDIA Jetson Nano with 128 CUDA cores, or Intel NUC with 8 CPU cores). On a Raspberry Pi 4 (ARM Cortex-A72, 4 cores, 1.8GHz), realistic decoder throughput is 2–5 tokens/second without a GPU → ~60–150 seconds for a 300-token explanation. This is the **14× slowdown rule** — desktop inference results must be multiplied by ~14 to estimate ARM embedded performance.

**Implication for system design:** On Raspberry Pi-class hardware, explanation generation may take 1–3 minutes rather than 1–3 seconds. This is acceptable architecturally (explanations are advisory, not blocking) but must be disclosed in the thesis as a limitation and measured empirically using CodeCarbon on the actual target hardware.

### 14.3 The 14× Slowdown Rule

Paper SS-2 (Salah et al.) empirically validates that inference latency on ARM Cortex-A53 embedded hardware is approximately 14× higher than on a desktop x86 CPU (Intel i5/i7). This rule is validated for TinyBERT and MiniLM inference specifically. The rule is essential for:
- Converting published benchmark results (typically measured on x86 servers or GPUs) to realistic edge device estimates
- Arguing in the thesis that a model achieves X ms on desktop → 14X ms on target ARM hardware
- Planning which model to use (a model that is 100ms on x86 will be 1,400ms on ARM)

---

## 15. Role of Every Tool, Framework, and Component

### Network Traffic and Flow Processing

**Zeek IDS:** The entry point to the system. Zeek is a passive network monitoring framework that processes raw packets and applies a domain-specific scripting language to extract structured information. For this project, it runs in offline mode over PCAP files from CIC IIoT 2025 and writes `conn.log`. Its key advantage is that it produces `conn_state` and `history` fields — native TCP state machine tracking — that encode Slowloris and RUDY attack signatures without requiring post-hoc feature engineering.

**Scapy:** Python library for packet-level manipulation. Used to extract first-N-packet features (TCP window size from the first 5 packets per flow, timestamp deltas between packets) that are necessary for distinguishing Slowread from Slowloris at the packet level. Zeek's flow-level summary does not preserve per-packet TCP window size sequences — Scapy is used to supplement Zeek's output with these packet-level details.

**tshark:** Command-line version of Wireshark. Used for initial dataset exploration and for filtering PCAP files to specific port ranges (e.g., `tshark -r capture.pcap -Y "tcp.port == 80" -w http_only.pcap`) before Zeek processing — reducing processing time for large captures.

### Data Processing and Machine Learning

**pandas:** The workhorse of data preprocessing. All tabular operations — loading `conn.log`, replacing infinities, dropping duplicates, grouping flows into temporal windows, computing per-window aggregates — are done in pandas DataFrames. Its `groupby().agg()` pattern is used for temporal windowing.

**NumPy:** Underlying numerical computation for scikit-learn and pandas. Used directly for constructing the tabular feature matrix (shape: [n_samples, 15]) and for computing boolean domain flags (vectorized comparisons).

**imbalanced-learn:** Provides `SMOTE(k_neighbors=5, random_state=42)` for oversampling the minority slow-rate DoS class in the training split. The key usage pattern: `X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)` — applied only to training data, never to validation or test.

**scikit-learn:** Provides scalers (`MinMaxScaler`, `StandardScaler`), feature selection (`mutual_info_classif`, `SelectKBest`), cross-validation (`StratifiedKFold`), metrics (`classification_report`, `roc_auc_score`, `confusion_matrix`), and classical classifiers (`RandomForestClassifier`). The scaler's `fit()` → `transform()` pattern enforces the separation between training statistics and test-time application.

**XGBoost:** The primary ML baseline classifier and the exemplar retriever. As a classifier: gradient-boosted decision trees, trained with `objective='multi:softmax'`, `num_class=4`, `max_depth=6`, `n_estimators=200`. As a retriever: trained in a one-vs-all fashion to score similarity between incoming flow features and each exemplar in the library.

**joblib:** Serializes trained scikit-learn and XGBoost models to disk using pickle-based compression. `joblib.dump(model, 'model.pkl')` / `joblib.load('model.pkl')`. Models are typically 1–5MB, negligible for edge deployment.

### Encoder SLM Training and Inference

**HuggingFace Transformers:** The unified API for loading, fine-tuning, and running BERT-class encoder models. Key classes:
- `AutoTokenizer.from_pretrained()`: loads the appropriate tokenizer for any checkpoint
- `AutoModelForSequenceClassification.from_pretrained()`: loads the encoder with a classification head
- `Trainer` + `TrainingArguments`: manages the complete training loop (data loading, optimizer steps, evaluation, checkpointing, early stopping)
- `EarlyStoppingCallback`: stops training when validation macro-F1 does not improve for `patience` evaluations

**HuggingFace datasets:** Provides `Dataset` and `DatasetDict` classes for efficient data loading and tokenization. The `.map()` method applies tokenization to the entire dataset in batches, using multiprocessing. The resulting `Dataset` is an Arrow-backed memory-mapped file, enabling out-of-core processing for large datasets.

**HuggingFace evaluate:** Provides `evaluate.load("f1")` for computing macro-F1 and per-class F1 during training evaluation. Also provides `evaluate.load("rouge")` and `evaluate.load("bertscore")` for explanation quality evaluation.

**Captum:** PyTorch interpretability library. `IntegratedGradients(encoder_model).attribute(input_ids, target=predicted_class)` computes token-level importance scores for the encoder's classification decision. Used during development to verify the encoder attends to semantically meaningful feature tokens (not padding or punctuation tokens). Not used at inference time.

**Weights & Biases (wandb):** Experiment tracking. `wandb.log({"train_loss": loss, "val_macro_f1": f1, "epoch": epoch})` called after each evaluation. Creates interactive dashboards comparing TinyBERT vs. MiniLM runs, showing learning curves, and tracking the effect of different hyperparameters. The `sweep` functionality enables automated hyperparameter search across (lr, batch_size, num_epochs).

**ONNX + optimum:** `optimum-cli export onnx --model ./checkpoint --task sequence-classification ./onnx_model/` converts the PyTorch checkpoint to ONNX graph format. The ONNX graph represents the encoder as a computation graph of standard operators (MatMul, Add, Softmax, LayerNorm) that ONNX Runtime can optimize and execute.

**ONNX Runtime:** Executes the ONNX model on CPU with optimizations. `onnxruntime.quantization.quantize_dynamic()` applies INT8 quantization to the ONNX model. `onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])` loads the optimized model for inference. On ARM CPUs, ONNX Runtime uses NEON intrinsics (ARM's SIMD instruction set) for vectorized INT8 matrix multiplication.

### XAI

**SHAP:** `shap.TreeExplainer(xgb_model)` initializes the explainer once at startup. `explainer.shap_values(feature_vector)` computes exact Shapley values for a single sample in <100ms. Returns a matrix of shape [num_classes, num_features] — the contribution of each feature to each class's prediction score. The column for the predicted class is sorted by absolute value to get the feature importance ranking.

**LIME:** `lime.lime_tabular.LimeTabularExplainer(training_data, feature_names=feature_names, class_names=class_names)` creates a local surrogate explainer. `explainer.explain_instance(feature_vector, xgb_model.predict_proba, num_features=6)` trains a local linear model around the specific sample and returns feature weights. Used as a cross-check against SHAP values — if both agree on the top features, the attribution is more trustworthy.

### Decoder SLM Training and Inference

**bitsandbytes:** Provides 4-bit NF4 quantization of the base model weights. The `BitsAndBytesConfig` object is passed to `AutoModelForCausalLM.from_pretrained()` to load the model in quantized form directly into GPU memory. Also implements "double quantization" — quantizing the quantization constants themselves — for additional memory savings.

**HuggingFace PEFT:** `LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"], lora_dropout=0.05)` defines the LoRA adapter configuration. `get_peft_model(base_model, lora_config)` inserts the adapter matrices into the base model. `model.merge_and_unload()` fuses the trained adapters back into the base weights for export.

**trl SFTTrainer:** Wraps HuggingFace `Trainer` for supervised instruction fine-tuning. Automatically handles the instruction format (system/user/assistant turns), applies the chat template, and computes loss only on the assistant (response) tokens — not on the system prompt or user input. This ensures the model is trained to generate good responses, not to predict input tokens.

**accelerate:** Handles distributed training coordination and mixed-precision training (BF16 for adapter weights while base model stays in NF4). `accelerate launch train.py` automatically detects available hardware and configures the training accordingly.

**llama.cpp:** The C++ inference engine that runs quantized decoder models on CPU. Implements optimized GGUF model loading, KV-cache management, speculative decoding, and GBNF grammar-constrained generation. The GGUF quantization pipeline: `python convert.py merged_model/` → `./quantize model.gguf model_q4km.gguf q4_k_m`.

**Ollama:** A user-friendly wrapper around llama.cpp that adds model management (downloading, versioning, hot-swapping), a REST API server, and concurrency management. The `Modelfile` defines the model configuration:
```
FROM /path/to/decoder_q4km.gguf
SYSTEM "You are a network security analyst..."
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_ctx 2048
```
`ollama create slm-ids -f Modelfile` registers the model. `ollama serve` starts the API server.

**Jinja2:** A Python template engine. Prompt templates are stored as `.j2` files with variable placeholders `{{ sir.detection_label }}`, `{{ exemplar.observation }}`, etc. At inference time, `template.render(sir=sir_dict, exemplar=exemplar_dict, ...)` fills in the current alert data without string concatenation errors. Enables versioned prompt management — different prompt versions can be A/B tested without changing the inference code.

**Optuna:** Bayesian hyperparameter optimization. `study = optuna.create_study(direction="maximize")` / `study.optimize(objective, n_trials=20)`. For QLoRA, the objective function fine-tunes for 1 epoch with a given (r, alpha, lr) combination and returns the validation loss. Optuna uses Tree-structured Parzen Estimator (TPE) to select the next trial, converging on good hyperparameters faster than grid search.

### Retrieval and Knowledge Storage

**ChromaDB:** A vector database for storing exemplar SIR feature embeddings. `chromadb.Client().create_collection("exemplars")` creates a persistent local store. Embeddings are computed from the feature vector (using the encoder's [CLS] embedding or XGBoost's leaf node indices). At retrieval time, the closest embedding in the database is found using cosine similarity. For small exemplar libraries (≤50), plain Python lists are faster.

**FAISS:** Facebook's vector similarity search library. `faiss.IndexFlatL2(d)` creates an exact nearest-neighbor index in d-dimensional space. `index.add(embedding_matrix)` adds all exemplar embeddings. `index.search(query_embedding, k=1)` returns the closest exemplar in O(n) time. More efficient than ChromaDB for large libraries but requires more configuration.

### Evaluation and Monitoring

**CodeCarbon:** `from codecarbon import EmissionsTracker; tracker = EmissionsTracker(project_name="slm-ids-training")`. Tracks CPU/GPU power consumption during training and inference, estimates CO2 emissions based on the electricity grid's carbon intensity at the user's location. Produces a CSV with energy_consumed (kWh) and emissions (kg CO2). Required to argue energy efficiency of the SLM approach versus larger LLM alternatives.

**psutil:** `psutil.Process(os.getpid()).memory_info().rss` returns current RSS (Resident Set Size) memory usage in bytes. Used to monitor RAM consumption during inference and verify the system stays within the 4GB budget. Can also monitor CPU utilization: `psutil.cpu_percent(interval=1)`.

**scipy.stats:** `scipy.stats.friedmanchisquare(*groups)` tests whether multiple models' performance distributions are statistically different (non-parametric, for non-Gaussian distributions). `scipy.stats.wilcoxon(scores_model1, scores_model2)` performs a paired comparison. Required for thesis claims about one model being significantly better than another.

**py-readability-metrics:** `flesch_reading_ease(text)` and `gunning_fog(text)` compute standardized readability scores for the generated explanations. Flesch Reading Ease of 60–70 indicates plain English accessible to high school graduates — the target for security analyst explanations. Used to verify that fine-tuned explanations are more readable than zero-shot LLM outputs.

---

## How It All Fits Together — A Complete Trace

To conclude, here is a complete trace of one Slowloris detection event through the entire system:

1. **t=0s:** Zeek detects a new TCP connection from 10.0.0.45:54321 to 192.168.1.1:80. The connection is established (SYN + SYN-ACK seen) but no HTTP header arrives.

2. **t=15s:** The W=15s window closes. Zeek's conn.log shows `conn_state=S1`, `duration=15.0s`, `orig_pkts=2`, `orig_bytes=120`, `resp_bytes=0`. Behavioral aggregation computes: `header_completeness=0.05` (only the first GET line received, not the complete header), `pkt_rate=0.13 pkt/s`.

3. **t=15.001s:** The preprocessing pipeline loads this window, normalizes features via the saved MinMaxScaler, and constructs the Format B verbalized string: `"FLOW_DURATION: 15.0s PKT_RATE: 0.13pkt/s BYTE_RATE: 8.0B/s HEADER_COMPLETE: 0.05 CONN_STATE: S1 INCOMPLETE_HEADER: True SLOW_RATE: True ..."`.

4. **t=15.002s:** XGBoost classifier receives the tabular feature vector and predicts "Slowloris" with confidence 0.91.

5. **t=15.004s:** Tokenizer converts the verbalized string to 147 tokens. ONNX Runtime runs the INT8 TinyBERT encoder. The `[CLS]` embedding and logits are computed.

6. **t=15.085s:** Encoder output: predicted class "Slowloris", confidence 0.97.

7. **t=15.090s:** SHAP TreeExplainer computes Shapley values on the tabular XGBoost model. Top features: `flow_duration (+0.38)`, `header_completeness (+0.33)`, `pkt_rate (+0.29)`, `conn_state (+0.12)`.

8. **t=15.095s:** SIR Builder assembles the JSON object with all fields populated. Detection result returned to the network monitoring system: "Slowloris, confidence 0.97." The monitoring system immediately blocks new connections from 10.0.0.45.

9. **t=15.096s:** SIR is pushed onto `sir_queue`. Main detection thread returns to monitoring the next window.

10. **t=15.096s:** Background explanation worker picks up the SIR. XGBoost retriever finds the closest Slowloris exemplar in the library (one that also had `header_completeness ≈ 0.08` and `pkt_rate ≈ 0.02`).

11. **t=15.100s:** Jinja2 renders the full prompt: SSRP 16-factor system prompt (loaded from template) + domain knowledge block + retrieved Slowloris exemplar with CoT trace + the SIR JSON block.

12. **t=15.105s:** Ollama receives the POST request with the rendered prompt and GBNF grammar. Gemma3-4B Q4_K_M begins generating the response token by token.

13. **t=17.4s:** Decoder generates the 287-token JSON explanation. Grammar constraints ensure valid JSON output. The explanation is stored in the alert log and dispatched to the security analyst dashboard.

14. **t=17.4s:** The security analyst sees the alert (raised at t=15.085s) and, two seconds later, the explanation: "Observation: A TCP connection was maintained for 15 seconds with only 2 packets and 120 bytes... Evidence: header_completeness of 0.05 confirms only the first line of the HTTP GET request was transmitted... Conclusion: Slowloris HTTP connection-exhaustion DoS attack... Mitigation: 1. Enforce server-side HTTP header timeout of 10 seconds..."

The entire detection-to-explanation cycle takes 2.3 seconds. The detection decision (which triggers the blocking action) is made in 85ms. The explanation arrives 2.3 seconds later — advisory, contextual, and grounded in the exact features that caused the alert.

---

*End of architecture_deep_dive.md*
*This document explains the complete proposed system architecture based on synthesis of 46 reviewed papers for the SLM-based slow-rate DoS detection project.*
