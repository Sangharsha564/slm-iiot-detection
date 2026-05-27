# Research Notes — Folder 1: Slow-Rate Detection
## Session saved: 2026-05-24
## Papers read: 7 of 11

---

## PAPER 1: Delays Have Dangerous Ends: Slow HTTP/2 DoS Attacks Into the Wild and Their Real-Time Detection Using Event Sequence Analysis
- **Author**: Tripathi, N. | **Venue**: IEEE TDSC 2024 (Q1, IF~7)
- **DOI**: 10.1109/TDSC.2023.3276062
- **Focus**: First real-time detection scheme for Slow HTTP/2 DoS attacks

### Detection Method
- Event sequence analysis with two databases:
  - **D_lookahead**: Normal HTTP/2 event pair patterns (window size n)
  - **D_delay**: Max inter-event delays
- Assigns mismatch score to each incoming event sequence
- Anomalous if mismatch score > threshold t
- Input: cleartext HTTP/2 flows captured via PolarProxy reverse TLS proxy
- Attack types covered: Zero Window, Incomplete POST/GET body, Incomplete HEADERS frame, Unacknowledged SETTINGS, Connection Preface only

### Dataset
- Custom dataset: ~5GB HTTP/2 traffic from academic quiz site (COVID period)
- Geographically diverse students + 5 attack types

### Results
- Accuracy ~98.65%, FPR ~1.7%, Recall 100%, Precision ~93%
- Best: window n=5, t=0.02
- RAM: ~1.92GB; detects virtually all attacks in <30 seconds

### Comparison
- Outperforms: Chi-square approach, DBSCAN, K-means

### Limitations
- No public dataset, no ML, no explanation, HTTP/2 only, threshold tuning required

---

## PAPER 2: HTTP Low and Slow DoS Attack Detection using LSTM-based deep learning
- **Author**: Gogoi & Ahmed | **Venue**: IEEE INDICON 2022

### Detection Method
- LSTM sequence classification; input = per-IP time series of (timestamp, requests_per_second), 180 timesteps

### Dataset
- CIC DoS 2017 (4.6GB, 24h, 261,026 HTTP requests, 32,318 attack from 8 tools)
- + 10,000 synthetic attack + 10,000 synthetic normal traces
- Attack tools: hulk, rudy, goldeneye, slowloris, slowbody2, ddossim, slowheaders, slowread

### Architecture
- Masking → LSTM(64) → Dropout(0.3) → LSTM(32) → Dropout(0.3) → Dense(8) → Dropout(0.3) → Dense(1, sigmoid)
- Adam optimizer, Binary CE loss, LR=0.01, 40 epochs, batch=512, dropout=0.3, uniform weight init

### Results
- Accuracy=0.99, AUC=0.998, FP/FN ~0.01
- Train/test: 70/30 split

### Preprocessing
- Parse PCAP using scapy/tcpdump; isolate attack traffic by matching timestamps + IPs
- Generate synthetic data mimicking CIC distribution

### Limitations
- Tiny real attack trace set (only 8 real attack traces); relies heavily on synthetic data; no comparison to other DL methods

---

## PAPER 3: slowTrack — Detecting slow rate Denial of Service attacks against HTTP with behavioral parameters
- **Author**: Sood & Hubballi | **Venue**: Journal of Supercomputing 2024

### Detection Method
- 8 behavioral parameters measured every W=15s window, 3 detection algorithms
- **Parameters**: (1) Incomplete Requests, (2) Response Time, (3) Response Messages, (4) Context Switches, (5) HTTP 2xx Success Codes, (6) HTTP 4xx Error Codes, (7) TCP Sockets on server port, (8) Server Processes
- Correlation with incomplete requests: positive (RT, CS, 4xx, Sockets, Processes); negative (RM, 2xx)
- **Algorithm 1 (Majority Voting)**: if InComp ≥ upper threshold → attack; else check 7 params, attack if ≥5 violate
- **Algorithm 2 (Decision Tree J48)**: binarized parameters fed to Weka
- **Algorithm 3 (Association Rule Mining)**: top 6 association rules from Weka Apriori
- Thresholds: avg ± 25% from normal training data

### Dataset
- Custom testbed (4 VMs: Apache, script-2 client, Kali+Slowloris, Ubuntu+JMeter; 10h, 2h attack)
- + live academic network (48h, Slowloris from AWS VM)

### Tools Used
- Python, psutil (num_ctx_switches, net_connections), scapy (AsyncSniffer), subprocess, Apache access logs, Weka, JMeter, Slowloris, Kali Linux

### Results
- Exp-1: Majority Voting F1=99.89%, DT F1=99.58%, ARM F1=98.53%
- Exp-2: Majority Voting F1=99.79%, DT F1=99.92%, ARM F1=99.66%
- Robustness: Tested with random UA, SOCKS5 proxy, 50-5000 socket variants, Nginx server, 30-90 req/s load

### Limitations
- No explanation; needs both server+client agents; Apache/Nginx only; threshold tuning manual

---

## PAPER 4: Optimizing Feature Selection for LDDoS Detection (DNN + SMOTE + RFE)
- **Author**: Al-Shukaili, Kiah & Ahmedy | **Venue**: Discover Internet of Things 2025

### Detection Method
- Attack types: Slowloris + Slowhttptest
- Feature Selection: Filter (SelectKBest + MI) vs Wrapper (RFE + RF), k=20 or 50
- DNN with class balancing (SMOTE on training only)

### Dataset
- CIC-IDS2017 (79 features, 692,703 instances)
- Preprocessing: Infinite→NaN, mean imputation, MinMaxScaler, LabelEncoder, stratified 70/30

### Architecture
- Input(n) → Dense(128, ReLU, L2=0.01) → BatchNorm → Dropout(0.45) → Dense(64, ReLU, L2=0.01) → BatchNorm → Dropout(0.45) → Dense(1, sigmoid)
- Adam optimizer, Binary CE, Early Stopping (patience=10), 5-fold CV, threshold tuning via PR curve

### Results
- **Best (Wrapper-50)**: Accuracy=99.77%, Precision=95.27%, Recall=95.63%, F1=95.45%, AUC=97.76%
- Key finding: Wrapper-based RFE > Filter-based SelectKBest; 50 features > 20 features

### Tools
- TensorFlow Keras (Sequential API), scikit-learn, Python
- Hardware: Intel i3-3227U, 4GB RAM (also Raspberry Pi 4 compatible)

### Limitations
- CIC-IDS2017 only, no adversarial testing, computationally heavy wrapper selection

---

## PAPER 5: Resource-Efficient Low-Rate DDoS Mitigation With Moving Target Defense in Edge Clouds (RE-MTD)
- **Author**: Zhou et al. | **Venue**: IEEE TNSM 2025 (Vol 22, No 1)
- **DOI**: 10.1109/TNSM.2024.3413685

### Detection/Mitigation Method
- **NOT a detection paper** — this is a proactive Moving Target Defense (MTD) + mitigation paper
- Three lightweight MTD mechanisms for container-based edge clouds:
  1. **Service Scaling**: Dynamically scale up/down containers to handle more connections
  2. **Service Replicas**: Create/delete container copies to redistribute load
  3. **Port Hopping**: Change port numbers using SHA-256 hash with shared session key + NTP sync
- DRL-based optimal deployment using Deep Q-Network (DQN) algorithm
- Formulates attack-defense as Markov Decision Process (MDP)
- State space: per-service tuple (running status, connections, resource consumption, port number)
- Action space: Scale up/down, create/delete replicas, hop port
- Reward function: 4 factors — service quality, security status, deployment resources, time overhead
- **No traffic classification** — detects via service state changes, reacts proactively

### Dataset/Evaluation
- Custom EC simulation using OpenAI Gym
- Four attack scenarios: single/multiple targets, constant/variable attackers (10–50 attackers)
- Real testbed: Dell PowerEdge servers, OpenStack, Docker/bWAPP, SlowHTTPTest tool
- Compared to: OpenMTD, FastMove, CMDP-MOS, ID-HAM

### Results
- Security improvement: up to 31.7% over baselines
- Service quality improvement: up to 26.95% (QoS)
- Lowest response time: 276.66 ms (vs >1s for alternatives at μ=800)
- Lowest page load time: 1.413 s
- Only 2.44% additional memory usage
- DQN > Deterministic > Random strategy

### Tools/Frameworks
- Python, OpenAI Gym, DQN (TensorFlow/PyTorch implied), Kubernetes/Docker Swarm (mentioned), Ryu SDN controller, OpenStack, SlowHTTPTest, Apache benchmarking tool (ab)

### Limitations
- No attack detection module; cannot handle unlimited attacker resources; fails against insider attackers; may degrade with network environment changes

---

## PAPER 6: Enhanced detection of low-rate DDoS attack patterns using machine learning models (FLD-LRDDoS)
- **Author**: Bocu & Iavich | **Venue**: Journal of Network and Computer Applications 2024 (Elsevier JNCA)

### Detection Method
- **Federated Learning (FL) + Bidirectional LSTM (Bi-LSTM) + Attention mechanism**
- Three modules: Data Preprocessing, Detection (Bi-LSTM), Global Aggregation (Async FL)
- Sliding windows mechanism for time-series feature extraction (30s intervals)
- Features: DPSF (statistical aggregation of IP packet features at 30s), CT (communication time), LFDP (Length of First Data Packet)
- Bidirectional LSTM: 2 LSTM layers with 128 neurons each (forward + backward)
- Attention mechanism to reweight feature importance
- 2 fully connected Dense layers (128, 1)
- Dropout rate: 0.29 (optimized via grid search)
- L2 normalization for overfitting prevention
- SGD optimizer, Binary Cross-Entropy loss
- **Asynchronous FL**: main node selection algorithm based on IP pool size and data recency
- Cache memory layer for classification speedup

### Dataset
- **300 million data samples** from real corporate Romanian IT company network
- 150M LRDDoS, 150M legitimate; 15% transformed to random noise
- Attack types: slow body, slow header, RUDY, HULK
- Extended evaluation: Shrew, CICADAS, Slowloris, SlowDrop, LoRDAS
- 10 FL nodes with different dataset sizes (30M–280M)
- Vagrant virtualization; AMD EPYC 7702, 256 GB RAM

### Results
- Best accuracy: 98.79% (FL distributed), cross-entropy loss 0.076 (at 300M samples)
- Local node only: 92.29% max accuracy
- Average detection time: 0.78-0.81s per sample
- Outperforms: LSTM (96.29%), Bi-LSTM (96.55%), Bi-LSTM+Dropout (96.82%), TAE, VLSTM, FL-LSTM, CNN-LSTM, Hybrid CNN+GRU, FSL-SCNN, RF, SVM, KNN
- Comparison vs Yungaicela-Naula2022 (~95.67%) and Tang2021 (~93.27%)

### Tools
- Python, Vagrant, Bi-LSTM (Keras implied), scikit-learn equivalents, grid search

### Limitations
- Data is confidential, not publicly available; only 3 features extracted; no adversarial evaluation

---

## PAPER 7: Minimal Overhead Modelling of Slow DoS Attack Detection for Resource-Constrained IoT Networks
- **Author**: Reed, Dooley, Kouadri Mostefaoui (Open University) | **Venue**: Future Internet 2025, 17, 432 (MDPI Open Access)

### Detection Method
- **Limited Packet Capture (LPC)** strategy — only 2 attributes: packet length (lp) and inter-arrival time (Δt)
- Decision Tree classifier (J48 in Weka) with default parameters
- Binary classification: Legitimate Node + Slow Node (SN) vs Malicious Node (MN)
- Key innovation: includes genuine slow nodes (high latency, 1500-3000ms) in evaluation

### Dataset
- Custom HTTP slow DoS dataset from LIVE IoT network (Open University)
- Nodes: DHT22, TME, 2TH sensors on Raspberry Pi boards
- Apache web server (max 150 concurrent connections, Ubuntu 20.04)
- Attack tools: slowhttptest at non-default settings (500 CNX, 50 RPC vs default 1000/200)
- Packet capture via Wireshark → CSV
- LPC dataset: 4905 packets, 0.52 MB (vs full 620,678 packets, 396MB = 99.8% reduction)
- Three attack variants: slow GET (238s), slow Read (240s), slow POST (120s)
- Compared to CIC-IoT dataset 2023

### Feature Selection Justification
- Information gain ranking: lp=0.84, Δt=0.55 (highest among 8 attributes evaluated)
- Source port and destination IP excluded (node-identity specific)

### Results
- Without SN: 96.2% accuracy, LN precision=91.5%, MN precision=98.3%
- With SN: 95.9% accuracy, LN/SN precision=94.3%, MN precision=97.0%
- Key finding: SN degrades accuracy — misclassifications increase from 47 to 88 instances

### MPerf Model
- Equation: MPerf(n) = ωA·A(n) − ωC·C(n)
- Quasi-optimal at n≈2000 packets → ~98% accuracy achievable
- Accuracy saturates between 10,000-20,000 packets

### Tools
- Weka (J48 Decision Tree), Wireshark, SPSS (info gain ranking), JMeter (traffic generation), Python Scapy (implied), Cisco 2800 routers, Apache 2.4

### Limitations
- Single IoT network testbed; stealthy attacks only partially explored; no real adversarial adaptation; Linear cost model assumption

---

## PAPER 8: P4-Assisted Slowloris DDoS Attack Detection in IoT Environments by Using ML and DL
- **Author**: Ramirez-Martinez, Pérez-Díaz, Yungaicela-Naula | **Venue**: Computer Networks 2025 (Elsevier)
- **DOI**: 10.1016/j.comnet.2025.111364

### Detection Method
- **P4 programmable switches** + external IDS/IPS framework
- Hybrid data plane (P4) + control plane (ML/DL IDS) approach
- P4 switch calculates 12 features per flow using TCP FIN flags as flow delimiter
- Features sent to external IDS via cloned packet with custom header
- IPS modifies P4 match-action tables to block attacker IPs (after 3 consecutive detections)

### Features (12 total, calculated in P4 from IP+TCP headers)
- Duration, Packet count, ACK/PSH/RST/URG/SYN counts, Min/Max packet IAT, Length Min/Max, Flow IAT
- All computable in linear time (no loops/float ops — P4 constraint)
- Flow identified by {IPSrc, IPDst} tuple, end marked by FIN flag

### Dataset
- CICIoT2023 (Canadian Institute for Cybersecurity): 105 IoT devices, 33 attack types
- Slowloris: 41,437 flows; Benign: 81,490 flows
- 80/20 train/test split; 20 runs × 3 folds averaged
- Feature normalization using mean/STD (z-score) saved from training

### Models Evaluated
| Model | Accuracy | Training Time | Testing Time |
|-------|----------|--------------|--------------|
| KNN | 96.34% | 0.34s | 2.96s |
| RF | 97.02% | 82.86s | 1.73s |
| DT | 96.06% | 0.22s | 0.01s |
| LSTM | 95.76% | 85.77s | 1.83s |
| CNN | 95.77% | 52.79s | 1.71s |
| GRU | 95.80% | 119.92s | 2.72s |
| MLP | 95.42% | 23.97s | 1.40s |

- ML > DL in accuracy due to simple P4-computable features
- DT = fastest; RF = highest accuracy → selected for deployment
- CNN selected for final experiments (balance accuracy/speed)

### Testbed Results (RF + CNN after fine-tuning with Slowloris tool traffic)
- RF: 98.28% accuracy, Benign recall=97.57%, Slowloris F1=97.21%
- CNN: 98.11% accuracy, Benign recall=97.46%, Slowloris F1=96.99%
- Detection time: ~20.629s average (expected for slow-rate)
- Mitigation time: ~0.128s average

### Testbed Setup
- Mininet + BMv2 switches; Ubuntu VM (4 cores, 16GB RAM)
- Topology: border P4 switches protecting infrastructure from IoT attacker network
- Slowloris tool (gkbrk/slowloris) targeting Apache server
- Benign traffic replayed via Tcprewrite/Tcpreplay from CICIoT2023 PCAPs
- Python Scapy library for IDS packet sniffing
- P4Utils for match-action table modification (mitigation)
- Additional data: Aposemat IoT-23 dataset for testing

### Tools/Frameworks
- P4 language, BMv2 software switch, Mininet, Python Scapy, DPKT, Pandas, Jupyter notebooks
- Hyperparameter tuning: random search + grid search
- Weka mentioned indirectly; mainly Python/Sklearn equivalents

### Limitations
- Only Slowloris evaluated (not full slow-rate family); simulation only (not physical hardware); MAC-address-based blocking may fail in multi-segment networks; Benign traffic misclassification issue

---

---

## PAPER 9: Detection of Slowloris Attacks using Machine Learning Algorithms (FRE)
- **Author**: Rios, Inácio, Magoni, Freire | **Venue**: ACM SAC 2024, Avila, Spain
- **DOI**: 10.1145/3605098.3635919
- **Focus**: Comparing 9 ML algorithms + novel FRE hybrid method for Slowloris detection

### Detection Method
- **Features (2 only)**: Shannon entropy + amount of distinct source ports (sliding window ΔT=1s)
- After testing 4 features (number of packets, average IAT, entropy, distinct ports), selected only 2 most discriminative
- **9 ML algorithms**: KNN, GNB, MLP, SVM, DT, MNB, RF, XGB (Gradient Boosting), LGBM
- **Novel FRE method** (Fuzzy Logic + Random Forest + Euclidean Distance):
  1. Classify with FL (trapezoidal membership, 9 Mamdani rules, low/medium/high for entropy and ports)
  2. Classify with RF
  3. If FL==RF → final decision; else apply Euclidean Distance to training data
  4. Warning alert if ED gives split result; blocks traffic if distinct ports > 80

### Dataset
- 4 custom distributed Slowloris datasets (publicly available on Zenodo):
  - Emulated (Netkit, 4 attackers): 40.6 MB
  - LAN (IFTO Brazil, 5 attackers): 1.0 GB
  - MAN-IF (4 city-distributed attackers → IFTO server): 165.51 MB
  - MAN-Pal (4 city-distributed attackers → Palmas municipal server): 23.6 MB
- Merged into one dataset for 10-fold cross-validation
- Attack: 180s normal → 60s Slowloris (slowhttptest tool) → normal
- Train/test: 75/25 split

### Results (all averaged over 10 folds)
| Algorithm | Test Acc | Precision (attack) | Recall (attack) | F1 (attack) |
|-----------|----------|--------------------|-----------------|-------------|
| MNB | 97.98% | 93.47% | 89.25% | 91.30% |
| SVM | 98.67% | 99.01% | 89.70% | 94.10% |
| GNB | 99.10% | 93.31% | 99.70% | 96.36% |
| MLP | 99.24% | 98.61% | 94.93% | 96.72% |
| DT | 99.29% | 96.79% | 97.31% | 97.04% |
| KNN | 99.35% | 97.35% | 97.16% | 97.24% |
| XGB | 99.49% | 97.39% | 98.36% | 97.85% |
| LGBM | 99.52% | 97.66% | 98.36% | 98.00% |
| RF | 99.52% | 97.25% | 98.81% | 98.01% |
| **FRE** | **99.80%** | **98.68%** | **99.70%** | **99.18%** |

- FRE consumes least CPU (137.10%), more RAM (112.85MB), execution time 0.69s
- Hyperparameter optimization via RandomizedSearchCV gave only marginal improvement
- 5 algorithms (DT, KNN, XGB, LGBM, RF) all >95% → suitable for deployment

### Tools/Frameworks
- scikit-learn (all 9 ML algorithms), Python
- Netkit (emulated environment), slowhttptest, tshark (traffic capture)
- Datasets on CERN Zenodo (DOI: 10.5281/zenodo.8316038)

### Limitations
- Only Slowloris attack covered (not full slow-rate family)
- 2-feature approach may produce false positives from dense legitimate traffic (multiple sources in 1s window)
- Custom datasets; not tested against encrypted traffic or adversarial variants

---

## PAPER 10: Survey on Low-Rate DDoS Attacks, Detection and Defense
- **Author**: Drinić & Čiča | **Venue**: INFOTEH-JAHORINA 2024 (IEEE)
- **DOI**: 10.1109/INFOTEH60418.2024.10496020
- **Type**: Survey paper (not a detection implementation)

### Key Classification of Detection Methods (with % share in literature):
1. **Filter detection** (time/frequency domain) — 20%
2. **Feature-based detection** — 10%
3. **Anomaly-based detection** — 5%
4. **ML using supervised learning classifiers** — 30% (most popular)
5. **ML using neural networks** — 20%
6. **Target-specific detection** — 15%

### Detection Categories Covered:
- **Filter**: RED/RRED (Robust RED), FRRED (Fair Robust RED), UTR analysis, network multifractal
- **Feature-based**: Factorization Machine (SDN flow rule duration), self-similarity + Hurst coefficient (botnet detection)
- **Anomaly-based**: Generalized entropy + information distance metrics (Xiang et al.)
- **Supervised ML**: KNN, SVM, DT, RF, LG, DNN (Siracusano et al.); RF > CNN for encrypted traffic; Federated Learning (Liu et al., Ali et al.)
- **Neural Networks**: FeedForward + CNN hybrid (Ilango et al.); Deep Learning for SDN-IoT (Alashhab et al.)
- **Target-specific**: Cloud/data center (CNN + autoencoder + dynamic mitigation); TOR network (delta-time classification for Tor's Hammer)

### LR-DDoS Attack Types Covered:
- TCP-targeted: Shrew (square-wave, synchronizes with TCP RTO)
- Application layer: Slowloris, RUDY, Slow Read, SlowDroid, HTTP/2 slow attacks
- SDN-targeted: TCAM exhaustion/saturation
- Infrastructure: LoRDAS (service queue exhaustion)

### Key Findings/Gaps Identified:
- ML (supervised classifiers) is the dominant approach (30% of papers)
- Most papers target SDN or modern network topologies
- Critical gaps: need for public datasets from diverse environments; mobile phones as attack vectors; IoT/industrial networks; virtual machine networks
- Detection success range for ML: 92%-99.5%

### Tools Referenced
- CIC-IDS2017 dataset, CICFlowMeter, ns-2/ns-3, Mininet, Ryu SDN controller
- Federated learning for distributed detection (IoT/data centers)

---

## PAPER 11: Detection and Mitigation of Low-Rate Denial-of-Service Attacks: A Survey
- **Author**: Rios, Inácio, Magoni, Freire | **Venue**: IEEE Access 2022 (Open Access)
- **DOI**: 10.1109/ACCESS.2022.3191430
- **Type**: Comprehensive survey — the most detailed taxonomy of LDoS attacks in the literature

### Taxonomy of LDoS Attacks (3 categories):

#### A. QoS Attacks (Transport/Network Layer — TCP throughput degradation)
1. **Shrew Attack** (2003): Square-wave pulses synchronized with TCP RTO → reduces throughput to near zero. IP spoofable. No tool.
2. **NoTuLA** (2005): Tunable bursts (monitoring phase + link capacity estimation) → more adaptive than Shrew
3. **PDoS** (2005): Targets TCP RTO (timeout-based) AND congestion window/AIMD (aimd-based); sync and async modes
4. **NewShrew** (2014): Targets RTO + slow start mechanism simultaneously
5. **Full-Buffer Shrew** (2006): High-rate bursts only when router buffer is full → max damage, min resources
6. **RoQ Attack** (2004): Reduces QoS by flooding border router queue; targets any transport protocol
7. **CICADAS** (2016): Decentralized coordinated bots; no central controller; synchronizes via feedback-based RTT adjustment

#### B. Slow DoS Attacks (Application Layer — HTTP connection exhaustion)
1. **Slowloris** (2009): Partial HTTP GET requests with \r\n; keeps connections open; IP non-spoofable; tool: slowloris.pl/py
2. **SlowReq/Slowcomm** (2014): Mix of Slowloris + RUDY; GET requests + single-char packets
3. **RUDY** (2016): Slow HTTP POST; 1-byte per 10s; fake form data with huge Content-Length; tool: R-U-Dead-Yet
4. **Slow Read** (2012): TCP flow control exploit; tiny receive window; tool: SlowHTTPTest
5. **SlowDroid** (2014): SlowReq on Android smartphone; tool: SlowDroid
6. **Slow Next** (2015): Exploits HTTP persistent connections; bogus timeout values
7. **SlowDrop** (2019): Drops HTTP responses to simulate poor wireless; forces server to retransmit endlessly
8. **HTTP/2 DoS** (2018): 5 variants: zero window_size, reset end_headers/stream, connection preface only, incomplete headers, unacknowledged SETTINGS

#### C. Service Queue Attacks
1. **LoRDAS** (2007): Overloads server service queue; watches for response timing to inject attack requests

### Detection Methods (per attack type):
**Shrew**: RRED, FRRED, DTW, HAWK (weighted choking), DFT/NCAS, entropy (Shannon/Renyi/Hartley/Generalized), EWMA, CPR, SEDP, KPCA, SSM, PCA, CUSUM, SADBSCAN (self-adaptive DBSCAN), P&F framework (SDN)
**RoQ**: FMT flow monitoring, spectral analysis + hypothesis testing, CUSUM + autocorrelation, CCID, self-adjusting SVM + APSO
**Slowloris**: SBID (statistical distribution), SeVen (state-based), FFT + mutual information, Naive Bayes IDS, threshold-based (data volume/context switches/sockets), 3-step connection count, MLP-GA, SDToW (HTTP GET pattern)
**NewShrew**: Fisher g-statistics test, wavelet + CUSUM (Vanguard)
**RUDY**: KNN + C4.5 decision tree (C4.5N best)
**HTTP/2 DoS**: Chi-square test on 5 HTTP/2 features (FPR=0%, Recall 8.33%-100% across scenarios)
**LoRDAS**: RAI, RST, RTQB, IRTQB (randomized service time + queue blocking)

### Attack Tools Catalog (Table 7):
| Tool | Year | Type | OS | Language |
|------|------|------|----|----------|
| slowloris.pl | 2009 | Slowloris DoS | Linux | Perl |
| PyLoris | 2009 | Slowloris DDoS | Win/Mac/Linux | Python |
| QSlowloris | 2009 | Slowloris DDoS | Windows | - |
| SlowHTTPTest | 2011 | Slowloris+SlowRead | Win/Mac/Linux | Python |
| Http Bog | 2011 | Slow Read | Windows | C# |
| Torshammer | 2012 | RUDY | Linux | Python |
| Goloris | 2014 | Slowloris DoS | Mac/Linux | Go |
| SlowDroid | 2014 | SlowReq/Slowcomm | Android | - |
| slowloris.py (gkbrk) | 2015 | Slowloris DDoS | Mac/Linux | Python |
| R-U-Dead-Yet | 2016 | RUDY | Linux | Python |
| Cyphon | 2018 | Slowloris DoS | Mac | Perl |
| sloww | 2018 | Slowloris DoS | Linux | JavaScript |
| dotloris | 2018 | Slowloris DDoS | Windows | C# |
| pwnloris | 2018 | Slowloris DDoS | Win/Linux | Python |

### Key Reference for Taxonomy (Figure 2 in paper):
- Full taxonomy diagram: DDoS → High-Rate / Low-Rate → QoS attacks / Slow DoS attacks / Service queue attacks

### Limitations of Existing Work (per the survey)
- Most papers target Shrew attacks; Slow DoS attacks less covered in detection literature
- HTTP/2 DoS detection very limited (only Chi-square from Tripathi)
- SlowDrop, SlowNext, SlowReq — almost no detection methods exist
- Public datasets for distributed attacks very scarce (most self-generated)

---

## FOLDER 1 COMPLETE: All 11 papers read. Session saved: 2026-05-24
## Papers read: 11 of 11

## ALSO TO READ:
- Folder 2: LLM for IDS (12 papers)
- Folder 3: Detect + Explain (9 papers)
- Folder 4: Background (7 papers)
- Supervisor shared (7 papers)
