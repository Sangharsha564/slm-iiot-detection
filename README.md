# Slowloris Detection and Explanation for IIoT

A two-stage system for detecting Slowloris slow-rate DoS attacks in Industrial IoT networks and explaining each alert in plain language.

- **Stage 1 – Detection.** Network flows are written as short text and classified by a compact language model (BERT-mini, 11M parameters) that runs on a Raspberry Pi.
- **Stage 2 – Explanation.** For each alert, Integrated Gradients finds the features that drove the decision, and a small language model (Qwen2.5-1.5B) turns them into an explanation and mitigation advice, optionally with retrieved security knowledge (RAG).

## Key results (BERT-mini, unseen test device)

| Metric | Value |
|---|---|
| MCC | 0.937 |
| Precision | 1.000 |
| Recall | 0.899 |
| Latency on Raspberry Pi 4 (1 flow) | 72.5 ms |

The model is trained on one device (edge1) and tested on a device it has never seen (wisenet-camera).

## Repository structure

| Folder | Contents |
|---|---|
| `Dataset_preprocessing/` | Builds the dataset from packet captures and selects features |
| `data/processed/` | Processed dataset (`v3` readable text, `v3_rawnames` raw feature names) |
| `configs/` | Model, experiment and environment settings |
| `src/` | Training, explainability (`xai_*`), Stage 2 (`stage2_*`, `rag_*`) and pipeline scripts |
| `experiments/` | Metrics and predictions for every training run |
| `results_main/` | Final Stage 1 comparison tables and figures |
| `results/` | XAI (`07_xai`), Stage 2 (`08_stage2`) and live pipeline (`10_pipeline`) results. Folders `01–06` are an earlier version of the Stage 1 results |
| `pi/` | Deployed model and Raspberry Pi scripts |

## Setup

```
conda env create -f configs/environment.yml
conda activate slm_stage1
```

## How to run

Run all commands from the repository root.

**1. Build the dataset** (needs the original pcap files, not included)

```
editcap -i 62 benign_whole-network3.pcap benign_chunks/chunk.pcap
python Dataset_preprocessing/datasetbuildv3.py <pcap_folder> ./benign_chunks
python Dataset_preprocessing/preprocess.py ./dataset_out_v3
# copy the contents of dataset_out_v3/ into data/processed/v3/
python src/preprocess_rawnames.py .
```

**2. Train and compare models (Stage 1)**

```
python src/train.py --model bert-mini --exp-config experiment_readable192.yaml --seed 42
python src/compare_main.py
```

**3. Explain decisions (XAI)**

```
python src/xai_ig.py --run experiments/stage1_full_13_readable192/bert-mini_s42
```

**4. Generate and score explanations (Stage 2)**

```
python src/rag_build.py --docs docs/
python src/select_explainer_sample.py --run experiments/stage1_full_13_readable192/bert-mini_s42
python src/select_fewshot_examples.py --run experiments/stage1_full_13_readable192/bert-mini_s42
python src/stage2_generate.py --condition rag        # also zero_shot, few_shot
python src/stage2_eval.py --condition rag
python src/stage2_report.py
```

**5. Live pipeline (Raspberry Pi + laptop)**

```
# laptop
python src/analyst_explain.py --broker <pi-ip>
# Raspberry Pi
python3 pi/edge_detect_final.py --ckpt pi --csv demo_flows.csv
```

## Notes

- **Model weights.** Only the deployed BERT-mini model is included (`pi/model.safetensors`). Other models can be recreated with `src/train.py`.
- **Not included.** The raw pcap files and the RAG source documents (`docs/`, `rag/`) are not included for size and copyright reasons. The RAG index is built from NIST SP 800-94, RFC 9293, MITRE ATT&CK T0814, and articles on Slowloris from Cloudflare, Imperva and the original ha.ckers.org post.

## Limitations

- Benign traffic comes from a single capture, so capture-specific patterns may remain.
- The validation split for benign traffic is random rather than by time window.
- The Pi–laptop link uses MQTT without authentication or encryption. It's a demonstration setup, not a production deployment.
