"""
Encoder SLM — Fine-tuning Script v2
=====================================
Fine-tunes DistilBERT + LoRA as an 8-class network-flow classifier.
Input: key-value verbalized text with Boolean domain flags (Verbalizer v2).
Loss : class-weighted CrossEntropyLoss — no SMOTE (original class dist).
Logs everything to MLflow.  Saves model, report, plots.

Changes from v1:
  - 8 classes (MitM and malware separated)
  - Key-value input format with Boolean domain flags
  - 34 selected features (XGBoost + MI combined selection)
  - Log1p + RobustScaler preprocessing

Run from project root:
    python src/training/train_encoder_slm.py

Outputs (in models/encoder_slm/):
    best_model/          ← HuggingFace model dir (LoRA weights)
    classification_report.txt
    training_curves.png
    confusion_matrix.png

MLflow:
    experiment → 'encoder-slm'
    run        → logged params, metrics, artifacts
"""

import os, sys, time, json, pickle, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mlflow
from sklearn.metrics import (
    classification_report, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
import yaml

# ── Project root ───────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.utils.reproducibility import set_seed, get_device
from src.preprocessing.verbalize import Verbalizer

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════
with open(os.path.join(ROOT, 'configs', 'project_config.yaml')) as f:
    cfg = yaml.safe_load(f)

PREP_DIR  = os.path.join(ROOT, cfg['paths']['preprocessed_dir'])
MODEL_DIR = os.path.join(ROOT, 'models', 'encoder_slm')
FIG_DIR   = os.path.join(ROOT, cfg['paths']['figures_dir'])
LOG_DIR   = os.path.join(ROOT, cfg['paths']['logs_dir'])
for d in [MODEL_DIR, FIG_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

SEED      = cfg['project']['seed']
LABEL_MAP = {int(k): v for k, v in cfg['dataset']['label_map'].items()}
N_CLASSES = cfg['dataset']['n_classes']

# ══════════════════════════════════════════════════════════════════════════
# ★ MODEL SELECTION — change this one line to switch models
# ══════════════════════════════════════════════════════════════════════════
# Options:
#   'distilbert'  → distilbert-base-uncased   (67M)  — our baseline
#   'roberta'     → roberta-base              (125M) — stronger encoder
#   'securebert'  → ehsanaghaei/SecureBERT    (125M) — cybersecurity domain
MODEL_CHOICE = 'securebert'   # ← change this to 'roberta' or 'securebert'

# Model registry — HuggingFace ID + correct LoRA attention layer names
MODEL_REGISTRY = {
    'distilbert': {
        'hf_id'          : 'distilbert-base-uncased',
        'lora_modules'   : ['q_lin', 'k_lin', 'v_lin', 'out_lin'],
        'short_name'     : 'DistilBERT-67M',
    },
    'roberta': {
        'hf_id'          : 'roberta-base',
        'lora_modules'   : ['query', 'key', 'value', 'dense'],
        'short_name'     : 'RoBERTa-125M',
    },
    'securebert': {
        'hf_id'          : 'ehsanaghaei/SecureBERT',
        'lora_modules'   : ['query', 'key', 'value', 'dense'],
        'short_name'     : 'SecureBERT-125M',
    },
}

assert MODEL_CHOICE in MODEL_REGISTRY, \
    f"Unknown model: {MODEL_CHOICE}. Choose from {list(MODEL_REGISTRY.keys())}"

BASE_MODEL    = MODEL_REGISTRY[MODEL_CHOICE]['hf_id']
LORA_MODULES  = MODEL_REGISTRY[MODEL_CHOICE]['lora_modules']
MODEL_SHORT   = MODEL_REGISTRY[MODEL_CHOICE]['short_name']

# ── Other hyper-parameters ────────────────────────────────────────────────
MAX_LENGTH   = cfg['encoder_slm']['max_length']        # 128
BATCH_SIZE   = cfg['encoder_slm']['batch_size']        # 32
LORA_R       = cfg['encoder_slm']['lora_r']            # 8
LORA_ALPHA   = cfg['encoder_slm']['lora_alpha']        # 16

# Training hyper-parameters — fixed at 5 epochs for fair model comparison
N_EPOCHS      = 5
LEARNING_RATE = 2e-4
WARMUP_RATIO  = 0.1
WEIGHT_DECAY  = 0.01
VAL_FRAC      = 0.1

# Output folder per model — keeps checkpoints separate
MODEL_DIR = os.path.join(ROOT, 'models', 'encoder_slm', MODEL_CHOICE)
os.makedirs(MODEL_DIR, exist_ok=True)

set_seed(SEED)
device = get_device()

print("\n" + "═"*65)
print(f"  Encoder SLM — Training ({MODEL_SHORT} + LoRA)")
print("═"*65)
print(f"  Base model : {BASE_MODEL}")
print(f"  Device     : {device}")
print(f"  Seed       : {SEED}")
print(f"  Max tokens : {MAX_LENGTH}")
print(f"  Batch size : {BATCH_SIZE}")
print(f"  LoRA r     : {LORA_R},  alpha={LORA_ALPHA}")
print(f"  Epochs     : {N_EPOCHS}")
print(f"  LR         : {LEARNING_RATE}")

# ══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
print("\n[1/7] Loading preprocessed data ...")

# Use X_train.npy — pre-SMOTE, original class distribution
X_train_full = np.load(os.path.join(PREP_DIR, 'X_train.npy'))
y_train_full = np.load(os.path.join(PREP_DIR, 'y_train.npy'))
X_test       = np.load(os.path.join(PREP_DIR, 'X_test.npy'))
y_test       = np.load(os.path.join(PREP_DIR, 'y_test.npy'))

print(f"  Train (pre-SMOTE, original dist) : {X_train_full.shape}")
print(f"  Test  (held-out)                 : {X_test.shape}")

# Class distribution
print(f"\n  Training class distribution:")
for cls_id in sorted(np.unique(y_train_full)):
    n = (y_train_full == cls_id).sum()
    print(f"    {cls_id} ({LABEL_MAP[cls_id]:<12}): {n:>5}  ({n/len(y_train_full)*100:.1f}%)")

# Validation split — stratified
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=VAL_FRAC,
    random_state=SEED,
    stratify=y_train_full
)
print(f"\n  Train split : {X_train.shape[0]:,}  |  Val split : {X_val.shape[0]:,}")

# ══════════════════════════════════════════════════════════════════════════
# VERBALIZE
# ══════════════════════════════════════════════════════════════════════════
print(f"\n[2/7] Verbalizing features ...")
verbalizer = Verbalizer()
t0 = time.time()
train_texts = verbalizer.transform(X_train)
val_texts   = verbalizer.transform(X_val)
test_texts  = verbalizer.transform(X_test)
print(f"  Done in {time.time()-t0:.1f}s")
print(f"  Example: {train_texts[0][:120]}...")

# ══════════════════════════════════════════════════════════════════════════
# CLASS WEIGHTS
# ══════════════════════════════════════════════════════════════════════════
print(f"\n[3/7] Computing class weights ...")
with open(os.path.join(PREP_DIR, 'class_weights.pkl'), 'rb') as f:
    weight_dict_raw = pickle.load(f)

# class_weights.pkl stores a nested dict — extract the class weights sub-dict
weight_dict = weight_dict_raw['class_weights']

class_weights_tensor = torch.tensor(
    [weight_dict[i] for i in range(N_CLASSES)],
    dtype=torch.float32
).to(device)

print(f"  Class weights (higher = rarer class penalised more):")
for i in range(N_CLASSES):
    print(f"    {i} ({LABEL_MAP[i]:<12}): {weight_dict[i]:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════
class FlowDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# ══════════════════════════════════════════════════════════════════════════
# TOKENIZER & MODEL
# ══════════════════════════════════════════════════════════════════════════
print(f"\n[4/7] Loading tokenizer and model ...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Build datasets
train_dataset = FlowDataset(train_texts, y_train, tokenizer, MAX_LENGTH)
val_dataset   = FlowDataset(val_texts,   y_val,   tokenizer, MAX_LENGTH)
test_dataset  = FlowDataset(test_texts,  y_test,  tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=False)

# Base model with classification head
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=N_CLASSES,
    ignore_mismatched_sizes=True,
)

# LoRA config — target the attention projection layers
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.1,
    # DistilBERT attention: q_lin, k_lin, v_lin, out_lin
    target_modules=LORA_MODULES,
    bias='none',
)
model = get_peft_model(base_model, lora_config)
model.to(device)

# Print trainable parameters
trainable, total = model.get_nb_trainable_parameters()
print(f"  Trainable params : {trainable:,}  ({100*trainable/total:.2f}% of {total:,})")

# ══════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════
print(f"\n[5/7] Training ...")

# Loss with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# Optimizer — only update LoRA params
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

total_steps  = len(train_loader) * N_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler    = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

mlflow.set_experiment(cfg['mlflow']['experiment_slm'])

history = {'train_loss': [], 'val_loss': [], 'val_f1_macro': [], 'val_f1_slowloris': []}
best_val_f1  = 0.0
best_epoch   = 0
train_start  = time.time()

with mlflow.start_run(run_name=f'encoder-slm-{MODEL_CHOICE}-v2-5ep') as run:

    # Log hyper-parameters
    mlflow.log_params({
        'base_model':    BASE_MODEL,
        'max_length':    MAX_LENGTH,
        'batch_size':    BATCH_SIZE,
        'lora_r':        LORA_R,
        'lora_alpha':    LORA_ALPHA,
        'n_epochs':      N_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'warmup_ratio':  WARMUP_RATIO,
        'weight_decay':  WEIGHT_DECAY,
        'train_rows':    X_train.shape[0],
        'val_rows':      X_val.shape[0],
        'test_rows':     X_test.shape[0],
        'trainable_params': trainable,
        'smote_applied': False,
        'class_weighted_loss': True,
    })
    mlflow.set_tag('dataset', 'CIC-IIoT-2025')
    mlflow.set_tag('architecture', 'DistilBERT-LoRA')
    mlflow.set_tag('preprocessing', 'v2-8class-kv-flags')
    mlflow.set_tag('verbalization', 'key-value+domain-flags')
    mlflow.log_param('n_classes', N_CLASSES)
    mlflow.log_param('input_format', 'key_value_with_flags')

    # ── Experiment description — update this for every new run ────────────
    # This appears in the MLflow UI so you can identify what changed
    mlflow.set_tag('model', MODEL_SHORT)
    mlflow.set_tag('description',
        f"Model: {MODEL_SHORT}. "
        "Preprocessing v2: 8 classes (MitM+malware separated), "
        "34 selected features (XGBoost+MI), log1p+RobustScaler. "
        "Verbalization: key-value format with 6 Boolean domain flags "
        "(HIGH_PSH 75%, IP_FLAGS_2 70%, SLOW_INTERVAL 78%, "
        "HIGH_VOLUME 75%, LOW_PAYLOAD 100%, HIGH_SYN 63%). "
        f"Training: {N_EPOCHS} epochs, LR={LEARNING_RATE}, "
        f"LoRA r={LORA_R}/alpha={LORA_ALPHA}, class-weighted loss."
    )
    mlflow.set_tag('what_changed',
        f"Model swapped to {MODEL_SHORT}. "
        "Same preprocessing v2, same 5 epochs, same LR — fair comparison."
    )

    for epoch in range(1, N_EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = criterion(outputs.logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validate ───────────────────────────────────────────────────
        model.eval()
        val_loss  = 0.0
        val_preds = []
        val_true  = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels         = batch['labels'].to(device)

                outputs  = model(input_ids=input_ids, attention_mask=attention_mask)
                loss     = criterion(outputs.logits, labels)
                val_loss += loss.item()

                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(labels.cpu().numpy())

        val_loss   /= len(val_loader)
        val_f1_mac  = f1_score(val_true, val_preds, average='macro', zero_division=0)
        val_f1_sl   = f1_score(val_true, val_preds, average=None,
                               labels=list(range(N_CLASSES)), zero_division=0)[1]

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1_macro'].append(val_f1_mac)
        history['val_f1_slowloris'].append(val_f1_sl)

        print(f"  Epoch {epoch}/{N_EPOCHS}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_f1_macro={val_f1_mac:.4f}  val_f1_slowloris={val_f1_sl:.4f}")

        mlflow.log_metrics({
            'train_loss':     round(train_loss, 4),
            'val_loss':       round(val_loss,   4),
            'val_f1_macro':   round(val_f1_mac, 4),
            'val_f1_slowloris': round(val_f1_sl, 4),
        }, step=epoch)

        # Save best model checkpoint
        if val_f1_mac > best_val_f1:
            best_val_f1 = val_f1_mac
            best_epoch  = epoch
            best_dir    = os.path.join(MODEL_DIR, 'best_model')
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"    ✓ New best model saved (val_f1_macro={best_val_f1:.4f})")

    train_time = time.time() - train_start
    print(f"\n  Training complete in {train_time:.1f}s  |  Best epoch: {best_epoch}")

    # ── Final test evaluation ──────────────────────────────────────────
    print(f"\n[6/7] Evaluating best model on test set ...")

    # Reload best checkpoint
    best_base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=N_CLASSES, ignore_mismatched_sizes=True
    )
    best_model = PeftModel.from_pretrained(best_base, os.path.join(MODEL_DIR, 'best_model'))
    best_model.to(device)
    best_model.eval()

    test_preds = []
    test_probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs  = best_model(input_ids=input_ids, attention_mask=attention_mask)
            logits   = outputs.logits
            preds    = logits.argmax(dim=-1).cpu().numpy()
            probs    = torch.softmax(logits, dim=-1).cpu().numpy()
            test_preds.extend(preds)
            test_probs.extend(probs)

    test_preds = np.array(test_preds)
    test_probs = np.array(test_probs)

    f1_macro    = f1_score(y_test, test_preds, average='macro',    zero_division=0)
    f1_weighted = f1_score(y_test, test_preds, average='weighted', zero_division=0)
    f1_per_cls  = f1_score(y_test, test_preds, average=None,
                           labels=list(range(N_CLASSES)), zero_division=0)
    f1_slowloris = f1_per_cls[1]
    accuracy     = (test_preds == y_test).mean()

    target_names = [LABEL_MAP[i] for i in range(N_CLASSES)]
    report_str   = classification_report(
        y_test, test_preds,
        target_names=target_names,
        zero_division=0
    )

    print(f"\n  ── Test Results ──────────────────────────────────")
    print(f"  Accuracy         : {accuracy:.4f}")
    print(f"  F1 macro         : {f1_macro:.4f}")
    print(f"  F1 weighted      : {f1_weighted:.4f}")
    print(f"  F1 Slowloris     : {f1_slowloris:.4f}  ← key metric")
    print(f"  Train time       : {train_time:.1f}s")
    print(f"  Best epoch       : {best_epoch}")
    print(f"\n{report_str}")

    # Log final metrics
    final_metrics = {
        'test_accuracy':     round(accuracy,     4),
        'test_f1_macro':     round(f1_macro,      4),
        'test_f1_weighted':  round(f1_weighted,   4),
        'test_f1_slowloris': round(f1_slowloris,  4),
        'train_time_s':      round(train_time,    2),
        'best_epoch':        best_epoch,
    }
    for i, name in LABEL_MAP.items():
        final_metrics[f'test_f1_{name}'] = round(float(f1_per_cls[i]), 4)
    mlflow.log_metrics(final_metrics)

    # ── Plots ──────────────────────────────────────────────────────────
    # Training curves
    epochs = list(range(1, N_EPOCHS + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train loss')
    axes[0].plot(epochs, history['val_loss'],   'r-o', label='Val loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss'); axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['val_f1_macro'],     'g-o', label='Val F1 macro')
    axes[1].plot(epochs, history['val_f1_slowloris'], 'm-o', label='Val F1 Slowloris')
    axes[1].axhline(y=best_val_f1, color='gray', linestyle='--', alpha=0.5,
                    label=f'Best val F1 macro={best_val_f1:.3f}')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Validation F1 Scores'); axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f'Encoder SLM ({MODEL_SHORT} + LoRA) — '
        f'Test F1 macro={f1_macro:.3f}  F1 Slowloris={f1_slowloris:.3f}',
        fontsize=13
    )
    plt.tight_layout()
    curves_path = os.path.join(FIG_DIR, f'slm_{MODEL_CHOICE}_training_curves.png')
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(curves_path)

    # Confusion matrix
    cm   = confusion_matrix(y_test, test_preds)
    fig2, ax2 = plt.subplots(figsize=(9, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax2, cmap='Blues', colorbar=False)
    ax2.set_title(
        f'{MODEL_SHORT} Confusion Matrix  '
        f'(F1 macro={f1_macro:.3f}  F1 Slowloris={f1_slowloris:.3f})',
        fontsize=12
    )
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    cm_path = os.path.join(FIG_DIR, f'slm_{MODEL_CHOICE}_confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(cm_path)

    # ── Save report ────────────────────────────────────────────────────
    print(f"\n[7/7] Saving report ...")
    report_path = os.path.join(MODEL_DIR, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Encoder SLM ({MODEL_SHORT} + LoRA) — Classification Report\n")
        f.write("="*55 + "\n\n")
        f.write(f"Base model       : {BASE_MODEL}\n")
        f.write(f"LoRA r / alpha   : {LORA_R} / {LORA_ALPHA}\n")
        f.write(f"Epochs           : {N_EPOCHS}  (best: {best_epoch})\n")
        f.write(f"Trainable params : {trainable:,} ({100*trainable/total:.2f}%)\n\n")
        f.write(f"Accuracy         : {accuracy:.4f}\n")
        f.write(f"F1 macro         : {f1_macro:.4f}\n")
        f.write(f"F1 weighted      : {f1_weighted:.4f}\n")
        f.write(f"F1 Slowloris     : {f1_slowloris:.4f}\n")
        f.write(f"Train time       : {train_time:.1f}s\n\n")
        f.write(report_str)
    mlflow.log_artifact(report_path)

    print(f"  Report saved: {report_path}")
    print(f"  MLflow Run ID: {run.info.run_id}")

# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*65}")
print(f"  TRAINING COMPLETE")
print(f"{'═'*65}")
print(f"  Accuracy         : {accuracy:.4f}")
print(f"  F1 macro         : {f1_macro:.4f}")
print(f"  F1 Slowloris     : {f1_slowloris:.4f}")
print(f"  Best epoch       : {best_epoch} / {N_EPOCHS}")
print(f"  Train time       : {train_time:.1f}s")
print(f"\n  Outputs in : models/encoder_slm/")
print(f"    ✓ best_model/  (HuggingFace LoRA checkpoint)")
print(f"    ✓ classification_report.txt")
print(f"  Plots in   : reports/figures/")
print(f"    ✓ slm_training_curves.png")
print(f"    ✓ slm_confusion_matrix.png")
print(f"\n  View in MLflow UI:")
print(f"    mlflow-ui  (then open http://127.0.0.1:5001)")
print(f"    Experiment: encoder-slm\n")
