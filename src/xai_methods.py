
import numpy as np

SEP = " | "
JOIN = " is "



def split_segments(text: str) -> list[tuple[str, str]]:
    """'client max packet size is 234 | ...' -> [('client max packet size', '234'), ...]"""
    out = []
    for seg in text.split("|"):
        seg = seg.strip()
        if JOIN not in seg:
            continue
        label, value = seg.split(JOIN, 1)
        out.append((label.strip(), value.strip()))
    return out


def rebuild(segments: list[tuple[str, str]]) -> str:
    return SEP.join(f"{lab}{JOIN}{val}" for lab, val in segments)


def feature_labels(text: str) -> list[str]:
    return [lab for lab, _ in split_segments(text)]


def build_baseline(texts, labels=None) -> dict[str, str]:
    """Per feature, the OBSERVED value string closest to the benign median.

    Returning an observed string rather than a computed one guarantees the
    substituted value is both correctly formatted and in-distribution for the
    tokeniser. Pass benign training flows only.
    """
    if labels is None:
        labels = feature_labels(texts[0])

    collected = {lab: [] for lab in labels}
    for t in texts:
        for lab, val in split_segments(t):
            if lab in collected:
                collected[lab].append(val)

    baseline = {}
    for lab, vals in collected.items():
        nums = np.array([float(v.rstrip("s")) for v in vals])
        median = float(np.median(nums))
        baseline[lab] = vals[int(np.argmin(np.abs(nums - median)))]
    return baseline


def tokens_to_features(text: str, tokenizer, token_attr, max_length: int):
    enc = tokenizer(text, return_offsets_mapping=True, truncation=True,
                    max_length=max_length)
    offsets = enc["offset_mapping"]

    spans, pos = {}, 0
    for seg in text.split("|"):
        if JOIN not in seg:
            continue
        start = text.index(seg, pos)
        end = start + len(seg)
        pos = end
        spans[seg.split(JOIN, 1)[0].strip()] = (start, end)

    # token_attr is padded to max_length; offsets is not. They align from the
    # front (same tokenisation, same [CLS] at index 0), so positions at and
    # beyond len(offsets) are padding and stay unassigned.
    assigned = np.zeros(len(token_attr), dtype=bool)
    out = {}
    for lab, (s, e) in spans.items():
        idx = [i for i, (a, b) in enumerate(offsets)
               if b > a and a >= s and b <= e]
        # b > a excludes special and padding tokens, which carry a
        # zero-length (0, 0) offset because they map to no text span.
        assigned[idx] = True
        out[lab] = float(token_attr[idx].sum()) if idx else 0.0
    return out, assigned


def integrated_gradients(model, tokenizer, text: str, baseline_text: str,
                         steps: int = 50, max_length: int = 128,
                         chunk: int = 25):
    import torch

    device = next(model.parameters()).device
    enc = tokenizer(text, padding="max_length", truncation=True,
                    max_length=max_length, return_tensors="pt").to(device)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    emb = model.get_input_embeddings()
    x = emb(ids)                                        # (1, L, H)

    b = tokenizer(baseline_text, padding="max_length", truncation=True,
                  max_length=max_length, return_tensors="pt").to(device)
    x0 = emb(b["input_ids"])
    n_tok_base = int(b["attention_mask"].sum().item())

    def logodds(e, m):
        lg = model(inputs_embeds=e, attention_mask=m).logits
        return lg[:, 1] - lg[:, 0]

    alphas = (torch.arange(steps, device=device, dtype=x.dtype) + 0.5) / steps
    grads = torch.zeros_like(x)
    for lo in range(0, steps, chunk):
        a = alphas[lo:lo + chunk].view(-1, 1, 1)
        path = (x0 + a * (x - x0)).detach().requires_grad_(True)
        logodds(path, mask.expand(path.shape[0], -1)).sum().backward()
        grads = grads + path.grad.sum(dim=0, keepdim=True)
    grads = grads / steps

    token_attr = ((x - x0) * grads).sum(-1).squeeze(0).detach().cpu().numpy()

    with torch.no_grad():
        fx = logodds(x, mask).item()
        f0 = logodds(x0, mask).item()

    scores, assigned_mask = tokens_to_features(text, tokenizer, token_attr,
                                               max_length)

    abs_attr = np.abs(token_attr)
    abs_total = float(abs_attr.sum()) or 1.0
    abs_unassigned = float(abs_attr[~assigned_mask].sum())

    total = float(token_attr.sum())
    assigned = float(sum(scores.values()))
    diag = {
        "completeness_error": float(abs(total - (fx - f0))),
        "assigned_attr": assigned,
        "unassigned_attr": total - assigned,       
        "unassigned_frac": abs_unassigned / abs_total,
        "abs_attr_total": abs_total,
        "n_tokens_assigned": int(assigned_mask.sum()),
        "fx": fx, "f0": f0, "logodds_delta": fx - f0,
        "n_tokens_flow": int(mask.sum().item()),
        "n_tokens_baseline": n_tok_base,
    }
    return scores, diag
