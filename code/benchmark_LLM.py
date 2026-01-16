#!/usr/bin/env python3
import os
import re
import argparse
import time
import pandas as pd
from pathlib import Path
import math
import random
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

SYSTEM_PROMPT = """You are judging noun phrase grammaticality.

Output exactly one line and end it with <EOT>.
- If the noun phrase is grammatical as written, output: yes
- If it is ungrammatical due to a determiner error, output: no <corrected noun phrase>

Rules:
- For "no", only insert or delete determiners; do not change other words.
- No other words, punctuation, quotes, or explanations."""


# Use a hard stop marker to terminate generation early
STOP_TOKEN = "<EOT>"


# Tight question template
QUESTION_TEMPLATE = (
    'Sentence: "{sentence}"\n'
    'Noun phrase: "{noun_phrase}"\n'
    "Answer (yes, or no <correction>). End with <EOT>: "
)

def levenshtein(a: str, b: str) -> int:
    a, b = a or "", b or ""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    prev = list(range(lb + 1))
    cur = [0] * (lb + 1)
    for i in range(1, la + 1):
        cur[0] = i
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[lb]

def build_prompt(tokenizer, user_text: str) -> str:
    # Prefer chat formatting for instruct-tuned models
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Fallback to plain prompt
    return f"{SYSTEM_PROMPT}\n\n{user_text}"

# Stop when STOP_TOKEN or an echo of the next prompt shows up
class StopOnTokens(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings):
        self.tokenizer = tokenizer
        # pre-tokenize stop strings
        self.stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strings]

    def __call__(self, input_ids, scores, **kwargs):
        # check only the last tokens for each stop pattern
        for ids in self.stop_ids:
            if len(ids) == 0 or input_ids.shape[1] < len(ids):
                continue
            if input_ids[0, -len(ids):].tolist() == ids:
                return True
        return False

def _is_placeholder(text: str) -> bool:
    if not isinstance(text, str):
        return True
    t = text.strip().lower()
    if not t:
        return True
    # common placeholder artifacts to drop
    return (
        t in {"<noun phrase>", "<corrected noun phrase>", "noun phrase", "corrected noun phrase"} or
        t.startswith("<") or t.endswith(">")
    )

def _clean_after(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    # Drop leading numbering and boilerplate like "A:" / "B:" / "Answer:"
    s = re.sub(r'^\s*(?:\(?\d+\)?[.)-]\s*)', '', s)
    s = re.sub(r'^\s*(?:followed by|is|should be|output(?: is)?|answer(?: is)?)\s+', '', s, flags=re.IGNORECASE)
    # Drop leading punctuation and surrounding quotes
    s = re.sub(r'^\s*[,;:–—-]\s*', '', s).strip().strip('"“”\'‘’')
    return s.strip()

def _norm_np(s: str) -> str:
    # Normalize NP for equality checks (case/whitespace/punct-insensitive)
    if not isinstance(s, str):
        return ""
    t = s.strip().lower()
    # remove leading/trailing quotes and trivial trailing punctuation
    t = t.strip('"“”\'‘’').strip()
    t = re.sub(r"[.,;:]+$", "", t)
    # collapse whitespace
    t = re.sub(r"\s+", " ", t)
    return t

def parse_reply(raw: str):
    # Keep parsing short and resilient
    if not raw:
        return "", "", ""
    # Cut at stop marker if present
    raw = raw.split(STOP_TOKEN, 1)[0]
    # split into non-empty trimmed lines
    lines = [l.strip() for l in str(raw).splitlines() if l and l.strip()]
    # Remove obvious echoes
    lines = [
        l for l in lines
        if not l.lower().startswith("sentence:")
        and not l.lower().startswith("noun phrase:")
        and not l.lower().startswith("respond")
        and not l.lower().startswith("examples:")
    ]
    for l in lines:
        # Drop "Answer:" prefix if present and numeric bullets
        l = re.sub(r"^\s*\**\s*answer\s*[:\-]*\s*", "", l, flags=re.IGNORECASE)
        l = re.sub(r'^\s*(?:\(?\d+\)?[.)-]\s*)', '', l)
        # yes
        m = re.match(r"^\s*yes\s*[:\-]?\s*(.*)$", l, flags=re.IGNORECASE)
        if m:
            after = _clean_after(m.group(1))
            return "yes", after, after.casefold().strip()
        # no <fix>
        m = re.match(r"^\s*no\s*[:\-]?\s*(.*)$", l, flags=re.IGNORECASE)
        if m:
            after = _clean_after(m.group(1))
            return "no", after, after.casefold().strip()
        # FALLBACK: accept A/B and map to yes/no
        m = re.match(r"^\s*A\s*[:\-]?\s*(.*)$", l, flags=re.IGNORECASE)
        if m:
            after = _clean_after(m.group(1))
            return "yes", after, after.casefold().strip()
        m = re.match(r"^\s*B\s*[:\-]?\s*(.*)$", l, flags=re.IGNORECASE)
        if m:
            after = _clean_after(m.group(1))
            return "no", after, after.casefold().strip()
    return "", "", ""

def set_seed(seed: int | None, device: str | None = None):
    if seed is None:
        return None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Build a per-device generator for HF generate()
    gen = torch.Generator(device=device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    gen.manual_seed(seed)
    return gen

def build_gen_kwargs(args):
    gen_kwargs = {}
    # Enable sampling if requested or if temperature/top_p imply sampling
    if getattr(args, "do_sample", False) or (getattr(args, "temperature", 0.0) and args.temperature > 0.0) or (hasattr(args, "top_p") and args.top_p < 1.0):
        gen_kwargs["do_sample"] = True
        if hasattr(args, "temperature") and args.temperature is not None:
            gen_kwargs["temperature"] = float(args.temperature)
        if hasattr(args, "top_p") and args.top_p is not None:
            gen_kwargs["top_p"] = float(args.top_p)
    return gen_kwargs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.getenv("MODEL_ID"), help="HF model id (e.g., meta-llama/Llama-3.2-1B-Instruct)")
    ap.add_argument("--input", default="formatted.csv", help="Path to formatted.csv")
    ap.add_argument("--max-new-tokens", type=int, default=int(os.getenv("MAX_TOKENS", "50")))
    # Add if missing:
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (>0 enables sampling)")
    ap.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling p")
    ap.add_argument("--do-sample", dest="do_sample", action="store_true", help="Force sampling")
    ap.add_argument("--seed", type=int, help="Random seed")
    ap.add_argument("--token", default=os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN"))
    # IMPORTANT: if you used parse_args() before, keep it; otherwise ignore unknowns safely:
    args = ap.parse_args()  # or: args = ap.parse_known_args()[0]

    if not args.model:
        raise SystemExit("Please provide --model or set MODEL_ID")

    # Load data
    df = pd.read_csv(args.input)
    for col in ("sentence", "noun_phrase", "grammatical", "corrected"):
        if col not in df.columns:
            raise SystemExit(f"Input missing required column: {col}")
    df = df.copy()

    # Load model locally
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)
    tok_kwargs = {"use_fast": True, "trust_remote_code": True}
    mdl_kwargs = {"torch_dtype": dtype, "device_map": "auto", "low_cpu_mem_usage": True, "trust_remote_code": True}
    if args.token:
        tok_kwargs["token"] = args.token
        mdl_kwargs["token"] = args.token

    # No stdout prints
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_kwargs)
    model = AutoModelForCausalLM.from_pretrained(args.model, **mdl_kwargs)
    model.eval()

    # Ensure pad/eos tokens for generate()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Build stopping criteria: stop only on our marker
    stop_criteria = StopOnTokens(
        tokenizer,
        stop_strings=[STOP_TOKEN]
    )

    # Derive device string if you use it
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = set_seed(getattr(args, "seed", None), device=device)
    gen_kwargs = build_gen_kwargs(args)

    # Run
    y_true = []        # 'yes' or 'no' (ground truth)
    y_pred = []        # 'yes' or 'no' (model)
    fixes_pred = []    # model fixes (only for 'no' predictions)
    # Per-row output for file
    pred_correction = []  # correction text for 'no', else original NP
    start = time.time()

    for _, row in df.iterrows():
        q = QUESTION_TEMPLATE.format(sentence=row["sentence"], noun_phrase=row["noun_phrase"])
        prompt = build_prompt(tokenizer, q)

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = model.generate(**inputs, **gen_kwargs)
        # Decode only newly generated tokens
        gen_only = out_ids[0, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_only, skip_special_tokens=True).strip()

        # Make parsing tolerant: append marker if missing (but parsing won’t depend on it)
        if STOP_TOKEN not in text:
            text = text + STOP_TOKEN

        yn, fix_raw, _ = parse_reply(text)
        if yn not in ("yes", "no"):
            # Fallback 1: regex scan anywhere in the output
            m = re.search(r"\b(yes|no)\b[:\-]?\s*(.*)", text, flags=re.IGNORECASE)
            if m:
                yn = m.group(1).lower()
                fix_raw = (m.group(2) or "").splitlines()[0].strip()

        # Final fallback: default to yes if still undecided
        if yn not in ("yes", "no"):
            yn, fix_raw = "yes", ""

        # Normalize fix (drop placeholders)
        np_or_fix = _clean_after((fix_raw or ""))
        if _is_placeholder(np_or_fix) and yn == "no":
            np_or_fix = ""  # model failed to propose a fix

        # If model says "no" but proposes no effective change (same NP), treat as "yes"
        if yn == "no":
            np_norm = _norm_np(row["noun_phrase"])
            fix_norm = _norm_np(np_or_fix)
            if not fix_norm or fix_norm == np_norm:
                yn = "yes"
                np_or_fix = ""

        # Ground truth to 'yes'/'no'
        def to_yesno(v: str) -> str:
            v = (str(v) if pd.notna(v) else "").strip().lower()
            if v in ("a", "yes", "y", "true", "1"): return "yes"
            if v in ("b", "no", "n", "false", "0"): return "no"
            return ""
        y_true.append(to_yesno(row["grammatical"]))
        y_pred.append(yn)  # never empty now
        fixes_pred.append(np_or_fix if yn == "no" else "")

        # Always populate 'correction'
        pred_fix = np_or_fix if yn == "no" else str(row["noun_phrase"])
        pred_correction.append(pred_fix)

    elapsed = time.time() - start

    # Metrics for class 'no' (ungrammatical)
    tp_no = sum(1 for t, p in zip(y_true, y_pred) if t == "no" and p == "no")
    fp_no = sum(1 for t, p in zip(y_true, y_pred) if t != "no" and p == "no")
    fn_no = sum(1 for t, p in zip(y_true, y_pred) if t == "no" and p != "no")
    precision_no = tp_no / (tp_no + fp_no) if (tp_no + fp_no) else float("nan")
    recall_no = tp_no / (tp_no + fn_no) if (tp_no + fn_no) else float("nan")
    f1_no = (2 * precision_no * recall_no) / (precision_no + recall_no) if (precision_no + recall_no) else float("nan")

    # Metrics for class 'yes' (grammatical)
    tp_yes = sum(1 for t, p in zip(y_true, y_pred) if t == "yes" and p == "yes")
    fp_yes = sum(1 for t, p in zip(y_true, y_pred) if t != "yes" and p == "yes")
    fn_yes = sum(1 for t, p in zip(y_true, y_pred) if t == "yes" and p != "yes")
    precision_yes = tp_yes / (tp_yes + fp_yes) if (tp_yes + fp_yes) else float("nan")
    recall_yes = tp_yes / (tp_yes + fn_yes) if (tp_yes + fn_yes) else float("nan")
    f1_yes = (2 * precision_yes * recall_yes) / (precision_yes + recall_yes) if (precision_yes + recall_yes) else float("nan")

    # Macro-F1 over classes yes and no
    macro_f1 = (
        (f1_yes + f1_no) / 2.0
        if (not math.isnan(f1_yes) and not math.isnan(f1_no))
        else (f1_yes if not math.isnan(f1_yes) else (f1_no if not math.isnan(f1_no) else float("nan")))
    )

    # Overall accuracy (label match)
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t and p and (t == p))
    accuracy = (correct / total) if total else float("nan")
    n_gold_yes = sum(1 for t in y_true if t == "yes")
    n_gold_no = sum(1 for t in y_true if t == "no")

    # Edit distance for correctly targeted 'no' cases (gold no and pred no)
    dists = []
    for (t, p, fix_pred, corr) in zip(y_true, y_pred, fixes_pred, df["corrected"].astype(str).tolist()):
        if t == "no" and p == "no":
            dists.append(levenshtein((corr or "").strip(), (fix_pred or "").strip()))
    mean_edit = (sum(dists) / len(dists)) if dists else float("nan")

    # Build output file paths
    model_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(args.model or "model"))
    in_name = Path(args.input).name
    out_csv_path = f"{model_safe}__{in_name}__predictions.csv"
    out_metrics_txt_path = f"{model_safe}__{in_name}__metrics.txt"
    out_metrics_csv_path = f"{model_safe}__{in_name}__metrics.csv"

    # ---- Build requested output columns ----
    grammatical_pred = y_pred[:]        # yes/no (never empty)
    correction = pred_correction[:]

    out_df = df.copy()
    # Normalize GT to yes/no in case source has A/B
    def _to_yesno(v: str) -> str:
        v = (str(v) if pd.notna(v) else "").strip().lower()
        if v in ("a", "yes", "y", "true", "1"): return "yes"
        if v in ("b", "no", "n", "false", "0"): return "no"
        return v
    out_df["grammatical"] = out_df["grammatical"].map(_to_yesno)
    out_df["grammatical_pred"] = grammatical_pred
    out_df["correction"] = correction
    out_df = out_df[["sentence", "noun_phrase", "grammatical", "corrected", "grammatical_pred", "correction"]]
    out_df.to_csv(out_csv_path, index=False, encoding="utf-8")

    # Write metrics (yes/no)
    metrics_row = {
        "model": args.model,
        "input": args.input,
        "elapsed_sec": round(elapsed, 3),
        "n_total": total,
        "n_gold_yes": n_gold_yes,
        "n_gold_no": n_gold_no,
        "tp_yes": tp_yes, "fp_yes": fp_yes, "fn_yes": fn_yes,
        "tp_no": tp_no, "fp_no": fp_no, "fn_no": fn_no,
        "accuracy": accuracy,
        "precision_yes": precision_yes,
        "recall_yes": recall_yes,
        "f1_yes": f1_yes,
        "precision_no": precision_no,
        "recall_no": recall_no,
        "f1_no": f1_no,
        "macro_f1": macro_f1,
        "mean_edit_distance_TP_no": (mean_edit if dists else float("nan")),
    }
    pd.DataFrame([metrics_row]).to_csv(out_metrics_csv_path, index=False, encoding="utf-8")

    with open(out_metrics_txt_path, "w", encoding="utf-8") as mf:
        mf.write(f"model={args.model}\n")
        mf.write(f"input={args.input}\n")
        mf.write(f"elapsed_sec={elapsed:.1f}\n")
        mf.write(f"n_total={total}\n")
        mf.write(f"n_gold_yes={n_gold_yes}\n")
        mf.write(f"n_gold_no={n_gold_no}\n")
        mf.write(f"tp_yes={tp_yes}\n")
        mf.write(f"fp_yes={fp_yes}\n")
        mf.write(f"fn_yes={fn_yes}\n")
        mf.write(f"tp_no={tp_no}\n")
        mf.write(f"fp_no={fp_no}\n")
        mf.write(f"fn_no={fn_no}\n")
        mf.write(f"accuracy={accuracy:.6f}\n")
        mf.write(f"precision_yes={precision_yes:.6f}\n")
        mf.write(f"recall_yes={recall_yes:.6f}\n")
        mf.write(f"f1_yes={f1_yes:.6f}\n")
        mf.write(f"precision_no={precision_no:.6f}\n")
        mf.write(f"recall_no={recall_no:.6f}\n")
        mf.write(f"f1_no={f1_no:.6f}\n")
        mf.write(f"macro_f1={macro_f1:.6f}\n")
        mf.write(f"mean_edit_distance_TP_no={mean_edit if dists else float('nan')}\n")

if __name__ == "__main__":
    main()