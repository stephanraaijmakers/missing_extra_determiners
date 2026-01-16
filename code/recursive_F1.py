#!/usr/bin/env python3
"""
Recursive F1 / accuracy computation.

Modes:
  columns (default): detect ground-truth & prediction columns and count:
     TP: gt=yes & pred=yes
     FP: gt=no  & pred=yes
     FN: gt=yes & pred=no
     TN: gt=no  & pred=no
  regex: raw line patterns (exact schema requested):
     TP: ,yes,.*?,yes,
     FN: ,yes,.*?,no,
     FP: ,no,.*?,yes,
     TN: ,no,.*?,no,

Directory mode:
  --dir <folder> --pattern "*__predictions.csv" aggregates mean/std of metric (--metric f1|accuracy).

Single-file mode:
  --file <csv>

Outputs JSON with --json, else plain text.
Exit code 0 on success, 1 on error.
"""
from __future__ import annotations
import argparse, json, math, re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import List, Dict, Tuple, Optional

try:
    import pandas as pd
except ImportError:
    pd = None  # only needed for columns mode

def safe_div(n: float, d: float) -> float:
    return n / d if d else float("nan")

def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    return safe_div(2 * p * r, p + r)

@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

# ---------- Columns mode ----------
def normalize(v, case_sensitive: bool) -> Optional[str]:
    if v is None or (isinstance(v, float) and pd is not None and pd.isna(v)):
        return None
    s = str(v).strip()
    if not case_sensitive:
        s = s.lower()
    # Map A/B -> yes/no
    if s in ("a", "A"):
        return "yes"
    if s in ("b", "B"):
        return "no"
    if not case_sensitive and s in ("y", "true", "1"):
        return "yes"
    if not case_sensitive and s in ("n", "false", "0"):
        return "no"
    return s

def detect_columns(df, gt_col: Optional[str], pred_col: Optional[str]) -> Tuple[str,str]:
    if gt_col and gt_col in df.columns:
        g = gt_col
    else:
        for c in ("grammatical", "grammatical_gt"):
            if c in df.columns:
                g = c
                break
        else:
            raise ValueError("Could not detect ground-truth column")
    if pred_col and pred_col in df.columns:
        p = pred_col
    else:
        for c in ("grammatical_pred", "prediction"):
            if c in df.columns:
                p = c
                break
        else:
            raise ValueError("Could not detect prediction column")
    return g, p

def counts_from_columns(path: str, gt_col: Optional[str], pred_col: Optional[str], case_sensitive: bool) -> Counts:
    if pd is None:
        raise RuntimeError("pandas not installed; cannot use columns mode")
    df = pd.read_csv(path)
    gcol, pcol = detect_columns(df, gt_col, pred_col)
    gvals = [normalize(v, case_sensitive) for v in df[gcol].tolist()]
    pvals = [normalize(v, case_sensitive) for v in df[pcol].tolist()]
    c = Counts()
    for g, p in zip(gvals, pvals):
        if g not in ("yes", "no") or p not in ("yes", "no"):
            continue
        if g == "yes" and p == "yes":
            c.tp += 1
        elif g == "yes" and p == "no":
            c.fn += 1
        elif g == "no" and p == "yes":
            c.fp += 1
        elif g == "no" and p == "no":
            c.tn += 1
    return c

# ---------- Regex mode ----------
PAT_TP = re.compile(r",\s*yes\s*,[^,]*,\s*yes\s*(?:,|$)", re.IGNORECASE)
PAT_FN = re.compile(r",\s*yes\s*,[^,]*,\s*no\s*(?:,|$)", re.IGNORECASE)
PAT_FP = re.compile(r",\s*no\s*,[^,]*,\s*yes\s*(?:,|$)", re.IGNORECASE)
PAT_TN = re.compile(r",\s*no\s*,[^,]*,\s*no\s*(?:,|$)", re.IGNORECASE)

def counts_from_regex(path: str, case_sensitive: bool) -> Counts:
    tp = fp = fn = tn = 0
    # Adjust patterns if case-sensitive requested
    if case_sensitive:
        ptp = re.compile(PAT_TP.pattern)
        pfn = re.compile(PAT_FN.pattern)
        pfp = re.compile(PAT_FP.pattern)
        ptn = re.compile(PAT_TN.pattern)
    else:
        ptp, pfn, pfp, ptn = PAT_TP, PAT_FN, PAT_FP, PAT_TN
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            s = line.rstrip("\r\n")
            if not s or s.lower().startswith("sentence,"):
                continue
            if ptp.search(s):
                tp += 1
            elif pfn.search(s):
                fn += 1
            elif pfp.search(s):
                fp += 1
            elif ptn.search(s):
                tn += 1
            else:
                pass
    return Counts(tp=tp, fp=fp, fn=fn, tn=tn)

# ---------- Metrics / aggregation ----------
def metrics_from_counts(c: Counts) -> Dict[str, float]:
    precision = safe_div(c.tp, c.tp + c.fp)
    recall = safe_div(c.tp, c.tp + c.fn)
    f1 = f1_from_counts(c.tp, c.fp, c.fn)
    accuracy = safe_div(c.tp + c.tn, c.total())
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

def process_file(path: str, mode: str, gt_col: Optional[str], pred_col: Optional[str], case_sensitive: bool) -> Dict:
    if mode == "regex":
        c = counts_from_regex(path, case_sensitive)
    else:
        c = counts_from_columns(path, gt_col, pred_col, case_sensitive)
    m = metrics_from_counts(c)
    return {
        "path": path,
        "confusion": {
            "tp": c.tp, "fp": c.fp, "fn": c.fn, "tn": c.tn, "n_total": c.total()
        },
        "metrics": m
    }

def aggregate(results: List[Dict], metric: str) -> Dict:
    vals = [r["metrics"][metric] for r in results if not (isinstance(r["metrics"][metric], float) and math.isnan(r["metrics"][metric]))]
    return {
        "count": len(results),
        "used": len(vals),
        "metric": metric,
        "mean": mean(vals) if vals else float("nan"),
        "std": pstdev(vals) if len(vals) > 1 else (0.0 if len(vals) == 1 else float("nan")),
    }

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--dir")
    ap.add_argument("--pattern", default="*__predictions.csv")
    ap.add_argument("--mode", choices=["columns","regex"], default="regex")
    ap.add_argument("--gt-col")
    ap.add_argument("--pred-col")
    ap.add_argument("--metric", choices=["f1","accuracy"], default="f1")
    ap.add_argument("--case-sensitive", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.dir and not args.file:
        files = sorted(Path(args.dir).rglob(args.pattern))
        results = []
        for f in files:
            try:
                results.append(process_file(str(f), args.mode, args.gt_col, args.pred_col, args.case_sensitive))
            except Exception as e:
                results.append({"path": str(f), "error": str(e)})
        good = [r for r in results if "metrics" in r]
        agg = aggregate(good, args.metric) if good else {"count":0,"used":0,"metric":args.metric,"mean":float("nan"),"std":float("nan")}
        if args.json:
            print(json.dumps({"files": results, "aggregate": agg}, indent=2))
        else:
            for r in results:
                if "metrics" in r:
                    v = r["metrics"][args.metric]
                    print(f"{r['path']}: tp={r['confusion']['tp']} fp={r['confusion']['fp']} fn={r['confusion']['fn']} tn={r['confusion']['tn']} {args.metric}={v}")
                else:
                    print(f"{r['path']}: ERROR {r['error']}")
            print(f"AGG {args.metric}: mean={agg['mean']} std={agg['std']} files={agg['count']} used={agg['used']}")
        return 0

    if not args.file:
        msg = "Provide --file or --dir"
        if args.json:
            print(json.dumps({"error": msg}))
        else:
            print(f"Error: {msg}")
        return 1

    try:
        r = process_file(args.file, args.mode, args.gt_col, args.pred_col, args.case_sensitive)
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Error: {e}")
        return 1

    if args.json:
        print(json.dumps(r, indent=2))
    else:
        c = r["confusion"]; m = r["metrics"]
        print(f"{r['path']}")
        print(f"tp={c['tp']} fp={c['fp']} fn={c['fn']} tn={c['tn']} n_total={c['n_total']}")
        print(f"precision={m['precision']:.6f} recall={m['recall']:.6f} f1={m['f1']:.6f} accuracy={m['accuracy']:.6f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())