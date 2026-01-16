#!/usr/bin/env python3
import argparse
import pandas as pd
import csv
import re
import unicodedata
import sys

def clean_series(s: pd.Series) -> pd.Series:
    # Normalize unicode, collapse all whitespace (incl. newlines/tabs) to single space
    def _norm(x: str) -> str:
        t = unicodedata.normalize("NFKC", str(x))
        t = t.replace("\u00A0", " ")
        t = re.sub(r"\s+", " ", t.strip())
        # Treat common "missing" string tokens as empty
        if t.lower() in ("", "nan", "none", "nat", "null"):
            return ""
        return t
    return s.astype(str).map(_norm)

def _to_yesno(label: str) -> str:
    s = (str(label) if label is not None else "").strip().lower()
    if s in ("a", "yes"): return "yes"
    if s in ("b", "no"):  return "no"
    return s  # pass through if already yes/no

def main():
    ap = argparse.ArgumentParser(description="Produce sentence,noun_phrase,grammatical,corrected from source CSV")
    ap.add_argument("--input", "-i", required=True, help="Input CSV path")
    ap.add_argument("--output", "-o", required=True, help="Output CSV path")
    ap.add_argument("--encoding", "-e", default="auto", help="Input encoding or 'auto'")
    args = ap.parse_args()

    # Read CSV with simple encoding fallback
    read_kwargs = {}
    if args.encoding != "auto":
        read_kwargs["encoding"] = args.encoding
    try:
        df = pd.read_csv(args.input, **read_kwargs)
    except UnicodeDecodeError:
        if args.encoding == "auto":
            for enc in ("utf-8-sig", "cp1252", "latin1"):
                try:
                    df = pd.read_csv(args.input, encoding=enc)
                    print(f"Read input using encoding={enc}", file=sys.stderr)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise
        else:
            raise

    # Required columns
    for col in ["sentence", "noun_phrase"]:
        if col not in df.columns:
            raise SystemExit(f"Missing required column: {col}")

    sentence = clean_series(df["sentence"])
    noun = clean_series(df["noun_phrase"])
    gold_fix = clean_series(df["gold_fix"]) if "gold_fix" in df.columns else pd.Series([""] * len(df), index=df.index)

    has_fix = gold_fix.str.len() > 0
    # yes = grammatical (no gold_fix); no = ungrammatical (has gold_fix)
    grammatical = has_fix.replace({True: "no", False: "yes"})
    corrected = gold_fix.where(has_fix, noun)

    out = pd.DataFrame({
        "sentence": sentence,
        "noun_phrase": noun,
        "grammatical": grammatical,
        "corrected": corrected,
    })

    # Idempotent normalization (handles any stray A/B)
    out["grammatical"] = out["grammatical"].apply(_to_yesno)

    # Write with all fields quoted
    out.to_csv(
        args.output,
        index=False,
        quoting=csv.QUOTE_ALL,
        quotechar='"',
        doublequote=True,
        lineterminator="\n"
    )

if __name__ == "__main__":
    main()
