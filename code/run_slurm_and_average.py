#!/usr/bin/env python3
"""
Run all SLURM*.sh scripts N times, compute metrics with recursive_F1.py after each run,
and aggregate (average) results per script.

Features:
- Auto-discovers scripts matching SLURM*.sh in the working directory
- Supports two execution modes:
  * sbatch (default if available)
  * bash (fallback, run the script directly)
- Parses the target predictions CSV from each SLURM script (model + input)
- After each run, calls recursive_F1.py --file --mode regex --json and stores per-run metrics
- Writes per-script JSON/CSV summaries with mean/std and sums

Usage examples:
  python run_slurm_and_average.py                     # auto-discover 4 scripts, 10 runs, sbatch if present
  python run_slurm_and_average.py --runs 5 --mode bash
  python run_slurm_and_average.py --pattern 'SLURM_DETERMINERS_*.sh' --timeout-sec 43200

Outputs:
- <script>.10runs.metrics.json  : Aggregated metrics (mean/std) + confusion (sum/mean)
- <script>.10runs.metrics.csv   : Per-run metrics table + final mean/std row
- summary_10runs.json           : One aggregated entry per script

Notes:
- This script waits for sbatch jobs by polling `squeue -j <jobid>` until the job finishes.
- If you don't have Slurm, use --mode bash to run the scripts directly.
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from shutil import copy2


HERE = Path(__file__).resolve().parent
# Use recursive_F1.py instead of compute_F1.py
METRICS_SCRIPT = HERE / "recursive_F1.py"


@dataclass
class RunResult:
    confusion: Dict[str, float]
    metrics: Dict[str, float]


def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def parse_benchmark_args_from_slurm(script_path: Path) -> tuple[str, str]:
    """
    Extract --model and --input from the benchmark_LLM.py invocation inside the SLURM script.
    Robust to line continuations (\\) and quoted or =-style args.
    Returns (model, input_basename).
    """
    txt = script_path.read_text(encoding="utf-8", errors="ignore")

    # Join lines that end with a backslash into one logical line
    merged_lines = []
    buf = ""
    for line in txt.splitlines():
        if line.rstrip().endswith("\\"):
            buf += line.rstrip()[:-1] + " "
        else:
            buf += line
            merged_lines.append(buf)
            buf = ""
    if buf:
        merged_lines.append(buf)
    merged = "\n".join(merged_lines)

    # Try to isolate the python ... benchmark_LLM.py command; fall back to full text if not found
    m_cmd = re.search(r"(?:^|\n)\s*(?:python3?|/usr/bin/python[0-9.]*)\s+[^\\\n]*benchmark_LLM\.py[^\n]*", merged)
    cmd_str = m_cmd.group(0) if m_cmd else merged

    def find_arg(name: str) -> str | None:
        # Supports: --arg value, --arg="value with spaces", --arg='value', --arg=value
        pat = re.compile(rf"--{name}(?:\s+|=)(?P<val>(\"[^\"]*\"|'[^']*'|[^\s\"']+))")
        m = pat.search(cmd_str)
        if not m:
            return None
        v = m.group("val")
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        return v

    model = find_arg("model")
    input_arg = find_arg("input")

    if not model:
        raise RuntimeError(f"--model not found in {script_path}")
    if not input_arg:
        raise RuntimeError(f"--input not found in {script_path}")

    return model, Path(input_arg).name


def model_safe_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def predictions_csv_for(model: str, input_basename: str) -> Path:
    return HERE / f"{model_safe_name(model)}__{input_basename}__predictions.csv"


def run_command(cmd: List[str], cwd: Optional[Path] = None, capture: bool = True, check: bool = True, env: Optional[dict] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
        check=check,
        env=env if env is not None else os.environ.copy(),
    )


def submit_sbatch(script: Path, exports: Dict[str, str]) -> int:
    """
    Submit sbatch with explicit environment export.
    """
    export_str = ",".join(f"{k}={v}" for k, v in exports.items()) or "NONE"
    res = run_command(["sbatch", f"--export={export_str}", str(script)])
    m = re.search(r"Submitted batch job\s+(\d+)", res.stdout or "")
    if not m:
        raise RuntimeError(f"Could not parse job id from sbatch output: {res.stdout!r} {res.stderr!r}")
    return int(m.group(1))


def wait_for_job(job_id: int, poll_sec: float = 10.0, timeout_sec: float = 24*3600) -> None:
    start = time.time()
    while True:
        if time.time() - start > timeout_sec:
            raise TimeoutError(f"Timed out waiting for job {job_id}")
        try:
            q = run_command(["squeue", "-j", str(job_id), "-h"], capture=True, check=False)
            still_there = (q.stdout or "").strip() != ""
        except Exception:
            # If squeue is unavailable intermittently, sleep and retry
            
            still_there = True
        if not still_there:
            return
        time.sleep(poll_sec)


def run_slurm_script(
    script: Path,
    mode: str,
    timeout_sec: float,
    run_index: int,
    stdout_path: Optional[Path] = None,
    stderr_path: Optional[Path] = None,
) -> Optional[int]:
    """
    Run one SLURM script; inject RUN_INDEX (and SEED) env.
    Returns job_id for sbatch mode, or None for bash mode.
    """
    exports = {"RUN_INDEX": str(run_index), "SEED": str(run_index)}
    if mode == "sbatch":
        job_id = submit_sbatch(script, exports)
        wait_for_job(job_id, timeout_sec=timeout_sec)
        return job_id
    elif mode == "bash":
        env = os.environ.copy()
        env.update(exports)
        stdout_f = open(stdout_path, "w", encoding="utf-8") if stdout_path else None
        stderr_f = open(stderr_path, "w", encoding="utf-8") if stderr_path else None
        try:
            proc = subprocess.Popen(
                ["bash", str(script)],
                cwd=str(HERE),
                stdout=stdout_f if stdout_f else None,
                stderr=stderr_f if stderr_f else None,
                text=True,
                env=env,
            )
            try:
                proc.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                proc.kill()
                raise
            if proc.returncode != 0:
                raise RuntimeError(f"Script {script.name} failed with exit code {proc.returncode}")
        finally:
            if stdout_f: stdout_f.close()
            if stderr_f: stderr_f.close()
        return None
    else:
        raise ValueError(f"Unknown mode: {mode}")


def compute_metrics_from_csv(csv_path: Path) -> RunResult:
    if not METRICS_SCRIPT.exists():
        raise FileNotFoundError(f"{METRICS_SCRIPT} missing")
    try:
        res = run_command(
            [sys.executable, str(METRICS_SCRIPT), "--file", str(csv_path), "--mode", "regex", "--json"],
            capture=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"recursive_F1.py failed for {csv_path} (exit {e.returncode})\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
        ) from e
    data = json.loads(res.stdout)
    if "metrics" not in data or "confusion" not in data:
        raise RuntimeError(f"Unexpected JSON from recursive_F1.py: {data}")
    return RunResult(confusion=data["confusion"], metrics=data["metrics"])


def agg_mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    # population std (consistent across small N)
    m = mean(values)
    try:
        s = pstdev(values)
    except Exception:
        s = float("nan")
    return m, s


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run SLURM scripts multiple times and average metrics")
    ap.add_argument("--pattern", default="SLURM*.sh", help="Glob for SLURM scripts to run")
    ap.add_argument("--runs", type=int, default=10, help="Times to run each script")
    ap.add_argument("--mode", choices=["auto", "sbatch", "bash"], default="auto", help="How to run scripts")
    ap.add_argument("--timeout-sec", type=float, default=24*3600, help="Timeout per run")
    ap.add_argument("--sleep-after", type=float, default=5.0, help="Extra sleep after job completes (sec)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out-dir", default="runs", help="Directory to store per-run artifacts")
    args = ap.parse_args(argv)

    scripts = sorted(HERE.glob(args.pattern))
    if not scripts:
        raise SystemExit(f"No scripts found for pattern {args.pattern} in {HERE}")

    # Decide mode
    if args.mode == "auto":
        mode = "sbatch" if which("sbatch") else "bash"
    else:
        mode = args.mode

    if args.verbose:
        print(f"Found {len(scripts)} scripts: {[s.name for s in scripts]}")
        print(f"Execution mode: {mode}")

    # Session directory for this invocation
    session_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = HERE / args.out_dir / session_tag
    session_dir.mkdir(parents=True, exist_ok=True)

    all_summary = []

    for script in scripts:
        model, input_base = parse_benchmark_args_from_slurm(script)
        pred_csv = predictions_csv_for(model, input_base)
        if args.verbose:
            print(f"\n== {script.name} => model={model} input={input_base}")
            print(f"Predictions file: {pred_csv.name}")

        per_run: List[RunResult] = []
        script_dir = session_dir / script.stem / f"{model_safe_name(model)}__{input_base}"
        script_dir.mkdir(parents=True, exist_ok=True)

        for r in range(1, args.runs + 1):
            before_mtime = pred_csv.stat().st_mtime if pred_csv.exists() else 0.0
            if args.verbose:
                print(f"[Run {r}/{args.runs}] Starting {script.name} (prev mtime={before_mtime})")

            run_dir = script_dir / f"run_{r:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            stdout_log = run_dir / "stdout.log"
            stderr_log = run_dir / "stderr.log"

            start_ts = datetime.now().isoformat(timespec="seconds")
            job_id = run_slurm_script(
                script,
                mode=mode,
                timeout_sec=args.timeout_sec,
                run_index=r,
                stdout_path=stdout_log if mode == "bash" else None,
                stderr_path=stderr_log if mode == "bash" else None,
            )
            time.sleep(args.sleep_after)

            # Wait for predictions update
            t0 = time.time()
            while True:
                if pred_csv.exists() and pred_csv.stat().st_mtime > before_mtime:
                    break
                if time.time() - t0 > args.timeout_sec:
                    raise TimeoutError(f"Timeout waiting for {pred_csv.name} to be created/updated")
                time.sleep(5)

            # Preserve predictions for this run
            try:
                copy2(pred_csv, run_dir / pred_csv.name)
            except Exception as e:
                if args.verbose:
                    print(f"[Run {r}] Warning: failed to copy predictions: {e}")

            # Compute metrics immediately
            res = compute_metrics_from_csv(pred_csv)
            per_run.append(res)
            if args.verbose:
                f1 = res.metrics.get("f1", float("nan"))
                acc = res.metrics.get("accuracy", float("nan"))
                print(f"[Run {r}] f1={f1:.4f} acc={acc:.4f}")

            # Save per-run metrics and metadata
            run_info = {
                "script": script.name,
                "model": model,
                "input": input_base,
                "mode": mode,
                "job_id": job_id,
                "run_index": r,
                "seed": r,
                "start_time": start_ts,
                "end_time": datetime.now().isoformat(timespec="seconds"),
                "predictions_file": pred_csv.name,
            }
            write_json(run_dir / "metrics.json", {"confusion": res.confusion, "metrics": res.metrics})
            write_json(run_dir / "run_info.json", run_info)

        # Aggregate across runs
        metric_keys = sorted(per_run[0].metrics.keys()) if per_run else []
        conf_keys = sorted(per_run[0].confusion.keys()) if per_run else []

        rows_csv = []
        for i, rr in enumerate(per_run, start=1):
            row = {"run": i}
            for k in metric_keys:
                row[k] = rr.metrics.get(k)
            rows_csv.append(row)

        metrics_mean, metrics_std = {}, {}
        for k in metric_keys:
            vals = [rr.metrics.get(k, float("nan")) for rr in per_run]
            vals = [v for v in vals if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))]
            m, s = agg_mean_std(vals)
            metrics_mean[k] = m
            metrics_std[k] = s

        confusion_sum = {k: sum(float(rr.confusion.get(k, 0.0)) for rr in per_run) for k in conf_keys}
        confusion_mean = {k: (confusion_sum[k] / len(per_run) if per_run else float("nan")) for k in conf_keys}

        agg_row_mean = {"run": "mean", **metrics_mean}
        agg_row_std = {"run": "std", **metrics_std}
        rows_csv.extend([agg_row_mean, agg_row_std])

        base = script.name + f".{args.runs}runs"
        out_json = HERE / f"{base}.metrics.json"
        out_csv = HERE / f"{base}.metrics.csv"
        write_json(out_json, {
            "script": script.name,
            "model": model,
            "input": input_base,
            "runs": args.runs,
            "metrics_mean": metrics_mean,
            "metrics_std": metrics_std,
            "confusion_sum": confusion_sum,
            "confusion_mean": confusion_mean,
        })
        write_csv(out_csv, fieldnames=["run", *metric_keys], rows=rows_csv)

        # Copy aggregates alongside per-run artifacts
        try:
            copy2(out_json, script_dir / out_json.name)
            copy2(out_csv, script_dir / out_csv.name)
        except Exception as e:
            if args.verbose:
                print(f"Warning: failed to copy aggregate files: {e}")

        all_summary.append({
            "script": script.name,
            "model": model,
            "input": input_base,
            "runs": args.runs,
            "metrics_mean": metrics_mean,
            "metrics_std": metrics_std,
            "confusion_sum": confusion_sum,
            "confusion_mean": confusion_mean,
        })

    write_json(HERE / f"summary_{args.runs}runs.json", {"scripts": all_summary})
    print(f"\nDone. Wrote per-script metrics and summary_{args.runs}runs.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
