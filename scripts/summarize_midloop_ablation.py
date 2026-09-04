#!/usr/bin/env python3
"""Summarize the mid-layer-loop ablation runs under outputs/.

For every outputs/gsm-lotus-*/ directory it reports the training progress
(epochs finished, best / last validation accuracy parsed from train.log) and,
once eval has run, the GSM8K test accuracy, OOD accuracies and the per-example
"thought" latency from results_{gsm8k,ood}.json. Prints a Markdown table.

    python scripts/summarize_midloop_ablation.py [--glob 'outputs/gsm-lotus-gpt2-*']
"""
import argparse
import glob
import json
import os
import re

ACC_RE = re.compile(r"Accuracy on validation set: (\d+) / (\d+) = ([0-9.]+)")
EPOCH_RE = re.compile(r"Training Epoch: (\d+)/(\d+) \(Stage (\d+)\)")
MEM_RE = re.compile(r"Epoch (\d+) done: peak GPU mem allocated=([0-9.]+) GB")


def parse_train_log(path):
    info = {"val_accs": [], "epochs_done": 0, "num_epochs": None, "stage": None, "peak_mem_gb": None}
    if not os.path.exists(path):
        return info
    with open(path, "rb") as f:
        text = f.read().decode("utf-8", errors="replace").replace("\r", "\n")
    for m in ACC_RE.finditer(text):
        info["val_accs"].append(float(m.group(3)))
    info["epochs_done"] = len(info["val_accs"])
    last = None
    for m in EPOCH_RE.finditer(text):
        last = m
    if last:
        info["num_epochs"] = int(last.group(2))
        info["stage"] = int(last.group(3))
    for m in MEM_RE.finditer(text):
        info["peak_mem_gb"] = float(m.group(2))
    return info


def load_results(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def fmt_pct(x):
    return "-" if x is None else f"{100 * x:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/gsm-lotus-gpt2-*")
    args = ap.parse_args()

    rows = []
    for d in sorted(glob.glob(args.glob)):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        tr = parse_train_log(os.path.join(d, "train.log"))
        g = load_results(os.path.join(d, "results_gsm8k.json")).get("gsm8k", {})
        ood = load_results(os.path.join(d, "results_ood.json"))
        status = "DONE" if os.path.exists(os.path.join(d, "DONE")) else (
            "trained" if os.path.exists(os.path.join(d, "TRAIN_DONE")) else "running/pending")
        best = max(tr["val_accs"]) if tr["val_accs"] else None
        last = tr["val_accs"][-1] if tr["val_accs"] else None
        thought_ms = None
        if g.get("total"):
            thought_ms = 1000 * g["thought_time"] / g["total"]
        ood_accs = [ood.get(k, {}).get("accuracy") for k in ("gsm-hard", "multi-arith", "svamp")]
        ood_avg = None if any(a is None for a in ood_accs) else sum(ood_accs) / 3
        rows.append({
            "run": name.replace("gsm-lotus-gpt2-", ""),
            "status": status,
            "epochs": f"{tr['epochs_done']}/{tr['num_epochs'] or '?'}",
            "val_best": best, "val_last": last,
            "test": g.get("accuracy"),
            "gsm_hard": ood_accs[0], "multi_arith": ood_accs[1], "svamp": ood_accs[2], "ood_avg": ood_avg,
            "thought_ms": thought_ms,
            "peak_mem": tr["peak_mem_gb"],
        })

    hdr = "| run | status | epochs | val best | val last | GSM8K test | GSM-Hard | MultiArith | SVAMP | OOD avg | thought ms/ex | peak GB |"
    print(hdr)
    print("|" + "---|" * (hdr.count("|") - 1))
    for r in rows:
        print(
            f"| {r['run']} | {r['status']} | {r['epochs']} | {fmt_pct(r['val_best'])} | {fmt_pct(r['val_last'])} | "
            f"{fmt_pct(r['test'])} | {fmt_pct(r['gsm_hard'])} | {fmt_pct(r['multi_arith'])} | {fmt_pct(r['svamp'])} | "
            f"{fmt_pct(r['ood_avg'])} | {'-' if r['thought_ms'] is None else f'{r['thought_ms']:.1f}'} | "
            f"{'-' if r['peak_mem'] is None else f'{r['peak_mem']:.1f}'} |"
        )


if __name__ == "__main__":
    main()
