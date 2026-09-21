#!/usr/bin/env python3
"""One-command extraction status. Usage: .venv312/bin/python xstat.py"""
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))


def procs():
    out = subprocess.run(["pgrep", "-f", "wextract\\.py"],
                         capture_output=True, text=True).stdout.strip()
    return out.split() if out else []


def workers():
    out = subprocess.run(["ps", "aux"], capture_output=True, text=True).stdout
    return sum(1 for ln in out.splitlines() if "multiprocessing.spawn" in ln)


def tail_bar(log):
    try:
        with open(os.path.join(ROOT, log), "rb") as f:
            f.seek(max(0, os.path.getsize(os.path.join(ROOT, log)) - 400))
            chunk = f.read().decode(errors="replace")
        bars = re.findall(r"(\d+/\d+ \[[\d:]+\<[^\]]+\])", chunk)
        return bars[-1] if bars else "no bar yet"
    except FileNotFoundError:
        return "no log"


n_new = sum(1 for f in os.listdir(os.path.join(ROOT, "keypoints"))
            if f.endswith(".npz") and "sentence_id_" in f)
print(f"keypoints (new stems): {n_new} / 75000")
print(f"mac runs alive: {procs() or 'none'} | workers: {workers()}")
print(f"mac sid1-333 run : {tail_bar('wextract_mac.log')}")
print(f"mac sid334-500 run: {tail_bar('wextract_mac2.log')}")
