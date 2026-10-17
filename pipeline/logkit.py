"""Paper-grade logging for pipeline runs.

One run directory under logs/<name>/ holding everything for later analysis:
  run.json          full config snapshot + data manifest + versions
  epochs.csv        one row per epoch, all decision + diagnostic numbers
  steps.jsonl       every Nth training step (losses, sims, grads, lr, pace)
  train.log         human-readable epoch lines + samples (the tail-able one)
Console stays lean: only the decision line per epoch.
"""
import csv
import datetime as dt
import json
import subprocess
import time
from pathlib import Path


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return "unknown"


class RunLog:
    def __init__(self, name, cfg, extra=None):
        self.dir = Path(__file__).resolve().parents[1] / "logs" / name
        self.dir.mkdir(parents=True, exist_ok=True)
        import torch
        sacrebleu = None
        try:
            import sacrebleu as sb
            sacrebleu = sb.__version__
        except Exception:
            pass
        payload = {"cfg": cfg, "extra": extra or {},
                   "torch": torch.__version__, "sacrebleu": sacrebleu,
                   "git": git_head(), "device": cfg.get("device"),
                   "started": dt.datetime.now().isoformat(timespec="seconds")}
        json.dump(payload, open(self.dir / "run.json", "w"), indent=2)
        self.log = open(self.dir / "train.log", "a")
        self.csv = open(self.dir / "epochs.csv", "a", newline="")
        self.csvw = csv.writer(self.csv)
        self._csv_header = None

    def epoch_header(self, cols):
        self.csvw.writerow(cols)
        self.csv.flush()

    def epoch(self, row):
        self.csvw.writerow(row)
        self.csv.flush()

    def step(self, payload):
        with open(self.dir / "steps.jsonl", "a") as f:
            f.write(json.dumps(payload, default=float) + "\n")

    def line(self, msg, console=True):
        out = f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
        self.log.write(out + "\n")
        self.log.flush()
        if console:
            print(out, flush=True)

    def samples(self, pairs):
        for got, want in pairs:
            self.log.write(f"  ex pred={got!r} true={want!r}\n")
        self.log.flush()

    def t(self):
        return time.time()
