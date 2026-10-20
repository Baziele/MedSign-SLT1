"""Final-report evaluation for a finished stage-2 run. Read-only: no training,
no checkpoint writes. Loads the `best` checkpoint and scores it under no_grad
on any named pool with the stage-2 protocol: corpus BLEU (same scorer as
train.val_metrics) + exact matches + a "correct translation" rate, defined as
a normalized string match OR >= 80% word-level F1 overlap with the reference.

  pool "all_val"   3,494 GH+MV validation rows (pooled + per-source BLEU)
  pool "mv_test"   MV signer-holdout test split (unseen signer S5)
  pool "jw"        held-out JW val and test splits

Usage:
  .venv312/bin/python scripts/eval_final.py --ckpt models/checkpoints/best_p1t2.pt \
      --pools all_val,mv_test --out logs/eval_final_p1t2.json
"""
import argparse
import collections
import json
import string
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

import pipeline.data as D
from dataset import collate_var, KIND_DIM
from model import SignTransformerV2
from tokenizer import PAD_ID
from train import _bleu_detail

PUNCT = str.maketrans("", "", string.punctuation)


def _norm(t):
    return " ".join(t.translate(PUNCT).lower().split())


def _word_f1(h, r):
    H = collections.Counter(_norm(h).split())
    R = collections.Counter(_norm(r).split())
    ov = sum((H & R).values())
    p = ov / max(1, sum(H.values()))
    rc = ov / max(1, sum(R.values()))
    return 2 * p * rc / max(1e-9, p + rc)


def _is_correct(h, r, thr=0.8):
    return _norm(h) == _norm(r) or _word_f1(h, r) >= thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pools", default="all_val")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="logs/eval_final.json")
    args = ap.parse_args()

    device = torch.device("mps")
    ck = torch.load(args.ckpt, map_location=device)
    cfg = ck["cfg"]
    print(f"ckpt {args.ckpt}"),

    from bpe_tokenizer import BPETokenizer
    tok = BPETokenizer.load(cfg["tokenizer"])

    ck_source = [s.strip() for s in (cfg.get("jw_source") or "").split(",")
                 if s.strip()]
    model = SignTransformerV2(
        len(tok), in_dim=KIND_DIM[cfg["kind"]],
        d_model=cfg.get("d_model", 256),
        enc_layers=cfg.get("enc", 4), dec_layers=cfg.get("dec", 4),
        frontend_layers=cfg.get("frontend_layers", 2)).to(device)
    model.load_state_dict(ck["state"])
    model.eval()

    def rows_for(pool):
        if pool == "all_val":
            return D.load_main_rows("all_val")
        if pool == "mv_test":
            return D.load_main_rows("mv_test")
        if pool == "jw":
            jw = D.load_jw_rows(ck_source)
            return [r for r in jw if r["split"] == "val"] + \
                   [r for r in jw if r["split"] == "test"]
        raise SystemExit(f"unknown pool {pool}")

    def pool(name):
        rows = rows_for(name)
        ds = D.VarRowDataset(rows, tok, kind=cfg["kind"])
        lens = [D.len_of(r, cfg["kind"]) for r in rows]
        loader = DataLoader(ds, batch_sampler=D.bucketed_batches(
            ds, lens, args.batch, shuffle=False),
            collate_fn=collate_var, num_workers=2)
        t0 = time.time()
        preds, refs = [], []
        n_tok = 0
        with torch.no_grad():
            for b in loader:
                src = b["src"].to(device)
                mask = b["src_mask"].to(device)
                ys = model.greedy(src, mask)
                preds += [tok.decode(y.tolist()) for y in ys]
                refs += b["texts"]
        d = _bleu_detail(preds, refs)
        exact = sum(1 for p, r in zip(preds, refs)
                    if p.strip().lower() == r.strip().lower())
        correct = sum(1 for p, r in zip(preds, refs) if _is_correct(p, r))
        n = len(preds)
        per_gram = {k: round(d[k], 3) for k in ("bleu", "p1", "p2", "p3",
                                                "p4", "bp", "sys_len",
                                                "ref_len")}
        m = {
            "n": n, "bleu": round(d["bleu"], 2),
            "exact_n": exact, "exact_pct": round(100 * exact / n, 1),
            "correct_n": correct, "correct_pct": round(100 * correct / n, 1),
            "time_s": round(time.time() - t0, 1), **per_gram,
        }
        print(f"  [{name}] {json.dumps(m)}", flush=True)
        return m

    results = {"ckpt": str(args.ckpt), "correct_def":
               "normalized match or >=80% word-F1 overlap", "pools": {}}
    for name in [p.strip() for p in args.pools.split(",") if p.strip()]:
        results["pools"][name] = pool(name)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
