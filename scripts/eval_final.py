"""Final-report evaluation for a finished stage-2 run. Read-only: no training,
no checkpoint writes. Loads the `best` checkpoint and scores it under no_grad
on every held-out pool, with the state-2 protocol (teacher-forced loss/acc +
greedy BLEU + exact matches):

  pool 1  all_val  (3,494 GH+MV rows; per-domain bleu_gh / bleu_mv)
  pool 2  jw val   (held-out chapters, any cross-source bias-free probe)
  pool 3  jw TEST  (chapter-level holdout untouched during ALL training)

Usage:
  .venv312/bin/python scripts/eval_final.py --ckpt models/checkpoints/best_p1t2.pt \
      [--source nwt-ghs,mwb,lfb] [--out logs/eval_final_p1t2.json]
"""
import argparse
import json
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
from train import val_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--source", default="nwt-ghs,mwb,lfb")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="logs/eval_final.json")
    args = ap.parse_args()

    device = torch.device("mps")
    ck = torch.load(args.ckpt, map_location=device)
    cfg = ck["cfg"]
    print(f"ckpt {args.ckpt}: best {ck.get('best_ep')} @ BLEU (run) "
          f"{ck.get('best')}", flush=True)

    from bpe_tokenizer import BPETokenizer
    tok = BPETokenizer.load(cfg["tokenizer"])

    ck_source = [s.strip() for s in (cfg.get("jw_source") or "").split(",")
                 if s.strip()] or [s.strip() for s in args.source.split(",")
                                   if s.strip()]
    model = SignTransformerV2(
        len(tok), in_dim=KIND_DIM[cfg["kind"]],
        d_model=cfg.get("d_model", 256),
        enc_layers=cfg.get("enc", 4), dec_layers=cfg.get("dec", 4),
        frontend_layers=cfg.get("frontend_layers", 2)).to(device)
    model.load_state_dict(ck["state"])
    model.eval()

    def pool(rows, name):
        if not rows:
            return None
        ds = D.VarRowDataset(rows, tok, kind=cfg["kind"])
        lens = [D.len_of(r, cfg["kind"]) for r in rows]
        loader = DataLoader(ds, batch_sampler=D.bucketed_batches(
            ds, lens, args.batch, shuffle=False),
            collate_fn=collate_var, num_workers=2)
        t0 = time.time()
        m = val_metrics(model, loader, tok, device, True, crit)
        m.pop("samples", None)
        m["time_s"] = round(time.time() - t0, 1)
        m = {k: (round(v, 4) if isinstance(v, float) else v)
             for k, v in m.items()}
        print(f"  [{name}] {json.dumps(m)}", flush=True)
        return m

    results = {"ckpt": str(args.ckpt), "best_ep": ck.get("best_ep"),
               "run_best": ck.get("best"), "pools": {}}
    crit = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)

    val_rows = D.load_main_rows("all_val")
    results["pools"]["all_val"] = pool(val_rows, "all_val GH+MV")

    jw_rows = D.load_jw_rows(ck_source)
    for split in ("val", "test"):
        rows = [r for r in jw_rows if r["split"] == split]
        results["pools"][f"jw_{split}"] = pool(rows, f"jw {split} "
                                               f"({len(rows)} rows)")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
