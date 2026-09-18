"""Inference: greedy or beam-5 decode for a stem, list of stems, or a split.

Usage:
  .venv312/bin/python translate.py 1A
  .venv312/bin/python translate.py --split test_T --beam 5 -o translation/preds_test_T.csv
"""
import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset import SignDataset, feat
from model import SignTransformer
from tokenizer import EOS_ID, SOS_ID, Tokenizer


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def beam_search(model, src, beam=5, max_len=60, lp_alpha=0.6):
    """Beam search with length penalty. src: (1, 64, 225). Returns id list."""
    model.eval()
    mem = model.encode(src)
    beams = [([SOS_ID], 0.0)]
    done = []
    for _ in range(max_len):
        cands = []
        for seq, score in beams:
            if seq[-1] == EOS_ID:
                done.append((seq, score))
                continue
            y = torch.tensor([seq], device=src.device)
            causal = torch.triu(
                torch.ones(len(seq), len(seq), dtype=torch.bool, device=src.device), diagonal=1)
            logp = F.log_softmax(model.head(model.decoder(
                model.pos(model.tgt_emb(y)), mem, tgt_mask=causal))[:, -1], -1)
            topk = logp[0].topk(beam)
            for v, lp in zip(topk.indices.tolist(), topk.values.tolist()):
                cands.append((seq + [v], score + lp))
        beams = sorted(cands, key=lambda x: x[1], reverse=True)[:beam]
        if len(done) >= beam:
            break
    pool = done + [b for b in beams]
    lp = lambda s: ((5 + len(s)) / 6) ** lp_alpha  # length-norm like GNMT
    return max(pool, key=lambda x: x[1] / lp(x[0]))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", nargs="?")
    ap.add_argument("--split", default=None)
    ap.add_argument("--beam", type=int, default=0, help="0=greedy, else beam width")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("-o", default=None)
    args = ap.parse_args()

    device = get_device()
    tok = Tokenizer.load()
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["cfg"]
    model = SignTransformer(cfg["vocab"],
                            in_dim=cfg.get("in_dim", 225),
                            enc_layers=cfg.get("enc", 4),
                            dec_layers=cfg.get("dec", 4)).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()

    def decode(stem):
        src = torch.from_numpy(
            feat(stem, velocity=cfg.get("in_dim", 225) == 450)).unsqueeze(0).to(device)
        if args.beam and args.beam > 1:
            ids = beam_search(model, src, beam=args.beam)
        else:
            ids = model.greedy(src, max_len=60)[0].tolist()
        return tok.decode(ids)

    if args.split:
        rows = SignDataset(args.split, tok).rows
        out = [{"video": r["video"], "signer": r["signer"], "category": r["category"],
                "true": r["text"], "pred": decode(r["stem"])} for r in rows]
        o = args.o or f"translation/preds_{args.split}.csv"
        Path(o).parent.mkdir(parents=True, exist_ok=True)
        with open(o, "w", newline="") as f:
            w = csv.DictWriter(f, ["video", "signer", "category", "true", "pred"])
            w.writeheader()
            w.writerows(out)
        print(f"wrote {o} ({len(out)} clips)")
    else:
        print(decode(args.stem))


if __name__ == "__main__":
    main()
