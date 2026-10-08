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
from dataset import SignDatasetVar, feat_var
from bpe_tokenizer import BPETokenizer
from model import SignTransformer, SignTransformerV2
from tokenizer import EOS_ID, SOS_ID, Tokenizer


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def beam_search(model, src, beam=5, max_len=60, lp_alpha=0.6, src_mask=None):
    """Beam search with length penalty. src: (1, Tv, D). Returns id list."""
    model.eval()
    if isinstance(model, SignTransformerV2):
        mem = model.encode(src, src_mask)
        mem_mask = model.frontend.downsample_mask(src_mask).to(src.device) \
            if src_mask is not None else None
    else:
        mem = model.encode(src)
        mem_mask = None
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
            if isinstance(model, SignTransformerV2):
                logp = F.log_softmax(model.head(model.decoder(
                    model.pos(model.tgt_emb(y)), mem, tgt_mask=causal,
                    memory_key_padding_mask=mem_mask))[:, -1], -1)
            else:
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
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer file (default: from ckpt cfg, else tokenizer.json)")
    ap.add_argument("-o", default=None)
    args = ap.parse_args()

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["cfg"]
    TokCls = BPETokenizer if cfg.get("tok_type") == "bpe" else Tokenizer
    tok = TokCls.load(args.tokenizer or cfg.get("tokenizer", "tokenizer.json"))
    variable = cfg.get("variable", False)
    if variable:
        model = SignTransformerV2(cfg["vocab"],
                                  in_dim=cfg.get("in_dim", 633),
                                  enc_layers=cfg.get("enc", 4),
                                  dec_layers=cfg.get("dec", 4),
                                  frontend_layers=cfg.get("frontend_layers", 2)).to(device)
    else:
        model = SignTransformer(cfg["vocab"],
                                in_dim=cfg.get("in_dim", 225),
                                enc_layers=cfg.get("enc", 4),
                                dec_layers=cfg.get("dec", 4)).to(device)
    model.load_state_dict(ckpt["state"], strict=False)  # old ckpts lack ctc_head
    model.eval()

    def decode(stem):
        if variable:
            kind = cfg.get("kind", "face256")
            f = feat_var(stem, kind=kind)
            src = torch.from_numpy(f).unsqueeze(0).to(device)
            src_mask = torch.zeros(1, len(f), dtype=torch.bool, device=device)
            if args.beam and args.beam > 1:
                ids = beam_search(model, src, beam=args.beam, src_mask=src_mask)
            else:
                ids = model.greedy(src, src_mask=src_mask, max_len=60)[0].tolist()
            return tok.decode(ids)
        kind = cfg.get("kind") or ({225: "pos", 450: "posvel"}.get(cfg.get("in_dim", 225), "pos"))
        src = torch.from_numpy(feat(stem, kind=kind)).unsqueeze(0).to(device)
        if args.beam and args.beam > 1:
            ids = beam_search(model, src, beam=args.beam)
        else:
            ids = model.greedy(src, max_len=60)[0].tolist()
        return tok.decode(ids)

    if args.split:
        rows = (SignDatasetVar if variable else SignDataset)(args.split, tok).rows
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
