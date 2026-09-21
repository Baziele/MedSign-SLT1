"""Train SignTransformer with teacher forcing; checkpoint on val BLEU.

Usage: .venv312/bin/python train.py [--epochs 100] [--batch 32] [--lr 3e-4]
Logs to train.log, best checkpoint to best.pt (model + config only;
tokenizer lives in tokenizer.json).
"""
import argparse
import time

import torch
import torch.nn as nn
from sacrebleu import corpus_bleu
from torch.utils.data import DataLoader

from dataset import SignDataset, collate
from dataset import SignDatasetVar, bucketed_loader, collate_var
from model import SignTransformer, SignTransformerV2
from tokenizer import PAD_ID, Tokenizer


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def val_bleu(model, loader, tok, device, variable=False):
    model.eval()
    preds, refs = [], []
    for b in loader:
        if variable:
            ys = model.greedy(b["src"].to(device), b["src_mask"].to(device))
        else:
            ys = model.greedy(b["src"].to(device))
        preds += [tok.decode(y.tolist()) for y in ys]
        refs += b["texts"]
    return corpus_bleu(preds, [refs]).score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--resume", action="store_true",
                    help="resume from last.pt (optimizer + epoch + best)")
    ap.add_argument("--tag", default="",
                    help="run tag: checkpoints become best<TAG>.pt / last<TAG>.pt")
    ap.add_argument("--enc-layers", type=int, default=6)
    ap.add_argument("--dec-layers", type=int, default=2)
    ap.add_argument("--warmup-steps", type=int, default=1000,
                    help="linear LR ramp-up steps, then cosine decay; 0 = flat LR")
    ap.add_argument("--in-dim", type=int, default=None,
                    help="override input dim (default: from --kind)")
    ap.add_argument("--kind", default="face",
                    help="feature set: pos (225) | posvel (450) | face (633) | pos256 (225 var) | face256 (633 var)")
    ap.add_argument("--augment", action="store_true",
                    help="train-only augmentation (speed/spatial/jitter)")
    ap.add_argument("--variable", action="store_true",
                    help="variable-length path: Tv<=256 + padding mask + length bucketing + conv frontend (no 64-squash)")
    ap.add_argument("--frontend-layers", type=int, default=2, choices=[2, 3],
                    help="conv frontend depth: 2 = 4x downsample (256->64), 3 = 8x (256->32)")
    ap.add_argument("--max-frames", type=int, default=256,
                    help="variable-length cap T_MAX (longer clips downsampled, shorter kept native)")
    args = ap.parse_args()

    device = get_device()
    print("device:", device)
    from dataset import KIND_DIM
    from preprocess import T_MAX
    if args.variable and args.kind in ("pos", "face"):
        args.kind = {"pos": "pos256", "face": "face256"}[args.kind]
        print(f"variable path: kind auto-upgraded to {args.kind}")
    in_dim = args.in_dim or KIND_DIM[args.kind]
    variable = args.variable or args.kind in ("pos256", "face256")
    tok = Tokenizer.load()
    if variable:
        from dataset import VAR_KINDS
        assert args.kind in VAR_KINDS, "--variable needs --kind pos256|face256"
        train_ds = SignDatasetVar("train", tok, augment_flag=args.augment,
                                  kind=args.kind)
        val_ds = SignDatasetVar("val", tok, kind=args.kind)
        train_loader = bucketed_loader(train_ds, batch=args.batch, shuffle=True)
        val_loader = bucketed_loader(val_ds, batch=args.batch, shuffle=False)
    else:
        train_ds = SignDataset("train", tok, augment_flag=args.augment, kind=args.kind)
        val_ds = SignDataset("val", tok, kind=args.kind)
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val_ds, batch_size=args.batch, collate_fn=collate)

    if variable:
        model = SignTransformerV2(len(tok), in_dim=in_dim,
                                  enc_layers=args.enc_layers,
                                  dec_layers=args.dec_layers,
                                  frontend_layers=args.frontend_layers).to(device)
    else:
        model = SignTransformer(len(tok), in_dim=in_dim,
                                enc_layers=args.enc_layers,
                                dec_layers=args.dec_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.1f}M enc={args.enc_layers} dec={args.dec_layers}"
          f"{f' frontend{args.frontend_layers}x' if variable else ''}"
          f" kind={args.kind} variable={variable}")
    cfg = {"vocab": len(tok), "kind": args.kind, "in_dim": in_dim,
           "enc": args.enc_layers, "dec": args.dec_layers,
           "variable": variable, "frontend_layers": args.frontend_layers,
           "max_frames": args.max_frames if variable else 64}
    best_pt, last_pt = f"best{args.tag}.pt", f"last{args.tag}.pt"
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)
    total_steps = args.epochs * len(train_loader)
    sched = None
    if args.warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-3,
                                                   total_iters=args.warmup_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, total_steps - args.warmup_steps))
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, [warmup, cosine], milestones=[args.warmup_steps])

    log = open("train.log", "a")
    best, bad, start_ep = -1.0, 0, 1
    if args.resume:
        ckpt = torch.load(last_pt, map_location=device)
        model.load_state_dict(ckpt["state"])
        opt.load_state_dict(ckpt["opt"])
        if sched is not None and ckpt.get("sched") is not None:
            sched.load_state_dict(ckpt["sched"])
        best, bad, start_ep = ckpt["best"], ckpt["bad"], ckpt["ep"] + 1
        print(f"resumed at ep {start_ep} (best val BLEU {best:.1f})")
    for ep in range(start_ep, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        t0 = time.time()
        for b in train_loader:
            opt.zero_grad()
            if variable:
                logits = model(b["src"].to(device), b["tgt_in"].to(device),
                               b["pad_mask"].to(device), b["src_mask"].to(device))
            else:
                logits = model(b["src"].to(device), b["tgt_in"].to(device),
                               b["pad_mask"].to(device))
            loss = crit(logits.reshape(-1, len(tok)), b["tgt_out"].reshape(-1).to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched is not None:
                sched.step()
            tot += loss.item() * len(b["stems"])
            n += len(b["stems"])
        bleu = val_bleu(model, val_loader, tok, device, variable=variable)
        line = f"ep {ep}: loss={tot/n:.3f} val_bleu={bleu:.1f} ({time.time()-t0:.0f}s)"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
        if bleu > best:
            best, bad = bleu, 0
            torch.save({"cfg": cfg, "state": model.state_dict()}, best_pt)
        else:
            bad += 1
        torch.save({"ep": ep, "best": best, "bad": bad,
                    "state": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict() if sched else None}, last_pt)
        if bad >= args.patience:
            print(f"early stop at ep {ep}, best val BLEU {best:.1f}")
            break
    print(f"done. best val BLEU {best:.1f} -> {best_pt}")


if __name__ == "__main__":
    main()
