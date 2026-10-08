"""Train SignTransformer with teacher forcing; checkpoint on val BLEU.

Usage: .venv312/bin/python train.py [--epochs 100] [--batch 32] [--lr 3e-4]
  --variable --kind face256 --augment --enc-layers 4 --dec-layers 4 --tag _v7
Logs per-epoch to train<TAG>.log + metrics<TAG>.csv, best checkpoint to
best<TAG>.pt (model + config only; tokenizer lives in tokenizer.json).
"""
import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sacrebleu import corpus_bleu
from torch.utils.data import DataLoader

from dataset import SignDataset, collate
from dataset import SignDatasetVar, bucketed_loader, collate_var, _seed_worker
from bpe_tokenizer import BPETokenizer
from model import SignTransformer, SignTransformerV2
from tokenizer import PAD_ID, Tokenizer


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def _bleu_detail(preds, refs):
    """corpus_bleu plus the paper-grade breakdown (n-gram precisions, BP, lens)."""
    r = corpus_bleu(preds, [refs])
    return {"bleu": r.score, "p1": r.precisions[0], "p2": r.precisions[1],
            "p3": r.precisions[2], "p4": r.precisions[3], "bp": r.bp,
            "sys_len": r.sys_len, "ref_len": r.ref_len}


def _is_mv(stem):
    return stem.startswith("train_") or stem.startswith("test_")


@torch.no_grad()
def val_metrics(model, loader, tok, device, variable, crit):
    """Teacher-forced val loss/acc + greedy BLEU/exact + 10 samples. Returns dict.

    BLEU is reported for the pooled val AND per domain (GH unseen-text vs MV
    take-holdout) — the pooled number hides which half drives selection, and
    that distinction is the core result of the GH+MV runs.
    """
    model.eval()
    sum_loss, n_tok, n_correct, n_exact, n = 0.0, 0, 0, 0, 0
    preds, refs, stems_all = [], [], []
    t0 = time.time()
    for b in loader:
        src, tgt_in = b["src"].to(device), b["tgt_in"].to(device)
        tgt_out, pad_mask = b["tgt_out"].to(device), b["pad_mask"].to(device)
        if variable:
            logits = model(src, tgt_in, pad_mask, b["src_mask"].to(device))
            ys = model.greedy(src, b["src_mask"].to(device))
        else:
            logits = model(src, tgt_in, pad_mask)
            ys = model.greedy(src)
        loss = crit(logits.reshape(-1, len(tok)), tgt_out.reshape(-1))
        nt = int((tgt_out != PAD_ID).sum())
        sum_loss += loss.item() * nt
        n_tok += nt
        n_correct += int(((logits.argmax(-1) == tgt_out) & (tgt_out != PAD_ID)).sum())
        bp = [tok.decode(y.tolist()) for y in ys]
        preds += bp
        refs += b["texts"]
        stems_all += b["stems"]
        n += len(b["stems"])
        for p, r in zip(bp, b["texts"]):
            n_exact += (p.strip().lower() == r.strip().lower())
    d = _bleu_detail(preds, refs)
    out = {"loss": sum_loss / max(1, n_tok), "acc": n_correct / max(1, n_tok),
           "exact": n_exact, "n": n, "n_tok": n_tok,
           "time": time.time() - t0, "samples": list(zip(preds[:10], refs[:10]))}
    out.update({f"val_{k}": v for k, v in d.items()})
    for dom, want_mv in (("gh", False), ("mv", True)):
        idx = [i for i, s in enumerate(stems_all) if _is_mv(s) == want_mv]
        if idx:
            dd = _bleu_detail([preds[i] for i in idx], [refs[i] for i in idx])
            ex = sum(preds[i].strip().lower() == refs[i].strip().lower()
                     for i in idx)
            out[f"bleu_{dom}"] = dd["bleu"]
            out[f"exact_{dom}"] = ex
            out[f"n_{dom}"] = len(idx)
        else:
            out[f"bleu_{dom}"] = -1.0
            out[f"exact_{dom}"] = 0
            out[f"n_{dom}"] = 0
    return out


def _train_step(model, src_d, tgt_in_d, tgt_out_d, pad_d, src_m, need_mem,
                samp_p, variable, ctc_crit, tok, b, ctc_weight, crit, device):
    """One forward (+optional sampling pass + TxtCTC). Returns
    (logits, loss, lctc_or_None, nvalid, nskip)."""
    if variable and samp_p > 0:
        # pass 1: teacher-forced predictions to mix in. Runs as a PLAIN
        # train-mode forward (NOT no_grad: on MPS, grad-disabled + dropout
        # flips attention onto a kernel without dropout support and crashes).
        # argmax is non-differentiable so no graph flows into pass 2; the
        # pass-1 graph is freed with `del` below. mem reused in pass 2 so
        # only the (cheap) decoder runs twice.
        logits_tf, mem = model(src_d, tgt_in_d, pad_d, src_m,
                               return_mem=True)
        pred_ids = logits_tf.argmax(-1)
        del logits_tf
        use_own = torch.rand_like(tgt_in_d.float()) < samp_p
        use_own[:, 0] = False                     # SOS stays teacher
        use_own = use_own & (tgt_in_d != PAD_ID)  # pads stay pads
        tgt_mixed = torch.where(use_own, pred_ids, tgt_in_d)
        logits, mem = model(src_d, tgt_mixed, tgt_mixed == PAD_ID, src_m,
                            mem=mem, return_mem=True)
    elif need_mem:
        logits, mem = model(src_d, tgt_in_d, pad_d, src_m, return_mem=True)
    elif variable:
        logits, mem = model(src_d, tgt_in_d, pad_d, src_m), None
    else:
        logits, mem = model(src_d, tgt_in_d, pad_d), None
    loss_ce = crit(logits.reshape(-1, len(tok)), tgt_out_d.reshape(-1))
    loss, lctc_v, nvalid, nskip = loss_ce, None, 0, 0
    if ctc_crit is not None:
        # TxtCTC over the transcript's own token ids (no SOS/EOS).
        # Skips takes whose frontend token count can't host the target.
        tgts = [tok.encode(t, add_markers=False) for t in b["texts"]]
        in_len = model.frontend_lens(b["src_lens"].to(device))
        tg_len = torch.tensor([len(t) for t in tgts], dtype=torch.long)
        ok = (in_len.cpu() >= tg_len) & (tg_len > 0)
        nskip = int((~ok).sum())
        if ok.any():
            ok_d = ok.to(device)
            flat = torch.tensor(
                [i for t, k in zip(tgts, ok.tolist()) for i in (t if k else [])],
                dtype=torch.long, device=device)
            lctc = ctc_crit(model.ctc_log_probs(mem[ok_d]),
                            flat, in_len[ok_d].to(torch.long),
                            tg_len[ok].to(device))
            loss = loss + ctc_weight * lctc
            lctc_v, nvalid = float(lctc.detach()), int(ok.sum())
    return logits, loss, lctc_v, nvalid, nskip


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
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader worker processes for CPU feeding (0=main process only). "
                         "6 is a good M1 Pro value; SSD speed only matters with workers to use it.")
    ap.add_argument("--tokenizer", default="tokenizer.json",
                    help="tokenizer file (default GH vocab; use tokenizer_all.json for GH+multiview runs)")
    ap.add_argument("--data", default="",
                    help="split prefix: '' = train/val (GH only), 'all' = all_train/all_val (GH+multiview), 'mv' = mv_train/mv_val")
    ap.add_argument("--tok-type", default="word", choices=["word", "bpe"],
                    help="word = tokenizer.py vocab, bpe = bpe_tokenizer.py subwords (pairs with --tokenizer bpe file)")
    ap.add_argument("--ctc-weight", type=float, default=0.0,
                    help="TxtCTC weight on top of CE (0 = off). 0.3 is the literature value. Variable path only.")
    ap.add_argument("--sched-samp", type=float, default=0.0,
                    help="scheduled-sampling target prob of feeding own prediction instead of teacher (0 = off, teammate uses 0.8)")
    ap.add_argument("--sched-ramp", type=int, default=50,
                    help="epochs to ramp sampling prob 0 -> target (linear)")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed for init/shuffle/augment/sampling (None = legacy unseeded). "
                         "Set it for paper reproducibility; logged to run<TAG>.json.")
    args = ap.parse_args()

    import random
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed % 2**32)
        torch.manual_seed(args.seed)

    device = get_device()
    print("device:", device)
    from dataset import KIND_DIM
    from preprocess import T_MAX
    if args.variable and args.kind in ("pos", "face"):
        args.kind = {"pos": "pos256", "face": "face256"}[args.kind]
        print(f"variable path: kind auto-upgraded to {args.kind}")
    in_dim = args.in_dim or KIND_DIM[args.kind]
    variable = args.variable or args.kind in ("pos256", "face256")
    tok = Tokenizer.load(args.tokenizer) if args.tok_type == "word" \
        else BPETokenizer.load(args.tokenizer)
    if args.ctc_weight > 0 and not variable:
        raise SystemExit("--ctc-weight needs the variable path (--variable/--kind *256)")
    tr_split, va_split = f"{args.data}_train".lstrip("_"), f"{args.data}_val".lstrip("_")
    if variable:
        from dataset import VAR_KINDS
        assert args.kind in VAR_KINDS, "--variable needs --kind pos256|face256"
        train_ds = SignDatasetVar(tr_split, tok, augment_flag=args.augment,
                                  kind=args.kind)
        val_ds = SignDatasetVar(va_split, tok, kind=args.kind)
        train_loader = bucketed_loader(train_ds, batch=args.batch, shuffle=True,
                                       num_workers=args.workers)
        val_loader = bucketed_loader(val_ds, batch=args.batch, shuffle=False,
                                     num_workers=min(args.workers, 2))
    else:
        train_ds = SignDataset(tr_split, tok, augment_flag=args.augment, kind=args.kind)
        val_ds = SignDataset(va_split, tok, kind=args.kind)
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate,
                                  num_workers=args.workers,
                                  worker_init_fn=_seed_worker if args.workers > 0 else None)
        val_loader = DataLoader(val_ds, batch_size=args.batch, collate_fn=collate,
                                num_workers=min(args.workers, 2),
                                worker_init_fn=_seed_worker if args.workers > 0 else None)

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
           "max_frames": args.max_frames if variable else 64,
           "tokenizer": args.tokenizer, "tok_type": args.tok_type,
           "ctc_weight": args.ctc_weight, "sched_samp": args.sched_samp,
           "sched_ramp": args.sched_ramp}
    REPO = Path(__file__).resolve().parent
    CKPT = REPO / "models" / "checkpoints"
    CKPT.mkdir(parents=True, exist_ok=True)
    LOGS = REPO / "logs"
    LOGS.mkdir(parents=True, exist_ok=True)
    best_pt, last_pt = (str(CKPT / f"best{args.tag}.pt"),
                        str(CKPT / f"last{args.tag}.pt"))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)
    ctc_crit = (nn.CTCLoss(blank=PAD_ID, zero_infinity=True)
                if args.ctc_weight > 0 else None)
    total_steps = args.epochs * len(train_loader)
    sched = None
    if args.warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-3,
                                                   total_iters=args.warmup_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, total_steps - args.warmup_steps))
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, [warmup, cosine], milestones=[args.warmup_steps])

    log_path = str(LOGS / f"train{args.tag}.log")
    csv_path = str(LOGS / f"metrics{args.tag}.csv")
    new_run = not (args.resume and Path(log_path).exists())
    log = open(log_path, "a")
    csv_new = not Path(csv_path).exists()
    csvf = open(csv_path, "a", newline="")
    csvw = csv.writer(csvf)
    if csv_new:
        csvw.writerow(["ep", "clock", "train_loss", "train_acc", "train_ppl",
                       "train_tok", "val_loss", "val_ppl", "val_acc", "val_tok",
                       "val_bleu", "val_bleu_s5", "val_p1", "val_p2", "val_p3",
                       "val_p4", "val_bp", "val_sys_len", "val_ref_len",
                       "val_exact_n", "val_exact_frac",
                       "val_bleu_gh", "val_exact_gh", "val_n_gh",
                       "val_bleu_mv", "val_exact_mv", "val_n_mv",
                       "best_ep", "lr", "grad_avg", "grad_max",
                       "train_s", "val_s", "ep_s", "ctc", "ctc_skip",
                       "samp_p"])
        csvf.flush()
    best, bad, best_ep, start_ep = -1.0, 0, 0, 1
    bleu_hist = []
    if args.resume:
        ckpt = torch.load(last_pt, map_location=device)
        model.load_state_dict(ckpt["state"])
        opt.load_state_dict(ckpt["opt"])
        if sched is not None and ckpt.get("sched") is not None:
            sched.load_state_dict(ckpt["sched"])
        best, bad, start_ep = ckpt["best"], ckpt["bad"], ckpt["ep"] + 1
        bleu_hist = ckpt.get("bleu_hist", [])
        best_ep = ckpt.get("best_ep", 0)
        print(f"resumed at ep {start_ep} (best val BLEU {best:.1f} @ ep {best_ep})")
    import datetime
    print("ep/total | train_loss | val_loss | val_BLEU (smooth5) | exact | "
          "best (epochs-back) | pace | FINISH-ETA", flush=True)
    print("detail per epoch -> train<TAG>.log (samples) + metrics<TAG>.csv (all numbers, graph this)",
          flush=True)
    if new_run:
        log.write(f"# run: kind={args.kind} variable={variable} "
                  f"tok={args.tok_type}:{args.tokenizer} "
                  f"ctc_w={args.ctc_weight} samp={args.sched_samp}/{args.sched_ramp} "
                  f"enc={args.enc_layers} dec={args.dec_layers} "
                  f"frontend={args.frontend_layers if variable else '-'} "
                  f"batch={args.batch} lr={args.lr} warmup={args.warmup_steps} "
                  f"patience={args.patience} train={len(train_ds)} val={len(val_ds)}\n")
        log.flush()
    print(f"run: data={tr_split}/{va_split} tok={args.tokenizer} kind={args.kind} variable={variable} params={n_params/1e6:.1f}M "
          f"train={len(train_ds)} val={len(val_ds)} steps/ep={len(train_loader)} "
          f"total_steps={total_steps} -> {best_pt}/{last_pt} | {log_path} {csv_path}",
          flush=True)
    import json
    import sacrebleu
    run_cfg = {"args": vars(args), "model_cfg": cfg, "n_train": len(train_ds),
               "n_val": len(val_ds), "params_M": round(n_params / 1e6, 2),
               "torch": torch.__version__, "sacrebleu": sacrebleu.__version__,
               "device": str(device),
               "started": datetime.datetime.now().isoformat(timespec="seconds")}
    if new_run:
        json.dump(run_cfg, open(f"run{args.tag}.json", "w"), indent=2)
        print(f"config snapshot -> run{args.tag}.json", flush=True)
    ep_times = []
    run_t0 = time.time()
    for ep in range(start_ep, args.epochs + 1):
        model.train()
        tot, ntok, ncorr, nsamp, ctc_tot, ctc_n, ctc_skip = 0.0, 0, 0, 0, 0.0, 0, 0
        samp_p = (args.sched_samp * min(1.0, ep / max(1, args.sched_ramp))
                  if args.sched_samp > 0 else 0.0)
        grad_sum, grad_max, nb = 0.0, 0.0, 0
        t0 = time.time()
        last_pb = t0
        for i, b in enumerate(train_loader, 1):
            opt.zero_grad()
            src_d = b["src"].to(device)
            tgt_in_d = b["tgt_in"].to(device)
            tgt_out_d = b["tgt_out"].to(device)
            pad_d = b["pad_mask"].to(device)
            src_m = b["src_mask"].to(device) if variable else None
            need_mem = variable and (ctc_crit is not None or samp_p > 0)
            try:
                logits, loss, lctc_v, nvalid, nskip = _train_step(
                    model, src_d, tgt_in_d, tgt_out_d, pad_d, src_m,
                    need_mem, samp_p, variable, ctc_crit, tok, b,
                    args.ctc_weight, crit, device)
            except NotImplementedError as e:
                # Rare MPS auto-dispatch corner (tiny batch + heavy padding +
                # dropout): skip the batch rather than killing a 7h run.
                # Proven absent at batch-32 bucketed over 90k steps.
                print(f"  ep {ep} batch {i}: skipped ({str(e)[:80]})", flush=True)
                continue
            if lctc_v is not None:
                ctc_tot += lctc_v * nvalid
                ctc_n += nvalid
            ctc_skip += nskip
            loss.backward()
            gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            gn = float(gn)
            opt.step()
            if sched is not None:
                sched.step()
            nt = int((tgt_out_d != PAD_ID).sum())
            tot += loss.item() * nt
            ntok += nt
            ncorr += int(((logits.argmax(-1) == tgt_out_d) &
                          (tgt_out_d != PAD_ID)).sum())
            nsamp += len(b["stems"])
            grad_sum += gn
            grad_max = max(grad_max, gn)
            nb += 1
            now = time.time()
            if now - last_pb > 180 and i < len(train_loader):
                last_pb = now
                el = now - t0
                eta_ep = el / i * (len(train_loader) - i)
                print(f"  ep {ep}: {i}/{len(train_loader)} batches "
                      f"({nsamp} clips, {nsamp/max(el,1e-3):.0f}/s), "
                      f"~{eta_ep/60:.0f}m left in epoch", flush=True)
        train_time = time.time() - t0
        tr_loss = tot / max(1, ntok)
        tr_acc = ncorr / max(1, ntok)
        vm = val_metrics(model, val_loader, tok, device, variable, crit)
        bleu_hist.append(vm["val_bleu"])
        s5 = sum(bleu_hist[-5:]) / min(5, len(bleu_hist))
        lr_now = opt.param_groups[0]["lr"]
        ep_s = train_time + vm["time"]
        ep_times.append(ep_s)
        avg_ep = sum(ep_times) / len(ep_times)
        left_h = avg_ep * (args.epochs - ep) / 3600
        finish = (datetime.datetime.now() +
                  datetime.timedelta(seconds=avg_ep * (args.epochs - ep)))
        improved = vm["val_bleu"] > best
        grad_avg = grad_sum / max(1, nb)  # csv only; console stays lean
        # console: only the decision numbers. everything else -> csv (graph it).
        line = (f"ep {ep}/{args.epochs} | train {tr_loss:.2f} | val {vm['loss']:.2f} | "
                f"BLEU {vm['val_bleu']:.1f} (s5 {s5:.1f}) | exact {vm['exact']}/{vm['n']} | "
                f"best {best:.1f} ({bad} back) | {ep_s/60:.0f}m/ep | "
                f"done ~{finish.strftime('%a %H:%M')} ({left_h:.1f}h left)")
        print(line, flush=True)
        log.write(line + "\n")
        for p, r in vm["samples"][:5]:
            log.write(f"  ex pred={p!r} true={r!r}\n")
        log.flush()
        clock = datetime.datetime.now().isoformat(timespec="seconds")
        csvw.writerow([ep, clock, f"{tr_loss:.4f}", f"{tr_acc:.4f}",
                       f"{math.exp(min(tr_loss, 20)):.2f}", ntok,
                       f"{vm['loss']:.4f}", f"{math.exp(min(vm['loss'], 20)):.2f}",
                       f"{vm['acc']:.4f}", vm["n_tok"],
                       f"{vm['val_bleu']:.2f}", f"{s5:.2f}",
                       f"{vm['val_p1']:.2f}", f"{vm['val_p2']:.2f}",
                       f"{vm['val_p3']:.2f}", f"{vm['val_p4']:.2f}",
                       f"{vm['val_bp']:.4f}", vm["val_sys_len"], vm["val_ref_len"],
                       vm["exact"], f"{vm['exact']/max(1,vm['n']):.4f}",
                       f"{vm['bleu_gh']:.2f}", vm["exact_gh"], vm["n_gh"],
                       f"{vm['bleu_mv']:.2f}", vm["exact_mv"], vm["n_mv"],
                       best_ep, f"{lr_now:.3e}",
                       f"{grad_avg:.3f}", f"{grad_max:.3f}",
                       f"{train_time:.0f}", f"{vm['time']:.0f}",
                       f"{ep_s:.0f}",
                       f"{ctc_tot/max(1,ctc_n):.4f}" if ctc_n else "-",
                       ctc_skip, f"{samp_p:.3f}"])
        if ctc_crit is not None or samp_p > 0:
            log.write(f"  ctc={ctc_tot/max(1,ctc_n):.3f} (n={ctc_n} skip={ctc_skip}) "
                      f"samp_p={samp_p:.3f}\n")
            log.flush()
        csvf.flush()
        if improved:
            best, bad, best_ep = vm['val_bleu'], 0, ep
            torch.save({"cfg": cfg, "state": model.state_dict()}, best_pt)
            print(f"  new best {best:.1f} @ ep {ep} -> {best_pt}", flush=True)
        else:
            bad += 1
        torch.save({"ep": ep, "best": best, "bad": bad, "best_ep": best_ep,
                    "bleu_hist": bleu_hist,
                    "state": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict() if sched else None}, last_pt)
        if bad >= args.patience:
            print(f"early stop at ep {ep}, best val BLEU {best:.1f} "
                  f"({(time.time()-run_t0)/3600:.1f}h total)")
            break
    print(f"done. best val BLEU {best:.1f} -> {best_pt}")


if __name__ == "__main__":
    main()
