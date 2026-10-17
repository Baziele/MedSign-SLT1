"""Stage 2: translation training from a stage-1 pose checkpoint (phase-1).

Usage:
  .venv312/bin/python -m pipeline.stage2 --tag p1t1 \
      --from-pose models/checkpoints/pose_text_best_v1.pt

Same trainer recipe as train.py --variable (SignTransformerV2, CTC-over-BPE
w=0.3, scheduled sampling) with three phase-1 changes, all logged:
  1. pose tower (frontend/pos/encoder) initialized from the contrastive ckpt
     with a lower LR (--enc-lr-scale); decoder+heads start fresh.
  2. multiview contribution capped: `--mv-takes` random takes per unique
     sentence per epoch (gh = all rows) -> stops the 500-sentence recitation.
  3. paper-grade logging under logs/stage2_<TAG>/{run.json,epochs.csv,
     steps.jsonl,train.log} with the exact per-epoch mix recorded.

Evaluation stays the untouched probes (GH unseen-text, MV signer-holdout) and
val selection is corpus BLEU, as in train.py, so take-1 vs take-2 and the old
all_bpe_v1 baseline are directly comparable.
"""
import argparse
import datetime as dt
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]

from dataset import VAR_KINDS, collate_var
from dataset import KIND_DIM, _seed_worker
from model import SignTransformerV2
from tokenizer import PAD_ID
from train import _train_step, val_metrics

from pipeline import data as D
from pipeline.logkit import RunLog


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pose_into(model, pose_ckpt_path, dev):
    """Copy pose tower weights (pose.* from the stage-1 VideoText state) into
    the matching model.frontend/pos/encoder keys. Returns match report."""
    ck = (pose_ckpt_path if isinstance(pose_ckpt_path, dict)
          else torch.load(pose_ckpt_path, map_location=dev))
    src = ck["state"]
    pose_state = {k[len("pose."):]: v for k, v in src.items()
                  if k.startswith("pose.")}
    allowed = model.state_dict().keys()
    take = {k: v for k, v in pose_state.items() if k in allowed}
    dropped = sorted(set(pose_state) - set(take))
    model_state = model.state_dict()
    model_state.update(take)
    model.load_state_dict(model_state)
    n_tensors = len(take)
    report = {"src": str(pose_ckpt_path), "ep": ck.get("ep"),
              "score": ck.get("score"), "tensors": n_tensors,
              "skipped": dropped}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--from-pose", required=True,
                    help="stage-1 VideoText checkpoint (pose_text_best_*.pt)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--enc-lr-scale", type=float, default=0.1,
                    help="LR multiplier for the pretrained pose tower")
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--mv-takes", type=int, default=4,
                    help="multiview takes per unique sentence per epoch (cap)")
    ap.add_argument("--no-jw", action="store_true",
                    help="exclude JW verse data (main source by default)")
    ap.add_argument("--jw-source", default="nwt-ghs",
                    help="which JW set to index as the main source")
    ap.add_argument("--tokenizer", default="bpe_all_8k.json.wrap")
    ap.add_argument("--ctc-weight", type=float, default=0.3)
    ap.add_argument("--sched-samp", type=float, default=0.8)
    ap.add_argument("--sched-ramp", type=int, default=50)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--enc-layers", type=int, default=4)
    ap.add_argument("--dec-layers", type=int, default=4)
    ap.add_argument("--frontend-layers", type=int, default=2, choices=[2, 3])
    ap.add_argument("--kind", default="face256")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--log-steps", type=int, default=200)
    ap.add_argument("--limit-clips", type=int, default=0,
                    help="debug: cap per-epoch rows (0 = full epoch)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed % 2**32)
    torch.manual_seed(args.seed)
    device = get_device()
    from bpe_tokenizer import BPETokenizer
    tok = BPETokenizer.load(args.tokenizer)

    # ---- data -------------------------------------------------------------
    train_rows = D.load_main_rows("all_train")
    jw_rows = [] if args.no_jw else D.load_jw_rows(args.jw_source)
    pool_rows = train_rows + jw_rows
    val_rows = D.load_main_rows("all_val")
    train_groups = D.build_groups(pool_rows)
    gsrc = D.groups_by_src(pool_rows, train_groups)
    src_by_stem = {r["stem"]: r["src"] for r in pool_rows}
    val_groups = D.build_groups(val_rows)
    lens_all = [D.len_of(r, args.kind) for r in train_rows] + \
               [D.len_of(r, args.kind) for r in jw_rows]
    src_line = " | ".join(
        f"{s} {sum(1 for r in pool_rows if r['src'] == s)} rows/"
        f"{len(gsrc.get(s, []))} texts" for s in sorted({r['src'] for r in pool_rows}))
    print(f"data: {src_line} | val {len(val_rows)} rows/{len(val_groups)} texts",
          flush=True)

    model = SignTransformerV2(len(tok), in_dim=KIND_DIM[args.kind],
                              d_model=args.d_model,
                              enc_layers=args.enc_layers,
                              dec_layers=args.dec_layers,
                              frontend_layers=args.frontend_layers).to(device)
    init_report = load_pose_into(model, args.from_pose, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"pose tower init: {init_report['tensors']} tensors "
          f"(stage-1 ep {init_report['ep']}); params {n_params/1e6:.1f}M; "
          f"device={device}", flush=True)

    # ---- LRs: pretrained pose tower slow, decoder normal -------------------
    pose_param_ids = {id(p) for p in list(model.frontend.parameters()) +
                      list(model.encoder.parameters())}
    pose_params = [p for p in model.parameters() if id(p) in pose_param_ids]
    fresh_params = [p for p in model.parameters() if id(p) not in pose_param_ids]
    opt = torch.optim.AdamW([
        {"params": pose_params, "lr": args.lr * args.enc_lr_scale},
        {"params": fresh_params, "lr": args.lr},
    ])
    crit = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)
    ctc_crit = (nn.CTCLoss(blank=PAD_ID, zero_infinity=True)
                if args.ctc_weight > 0 else None)

    gh_row_n = sum(len(train_groups[k]) for k in gsrc.get("gh", []))
    mv_row_n = args.mv_takes * len(gsrc.get("mv", []))
    total_steps = args.epochs * ((gh_row_n + mv_row_n) // args.batch + 1)
    sched = None
    if args.warmup_steps > 0:
        from torch.optim.lr_scheduler import (CosineAnnealingLR, LinearLR,
                                              SequentialLR)
        warmup = LinearLR(opt, start_factor=1e-3,
                          total_iters=args.warmup_steps)
        cosine = CosineAnnealingLR(
            opt, T_max=max(1, total_steps - args.warmup_steps))
        sched = SequentialLR(opt, [warmup, cosine],
                             milestones=[args.warmup_steps])

    # ---- eval loader (fixed, full val set) --------------------------------
    val_ds = D.VarRowDataset(val_rows, tok, kind=args.kind)
    val_lens = [D.len_of(r, args.kind) for r in val_rows]
    val_batches = D.bucketed_batches(val_ds, val_lens, args.batch,
                                     shuffle=False)
    val_loader = DataLoader(val_ds, batch_sampler=val_batches,
                            collate_fn=collate_var, num_workers=2)

    cfg = {"vocab": len(tok), "kind": args.kind,
                       "in_dim": KIND_DIM[args.kind],
                       "enc": args.enc_layers, "dec": args.dec_layers,
                       "variable": True,
                       "frontend_layers": args.frontend_layers,
                       "max_frames": 256, "tokenizer": args.tokenizer,
                       "tok_type": "bpe", "ctc_weight": args.ctc_weight,
                       "sched_samp": args.sched_samp,
                       "sched_ramp": args.sched_ramp, "stage2": True}
    best_pt = ROOT / "models" / "checkpoints" / f"best_{args.tag}.pt"
    last_pt = ROOT / "models" / "checkpoints" / f"last_{args.tag}.pt"

    manifest = {"from_pose": init_report,
                "train_texts": {s: len(gsrc[s]) for s in gsrc},
                "train_rows": {s: sum(len(train_groups[k]) for k in gsrc[s])
                               for s in gsrc},
                "mv_takes_per_epoch": args.mv_takes,
                "jw_source": None if args.no_jw else args.jw_source,
                "jw_rows": {s: sum(1 for r in jw_rows if r["src"] == s)
                            for s in {r["src"] for r in jw_rows}},
                "val_rows": len(val_rows), "val_texts": len(val_groups)}
    rl = RunLog(f"stage2_{args.tag}", cfg, extra=manifest)
    ep_cols = ["ep", "clock", "train_s", "val_s", "tr_loss", "tr_acc",
               "ctc", "ctc_skip", "samp_p", "val_loss", "val_acc",
               "val_bleu", "val_bleu_s5", "exact_n", "exact_frac",
               "bleu_gh", "exact_gh", "n_gh", "bleu_mv", "exact_mv", "n_mv",
               "lr", "lr_pose", "grad_avg", "grad_max", "n_batches",
               "tok_s", "best_ep"]
    fresh_log = not (args.resume and (rl.dir / "epochs.csv").exists())
    if fresh_log:
        rl.epoch_header(ep_cols)
    best, bad, best_ep, start_ep = -1.0, 0, 0, 1
    bleu_hist = []
    if args.resume and last_pt.exists():
        ck = torch.load(last_pt, map_location=device)
        model.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        if sched is not None and ck.get("sched") is not None:
            sched.load_state_dict(ck["sched"])
        best, bad, best_ep, start_ep = (ck["best"], ck["bad"],
                                        ck["best_ep"], ck["ep"] + 1)
        bleu_hist = ck.get("bleu_hist", [])
        print(f"resumed ep {start_ep} (best {best:.2f} @ {best_ep})",
              flush=True)
    else:
        rl.line(f"run: init pose[]={init_report['tensors']} tensors "
                f"(stage1 ep {init_report['ep']}), dec/heads fresh | "
                f"mv takes/sentence={args.mv_takes} | tok=bpe_8k | "
                f"ctc={args.ctc_weight} samp={args.sched_samp}/"
                f"{args.sched_ramp} | enc_lr={args.lr*args.enc_lr_scale:.2e} "
                f"dec_lr={args.lr:.2e} | {n_params/1e6:.1f}M params")

    run_t0 = time.time()
    for ep in range(start_ep, args.epochs + 1):
        # ---- per-epoch controlled mix -------------------------------------
        rng = random.Random(args.seed * 1000 + ep)
        idx = D.stage2_index(pool_rows, train_groups, args.mv_takes, rng)
        mix = D.index_stats(pool_rows, idx)
        if args.limit_clips:
            rng.shuffle(idx)
            idx = idx[:args.limit_clips]
            mix = D.index_stats(pool_rows, idx)
        ep_rows = [pool_rows[i] for i in idx]
        ep_lens = [lens_all[i] for i in idx]
        ds = D.VarRowDataset(ep_rows, tok, kind=args.kind, augment_flag=True)
        train_batches = D.bucketed_batches(ds, ep_lens, args.batch,
                                           shuffle=True, rng=rng)
        loader = DataLoader(ds, batch_sampler=train_batches,
                            collate_fn=collate_var, num_workers=args.workers,
                            worker_init_fn=_seed_worker
                            if args.workers > 0 else None)

        model.train()
        samp_p = (args.sched_samp * min(1.0, ep / max(1, args.sched_ramp))
                  if args.sched_samp > 0 else 0.0)
        t0, tot, ntok, ncorr, ctc_tot, ctc_n, ctc_skip = (
            time.time(), 0.0, 0, 0, 0.0, 0, 0)
        gsum, gmax, nb = 0.0, 0.0, 0
        need_mem = ctc_crit is not None or samp_p > 0
        for i, b in enumerate(loader, 1):
            opt.zero_grad()
            src_d = b["src"].to(device)
            tgt_in_d = b["tgt_in"].to(device)
            tgt_out_d = b["tgt_out"].to(device)
            pad_d = b["pad_mask"].to(device)
            src_m = b["src_mask"].to(device)
            try:
                logits, loss, lctc_v, nvalid, nskip = _train_step(
                    model, src_d, tgt_in_d, tgt_out_d, pad_d, src_m,
                    need_mem, samp_p, True, ctc_crit, tok, b,
                    args.ctc_weight, crit, device)
            except NotImplementedError as e:
                # same MPS corner-case guard as train.py: skip, don't kill
                rl.line(f"ep {ep} batch {i}: skipped ({str(e)[:80]})",
                        console=False)
                continue
            if lctc_v is not None:
                ctc_tot += lctc_v * nvalid
                ctc_n += nvalid
            ctc_skip += nskip
            loss.backward()
            gn = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            opt.step()
            if sched is not None:
                sched.step()
            nt = int((tgt_out_d != PAD_ID).sum())
            tot += loss.item() * nt
            ntok += nt
            ncorr += int(((logits.argmax(-1) == tgt_out_d) &
                          (tgt_out_d != PAD_ID)).sum())
            gsum += gn
            gmax = max(gmax, gn)
            nb += 1
            if i % args.log_steps == 0 or i == len(loader):
                el = time.time() - t0
                bs = {}
                for s in b["stems"]:
                    src_tag = src_by_stem.get(s, "?")
                    bs[src_tag] = bs.get(src_tag, 0) + 1
                rl.step({"ep": ep, "batch": i, "loss": tot / max(1, ntok),
                         "ctc": ctc_tot / max(1, ctc_n) if ctc_n else None,
                         "ctc_skip": ctc_skip, "samp_p": samp_p,
                         "lr": opt.param_groups[1]["lr"],
                         "lr_pose": opt.param_groups[0]["lr"],
                         "grad": gn, "tok_s": ntok / max(el, 1e-3),
                         "epoch_mix": mix, "batch_src_counts": bs})
        train_s = time.time() - t0

        vm = val_metrics(model, val_loader, tok, device, True, crit)
        bleu_hist.append(vm["val_bleu"])
        s5 = sum(bleu_hist[-5:]) / min(5, len(bleu_hist))
        lr_now = opt.param_groups[1]["lr"]
        ep_s = train_s + vm["time"]
        left_h = ep_s * (args.epochs - ep) / 3600
        done_at = dt.datetime.now() + dt.timedelta(seconds=ep_s *
                                                   (args.epochs - ep))
        improved = vm["val_bleu"] > best
        if improved:
            best, bad, best_ep = vm["val_bleu"], 0, ep
        else:
            bad += 1
        rl.epoch([ep, dt.datetime.now().isoformat(timespec="seconds"),
                  f"{train_s:.0f}", f"{vm['time']:.0f}",
                  f"{tot/max(1,ntok):.4f}", f"{ncorr/max(1,ntok):.4f}",
                  f"{ctc_tot/max(1,ctc_n):.4f}" if ctc_n else "-",
                  ctc_skip, f"{samp_p:.3f}",
                  f"{vm['loss']:.4f}", f"{vm['acc']:.4f}",
                  f"{vm['val_bleu']:.2f}", f"{s5:.2f}", vm["exact"],
                  f"{vm['exact']/max(1,vm['n']):.4f}",
                  f"{vm['bleu_gh']:.2f}", vm["exact_gh"], vm["n_gh"],
                  f"{vm['bleu_mv']:.2f}", vm["exact_mv"], vm["n_mv"],
                  f"{lr_now:.3e}", f"{opt.param_groups[0]['lr']:.3e}",
                  f"{gsum/max(1,nb):.3f}", f"{gmax:.3f}", nb,
                  f"{ntok/train_s:.0f}", best_ep])
        mixstr = " + ".join(f"{k} {v}" for k, v in sorted(mix.items()))
        line = (f"ep {ep}/{args.epochs} | mix {mixstr} | "
                f"train {tot/max(1,ntok):.2f} | val {vm['loss']:.2f} | "
                f"BLEU {vm['val_bleu']:.1f} (s5 {s5:.1f}) gh "
                f"{vm['bleu_gh']:.1f} mv {vm['bleu_mv']:.1f} | "
                f"exact {vm['exact']}/{vm['n']} | "
                f"best {best:.1f} ({bad} back) | {ep_s/60:.0f}m/ep | "
                f"~{done_at.strftime('%a %H:%M')} ({left_h:.1f}h left)"
                + (" | NEW BEST" if improved else ""))
        rl.line(line)
        rl.samples(vm["samples"][:5])
        if ctc_n:
            rl.log.write(f"  ctc={ctc_tot/ctc_n:.3f} skip={ctc_skip} "
                         f"samp_p={samp_p:.3f}\n")
            rl.log.flush()
        if improved:
            torch.save({"cfg": cfg, "state": model.state_dict()}, best_pt)
            rl.line(f"  new best -> {best_pt.name}")
        torch.save({"ep": ep, "best": best, "bad": bad, "best_ep": best_ep,
                    "bleu_hist": bleu_hist, "cfg": cfg,
                    "state": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict() if sched else None}, last_pt)
        if bad >= args.patience:
            rl.line(f"early stop at ep {ep}, best {best:.1f} "
                    f"({(time.time()-run_t0)/3600:.1f}h)")
            break
    rl.line(f"done {(time.time()-run_t0)/3600:.1f}h; best val BLEU "
            f"{best:.2f} @ ep {best_ep} -> {best_pt}")


if __name__ == "__main__":
    main()
