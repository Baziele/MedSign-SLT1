"""Stage 1: contrastive pose<->text pretraining (phase-1 pipeline).

Usage:
  .venv312/bin/python -m pipeline.contrastive --tag v1 [options]

Trains VideoText (pose tower + BART text tower) with symmetric InfoNCE plus
the SignCL anti-density video-video term. Data: SignTalk gh+mv (all_train.csv)
+ capped GSL dictionary share; every epoch resamples a controlled mixture by
unique text group (--mv-frac/--gsl-frac) and logs the exact composition.
Checkpoints: models/checkpoints/pose_text_{best,last}_<TAG>.pt
Artifacts:   logs/contrastive_<TAG>/{run.json,epochs.csv,steps.jsonl,train.log}
"""
import argparse
import datetime as dt
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))

from torch.utils.data import DataLoader

from dataset import KIND_DIM, _seed_worker
from pipeline import data as D
from pipeline.logkit import RunLog
from pipeline.pose_text import VideoText, cl_losses


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode_pool(model, rows, tok, device, kind, batch=64):
    """Encode every row's video view + text (eval mode). Returns V, T, texts, stems."""
    ds = D.EncodeDataset(rows, kind)
    model.eval()
    V, T, texts, stems = [], [], [], []
    for s in range(0, len(ds), batch):
        chunk = list(range(s, min(s + batch, len(ds))))
        items = [ds[i] for i in chunk]
        src = torch.nn.utils.rnn.pad_sequence(
            [it["src"] for it in items], batch_first=True)
        lens = torch.tensor([len(it["src"]) for it in items])
        mask = torch.arange(src.size(1)).unsqueeze(0) >= lens.unsqueeze(1)
        with torch.no_grad():
            V.append(model.pose.embed(src.to(device), mask.to(device)).cpu())
        tobj = tok([it["text"] for it in items], padding=True, truncation=True,
                   max_length=48, return_tensors="pt")
        with torch.no_grad():
            T.append(model.text.embed(tobj["input_ids"].to(device),
                                      tobj["attention_mask"].to(device)).cpu())
        texts += [it["text"] for it in items]
        stems += [it["stem"] for it in items]
    return torch.cat(V), torch.cat(T), texts, stems


class PairCollator:
    """Module-level (picklable under macOS spawn): pads src/src2 to local max
    and tokenizes the batch texts with the HF tokenizer."""

    def __init__(self, tok):
        self.tok = tok

    def __call__(self, batch):
        lens = torch.tensor([len(b["src"]) for b in batch])
        lens2 = torch.tensor([len(b["src2"]) for b in batch])
        src = torch.nn.utils.rnn.pad_sequence([b["src"] for b in batch],
                                              batch_first=True)
        src2 = torch.nn.utils.rnn.pad_sequence([b["src2"] for b in batch],
                                               batch_first=True)
        smask = torch.arange(src.size(1)).unsqueeze(0) >= lens.unsqueeze(1)
        s2mask = torch.arange(src2.size(1)).unsqueeze(0) >= lens2.unsqueeze(1)
        toks = self.tok([b["text"] for b in batch], padding=True,
                        truncation=True, max_length=48, return_tensors="pt")
        return {**toks, "src": src, "src2": src2, "src_mask": smask,
                "src2_mask": s2mask,
                "texts": [b["text"] for b in batch],
                "stems": [b["stem"] for b in batch]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--clips-per-epoch", type=int, default=20000,
                    help="clips per epoch; mixture per --mv-frac/--gsl-frac")
    ap.add_argument("--lr", type=float, default=3e-4, help="pose tower LR")
    ap.add_argument("--text-lr", type=float, default=1e-4, help="BART tower LR")
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--mv-frac", type=float, default=0.5,
                    help="share of epoch slots from multiview text groups")
    ap.add_argument("--gsl-frac", type=float, default=0.1,
                    help="share of epoch slots from GHS dictionary (capped)")
    ap.add_argument("--vv-weight", type=float, default=0.1,
                    help="SignCL video-video anti-density weight")
    ap.add_argument("--text-model", default="facebook/bart-base")
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--enc-layers", type=int, default=4)
    ap.add_argument("--frontend-layers", type=int, default=2, choices=[2, 3])
    ap.add_argument("--kind", default="face256")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--log-steps", type=int, default=200)
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed % 2**32)
    torch.manual_seed(args.seed)
    device = get_device()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.text_model)

    # ---- data pools -----------------------------------------------------
    main_rows = D.load_main_rows("all_train")
    gsl_rows = D.load_gsl_rows()
    rows = main_rows + gsl_rows
    groups = D.build_groups(rows)
    gsrc = D.groups_by_src(rows, groups)
    val_rows = D.load_main_rows("all_val")
    val_groups = D.build_groups(val_rows)
    lens_all = [D.len_of(r, args.kind) for r in rows]

    pose_kw = dict(in_dim=KIND_DIM[args.kind], d_model=args.d_model,
                   enc_layers=args.enc_layers,
                   frontend_layers=args.frontend_layers, proj_dim=args.proj_dim)
    model = VideoText(pose_kw, args.text_model, args.proj_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"pools: groups gh={len(gsrc['gh'])} mv={len(gsrc['mv'])} "
          f"gsl={len(gsrc['gsl'])} | rows={len(rows)} | val={len(val_rows)} "
          f"({len(val_groups)} val texts) | params {n_params/1e6:.1f}M | "
          f"device={device}", flush=True)

    pose_params = list(model.pose.parameters()) + [model.scale]
    text_params = list(model.text.parameters())
    opt = torch.optim.AdamW([
        {"params": pose_params, "lr": args.lr},
        {"params": text_params, "lr": args.text_lr},
    ])
    sched = None
    if args.warmup_steps > 0:
        from torch.optim.lr_scheduler import (CosineAnnealingLR, LinearLR,
                                              SequentialLR)
        total_steps = args.epochs * (args.clips_per_epoch // args.batch + 1)
        warmup = LinearLR(opt, start_factor=1e-3,
                          total_iters=args.warmup_steps)
        cosine = CosineAnnealingLR(
            opt, T_max=max(1, total_steps - args.warmup_steps))
        sched = SequentialLR(opt, [warmup, cosine],
                             milestones=[args.warmup_steps])

    best_pt = ROOT / "models" / "checkpoints" / f"pose_text_best_{args.tag}.pt"
    last_pt = ROOT / "models" / "checkpoints" / f"pose_text_last_{args.tag}.pt"

    manifest = {
        "groups": {s: len(gsrc[s]) for s in gsrc},
        "total_clips": len(rows),
        "clips_per_epoch": args.clips_per_epoch,
        "fractions": {"mv": args.mv_frac, "gsl": args.gsl_frac,
                      "gh": round(1 - args.mv_frac - args.gsl_frac, 2)},
        "val_rows": len(val_rows), "val_texts": len(val_groups),
        "vocab": {"pose": None, "text": args.text_model},
    }
    cfg = {"cmd": "contrastive", "tag": args.tag, "device": str(device),
           "batch": args.batch, "clips_per_epoch": args.clips_per_epoch,
           "lr": args.lr, "text_lr": args.text_lr,
           "warmup_steps": args.warmup_steps, "mv_frac": args.mv_frac,
           "gsl_frac": args.gsl_frac, "vv_weight": args.vv_weight,
           "text_model": args.text_model, "proj_dim": args.proj_dim,
           "pose_kw": pose_kw, "kind": args.kind, "epochs": args.epochs,
           "seed": args.seed, "n_params_M": round(n_params / 1e6, 2)}
    rl = RunLog(f"contrastive_{args.tag}", cfg, extra=manifest)
    ep_cols = ["ep", "clock", "train_s", "val_s", "loss", "l_vt", "l_tv",
               "l_vv", "sim_pos", "sim_neg", "r1_v2t", "r5_v2t", "r1_t2v",
               "r5_t2v", "lr_pose", "lr_text", "scale", "grad_avg",
               "grad_max", "n_batches", "clips_s"]
    fresh = not (args.resume and (rl.dir / "epochs.csv").exists())
    if fresh:
        rl.epoch_header(ep_cols)
    best, bad, start_ep = -1.0, 0, 1
    if args.resume and last_pt.exists():
        ck = torch.load(last_pt, map_location=device)
        model.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        if sched is not None and ck.get("sched") is not None:
            sched.load_state_dict(ck["sched"])
        best, start_ep = ck["best"], ck["ep"] + 1
        print(f"resumed at ep {start_ep} (best r@1 avg {best:.4f})", flush=True)
    else:
        rl.line(f"run: mix(mv/gsl/gh)={args.mv_frac}/{args.gsl_frac}/"
                f"{1 - args.mv_frac - args.gsl_frac:.2f} clips/ep="
                f"{args.clips_per_epoch} batch={args.batch} vv="
                f"{args.vv_weight} {n_params/1e6:.1f}M params -> "
                f"{best_pt.name}")

    collate = PairCollator(tok)
    t_run, step = time.time(), 0
    for ep in range(start_ep, args.epochs + 1):
        rng = random.Random(args.seed * 1000 + ep)
        idx = D.mix_index(rows, groups, args.clips_per_epoch,
                          {"mv": args.mv_frac, "gsl": args.gsl_frac}, rng)
        mix = D.index_stats(rows, idx)
        ep_rows = [rows[i] for i in idx]
        ep_lens = [lens_all[i] for i in idx]
        ds = D.PairDataset(ep_rows, args.kind)
        batches = D.bucketed_batches(ds, ep_lens, args.batch, shuffle=True,
                                     rng=rng)
        loader = DataLoader(ds, batch_sampler=batches, collate_fn=collate,
                            num_workers=args.workers,
                            worker_init_fn=_seed_worker if args.workers > 0
                            else None)

        model.train()
        t0, agg, gsum, gmax, nb = time.time(), {}, 0.0, 0.0, 0
        for i, b in enumerate(loader, 1):
            out = model(b["src"].to(device), b["src_mask"].to(device),
                        b["src2"].to(device), b["src2_mask"].to(device),
                        b["input_ids"].to(device),
                        b["attention_mask"].to(device))
            losses = cl_losses(out, args.vv_weight)
            with torch.no_grad():
                C = out["v"] @ out["t"].t()
                n = C.shape[0]
                diag = torch.arange(n, device=C.device)
                sim_pos = C[diag].mean()
                sim_neg = (C.sum() - C.diag().sum()) / max(1, n * n - n)
            opt.zero_grad()
            losses["loss"].backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            opt.step()
            if sched is not None:
                sched.step()
            for k in ("loss", "l_vt", "l_tv", "l_vv"):
                v = losses.get(k)
                agg[k] = agg.get(k, 0.0) + (float(v.detach())
                                            if v is not None else 0.0)
            agg["sim_pos"] = agg.get("sim_pos", 0.0) + float(sim_pos)
            agg["sim_neg"] = agg.get("sim_neg", 0.0) + float(sim_neg)
            gsum += gn
            gmax = max(gmax, gn)
            nb += 1
            step += 1
            if step % args.log_steps == 0:
                el = time.time() - t0
                rl.step({
                    "step": step, "ep": ep,
                    "loss": agg["loss"] / nb, "l_vt": agg["l_vt"] / nb,
                    "l_tv": agg["l_tv"] / nb,
                    "l_vv": agg["l_vv"] / max(1, nb),
                    "sim_pos": agg["sim_pos"] / nb,
                    "sim_neg": agg["sim_neg"] / nb,
                    "scale": float(model.scale.detach()),
                    "lr_pose": opt.param_groups[0]["lr"],
                    "lr_text": opt.param_groups[1]["lr"],
                    "grad": gn,
                    "clips_s": nb * args.batch / max(el, 1e-3),
                    "epoch_mix": mix})
        train_s = time.time() - t0

        # ---- retrieval probe (full val pool) -----------------------------
        t1 = time.time()
        V, T, tx, st = encode_pool(model, val_rows, tok, device, args.kind)
        from pipeline.pose_text import retrieval_metrics
        vm = retrieval_metrics(V, T)
        val_s = time.time() - t1
        score = (vm["r1_v2t"] + vm["r1_t2v"]) / 2

        clock = dt.datetime.now().isoformat(timespec="seconds")
        rl.epoch([ep, clock, f"{train_s:.0f}", f"{val_s:.0f}",
                  f"{agg['loss']/nb:.4f}", f"{agg['l_vt']/nb:.4f}",
                  f"{agg['l_tv']/nb:.4f}", f"{agg['l_vv']/max(1,nb):.4f}",
                  f"{vm['sim_pos']:.4f}", f"{vm['sim_neg']:.4f}",
                  f"{vm['r1_v2t']:.4f}", f"{vm['r5_v2t']:.4f}",
                  f"{vm['r1_t2v']:.4f}", f"{vm['r5_t2v']:.4f}",
                  f"{opt.param_groups[0]['lr']:.3e}",
                  f"{opt.param_groups[1]['lr']:.3e}",
                  f"{float(model.scale.detach()):.3f}",
                  f"{gsum/max(1,nb):.3f}", f"{gmax:.3f}", nb,
                  f"{nb*args.batch/train_s:.1f}"])
        left_h = (train_s + val_s) * (args.epochs - ep) / 3600
        done_at = (dt.datetime.now() +
                   dt.timedelta(seconds=(train_s + val_s) * (args.epochs - ep)))
        improved = score > best
        if improved:
            best, bad = score, 0
        else:
            bad += 1
        rl.line(f"ep {ep}/{args.epochs} | loss {agg['loss']/nb:.2f} "
                f"(vt {agg['l_vt']/nb:.2f} tv {agg['l_tv']/nb:.2f} "
                f"vv {agg['l_vv']/max(1,nb):.2f}) | "
                f"R@1 v2t {vm['r1_v2t']*100:.1f}% t2v {vm['r1_t2v']*100:.1f}% "
                f"| R@5 {vm['r5_v2t']*100:.0f}%/{vm['r5_t2v']*100:.0f}% | "
                f"sim +{vm['sim_pos']:.2f}/{vm['sim_neg']:.2f} | "
                f"{train_s/60:.0f}m+{val_s/60:.0f}m val | "
                f"~{done_at.strftime('%a %H:%M')} ({left_h:.1f}h left)"
                + (" | NEW BEST" if improved else ""))
        for j in (0, len(tx) // 5, len(tx) // 2, 3 * len(tx) // 5,
                  len(tx) - 2):
            S = V[j] @ T.t()
            nn_id = int(S.argmax())
            rl.log.write(f"  ex video {st[j][:34]} -> '{tx[nn_id]}' "
                         f"(true: '{tx[j]}')\n")
        rl.log.flush()
        if improved:
            torch.save({"ep": ep, "score": score, "args": vars(args),
                        "state": model.state_dict()}, best_pt)
        torch.save({"ep": ep, "best": best, "state": model.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict() if sched else None}, last_pt)
        if bad >= args.patience:
            rl.line(f"early stop ep {ep}; best r@1 avg {best:.4f}")
            break
    rl.line(f"done {(time.time()-t_run)/3600:.1f}h; best r@1 avg {best:.4f}; "
            f"restore pose tower for stage 2 from: {best_pt}")


if __name__ == "__main__":
    main()
