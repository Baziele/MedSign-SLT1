# Phase-1 training pipeline (contrastive pretrain → translation)

Two stages, one pose tower handed over between them. Stage 1 learns a shared
pose↔text space; stage 2 trains the translation model initialized from that
tower. Take-1 uses the from-scratch decoder so improvements are attributable
against the `all_bpe_v1` baseline; take-2 (mBART decoder) is planned.

## Stage 1 — contrastive pretraining

```bash
.venv312/bin/python -m pipeline.contrastive --tag v1
```

- Model: pose tower (conv frontend + 4-layer encoder, same module layout as
  `SignTransformerV2`) + BART-family text tower (`--text-model`) + learnable
  temperature. Loss: symmetric InfoNCE + SignCL video-video term
  (`--vv-weight 0.1`).
- Data (`pipeline/data.py`):
  - `gh` — original SignTalk hospital sentences (all_train.csv),
  - `mv` — Medsign multiview: 75k takes, 493 unique texts,
  - `gsl` — GHS dictionary npz, capped by `--gsl-frac` (text exposure, not
    rows — before the cap it would be <10% of unique texts anyway).
- Each epoch is a fresh controlled mixture by **unique text group**
  (`--mv-frac 0.5`), logged per epoch; the exact selected clips are
  reproducible from `--seed` + epoch number.
- Best selection: average retrieval R@1 (video→text, text→video) over the
  untouched val pool (3,494 rows / 891 texts).
- Outputs: `models/checkpoints/pose_text_{best,last}_<TAG>.pt`;
  `logs/contrastive_<TAG>/{run.json, epochs.csv, steps.jsonl, train.log}`.

## Stage 2 — translation training (CTC + CE), from-scratch decoder

```bash
.venv312/bin/python -m pipeline.stage2 --tag p1t1 \
    --from-pose models/checkpoints/pose_text_best_v1.pt
```

- `SignTransformerV2` (4enc/4dec, frontend 4x, face256), decoder + CTC/BPE
  head initialized fresh; `pose.*` weights (53 tensors) copied from the
  stage-1 checkpoint and trained at `--enc-lr-scale 0.1` of the decoder LR.
- Sampling: **all** gh rows + `--mv-takes` (default 4) random takes per
  multiview unique sentence per epoch — removes the recitation pressure while
  keeping signer/camera robustness. GSL is never a CE target.
- Recipe matches `all_bpe_v1` otherwise (CTC w=0.3, scheduled sampling 0→0.8
  over 50 ep, bpe_all_8k, face256) so deltas are attributable:
  1. pose-tower init from contrastive stage,
  2. mv-cap sampling.
- Selection: val corpus BLEU (same as train.py), probes: GH unseen-text and
  MV signer-holdout reported separately every epoch.
- Outputs: `models/checkpoints/{best,last}_<TAG>.pt`;
  `logs/stage2_<TAG>/{run.json, epochs.csv, steps.jsonl, train.log}`.

## Logging map (research paper hydration)

Every run directory contains:

| File | Granularity | Contents |
|---|---|---|
| `run.json` | once | full config, data manifest, git HEAD, versions |
| `epochs.csv` | per epoch | all decision + diagnostic numbers (graph sheet) |
| `steps.jsonl` | every N steps | losses, sim stats, LRs, grad norm, pace, epoch mix |
| `train.log` | per epoch | human-readable lines + sample predictions (tail this) |

Console prints only the epoch decision line. Debug caps: stage2
`--limit-clips N` (boot test without a full epoch).

## What's deliberately NOT in take 1

No mBART decoder (take 2), no EM/DTW refinement and no pseudo-glosses (that
phase belongs to the weakly-aligned long-video / YouTube data), no masked
pose modeling (optional, can bolt on later).

## Smoke test

```bash
.venv312/bin/python scripts/dev/smoke_phase1.py
```

CPU, real rows, no training: data plumbing, two-view pair forward, backprop,
pose-tower→translation key mapping (53 tensors, byte-identical).
