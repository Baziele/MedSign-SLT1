"""Shared data plumbing for the phase-1 pipeline (contrastive + stage2).

Row dicts carry a `src` tag so every sampler/stat can split by source:
  gh  = original SignTalk hospital sentences (all_train.csv, non-MV stems)
  mv  = Medsign multiview takes (all_train.csv, train_train_/test_test_ stems)
  gsl = GHS dictionary entries (data/gsl/gsl-dictionary/dictionary_clean.csv)

All sampling is index-list based: datasets hold the full pool, and each epoch
gets a fresh deterministic index list. The exact composition of every epoch is
logged (pipeline.logkit) so runs are reproducible and paper-reportable.
"""
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from dataset import FEAT_CACHE, VAR_KINDS, feat_var, augment_var, collate_var
from preprocess import SPLITS_DIR, clip_features_var

ROOT = Path(__file__).resolve().parents[1]
GSL_DIR = ROOT / "data" / "gsl" / "gsl-dictionary"


# ---------------------------------------------------------------- row loading

def load_main_rows(split="all_train"):
    """GH + multiview rows from the combined splits (GH stems are everything
    that is not a multiview take). Adds `src` in place."""
    rows = []
    with open(SPLITS_DIR / f"{split}.csv") as f:
        for r in csv.DictReader(f):
            stem = r["stem"]
            r["src"] = ("mv" if stem.startswith(("train_train_", "test_test_"))
                        else "gh")
            rows.append(r)
    return rows


def load_gsl_rows(path=GSL_DIR / "dictionary_clean.csv"):
    """Dictionary entries -> contrastive pairs (text = the word it teaches).
    Variants map to the same video; exact (stem, text) duplicates are dropped
    but synonym rows are kept: more positives per video is free signal.
    Metadata sometimes lists npz that never landed on disk -> skipped."""
    rows, seen, missing = [], set(), 0
    with open(path) as f:
        for r in csv.DictReader(f):
            stem = r["npz"].removesuffix(".npz")
            text = r["word"].strip().lower()
            if (stem, text) in seen:
                continue
            if not (GSL_DIR / "keypoints" / f"{stem}.npz").exists():
                missing += 1
                continue
            seen.add((stem, text))
            rows.append({"stem": stem, "text": text, "src": "gsl"})
    if missing:
        print(f"gsl: skipped {missing} rows whose npz is missing on disk",
              flush=True)
    return rows


def source_of(stem):
    return "mv" if stem.startswith(("train_train_", "test_test_")) else "gh"


# ------------------------------------------------------------- feature access

def kp_dir_for(row):
    """None -> preprocess default (data/multiview/keypoints, GH+MV); GSL npz
    live in their own dir so they need an explicit one."""
    return (GSL_DIR / "keypoints") if row["src"] == "gsl" else None


def feat_any(row, kind="face256", t_max=256):
    """dataset.feat_var with per-source kp dirs. Cache layout is shared with
    dataset.py so stage-2 training never recomputes what stage-1 already saw."""
    cache = FEAT_CACHE[kind]
    p = cache / f"{row['stem']}.npy"
    if p.exists():
        return np.load(p)
    kp = kp_dir_for(row)
    if kp is None:
        return feat_var(row["stem"], kind)
    feats, _ = clip_features_var(row["stem"], t_max=t_max,
                                 kind=VAR_KINDS[kind], kp_dir=kp)
    cache.mkdir(parents=True, exist_ok=True)
    np.save(p, np.ascontiguousarray(feats, dtype=np.float32))
    return feats


def len_of(row, kind="face256"):
    """Cached clip length (npy header read) or computes it once."""
    p = FEAT_CACHE[kind] / f"{row['stem']}.npy"
    if p.exists():
        with open(p, "rb") as f:
            np.lib.format.read_magic(f)
            shape, _, _ = np.lib.format.read_array_header_1_0(f)
            return int(shape[0])
    return len(feat_any(row, kind))


def warm_lengths(rows, kind="face256"):
    """Materialize any missing caches once (GSL first-access) and return the
    full length list needed for bucketing."""
    return [len_of(r, kind) for r in rows]


# ------------------------------------------------------------------ grouping

def build_groups(rows):
    """text-normalized groups: {norm_text: [row indices]}."""
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(r["text"].strip().lower(), []).append(i)
    return groups


def groups_by_src(rows, groups):
    """Split group keys by source (mixed-source groups don't occur: each
    text lives in exactly one of gh/mv/gsl)."""
    out = {"gh": [], "mv": [], "gsl": []}
    for key, idx in groups.items():
        out[rows[idx[0]]["src"]].append(key)
    return out


# ------------------------------------------------------------------ sampling

def mix_index(rows, groups, n_target, fractions, rng):
    """One epoch's index list: `n_target` slots, each slot samples a source
    by fraction, a text group uniformly within the source, and a take
    uniformly within the group. Fractions are relative; gh gets the remainder.
    Repeats are identical-sampling (new take each time), so every appearance
    of a text stays visually fresh while text exposure stays balanced."""
    gsrc = groups_by_src(rows, groups)
    weights = {"gh": max(0.0, 1.0 - fractions["mv"] - fractions["gsl"]),
               "mv": fractions["mv"], "gsl": fractions["gsl"]}
    # uniform group choice within each source
    picks = {s: [] for s in weights}
    for s, keys in gsrc.items():
        if weights[s] > 0 and keys:
            picks[s] = keys
    eff = {s: w for s, w in weights.items() if picks[s]}
    tot = sum(eff.values())
    srcs = list(eff)
    probs = [eff[s] / tot for s in srcs]
    idx = []
    for _ in range(n_target):
        s = rng.choices(srcs, weights=probs)[0]
        key = rng.choice(picks[s])
        idx.append(rng.choice(groups[key]))
    # avoid two adjacent identical takes
    rng.shuffle(idx)
    return idx


def stage2_index(rows, groups, mv_takes, rng):
    """Stage-2 epoch: ALL gh rows + mv sampled as strikes-per-unique-sentence
    (`mv_takes` random takes per mv text group). gsl is never a CE target."""
    idx = [i for i, r in enumerate(rows) if r["src"] == "gh"]
    gsrc = groups_by_src(rows, groups)["mv"]
    for key in gsrc:
        pool = groups[key]
        k = min(mv_takes, len(pool))
        idx += rng.sample(pool, k)
    rng.shuffle(idx)
    return idx


def index_stats(rows, idx):
    """Source composition of one epoch's index list (for logs / paper)."""
    out = {"gh": 0, "mv": 0, "gsl": 0}
    for i in idx:
        out[rows[i]["src"]] += 1
    return out


# ------------------------------------------------------------------ datasets

class PairDataset(Dataset):
    """Contrastive pairs: two independent augmented video views + raw text."""

    def __init__(self, rows, kind="face256"):
        self.rows = rows
        self.kind = kind

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x1 = feat_any(r, self.kind)
        x2 = augment_var(x1.copy())
        return {"src": torch.from_numpy(x1), "src2": torch.from_numpy(x2),
                "text": r["text"], "stem": r["stem"]}


class EncodeDataset(Dataset):
    """Eval-side: no augmentation, single view per row."""

    def __init__(self, rows, kind="face256"):
        self.rows = rows
        self.kind = kind

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x = feat_any(r, self.kind)
        return {"src": torch.from_numpy(x), "src2": torch.from_numpy(x),
                "text": r["text"], "stem": r["stem"]}


class VarRowDataset(Dataset):
    """Stage-2 CE/CTC dataset: rows-subset of SignDatasetVar semantics,
    implemented over shared reprs so checkpoint/data format never diverges."""

    def __init__(self, split_rows, tok, kind="face256", augment_flag=False):
        self.rows = split_rows
        self.tok = tok
        self.kind = kind
        self.augment_flag = augment_flag

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        f = feat_any(r, self.kind)
        if self.augment_flag:
            f = augment_var(f)
        return {"src": torch.from_numpy(f), "tgt": torch.tensor(
            self.tok.encode(r["text"]), dtype=torch.long),
            "stem": r["stem"], "text": r["text"]}


# ------------------------------------------------------------------ batching

def var_lens(idx, kind="face256"):
    return [len_of(v) for v in idx]


def bucketed_batches(ds, lens, batch, shuffle=True, bucket_batches=20, rng=None):
    """Same bucketing scheme as dataset.bucketed_loader, index-list based:
    sort into length buckets so a batch pads only ~20-30% waste, shuffle
    bucket order + intra-bucket order every call."""
    idx = list(range(len(ds)))
    if shuffle:
        rng = rng or random
        rng.shuffle(idx)
        idx.sort(key=lambda i: lens[i])
        buckets = [idx[i:i + batch * bucket_batches]
                   for i in range(0, len(idx), batch * bucket_batches)]
        rng.shuffle(buckets)
        batches = []
        for bkt in buckets:
            bl = [bkt[i:i + batch] for i in range(0, len(bkt), batch)]
            rng.shuffle(bl)
            batches.extend(bl)
    else:
        idx.sort(key=lambda i: lens[i])
        batches = [idx[i:i + batch] for i in range(0, len(idx), batch)]
    return batches
