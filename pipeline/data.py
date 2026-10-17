"""Shared data plumbing for the phase-1 pipeline (contrastive + stage2).

Row dicts carry a `src` tag so every sampler/stat can split by source:
  gh  = original SignTalk hospital sentences (all_train.csv, non-MV stems)
  mv  = Medsign multiview takes (all_train.csv, train_train_/test_test_ stems)
  gsl = GHS dictionary entries (data/gsl/gsl-dictionary/dictionary_clean.csv)
  jw* = JW continuous chapters, sliced verse/section level. Timing + English
        text live INSIDE each npz (`verses_json` / `sections_json`: unit rows
        with start_s/dur_s/text_en). Feature cache = chapter normalized once,
        then the [start_f, end_f) window saved per unit.

All sampling is index-list based: datasets hold the full pool, and each epoch
gets a fresh deterministic index list. The exact composition of every epoch is
logged (pipeline.logkit) so runs are reproducible and paper-reportable.
"""
import csv
import json
import re
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from dataset import FEAT_CACHE, VAR_KINDS, feat_var, augment_var, collate_var
from preprocess import SPLITS_DIR, clip_features_var, T_MAX
from preprocess import _interp_missing, normalize as pp_normalize, resample as pp_resample

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


_TAG = re.compile(r"<[^>]+>")


def _clean_text(t):
    """JW embedded text carries HTML fragments, entity-encoded on top
    (e.g. &lt;span class=\"verse\"&gt;). Unescape first, THEN strip tags,
    then unescape again for double-encoded entities; collapse whitespace."""
    import html
    s = html.unescape(t or "")
    s = _TAG.sub(" ", s)
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


# ------------------------------------------------------- JW embedded timers

def _npz_units(npz_path):
    """(fps, n_frames, [unit dicts]) from one JW npz: verses_json preferred,
    sections_json fallback — same timed-unit schema (start_s/dur_s/text_en)."""
    z = np.load(npz_path, allow_pickle=True)
    blocks = {}
    for field in ("verses_json", "sections_json"):
        if field in z:
            raw = z[field]
            raw = raw.decode() if hasattr(raw, "decode") else str(raw)
            try:
                blocks[field] = json.loads(raw)
            except json.JSONDecodeError:
                blocks[field] = []
    if not blocks:
        return float(z["fps"]), int(z["n_frames"]), []
    fps = float(z["fps"])
    n_frames = int(z["n_frames"])
    units = []
    for field, data in blocks.items():
        for u in data:
            st, du = float(u.get("start_s", -1)), float(u.get("dur_s", 0))
            text = _clean_text(u.get("text_en", ""))
            if st < 0 or du <= 0.2 or not text:
                continue
            units.append({"start_s": st, "dur_s": du, "text_en": text})
    return fps, n_frames, units


def load_jw_rows(source="nwt-ghs"):
    """Slice a JW set's chapters into timed-unit training rows (`src`=source,
    with jw prefix in slug). Chapter normalized once, [start_f, end_f) sliced.
    Index cache: data/splits/jw_{source}_index.csv (ignored, fine to delete)."""
    kpdir = ROOT / "data" / "jw" / source / "keypoints"
    idx_csv = SPLITS_DIR / f"jw_{source}_index.csv"
    if idx_csv.exists():
        with open(idx_csv) as f:
            return list(csv.DictReader(f))
    rows = []
    for p in sorted(kpdir.glob("*.npz")):
        fps, n_frames, units = _npz_units(p)
        for i, u in enumerate(units):
            start_f = int(round(u["start_s"] * fps))
            end_f = min(int(round((u["start_s"] + u["dur_s"]) * fps)), n_frames)
            if end_f - start_f < 2:
                continue
            rows.append({
                "stem": f"{source}__{p.stem}__{i:03d}",
                "chapter": p.stem, "npz": str(p),
                "start_f": str(start_f), "end_f": str(end_f),
                "dur_s": f"{u['dur_s']:.2f}", "src": source,
                "text": u["text_en"],
            })
    SPLITS_DIR.mkdir(exist_ok=True)
    with open(idx_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["stem"])
        w.writeheader()
        w.writerows(rows)
    print(f"jw[{source}]: indexed {len(rows)} units from "
          f"{len(set(r['chapter'] for r in rows))} chapters", flush=True)
    return rows


def chapter_verse_feats(npz, start_f, end_f, kind="face256", t_max=T_MAX):
    """Feature matrix (Tv, 633) for one JW slice. The chapter is normalized
    once, then the frame window is cut — per-verse stats would re-run the
    interp/normalize pipeline per unit for the same video."""
    from render_skeleton import FACE_VERTS

    z = np.load(npz, allow_pickle=True)
    face = _interp_missing(z["face_xyz"].astype(np.float64), z["face_present"])
    pose, hL, hR, face_n, _ = pp_normalize(
        z["pose_xyz"].astype(np.float64),
        z["handL_xyz"].astype(np.float64),
        z["handR_xyz"].astype(np.float64), face, z["pose_present"])
    hL = np.where(z["handL_present"][:, None, None], hL, 0.0)
    hR = np.where(z["handR_present"][:, None, None], hR, 0.0)
    sl = slice(int(start_f), int(end_f))
    parts = [pose[sl], hL[sl], hR[sl],
             face_n[sl][:, list(FACE_VERTS)]]
    n = parts[0].shape[0]
    if n > t_max:
        parts = [pp_resample(p, t_max) for p in parts]
    feats = np.concatenate(parts, axis=1)
    feats = np.where(np.isfinite(feats), feats, 0.0).astype(np.float32)
    return feats.reshape(len(feats), -1)


# ------------------------------------------------------------- feature access

def kp_dir_for(row):
    """None -> preprocess default (data/multiview/keypoints, GH+MV); GSL npz
    live in their own dir so they need an explicit one."""
    return (GSL_DIR / "keypoints") if row["src"] == "gsl" else None


def feat_any(row, kind="face256", t_max=256):
    """dataset.feat_var with per-source kp dirs. Cache layout is shared with
    dataset.py so stage-2 training never recomputes what stage-1 already saw.
    JW verse rows need their cache prebuilt (build_jw_cache.py): inline
    computation would reload + renormalize a whole chapter per verse."""
    cache = FEAT_CACHE[kind]
    p = cache / f"{row['stem']}.npy"
    if p.exists():
        return np.load(p)
    if "start_f" in row:
        raise RuntimeError(
            f"missing JW cache {p.name}; run scripts/data_tools/build_jw_cache.py first")
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
    text lives in exactly one of gh/mv/gsl/jw*)."""
    out = {}
    for key, idx in groups.items():
        out.setdefault(rows[idx[0]]["src"], []).append(key)
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
    """Stage-2 epoch: every gh AND jw verse row (JW is the main training
    source) + mv capped at `mv_takes` takes per unique sentence. gsl is never
    a CE target."""
    idx = [i for i, r in enumerate(rows) if r["src"] != "mv"]
    for key in groups_by_src(rows, groups).get("mv", []):
        pool = groups[key]
        k = min(mv_takes, len(pool))
        idx += rng.sample(pool, k)
    rng.shuffle(idx)
    return idx


def index_stats(rows, idx):
    """Source composition of one epoch's index list (for logs / paper)."""
    out = {}
    for i in idx:
        s = rows[i]["src"]
        out[s] = out.get(s, 0) + 1
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
