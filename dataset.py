"""Dataset: splits CSV + cached (64, 225) features + token ids.

Features are cached per-stem as cache/feat/<stem>.npy on first access, so
the ~80s full pass happens once and every later run (train, retrieval,
eval) is I/O-bound. Text is encoded on the fly (fast).
"""
import csv
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from preprocess import SPLITS_DIR, T_FRAMES, clip_features, clip_features_face
from preprocess import T_MAX, clip_features_var
from tokenizer import PAD_ID, Tokenizer

ROOT = Path(__file__).resolve().parent
FEAT_CACHE = {
    "pos": ROOT / "cache" / "feat",        # (64, 225) positions (rounds 1/4/5)
    "posvel": ROOT / "cache" / "feat_v2",  # (64, 450) positions + velocities
    "face": ROOT / "cache" / "feat_v3",    # (64, 633) pose+hands+face contours
    "pos256": ROOT / "cache" / "feat_var_pos",   # (Tv<=256, 225) variable-length
    "face256": ROOT / "cache" / "feat_var_face",  # (Tv<=256, 633) variable-length
}
KIND_DIM = {"pos": 225, "posvel": 450, "face": 633, "pos256": 225, "face256": 633}
VAR_KINDS = {"pos256": "pos", "face256": "face"}


def feat(stem, kind="face"):
    """(64, D) float32, from disk cache or computed once."""
    cache = FEAT_CACHE[kind]
    p = cache / f"{stem}.npy"
    if p.exists():
        return np.load(p)
    if kind == "face":
        f, _ = clip_features_face(stem)
    else:
        f, _ = clip_features(stem)
        if kind == "posvel":
            vel = np.zeros_like(f)
            vel[1:] = f[1:] - f[:-1]
            f = np.concatenate([f, vel], axis=1).astype(np.float32)
    cache.mkdir(parents=True, exist_ok=True)
    np.save(p, f)
    return f


def augment(x):
    """Train-only augmentation on pos-only (64, D): speed jitter via random
    crop+resize, plus joint jitter. No spatial warps (place carries meaning).
    No velocity support: velocity proved to be noise (rounds 2-3)."""
    T, D = x.shape
    pos = x.reshape(T, -1, 3).copy()
    # speed jitter: crop 48-64 frames, stretch back to 64
    c = np.random.randint(48, T + 1)
    s = np.random.randint(0, T - c + 1)
    seg = pos[s : s + c]
    idx = np.linspace(0, c - 1, T)
    flat = seg.reshape(c, -1)
    pos = np.stack([np.interp(idx, np.arange(c), flat[:, k])
                    for k in range(flat.shape[1])], axis=1).reshape(T, -1, 3)
    pos += np.random.normal(0, 0.01, size=pos.shape)  # keypoint jitter
    return pos.reshape(T, -1).astype(np.float32)


def feat_var(stem, kind="face256", t_max=T_MAX):
    """(Tv<=t_max, D) float32 variable-length, from disk cache or computed once."""
    assert kind in VAR_KINDS, f"var kind must be one of {sorted(VAR_KINDS)}"
    cache = FEAT_CACHE[kind]
    p = cache / f"{stem}.npy"
    if p.exists():
        return np.load(p)
    f, _ = clip_features_var(stem, t_max=t_max, kind=VAR_KINDS[kind])
    cache.mkdir(parents=True, exist_ok=True)
    np.save(p, f)
    return f


def var_len(stem, kind="face256"):
    """Cached clip length without loading full array (uses .npy header)."""
    p = FEAT_CACHE[kind] / f"{stem}.npy"
    if p.exists():
        with open(p, "rb") as f:
            ver = np.lib.format.read_magic(f)
            shape, _, _ = np.lib.format.read_array_header_1_0(f)
            return int(shape[0])
    return len(feat_var(stem, kind))


def augment_var(x):
    """Train-only augmentation for variable-length (Tv, D): random temporal
    stretch 0.85-1.15x (interp back to same Tv) + joint jitter σ0.01.
    Length-preserving so padding masks stay valid. No spatial warps."""
    T = len(x)
    if T < 8:
        return x.astype(np.float32)
    rate = float(np.random.uniform(0.85, 1.15))
    new_T = max(8, int(round(T * rate)))
    old_idx = np.linspace(0, T - 1, T)
    new_idx = np.linspace(0, T - 1, new_T)
    flat = x.reshape(T, -1).astype(np.float64)
    stretched = np.stack([np.interp(new_idx, old_idx, flat[:, k])
                          for k in range(flat.shape[1])], axis=1)
    back_idx = np.linspace(0, new_T - 1, T)
    out = np.stack([np.interp(back_idx, np.arange(new_T), stretched[:, k])
                    for k in range(stretched.shape[1])], axis=1)
    out = out.reshape(T, -1) + np.random.normal(0, 0.01, size=(T, x.shape[1]))
    return out.astype(np.float32)


class SignDataset(Dataset):
    def __init__(self, split, tok, augment_flag=False, kind="face"):
        with open(SPLITS_DIR / f"{split}.csv") as f:
            self.rows = list(csv.DictReader(f))
        self.tok = tok
        self.augment_flag = augment_flag
        self.kind = kind

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        f = feat(r["stem"], self.kind)
        if self.augment_flag:
            f = augment(f)
        src = torch.from_numpy(np.ascontiguousarray(f))  # (64, D)
        tgt = torch.tensor(self.tok.encode(r["text"]), dtype=torch.long)
        return {"src": src, "tgt": tgt, "stem": r["stem"], "text": r["text"]}


def collate(batch):
    src = torch.stack([b["src"] for b in batch])  # (B, 64, 225)
    tgt = pad_sequence([b["tgt"] for b in batch], batch_first=True,
                       padding_value=PAD_ID)
    return {
        "src": src,
        "tgt_in": tgt[:, :-1],   # <sos> ... w_{L-1}
        "tgt_out": tgt[:, 1:],   # w_1 ... <eos> (+ pads)
        "pad_mask": tgt[:, :-1] == PAD_ID,
        "stems": [b["stem"] for b in batch],
        "texts": [b["text"] for b in batch],
    }


class SignDatasetVar(Dataset):
    """Variable-length sister of SignDataset: src is (Tv, D), Tv<=256."""

    def __init__(self, split, tok, augment_flag=False, kind="face256"):
        assert kind in VAR_KINDS, f"var kind must be one of {sorted(VAR_KINDS)}"
        with open(SPLITS_DIR / f"{split}.csv") as f:
            self.rows = list(csv.DictReader(f))
        self.tok = tok
        self.augment_flag = augment_flag
        self.kind = kind

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        f = feat_var(r["stem"], self.kind)
        if self.augment_flag:
            f = augment_var(f)
        src = torch.from_numpy(np.ascontiguousarray(f))  # (Tv, D)
        tgt = torch.tensor(self.tok.encode(r["text"]), dtype=torch.long)
        return {"src": src, "tgt": tgt, "stem": r["stem"], "text": r["text"]}


def collate_var(batch):
    """Pad variable-length src to Lmax; src_key_padding_mask True = pad."""
    srcs = [b["src"] for b in batch]
    lens = torch.tensor([len(s) for s in srcs], dtype=torch.long)
    src = pad_sequence(srcs, batch_first=True)  # (B, Lmax, D) zero-padded
    Lmax = src.size(1)
    src_mask = torch.arange(Lmax).unsqueeze(0) >= lens.unsqueeze(1)  # (B, Lmax) bool
    tgt = pad_sequence([b["tgt"] for b in batch], batch_first=True,
                       padding_value=PAD_ID)
    return {
        "src": src,
        "src_mask": src_mask,
        "src_lens": lens,
        "tgt_in": tgt[:, :-1],
        "tgt_out": tgt[:, 1:],
        "pad_mask": tgt[:, :-1] == PAD_ID,
        "stems": [b["stem"] for b in batch],
        "texts": [b["text"] for b in batch],
    }


def bucketed_loader(ds, batch, shuffle=True, bucket_batches=20):
    """Length-bucketed DataLoader: sort indices by cached src length so each
    batch pads only ~20-30% waste instead of shortest-to-256. Token budget:
    B*Lmax <= ~8192 keeps MPS happy (32 x 256 worst case).
    batch_sampler is a plain list of index-lists; DataLoader consumes it
    directly (no BatchSampler wrapper needed)."""
    from torch.utils.data import DataLoader

    if isinstance(ds, SignDatasetVar):
        lens = []
        for r in ds.rows:
            p = FEAT_CACHE[ds.kind] / f"{r['stem']}.npy"
            if p.exists():
                with open(p, "rb") as f:
                    np.lib.format.read_magic(f)
                    shape, _, _ = np.lib.format.read_array_header_1_0(f)
                    lens.append(int(shape[0]))
            else:
                lens.append(len(feat_var(r["stem"], ds.kind)))
    else:
        lens = [T_FRAMES] * len(ds)
    idx = list(range(len(ds)))
    if shuffle:
        # sort into buckets of batch*bucket_batches, shuffle bucket order and
        # within-bucket batch order: similar lengths land together.
        import random

        random.shuffle(idx)
        idx.sort(key=lambda i: lens[i])
        buckets = [idx[i:i + batch * bucket_batches]
                   for i in range(0, len(idx), batch * bucket_batches)]
        random.shuffle(buckets)
        batches = []
        for bkt in buckets:
            bl = [bkt[i:i + batch] for i in range(0, len(bkt), batch)]
            random.shuffle(bl)
            batches.extend(bl)
    else:
        idx.sort(key=lambda i: lens[i])
        batches = [idx[i:i + batch] for i in range(0, len(idx), batch)]
    coll = collate_var if isinstance(ds, SignDatasetVar) else collate
    return DataLoader(ds, batch_sampler=batches, collate_fn=coll)
