"""Dataset: splits CSV + cached (64, 225) features + token ids.

Features are cached per-stem as data/cache/feat/<stem>.npy on first access, so
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
    "pos": ROOT / "data" / "cache" / "feat",        # (64, 225) positions (rounds 1/4/5)
    "posvel": ROOT / "data" / "cache" / "feat_v2",  # (64, 450) positions + velocities
    "face": ROOT / "data" / "cache" / "feat_v3",    # (64, 633) pose+hands+face contours
    "pos256": ROOT / "data" / "cache" / "feat_var_pos",   # (Tv<=256, 225) variable-length
    "face256": ROOT / "data" / "cache" / "feat_var_face",  # (Tv<=256, 633) variable-length
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


def _interp_rows(flat, pos):
    """Vectorized linear interpolation over rows: flat (N, D) float32,
    pos (M,) sample positions in [0, N-1] -> (M, D). Identical to per-column
    np.interp (linear; positions stay in-range by construction so edge
    behavior matches), one pass over D instead of D python calls."""
    lo = np.floor(pos).astype(np.int64)
    lo = np.clip(lo, 0, flat.shape[0] - 1)
    hi = np.minimum(lo + 1, flat.shape[0] - 1)
    w = (pos - lo).astype(np.float32)[:, None]
    return flat[lo] * (1.0 - w) + flat[hi] * w


def augment_var(x):
    """Train-only augmentation for variable-length (Tv, D): random temporal
    stretch 0.85-1.15x (interp back to same Tv) + joint jitter σ0.01.
    Length-preserving so padding masks stay valid. No spatial warps.
    Vectorized: two gather+lerp passes over D (no per-column python loop),
    float32 throughout."""
    T = len(x)
    if T < 8:
        return np.ascontiguousarray(x, dtype=np.float32)
    flat = np.ascontiguousarray(x.reshape(T, -1), dtype=np.float32)
    rate = float(np.random.uniform(0.85, 1.15))
    new_T = max(8, int(round(T * rate)))
    stretched = _interp_rows(flat, np.linspace(0, T - 1, new_T))
    out = _interp_rows(stretched, np.linspace(0, new_T - 1, T))
    out += np.random.normal(0, 0.01, size=out.shape).astype(np.float32)
    return np.ascontiguousarray(out.reshape(x.shape), dtype=np.float32)


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


def _seed_worker(wid):
    """Give each loader worker its own numpy/python RNG stream, otherwise
    every worker inherits identical state and produces identical
    augmentations. torch's seed is handled by DataLoader itself."""
    import random

    import numpy as np

    s = (torch.initial_seed() + wid) % 2**32
    np.random.seed(s)
    random.seed(s)


def bucketed_loader(ds, batch, shuffle=True, bucket_batches=20, num_workers=0):
    """Length-bucketed DataLoader: sort indices by cached src length so each
    batch pads only ~20-30% waste instead of shortest-to-256. Token budget:
    B*Lmax <= ~8192 keeps MPS happy (32 x 256 worst case).
    batch_sampler is a plain list of index-lists; DataLoader consumes it
    directly (no BatchSampler wrapper needed). num_workers>0 feeds batches
    from parallel CPU processes (workers return CPU tensors; the .to(device)
    copy stays in the main process, so MPS never crosses a process boundary).
    macOS uses spawn: dataset objects must be picklable (plain rows + Tokenizer)."""
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
    return DataLoader(ds, batch_sampler=batches, collate_fn=coll,
                      num_workers=num_workers,
                      worker_init_fn=_seed_worker if num_workers > 0 else None)
