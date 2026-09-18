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

from preprocess import SPLITS_DIR, T_FRAMES, clip_features
from tokenizer import PAD_ID, Tokenizer

ROOT = Path(__file__).resolve().parent
FEAT_CACHE_POS = ROOT / "cache" / "feat"       # (64, 225) positions (round 1)
FEAT_CACHE_VEL = ROOT / "cache" / "feat_v2"    # (64, 450) positions + velocities


def feat(stem, velocity=True):
    """(64, 225|450) float32, from disk cache or computed once."""
    cache = FEAT_CACHE_VEL if velocity else FEAT_CACHE_POS
    p = cache / f"{stem}.npy"
    if p.exists():
        return np.load(p)
    f, _ = clip_features(stem)
    if velocity:
        vel = np.zeros_like(f)
        vel[1:] = f[1:] - f[:-1]
        f = np.concatenate([f, vel], axis=1).astype(np.float32)
    cache.mkdir(parents=True, exist_ok=True)
    np.save(p, f)
    return f


def augment(x):
    """Train-only augmentation on (64, 225|450) positions(+vel): speed jitter
    via random crop+resize, small rotation/scale/shift, joint jitter."""
    T, D = x.shape
    J = D // 2 if D == 450 else D // 1  # joints*3 for positions half
    pos = x[:, : J].reshape(T, -1, 3).copy()
    # speed jitter: crop 48-64 frames, stretch back to 64
    c = np.random.randint(48, T + 1)
    s = np.random.randint(0, T - c + 1)
    seg = pos[s : s + c]
    idx = np.linspace(0, c - 1, T)
    flat = seg.reshape(c, -1)
    pos = np.stack([np.interp(idx, np.arange(c), flat[:, k])
                    for k in range(flat.shape[1])], axis=1).reshape(T, -1, 3)
    # NOTE: no rotation/scale/shift -- place of articulation carries meaning
    # in signing, so spatial warps teach the wrong invariance (round 4: 2.1).
    pos += np.random.normal(0, 0.01, size=pos.shape)  # keypoint jitter
    out = np.concatenate([pos.reshape(T, -1),
                          x[:, J:]], axis=1) if D == 450 else pos.reshape(T, -1)
    return out.astype(np.float32)


class SignDataset(Dataset):
    def __init__(self, split, tok, augment_flag=False, velocity=True):
        with open(SPLITS_DIR / f"{split}.csv") as f:
            self.rows = list(csv.DictReader(f))
        self.tok = tok
        self.augment_flag = augment_flag
        self.velocity = velocity

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        f = feat(r["stem"], self.velocity)
        if self.augment_flag:
            f = augment(f)
        src = torch.from_numpy(np.ascontiguousarray(f))  # (64, 225|450)
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
