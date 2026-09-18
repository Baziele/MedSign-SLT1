"""Retrieval baseline: mean-pooled (225,) clip vectors, cosine NN in train.

For each test clip: features = clip_features(stem) -> (64, 225),
vector = L2-normalized mean over time. Predict the English of the nearest
train vector. Score corpus BLEU + exact match, overall and per signer.

Val split is deliberately untouched (reserved for model selection).
Cache: retrieval/vectors.npz (stem -> vector) so re-runs are seconds.
"""
import csv
from pathlib import Path

import numpy as np

from preprocess import SPLITS_DIR, clip_features

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "retrieval"
CACHE = OUT_DIR / "vectors.npz"


def load_split_rows(name):
    with open(SPLITS_DIR / f"{name}.csv") as f:
        return list(csv.DictReader(f))


def get_vectors(stems):
    """Mean-pooled L2-normalized vectors, using disk cache where possible."""
    cached = dict(np.load(CACHE)) if CACHE.exists() else {}
    missing = [s for s in stems if s not in cached]
    if missing:
        print(f"computing {len(missing)} vectors...")
        for i, s in enumerate(missing):
            f, _ = clip_features(s)
            v = f.mean(axis=0)
            n = np.linalg.norm(v)
            cached[s] = (v / n if n > 0 else v).astype(np.float32)
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(missing)}")
        OUT_DIR.mkdir(exist_ok=True)
        np.savez_compressed(CACHE, **cached)
    return np.stack([cached[s] for s in stems])


def predict(test_rows, train_rows, train_mat):
    test_mat = get_vectors([r["stem"] for r in test_rows])
    sims = test_mat @ train_mat.T  # cosine (both L2-normalized)
    best = sims.argmax(axis=1)
    out = []
    for r, j, row_tr in zip(test_rows, best, [train_rows[k] for k in best]):
        out.append(
            {
                "video": r["video"],
                "signer": r["signer"],
                "category": r["category"],
                "true": r["text"],
                "pred": row_tr["text"],
                "match": row_tr["video"],
            }
        )
    return out


def report(preds, label):
    from sacrebleu import corpus_bleu

    print(f"--- {label} (n={len(preds)}) ---")
    em = np.mean([p["true"] == p["pred"] for p in preds])
    bleu = corpus_bleu([p["pred"] for p in preds], [[p["true"] for p in preds]])
    print(f"overall: BLEU={bleu.score:.1f} exact_match={em:.3f}")
    for signer in sorted({p["signer"] for p in preds}):
        sub = [p for p in preds if p["signer"] == signer]
        b = corpus_bleu([p["pred"] for p in sub], [[p["true"] for p in sub]])
        e = np.mean([p["true"] == p["pred"] for p in sub])
        print(f"  signer {signer} (n={len(sub)}): BLEU={b.score:.1f} exact_match={e:.3f}")
    return preds


def main():
    train_rows = load_split_rows("train")
    train_mat = get_vectors([r["stem"] for r in train_rows])
    print(f"train vectors: {train_mat.shape}")

    for split in ["test_R", "test_T"]:
        preds = predict(load_split_rows(split), train_rows, train_mat)
        report(preds, split)
        OUT_DIR.mkdir(exist_ok=True)
        with open(OUT_DIR / f"preds_{split}.csv", "w", newline="") as f:
            w = csv.DictWriter(
                f, ["video", "signer", "category", "true", "pred", "match"]
            )
            w.writeheader()
            w.writerows(preds)
        print(f"wrote retrieval/preds_{split}.csv")


if __name__ == "__main__":
    main()
