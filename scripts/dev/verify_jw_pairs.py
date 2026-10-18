"""Preflight verification for JW feature caches BEFORE any training launch.

Catches the failure class that poisoned contrastive v2 and stage2 p1t2:
feature slices whose absolute frames were not mapped through the per-source
extractor trims (mwb intro-cut via start_frame/skip_seconds; lfb kept-frame
list via frame_idx). Guards, per source:

  G1  every indexed unit has a cached .npy
  G2  cached length == min(t_max, independently mapped window length)
  G3  cached npy content == fresh recompute through the CURRENT slicing code
      (would have failed with the poisoned caches)
  G4  index row's a_lo/a_hi == an INDEPENDENT reimplementation of the mapping
      written here from the raw npz fields, not shared with pipeline/data
  G5  sampled unit text matches the raw embedded text after cleaning

Exit code 1 on any failure. Usage:
  .venv312/bin/python scripts/dev/verify_jw_pairs.py [--source mwb,nwt-ghs,lfb]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pipeline.data as D

FAILS = []


def fail(msg):
    FAILS.append(msg)
    print(f"  FAIL {msg}", flush=True)


def independent_map(z, start_f, end_f):
    """Standalone reimplementation (deliberately NOT calling data.py) of the
    abs-frame -> array-index mapping, from the raw npz fields."""
    keys = list(z.keys())
    if "frame_idx" in keys:  # lfb: kept original frame numbers
        fi = np.asarray(z["frame_idx"], dtype=np.int64)
        lo = int(np.searchsorted(fi, start_f, side="left"))
        hi = int(np.searchsorted(fi, end_f, side="left"))
        span_req = end_f - start_f
        if hi <= lo or hi - lo < 2:
            return None
        covered = int(fi[hi - 1]) - int(fi[lo]) + 1
        if covered < 0.5 * span_req:
            return None
        return lo, hi
    off = int(z["start_frame"]) if "start_frame" in keys else 0
    n = int(z["n_frames"]) if "n_frames" in keys else 1 << 30
    lo, hi = start_f - off, min(end_f - off, n)
    if lo < 0 or hi - lo < 2:
        return None
    return lo, hi


def raw_text_match(z, fields, row):
    units = None
    for field in fields:
        if field in z:
            raw = z[field]
            raw = raw.decode() if hasattr(raw, "decode") else str(raw)
            try:
                units = json.loads(raw)
            except json.JSONDecodeError:
                units = None
            if units:
                break
    if not units:
        return False
    pos = int(row["stem"].split("__")[-1])
    if not (0 <= pos < len(units)):
        return False
    cleaned = D._clean_text(units[pos].get("text_en", ""))
    return cleaned == row["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="nwt-ghs,mwb,lfb")
    ap.add_argument("--n-sample", type=int, default=20)
    args = ap.parse_args()
    sources = [s.strip() for s in args.source.split(",") if s.strip()]
    t_max = D.T_MAX

    for source in sources:
        print(f"== {source} ==", flush=True)
        rows = D.load_jw_rows(source)
        if not rows:
            fail(f"{source}: empty index")
            continue
        cache = D.FEAT_CACHE["face256"]
        cache_dir = Path(cache)

        # G1: every row cached
        missing = [r["stem"] for r in rows
                   if not (cache_dir / f"{r['stem']}.npy").exists()]
        if missing:
            fail(f"{source}: {len(missing)}/{len(rows)} units missing .npy, "
                 f"e.g. {missing[:3]}")
        n = sum(1 for _ in cache_dir.glob(f"{source}__*.npy"))
        print(f"  G1 cache files: {len(rows)} rows, {n} npy on disk",
              flush=True)

        rng = random.Random(6789 + len(rows))
        sample = rng.sample(rows, min(args.n_sample, len(rows)))
        n_g3_checked = 0
        for r in sample:
            stem = r["stem"]
            npz = r["npz"]
            sf, ef = int(r["start_f"]), int(r["end_f"])
            # G4: independent mapping
            z = np.load(npz, allow_pickle=True)
            m = independent_map(z, sf, ef)
            if m is None:
                fail(f"{stem}: independent mapping says unit has no kept "
                     f"footage (index kept it)")
                continue
            lo2, hi2 = m
            a_lo = int(r["a_lo"]) if "a_lo" in r else sf
            a_hi = int(r["a_hi"]) if "a_hi" in r else min(
                ef, int(z["n_frames"]) if "n_frames" in z else 1 << 30)
            if (a_lo, a_hi) != (lo2, hi2):
                fail(f"{stem}: mapped window mismatch stored=({a_lo},{a_hi}) "
                     f"independent=({lo2},{hi2})")
                continue
            # G2: cached length vs expected
            p = cache_dir / f"{stem}.npy"
            got = np.load(p, allow_pickle=True)
            expect_len = min(t_max, hi2 - lo2)
            if got.shape[0] != expect_len:
                fail(f"{stem}: cached len {got.shape[0]} != expected "
                     f"{expect_len}")
                continue
            # G3: content == fresh recompute
            fresh = D.chapter_verse_feats(npz, sf, ef, a_lo=lo2, a_hi=hi2)
            if fresh is None or not np.allclose(fresh, got, atol=1e-5):
                dev = (None if fresh is None else
                       float(np.max(np.abs(fresh - got))))
                fail(f"{stem}: cached content != fresh recompute (maxdev "
                     f"{dev})")
                continue
            n_g3_checked += 1
            # G5: text <-> position
            if not raw_text_match(z, ("verses_json", "sections_json"), r):
                fail(f"{stem}: cleaned unit text != index row text")

        splits = {}
        for r in rows:
            splits[r["split"]] = splits.get(r["split"], 0) + 1
        print(f"  G2/G4 sampled {len(sample)} rows; G3 content-verified "
              f"{n_g3_checked}; splits {splits}", flush=True)

        # trim boundary bookkeeping: report units dropped vs indexed
        kpath = ROOT / "data" / "jw" / source / "keypoints"
        print(f"  ok: {source} ({len(rows)} units)", flush=True)

    print("", flush=True)
    if FAILS:
        print(f"PREFLIGHT: {len(FAILS)} FAILURES — do NOT train", flush=True)
        sys.exit(1)
    print("PREFLIGHT: ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
