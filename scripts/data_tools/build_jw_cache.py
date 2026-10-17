"""Build the feature cache for JW verse/section slices.

Reads each chapter npz ONCE (interp + normalize once per chapter), then saves
one (Tv, 633) float32 npy per timed verse/section to
data/cache/feat_var_face/<source>__<chapter>__<unit>.npy — the exact cache
location pipeline/data.feat_any expects.

The ptc index (data/splits/jw_<source>_index.csv) is rebuilt with --rebuild.

Usage: .venv312/bin/python scripts/data_tools/build_jw_cache.py \
           [--source nwt-ghs] [--rebuild]
Progress lands in logs/build_jw_cache_<source>.log (gitignored).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset import FEAT_CACHE
from pipeline import data as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="nwt-ghs")
    ap.add_argument("--rebuild", action="store_true",
                    help="regenerate the timed-unit index csv")
    args = ap.parse_args()

    if args.rebuild:
        idx = ROOT / "data" / "splits" / f"jw_{args.source}_index.csv"
        if idx.exists():
            idx.unlink()

    rows = D.load_jw_rows(args.source)
    if not rows:
        raise SystemExit(f"no units parsed for {args.source}")

    cache = FEAT_CACHE["face256"]
    by_chapter = {}
    for r in rows:
        by_chapter.setdefault(r["chapter"], []).append(r)

    todo = [ch for ch, rs in by_chapter.items()
            if not all((cache / f"{r['stem']}.npy").exists() for r in rs)]
    missing_units = sum(
        1 for ch, rs in by_chapter.items()
        for r in rs if not (cache / f"{r['stem']}.npy").exists())
    print(f"{args.source}: {len(by_chapter)} chapters, {len(rows)} units, "
          f"{len(todo)} chapters to compute ({missing_units} units)")

    t0, done_units, n_bad = time.time(), 0, 0
    for ci, ch in enumerate(sorted(todo), 1):
        rs = by_chapter[ch]
        try:
            from pipeline.data import chapter_verse_feats
            for r in sorted(rs, key=lambda x: int(x["start_f"])):
                out = cache / f"{r['stem']}.npy"
                if out.exists():
                    done_units += 1
                    continue
                f = chapter_verse_feats(r["npz"], r["start_f"], r["end_f"])
                cache.mkdir(parents=True, exist_ok=True)
                np.save(out, np.ascontiguousarray(f, dtype=np.float32))
                done_units += 1
        except Exception as e:  # keep the sweep going; report at the end
            n_bad += 1
            print(f"  BAD {ch}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        if ci % 10 == 0 or ci == len(todo):
            el = time.time() - t0
            rate = done_units / max(el, 1e-3)
            eta = (missing_units - done_units) / max(rate, 1e-3) / 60
            print(f"  chapter {ci}/{len(todo)} | {done_units} units cached | "
                  f"{rate:.1f} units/s | ETA {eta:.0f} min | bad {n_bad}",
                  flush=True)

    print(f"done: {done_units} units cached in {(time.time()-t0)/60:.1f} min"
          f" ({n_bad} bad chapters)", flush=True)


if __name__ == "__main__":
    main()
