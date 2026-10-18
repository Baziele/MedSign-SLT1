"""Build the feature cache for JW verse/section slices.

Reads each chapter npz ONCE (interp + normalize once per chapter), then saves
one (Tv, 633) float32 npy per timed verse/section to
data/cache/feat_var_face/<source>__<chapter>__<unit>.npy — the exact cache
location pipeline/data.feat_any expects. Incremental: existing npys skipped.

The unit index (data/splits/jw_<source>_index.csv) is rebuilt with --rebuild.

Usage: .venv312/bin/python scripts/data_tools/build_jw_cache.py \
           [--source nwt-ghs,mwb,lfb] [--rebuild]
Progress lands in the console (redirect into logs/ when backgrounded).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset import FEAT_CACHE
from pipeline.data import chapter_verse_feats, load_jw_rows


def build_source(source, cache):
    rows = load_jw_rows(source)
    if not rows:
        print(f"jw[{source}]: no units parsed — skipped", flush=True)
        return []
    by_chapter = {}
    for r in rows:
        by_chapter.setdefault(r["chapter"], []).append(r)

    todo = [ch for ch, rs in by_chapter.items()
            if not all((cache / f"{r['stem']}.npy").exists() for r in rs)]
    missing_units = sum(1 for ch, rs in by_chapter.items()
                        for r in rs if not (cache / f"{r['stem']}.npy").exists())
    print(f"jw[{source}]: {len(by_chapter)} chapters, {len(rows)} units, "
          f"{len(todo)} chapters to compute ({missing_units} units)",
          flush=True)

    t0, done_units, n_bad = time.time(), 0, 0
    total = len(todo)
    for ci, ch in enumerate(sorted(todo), 1):
        rs = by_chapter[ch]
        try:
            for r in sorted(rs, key=lambda x: int(x["start_f"])):
                out = cache / f"{r['stem']}.npy"
                if out.exists():
                    done_units += 1
                    continue
                f = chapter_verse_feats(r["npz"], r["start_f"], r["end_f"],
                                        a_lo=int(r["a_lo"]) if "a_lo" in r
                                        else None,
                                        a_hi=int(r["a_hi"]) if "a_hi" in r
                                        else None)
                if f is None:
                    n_short = getattr(build_source, "_n_short", 0) + 1
                    build_source._n_short = n_short
                    print(f"  SKIP {r['stem']}: no kept footage "
                          f"(index should have dropped it)", flush=True)
                    continue
                cache.mkdir(parents=True, exist_ok=True)
                np.save(out, np.ascontiguousarray(f, dtype=np.float32))
                done_units += 1
        except Exception as e:  # keep the sweep going; report inline
            n_bad += 1
            print(f"  BAD {source}/{ch}: {type(e).__name__}: {str(e)[:120]}",
                  flush=True)
            continue
        if ci % 10 == 0 or ci == total:
            el = time.time() - t0
            rate = done_units / max(el, 1e-3)
            eta = (missing_units - done_units) / max(rate, 1e-3) / 60
            print(f"  [{source}] chapter {ci}/{total} | {done_units} units "
                  f"cached | {rate:.1f} units/s | ETA {eta:.0f} min | bad "
                  f"{n_bad}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="nwt-ghs,mwb,lfb",
                    help="comma list of JW sets")
    ap.add_argument("--rebuild", action="store_true",
                    help="regenerate the timed-unit index csvs")
    args = ap.parse_args()

    sources = [s.strip() for s in args.source.split(",") if s.strip()]
    if args.rebuild:
        for source in sources:
            idx = ROOT / "data" / "splits" / f"jw_{source}_index.csv"
            if idx.exists():
                idx.unlink()

    cache = FEAT_CACHE["face256"]
    t_start = time.time()
    for source in sources:
        build_source(source, cache)
    n = len(list(cache.glob("nwt-ghs__*.*npy"))) + \
        len(list(cache.glob("mwb__*.npy"))) + \
        len(list(cache.glob("lfb__*.npy")))
    print(f"done in {(time.time()-t_start)/60:.1f} min; "
          f"~{n} unit features on disk", flush=True)


if __name__ == "__main__":
    main()
