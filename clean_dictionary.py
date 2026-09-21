#!/usr/bin/env python3
"""Split multi-word GSL entries so each word gets its own row (same video).

'abbreviate / summarize' -> two rows, one per word, both pointing at the
same mp4 but with distinct .npz outputs. Single-word rows keep the
original npz name so already-extracted files are skipped, not redone.

Usage: clean_dictionary.py  ->  gsl-dictionary/dictionary_clean.csv
"""
import csv
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
GSL = os.path.join(ROOT, "gsl-dictionary")


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:60] or "entry"


def main():
    rows = list(csv.DictReader(open(os.path.join(GSL, "dictionary.csv"))))
    out = []
    for r in rows:
        base = r["file"][:-4]  # video stem, e.g. 0001_abbreviate_summarize
        if "/" in r["name"]:
            for i, variant in enumerate(
                    [v.strip() for v in r["name"].split("/")]):
                if not variant:
                    continue
                out.append({
                    "entry_id": r["id"], "variant": i + 1, "word": variant,
                    "type": r["type"], "level": r["level"],
                    "category": r["category"], "video_url": r["video_url"],
                    "video_file": r["file"],
                    "npz": f"{int(r['id']):04d}_{slug(variant)}.npz",
                })
        else:
            out.append({
                "entry_id": r["id"], "variant": 1, "word": r["name"].strip(),
                "type": r["type"], "level": r["level"],
                "category": r["category"], "video_url": r["video_url"],
                "video_file": r["file"], "npz": base + ".npz",
            })
    # collision guard: two variants slug identically?
    seen = {}
    for o in out:
        if o["npz"] in seen:
            o["npz"] = o["npz"][:-4] + f"_v{o['variant']}.npz"
        seen[o["npz"]] = True
    with open(os.path.join(GSL, "dictionary_clean.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    n_split = sum(1 for r in rows if "/" in r["name"])
    print(f"{len(rows)} entries -> {len(out)} word rows "
          f"({n_split} split into multiples)")


if __name__ == "__main__":
    main()
