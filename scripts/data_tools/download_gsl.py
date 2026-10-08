#!/usr/bin/env python3
"""Download the full GSL dictionary (words + phrases) with videos.

1. POSTs the word list from ghsldictionary.com
2. Writes dictionary.csv (id, name, type, level, category, video_url, file)
3. Downloads every mp4 into videos/ (skip-if-exists resume, 3 retries)

Usage: download_gsl.py [--max N] [--out DIR]
Progress: download.log. Safe to re-run; failures listed at the end.
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.request

API = "https://ghsldictionary.com/api/students/words"


def slug(name, i):
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:60]
    return f"{i:04d}_{s or 'entry'}.mp4"


def fetch_words():
    req = urllib.request.Request(
        API, data=b"{}", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["words"]


def get(url, dest, tries=3):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "skipped"
    for a in range(tries):
        try:
            urllib.request.urlretrieve(url, dest + ".part")
            os.replace(dest + ".part", dest)
            return "done"
        except Exception as e:
            if a == tries - 1:
                return f"FAILED: {e}"
            time.sleep(2)
    return "FAILED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "data", "gsl"))
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "videos"), exist_ok=True)
    log = open(os.path.join(args.out, "download.log"), "a")

    def msg(s):
        print(s, flush=True)
        log.write(s + "\n")

    words = fetch_words()
    if args.max:
        words = words[:args.max]
    msg(f"fetched {len(words)} entries")

    rows = []
    for w in words:
        fn = slug(w["name"], w["id"])
        rows.append([w["id"], w["name"], w["type"], w.get("levelId"),
                     w.get("category"), w["video_url"], fn])
    with open(os.path.join(args.out, "dictionary.csv"), "w", newline="") as f:
        csv.writer(f).writerow(
            ["id", "name", "type", "level", "category", "video_url", "file"])
        csv.writer(f).writerows(rows)
    msg(f"wrote dictionary.csv ({len(rows)} rows)")

    fails = []
    for i, r in enumerate(rows):
        st = get(r[5], os.path.join(args.out, "videos", r[6]))
        if st.startswith("FAILED"):
            fails.append((r[0], r[1], st))
        if (i + 1) % 100 == 0:
            msg(f"progress: {i + 1}/{len(rows)}")
    msg(f"finished: {len(rows) - len(fails)} ok, {len(fails)} failed")
    for fid, name, e in fails:
        msg(f"  FAIL {fid} {name}: {e}")


if __name__ == "__main__":
    sys.exit(main())
