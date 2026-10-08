"""Extract only sentence_id dirs in [lo, hi]. Usage: whextract_sid.py <lo> <hi>.
Skips existing files. Run detached; logs progress."""
import os
import sys
import zipfile

Z = r"C:\Users\Baziele\Documents\Code\medsign-multiview\archive.zip"
D = r"C:\Users\Baziele\Documents\Code\medsign-multiview\data"

lo, hi = int(sys.argv[1]), int(sys.argv[2])

import re

pat = re.compile(r"sentence_id_(\d+)")
z = zipfile.ZipFile(Z)
mine = [n for n in z.namelist()
        if n.endswith(".mp4") and (lambda m: m and lo <= int(m.group(1)) <= hi)(pat.search(n))]
print(f"sids {lo}-{hi}: {len(mine)} videos", flush=True)
done = skip = 0
for i, n in enumerate(sorted(mine)):
    out = os.path.join(D, n)
    if os.path.exists(out):
        skip += 1
        continue
    z.extract(n, D)
    done += 1
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(mine)} done={done} skip={skip}", flush=True)
print(f"SIDS {lo}-{hi} DONE: extracted={done} skipped={skip}", flush=True)
