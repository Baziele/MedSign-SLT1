"""Chunked resumable extraction. Usage: whextract2.py <chunk 0-9> [chunks=10].
Skips files that already exist. One chunk ~7500 videos."""
import sys
import zipfile

Z = r"C:\Users\Baziele\Documents\Code\medsign-multiview\archive.zip"
D = r"C:\Users\Baziele\Documents\Code\medsign-multiview\data"

chunk = int(sys.argv[1])
nchunks = int(sys.argv[2]) if len(sys.argv) > 2 else 10

z = zipfile.ZipFile(Z)
names = z.namelist()
videos = sorted(n for n in names if n.endswith(".mp4"))
mine = videos[chunk::nchunks]
print(f"chunk {chunk}/{nchunks}: {len(mine)} videos", flush=True)

import os

done = skip = 0
for i, n in enumerate(mine):
    out = os.path.join(D, n)
    if os.path.exists(out):
        skip += 1
        continue
    z.extract(n, D)
    done += 1
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(mine)} done={done} skip={skip}", flush=True)
meta = os.path.join(D, "Metadata.csv")
if not os.path.exists(meta):
    z.extract("Metadata.csv", D)
print(f"CHUNK {chunk} DONE: extracted={done} skipped={skip}", flush=True)
