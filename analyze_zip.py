"""Analyze the multiview archive listing (no extraction). Run on Windows box."""
import re
import zipfile
from collections import Counter, defaultdict

Z = r"C:\Users\Baziele\Documents\Code\medsign-multiview\archive.zip"
z = zipfile.ZipFile(Z)
names = z.namelist()
print("total entries:", len(names))

pat = re.compile(r"^(train|test)/(?:train|test)/sentence_id_(\d+)/(\d+)_(\S+?)_cam(\d+)_rep(\d+)\.mp4$")
top = Counter(x.split("/")[0] for x in names)
print("top-level:", dict(top))
split, sids, sign, cam, rep = Counter(), set(), Counter(), Counter(), Counter()
pertxt = defaultdict(int)
bad = []
for n in names:
    if not n.endswith(".mp4"):
        continue
    m = pat.match(n)
    if not m:
        bad.append(n)
        continue
    s, sid, _vid, signer, c, r = m.groups()
    split[s] += 1
    sids.add((s, sid))
    sign[signer] += 1
    cam[c] += 1
    rep[r] += 1
    pertxt[(s, sid)] += 1
print("unmatched:", len(bad), bad[:5])
print("split:", dict(split))
print("sentence groups:", len(sids))
print("signers:", dict(sign))
print("cams:", dict(cam))
print("reps:", dict(rep))
import statistics
v = list(pertxt.values())
print("videos/sentence: min", min(v), "max", max(v), "mean", round(sum(v) / len(v), 2))
print("compressed bytes:", sum(i.file_size for i in z.infolist()))
