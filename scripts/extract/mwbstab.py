"""MWB pilot stability check: does the tracker hold the signer through
fade-in, left-right moves, and slide cutaways with other people?

Reads data/jw/mwb/keypoints/*.npz, reports per video:
- present rates (face/pose/hands) overall + in first 90 frames (fade-in)
- tracker jumps: frame-to-frame nose-x shifts > 0.15 width while pose
  continuously present (signer <-> slide-person flips show up here)
- signer x-trajectory: median nose-x in thirds of the video (left->right?)
Usage: .venv312/bin/python scripts/extract/mwbstab.py (repo root)
"""
import glob
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
KPD = os.path.join(ROOT, "data", "jw", "mwb", "keypoints")


def jumps(px, present, thresh=0.15):
    n = 0
    idx = []
    for i in range(1, len(px)):
        if present[i] and present[i - 1] and abs(px[i] - px[i - 1]) > thresh:
            n += 1
            idx.append(i)
    return n, idx


for p in sorted(glob.glob(os.path.join(KPD, "*.npz"))):
    d = np.load(p, allow_pickle=True)
    n = int(d["n_frames"])
    fps = float(d["fps"])
    pp = d["pose_present"]
    fp = d["face_present"]
    nose_x = d["pose_xyz"][:, 0, 0]  # landmark 0 = nose
    nj, jidx = jumps(nose_x, pp)
    thirds = [np.median(nose_x[int(n * a):int(n * b)][pp[int(n * a):int(n * b)]])
              if pp[int(n * a):int(n * b)].any() else float("nan")
              for a, b in ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1))]
    print(f"{os.path.basename(p)}: {n} frames @ {fps:.0f}fps")
    print(f"  pose {pp.mean():.2f} face {fp.mean():.2f} "
          f"hands {d['handL_present'].mean():.2f}/{d['handR_present'].mean():.2f}")
    print(f"  first-90f pose {pp[:90].mean():.2f} face {fp[:90].mean():.2f}")
    print(f"  jumps: {nj}" + (f" e.g. frames {jidx[:8]}" if jidx else ""))
    print(f"  nose-x thirds: {[round(t, 2) for t in thirds]}")
