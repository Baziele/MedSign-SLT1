"""Trim LFB keypoints: cut messy title-card intro.

Per npz in data/jw/lfb/keypoints/: finds the tracker lock frame (first
frame where a 90-frame window shows >90% pose presence with nose-x
spread < 0.12) and slices all arrays from there. Picture (Art) sections
are already skipped at extraction time; as a safety net any remaining
frames inside Art ranges (mapped via frame_idx) are dropped too.

Absolute video time of kept frame i is frame_idx[i] / fps.
Idempotent: skips npz already trimmed (marked by trim_cut field).
Usage (repo root):
  .venv312/bin/python scripts/extract/trim_lfb.py
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
KPD = os.path.join(ROOT, "data", "jw", "lfb", "keypoints")

ARR_KEYS = ("face_xyz", "pose_xyz", "pose_vis", "pose_world_xyz",
            "handL_xyz", "handR_xyz", "face_present", "pose_present",
            "handL_present", "handR_present")


def lock_frame(px, pp):
    n = len(pp)
    for i in range(0, max(1, n - 90), 30):
        seg = px[i:i + 90]
        p = pp[i:i + 90]
        if p.mean() > 0.9 and np.ptp(seg[p]) < 0.12:
            return i
    return 0


def main():
    for path in sorted(glob.glob(os.path.join(KPD, "*.npz"))):
        d = dict(np.load(path, allow_pickle=True))
        if "trim_cut" in d:
            continue
        n = int(d["n_frames"])
        fps = float(d["fps"])
        if "frame_idx" in d:
            fidx = d["frame_idx"]
        else:
            s0 = int(d.get("start_frame", 0))
            fidx = np.arange(s0, s0 + n, dtype=np.int32)
        pp = d["pose_present"]
        px = d["pose_xyz"][:, 0, 0]
        cut = lock_frame(px, pp)
        keep = np.ones(len(fidx), dtype=bool)
        keep[:cut] = False
        secs = json.loads(str(d["sections_json"]))
        for s in secs:
            if str(s.get("label", "")).startswith("Art"):
                a = int(s["start_s"] * fps)
                b = int((s["start_s"] + s["dur_s"]) * fps) + 1
                keep[(fidx >= a) & (fidx < b)] = False
        idx = np.nonzero(keep)[0]
        for k in ARR_KEYS:
            d[k] = d[k][idx]
        fidx = fidx[idx]
        d["frame_idx"] = fidx.astype(np.int32)
        d["n_frames"] = np.array(len(idx), dtype=np.int32)
        d["start_frame"] = np.array(int(fidx[0]), dtype=np.int32)
        d["skip_seconds"] = np.array(round(float(fidx[0]) / fps, 3))
        d["trim_cut"] = np.array(cut, dtype=np.int32)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **d)
        os.replace(tmp, path)
        print(f"  {os.path.basename(path)}: cut {cut}f, "
              f"kept {len(idx)}/{n}f", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
