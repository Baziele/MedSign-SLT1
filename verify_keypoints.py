#!/usr/bin/env python3
"""Integrity-check every keypoint .npz (keys, shapes, non-empty).

Run:  .venv312/bin/python verify_keypoints.py [--dir keypoints]
Exit code 0 = all clean, 1 = problems found (listed). Keep this script with
any copy of the data so the receiver can re-verify after transfer.
"""

import argparse
import glob
import os
import sys

import numpy as np

EXPECTED_KEYS = {
    "face_xyz", "pose_xyz", "pose_vis", "pose_world_xyz",
    "handL_xyz", "handR_xyz",
    "face_present", "pose_present", "handL_present", "handR_present",
    "frame_idx", "fps", "n_frames", "width", "height",
    "video_file", "sentence_id", "signer",
}


def check(path):
    errs = []
    try:
        d = np.load(path, allow_pickle=True)
    except Exception as e:
        return [f"unreadable: {e}"]
    if set(d.files) != EXPECTED_KEYS:
        errs.append(f"keys mismatch: missing={sorted(EXPECTED_KEYS - set(d.files))} "
                    f"extra={sorted(set(d.files) - EXPECTED_KEYS)}")
        return errs
    try:
        T = int(d["n_frames"])
    except Exception:
        return ["n_frames unreadable"]
    if T <= 0:
        errs.append("n_frames <= 0")
        return errs
    for key, shape in [
        ("face_xyz", (T, 478, 3)), ("pose_xyz", (T, 33, 3)),
        ("pose_vis", (T, 33)), ("pose_world_xyz", (T, 33, 3)),
        ("handL_xyz", (T, 21, 3)), ("handR_xyz", (T, 21, 3)),
    ]:
        if d[key].shape != shape:
            errs.append(f"{key}.shape={d[key].shape} expected {shape}")
    for key in ["face_present", "pose_present", "handL_present", "handR_present",
                "frame_idx"]:
        if len(d[key]) != T:
            errs.append(f"{key} len={len(d[key])} expected {T}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "keypoints"))
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.dir, "*.npz")))
    bad, empty_pose = 0, []
    for i, p in enumerate(files):
        errs = check(p)
        if errs:
            bad += 1
            print(f"BAD {os.path.basename(p)}: {'; '.join(errs)}")
        else:
            d = np.load(p, allow_pickle=True)
            if not d["pose_present"].any():
                empty_pose.append(os.path.basename(p))
        if (i + 1) % 2000 == 0:
            print(f"  ...{i + 1}/{len(files)}", flush=True)
    print(f"\nchecked={len(files)} bad={bad} pose_never_detected={len(empty_pose)}")
    for f in empty_pose[:20]:
        print(f"  NOPOSE {f}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
