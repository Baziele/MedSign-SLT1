#!/usr/bin/env python3
"""Extract MediaPipe Holistic keypoints for the GSL dictionary videos.

Same landmarks/pipeline as wextract.py (478 face + 33 pose + 2x21 hands),
but for flat video files mapped by dictionary.csv instead of the
multiview layout. Output: one .npz per video in data/gsl/keypoints/,
with word metadata embedded. Resumable (skip-if-exists), crash-proof
wave execution, progress.jsonl log.

Usage:
  .venv312/bin/python -u scripts/extract/gextract.py [--workers 8]
Logs: console (redirect to logs/extract.log) + data/gsl/progress.jsonl
"""
import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wextract import (N_FACE, N_HAND, N_POSE, _record, make_holistic,
                      run_farm, vis_array, xyz_array)

ROOT = os.path.dirname(os.path.abspath(__file__))
GSL = os.path.join(ROOT, "data", "gsl")
VID_DIR = os.path.join(GSL, "videos")
OUT_DIR = os.path.join(GSL, "keypoints")

_HOLISTIC = None


def _init_worker():
    global _HOLISTIC
    _HOLISTIC = make_holistic()


def _process_gsl(task):
    src, out, meta, max_width = task
    t0 = time.time()
    try:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError("cv2 could not open video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = max_width / width if (max_width and width > max_width) else 1.0
        F, P, PV, W, HL, HR = [], [], [], [], [], []
        fp, pp, hlp, hrp = [], [], [], []
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if scale != 1.0:
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)),
                                   interpolation=cv2.INTER_AREA)
            res = _HOLISTIC.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            F.append(xyz_array(res.face_landmarks, N_FACE))
            fp.append(res.face_landmarks is not None)
            P.append(xyz_array(res.pose_landmarks, N_POSE))
            PV.append(vis_array(res.pose_landmarks, N_POSE))
            W.append(xyz_array(res.pose_world_landmarks, N_POSE))
            pp.append(res.pose_landmarks is not None)
            HL.append(xyz_array(res.left_hand_landmarks, N_HAND))
            hlp.append(res.left_hand_landmarks is not None)
            HR.append(xyz_array(res.right_hand_landmarks, N_HAND))
            hrp.append(res.right_hand_landmarks is not None)
            n += 1
        cap.release()
        if n == 0:
            raise RuntimeError("video contained 0 readable frames")
        data = {
            "face_xyz": np.stack(F), "pose_xyz": np.stack(P),
            "pose_vis": np.stack(PV), "pose_world_xyz": np.stack(W),
            "handL_xyz": np.stack(HL), "handR_xyz": np.stack(HR),
            "face_present": np.array(fp, dtype=bool),
            "pose_present": np.array(pp, dtype=bool),
            "handL_present": np.array(hlp, dtype=bool),
            "handR_present": np.array(hrp, dtype=bool),
            "n_frames": np.array(n, dtype=np.int32),
            "fps": np.array(float(fps), dtype=np.float32),
            "video_file": np.array(os.path.basename(src)),
            "word_id": np.array(str(meta["id"])),
            "word": np.array(meta["name"]),
            "word_type": np.array(meta["type"]),
            "level": np.array(str(meta["level"])),
            "category": np.array(meta["category"] or ""),
        }
        tmp = out + ".tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **data)
        os.replace(tmp, out)
        dt = time.time() - t0
        return {"file": meta["file"], "status": "done", "frames": n,
                "seconds": round(dt, 1),
                "pose_hit": round(float(np.array(pp).mean()), 3)}
    except Exception as e:
        return {"file": meta["file"], "status": "failed", "error": str(e)}


# run_farm calls wextract._process_one(task); adapt our 4-tuples by
# monkey-patching at call time: wrap tasks so _process_one dispatches here.
import wextract as _wx  # noqa: F401  (import ensures same pipeline constants)


def main():
    ap = argparse.ArgumentParser(description="GSL dictionary keypoint extraction.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-width", type=int, default=1280)
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--csv", default="dictionary.csv",
                    help="word list in data/gsl/ (use dictionary_clean.csv "
                    "for split single-word rows)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(GSL, args.csv))))
    if args.max_videos:
        rows = rows[:args.max_videos]
    print(f"{len(rows)} dictionary rows from {args.csv}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    def src_of(r):
        return os.path.join(VID_DIR, r.get("video_file") or r["file"])

    def out_of(r):
        if r.get("npz"):
            return os.path.join(OUT_DIR, r["npz"])
        return os.path.join(OUT_DIR, r["file"][:-4] + ".npz")

    def meta_of(r):
        return {"file": r.get("video_file") or r["file"],
                "id": r.get("entry_id") or r.get("id"),
                "name": r.get("word") or r.get("name"),
                "type": r.get("word_type") or r.get("type"),
                "level": r.get("level"), "category": r.get("category")}

    _wx._process_one = lambda task: _process_gsl(task)  # noqa
    _wx._init_worker = _init_worker  # noqa
    import concurrent.futures as cf
    tasks = []
    skipped = 0
    for r in rows:
        src, out, meta = src_of(r), out_of(r), meta_of(r)
        if os.path.exists(out) and not args.overwrite:
            skipped += 1
            continue
        if not os.path.exists(src):
            print(f"  MISSING video file: {src}", flush=True)
            continue
        tasks.append((r, (src, out, meta,
                          args.max_width if args.max_width > 0 else None)))
    print(f"{len(tasks)} to process, {skipped} already done.", flush=True)

    print("loading holistic model ...", flush=True)
    t_all = time.time()
    stats = {"done": 0, "skipped": skipped, "failed": 0}
    log_path = os.path.join(GSL, "progress.jsonl")
    try:
        with open(log_path, "a") as log:
            from wextract import _record as rec
            queue = {}
            for r, tt in tasks:
                queue[out_of(r)] = [tt, 0]
            with tqdm(total=len(queue), desc="extracting",
                      unit="video") as pbar:
                while queue:
                    for o in [k for k, (_, a) in queue.items() if a >= 3]:
                        rec({"file": o, "status": "failed",
                             "error": "crashed 3 waves, skipped"},
                            log, stats)
                        del queue[o]
                        pbar.update(1)
                    if not queue:
                        break
                    wave = list(queue)[:64]
                    ex = cf.ProcessPoolExecutor(
                        max_workers=args.workers, initializer=_init_worker)
                    fut2o = {ex.submit(_process_gsl, queue[o][0]): o
                             for o in wave}
                    try:
                        for fut in cf.as_completed(fut2o, timeout=1200):
                            o = fut2o[fut]
                            try:
                                r_ = fut.result()
                            except Exception as e:
                                r_ = {"file": o, "status": "failed",
                                      "error": f"worker exception: {e}"}
                            rec(r_, log, stats)
                            queue.pop(o, None)
                            pbar.update(1)
                    except cf.BrokenProcessPool:
                        tqdm.write("  [pool] worker crashed -- wave retries, "
                                   f"{len(queue)} pending")
                        for o in fut2o.values():
                            if o in queue:
                                queue[o][1] += 1
                    except TimeoutError:
                        tqdm.write("  [pool] 20min idle -- next wave")
                        for o in fut2o.values():
                            if o in queue:
                                queue[o][1] += 1
                    finally:
                        ex.shutdown(wait=True)
    except KeyboardInterrupt:
        print("\ninterrupted -- partial progress saved, re-run to continue.",
              flush=True)
    print(f"\nfinished in {time.time() - t_all:.0f}s -- " + ", ".join(
        f"{k}={v}" for k, v in stats.items()), flush=True)
    print(f"progress log: {log_path}", flush=True)


if __name__ == "__main__":
    main()
