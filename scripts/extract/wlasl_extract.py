#!/usr/bin/env python3
"""Extract MediaPipe Holistic keypoints for the WLASL dataset.

Same landmarks/pipeline as wextract.py (478 face + 33 pose + 2x21 hands).
Input : <WLASL>/archive/videos/{video_id}.mp4  (+ WLASL_v0.3.json for gloss)
Output: <WLASL>/keypoints/{video_id}.npz (gloss embedded), all outputs and
        logs live in the WLASL folder, nothing in the medsign workspace.
Resumable (skip-if-exists), crash-proof wave execution.

Usage:
  .venv312/bin/python -u wlasl_extract.py [--workers 8]
Logs: <WLASL>/extract.log (console) + <WLASL>/progress.jsonl
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wextract import (N_FACE, N_HAND, N_POSE, make_holistic, vis_array,
                      xyz_array)

WLASL = "/Users/mazen/dev/research/Wlasl_data"
VID_DIR = os.path.join(WLASL, "archive", "videos")
OUT_DIR = os.path.join(WLASL, "keypoints")

_HOLISTIC = None


def _init_worker():
    global _HOLISTIC
    _HOLISTIC = make_holistic()


def _process_one(task):
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
            "video_id": np.array(meta["video_id"]),
            "gloss": np.array(meta["gloss"]),
            "split": np.array(meta.get("split") or ""),
        }
        tmp = out + ".tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **data)
        os.replace(tmp, out)
        dt = time.time() - t0
        return {"file": meta["video_id"], "status": "done", "frames": n,
                "seconds": round(dt, 1),
                "pose_hit": round(float(np.array(pp).mean()), 3)}
    except Exception as e:
        return {"file": meta["video_id"], "status": "failed", "error": str(e)}


def main():
    ap = argparse.ArgumentParser(description="WLASL keypoint extraction.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-width", type=int, default=1280)
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    meta = {}
    for g in json.load(open(os.path.join(WLASL, "archive", "WLASL_v0.3.json"))):
        for inst in g["instances"]:
            meta[str(inst["video_id"])] = {
                "video_id": str(inst["video_id"]), "gloss": g["gloss"],
                "split": inst.get("split", "")}
    vids = sorted(f[:-4] for f in os.listdir(VID_DIR)
                  if f.endswith(".mp4"))
    if args.max_videos:
        vids = vids[:args.max_videos]
    print(f"{len(vids)} videos on disk", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    import concurrent.futures as cf
    from wextract import _record as rec
    tasks = []
    skipped = 0
    for v in vids:
        out = os.path.join(OUT_DIR, v + ".npz")
        if os.path.exists(out) and not args.overwrite:
            skipped += 1
            continue
        m = meta.get(v, {"video_id": v, "gloss": "", "split": ""})
        tasks.append((os.path.join(VID_DIR, v + ".mp4"), out, m,
                      args.max_width if args.max_width > 0 else None))
    print(f"{len(tasks)} to process, {skipped} already done.", flush=True)

    print("loading holistic model ...", flush=True)
    t_all = time.time()
    stats = {"done": 0, "skipped": skipped, "failed": 0}
    try:
        with open(os.path.join(WLASL, "progress.jsonl"), "a") as log:
            queue = {t[1]: [t, 0] for t in tasks}
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
                    fut2o = {ex.submit(_process_one, queue[o][0]): o
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


if __name__ == "__main__":
    main()
