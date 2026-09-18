#!/usr/bin/env python3
"""Extract skeletal keypoints for SignTalk-GH videos (MediaPipe Holistic).

Run with:  .venv312/bin/python extract_keypoints.py --max-videos 30

What it extracts per frame:
  - FACE  : 478 landmarks (468 mesh + 10 iris). The mesh contains every
            region you asked for: eyes, eyebrows, glabella (between brows),
            nose ridge/arch, mouth/lips, jawline (face oval).
  - POSE  : 33 body landmarks (torso, arms, legs) + visibility + world coords.
  - HANDS : left + right x 21 landmarks (full finger detail).

Output: one compressed .npz per video in --out-dir, e.g. 1A.npz, with keys:
  face_xyz (T,478,3), pose_xyz (T,33,3), pose_vis (T,33),
  pose_world_xyz (T,33,3), handL_xyz (T,21,3), handR_xyz (T,21,3),
  face_present / pose_present / handL_present / handR_present (T,) bool,
  frame_idx (T,), fps, n_frames, width, height,
  video_file, sentence_id, signer.

Missing detections are NaN (with the *_present flags marking them), so
downstream code can mask instead of choking. Coordinates are NORMALIZED
(x, y in [0,1] of frame; z is relative depth) -- normalize against body
size at TRAINING time, not here (raw extraction stays lossless).

Resumability (for the 9,878-video full run):
  - Videos already having an output .npz are SKIPPED (unless --overwrite).
  - Each video's result is written atomically (temp file + rename), so an
    interrupted run never leaves a corrupt .npz behind.
  - Every outcome is appended to progress.jsonl IMMEDIATELY, so Ctrl+C
    loses at most the video in flight. Re-run the same command to continue.
  - A video that fails (corrupt file etc.) is logged as "failed" and the
    run continues with the next one.

Progress visibility:
  - tqdm bar in the terminal, one flushed log line per video,
  - `tail -f <out-dir>/progress.jsonl` from another terminal,
  - `ls <out-dir>/*.npz | wc -l` counts finished videos any time.

Usage:
  .venv312/bin/python extract_keypoints.py --max-videos 30
  .venv312/bin/python extract_keypoints.py --max-videos 30 --overwrite
  .venv312/bin/python extract_keypoints.py --max-videos 0    # 0 = ALL videos
  .venv312/bin/python extract_keypoints.py --max-videos 0 --workers 8  # overnight full run
"""

import argparse
import json
import os
import re
import sys
import time

import cv2
import numpy as np
from tqdm import tqdm

import mediapipe as mp

N_FACE, N_POSE, N_HAND = 478, 33, 21

VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi")
NAME_RE = re.compile(r"^(\d+)([A-Za-z]*)\.(?:mp4|mkv|mov|avi)$", re.IGNORECASE)


def natural_key(fname):
    """Sort 1A, 2A, ... 10A numerically (plain sorted() would put 10A before 2A)."""
    m = NAME_RE.match(fname)
    if m:
        return (int(m.group(1)), m.group(2).upper(), fname)
    return (10**9, "", fname)


def make_holistic():
    return mp.solutions.holistic.Holistic(
        static_image_mode=False,      # tracking mode: uses previous frame, faster + smoother
        model_complexity=1,           # 0 = fastest, 2 = most accurate
        enable_segmentation=False,    # we don't need person segmentation masks
        refine_face_landmarks=True,   # 478 face pts (468 mesh + 10 iris) instead of 468
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def xyz_array(lm_container, n):
    """Landmark list -> (n,3) float32 normalized xyz, or all-NaN if undetected."""
    arr = np.full((n, 3), np.nan, dtype=np.float32)
    if lm_container is not None:
        for i, lm in enumerate(lm_container.landmark[:n]):
            arr[i, 0], arr[i, 1], arr[i, 2] = lm.x, lm.y, lm.z
    return arr


def vis_array(lm_container, n):
    """Visibility scores -> (n,) float32, or all-NaN if undetected (pose only)."""
    arr = np.full((n,), np.nan, dtype=np.float32)
    if lm_container is not None:
        for i, lm in enumerate(lm_container.landmark[:n]):
            arr[i] = lm.visibility
    return arr


def process_video(path, holistic, max_width):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("cv2 could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:  # 0 / NaN guard (some files lack metadata)
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = 1.0
    if max_width and width > max_width:
        scale = max_width / width  # landmarks are normalized: coordinates unaffected

    F, P, PV, W, HL, HR = [], [], [], [], [], []
    fp, pp, hlp, hrp, idxs = [], [], [], [], []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if scale != 1.0:
            frame = cv2.resize(
                frame, (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

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
        idxs.append(frame_idx)
        frame_idx += 1
    cap.release()
    if frame_idx == 0:
        raise RuntimeError("video contained 0 readable frames")

    m = NAME_RE.match(os.path.basename(path))
    return {
        "face_xyz": np.stack(F), "pose_xyz": np.stack(P),
        "pose_vis": np.stack(PV), "pose_world_xyz": np.stack(W),
        "handL_xyz": np.stack(HL), "handR_xyz": np.stack(HR),
        "face_present": np.array(fp, dtype=bool),
        "pose_present": np.array(pp, dtype=bool),
        "handL_present": np.array(hlp, dtype=bool),
        "handR_present": np.array(hrp, dtype=bool),
        "frame_idx": np.array(idxs, dtype=np.int32),
        "fps": np.array(fps, dtype=np.float32),
        "n_frames": np.array(frame_idx, dtype=np.int32),
        "width": np.array(width, dtype=np.int32),
        "height": np.array(height, dtype=np.int32),
        "video_file": np.array(os.path.basename(path)),
        "sentence_id": np.array(m.group(1) if m else ""),
        "signer": np.array(m.group(2).upper() if m else ""),
    }


_HOLISTIC = None  # one Holistic instance per process (never shared across workers)


def _init_worker():
    """Pool initializer: each worker builds its own model instance."""
    global _HOLISTIC
    _HOLISTIC = make_holistic()


def _process_one(task):
    """Run one video -> save .npz atomically -> return a log record.

    Runs identically in the main process (workers=1) or a pool worker.
    Only ever called with tasks the main process has already checked as
    not-done, so no two processes can claim the same output.
    """
    videos_dir, fname, out, max_width = task
    t0 = time.time()
    try:
        data = process_video(
            os.path.join(videos_dir, fname), _HOLISTIC, max_width)
        tmp = out + ".tmp"
        # NOTE: pass an open handle -- savez_compressed appends ".npz"
        # to plain string paths not ending in .npz, breaking the rename.
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **data)
        os.replace(tmp, out)  # atomic: a killed run never leaves a partial .npz
        n, dt = int(data["n_frames"]), time.time() - t0
        return {
            "file": fname, "status": "done", "frames": n,
            "seconds": round(dt, 1),
            "face_hit": round(float(data["face_present"].mean()), 3),
            "pose_hit": round(float(data["pose_present"].mean()), 3),
            "hand_hit": round(float((data["handL_present"] | data["handR_present"]).mean()), 3),
        }
    except Exception as e:  # one bad video must never kill the batch
        return {"file": fname, "status": "failed", "error": str(e)}


def _record(rec, log, stats, show=True):
    stats[rec["status"]] = stats.get(rec["status"], 0) + 1
    rec["t"] = round(time.time(), 1)  # wall-clock: enables rate/ETA math
    log.write(json.dumps(rec) + "\n")
    log.flush()
    if show:
        if rec["status"] == "done":
            tqdm.write(f"  done {rec['file']}: {rec['frames']} frames in {rec['seconds']}s")
        elif rec["status"] == "failed":
            tqdm.write(f"  FAILED {rec['file']}: {rec.get('error')}")


def main():
    ap = argparse.ArgumentParser(description="MediaPipe Holistic keypoint extraction (resumable).")
    ap.add_argument("--videos-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Medsign", "archive", "SignTalk-GH", "Videos"))
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "keypoints"))
    ap.add_argument("--max-videos", type=int, default=30,
                    help="How many videos (natural-sorted) to process. 0 = all. Default 30.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-process videos that already have an output .npz.")
    ap.add_argument("--max-width", type=int, default=1280,
                    help="Downscale frames wider than this before inference (0 = keep native). "
                         "Landmarks are normalized, so coordinates are unaffected; only speed changes.")
    ap.add_argument("--workers", type=int, default=6,
                    help="Parallel worker processes, one Holistic instance each. "
                         "6 keeps an M1 Pro responsive; 8 for overnight runs. 1 = serial.")
    args = ap.parse_args()

    files = sorted(
        (f for f in os.listdir(args.videos_dir)
         if f.lower().endswith(VIDEO_EXTS) and NAME_RE.match(f)),
        key=natural_key,
    )
    if args.max_videos and args.max_videos > 0:
        files = files[: args.max_videos]
    print(f"{len(files)} videos selected from {args.videos_dir}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "progress.jsonl")
    max_width = args.max_width if args.max_width > 0 else None

    # Resume: the main process alone decides what is (not) done, so workers
    # never race on the same output.
    tasks = []
    stats = {"done": 0, "skipped": 0, "failed": 0}
    with open(log_path, "a") as log:
        for fname in files:
            out = os.path.join(args.out_dir, os.path.splitext(fname)[0] + ".npz")
            if os.path.exists(out) and not args.overwrite:
                _record({"file": fname, "status": "skipped"}, log, stats, show=False)
                continue
            tasks.append((args.videos_dir, fname, out, max_width))
        print(f"{len(tasks)} to process, {stats['skipped']} already done.", flush=True)

        print("loading holistic model ...", flush=True)
        t_all = time.time()
        try:
            if args.workers > 1:
                from multiprocessing import Pool
                with Pool(args.workers, initializer=_init_worker) as pool:
                    it = pool.imap_unordered(_process_one, tasks, chunksize=1)
                    for rec in tqdm(it, total=len(tasks), desc="extracting", unit="video"):
                        _record(rec, log, stats)
            else:
                _init_worker()
                print("model ready.", flush=True)
                for task in tqdm(tasks, desc="extracting", unit="video"):
                    _record(_process_one(task), log, stats)
        except KeyboardInterrupt:
            print("\ninterrupted -- partial progress is saved, re-run to continue.",
                  flush=True)
    print(f"\nfinished in {time.time() - t_all:.0f}s -- " + ", ".join(
        f"{k}={v}" for k, v in stats.items()), flush=True)
    print(f"progress log: {log_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
