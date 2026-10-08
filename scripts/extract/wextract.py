#!/usr/bin/env python3
"""Extract skeletal keypoints for the multiview GSL dataset (MediaPipe Holistic).

Same pipeline as extract_keypoints.py on the Mac: per-frame 478 face +
33 pose (+vis/world) + 2x21 hands, atomic .npz writes, skip-if-exists resume,
progress.jsonl. Adapted to the multiview layout:

  data/train/train/sentence_id_<N>/<vid>_<S>_cam<C>_rep<R>.mp4
  data/test/test/sentence_id_<N>/...

Output stem: <split>_sentence_id_<N>_<vid>_<S>_cam<C>_rep<R>.npz
Extra fields per file: split, cam, rep (plus video_file, sentence_id, signer).

Tuned for a crash-prone 4C/8T box: --workers defaults to 3, small chunks.
Usage:
  .venv312w\\Scripts\\python wextract.py --max-videos 10
  .venv312w\\Scripts\\python wextract.py --max-videos 0 --workers 3
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

VID_RE = re.compile(
    r"^(train|test)[/\\](?:train|test)[/\\](sentence_id_\d+)[/\\]"
    r"(\d+)_([A-Za-z0-9]+)_cam(\d+)_rep(\d+)\.(mp4|mkv|mov|avi)$",
    re.IGNORECASE,
)


def collect(data_dir):
    found = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith((".mp4", ".mkv", ".mov", ".avi")):
                rel = os.path.relpath(os.path.join(root, f), data_dir)
                if VID_RE.match(rel):
                    found.append(rel)
    return sorted(found)


def make_holistic():
    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def xyz_array(lm_container, n):
    arr = np.full((n, 3), np.nan, dtype=np.float32)
    if lm_container is not None:
        for i, lm in enumerate(lm_container.landmark[:n]):
            arr[i, 0], arr[i, 1], arr[i, 2] = lm.x, lm.y, lm.z
    return arr


def vis_array(lm_container, n):
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
    if not fps or fps != fps or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = 1.0
    if max_width and width > max_width:
        scale = max_width / width

    F, P, PV, W, HL, HR = [], [], [], [], [], []
    fp, pp, hlp, hrp, idxs = [], [], [], [], []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)),
                               interpolation=cv2.INTER_AREA)
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

    m = VID_RE.match(os.path.relpath(path, _DATA_DIR))
    split, siddir, vid, signer, cam, rep = (
        m.group(1), m.group(2), m.group(3), m.group(4).upper(),
        m.group(5), m.group(6)) if m else ("", "", "", "", "", "")
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
        "sentence_id": np.array(vid),
        "signer": np.array(signer),
        "split": np.array(split),
        "cam": np.array(cam),
        "rep": np.array(rep),
    }


_HOLISTIC = None
_DATA_DIR = ""


def _init_worker():
    global _HOLISTIC
    _HOLISTIC = make_holistic()


def _process_one(task):
    data_dir, rel, out, max_width = task
    t0 = time.time()
    try:
        data = process_video(os.path.join(data_dir, rel), _HOLISTIC, max_width)
        tmp = out + ".tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **data)
        os.replace(tmp, out)
        n, dt = int(data["n_frames"]), time.time() - t0
        return {
            "file": rel, "status": "done", "frames": n,
            "seconds": round(dt, 1),
            "face_hit": round(float(data["face_present"].mean()), 3),
            "pose_hit": round(float(data["pose_present"].mean()), 3),
            "hand_hit": round(float((data["handL_present"] | data["handR_present"]).mean()), 3),
        }
    except Exception as e:
        return {"file": rel, "status": "failed", "error": str(e)}


def _record(rec, log, stats, show=True):
    stats[rec["status"]] = stats.get(rec["status"], 0) + 1
    rec["t"] = round(time.time(), 1)
    log.write(json.dumps(rec) + "\n")
    log.flush()
    if show:
        if rec["status"] == "done":
            tqdm.write(f"  done {rec['file']}: {rec['frames']} frames in {rec['seconds']}s")
        elif rec["status"] == "failed":
            tqdm.write(f"  FAILED {rec['file']}: {rec.get('error')}")


def run_farm(tasks, args, log, stats):
    """Crash-proof execution in small waves: each wave gets a fresh pool
    that is cleanly shut down at the wave boundary. A native crash
    (MediaPipe 0xc0000005) breaks only its wave; uncompleted videos are
    retried (max 3 crashes each, then recorded failed). The parent never
    loads MediaPipe, so it cannot segfault. Small waves also avoid the
    mass worker-recycle stall seen with max_tasks_per_child."""
    import concurrent.futures as cf

    WAVE = 64
    pending = {t[1]: [t, 0] for t in tasks}  # rel -> [task, crash_attempts]
    with tqdm(total=len(tasks), desc="extracting", unit="video") as pbar:
        while pending:
            # give up on videos that crashed 3 waves in a row
            for rel in [r for r, (_, a) in pending.items() if a >= 3]:
                _record({"file": rel, "status": "failed",
                         "error": "crashed 3 waves, skipped"}, log, stats)
                del pending[rel]
                pbar.update(1)
            if not pending:
                break
            wave_rels = list(pending)[:WAVE]
            ex = cf.ProcessPoolExecutor(
                max_workers=args.workers, initializer=_init_worker)
            fut2rel = {ex.submit(_process_one, pending[rel][0]): rel
                       for rel in wave_rels}
            try:
                for fut in cf.as_completed(fut2rel, timeout=1200):
                    rel = fut2rel[fut]
                    try:
                        rec = fut.result()
                    except Exception as e:
                        rec = {"file": rel, "status": "failed",
                               "error": f"worker exception: {e}"}
                    _record(rec, log, stats)
                    pending.pop(rel, None)
                    pbar.update(1)
            except cf.BrokenProcessPool:
                tqdm.write("  [pool] worker crashed -- wave will retry, "
                           f"{len(pending)} pending")
                for rel in fut2rel.values():
                    if rel in pending:
                        pending[rel][1] += 1
            except TimeoutError:
                tqdm.write("  [pool] 20min without a finish -- next wave")
                for rel in fut2rel.values():
                    if rel in pending:
                        pending[rel][1] += 1
            finally:
                ex.shutdown(wait=True)


def stem_of(rel):
    return rel.replace("\\", "_").replace("/", "_")[:-4] + ".npz"


def main():
    global _DATA_DIR
    ap = argparse.ArgumentParser(description="Multiview keypoint extraction (resumable).")
    ap.add_argument("--data-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "data", "multiview", "mv-data"))
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "data", "multiview", "keypoints"))
    ap.add_argument("--max-videos", type=int, default=10,
                    help="How many videos to process. 0 = all. Default 10.")
    ap.add_argument("--min-sid", type=int, default=0,
                    help="Only sentence_id >= this (0 = no filter).")
    ap.add_argument("--max-sid", type=int, default=0,
                    help="Only sentence_id <= this (0 = no filter).")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-width", type=int, default=1280)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--skip-list", default="",
                    help="text file with already-done output stems "
                    "(one per line) to skip without checking disk, e.g. "
                    "a listing from another machine's out dir")
    args = ap.parse_args()
    _DATA_DIR = args.data_dir

    files = collect(args.data_dir)
    if args.min_sid or args.max_sid:
        lo = args.min_sid or 0
        hi = args.max_sid or 10**9
        keep = []
        for rel in files:
            m = re.search(r"sentence_id_(\d+)", rel)
            if m and lo <= int(m.group(1)) <= hi:
                keep.append(rel)
        files = keep
        print(f"sid filter {lo}-{hi}: {len(files)} videos", flush=True)
    if args.max_videos and args.max_videos > 0:
        files = files[: args.max_videos]
    print(f"{len(files)} videos selected from {args.data_dir}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "progress.jsonl")
    max_width = args.max_width if args.max_width > 0 else None

    skip_stems = set()
    if args.skip_list:
        with open(args.skip_list, errors="replace") as f:
            skip_stems = {ln.strip() for ln in f if ln.strip()}
        print(f"skip-list {args.skip_list}: {len(skip_stems)} stems", flush=True)

    tasks = []
    stats = {"done": 0, "skipped": 0, "failed": 0}
    with open(log_path, "a") as log:
        for rel in files:
            out = os.path.join(args.out_dir, stem_of(rel))
            if os.path.exists(out) and not args.overwrite:
                _record({"file": rel, "status": "skipped"}, log, stats, show=False)
                continue
            if os.path.basename(out) in skip_stems and not args.overwrite:
                _record({"file": rel, "status": "skipped-remote"}, log,
                        stats, show=False)
                continue
            tasks.append((args.data_dir, rel, out, max_width))
        print(f"{len(tasks)} to process, {stats['skipped']} already done.", flush=True)

        print("loading holistic model ...", flush=True)
        t_all = time.time()
        try:
            if args.workers > 1:
                run_farm(tasks, args, log, stats)
            else:
                _init_worker()
                print("model ready.", flush=True)
                for task in tqdm(tasks, desc="extracting", unit="video"):
                    _record(_process_one(task), log, stats)
        except KeyboardInterrupt:
            print("\ninterrupted -- partial progress is saved, re-run to continue.", flush=True)
    print(f"\nfinished in {time.time() - t_all:.0f}s -- " + ", ".join(
        f"{k}={v}" for k, v in stats.items()), flush=True)
    print(f"progress log: {log_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
