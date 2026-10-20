#!/usr/bin/env python3
"""Extract MediaPipe Holistic keypoints for JW NWT Bible-chapter videos.

Same landmarks/pipeline as wextract.py (478 face + 33 pose + 2x21 hands).
Input:  <data-pipeline>/nwt-ghs/<book>/chNNN.mp4 (+ chNNN.json verse metadata).
Output: data/jw/nwt-ghs/keypoints/book<BB>_<slug>_ch<NNN>.npz with verse
metadata embedded. Resumable (skip-if-exists), crash-proof wave execution,
progress.jsonl log.

Usage:
  .venv312/bin/python -u scripts/extract/jwextract.py [--workers 8]
  .venv312/bin/python -u scripts/extract/jwextract.py --src <videos> --dst <outdir>
      (e.g. --src /path/to/nns --dst data/jw/nwt-nns)
Logs: console (redirect to extract log) + <dst>/progress.jsonl
"""
import argparse
import glob
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
REPO = os.path.dirname(os.path.dirname(ROOT))
NWT = "/Users/mazen/dev/projects/medsign-data-pipeline/nwt-ghs"
JW = os.path.join(REPO, "data", "jw", "nwt-ghs")
OUT_DIR = os.path.join(JW, "keypoints")

_HOLISTIC = None


def _init_worker():
    global _HOLISTIC
    _HOLISTIC = make_holistic()


def _meta_payload(meta):
    """Per-corpus metadata embedded in each npz. NWT chapters carry verse
    timings; MWB weeks carry section timings + paragraph ids."""
    base = {"docid": np.array(str(meta.get("docid") or ""))}
    if meta.get("kind") == "mwb":
        return {**base,
                "issue": np.array(meta["issue"]),
                "title": np.array(meta["title"]),
                "sections_json": np.array(json.dumps(meta["sections"]))}
    return {**base,
            "book": np.array(meta["book"]),
            "booknum": np.array(meta["booknum"], dtype=np.int32),
            "chapter": np.array(meta["chapter"], dtype=np.int32),
            "verses_json": np.array(json.dumps(meta["verses"]))}


def _process_jw(task):
    src, out, meta, max_width, stag, skip_secs = task
    t0 = time.time()
    try:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError("cv2 could not open video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        skip_frames = int(round((skip_secs or 0) * fps))
        if skip_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
            # mp4 seeks land on keyframes; record where decoding resumes
            skip_frames = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or skip_frames)
        # Picture sections (Art*) hold static images, not signing: jump over
        # them entirely instead of paying Holistic cost per frame. Ranges
        # come from section timestamps, so no detection is needed upfront.
        drop = []
        if meta.get("kind") == "mwb":
            for s in meta.get("sections", []):
                if str(s.get("label", "")).startswith("Art"):
                    a = int(s["start_s"] * fps)
                    b = int((s["start_s"] + s["dur_s"]) * fps) + 1
                    drop.append((a, b))
        drop.sort()

        def in_drop(fr):
            for a, b in drop:
                if a <= fr < b:
                    return True
                if fr < a:
                    break
            return False

        def drop_end(fr):
            for a, b in drop:
                if a <= fr < b:
                    return b
            return fr
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = max_width / width if (max_width and width > max_width) else 1.0
        F, P, PV, W, HL, HR = [], [], [], [], [], []
        fp, pp, hlp, hrp = [], [], [], []
        kept = []
        n = skip_frames
        while True:
            if in_drop(n):
                n = drop_end(n)
                cap.set(cv2.CAP_PROP_POS_FRAMES, n)
                continue
            ok, frame = cap.read()
            if not ok:
                break
            actual = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0) - 1
            if in_drop(actual):
                # seek landed inside a drop range (keyframe slop)
                n = actual + 1
                continue
            n = actual + 1
            kept.append(actual)
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
        cap.release()
        if not F:
            raise RuntimeError("video contained 0 readable frames")
        n = len(F)
        start = kept[0]
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
            # kept[] holds the original video frame behind each npz frame
            # (leading skips + dropped picture ranges removed); absolute
            # video time of npz frame i is frame_idx[i] / fps
            "frame_idx": np.array(kept, dtype=np.int32),
            "skip_seconds": np.array(round(start / fps, 3)),
            "start_frame": np.array(start, dtype=np.int32),
            "video_file": np.array(os.path.basename(src)),
            "source": np.array(stag),
            **_meta_payload(meta),
        }
        tmp = out + ".tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **data)
        os.replace(tmp, out)
        dt = time.time() - t0
        return {"file": meta["stem"], "status": "done", "frames": n,
                "seconds": round(dt, 1),
                "pose_hit": round(float(np.array(pp).mean()), 3)}
    except Exception as e:
        return {"file": meta["stem"], "status": "failed", "error": str(e)}


import wextract as _wx  # noqa: F401  (import ensures same pipeline constants)


import signal as _signal


def _log_signal(signum, frame):
    # Diagnostic trap: external kills (reboots/session cleanup) arrive as
    # SIGTERM/SIGHUP and otherwise leave no trace. Log and exit loudly.
    # SIGKILL/OOM cannot be trapped: no message then means SIGKILL or power.
    print(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%S')}] "
          f"received signal {signum} -- shutting down", flush=True)
    raise SystemExit(128 + signum)


for _s in (_signal.SIGTERM, _signal.SIGHUP):
    _signal.signal(_s, _log_signal)
del _s


def main():
    ap = argparse.ArgumentParser(description="JW NWT keypoint extraction.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-width", type=int, default=1280)
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--src", default=NWT,
                    help="dir of book*/ch*.mp4 (+ch*.json) videos")
    ap.add_argument("--kind", choices=("nwt", "mwb"), default="nwt",
                    help="nwt: book*/ch*.mp4 + verse JSONs; "
                         "mwb: <issue>/*.mp4 + section JSONs")
    ap.add_argument("--skip-seconds", type=float, default=0.0,
                    help="seek past this many seconds before tracking "
                         "(e.g. 4 to skip title cards); offset recorded "
                         "in each npz as skip_seconds/start_frame")
    ap.add_argument("--dst", default=os.path.join(
        REPO, "data", "jw", "nwt-ghs"),
                    help="output dir (keypoints/ + progress.jsonl land here)")
    args = ap.parse_args()
    JW = args.dst
    OUT_DIR = os.path.join(JW, "keypoints")
    stag = os.path.basename(os.path.normpath(JW))

    pat = "ch*.mp4" if args.kind == "nwt" else "*.mp4"
    mp4s = sorted(glob.glob(os.path.join(args.src, "*", pat)))
    if args.kind == "mwb":
        # section-based corpora may also be flat (lfb: lessonNNN.mp4)
        mp4s += sorted(glob.glob(os.path.join(args.src, pat)))
        mp4s = sorted(set(mp4s))
    if args.max_videos:
        mp4s = mp4s[:args.max_videos]
    print(f"{len(mp4s)} {args.kind} videos found ({stag})", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    def out_of(src):
        book = os.path.basename(os.path.dirname(src))
        stem = os.path.splitext(os.path.basename(src))[0]
        return os.path.join(OUT_DIR, f"{book}_{stem}.npz")

    def meta_of(src):
        js = os.path.splitext(src)[0] + ".json"
        d = json.load(open(js))
        book = os.path.basename(os.path.dirname(src))
        stem = os.path.splitext(os.path.basename(src))[0]
        if args.kind == "mwb":
            return {"stem": f"{book}/{stem}", "kind": "mwb",
                    "issue": d.get("issue", book),
                    "title": d.get("title", stem),
                    "docid": d.get("docid"),
                    "sections": d.get("sections", [])}
        return {"stem": f"{book}/{stem}",
                "book": d["book"], "booknum": d["booknum"],
                "chapter": d["chapter"], "docid": d.get("docid"),
                "verses": d.get("verses", [])}

    _wx._process_one = lambda task: _process_jw(task)  # noqa
    _wx._init_worker = _init_worker  # noqa
    import concurrent.futures as cf
    from concurrent.futures.process import BrokenProcessPool
    tasks = []
    skipped = 0
    for src in mp4s:
        out = out_of(src)
        if os.path.exists(out) and not args.overwrite:
            skipped += 1
            continue
        tasks.append((src, (src, out, meta_of(src),
                            args.max_width if args.max_width > 0 else None,
                            stag, args.skip_seconds)))
    print(f"{len(tasks)} to process, {skipped} already done.", flush=True)

    print("loading holistic model ...", flush=True)
    t_all = time.time()
    stats = {"done": 0, "skipped": skipped, "failed": 0}
    log_path = os.path.join(JW, "progress.jsonl")
    try:
        with open(log_path, "a") as log:
            from wextract import _record as rec
            queue = {}
            for src, tt in tasks:
                queue[out_of(src)] = [tt, 0]
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
                    wave = list(queue)[:32]
                    ex = cf.ProcessPoolExecutor(
                        max_workers=args.workers, initializer=_init_worker)
                    fut2o = {ex.submit(_process_jw, queue[o][0]): o
                             for o in wave}
                    try:
                        # NOTE: as_completed timeout is a TOTAL deadline for
                        # the whole wave, not per-video idle. Full-chapter
                        # videos run 10-20+ min each; a 1200s cap aborts
                        # every long-video wave and discards in-flight
                        # results, redoing them forever. 7200s covers a
                        # full 32-video wave of the longest chapters.
                        for fut in cf.as_completed(fut2o, timeout=7200):
                            o = fut2o[fut]
                            try:
                                r_ = fut.result()
                            except Exception as e:
                                r_ = {"file": o, "status": "failed",
                                      "error": f"worker exception: {e}"}
                            rec(r_, log, stats)
                            queue.pop(o, None)
                            pbar.update(1)
                    except BrokenProcessPool:
                        tqdm.write("  [pool] worker crashed -- wave retries, "
                                   f"{len(queue)} pending")
                        for o in fut2o.values():
                            if o in queue:
                                queue[o][1] += 1
                    except TimeoutError:
                        tqdm.write("  [pool] 2h wave timeout -- next wave")
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
