#!/usr/bin/env python3
"""MediaPipe Tasks benchmark: Pose + Hands + Face landmarkers per frame.
Equivalent workload to Holistic (same model family). Short test + 25x
sustained run on 4 workers, per-iteration times expose throttling.
"""
import multiprocessing as mp
import os
import time
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent.parent
BDIR = REPO / "bench" / "kbundles"
MVD = REPO / "data" / "multiview" / "mv-data"
VIDS = [str(f) for f in (MVD / "train/train/sentence_id_1/001_S1_cam3_rep1.mp4",
                          MVD / "test/test/sentence_id_100/100_S5_cam2_rep4.mp4")
        if os.path.exists(f)]


def make_landmarkers():
    import mediapipe as mp_sol
    from mediapipe.tasks import python as mp_py
    from mediapipe.tasks.python import vision as mp_vision

    base = mp_py.BaseOptions
    pose = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=base(model_asset_path=os.path.join(BDIR, "pose.task")),
            running_mode=mp_vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5))
    hand = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=base(model_asset_path=os.path.join(BDIR, "hand.task")),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2, min_hand_detection_confidence=0.5))
    face = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=base(model_asset_path=os.path.join(BDIR, "face.task")),
            running_mode=mp_vision.RunningMode.VIDEO,
            min_face_detection_confidence=0.5))
    return pose, hand, face


def load(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def bench_one(path):
    import mediapipe as mp_sol
    pose, hand, face = make_landmarkers()
    frames = load(path)
    t0 = time.time()
    for i, f in enumerate(frames):
        img = mp_sol.Image(image_format=mp_sol.ImageFormat.SRGB, data=f)
        t = i
        pose.detect_for_video(img, t)
        hand.detect_for_video(img, t)
        face.detect_for_video(img, t)
    dt = time.time() - t0
    return len(frames), dt


def main():
    print("cwd:", os.getcwd(), flush=True)
    for d in [".", "/kaggle/src", "/kaggle/working"]:
        try:
            print(d, sorted(os.listdir(d))[:15], flush=True)
        except Exception as e:
            print(d, "ERR", str(e)[:80], flush=True)
    print("input root:", sorted(os.listdir("/kaggle/input"))
          if os.path.isdir("/kaggle/input") else "NO /kaggle/input",
          flush=True)
    for v in VIDS:
        print("exists:", v, os.path.exists(v), flush=True)
    print("== SHORT ==", flush=True)
    for v in VIDS:
        n, dt = bench_one(v)
        print(f"{v}: {n} frames in {dt:.1f}s ({dt/n*1000:.0f} ms/frame)",
              flush=True)
    print("== SUSTAINED: b2 x8 on 8 workers ==", flush=True)
    v = VIDS[-1]
    t_all = time.time()
    with mp.Pool(8) as pool:
        for i, (n, dt) in enumerate(pool.imap_unordered(bench_one, [v] * 8)):
            print(f"iter {i+1:02d}: {n} frames in {dt:.1f}s "
                  f"({dt/n*1000:.0f} ms/frame)", flush=True)
    print(f"sustained total: {time.time()-t_all:.0f}s for 8 iters",
          flush=True)


if __name__ == "__main__":
    try:
        import mediapipe  # noqa
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call(
            [sys.executable, "-m", "pip", "-q", "install", "--no-index",
             "--find-links", "/kaggle/input/mp-wheels",
             "--find-links", "/kaggle/input/mp-wheel", "mediapipe==1.0.1"])
    main()
