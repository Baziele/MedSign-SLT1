#!/usr/bin/env python3
"""MediaPipe Tasks benchmark with GPU delegate (T4).
Pose + Hands + Face per frame, short test + 8 sustained iters.
"""
import os
import time

import cv2

BDIR = "/kaggle/input/mp-bundles"
VIDS = ["/kaggle/input/mp-bundles/b1.mp4",
        "/kaggle/input/mp-bundles/b2.mp4"]

BUNDLES = {
    "pose.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",  # noqa
    "hand.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",  # noqa
    "face.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",  # noqa
}


def ensure_bundles():
    import urllib.request
    os.makedirs("/tmp/bundles", exist_ok=True)
    for name, url in BUNDLES.items():
        dest = os.path.join("/tmp/bundles", name)
        if not os.path.exists(dest) or os.path.getsize(dest) < 1000000:
            print("downloading", name, flush=True)
            urllib.request.urlretrieve(url, dest)
    return "/tmp/bundles"


def make_landmarkers():
    import mediapipe as mp_sol
    from mediapipe.tasks import python as mp_py
    from mediapipe.tasks.python import vision as mp_vision

    bdir = ensure_bundles()

    try:
        delegate = mp_py.BaseOptions.Delegate.GPU
    except Exception:
        delegate = None
    print("delegate:", delegate, flush=True)

    def base(path):
        kw = {"model_asset_path": path}
        if delegate is not None:
            kw["delegate"] = delegate
        return mp_py.BaseOptions(**kw)

    pose = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=base(os.path.join(bdir, "pose.task")),
            running_mode=mp_vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5))
    hand = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=base(os.path.join(bdir, "hand.task")),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2, min_hand_detection_confidence=0.5))
    face = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=base(os.path.join(bdir, "face.task")),
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
        pose.detect_for_video(img, i)
        hand.detect_for_video(img, i)
        face.detect_for_video(img, i)
    dt = time.time() - t0
    return len(frames), dt


def main():
    import subprocess
    import sys
    print(subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip(), flush=True)
    print("== SHORT ==", flush=True)
    for v in VIDS:
        n, dt = bench_one(v)
        print(f"{v}: {n} frames in {dt:.1f}s ({dt/n*1000:.0f} ms/frame)",
              flush=True)
    print("== SUSTAINED x8 ==", flush=True)
    t_all = time.time()
    for i in range(8):
        n, dt = bench_one(VIDS[-1])
        print(f"iter {i+1:02d}: {n} frames in {dt:.1f}s "
              f"({dt/n*1000:.0f} ms/frame)", flush=True)
    print(f"sustained total: {time.time()-t_all:.0f}s", flush=True)


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
