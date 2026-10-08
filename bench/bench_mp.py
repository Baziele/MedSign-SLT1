#!/usr/bin/env python3
"""MediaPipe Holistic benchmark: same settings as wextract.py.
Usage: bench_mp.py video1.mp4 [video2.mp4 ...]
Reports per-video seconds + ms/frame (inference only, frames pre-decoded).
"""
import sys
import time

import cv2
import mediapipe as mp


def main(paths):
    h = mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        enable_segmentation=False, refine_face_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    for p in paths:
        cap = cv2.VideoCapture(p)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        t0 = time.time()
        for f in frames:
            h.process(f)
        dt = time.time() - t0
        print(f"{p}: {len(frames)} frames in {dt:.1f}s "
              f"({dt/len(frames)*1000:.0f} ms/frame)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
