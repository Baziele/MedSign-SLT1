#!/usr/bin/env python3
"""Classic Holistic benchmark on Kaggle (needs run internet).
Bootstraps an isolated venv, installs mediapipe 0.10.18 with resolved
deps, re-execs into it, then runs short + sustained benchmark.
"""
import os
import sys

VENV = "/tmp/mpvenv"
VIDS = ["/kaggle/input/mp-bundles/b1.mp4",
        "/kaggle/input/mp-bundles/b2.mp4"]


def main():
    import multiprocessing as mp_mod
    import time

    import cv2
    import mediapipe as mp_sol

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
        h = mp_sol.solutions.holistic.Holistic(
            static_image_mode=False, model_complexity=1,
            enable_segmentation=False, refine_face_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        frames = load(path)
        t0 = time.time()
        for f in frames:
            h.process(f)
        dt = time.time() - t0
        return len(frames), dt

    print("cpus:", os.cpu_count(), flush=True)
    print("mp:", mp_sol.__version__, flush=True)
    print("== SHORT ==", flush=True)
    for v in VIDS:
        n, dt = bench_one(v)
        print(f"{v}: {n} frames in {dt:.1f}s ({dt/n*1000:.0f} ms/frame)",
              flush=True)
    print("== SUSTAINED: b2 x25 on 4 workers ==", flush=True)
    t_all = time.time()
    with mp_mod.Pool(4) as pool:
        for i, (n, dt) in enumerate(pool.imap_unordered(bench_one, [VIDS[-1]] * 25)):
            print(f"iter {i+1:02d}: {n} frames in {dt:.1f}s "
                  f"({dt/n*1000:.0f} ms/frame)", flush=True)
    print(f"sustained total: {time.time()-t_all:.0f}s for 25 iters",
          flush=True)


if __name__ == "__main__":
    if not os.environ.get("HBENCH_VENV"):
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        subprocess.check_call(
            [sys.executable, "-m", "venv", "--system-site-packages",
             "--without-pip", VENV])
        vp = os.path.join(VENV, "bin", "python")
        subprocess.check_call(
            [vp, "/kaggle/input/mp-bundles/get-pip.py", "-q"])
        subprocess.check_call(
            [vp, "-m", "pip", "-q", "install", "mediapipe==0.10.18",
             "opencv-contrib-python", "numpy==1.26.4"])
        # mp0.10.18 metadata caps protobuf<5, but its graphs only parse
        # with protobuf>=5 (proven on Ubuntu server) -- upgrade past the cap
        subprocess.check_call(
            [vp, "-m", "pip", "-q", "install", "--upgrade", "protobuf"])
        os.environ["HBENCH_VENV"] = "1"
        os.execv(vp, [vp] + sys.argv)
    main()
