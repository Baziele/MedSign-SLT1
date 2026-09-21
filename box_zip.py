"""Zip keypoints/*.npz for shipment. Writes heartbeat to zip_status.txt."""
import os
import zipfile

SRC = r"C:\Users\Baziele\Documents\Code\medsign-multiview\keypoints"
DST = r"C:\Users\Baziele\Documents\Code\medsign-multiview\npz_ship.zip"
STATUS = r"C:\Users\Baziele\Documents\Code\medsign-multiview\zip_status.txt"


def beat(msg):
    with open(STATUS, "w") as f:
        f.write(msg)
    print(msg, flush=True)


files = sorted(f for f in os.listdir(SRC) if f.endswith(".npz"))
mode = "w"
have = set()
if os.path.exists(DST):
    with zipfile.ZipFile(DST, "r") as z:
        have = set(z.namelist())
    mode = "a"
beat(f"start {len(files)} have={len(have)}")
with zipfile.ZipFile(DST, mode, zipfile.ZIP_STORED) as z:
    done = 0
    for i, f in enumerate(files):
        if f in have:
            continue
        z.write(os.path.join(SRC, f), f)
        done += 1
        if done % 500 == 0:
            beat(f"{i + 1}/{len(files)}")
beat(f"done files={len(files)} size={os.path.getsize(DST)}")
