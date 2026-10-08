"""NNS NWT extraction progress: counts, rate, ETA (mirrors jwstat.py). Run from repo root."""
import glob, json, os, time
ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(ROOT))
JW = os.path.join(REPO, "data", "jw", "nwt-nns")
npz = glob.glob(os.path.join(JW, "keypoints", "*.npz"))
TOTAL = 697
lp = os.path.join(JW, "progress.jsonl")
done = failed = 0
t0 = t1 = None
if os.path.exists(lp):
    for line in open(lp):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("status") == "done":
            done += 1
        elif d.get("status") == "failed":
            failed += 1
        if "ts" in d:
            t = time.mktime(time.strptime(d["ts"], "%Y-%m-%dT%H:%M:%S"))
            t0 = t if t0 is None else min(t0, t)
            t1 = max(t1 or 0, t)
rate = (done / (t1 - t0) * 3600) if t0 and t1 and t1 > t0 and done else 0
left = TOTAL - len(npz)
eta = (left / rate) if rate else float("inf")
print(f"npz on disk: {len(npz)}/{TOTAL}  log done: {done}  failed: {failed}")
print(f"rate: {rate:.0f} vids/h  left: {left}  ETA: "
      f"{eta:.1f}h" if rate else "ETA: -- (no rate yet)")
