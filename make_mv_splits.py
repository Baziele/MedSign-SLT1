"""Build train/val/test CSVs for the multiview 500-sentence set.

Source of truth (already on disk, dataset-author split):
  data/multiview/mv-data/train/train/sentence_id_<N>/<NNN>_S<S>_cam<C>_rep<R>.mp4  (S1-S4, 120/sent)
  data/multiview/mv-data/test/test/sentence_id_<N>/<NNN>_S<S>_cam<C>_rep<R>.mp4    (S5, 30/sent)
  data/multiview/mv-data/Metadata.csv  (SENTENCE ID -> SENTENCE, HOSPITAL DEPARTMENT)
  data/multiview/keypoints/train_train_sentence_id_<N>_<N>_S<S>_cam<C>_rep<R>.npz (60k)
  data/multiview/keypoints/test_test_sentence_id_<N>_<N>_S<S>_cam<C>_rep<R>.npz  (15k)

Split:
  test  = all of S5 (15,000) — true signer-holdout, decoded once at the end.
  val   = S4 rep6, all cams (2,500) — take-holdout within a seen signer.
          Tracks training cheaply (~50s decode); optimistic vs S5 test.
  train = all other S1-S4 (57,500).

Writes splits/mv_{train,val,test}.csv + splits/mv_summary.json.
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MV = ROOT / "data" / "multiview" / "mv-data"
KP = ROOT / "data" / "multiview" / "keypoints"
SPLITS = ROOT / "data" / "splits"

VID = re.compile(r"(\d+)_S(\d)_cam(\d)_rep(\d)\.mp4")


def load_meta():
    meta = {}
    with open(MV / "Metadata.csv") as f:
        for r in csv.DictReader(f):
            sid = int(r["SENTENCE ID"])
            meta[sid] = (r["SENTENCE"], r["HOSPITAL DEPARTMENT"])
    assert len(meta) == 500, f"want 500 sentences, got {len(meta)}"
    return meta


def _npz_ok(stem):
    """True iff data/multiview/keypoints/<stem>.npz opens as a zip (catches truncated copies)."""
    import zipfile
    try:
        zipfile.ZipFile(KP / f"{stem}.npz").close()
        return True
    except Exception:
        return False


def scan(sub):
    """Walk mv-data/<sub>/<sub>/sentence_id_N/*.mp4 -> row dicts.

    Rows whose .npz is missing/corrupt (see data/multiview/keypoints_corrupt/) are skipped
    and reported — a dead take must never kill a training run.
    """
    rows, skipped = [], []
    base = MV / sub / sub
    sdirs = sorted([p for p in base.iterdir() if p.is_dir()],
                   key=lambda p: int(p.name.split("_")[-1]))
    for sdir in sdirs:
        sid = int(sdir.name.split("_")[-1])
        for mp4 in sorted(sdir.glob("*.mp4")):
            m = VID.fullmatch(mp4.name)
            assert m, f"bad name {mp4}"
            num, signer, cam, rep = int(m[1]), f"S{m[2]}", int(m[3]), int(m[4])
            assert num == sid, f"{mp4}: file id {num} != folder {sid}"
            stem = (f"{sub}_{sub}_sentence_id_{sid}_{num:03d}"
                    f"_S{m[2]}_cam{cam}_rep{rep}")
            if not _npz_ok(stem):
                skipped.append(stem)
                continue
            rows.append({"video": f"{sub}/{sdir.name}/{mp4.name}", "stem": stem,
                         "sentence_id": sid, "signer": signer,
                         "cam": cam, "rep": rep})
    return rows, skipped


def main():
    meta = load_meta()
    train_all, skip_tr = scan("train")
    test, skip_te = scan("test")
    skipped = skip_tr + skip_te
    print(f"videos: {len(train_all)+len(test)} ok + {len(skipped)} quarantined "
          f"(see data/multiview/keypoints_corrupt/)")
    assert len(train_all) + len(test) + len(skipped) == 75000
    assert not any("S4" in s and "rep6" in s for s in skipped), \
        f"val take quarantined: {[s for s in skipped if 'S4' in s and 'rep6' in s]}"

    for r in train_all + test:
        r["text"], r["category"] = meta[r["sentence_id"]]

    val = [r for r in train_all if r["signer"] == "S4" and r["rep"] == 6]
    train = [r for r in train_all
             if not (r["signer"] == "S4" and r["rep"] == 6)]
    assert len(val) == 2500, len(val)
    assert len(train) + len(val) == len(train_all)
    assert {r["sentence_id"] for r in val} == set(range(1, 501))

    SPLITS.mkdir(exist_ok=True)
    for name, rows in [("mv_train", train), ("mv_val", val), ("mv_test", test)]:
        with open(SPLITS / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, ["video", "stem", "sentence_id", "text",
                                   "category", "signer", "cam", "rep"])
            w.writeheader()
            w.writerows(rows)
        print(f"{name}: {len(rows)} clips")

    summary = {
        "train": len(train), "val": len(val), "test": len(test),
        "quarantined": skipped,
        "val_design": "S4 rep6 all cams (take-holdout, seen signer; "
                      "optimistic vs S5 signer-holdout test)",
        "per_sentence_train_takes": "~115", "per_sentence_note":
            "120 takes/sent in S1-S4 minus 5 val takes (S4 rep6, 5 cams), "
            "minus rare quarantined takes",
    }
    import json
    json.dump(summary, open(SPLITS / "mv_summary.json", "w"), indent=2)
    print("wrote splits/mv_{train,val,test}.csv + mv_summary.json")


if __name__ == "__main__":
    main()
