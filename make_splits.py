"""Frozen splits for SignTalk-GH keypoints -> English prototype.

Design (agreed with user):
- Group by NORMALIZED text (lower, strip, collapse whitespace), not Sentence ID.
  78 exact-dup texts / 9 norm texts span >1 category -> one primary category
  per text (most videos, tie-break smallest ID) for stratification.
- Stratify by category (stripped): every split sees all 26 categories.
- All signers (A-E) in every split: no signer holdout (GSL style variation
  is training signal, not a holdout dimension).
- test-T / val = UNSEEN texts. test-R = seen texts, held-out renditions
  (robustness to signer-style variation). train = rest.
- Guards: drop 1088Z video; pin 3006C + 2945B outlier videos to train by
  forcing their text groups into the train pool.

Outputs: data/splits/{train,val,test_T,test_R}.csv + data/splits/summary.json
CSV cols: video,stem,sentence_id,text,category,signer
"""
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VID_DIR = ROOT / "data" / "multiview" / "Medsign" / "archive" / "SignTalk-GH" / "Videos"
META = ROOT / "data" / "multiview" / "Medsign" / "archive" / "SignTalk-GH" / "Metadata.xlsx"
OUT_DIR = ROOT / "data" / "splits"

SEED = 42
TRAIN_FRAC, VAL_FRAC, TESTT_FRAC = 0.80, 0.10, 0.10
TEST_R_FRAC = 0.07  # of train-pool videos, constraint: group keeps >=1 in train

DROP_FILES = {"1088Z.mkv", "1088Z.mp4"}
QUARANTINE_FILES = {"3006C.mp4", "2945B.mp4"}

VID_PAT = re.compile(r"^(\d+)([A-Z])\.(mp4|mkv)$")


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def load_meta():
    # no pandas in .venv312 at build time? try pandas, fall back to openpyxl
    try:
        import pandas as pd

        df = pd.read_excel(META)
        rows = df.to_dict("records")
    except ImportError:
        from openpyxl import load_workbook

        wb = load_workbook(META, read_only=True, data_only=True)
        ws = wb.active
        header = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
        rows = [dict(zip(header, [c.value for c in r])) for r in ws.iter_rows(min_row=2)]
    meta = {}
    for r in rows:
        sid = int(r["Sentence ID"])
        meta[sid] = {
            "text": str(r["Sentence Text"]).strip(),
            "norm": norm_text(r["Sentence Text"]),
            "category": str(r["Category"]).strip(),
        }
    return meta


def main():
    rng = random.Random(SEED)
    meta = load_meta()

    videos = []  # dicts
    skipped = Counter()
    for p in sorted(VID_DIR.iterdir()):
        m = VID_PAT.match(p.name)
        if not m:
            skipped["nonmatching_name"] += 1
            continue
        if p.name in DROP_FILES:
            skipped["dropped_Z"] += 1
            continue
        sid, signer = int(m.group(1)), m.group(2)
        if sid not in meta:
            skipped["no_metadata"] += 1
            continue
        info = meta[sid]
        videos.append(
            {
                "video": p.name,
                "stem": p.stem,
                "sentence_id": sid,
                "text": info["text"],
                "norm": info["norm"],
                "category": info["category"],
                "signer": signer,
            }
        )

    # group norms -> videos; primary category = most videos, tie-break min sid
    groups = defaultdict(list)
    for v in videos:
        groups[v["norm"]].append(v)
    print(f"text groups with video: {len(groups)}, videos kept: {len(videos)}")

    quarantine_norms = {v["norm"] for v in videos if v["video"] in QUARANTINE_FILES}
    print(f"quarantine groups forced to train: {len(quarantine_norms)}")

    group_cats = {}
    for norm, vs in groups.items():
        cc = Counter(v["category"] for v in vs)
        top = max(cc.values())
        cands = sorted(c for c, n in cc.items() if n == top)
        if len(cands) == 1:
            group_cats[norm] = cands[0]
        else:  # tie-break: category of smallest sentence_id
            group_cats[norm] = min(vs, key=lambda v: v["sentence_id"])["category"]

    # stratify: per category, shuffle groups, allocate 80/10/10; quarantine -> train
    cat_groups = defaultdict(list)
    for norm in groups:
        if norm in quarantine_norms:
            continue
        cat_groups[group_cats[norm]].append(norm)

    train_norms, val_norms, testt_norms = set(quarantine_norms), set(), set()
    for cat, norms in sorted(cat_groups.items()):
        rng.shuffle(norms)
        n = len(norms)
        n_test = max(1, round(n * TESTT_FRAC)) if n >= 5 else 0
        n_val = max(1, round(n * VAL_FRAC)) if n >= 5 else 0
        testt_norms.update(norms[:n_test])
        val_norms.update(norms[n_test : n_test + n_val])
        train_norms.update(norms[n_test + n_val :])

    # video pools from text split
    train_pool = [v for v in videos if v["norm"] in train_norms]
    val_vs = [v for v in videos if v["norm"] in val_norms]
    testt_vs = [v for v in videos if v["norm"] in testt_norms]

    # test-R: stratified ~7% of train_pool by (category, signer), keep >=1/group in train
    by_group = defaultdict(list)
    for v in train_pool:
        by_group[v["norm"]].append(v)
    # eligible: groups with >=2 videos
    candidates = [v for v in train_pool if len(by_group[v["norm"]]) >= 2]
    # stratify by (cat, signer): sample proportionally
    strata = defaultdict(list)
    for v in candidates:
        strata[(v["category"], v["signer"])].append(v)
    for s in strata.values():
        rng.shuffle(s)
    n_target = round(len(train_pool) * TEST_R_FRAC)
    # round-robin across strata for balance, greedily respecting the
    # keep->=1-in-train constraint
    test_r_vs, test_r_keys = [], set()
    strata_lists = [strata[k] for k in sorted(strata)]
    progressed = True
    while len(test_r_vs) < n_target and progressed:
        progressed = False
        for s in strata_lists:
            if len(test_r_vs) >= n_target or not s:
                continue
            v = s.pop(0)
            selected_in_group = sum(1 for t in test_r_vs if t["norm"] == v["norm"])
            if len(by_group[v["norm"]]) - selected_in_group - 1 >= 1:
                test_r_vs.append(v)
                test_r_keys.add((v["norm"], v["video"]))
                progressed = True
    test_r_ids = {(v["norm"], v["video"]) for v in test_r_vs}
    train_vs = [v for v in train_pool if (v["norm"], v["video"]) not in test_r_ids]

    splits = {"train": train_vs, "val": val_vs, "test_T": testt_vs, "test_R": test_r_vs}

    # leakage checks
    train_norm_set = {v["norm"] for v in train_vs} | {v["norm"] for v in test_r_vs}
    assert not (set(v["norm"] for v in val_vs) & train_norm_set), "val text leaks into train"
    assert not (set(v["norm"] for v in testt_vs) & train_norm_set), "test_T text leaks into train"
    assert not (set(v["norm"] for v in val_vs) & set(v["norm"] for v in testt_vs)), "val/test_T overlap"
    for v in test_r_vs:  # every test-R text still has >=1 video in train
        assert any(
            t["norm"] == v["norm"] for t in train_vs
        ), f"test-R text orphaned: {v['norm'][:40]}"

    OUT_DIR.mkdir(exist_ok=True)
    for name, vs in splits.items():
        vs_sorted = sorted(vs, key=lambda v: (v["sentence_id"], v["video"]))
        with open(OUT_DIR / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["video", "stem", "sentence_id", "text", "category", "signer"]
            )
            w.writeheader()
            for v in vs_sorted:
                w.writerow({k: v[k] for k in w.fieldnames})

    summary = {
        "seed": SEED,
        "text_fracs": {"train": TRAIN_FRAC, "val": VAL_FRAC, "test_T": TESTT_FRAC},
        "test_R_frac_of_train_pool": TEST_R_FRAC,
        "guards": {
            "dropped": sorted(DROP_FILES),
            "quarantined_to_train": sorted(QUARANTINE_FILES),
            "skipped": dict(skipped),
        },
        "counts": {k: len(v) for k, v in splits.items()},
        "text_groups": {
            "train": len(train_norms),
            "val": len(val_norms),
            "test_T": len(testt_norms),
        },
        "per_split_category": {
            k: dict(Counter(v["category"] for v in vs)) for k, vs in splits.items()
        },
        "per_split_signer": {k: dict(Counter(v["signer"] for v in vs)) for k, vs in splits.items()},
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    total = sum(len(v) for v in splits.values())
    print(f"videos: train={len(train_vs)} val={len(val_vs)} test_T={len(testt_vs)} test_R={len(test_r_vs)} total={total}")
    for k, vs in splits.items():
        print(k, dict(Counter(v["signer"] for v in vs)))
    print("leakage checks passed: val/test_T text-disjoint from train; test_R texts all in train")


if __name__ == "__main__":
    main()
