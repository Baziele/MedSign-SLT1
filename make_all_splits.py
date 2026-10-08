"""Combine GH + multiview splits into all_{train,val} (tests stay separate).

  all_train = data/splits/train.csv (7,337 GH) + data/splits/mv_train.csv (57,500 MV)
  all_val   = data/splits/val.csv   (994 GH, unseen-text) + data/splits/mv_val.csv (2,500 MV, take-holdout)
  Pooled val (3,494) = bigger selection signal; GH test_T/test_R + MV test
  stay untouched for final reporting.
  Also builds tokenizer_all.json from all_train texts (MV adds ~7% new words;
  without it those targets train as <unk>).

Writes data/splits/all_{train,val}.csv + data/splits/all_summary.json + tokenizer_all.json.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPLITS = ROOT / "data" / "splits"
COLS = ["video", "stem", "sentence_id", "text", "category", "signer", "cam", "rep"]


def load(name):
    with open(SPLITS / f"{name}.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.setdefault("cam", "")
        r.setdefault("rep", "")
    return rows


def main():
    gh_train, gh_val = load("train"), load("val")
    mv_train, mv_val = load("mv_train"), load("mv_val")
    assert len(gh_train) == 7337 and len(gh_val) == 994, "GH splits changed!"
    assert len(mv_train) >= 50000 and len(mv_val) >= 2000, "MV splits look wrong"
    train = gh_train + mv_train
    val = gh_val + mv_val
    for name, rows in [("all_train", train), ("all_val", val)]:
        with open(SPLITS / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"{name}: {len(rows)} clips")

    from tokenizer import Tokenizer
    tok = Tokenizer.build([r["text"] for r in train])
    tok.save(ROOT / "tokenizer_all.json")
    old_n = len(json.load(open(ROOT / "tokenizer.json"))["tok2id"])
    print(f"tokenizer_all.json: {len(tok)} tokens (GH vocab was {old_n}, "
          f"+{len(tok) - old_n} new)")

    json.dump({"train": len(train), "val": len(val),
               "train_GH": len(gh_train), "train_MV": len(mv_train),
               "val_GH_unseen_text": len(gh_val), "val_MV_take_holdout": len(mv_val),
               "quarantined_npz": 8,
               "tests_untouched": ["test_T", "test_R", "mv_test"]},
              open(SPLITS / "all_summary.json", "w"), indent=2)
    print("wrote data/splits/all_{train,val}.csv + all_summary.json + tokenizer_all.json")


if __name__ == "__main__":
    main()
