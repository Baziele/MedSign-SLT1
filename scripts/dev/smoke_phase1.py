"""Phase-1 pipeline smoke test (CPU, tiny batch): not a training run.

Verifies, without launching a real run:
  1. rows / groups / mixture sampling work on real splits
  2. feature access + two-view pair dataset + collate produce the right shapes
  3. VideoText forward + symmetric InfoNCE + SignCL term compute and backprop
  4. stage-2 key mapping: pose tower state drops into SignTransformerV2 exactly

Run: .venv312/bin/python scripts/dev/smoke_phase1.py
"""
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import os
os.environ["HF_HOME"] = str(Path(__file__).resolve().parents[2] / "models" / "hf")

from pipeline import data as D
from pipeline.pose_text import VideoText, cl_losses


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    return cond


def main():
    ok = True

    # 1. data plumbing -----------------------------------------------------
    rows = D.load_main_rows("all_train")
    srcs = {}
    for r in rows[:5000]:
        srcs[r["src"]] = srcs.get(r["src"], 0) + 1
    ok &= check(len(rows) > 60_000, f"all_train rows loaded: {len(rows)} "
                f"(+head mix {srcs})")
    groups = D.build_groups(rows)
    idx = D.mix_index(rows, groups, 32, {"mv": 0.5, "gsl": 0.0},
                      random.Random(1))
    ok &= check(len(idx) == 32 and all(0 <= i < len(rows) for i in idx),
                f"mix_index: {D.index_stats(rows, idx)}")
    seen = [rows[i]["src"] for i in idx]
    ok &= check("gh" in seen and "mv" in seen, f"mixture has both sources "
                f"({{gh: {seen.count('gh')}, mv: {seen.count('mv')}}})")

    # 2. pair dataset + collate + forward + losses -------------------------
    t0 = time.time()
    feats = [D.feat_any(rows[i]) for i in idx[:4]]
    dims = {(f.shape) for f in feats}
    ok &= check(all(d[1] == 633 and 0 < d[0] <= 256 for d in dims),
                f"feat_any shapes {list(dims)} in {time.time()-t0:.2f}s")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("facebook/bart-base")

    model = VideoText(dict(in_dim=633, d_model=256, enc_layers=4,
                           frontend_layers=2, proj_dim=64), proj_dim=64)
    sys.path.insert(0, "/Users/mazen/dev/projects/medsign")
    from pipeline.contrastive import make_collate
    mini_rows = [rows[i] for i in idx[:4]]
    import pipeline.data as DD
    pair = DD.PairDataset(mini_rows)
    batch = [pair[i] for i in range(4)]
    b = make_collate(tok)(batch)
    out = model(b["src"], b["src_mask"], b["src2"], b["src2_mask"],
                b["input_ids"], b["attention_mask"])
    ok &= check(out["v"].shape == (4, 64) and out["t"].shape == (4, 64),
                f"forward: v {tuple(out['v'].shape)}, t {tuple(out['t'].shape)}")
    ok &= check(abs(out["v"].norm(dim=-1) - 1).max() < 1e-4,
                "embeddings are L2-normalized")
    losses = cl_losses(out, 0.1)
    ok &= check(torch.isfinite(losses["loss"]).item(),
                f"losses finite: loss={losses['loss']:.3f} "
                f"vt={losses['l_vt']:.3f} tv={losses['l_tv']:.3f} "
                f"vv={losses['l_vv']:.3f}")
    t0 = time.time()
    losses["loss"].backward()
    ok &= check(all(p.grad is not None and p.grad.abs().sum() >= 0
                    for p in model.parameters() if p.requires_grad),
                f"backward ok ({time.time()-t0:.2f}s, cpu)")

    # 3. stage-2 key mapping ----------------------------------------------
    from bpe_tokenizer import BPETokenizer
    from model import SignTransformerV2
    from pipeline.stage2 import load_pose_into
    tokb = BPETokenizer.load("bpe_all_8k.json.wrap")
    s2 = SignTransformerV2(len(tokb), in_dim=633, enc_layers=4, dec_layers=4)
    video_state = {"pose." + k: v for k, v in model.pose.state_dict().items()}
    rep = load_pose_into(s2, {"state": video_state, "ep": 0}, torch.device("cpu"))
    copied = rep["tensors"]
    ok &= check(copied >= 40, f"pose tower mapped into stage2: "
                f"{copied} tensors (proj head correctly dropped="
                f"{len(rep['skipped'])})")
    enc_w_identical = torch.equal(
        s2.encoder.layers[0].self_attn.in_proj_weight,
        model.pose.encoder.layers[0].self_attn.in_proj_weight)
    ok &= check(enc_w_identical, "encoder weights byte-identical after map")

    print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
