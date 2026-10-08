"""Translate one self-recorded video with a trained checkpoint.

Identical pipeline to training (same extractor settings, same normalize,
same features) so a live clip is treated exactly like a dataset clip:

  .venv312/bin/python translate_video.py me_trying_kin.mp4
  .venv312/bin/python translate_video.py clip.mp4 --ckpt best_all.pt --max-width 1280

Writes data/live/<name>.npz (keypoints, reusable) and prints the translation plus
detection diagnostics (low pose/hand presence = framing problem, not model).
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from extract_keypoints import make_holistic, process_video
from bpe_tokenizer import BPETokenizer
from model import SignTransformerV2
from preprocess import clip_features_var
from tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent
LIVE = ROOT / "data" / "live"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--ckpt", default="best_all.pt")
    ap.add_argument("--max-width", type=int, default=1280,
                    help="downscale wide videos for speed (landmarks are normalized: harmless)")
    args = ap.parse_args()

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["cfg"]
    assert cfg.get("variable"), f"{args.ckpt} is a fixed-64 ckpt; need a variable one"
    TokCls = BPETokenizer if cfg.get("tok_type") == "bpe" else Tokenizer
    tok = TokCls.load(cfg.get("tokenizer", "tokenizer.json"))
    base_kind = {"face256": "face", "pos256": "pos"}[cfg["kind"]]

    holistic = make_holistic()
    print("extracting keypoints...", flush=True)
    d = process_video(args.video, holistic, args.max_width)
    n = int(d["n_frames"])
    print(f"frames: {n} ({float(d['fps']):.0f} fps), "
          f"pose {d['pose_present'].mean():.0%} | "
          f"handL {d['handL_present'].mean():.0%} | "
          f"handR {d['handR_present'].mean():.0%} | "
          f"face {d['face_present'].mean():.0%}")

    LIVE.mkdir(exist_ok=True)
    stem = Path(args.video).stem
    np.savez_compressed(LIVE / f"{stem}.npz", **d)

    feats, info = clip_features_var(stem, kind=base_kind, kp_dir=LIVE)
    print(f"features: {feats.shape} (src frames {info['n_src']})")
    if n > cfg.get("max_frames", 256):
        print(f"note: {n} frames > {cfg['max_frames']} cap, downsampled (sign slower / trim pauses)")

    model = SignTransformerV2(cfg["vocab"], in_dim=cfg["in_dim"],
                              enc_layers=cfg.get("enc", 4),
                              dec_layers=cfg.get("dec", 4),
                              frontend_layers=cfg.get("frontend_layers", 2)).to(device)
    model.load_state_dict(ckpt["state"], strict=False)  # old ckpts lack ctc_head
    src = torch.from_numpy(feats).unsqueeze(0).to(device)
    mask = torch.zeros(1, len(feats), dtype=torch.bool, device=device)
    ids = model.greedy(src, src_mask=mask, max_len=60)[0].tolist()
    print(f"TRANSLATION: {tok.decode(ids)}")


if __name__ == "__main__":
    main()
