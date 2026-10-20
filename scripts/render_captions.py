#!/usr/bin/env python3
"""Render captioned skeleton previews from JW keypoint npz files.

Purpose: visually verify that keypoint windows align with their transcript
units (verses_json / sections_json). Each output frame shows the animated
skeleton plus a caption band with the label/ref and the cleaned English text
of whatever timed unit covers that absolute video moment.

Timing uses the ABSOLUTE video clock: kept frame t maps through frame_idx
(intro cuts, dropped Art ranges) when present; older npz without frame_idx
uses t/fps directly. Interior gaps (dropped ranges) are flagged in the
caption corner so the jump is visible instead of silent.

Usage: .venv312/bin/python scripts/render_captions.py <npz...> [--out-dir ...]
"""
import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.data import _clean_text, _npz_units  # noqa: E402
from render_skeleton import draw_skeleton  # noqa: E402

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

_FONT = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
_FONT_SMALL = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)


def caption_text(fps, d):
    """[(start_s, dur_s, label, text)] covering the whole timeline."""
    out = []
    data = None
    for field in ("verses_json", "sections_json"):
        if field in d:
            raw = d[field]
            raw = raw.decode() if hasattr(raw, "decode") else str(raw)
            try:
                data = __import__("json").loads(raw)
            except Exception:
                data = None
            if data:
                break
    if not data:
        return out
    for u in data:
        if not isinstance(u, dict):
            continue
        st, du = float(u.get("start_s", -1)), float(u.get("dur_s", 0))
        txt = _clean_text(u.get("text_en", ""))
        label = str(u.get("label") or u.get("id") or "")
        if st < 0 or du <= 0 or not txt:
            continue
        out.append((st, du, label, txt))
    out.sort(key=lambda x: x[0])
    return out


def pick_unit(units, ts):
    best = None
    for st, du, label, txt in units:
        if st <= ts < st + du:
            if best is None or len(txt) > len(best[3]):
                best = (st, du, label, txt)
    return best


def wrap(draw, text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def caption(img, unit, ts, gap, fps=30.0):
    """Bottom band: label line + up to 3 wrapped text lines."""
    h, w = img.shape[:2]
    band_h = 96
    band = np.zeros((band_h, w, 3), dtype=np.uint8)
    rgb = Image.fromarray(cv2.cvtColor(band, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(rgb)
    max_w = w - 24
    if unit:
        st, du, label, txt = unit
        head = f"{label}  [{ts:.1f}s of {st:.1f}+{du:.1f}s]"
        n = 0
        y = 4
        for ln in wrap(dr, head, _FONT_SMALL, max_w)[:1]:
            dr.text((12, y), ln, font=_FONT_SMALL, fill=(80, 220, 255))
            y += 17
        for ln in wrap(dr, txt, _FONT, max_w)[:3]:
            dr.text((12, y), ln, font=_FONT, fill=(255, 255, 255))
            y += 19
    else:
        dr.text((12, 40), "(no transcript unit here)", font=_FONT,
                fill=(160, 160, 160))
    if gap:
        dr.text((w - 280, 4), f"gap: {gap/fps:.1f}s dropped",
                font=_FONT_SMALL, fill=(255, 120, 80))
    img[-band_h:] = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    return img


def render(p, out_dir, W=640, H=360, max_frames=None):
    d = dict(np.load(p, allow_pickle=True))
    fps = float(d["fps"]) or 30.0
    T = int(d["n_frames"])
    if max_frames:
        T = min(T, max_frames)
    units = caption_text(fps, d)
    fi = d["frame_idx"] if "frame_idx" in d else None
    off = int(d["start_frame"]) if ("frame_idx" not in d and
                                    "start_frame" in d) else 0
    stem = os.path.splitext(os.path.basename(p))[0]
    out = os.path.join(out_dir, f"captioned_{stem}.mp4")
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not vw.isOpened():
        print(f"  SKIP {p}: no VideoWriter")
        return
    prev_fi = None
    mid = None
    for t in range(T):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        draw_skeleton(img, d, t, label=True)
        if fi is not None:
            abs_f = int(fi[t])
            gap = (abs_f - prev_fi - 1) if (prev_fi is not None and
                                            abs_f - prev_fi > 1) else 0
            prev_fi = abs_f
            ts = abs_f / fps
        else:
            gap, ts = 0, (t + off) / fps
        img = caption(img, pick_unit(units, ts), ts, gap, fps)
        if mid is None and t >= T // 2:
            mid = img.copy()
        vw.write(img)
    vw.release()
    if mid is not None:
        cv2.imwrite(os.path.join(out_dir, f"captioned_{stem}.png"), mid)
    print(f"wrote {out} ({T} frames @ {fps:.0f}fps, {len(units)} units)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="keypoint npz files")
    ap.add_argument("--out-dir",
                    default=os.path.join(ROOT, "previews", "captions"))
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap rendered frames per video (0 = full)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for p in a.inputs:
        render(p, a.out_dir, max_frames=a.max_frames or None)


if __name__ == "__main__":
    main()
