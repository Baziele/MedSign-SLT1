#!/usr/bin/env python3
"""Render black-background skeleton animations from keypoint .npz files.

One panel per input file, side by side -- ideal for comparing signers on
the same sentence, or spot-checking extraction quality.

Usage:
  .venv312/bin/python render_skeleton.py keypoints/1A.npz keypoints/1C.npz keypoints/1D.npz -o preview_1ACD.mp4
  .venv312/bin/python render_skeleton.py keypoints/1A.npz -o preview_1A.mp4

Overlay mode (skeleton drawn on the real video frames, one output per input):
  .venv312/bin/python render_skeleton.py --overlay keypoints/1A.npz keypoints/1C.npz

Outputs <name>.mp4 plus <name>.png (a mid-video contact sheet, one frame
per panel, for quick inline viewing). Overlay outputs land in --out-dir
(default ./previews) as overlay_<stem>.mp4/.png.
"""

import argparse
import os

import cv2
import numpy as np

# BlazePose 33-point topology (matches mp.solutions.pose.POSE_CONNECTIONS).
POSE_LINKS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

# Body-pose head (0-10: nose/eyes/ears/mouth stubs) duplicates the detailed
# face mesh, crudely: segment 9-10 draws a fake lip line, and the nose-ear
# fan draws a V over the brows. Overlay mode uses these headless sets;
# black-background mode keeps the full skeleton.
POSE_HEAD_VERTS = list(range(11))
POSE_BODY_LINKS = [(a, b) for (a, b) in POSE_LINKS if a >= 11 and b >= 11]
POSE_BODY_VERTS = list(range(11, 33))

# 21-point hand topology (matches mp.solutions.hands.HAND_CONNECTIONS).
HAND_LINKS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

# --- Reference-style face: sparse contours, not the full mesh ---
# Link sets come from MediaPipe's own contour definitions (guaranteed to
# match the landmark numbering): dense double-ring lips, eye outlines,
# eyebrows, nose ridge+base, and the jawline arc of the face oval
# (temple 234 -> chin 152 -> temple 323), irises excluded.
from mediapipe.python.solutions import face_mesh_connections as _FMC

_FACE_JAW = frozenset({
    58, 93, 132, 136, 148, 149, 150, 152, 172, 176, 234,
    288, 323, 361, 365, 377, 378, 379, 397, 400,
})
# Face = FULL contours (lips incl. inner ring, nose incl. walls, eyes,
# brows, jawline). NOTE: an earlier revision excluded the inner-lip ring
# and nose walls, mistaking them for overlay artifacts -- wrong call. The
# artifacts were the body-pose model's crude head stubs (mouth segment 9-10
# reads as a fake lip line; the nose-ear fan reads as a V over the brows).
# So facial detail is fully restored; the pose head is dropped in overlay
# mode instead (see POSE_BODY_LINKS below). The _LIP_INNER / _NOSE_KEEP sets
# remain defined: landmark_map.json documents them as contour subsets.
_LIP_INNER = frozenset({
    13, 14, 78, 80, 81, 82, 87, 88, 95, 178,
    191, 308, 310, 311, 312, 317, 318, 324, 402, 415,
})
def _norm(p):
    return (p[0], p[1]) if p[0] <= p[1] else (p[1], p[0])


_NOSE_KEEP = frozenset(map(_norm, [
    (168, 6), (6, 197), (197, 195), (195, 5), (5, 4), (4, 1),
    (1, 19), (19, 94), (94, 2),                          # bridge ridge
    (64, 98), (98, 97), (97, 2), (2, 326),               # nostril floor
]))
# Full face: every contour set, nothing excluded.
FACE_LINKS = frozenset(
    [_norm(p) for p in _FMC.FACEMESH_LIPS]
    + [_norm(p) for p in _FMC.FACEMESH_LEFT_EYE]
    + [_norm(p) for p in _FMC.FACEMESH_RIGHT_EYE]
    + [_norm(p) for p in _FMC.FACEMESH_LEFT_EYEBROW]
    + [_norm(p) for p in _FMC.FACEMESH_RIGHT_EYEBROW]
    + [_norm(p) for p in _FMC.FACEMESH_NOSE]
    + [p for p in _FMC.FACEMESH_FACE_OVAL if p[0] in _FACE_JAW and p[1] in _FACE_JAW]
)
FACE_VERTS = sorted({i for p in FACE_LINKS for i in p})  # ~120 pts, not 478

# Two-tone style (links dim, joints bright) so small joints stay visible.
# NOTE: cv2 is BGR, so "cyan" is (170,170,0), not (0,170,170).
C_FACE_LINK, C_FACE_JOINT = (170, 170, 0), (230, 255, 255)
C_POSE_LINK, C_POSE_JOINT = (170, 170, 0), (255, 255, 255)
C_HAND_L_LINK, C_HAND_L_JOINT = (0, 160, 0), (170, 255, 170)
C_HAND_R_LINK, C_HAND_R_JOINT = (160, 0, 160), (255, 170, 255)


def valid(pt):
    return not (np.isnan(pt[0]) or np.isnan(pt[1]))


def draw_links(img, pts, links, color, thick):
    h, w = img.shape[:2]
    for a, b in links:
        if a < len(pts) and b < len(pts) and valid(pts[a]) and valid(pts[b]):
            pa = (int(pts[a][0] * w), int(pts[a][1] * h))
            pb = (int(pts[b][0] * w), int(pts[b][1] * h))
            cv2.line(img, pa, pb, color, thick, cv2.LINE_AA)


def draw_dots(img, pts, color, radius, idx=None):
    h, w = img.shape[:2]
    seq = pts if idx is None else (pts[i] for i in idx)
    for pt in seq:
        if valid(pt):
            cv2.circle(img, (int(pt[0] * w), int(pt[1] * h)), radius, color, -1,
                       cv2.LINE_AA)


def draw_skeleton(img, d, t, label=True, draw_face=True, pose_head=True):
    """Draw frame t of one npz dict onto img (black canvas or video frame)."""
    H, W = img.shape[:2]
    s = max(1, H // 360)  # scale line thickness with resolution
    T = int(d["n_frames"])
    tt = min(t, T - 1)  # freeze last frame once a shorter input ends
    ended = t >= T

    if draw_face:
        draw_links(img, d["face_xyz"][tt], FACE_LINKS, C_FACE_LINK, s)
        draw_dots(img, d["face_xyz"][tt], C_FACE_JOINT, s, idx=FACE_VERTS)
    if pose_head:
        draw_links(img, d["pose_xyz"][tt], POSE_LINKS, C_POSE_LINK, s)
        draw_dots(img, d["pose_xyz"][tt], C_POSE_JOINT, s)
    else:
        draw_links(img, d["pose_xyz"][tt], POSE_BODY_LINKS, C_POSE_LINK, s)
        draw_dots(img, d["pose_xyz"][tt], C_POSE_JOINT, s, idx=POSE_BODY_VERTS)
    draw_links(img, d["handL_xyz"][tt], HAND_LINKS, C_HAND_L_LINK, s)
    draw_dots(img, d["handL_xyz"][tt], C_HAND_L_JOINT, s)
    draw_links(img, d["handR_xyz"][tt], HAND_LINKS, C_HAND_R_LINK, s)
    draw_dots(img, d["handR_xyz"][tt], C_HAND_R_JOINT, s)

    if label:
        signer = str(d.get("signer", d.get("source", "")))
        text = f"{str(d['video_file'])}  signer={signer}  f{tt}/{T}"
        if ended:
            text += "  END"
        cv2.putText(img, text, (10, int(24 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * s, (255, 255, 255), s, cv2.LINE_AA)
    if ended:
        img[:] = (img * 0.45).astype(np.uint8)
    return img


def draw_frame(d, t, W, H, draw_face=True, pose_head=True):
    """Render frame t of one npz dict onto a WxH black canvas."""
    return draw_skeleton(np.zeros((H, W, 3), dtype=np.uint8), d, t,
                         draw_face=draw_face, pose_head=pose_head)


def render_overlay(paths, videos_dir, out_dir, draw_face=True):
    """One overlay_<stem>.mp4 (+ mid-frame .png) per input npz.

    Overlays always drop the pose head (pose_head=False): the body's crude
    nose/eyes/ears/mouth stubs would scribble over the detailed face mesh.
    """
    os.makedirs(out_dir, exist_ok=True)
    for p in paths:
        d = dict(np.load(p, allow_pickle=True))
        vpath = os.path.join(videos_dir, str(d["video_file"]))
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"  SKIP {p}: cannot open {vpath}")
            continue
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps <= 0:
            fps = round(float(d["fps"])) or 30
        stem = os.path.splitext(os.path.basename(p))[0]
        out = os.path.join(out_dir, f"overlay_{stem}.mp4")
        vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        if not vw.isOpened():
            print(f"  SKIP {p}: could not open VideoWriter")
            cap.release()
            continue
        T = int(d["n_frames"])
        t, mid_png = 0, None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            draw_skeleton(frame, d, t, draw_face=draw_face, pose_head=False)
            if t == T // 2:
                mid_png = frame.copy()
            vw.write(frame)
            t += 1
        cap.release()
        vw.release()
        if mid_png is not None:
            cv2.imwrite(os.path.join(out_dir, f"overlay_{stem}.png"), mid_png)
        print(f"wrote {out} ({t} frames)")


def render_train_view(paths, out_dir, mode):
    """Render exactly the joints a training run sees (black background).

    mode=new: face contours (FACE_VERTS, 136) + FULL pose incl. head + hands.
    mode=old: previous runs -- full pose + hands, no face.
    """
    os.makedirs(out_dir, exist_ok=True)
    face = mode == "new"
    for p in paths:
        d = dict(np.load(p, allow_pickle=True))
        T = int(d["n_frames"])
        fps = round(float(d["fps"])) or 30
        W, H = 640, 360
        stem = os.path.splitext(os.path.basename(p))[0]
        out = os.path.join(out_dir, f"trainview_{stem}_{mode}.mp4")
        vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        for t in range(T):
            vw.write(draw_frame(d, t, W, H, draw_face=face, pose_head=True))
        vw.release()
        png = os.path.join(out_dir, f"trainview_{stem}_{mode}.png")
        cv2.imwrite(png, draw_frame(d, T // 2, W, H,
                                    draw_face=face, pose_head=True))
        print(f"wrote {out} and {png}")


def main():
    ap = argparse.ArgumentParser(description="Render skeleton preview video.")
    ap.add_argument("inputs", nargs="+", help=".npz files, one panel each")
    ap.add_argument("-o", "--output", default=None,
                    help="output .mp4 path (side-by-side mode)")
    ap.add_argument("--overlay", action="store_true",
                    help="draw the skeleton over the original video frames; "
                         "writes overlay_<stem>.mp4 per input into --out-dir")
    ap.add_argument("--videos-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data", "multiview", "Medsign", "archive", "SignTalk-GH", "Videos"))
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "previews"),
        help="output directory for --overlay mode")
    ap.add_argument("--no-face", action="store_true",
                    help="omit the face/head skeleton (data untouched; viewing only)")
    ap.add_argument("--train-view", choices=["new", "old"], default=None,
                    help="render exactly the training joints: new (face contours "
                         "+ full pose + hands) or old (pose + hands, no face)")
    ap.add_argument("--panel-w", type=int, default=640)
    ap.add_argument("--panel-h", type=int, default=360)
    args = ap.parse_args()

    if args.train_view:
        render_train_view(args.inputs, args.out_dir, args.train_view)
        return

    if args.overlay:
        render_overlay(args.inputs, args.videos_dir, args.out_dir,
                       draw_face=not args.no_face)
        return

    if not args.output:
        ap.error("-o/--output is required in side-by-side mode (omit it with --overlay)")
    datas = [dict(np.load(p, allow_pickle=True)) for p in args.inputs]
    T = max(int(d["n_frames"]) for d in datas)
    fps = round(float(datas[0]["fps"])) or 30

    W, H = args.panel_w, args.panel_h
    out = args.output
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W * len(datas), H))
    if not vw.isOpened():
        raise RuntimeError("could not open VideoWriter (missing mp4 codec?)")
    for t in range(T):
        vw.write(np.hstack([draw_frame(d, t, W, H, draw_face=not args.no_face)
                            for d in datas]))
    vw.release()

    # Contact sheet: middle frame of each panel, for quick inline viewing.
    mid = T // 2
    sheet = np.hstack([draw_frame(d, mid, W, H, draw_face=not args.no_face)
                       for d in datas])
    png = os.path.splitext(out)[0] + ".png"
    cv2.imwrite(png, sheet)
    print(f"wrote {out} ({T} frames @ {fps}fps) and {png}")


if __name__ == "__main__":
    main()
