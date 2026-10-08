"""Normalize + resample SignTalk-GH keypoints.

Pipeline per clip (raw .npz -> (T, D) feature matrix):
  1. Interpolate pose frames where pose_present==False (raw image-coord space).
  2. Per-frame: center = midpoint of shoulders (pose 11, 12); subtract from
     x,y of ALL streams (pose/hands/face share the image-coord space, so one
     transform keeps them aligned). Scale = shoulder width; divide x,y,z.
  3. Per-clip single rotation: median shoulder-line angle over the clip,
     rotate x,y about the (now centered) origin. No per-frame leveling.
  4. Missing hands -> zeros (hand out of frame is signal, not noise).
  5. Linear resample every stream to T=64 frames.

Default features: pose 33 + handL 21 + handR 21 = 75 joints x 3 = 225 dims.
Face loads too (for later lip-subset work) but is excluded from features:
478 dense points would dominate any distance metric.
"""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
KP_DIR = ROOT / "data" / "multiview" / "keypoints"
SPLITS_DIR = ROOT / "data" / "splits"

T_FRAMES = 64
T_MAX = 256  # variable-length cap: clips shorter than this keep native frames;
# only longer clips are linearly downsampled to T_MAX (no more squashing to 64).
SHOULDER_L, SHOULDER_R = 11, 12
EPS = 1e-6


def load_raw(stem, kp_dir=KP_DIR):
    """Load one clip's raw arrays. Returns dict of xyz (N, J, 3) + present (N,)."""
    d = np.load(kp_dir / f"{stem}.npz")
    return {
        "pose": d["pose_xyz"].astype(np.float64),
        "pose_present": d["pose_present"],
        "handL": d["handL_xyz"].astype(np.float64),
        "handL_present": d["handL_present"],
        "handR": d["handR_xyz"].astype(np.float64),
        "handR_present": d["handR_present"],
        "face": d["face_xyz"].astype(np.float64),
        "face_present": d["face_present"],
    }


def _interp_missing(x, present):
    """Linear-interpolate rows where present==False (per joint/coordinate)."""
    x = x.copy()
    n = len(x)
    idx = np.arange(n)
    good = np.where(present)[0]
    if len(good) == 0:
        return x  # caller guards; verified never happens for pose
    for j in range(x.shape[1]):
        for c in range(x.shape[2]):
            x[:, j, c] = np.interp(idx, good, x[good, j, c])
    return x


def normalize(pose, handL, handR, face, pose_present):
    """Center/scale per frame + one median rotation per clip. Returns same shapes."""
    pose = _interp_missing(pose, pose_present)
    n = len(pose)

    L = pose[:, SHOULDER_L, :2]
    R = pose[:, SHOULDER_R, :2]
    center = (L + R) / 2.0
    width = np.linalg.norm(R - L, axis=1)
    width = np.where(width < EPS, 1.0, width)

    def tfm(x):
        x = x.copy()
        x[..., 0] -= center[:, None, 0]
        x[..., 1] -= center[:, None, 1]
        x /= width[:, None, None]
        return x

    pose, handL, handR, face = tfm(pose), tfm(handL), tfm(handR), tfm(face)

    # single per-clip rotation from median shoulder angle (xy plane).
    # The shoulder line is an UNDIRECTED axis: fold the angle into
    # [-pi/2, pi/2) so we only correct camera tilt, never flip head-down
    # (subjects face the camera, so raw dx<0 puts the angle near +/-pi).
    dx = R[:, 0] - L[:, 0]
    dy = R[:, 1] - L[:, 1]
    ang = float(np.median(np.arctan2(dy, dx)))
    ang = (ang + np.pi / 2) % np.pi - np.pi / 2
    ca, sa = np.cos(-ang), np.sin(-ang)
    rot = np.array([[ca, -sa], [sa, ca]])

    def rot_xy(x):
        x = x.copy()
        x[..., :2] = x[..., :2] @ rot.T
        return x

    return rot_xy(pose), rot_xy(handL), rot_xy(handR), rot_xy(face), ang


def resample(x, t=T_FRAMES):
    """Linear resample (N, J, C) -> (t, J, C). Bool presence via threshold."""
    n = len(x)
    if n == t:
        return x
    if n == 1:
        return np.repeat(x, t, axis=0)
    old = np.linspace(0, 1, n)
    new = np.linspace(0, 1, t)
    flat = x.reshape(n, -1).astype(np.float64)
    out = np.stack([np.interp(new, old, flat[:, k]) for k in range(flat.shape[1])], axis=1)
    out = out.reshape(t, *x.shape[1:])
    if x.dtype == bool:
        out = out > 0.5
    return out.astype(x.dtype if x.dtype != np.float64 else np.float32)


def clip_features(stem, t=T_FRAMES, kp_dir=KP_DIR):
    """Full pipeline for one clip -> (t, 225) float32 + debug info."""
    r = load_raw(stem, kp_dir)
    pose, hL, hR, face, ang = normalize(
        r["pose"], r["handL"], r["handR"], r["face"], r["pose_present"]
    )
    # missing hands -> zeros (after transform; keeps scale consistent)
    hL = np.where(r["handL_present"][:, None, None], hL, 0.0)
    hR = np.where(r["handR_present"][:, None, None], hR, 0.0)
    feats = np.concatenate(
        [resample(pose, t), resample(hL, t), resample(hR, t)], axis=1
    )  # (t, 75, 3)
    feats = np.where(np.isfinite(feats), feats, 0.0).astype(np.float32)
    return feats.reshape(t, -1), {"median_angle": ang, "n_src": len(r["pose"])}


# Training joints = exactly what the overlay videos render, plus the pose
# head back: full pose 33 + full hands 21+21 + face contour subset
# (FACE_VERTS: lips/eyes/brows/nose/jawline, 136 pts -- not the full 478).
def clip_features_face(stem, t=T_FRAMES, kp_dir=KP_DIR):
    """Full pipeline for one clip -> (t, 633) float32 + debug info."""
    from render_skeleton import FACE_VERTS

    r = load_raw(stem, kp_dir)
    face = _interp_missing(r["face"], r["face_present"])
    pose, hL, hR, face, ang = normalize(
        r["pose"], r["handL"], r["handR"], face, r["pose_present"]
    )
    hL = np.where(r["handL_present"][:, None, None], hL, 0.0)
    hR = np.where(r["handR_present"][:, None, None], hR, 0.0)
    feats = np.concatenate(
        [resample(pose, t), resample(hL, t), resample(hR, t),
         resample(face[:, list(FACE_VERTS)], t)], axis=1
    )  # (t, 211, 3)
    feats = np.where(np.isfinite(feats), feats, 0.0).astype(np.float32)
    return feats.reshape(t, -1), {"median_angle": ang, "n_src": len(r["pose"])}


def _assemble(pose, hL, hR, face_part):
    """Concatenate joint streams -> (N, D) float32. face_part=None for pos-only."""
    parts = [pose, hL, hR] + ([face_part] if face_part is not None else [])
    feats = np.concatenate(parts, axis=1)  # (N, J, 3)
    feats = np.where(np.isfinite(feats), feats, 0.0).astype(np.float32)
    return feats.reshape(len(feats), -1)


def clip_features_var(stem, t_max=T_MAX, kind="face", kp_dir=KP_DIR):
    """Variable-length pipeline for one clip -> (Tv, D) float32 + debug info.

    Same normalize as fixed path, but NO squash to 64: clips with N <= t_max
    keep native frame count (temporal resolution preserved); only N > t_max
    clips are linearly downsampled to t_max. Caller pads + masks per batch.
    kind: "pos" -> (Tv, 225) | "face" -> (Tv, 633).
    """
    from render_skeleton import FACE_VERTS

    r = load_raw(stem, kp_dir)
    n_src = len(r["pose"])
    if kind == "face":
        face = _interp_missing(r["face"], r["face_present"])
        pose, hL, hR, face, ang = normalize(
            r["pose"], r["handL"], r["handR"], face, r["pose_present"]
        )
        face_sub = face[:, list(FACE_VERTS)]
    elif kind == "pos":
        pose, hL, hR, _, ang = normalize(
            r["pose"], r["handL"], r["handR"], r["face"], r["pose_present"]
        )
        face_sub = None
    else:
        raise ValueError(f"unknown kind {kind!r} (want 'pos' or 'face')")
    hL = np.where(r["handL_present"][:, None, None], hL, 0.0)
    hR = np.where(r["handR_present"][:, None, None], hR, 0.0)
    if n_src > t_max:
        pose = resample(pose, t_max)
        hL = resample(hL, t_max)
        hR = resample(hR, t_max)
        if face_sub is not None:
            face_sub = resample(face_sub, t_max)
    feats = _assemble(pose, hL, hR, face_sub)
    return feats, {"median_angle": ang, "n_src": n_src, "t_var": len(feats)}


def iter_split(name, splits_dir=SPLITS_DIR):
    with open(splits_dir / f"{name}.csv") as f:
        yield from csv.DictReader(f)


if __name__ == "__main__":
    import time

    for stem in ["1A", "3006C", "2945B"]:
        f, info = clip_features(stem)
        # post-norm check on resampled pose part
        p = f[:, : 33 * 3].reshape(T_FRAMES, 33, 3)
        mid = (p[:, SHOULDER_L, :2] + p[:, SHOULDER_R, :2]) / 2
        w = np.linalg.norm(p[:, SHOULDER_R, :2] - p[:, SHOULDER_L, :2], axis=1)
        print(
            f"{stem}: src={info['n_src']} out={f.shape} "
            f"|mid|~{np.abs(mid).max():.2e} width~{w.mean():.3f} "
            f"med_ang={info['median_angle']:.3f} nan={np.isnan(f).sum()} "
            f"nose_above_shoulders={bool((p[:, 0, 1] - p[:, SHOULDER_L, 1]).mean() < 0)}"
        )
    t0 = time.time()
    n = 0
    for split in ["train", "val", "test_T", "test_R"]:
        for row in iter_split(split):
            f, _ = clip_features(row["stem"])
            assert f.shape == (T_FRAMES, 225) and np.isfinite(f).all()
            n += 1
    print(f"full pass: {n} clips in {time.time()-t0:.1f}s")
