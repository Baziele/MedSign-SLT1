#!/usr/bin/env python3
"""One-off: export guaranteed-correct landmark index maps.

Pulls face-contour index sets from the installed MediaPipe package itself
(mediapipe 0.10.21) and the curated subsets from render_skeleton.py, so no
hand-typed landmark index can be wrong. Output: keypoints/landmark_map.json
Run:  .venv312/bin/python gen_landmark_map.py
"""

import json
import os

import render_skeleton as R
from mediapipe.python.solutions import face_mesh_connections as FMC

POSE_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

HAND_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

NOSE_RIDGE_PAIRS = [
    (168, 6), (6, 197), (197, 195), (195, 5), (5, 4),
    (4, 1), (1, 19), (19, 94), (94, 2),
]
NOSE_FLOOR_PAIRS = [(64, 98), (98, 97), (97, 2), (2, 326)]


def vset(pairs):
    return sorted({i for p in pairs for i in p})


lips_all = vset(FMC.FACEMESH_LIPS)
lips_inner = sorted(R._LIP_INNER)
lips_outer = sorted(set(lips_all) - set(lips_inner))

landmark_map = {
    "meta": {
        "extractor": "MediaPipe Holistic (mp.solutions.holistic), mediapipe==0.10.21",
        "settings": {
            "static_image_mode": False,
            "model_complexity": 1,
            "refine_face_landmarks": True,
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5,
        },
        "note": "Face contour subsets mirror render_skeleton.py (visualization == validated model-input candidate).",
    },
    "pose_33": {
        "names": POSE_NAMES,
        "key_indices": {
            "nose": 0,
            "left_shoulder": 11, "right_shoulder": 12,
            "left_elbow": 13, "right_elbow": 14,
            "left_wrist": 15, "right_wrist": 16,
            "left_hip": 23, "right_hip": 24,
        },
        "head_indices": list(range(11)),
        "body_indices": list(range(11, 33)),
        "head_note": "0-10 are crude nose/eyes/ears/mouth stubs duplicating the "
                     "face mesh. Overlay visualizations drop them (segment 9-10 "
                     "fakes a lip line; the nose-ear fan fakes a brow V); the "
                     "data keeps them.",
        "links": [list(p) for p in R.POSE_LINKS],
        "body_links": [list(p) for p in R.POSE_BODY_LINKS],
    },
    "hand_21": {
        "names": HAND_NAMES,
        "wrist": 0,
        "fingertip_indices": [4, 8, 12, 16, 20],
        "links": [list(p) for p in R.HAND_LINKS],
    },
    "face_478": {
        "count": 478,
        "layout": "indices 0-467 = base mesh, 468-477 = iris (10 pts, excluded from contours)",
        "contours": {
            "lips_outer": lips_outer,
            "lips_inner": lips_inner,
            "eye_left": vset(FMC.FACEMESH_LEFT_EYE),
            "eye_right": vset(FMC.FACEMESH_RIGHT_EYE),
            "brow_left": vset(FMC.FACEMESH_LEFT_EYEBROW),
            "brow_right": vset(FMC.FACEMESH_RIGHT_EYEBROW),
            "nose_ridge": vset(NOSE_RIDGE_PAIRS),
            "nose_floor": vset(NOSE_FLOOR_PAIRS),
            "jawline": sorted(R._FACE_JAW),
            "iris": list(range(468, 478)),
        },
        "visualization_subset": R.FACE_VERTS,
        "visualization_links": [list(p) for p in sorted(R.FACE_LINKS)],
    },
    "coordinate_conventions": {
        "x": "normalized [0,1], 0 = left edge of frame",
        "y": "normalized [0,1], 0 = TOP edge of frame (image coords, y grows downward)",
        "z": "MediaPipe relative depth, same scale as x; smaller = closer to camera. Ordinal, not metric.",
        "pose_world_xyz": "meters, origin at hips center. Metric, unlike pose_xyz.",
        "missing": "NaN in xyz arrays; see *_present boolean masks. Visibility in pose_vis.",
    },
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keypoints",
                   "landmark_map.json")
with open(out, "w") as f:
    json.dump(landmark_map, f, indent=1)
print(f"wrote {out} "
      f"({len(landmark_map['face_478']['visualization_subset'])} face subset verts)")
