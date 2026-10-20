"""Shared data plumbing for the phase-1 pipeline (contrastive + stage2).

Row dicts carry a `src` tag so every sampler/stat can split by source:
  gh  = original SignTalk hospital sentences (all_train.csv, non-MV stems)
  mv  = Medsign multiview takes (all_train.csv, train_train_/test_test_ stems)
  gsl = GHS dictionary entries (data/gsl/gsl-dictionary/dictionary_clean.csv)
  jw* = JW continuous chapters, sliced verse/section level. Timing + English
        text live INSIDE each npz (`verses_json` / `sections_json`: unit rows
        with start_s/dur_s/text_en). Feature cache = chapter normalized once,
        then the [start_f, end_f) window saved per unit.

All sampling is index-list based: datasets hold the full pool, and each epoch
gets a fresh deterministic index list. The exact composition of every epoch is
logged (pipeline.logkit) so runs are reproducible and paper-reportable.
"""
import csv
import hashlib
import json
import re
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from dataset import FEAT_CACHE, VAR_KINDS, feat_var, augment_var, collate_var
from preprocess import SPLITS_DIR, clip_features_var, T_MAX
from preprocess import _interp_missing, normalize as pp_normalize, resample as pp_resample

ROOT = Path(__file__).resolve().parents[1]
GSL_DIR = ROOT / "data" / "gsl" / "gsl-dictionary"


# ---------------------------------------------------------------- row loading

def load_main_rows(split="all_train"):
    """GH + multiview rows from the combined splits (GH stems are everything
    that is not a multiview take). Adds `src` in place."""
    rows = []
    with open(SPLITS_DIR / f"{split}.csv") as f:
        for r in csv.DictReader(f):
            stem = r["stem"]
            r["src"] = ("mv" if stem.startswith(("train_train_", "test_test_"))
                        else "gh")
            rows.append(r)
    return rows


def load_gsl_rows(path=GSL_DIR / "dictionary_clean.csv"):
    """Dictionary entries -> contrastive pairs (text = the word it teaches).
    Variants map to the same video; exact (stem, text) duplicates are dropped
    but synonym rows are kept: more positives per video is free signal.
    Metadata sometimes lists npz that never landed on disk -> skipped."""
    rows, seen, missing = [], set(), 0
    with open(path) as f:
        for r in csv.DictReader(f):
            stem = r["npz"].removesuffix(".npz")
            text = r["word"].strip().lower()
            if (stem, text) in seen:
                continue
            if not (GSL_DIR / "keypoints" / f"{stem}.npz").exists():
                missing += 1
                continue
            seen.add((stem, text))
            rows.append({"stem": stem, "text": text, "src": "gsl"})
    if missing:
        print(f"gsl: skipped {missing} rows whose npz is missing on disk",
              flush=True)
    return rows


def source_of(stem):
    return "mv" if stem.startswith(("train_train_", "test_test_")) else "gh"


_TAG = re.compile(r"<[^>]+>")


_BOOKS = (r"Genesis|Exodus|Leviticus|Numbers|Deuteronomy|Joshua|Judges|Ruth|"
          r"Samuel|Kings|Chronicles|Ezra|Nehemiah|Esther|Job|Psalm\w*|Proverbs|"
          r"Ecclesiastes|Isaiah|Jeremiah|Lamentations|Ezekiel|Daniel|Hosea|"
          r"Joel|Amos|Obadiah|Jonah|Micah|Nahum|Habakkuk|Zephaniah|Haggai|"
          r"Zechariah|Malachi|Matthew|Mark|Luke|John|Acts|Romans|Corinthians|"
          r"Galatians|Ephesians|Philippians|Colossians|Thessalonians|Timothy|"
          r"Titus|Philemon|Hebrews|James|Peter|Jude|Revelation")
# trailing on-screen citation: '\u2019— Genesis 1:1' / '\u2019— Matthew 24:38, 39'
_CITE = re.compile(
    rf"[\u200b\s]*[—]\s*(?:[1-3]\s)?(?:{_BOOKS})\s+\d+"
    r"(?::\d+)?(?:\s*[;,]\s*\d+(?::\d+)?)?\s*\.?$", re.I)
# one segment of a reference-only line: 'Genesis 6:1-22' / 'Jude 6' / '2 Peter 2:5'
_REFSEG = re.compile(
    rf"(?:[1-3]\s)?(?:{_BOOKS})\s+\d+(?::\d+)?"
    r"(?:\s*[-–]\s*\d+(?::\d+)?)*(?:\s*[,;.]\s*\d+(?::\d+)?)*$", re.I)
# production metadata on MWB on-screen cards
_META_PAREN = (
    re.compile(r"\(\s*\d{1,3}\s*min\.\s*\)", re.I),      # (2 min.)
    re.compile(r"\(\s*(?:lmd|th)\s[^)]*\)", re.I),       # ( lmd lesson 9 point 4 )
)


def _ref_only(t):
    """True when the line is nothing but bible references ('Genesis 6:1-22;
    Matthew 24:37-41; Jude 6') — the reference card, not signed prose."""
    return t and all(_REFSEG.match(seg.strip(" \u200b"))
                     for seg in t.split(";"))


def _clean_text(t):
    """JW embedded text carries HTML fragments, entity-encoded on top
    (e.g. &lt;span class=\"verse\"&gt;), sometimes truncated mid-tag by the
    extractor and never closed (e.g. 'earth. <span class=\"verse\"').
    Unescape first, THEN strip tags, then unescape again for double-encoded
    entities; any '<' with no '>' after it is a dangling truncated tag —
    drop it and everything it swallowed. Collapses whitespace.
    Also strips trailing lesson-locate digit lists (>=2 trailing integers:
    'Anger is dangerous 4, 12, 41, 49' -> 'Anger is dangerous') — these are
    page pointers, not signed content.
    Also strips on-screen artifacts that are not signed content: trailing
    reference citations ('\u2019— Genesis 1:1'), pure reference-list lines
    ('Genesis 6:1-22; Matthew 24:37-41'), and MWB production markers
    ('(2 min.)', '( lmd lesson 9 point 4 )'). References inside a real
    sentence ('Read John 11:21-24. Then ask the audience:') are kept."""
    import html
    s = html.unescape(t or "")
    s = _TAG.sub(" ", s)
    s = html.unescape(s)
    s = _TAG.sub(" ", s)
    lt = s.rfind("<")  # dangling truncated tag: no closing '>' after it
    if lt != -1 and ">" not in s[lt:]:
        s = s[:lt]
    for pat in _META_PAREN:
        s = pat.sub(" ", s)
    s = _CITE.sub(" ", s)
    if _ref_only(s):
        s = ""
    # trailing lesson-locate page pointers — the comma carries no trailing
    # space in the pattern so real thousands ('was 40,500.') are never touched
    s = re.sub(r"(?:\s\d{1,3},)+\s(?:and\s+)?\d{1,3}[.,;:]*\s*$", " ", s)
    s = re.sub(r"(?:\s(?:and\s+)?\d{1,3}){2,}[.,;:!]*\s*$", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_MIN_WORDS = 3  # drops 'October 13-19'-style section headers


def _npz_units(npz_path):
    """(fps, n_frames, [unit dicts]) from one JW npz: verses_json preferred,
    sections_json fallback — same timed-unit schema (start_s/dur_s/text_en)."""
    z = np.load(npz_path, allow_pickle=True)
    data = []
    for field in ("verses_json", "sections_json"):  # first that parses wins
        if field in z:
            raw = z[field]
            raw = raw.decode() if hasattr(raw, "decode") else str(raw)
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = []
            if data:
                break
    if not data:
        return float(z["fps"]), int(z["n_frames"]), []
    fps = float(z["fps"])
    n_frames = int(z["n_frames"])
    units = []
    for pos, u in enumerate(data):
        st, du = float(u.get("start_s", -1)), float(u.get("dur_s", 0))
        text = _clean_text(u.get("text_en", ""))
        if st < 0 or du <= 0.2 or not text:
            continue
        if len(text.split()) < _MIN_WORDS:
            continue
        u["pos"] = pos  # raw position = stable cache name, filter-proof
        u["text_en"] = text  # store the cleaned text, not the raw fragment
        units.append(u)
    return fps, n_frames, units


def split_jw(rows, val_frac=0.03, test_frac=0.03):
    """Deterministic chapter-level holdout: order chapters by content hash and
    hand out the first slice as test, next as val, rest train. Chapter-level
    (not verse-level) so no signer/content leakage across splits, and pure
    md5 ordering so every run/process sees identical splits without a state
    file. Adds a `split` column to each row in place."""
    chapters = sorted(set(r["chapter"] for r in rows))
    ranked = sorted(chapters,
                    key=lambda c: hashlib.md5(f"jwsplit|{c}".encode())
                    .hexdigest())
    n_test = round(len(ranked) * test_frac)
    n_val = round(len(ranked) * val_frac)
    test = set(ranked[:n_test])
    val = set(ranked[n_test:n_test + n_val])
    for r in rows:
        r["split"] = ("test" if r["chapter"] in test else
                      "val" if r["chapter"] in val else "train")
    c = {s: sum(1 for r in rows if r["split"] == s) for s in
         ("train", "val", "test")}
    print(f"jw split: {c} (chapter-level, {len(chapters)} chapters)",
          flush=True)
    return rows


def load_jw_rows(sources=("nwt-ghs",)):
    """Slice JW set(s)' chapters into timed-unit training rows (`src`=source
    slug). Chapter normalized once, [start_f, end_f) sliced. Index cache:
    data/splits/jw_{source}_index.csv (ignored, fine to delete)."""
    if isinstance(sources, str):
        sources = [sources]
    out = []
    for source in sources:
        kpath = ROOT / "data" / "jw" / source / "keypoints"
        idx_csv = SPLITS_DIR / f"jw_{source}_index.csv"
        if idx_csv.exists():
            with open(idx_csv) as f:
                rows = list(csv.DictReader(f))
            # self-heal: GPU-cached index may predate a text-cleaner fix —
            # cleaning is idempotent and costs nothing at load time;
            # re-filtering drops rows that became ref-only/marker-only
            # (e.g. '(2 min.) INFORMAL WITNESSING.') under the newer cleaner
            kept = []
            for r in rows:
                c = _clean_text(r["text"])
                if len(c.split()) >= _MIN_WORDS:
                    r["text"] = c
                    kept.append(r)
            rows = kept
        else:
            rows = []
            n_dropped = 0
            for p in sorted(kpath.glob("*.npz")):
                fps, n_frames, units = _npz_units(p)
                if not units:
                    continue
                zt = np.load(p, allow_pickle=True)  # trim metadata (cheap)
                for u in units:
                    start_f = int(round(u["start_s"] * fps))
                    end_f = int(round((u["start_s"] + u["dur_s"]) * fps))
                    if end_f - start_f < 2:
                        continue
                    m = _map_window_z(zt, start_f, end_f)
                    if m is None:  # unit has no usable kept footage
                        n_dropped += 1
                        continue
                    a_lo, a_hi = m
                    rows.append({
                        "stem": f"{source}__{p.stem}__{u['pos']:03d}",
                        "chapter": p.stem, "npz": str(p),
                        "start_f": str(start_f), "end_f": str(end_f),
                        "a_lo": str(a_lo), "a_hi": str(a_hi),
                        "dur_s": f"{u['dur_s']:.2f}", "src": source,
                        "text": u["text_en"],
                    })
            if n_dropped:
                print(f"jw[{source}]: dropped {n_dropped} units with no "
                      f"usable kept footage (trim cut them off)", flush=True)
            SPLITS_DIR.mkdir(exist_ok=True)
            with open(idx_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows
                                   else ["stem"])
                w.writeheader()
                w.writerows(rows)
            print(f"jw[{source}]: indexed {len(rows)} units from "
                  f"{len(set(r['chapter'] for r in rows))} chapters",
                  flush=True)
        rows = split_jw(rows)
        out.extend(rows)
    return out


def _map_window_z(z, start_f, end_f):
    """Map absolute video frames -> array indices for a possibly-trimmed
    chapter npz. Trims happen at EXTRACTION time, so npz arrays start above
    absolute frame 0: mwb stores start_frame/skip_seconds (intro cut), lfb
    stores frame_idx = kept original frame numbers (head cut; interior gaps
    possible if ranges were dropped), nwt-ghs has neither (identity).
    Ignoring this had sliced every mwb/lfb cache ~one cut-length late.
    Used BOTH at index time (drop un-mappable units) and at slice time.
    Returns (a_lo, a_hi) half-open array indices, or None if the unit has
    essentially no kept footage."""
    if "frame_idx" in z:
        fi = np.asarray(z["frame_idx"], dtype=np.int64)
        a_lo = int(np.searchsorted(fi, start_f, side="left"))
        a_hi = int(np.searchsorted(fi, end_f, side="left"))
        requested = end_f - start_f
        if a_hi <= a_lo or a_hi - a_lo < 2:
            return None
        covered = int(fi[a_hi - 1]) - int(fi[a_lo]) + 1
        if covered < 0.5 * requested:  # unit mostly inside a dropped range
            return None
        return a_lo, a_hi
    off = int(z["start_frame"]) if "start_frame" in z else 0
    n = int(z["n_frames"]) if "n_frames" in z else 1 << 30
    a_lo, a_hi = start_f - off, min(end_f - off, n)
    if a_lo < 0 or a_hi - a_lo < 2:
        return None
    return a_lo, a_hi


def _map_window(npz, start_f, end_f):
    return _map_window_z(np.load(npz, allow_pickle=True), start_f, end_f)


def chapter_verse_feats(npz, start_f, end_f, a_lo=None, a_hi=None,
                        kind="face256", t_max=T_MAX):
    """Feature matrix (Tv, 633) for one JW slice. The chapter is normalized
    once, then the frame window is cut — per-verse stats would re-run the
    interp/normalize pipeline per unit for the same video. a_lo/a_hi are the
    ARRAY indices pre-mapped by _map_window at index time; when omitted the
    absolute frames are mapped here instead."""
    from render_skeleton import FACE_VERTS

    z = np.load(npz, allow_pickle=True)
    if a_lo is None or a_hi is None:
        m = _map_window_z(z, start_f, end_f)
        if m is None:
            return None
        a_lo, a_hi = m
    face = _interp_missing(z["face_xyz"].astype(np.float64), z["face_present"])
    pose, hL, hR, face_n, _ = pp_normalize(
        z["pose_xyz"].astype(np.float64),
        z["handL_xyz"].astype(np.float64),
        z["handR_xyz"].astype(np.float64), face, z["pose_present"])
    hL = np.where(z["handL_present"][:, None, None], hL, 0.0)
    hR = np.where(z["handR_present"][:, None, None], hR, 0.0)
    sl = slice(int(a_lo), int(a_hi))
    parts = [pose[sl], hL[sl], hR[sl],
             face_n[sl][:, list(FACE_VERTS)]]
    n = parts[0].shape[0]
    if n > t_max:
        parts = [pp_resample(p, t_max) for p in parts]
    feats = np.concatenate(parts, axis=1)
    feats = np.where(np.isfinite(feats), feats, 0.0).astype(np.float32)
    return feats.reshape(len(feats), -1)


# ------------------------------------------------------------- feature access

def kp_dir_for(row):
    """None -> preprocess default (data/multiview/keypoints, GH+MV); GSL npz
    live in their own dir so they need an explicit one."""
    return (GSL_DIR / "keypoints") if row["src"] == "gsl" else None


def feat_any(row, kind="face256", t_max=256):
    """dataset.feat_var with per-source kp dirs. Cache layout is shared with
    dataset.py so stage-2 training never recomputes what stage-1 already saw.
    JW verse rows need their cache prebuilt (build_jw_cache.py): inline
    computation would reload + renormalize a whole chapter per verse."""
    cache = FEAT_CACHE[kind]
    p = cache / f"{row['stem']}.npy"
    if p.exists():
        return np.load(p)
    if "start_f" in row:
        raise RuntimeError(
            f"missing JW cache {p.name}; run scripts/data_tools/build_jw_cache.py first")
    kp = kp_dir_for(row)
    if kp is None:
        return feat_var(row["stem"], kind)
    feats, _ = clip_features_var(row["stem"], t_max=t_max,
                                 kind=VAR_KINDS[kind], kp_dir=kp)
    cache.mkdir(parents=True, exist_ok=True)
    np.save(p, np.ascontiguousarray(feats, dtype=np.float32))
    return feats


def len_of(row, kind="face256"):
    """Cached clip length (npy header read) or computes it once."""
    p = FEAT_CACHE[kind] / f"{row['stem']}.npy"
    if p.exists():
        with open(p, "rb") as f:
            np.lib.format.read_magic(f)
            shape, _, _ = np.lib.format.read_array_header_1_0(f)
            return int(shape[0])
    return len(feat_any(row, kind))


def warm_lengths(rows, kind="face256"):
    """Materialize any missing caches once (GSL first-access) and return the
    full length list needed for bucketing."""
    return [len_of(r, kind) for r in rows]


# ------------------------------------------------------------------ grouping

def build_groups(rows):
    """text-normalized groups: {norm_text: [row indices]}."""
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(r["text"].strip().lower(), []).append(i)
    return groups


def groups_by_src(rows, groups):
    """Split group keys by source (mixed-source groups don't occur: each
    text lives in exactly one of gh/mv/gsl/jw*)."""
    out = {}
    for key, idx in groups.items():
        out.setdefault(rows[idx[0]]["src"], []).append(key)
    return out


# ------------------------------------------------------------------ sampling

def mix_index(rows, groups, n_target, fractions, rng):
    """One epoch's index list: `n_target` slots, each slot samples a source
    by fraction, a text group uniformly within the source, and a take
    uniformly within the group. Fractions: gh gets the remainder; a special
    'jw' share is spread evenly across every jw* source present. Repeats are
    identical-sampling (new take each time), so every appearance of a text
    stays visually fresh while text exposure stays balanced."""
    gsrc = groups_by_src(rows, groups)
    weights = {"gh": max(0.0, 1.0 - sum(fractions.values()))}
    for s, f in fractions.items():
        weights[s] = f
    picks = {s: [] for s in weights}
    jw_keys = []
    for s, keys in gsrc.items():
        if s in ("gh", "mv", "gsl"):
            if weights.get(s, 0) > 0 and keys:
                picks[s] = keys
        else:
            jw_keys += keys
    if jw_keys and weights.get("jw", 0) > 0:
        picks["jw"] = jw_keys
    eff = {s: w for s, w in weights.items() if picks.get(s) and w > 0}
    tot = sum(eff.values())
    srcs = list(eff)
    probs = [eff[s] / tot for s in srcs]
    idx = []
    for _ in range(n_target):
        s = rng.choices(srcs, weights=probs)[0]
        key = rng.choice(picks[s])
        idx.append(rng.choice(groups[key]))
    # avoid two adjacent identical takes
    rng.shuffle(idx)
    return idx


def stage2_index(rows, groups, mv_takes, rng):
    """Stage-2 epoch: every gh AND jw verse row (JW is the main training
    source) + mv capped at `mv_takes` takes per unique sentence. gsl is never
    a CE target."""
    idx = [i for i, r in enumerate(rows) if r["src"] != "mv"]
    for key in groups_by_src(rows, groups).get("mv", []):
        pool = groups[key]
        k = min(mv_takes, len(pool))
        idx += rng.sample(pool, k)
    rng.shuffle(idx)
    return idx


def index_stats(rows, idx):
    """Source composition of one epoch's index list (for logs / paper)."""
    out = {}
    for i in idx:
        s = rows[i]["src"]
        out[s] = out.get(s, 0) + 1
    return out


# ------------------------------------------------------------------ datasets

class PairDataset(Dataset):
    """Contrastive pairs: two independent augmented video views + raw text."""

    def __init__(self, rows, kind="face256"):
        self.rows = rows
        self.kind = kind

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x1 = feat_any(r, self.kind)
        x2 = augment_var(x1.copy())
        return {"src": torch.from_numpy(x1), "src2": torch.from_numpy(x2),
                "text": r["text"], "stem": r["stem"]}


class EncodeDataset(Dataset):
    """Eval-side: no augmentation, single view per row."""

    def __init__(self, rows, kind="face256"):
        self.rows = rows
        self.kind = kind

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x = feat_any(r, self.kind)
        return {"src": torch.from_numpy(x), "src2": torch.from_numpy(x),
                "text": r["text"], "stem": r["stem"]}


class VarRowDataset(Dataset):
    """Stage-2 CE/CTC dataset: rows-subset of SignDatasetVar semantics,
    implemented over shared reprs so checkpoint/data format never diverges."""

    def __init__(self, split_rows, tok, kind="face256", augment_flag=False):
        self.rows = split_rows
        self.tok = tok
        self.kind = kind
        self.augment_flag = augment_flag

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        f = feat_any(r, self.kind)
        if self.augment_flag:
            f = augment_var(f)
        return {"src": torch.from_numpy(f), "tgt": torch.tensor(
            self.tok.encode(r["text"]), dtype=torch.long),
            "stem": r["stem"], "text": r["text"]}


# ------------------------------------------------------------------ batching

def var_lens(idx, kind="face256"):
    return [len_of(v) for v in idx]


def bucketed_batches(ds, lens, batch, shuffle=True, bucket_batches=20, rng=None):
    """Same bucketing scheme as dataset.bucketed_loader, index-list based:
    sort into length buckets so a batch pads only ~20-30% waste, shuffle
    bucket order + intra-bucket order every call."""
    idx = list(range(len(ds)))
    if shuffle:
        rng = rng or random
        rng.shuffle(idx)
        idx.sort(key=lambda i: lens[i])
        buckets = [idx[i:i + batch * bucket_batches]
                   for i in range(0, len(idx), batch * bucket_batches)]
        rng.shuffle(buckets)
        batches = []
        for bkt in buckets:
            bl = [bkt[i:i + batch] for i in range(0, len(bkt), batch)]
            rng.shuffle(bl)
            batches.extend(bl)
    else:
        idx.sort(key=lambda i: lens[i])
        batches = [idx[i:i + batch] for i in range(0, len(idx), batch)]
    return batches
