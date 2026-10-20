# MedSign — Ghanaian/Nigerian Sign Language translation

## Layout

- `train.py`, `model.py`, `dataset.py`, `tokenizer.py`, `bpe_tokenizer.py`,
  `preprocess.py`, `render_skeleton.py`, `extract_keypoints.py`,
  `gen_landmark_map.py`, `make_*_splits.py`, `translate.py`,
  `translate_video.py`, `retrieval_baseline.py` — product code (flat on
  purpose: the modules cross-import as top-level names).
- `tokenizer.json`, `tokenizer_all.json`, `bpe_all_8k.json` — vocab files.
  They stay at root because frozen checkpoint configs reference these
  exact paths.
- `scripts/extract/` — keypoint extraction (`jwextract.py` = JW pipeline
  with `--src/--dst`, `wextract.py` = multiview, `gextract.py` = GSL
  dictionary, `wlasl_extract.py`) + status tools (`xstat.py`, `jwstat.py`,
  `nnsstat.py`) + `run_nns_extract.sh` watchdog.
- `scripts/windows/` — Windows-box helpers (`wssh.py`, `wput.py`,
  `run_*.bat`, `launch_*.ps1`, `babysit.py`).
- `scripts/data_tools/` — `download_gsl.py`, `clean_dictionary.py`,
  `analyze_zip.py`.
- `bench/` — MediaPipe/kernel benchmarks (Kaggle-oriented).
- `data/` — all datasets (mostly git-ignored, reproducible):
  - `data/jw/nwt-ghs/` — Ghanaian SL Bible keypoints + `progress.jsonl`
  - `data/jw/nwt-nns/` — Nigerian SL Bible keypoints + `progress.jsonl`
  - `data/jw/mwb/` — MWB conventions keypoints (78, intro-skip 4s) +
    `text/` per-issue section transcripts; + `progress.jsonl`
  - `data/jw/lfb/` — LFB lesson keypoints (120, intro trimmed, Art
    sections skipped, `reading_text` expanded) + `text/` lesson
    transcripts; + `progress.jsonl`
  - `data/jw/thumbs/` — JW thumbnail cache
  - `data/multiview/keypoints/` — SignTalk multiview keypoints
    (`keypoints_corrupt/` = quarantined bad files)
  - `data/multiview/mv-data/`, `data/multiview/Medsign/` — multiview videos
  - `data/gsl/` — GSL dictionary videos/keypoints/manifests
  - `data/splits/` — frozen train/val/test CSVs (**tracked**)
  - `data/cache/` — extracted feature cache (`dataset.py`)
  - `data/live/` — self-test clips + their keypoints
- `models/` — vocab copies, registry, released checkpoints;
  `models/checkpoints/` — training outputs (`best*.pt`, `last*.pt`).
- `translation/`, `retrieval/`, `previews/`, `experiments/` —
  generated outputs (git-ignored, reproducible).
- `logs/` — all run logs. `scratch/` — personal clips, notes, one-off
  artifacts (never committed). `docs/` — notes (`multiview_split.md`).
- `apk/`, `apkwork/` — pulled Android APKs (legacy, ignored).
- Raw JW videos live in the **separate, unversioned**
  `medsign-data-pipeline/` folder (`nwt-ghs/`, `nns/`, `mwb/`, `lfb/`,
  `samples/` + `download_*.py` fetchers). MWB/LFB section transcripts
  live here under `data/jw/{mwb,lfb}/text/`; the backfill scripts in
  `medsign-data-pipeline/legacy/tools/` regenerate them if needed.

## Common commands (run from repo root)

```bash
.venv312/bin/python train.py --tag v1 [--resume]
.venv312/bin/python translate.py --split test_T --beam 5 -o translation/preds_test_T.csv
.venv312/bin/python scripts/extract/jwstat.py    # GHS extraction status
.venv312/bin/python scripts/extract/nnsstat.py   # NNS extraction status
.venv312/bin/python scripts/extract/jwextract.py --workers 8 \
    --src /path/to/videos --dst data/jw/nwt-nns  # generic JW extraction
```

NNS extraction is kept alive by launchd (`com.medsign.nnsextract` →
`scripts/extract/run_nns_extract.sh`); remove with
`launchctl unload ~/Library/LaunchAgents/com.medsign.nnsextract.plist`
once it reports complete.
