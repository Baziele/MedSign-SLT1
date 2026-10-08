# Multiview extraction/training split (recorded 2026-09-18)

Total: 75,000 videos, 500 sentences, layout
`<split>/<split>/sentence_id_<N>/<vid>_<S>_cam<C>_rep<R>.mp4`.

Split is by sentence directory (sentence-complete on both sides:
all signers/cams/reps travel together):

- Mac (train 50k): sentence_id_1–333 in BOTH train/ and test/
  = 333×120 (train: S1–S4 × 5 cams × 6 reps) + 333×30 (test: S5 × 5 × 6)
  = 39,960 + 9,990 = 49,950 videos.
- Windows (train 25k): sentence_id_334–500 in BOTH train/ and test/
  = 167×120 + 167×30 = 20,040 + 5,010 = 25,050 videos.

Windows box: Documents/Code/medsign-multiview/ (archive.zip, data/,
wextract.py). E: drive gets full data/ mirror at E:\mv-data
(robocopy, re-runnable). When E: is on the Mac, copy only:
  train/train/sentence_id_1–333, test/test/sentence_id_1–333.
