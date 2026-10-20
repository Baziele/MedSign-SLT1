# MedSign

Machine translation of Ghanaian Sign Language video into plain English text.

Most sign-language "translation" systems stop at glosses: sequences of dictionary
labels that a human still has to interpret. MedSign skips that step entirely.
A video of a signer goes in, a natural-language sentence comes out. The model is
built for hospital communication first (clinics, maternity, ANC), with the broad
Ghanaian Sign Language vocabulary as the longer-term goal.

## Results

| Test set | BLEU | Exact sentences |
|---|---|---|
| hospital-held-out validation | 23.3 | 685 of 3,494 (20%) |
| signer-holdout validation | 30.2 | n/a |
| long-form prose (held-out test chapters) | 1.7 | 7 of 363 |

The previous baseline scored 19.0 BLEU on the same validation split and could
not reproduce a single full sentence exactly. Long-form prose is the honest
weak spot: verses run 25+ words and the current decoder is small, so it is the
first target for the next generation.

## Training data

The model was trained on more than 330 hours of sign language video.
Data preparation, text alignment, and train/val/test splitting are fully
scripted in this repository; raw source videos are not distributed here.

## How the model works

Sign language carries meaning in the hands, the face, and the body at once, so
the input is not a single stream. From each video frame the pipeline extracts
MediaPipe keypoints: 225-dim body pose plus 633-dim face and hand landmarks,
collected through the pipeline's extractor scripts.

The model itself is small (about 12M parameters), which keeps it trainable on a
laptop GPU. It runs in two training phases:

* **Alignment pretraining.** A contrastive (CLIP-style) phase pairs sign video
  embeddings with sentence embeddings from a BART text encoder, teaching the
  pose tower what its visual features mean in language before any translation
  happens.
* **Translation training.** A temporal-convolution frontend downsamples frames
  into tokens, a 4-layer transformer encoder codes them, and a 4-layer decoder
  generates 8k-BPE sentence pieces. The loss is plain cross-entropy plus an
  auxiliary CTC head that aligns frames to transcript tokens directly, so no
  human-written gloss labels are ever needed. The pose tower starts from the
  alignment-pretrained weights instead of randomly.

## Reproducing the pipeline

Set up the environment (Python 3.12):

```bash
python3.12 -m venv .venv312
.venv312/bin/pip install -r requirements.txt
```

The train/val/test split CSVs are tracked in `data/splits/`, so evaluation is
reproducible without touching the raw sources. For a full retrain, bring your
own videos, then:

```bash
# 1. Extract keypoints from raw videos (MediaPipe).
python extract_keypoints.py            # body-face-hands, per-video npz
python scripts/extract/jwextract.py --workers 8 --src /path/to/videos --dst data/jw/<source>

# 2. Verify every video/text pair before training (trims, alignment, lengths).
python scripts/dev/verify_jw_pairs.py

# 3. Alignment pretraining: contrastive video-to-text.
python -m pipeline.contrastive --tag align_v3 --epochs 50 --batch 32 \
    --jw-frac 0.25 --jw-source nwt-ghs,mwb,lfb

# 4. Translation training, initializing the pose tower from step 3.
python -m pipeline.stage2 --tag p1t2 --epochs 100 --batch 64 \
    --from-pose models/checkpoints/pose_text_best_v3.pt \
    --jw-source nwt-ghs,mwb,lfb

# 5. Evaluate a finished checkpoint on every held-out pool.
python scripts/eval_final.py --ckpt models/checkpoints/best_p1t2.pt \
    --out logs/eval_final_p1t2.json

# 6. Translate new video.
python translate.py --split test_T --beam 5 -o translation/preds.csv
python translate_video.py path/to/video.mp4
```

Training runs on Apple silicon (MPS) and CUDA, single GPU.

## Roadmap

The next generation replaces the fresh-trained decoder with an mBART-family
backbone (for stronger long-sentence generation), adds beam search at decode
time, and trains on additional video already collected for this purpose.
After that comes cross-lingual transfer from related sign languages and the
move from hospital sentences to everyday conversation.

## Contributing

Issues and pull requests are welcome. The pipeline scripts are plain and
single-purpose, so they should read easily. If you work with sign-language
data in Ghana and can contribute recordings or corrections, please open an
issue.

## License

License to be selected; until then, the code is all-rights-reserved by the
MedSign project authors. Please contact the maintainers before reuse.
