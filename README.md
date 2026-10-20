# MedSign

Machine translation of Ghanaian Sign Language video into plain English text.

Most sign-language "translation" systems stop at glosses: sequences of dictionary
labels that a human still has to interpret. MedSign skips that step entirely.
A video of a signer goes in, a natural-language sentence comes out. The model is
built for hospital communication first (clinics, maternity, ANC), with the broad
Ghanaian Sign Language vocabulary as the longer-term goal.

## Results

Put simply: on medical sentences the model has never seen, about one in three
translations comes back correct in meaning, and one in six comes back
word-for-word exact.

| Test set | BLEU score | Correct in meaning | Word-for-word exact |
|---|---|---|---|
| Medical sentences, signed by a person the model never saw | 20.0 | 29.4% (4,410 of 14,994) | 17.2% |
| Medical sentences, held back from training for validation | 23.3 | 32.9% (1,150 of 3,494) | 19.6% |

What these rows mean, for anyone meeting them for the first time:

* The test rows above are the measure that matters. The signer in it was never
  in the training videos, so the score reflects generalization to new people,
  not memorization. The vocabulary is tuned toward healthcare and similar
  everyday sentences; that is where the available data is, so that is the
  setting these numbers describe.
* "Correct in meaning" counts a translation as right when it matches the
  reference sentence after minor rewording (80% word overlap or better, or
  exact after normalizing capitalization and punctuation). Humans judge this
  as a good translation.
* BLEU score (0-100) is the standard machine-translation metric: it grades how
  much word overlap survives between the translation and a reference sentence,
  and it punishes paraphrasing. "Word-for-word exact" is the strictest bar,
  the sentence reproduced exactly.

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
