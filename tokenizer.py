"""Word-level tokenizer built from train texts.

Tokenization: lowercase, split punctuation ([.,?!]) into own tokens.
Specials: <pad>=0 <sos>=1 <eos>=2 <unk>=3. All train words kept (min_freq=1);
<unk> fires only on val/test words never seen in train -- the OOV report in
__main__ quantifies the ceiling this imposes on test_T.
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPLITS_DIR = ROOT / "splits"
TOK_PATH = ROOT / "tokenizer.json"

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
SPECIALS = [PAD, SOS, EOS, UNK]
PAD_ID, SOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

_WORD = re.compile(r"[a-z0-9']+|[.,?!]")


def tokenize(text):
    return _WORD.findall(text.lower())


class Tokenizer:
    def __init__(self, tok2id):
        self.tok2id = dict(tok2id)
        self.id2tok = [t for t, _ in sorted(tok2id.items(), key=lambda kv: kv[1])]

    @classmethod
    def build(cls, texts):
        cnt = Counter()
        for t in texts:
            cnt.update(tokenize(t))
        toks = SPECIALS + sorted(cnt, key=lambda w: (-cnt[w], w))
        return cls({t: i for i, t in enumerate(toks)})

    def encode(self, text, add_markers=True):
        ids = [self.tok2id.get(w, UNK_ID) for w in tokenize(text)]
        return ([SOS_ID] + ids + [EOS_ID]) if add_markers else ids

    def decode(self, ids, skip_specials=True):
        toks = []
        for i in ids:
            t = self.id2tok[i]
            if skip_specials and t in SPECIALS:
                continue
            toks.append(t)
        s = " ".join(toks)
        return re.sub(r"\s+([.,?!])", r"\1", s)  # reattach punctuation

    def save(self, path=TOK_PATH):
        json.dump({"tok2id": self.tok2id}, open(path, "w"))

    @classmethod
    def load(cls, path=TOK_PATH):
        return cls(json.load(open(path))["tok2id"])

    def __len__(self):
        return len(self.id2tok)


def _texts(split):
    with open(SPLITS_DIR / f"{split}.csv") as f:
        return [r["text"] for r in csv.DictReader(f)]


if __name__ == "__main__":
    tok = Tokenizer.build(_texts("train"))
    tok.save()
    print(f"vocab size: {len(tok)}")
    for split in ["val", "test_T", "test_R"]:
        words, oov = 0, 0
        for t in _texts(split):
            for w in tokenize(t):
                words += 1
                oov += w not in tok.tok2id
        print(f"{split}: {oov}/{words} OOV words ({oov/words:.3f})")
