"""BPE subword tokenizer with the same interface as tokenizer.Tokenizer.

IDs mirror the word tokenizer: <pad>=0 <sos>=1 <eos>=2 <unk>=3, pieces 4+.
SOS/EOS are added manually (not HF specials) so encode/decode stay fully
controlled; HF handles pieces + <unk> only. File: bpe_all_8k.json.

Train: .venv312/bin/python bpe_tokenizer.py  (reads splits/all_train.csv)
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPLITS_DIR = ROOT / "splits"

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
PAD_ID, SOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
SPECIALS = [PAD, SOS, EOS, UNK]


class BPETokenizer:
    def __init__(self, hf_path):
        from tokenizers import Tokenizer as HFTok
        p = Path(hf_path)
        if not p.is_absolute():
            p = ROOT / p
        self.hf_path = str(p)
        self.hf = HFTok.from_file(str(p))
        self._n = self.hf.get_vocab_size()

    @property
    def tok2id(self):  # compat shim (pieces live in HF vocab)
        return {t: i for t, i in self.hf.get_vocab().items()}

    def encode(self, text, add_markers=True):
        ids = self.hf.encode(text.lower()).ids
        ids = [UNK_ID if i == self.hf.token_to_id(UNK) else i for i in ids]
        return ([SOS_ID] + ids + [EOS_ID]) if add_markers else ids

    def decode(self, ids, skip_specials=True):
        ids = [i for i in ids
               if not (skip_specials and i in (PAD_ID, SOS_ID, EOS_ID, UNK_ID))]
        s = self.hf.decode(ids, skip_special_tokens=True)
        return " ".join(s.split())  # normalize BPE spacing artifacts

    def save(self, path):
        json.dump({"bpe": self.hf_path, "type": "bpe"}, open(path, "w"))

    @classmethod
    def load(cls, path):
        return cls(json.load(open(path))["bpe"])

    def __len__(self):
        return self._n


def train_bpe(texts, out="bpe_all_8k.json", vocab_size=8000):
    from tokenizers import Tokenizer as HFTok
    from tokenizers.decoders import ByteLevel as ByteLevelDec
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer
    tok = HFTok(BPE(unk_token=UNK))
    tok.pre_tokenizer = ByteLevel()
    tok.decoder = ByteLevelDec()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS,
                         show_progress=False)
    tok.train_from_iterator((t.lower() for t in texts), trainer=trainer)
    tok.save(str(ROOT / out))
    BPETokenizer(ROOT / out).save(ROOT / (out + ".wrap"))
    print(f"bpe vocab: {tok.get_vocab_size()} -> {out} (+ {out}.wrap)")
    return out


if __name__ == "__main__":
    texts = []
    for split in ["all_train"]:
        with open(SPLITS_DIR / f"{split}.csv") as f:
            texts += [r["text"] for r in csv.DictReader(f)]
    print(f"training texts: {len(texts)} clips")
    train_bpe(texts)
