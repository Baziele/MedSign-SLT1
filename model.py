"""SignTransformer: (B, 64, 225) keypoints -> vocab logits.

src_proj is the "tokenizer" (poses -> vectors), sinusoidal positions give
order, encoder builds signer-invariant context, decoder writes English with
cross-attention into the encoder states.
"""
import math

import torch
import torch.nn as nn

from tokenizer import EOS_ID, PAD_ID, SOS_ID


class SinusoidalPos(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class SignTransformer(nn.Module):
    def __init__(self, vocab_size, in_dim=450, d_model=256, nhead=4,
                 enc_layers=6, dec_layers=2, dim_ff=1024, dropout=0.1):
        super().__init__()
        self.src_proj = nn.Linear(in_dim, d_model)
        self.tgt_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = SinusoidalPos(d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout,
                                       batch_first=True), enc_layers)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout,
                                       batch_first=True), dec_layers)
        self.head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def encode(self, src):
        return self.encoder(self.pos(self.src_proj(src)))

    def _causal(self, L, device):
        return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, src, tgt_in, tgt_pad_mask):
        mem = self.encode(src)
        out = self.decoder(self.pos(self.tgt_emb(tgt_in)), mem,
                           tgt_mask=self._causal(tgt_in.size(1), src.device),
                           tgt_key_padding_mask=tgt_pad_mask)
        return self.head(out)

    @torch.no_grad()
    def greedy(self, src, max_len=60):
        self.eval()
        mem = self.encode(src)
        B = src.size(0)
        ys = torch.full((B, 1), SOS_ID, dtype=torch.long, device=src.device)
        for _ in range(max_len):
            nxt = self.head(self.decoder(
                self.pos(self.tgt_emb(ys)), mem,
                tgt_mask=self._causal(ys.size(1), src.device)))[:, -1:].argmax(-1)
            ys = torch.cat([ys, nxt], dim=1)
            if (nxt == EOS_ID).all():
                break
        return ys
