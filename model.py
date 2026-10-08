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
        # enable_nested_tensor=False: the nested-tensor fast path for padding
        # masks is not implemented on MPS (torch 2.x) and crashes; the masked
        # math is identical, just the slower standard kernel.
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout,
                                       batch_first=True), enc_layers,
            enable_nested_tensor=False)
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


class TemporalConvFrontend(nn.Module):
    """1D conv stack over time: T frames -> T/stride tokens.

    Sign is redundant at 30fps (each sign sampled 8-15x); convs learn what to
    keep per local window instead of blind uniform dropping. k=3/pad=1/stride=2
    per layer so out_len = ceil(in_len / 2) exactly; 2 layers = 4x (256->64),
    3 layers = 8x (256->32). Operates on (B, T, D) via transpose.
    """

    def __init__(self, in_dim, d_model=256, layers=2):
        super().__init__()
        assert layers in (2, 3)
        self.layers = layers
        self.stride_total = 2 ** layers
        ch = [in_dim, 256, 256, d_model][: layers + 1]
        convs = []
        for i in range(layers):
            convs += [nn.Conv1d(ch[i], ch[i + 1], kernel_size=3,
                                stride=2, padding=1), nn.ReLU()]
        self.convs = nn.Sequential(*convs)

    @staticmethod
    def out_len(n, layers):
        for _ in range(layers):
            n = (n + 1) // 2  # ceil(n/2) for k=3/pad=1/stride=2
        return n

    def forward(self, src):
        # src: (B, T, D) -> (B, D, T) -> convs -> (B, d_model, T') -> (B, T', d_model)
        return self.convs(src.transpose(1, 2)).transpose(1, 2)

    def downsample_mask(self, mask, in_len=None):
        """Strided-OR over time: output pad iff all receptive inputs are pad.

        mask: (B, T) bool True=pad. Approximated by max-pooling the inverse
        (valid=1) with k=3/stride=2/pad=1 per layer, then inverting back.
        """
        import torch.nn.functional as F

        valid = (~mask).float().unsqueeze(1)  # (B, 1, T)
        for _ in range(self.layers):
            valid = F.max_pool1d(valid, kernel_size=3, stride=2, padding=1)
        return valid.squeeze(1) < 0.5  # (B, T') bool True=pad


class SignTransformerV2(nn.Module):
    """Variable-length sister: (B, Tv<=256, D) -> vocab logits.

    Frontend downsamples time 4-8x, encoder runs on T'<=64 tokens with a real
    src_key_padding_mask (no more fixed-64 squash). Decoder identical to v1.
    ckpt cfg carries {"variable": True, "frontend_layers": 2}.
    """

    def __init__(self, vocab_size, in_dim=633, d_model=256, nhead=4,
                 enc_layers=4, dec_layers=4, dim_ff=1024, dropout=0.1,
                 frontend_layers=2):
        super().__init__()
        self.frontend = TemporalConvFrontend(in_dim, d_model, frontend_layers)
        self.tgt_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = SinusoidalPos(d_model, max_len=512)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout,
                                       batch_first=True), enc_layers,
            enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout,
                                       batch_first=True), dec_layers)
        self.head = nn.Linear(d_model, vocab_size)
        # TxtCTC head: frame-to-token alignment without glosses. Linear map
        # encoder states -> vocab distribution trained with CTC over the
        # transcript's own token ids (blank=PAD=0, never emitted by tokenizers).
        self.ctc_head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.frontend_layers = frontend_layers

    def encode(self, src, src_mask=None):
        tok = self.frontend(src)  # (B, T', d_model)
        if src_mask is not None:
            mem_mask = self.frontend.downsample_mask(src_mask).to(src.device)
        else:
            mem_mask = None
        # NOTE: MPS auto-dispatch only. Do NOT force an SDPA backend here:
        # forcing MATH crashes on dropout; auto mode is proven over 90k steps.
        return self.encoder(self.pos(tok), src_key_padding_mask=mem_mask)

    def _causal(self, L, device):
        return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, src, tgt_in, tgt_pad_mask, src_mask=None,
                mem=None, return_mem=False):
        # mem passthrough lets scheduled sampling reuse pass-1's encoding so
        # only the (cheap) decoder runs twice.
        if mem is None:
            mem = self.encode(src, src_mask)
        if src_mask is not None:
            mem_mask = self.frontend.downsample_mask(src_mask).to(src.device)
        else:
            mem_mask = None
        out = self.decoder(self.pos(self.tgt_emb(tgt_in)), mem,
                           tgt_mask=self._causal(tgt_in.size(1), src.device),
                           tgt_key_padding_mask=tgt_pad_mask,
                           memory_key_padding_mask=mem_mask)
        logits = self.head(out)
        return (logits, mem) if return_mem else logits

    def frontend_lens(self, src_lens):
        """Input frame counts -> frontend token counts (exact for k3/pad1/s2)."""
        return torch.tensor(
            [TemporalConvFrontend.out_len(int(n), self.frontend_layers)
             for n in src_lens], dtype=torch.long, device=src_lens.device)

    def ctc_log_probs(self, mem):
        """Encoder states -> per-token log-distributions over vocab (T', B, V)."""
        return self.ctc_head(mem).log_softmax(-1).transpose(0, 1)

    @torch.no_grad()
    def greedy(self, src, src_mask=None, max_len=60):
        self.eval()
        mem = self.encode(src, src_mask)
        if src_mask is not None:
            mem_mask = self.frontend.downsample_mask(src_mask).to(src.device)
        else:
            mem_mask = None
        B = src.size(0)
        ys = torch.full((B, 1), SOS_ID, dtype=torch.long, device=src.device)
        for _ in range(max_len):
            nxt = self.head(self.decoder(
                self.pos(self.tgt_emb(ys)), mem,
                tgt_mask=self._causal(ys.size(1), src.device),
                memory_key_padding_mask=mem_mask))[:, -1:].argmax(-1)
            ys = torch.cat([ys, nxt], dim=1)
            if (nxt == EOS_ID).all():
                break
        return ys
