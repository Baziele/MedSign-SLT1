"""Pose/text contrastive towers + shared-space projection (phase-1 stage 1).

Pose tower deliberately mirrors SignTransformerV2's frontend/pos/encoder
module layout (same classes, same names) so stage-2 loads its state with a
plain key-prefix strip — no architecture reconciliation at handoff:
  pose.frontend.*  <-> model.frontend.*
  pose.pos.*       <-> model.pos.*
  pose.encoder.*   <-> model.encoder.*

Text side is a BART-family encoder (user-definable), mean-pooled and
projected; both towers output L2-normalized vectors. `scale` implements the
CLIP learnable temperature (init = ln(1/0.07)).
"""
import torch
import torch.nn.functional as F
from torch import nn

from model import SinusoidalPos, TemporalConvFrontend


def masked_mean(x, keep):
    """x (B,T,D), keep (B,T) bool True=valid -> (B,D)."""
    keep = keep.float().unsqueeze(-1)
    return (x * keep).sum(1) / keep.sum(1).clamp(min=1.0)


class PoseTower(nn.Module):
    """Conv frontend + transformer encoder + linear projection head."""

    def __init__(self, in_dim=633, d_model=256, nhead=4, enc_layers=4,
                 dim_ff=1024, dropout=0.1, proj_dim=256, frontend_layers=2):
        super().__init__()
        self.frontend = TemporalConvFrontend(in_dim, d_model, frontend_layers)
        self.pos = SinusoidalPos(d_model, max_len=512)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout,
                                       batch_first=True), enc_layers,
            enable_nested_tensor=False)
        self.proj = nn.Linear(d_model, proj_dim)
        self.frontend_layers = frontend_layers

    def encode_seq(self, src, pad_mask=None):
        tok = self.frontend(src)
        m = (self.frontend.downsample_mask(pad_mask)
             if pad_mask is not None else None)
        return self.encoder(self.pos(tok), src_key_padding_mask=m), m

    def embed(self, src, pad_mask=None):
        enc, m = self.encode_seq(src, pad_mask)
        keep = (torch.ones(enc.shape[:2], dtype=torch.bool, device=enc.device)
                if m is None else ~m)
        z = self.proj(masked_mean(enc, keep))
        return F.normalize(z, dim=-1)


class TextTower(nn.Module):
    """BART-family encoder: mean-pool token states, project, L2-normalize."""

    def __init__(self, model_name="facebook/bart-base", proj_dim=256):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.backbone.config.d_model, proj_dim)

    def embed(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids,
                            attention_mask=attention_mask, return_dict=True)
        z = self.proj(masked_mean(out.last_hidden_state,
                                  attention_mask != 0))
        return F.normalize(z, dim=-1)


class VideoText(nn.Module):
    """Both towers + learnable temperature, saved/loaded as one state."""

    def __init__(self, pose_kw, text_model="facebook/bart-base", proj_dim=256):
        super().__init__()
        pose_kw = {**pose_kw}  # don't mutate the caller's dict
        pose_kw.pop("proj_dim", None)  # supplied below, exactly once
        self.pose = PoseTower(proj_dim=proj_dim, **pose_kw)
        self.text = TextTower(text_model, proj_dim)
        self.scale = nn.Parameter(torch.tensor(2.659))  # ln(1/0.07)

    def forward(self, src1, m1, src2, m2, input_ids, attention_mask):
        v = self.pose.embed(src1, m1)
        v2 = self.pose.embed(src2, m2)
        t = self.text.embed(input_ids, attention_mask)
        s = self.scale.clamp(max=4.605)  # ln(100) cap, per CLIP
        return {"v": v, "v2": v2, "t": t,
                "s_vt": s * (v @ t.t()), "s_tv": s * (t @ v.t()),
                "s_vv": s * (v @ v2.t())}


def cl_losses(out, vv_weight=0.1):
    """Symmetric InfoNCE on (video, text) + SignCL anti-density term on the
    two video views. Diagonal is the positive in every matrix."""
    B = out["v"].shape[0]
    eye = torch.arange(B, device=out["v"].device)
    l_vt = F.cross_entropy(out["s_vt"], eye)
    l_tv = F.cross_entropy(out["s_tv"], eye)
    l_vv = F.cross_entropy(out["s_vv"], eye) if vv_weight > 0 else None
    loss = l_vt + l_tv
    if l_vv is not None:
        loss = loss + vv_weight * l_vv
    return {"loss": loss, "l_vt": l_vt, "l_tv": l_tv, "l_vv": l_vv}


def retrieval_metrics(V, T):
    """V, T: (N, D) L2-normalized embeddings of same-index val pairs.
    Video->text = rows, text->video = columns; diagonal is gold."""
    S = V @ T.t()
    n = len(S)
    diag = torch.arange(n, device=S.device)
    def _rk(Smat, k):
        top = Smat.topk(k, dim=1).indices
        return (top == diag.unsqueeze(1)).any(1).float().mean().item()
    pos = S[diag].mean().item()
    neg = (S.sum() - S[diag].sum()) / max(1, n * n - n)
    return {"r1_v2t": _rk(S, 1), "r5_v2t": _rk(S, 5),
            "r1_t2v": _rk(S.T, 1), "r5_t2v": _rk(S.T, 5),
            "sim_pos": pos, "sim_neg": neg.item()}
