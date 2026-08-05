"""Foreground-weighted pixel loss for the visual decoder.

L1 (sharper than L2) weighted so the ~2px ball / shadow / reticle / HUD dominate
the gradient instead of the near-static background, plus a light edge term to
keep those few-pixel features crisp.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..models.visual.geometry import foreground_weight


def _grad(x):
    gx = x[..., :, 1:] - x[..., :, :-1]
    gy = x[..., 1:, :] - x[..., :-1, :]
    return gx, gy


class VisualLoss(nn.Module):
    def __init__(self, edge_w: float = 0.1):
        super().__init__()
        self.edge_w = edge_w

    def forward(self, pred, target, state, hints=None):
        w = foreground_weight(target, state, hints)          # [B,1,H,W]
        l1 = (w * (pred - target).abs()).mean()
        pgx, pgy = _grad(pred)
        tgx, tgy = _grad(target)
        edge = (pgx - tgx).abs().mean() + (pgy - tgy).abs().mean()
        total = l1 + self.edge_w * edge
        return total, {"l1": float(l1.detach()), "edge": float(edge.detach()),
                       "total": float(total.detach())}
