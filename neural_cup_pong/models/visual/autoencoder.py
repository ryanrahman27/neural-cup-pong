"""State-grounded neural decoder (the learned renderer) + optional encoder.

Decoder(state, hints[, z]) -> frame. Conditioned on the structured state via
FiLM and on rasterized geometry-hint planes, so it paints appearance onto known
geometry. Deterministic conv net, nearest-upsample (no checkerboard on the ~2px
ball), sigmoid RGB. Stage 1 ships state-only (z = zeros).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import N_HINT
from .. import layout as L

LATENT_C, LATENT_H, LATENT_W = 16, 12, 16


def cgn(ci, co, s=1):
    return nn.Sequential(nn.Conv2d(ci, co, 3, s, 1), nn.GroupNorm(8, co), nn.SiLU())


class StateFiLM(nn.Module):
    def __init__(self, w, state_dim=L.STATE_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 128), nn.SiLU(), nn.Linear(128, 2 * w))

    def forward(self, s):
        g, b = self.net(s).chunk(2, -1)
        return g[..., None, None], b[..., None, None]


class Up(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.c = cgn(ci, co)
        self.film = StateFiLM(co)

    def forward(self, x, s):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.c(x)
        g, b = self.film(s)
        return x * (1 + g) + b


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(LATENT_C + N_HINT, 128, 1)
        self.u1 = Up(128, 96)
        self.u2 = Up(96, 64)
        self.u3 = Up(64, 32)
        self.refine = cgn(32 + N_HINT, 32)
        self.out = nn.Conv2d(32, 3, 3, 1, 1)

    def forward(self, state, hints, z=None):
        B = state.shape[0]
        if z is None:
            z = torch.zeros(B, LATENT_C, LATENT_H, LATENT_W, device=state.device)
        hint_small = F.adaptive_avg_pool2d(hints, (LATENT_H, LATENT_W))
        x = self.stem(torch.cat([z, hint_small], 1))     # [B,128,12,16]
        x = self.u1(x, state)                            # 24x32
        x = self.u2(x, state)                            # 48x64
        x = self.u3(x, state)                            # 96x128
        x = self.refine(torch.cat([x, hints], 1))
        return torch.sigmoid(self.out(x))                # [B,3,96,128] in [0,1]

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class Encoder(nn.Module):
    """Optional appearance encoder (escalation #1; unused in the state-only ship)."""

    def __init__(self):
        super().__init__()
        self.stem = cgn(3, 32)
        self.d1, self.d2, self.d3 = cgn(32, 64, 2), cgn(64, 96, 2), cgn(96, 128, 2)
        self.res = cgn(128, 128)
        self.head = nn.Conv2d(128, LATENT_C, 1)

    def forward(self, frame01):
        x = frame01 * 2 - 1
        x = self.res(self.d3(self.d2(self.d1(self.stem(x)))))
        return torch.tanh(self.head(x))
