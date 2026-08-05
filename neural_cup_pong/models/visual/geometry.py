"""Torch port of the renderer projection + rasterized geometry-hint planes.

The decoder is conditioned on hint channels rasterized from the structured
state, so it learns *appearance*, never *localization* — ball/shadow/cups/
reticle inherit the Phase-3 model's controllability + legality for free. These
constants MUST match ``environment/renderer.py`` (unit-tested to <1px).
"""

from __future__ import annotations

import numpy as np
import torch

from ...environment import constants as C
from .. import layout as L

W, H = C.OBS_W, C.OBS_H            # 128, 96
NEAR_Y, FAR_Y = 0.90 * H, 0.28 * H
NEAR_XL, NEAR_XR = 0.08 * W, 0.92 * W
FAR_XL, FAR_XR = 0.34 * W, 0.66 * W
LIFT = 0.006 * H
N_HINT = 16    # 13 dynamic + 3 static empty-court backdrop (felt + boundary)
_CUPS = torch.tensor(C.cup_layout(), dtype=torch.float32)   # [6,2]


def _xspan(t):
    return NEAR_XL + (FAR_XL - NEAR_XL) * t, NEAR_XR + (FAR_XR - NEAR_XR) * t


def project(x, y, z):
    """Batched projection matching renderer.project(): returns (px, py, py_floor, scale)."""
    t = y / C.TABLE_D
    scale = 1.0 - 0.45 * t
    xl, xr = _xspan(t)
    px = xl + (x / C.TABLE_W) * (xr - xl)
    py_floor = NEAR_Y + (FAR_Y - NEAR_Y) * t
    py = py_floor - z * LIFT * scale
    return px, py, py_floor, scale


def dxdu(y):
    xl, xr = _xspan(y / C.TABLE_D)
    return (xr - xl) / C.TABLE_W


def predicted_landing(aim_x, power):
    angle = aim_x * C.MAX_AIM_ANGLE
    speed = C.POWER_MIN + (C.POWER_MAX - C.POWER_MIN) * power
    hs = speed * float(np.cos(C.LAUNCH_ELEV))
    vz = speed * float(np.sin(C.LAUNCH_ELEV))
    z0 = float(C.THROW_ORIGIN[2])
    t = (vz + torch.sqrt(vz * vz + 2 * C.GRAVITY * z0)) / C.GRAVITY
    lx = float(C.THROW_ORIGIN[0]) + hs * torch.sin(angle) * t
    ly = float(C.THROW_ORIGIN[1]) + hs * torch.cos(angle) * t
    return lx.clamp(0, C.TABLE_W), ly.clamp(0, C.TABLE_D)


def _gauss(gx, gy, cx, cy, sigma):
    # gx,gy: [H,W]; cx,cy,sigma: [B,1,1] -> [B,H,W]
    return torch.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2.0 * sigma ** 2))


_CACHE: dict = {}       # per-device: grid + static cup-disk masks + power-bar cols


def _cache(dev):
    key = str(dev)
    if key not in _CACHE:
        gy, gx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                                torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
        cups = _CUPS.to(dev)
        disks = torch.zeros(C.NUM_CUPS, H, W, device=dev)     # static cup geometry
        for c in range(C.NUM_CUPS):
            pcx, pcy, _, _ = project(cups[c, 0], cups[c, 1], torch.zeros((), device=dev))
            rpx = float(max(2.0, C.CUP_R * float(dxdu(cups[c, 1]))))
            disks[c] = (((gx - float(pcx)) ** 2 + (gy - float(pcy)) ** 2).sqrt() < rpx).float()
        rows = torch.arange(H, device=dev).view(H, 1)         # for the power-bar
        _CACHE[key] = (gx, gy, disks, rows, _backdrop(dev))
    return _CACHE[key]


def _backdrop(dev):
    """The static empty-court image (felt + boundary), identical every frame,
    as a [3,H,W] tensor in [0,1] — a known-geometry hint the decoder composites on."""
    import pygame
    from ...environment.renderer import Renderer
    r = Renderer(W, H)
    surf = pygame.Surface((W, H))
    surf.fill(C.COLOR_BG)
    r._draw_table(surf)
    arr = pygame.surfarray.array3d(surf).transpose(1, 0, 2)   # HWC uint8
    return torch.tensor(arr, dtype=torch.float32, device=dev).permute(2, 0, 1) / 255.0


@torch.no_grad()
def geometry_hints(state: torch.Tensor) -> torch.Tensor:
    """state [B,21] -> hint planes [B,13,H,W] in [0,1]. Vectorized + cached."""
    dev = state.device
    B = state.shape[0]
    gx, gy, disks, rows, backdrop = _cache(dev)
    out = torch.zeros(B, N_HINT, H, W, device=dev)

    bx, by, bz = state[:, 0], state[:, 1], state[:, 2]
    phase = state[:, L.PHASE].argmax(-1)
    airborne = ((phase == C.PHASE_AIM) | (phase == C.PHASE_FLIGHT)).float().view(B, 1, 1)

    px, py, pyf, _ = project(bx, by, bz)
    r = torch.clamp(C.BALL_R * dxdu(by) * 1.2, min=1.0)
    out[:, 0] = _gauss(gx, gy, px.view(B, 1, 1), py.view(B, 1, 1), r.view(B, 1, 1)) * airborne
    out[:, 1] = _gauss(gx, gy, px.view(B, 1, 1), pyf.view(B, 1, 1),
                       (r * 1.2).view(B, 1, 1)) * airborne

    # cups: static disks gated by cups_present
    out[:, 2:8] = disks.unsqueeze(0) * state[:, L.CUPS].view(B, C.NUM_CUPS, 1, 1)

    # reticle + power bar (AIM only), vectorized
    is_aim = (phase == C.PHASE_AIM).float().view(B, 1, 1)
    lx, ly = predicted_landing(state[:, L.AIM], state[:, L.POWER])
    rpx, rpy, _, _ = project(lx, ly, torch.zeros(B, device=dev))
    reticle = _gauss(gx, gy, rpx.view(B, 1, 1), rpy.view(B, 1, 1),
                     torch.full((B, 1, 1), 2.5, device=dev))
    bar_x0, bar_x1 = int(0.03 * W), int(0.03 * W) + max(1, int(0.04 * W))
    fill = (state[:, L.POWER] * 0.30 * H).clamp(0, H).view(B, 1, 1)   # [B,1,1]
    col_mask = torch.zeros(1, H, W, device=dev)
    col_mask[:, :, bar_x0:bar_x1] = 1.0
    bar = ((rows.view(1, H, 1) >= (H - fill)).float()) * col_mask
    out[:, 8] = (reticle + bar).clamp(0, 1) * is_aim

    out[:, 9:13] = state[:, L.PHASE].view(B, 4, 1, 1)
    out[:, 13:16] = backdrop.unsqueeze(0)          # static empty-court felt + boundary
    return out


@torch.no_grad()
def foreground_weight(frame01, state, hints=None):
    """Per-pixel loss weight [B,1,H,W]: upweight ball/shadow/reticle/HUD so the
    static background can't drown the few high-value pixels."""
    if hints is None:
        hints = geometry_hints(state)
    w = 1.0 + 7.0 * hints[:, 0:1] + 5.0 * hints[:, 1:2] + 3.0 * hints[:, 8:9]
    w[:, :, 0:int(0.10 * H), :] += 2.5             # HUD band (top rows)
    return w.clamp(1.0, 15.0)
