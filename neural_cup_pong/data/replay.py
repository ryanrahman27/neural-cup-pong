"""Headless inspection: tile sampled frames from an episode into one PNG."""

from __future__ import annotations

import os

import numpy as np

from .dataset import Episode, load_episode


def export_contact_sheet(episode, out_png, num=25, cols=5, scale=3) -> str:
    import pygame

    if not pygame.get_init():
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
    ep = load_episode(episode, with_frames=True) if isinstance(episode, str) else episode
    if not ep.frames.size:
        raise ValueError("episode has no frames")
    T, H, W, _ = ep.frames.shape
    idx = np.linspace(0, T - 1, min(num, T)).astype(int)
    rows = (len(idx) + cols - 1) // cols
    fw, fh = W * scale, H * scale
    sheet = pygame.Surface((cols * fw, rows * fh))
    sheet.fill((18, 18, 24))
    for k, t in enumerate(idx):
        surf = pygame.surfarray.make_surface(np.transpose(ep.frames[t], (1, 0, 2)))
        surf = pygame.transform.scale(surf, (fw, fh))
        r, c = divmod(k, cols)
        sheet.blit(surf, (c * fw, r * fh))
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    pygame.image.save(sheet, out_png)
    return out_png
