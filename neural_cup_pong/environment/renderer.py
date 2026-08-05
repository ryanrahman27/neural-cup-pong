"""Fixed-camera 2.5D renderer: table + cups + ball(+shadow) + aim reticle + HUD.

Deterministic. Ball carries a ground-pinned shadow (a depth cue the world model
can read), and the aim reticle marks the predicted landing point.
"""

from __future__ import annotations

import os

import numpy as np
import pygame

from . import constants as C
from .state import GameState


def _ensure_video() -> None:
    if pygame.display.get_init():
        return
    try:
        pygame.display.init()
    except pygame.error:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.display.init()


class Renderer:
    def __init__(self, width: int = C.OBS_W, height: int = C.OBS_H) -> None:
        if not pygame.get_init():
            pygame.init()
        _ensure_video()
        try:
            if not pygame.font.get_init():
                pygame.font.init()
        except Exception:
            pass
        self.W, self.H = width, height
        self.NEAR_Y, self.FAR_Y = 0.90 * height, 0.28 * height
        self.NEAR_XL, self.NEAR_XR = 0.08 * width, 0.92 * width
        self.FAR_XL, self.FAR_XR = 0.34 * width, 0.66 * width
        self.LIFT = 0.006 * height
        self.surface = pygame.Surface((width, height))
        self._cups = C.cup_layout()
        self._font = pygame.font.SysFont("consolas", max(8, int(height * 0.06)))

    # --- projection ----------------------------------------------------------
    def _xspan(self, t):
        return (self.NEAR_XL + (self.FAR_XL - self.NEAR_XL) * t,
                self.NEAR_XR + (self.FAR_XR - self.NEAR_XR) * t)

    def project(self, x, y, z=0.0):
        t = y / C.TABLE_D
        scale = 1.0 - 0.45 * t
        xl, xr = self._xspan(t)
        px = xl + (x / C.TABLE_W) * (xr - xl)
        py_floor = self.NEAR_Y + (self.FAR_Y - self.NEAR_Y) * t
        py = py_floor - z * self.LIFT * scale
        return px, py, py_floor, scale

    def _dxdu(self, y):
        t = y / C.TABLE_D
        xl, xr = self._xspan(t)
        return (xr - xl) / C.TABLE_W

    # --- render --------------------------------------------------------------
    def render(self, state: GameState) -> np.ndarray:
        surf = self.surface
        surf.fill(C.COLOR_BG)
        self._draw_table(surf)
        if state.game_phase == C.PHASE_AIM:
            self._draw_reticle(surf, state)
        # z-sorted cups + ball (far first)
        actors = [(self._cups[c, 1], "cup", c) for c in range(C.NUM_CUPS) if state.cups_present[c]]
        if state.game_phase in (C.PHASE_AIM, C.PHASE_FLIGHT):
            actors.append((float(state.ball_position[1]), "ball", -1))
        actors.sort(key=lambda a: a[0], reverse=True)
        for _, kind, idx in actors:
            if kind == "cup":
                self._draw_cup(surf, self._cups[idx])
            else:
                self._draw_ball(surf, state)
        self._draw_hud(surf, state)
        arr = pygame.surfarray.array3d(surf)
        return np.transpose(arr, (1, 0, 2)).copy()

    def _draw_table(self, surf) -> None:
        # felt trapezoid (near->far gradient via depth bands)
        for i in range(12):
            t0, t1 = i / 12.0, (i + 1) / 12.0
            xl0, xr0 = self._xspan(t0); xl1, xr1 = self._xspan(t1)
            y0 = self.NEAR_Y + (self.FAR_Y - self.NEAR_Y) * t0
            y1 = self.NEAR_Y + (self.FAR_Y - self.NEAR_Y) * t1
            col = tuple(int(C.COLOR_TABLE_NEAR[k] + (C.COLOR_TABLE_FAR[k] - C.COLOR_TABLE_NEAR[k]) * t0)
                        for k in range(3))
            pygame.draw.polygon(surf, col, [(xl0, y0), (xr0, y0), (xr1, y1), (xl1, y1)])
        # boundary
        poly = [self.project(0, 0)[:2], self.project(C.TABLE_W, 0)[:2],
                self.project(C.TABLE_W, C.TABLE_D)[:2], self.project(0, C.TABLE_D)[:2]]
        pygame.draw.polygon(surf, C.COLOR_TABLE_LINE, poly, max(1, self.W // 160))

    def _draw_cup(self, surf, cup) -> None:
        px, py, _, scale = self.project(float(cup[0]), float(cup[1]), 0.0)
        px, py, scale = float(px), float(py), float(scale)
        top_y = py - C.CUP_H * self.LIFT * scale
        rpx = float(max(2.0, C.CUP_R * self._dxdu(float(cup[1]))))
        ry = max(1.0, rpx * 0.42)

        def P(dx, dy):
            return (int(px + dx), int(dy))

        # base shadow
        _ellipse(surf, C.COLOR_BALL_SHADOW, px, py + ry * 0.4, rpx, ry, 90)
        # body
        pygame.draw.polygon(surf, C.COLOR_CUP_DARK,
                            [P(-rpx, py), P(rpx, py), P(rpx * 0.92, top_y), P(-rpx * 0.92, top_y)])
        pygame.draw.polygon(surf, C.COLOR_CUP,
                            [P(-rpx * 0.92, py), P(rpx * 0.5, py), P(rpx * 0.5, top_y), P(-rpx * 0.92, top_y)])
        # rim + inner opening
        pygame.draw.ellipse(surf, C.COLOR_CUP_RIM, (int(px - rpx), int(top_y - ry), int(rpx * 2), int(ry * 2)))
        pygame.draw.ellipse(surf, C.COLOR_CUP_INNER,
                            (int(px - rpx * 0.72), int(top_y - ry * 0.7), int(rpx * 1.44), int(ry * 1.4)))

    def _draw_ball(self, surf, state) -> None:
        bx, by, bz = (float(v) for v in state.ball_position)
        px, py, py_floor, scale = self.project(bx, by, bz)
        r = max(1, C.BALL_R * self._dxdu(by) * 1.2)
        # ground shadow (gap encodes height)
        _ellipse(surf, C.COLOR_BALL_SHADOW, px, py_floor, r * (1 + 0.03 * bz), r * 0.45,
                 max(40, int(150 - 4 * bz)))
        pygame.draw.circle(surf, (60, 60, 58), (int(px), int(py)), int(r + 1))
        pygame.draw.circle(surf, C.COLOR_BALL, (int(px), int(py)), int(r))

    def _draw_reticle(self, surf, state) -> None:
        lx, ly = self._predicted_landing(state)
        px, py, _, _ = self.project(lx, ly, 0.0)
        s = max(3, int(self.W * 0.03))
        pygame.draw.line(surf, C.COLOR_RETICLE, (int(px - s), int(py)), (int(px + s), int(py)), 1)
        pygame.draw.line(surf, C.COLOR_RETICLE, (int(px), int(py - s)), (int(px), int(py + s)), 1)
        pygame.draw.circle(surf, C.COLOR_RETICLE, (int(px), int(py)), s, 1)
        # power bar (bottom-left)
        bw, bh = int(self.W * 0.04), int(self.H * 0.30)
        bx0, by0 = int(self.W * 0.03), int(self.H * 0.60)
        pygame.draw.rect(surf, (60, 60, 66), (bx0, by0, bw, bh), 1)
        fh = int(bh * state.power)
        pygame.draw.rect(surf, C.COLOR_RETICLE, (bx0, by0 + bh - fh, bw, fh))

    def _predicted_landing(self, state):
        angle = state.aim_x * C.MAX_AIM_ANGLE
        speed = C.POWER_MIN + (C.POWER_MAX - C.POWER_MIN) * state.power
        hs = speed * float(np.cos(C.LAUNCH_ELEV))
        vz = speed * float(np.sin(C.LAUNCH_ELEV))
        z0 = float(C.THROW_ORIGIN[2])
        t = (vz + float(np.sqrt(vz * vz + 2 * C.GRAVITY * z0))) / C.GRAVITY
        lx = float(C.THROW_ORIGIN[0]) + hs * float(np.sin(angle)) * t
        ly = float(C.THROW_ORIGIN[1]) + hs * float(np.cos(angle)) * t
        return float(np.clip(lx, 0, C.TABLE_W)), float(np.clip(ly, 0, C.TABLE_D))

    def _draw_hud(self, surf, state) -> None:
        left = int(state.cups_present.sum())
        txt = self._font.render(f"CUPS {left}   THROWS {state.throws_used}", True, C.COLOR_HUD)
        surf.blit(txt, (4, 3))
        if state.game_phase == C.PHASE_GAME_OVER and self.H >= 200:
            big = pygame.font.SysFont("consolas", int(self.H * 0.12), bold=True)
            msg = "RACK CLEARED!" if left == 0 else "GAME OVER"
            t = big.render(msg, True, C.COLOR_RETICLE)
            surf.blit(t, (self.W // 2 - t.get_width() // 2, self.H // 2 - t.get_height() // 2))


def _ellipse(surf, color, cx, cy, rx, ry, alpha=255):
    rx, ry = max(1, int(rx)), max(1, int(ry))
    if alpha >= 255:
        pygame.draw.ellipse(surf, color, (int(cx - rx), int(cy - ry), rx * 2, ry * 2))
        return
    tmp = pygame.Surface((rx * 2, ry * 2), pygame.SRCALPHA)
    pygame.draw.ellipse(tmp, (*color, alpha), (0, 0, rx * 2, ry * 2))
    surf.blit(tmp, (int(cx - rx), int(cy - ry)))
