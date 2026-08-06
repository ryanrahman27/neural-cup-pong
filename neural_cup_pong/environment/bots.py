"""Scripted thrower: aims at a present cup, sets power to reach it, then throws.

Deterministic given the env's seeded generator; intentionally imperfect so it
generates varied makes/misses for training data.
"""

from __future__ import annotations

import numpy as np

from . import actions as A
from . import constants as C
from .state import GameState

_AIM_EPS = 0.04
_POW_EPS = 0.03


def _target_cup(state: GameState) -> np.ndarray:
    cups = C.cup_layout()
    present = np.where(state.cups_present > 0)[0]
    return cups[present[0]] if len(present) else cups[0]


_LANDING_GRID = None


def _grid():
    """Cache a (aim, power) -> true rim-plane landing table (via physics)."""
    global _LANDING_GRID
    if _LANDING_GRID is None:
        from . import physics
        aims = np.linspace(-1.0, 1.0, 41)
        pows = np.linspace(0.1, 1.0, 41)
        land = np.array([[physics.simulate_landing(a, p) for p in pows] for a in aims])
        _LANDING_GRID = (aims, pows, land)
    return _LANDING_GRID


def _desired_aim_power(target: np.ndarray, rng: np.random.Generator, skill: float):
    # pick the (aim, power) whose actual landing is nearest the target cup,
    # then jitter by (1-skill) so makes/misses both appear in the data.
    aims, pows, land = _grid()
    d = np.hypot(land[..., 0] - float(target[0]), land[..., 1] - float(target[1]))
    i, j = np.unravel_index(int(d.argmin()), d.shape)
    err = 1.0 - skill
    aim_x = float(np.clip(aims[i] + rng.uniform(-1, 1) * err * 0.4, -1.0, 1.0))
    power = float(np.clip(pows[j] + rng.uniform(-1, 1) * err * 0.1, 0.0, 1.0))
    return aim_x, power


def scripted_thrower(state: GameState, rng: np.random.Generator, skill: float = 0.8) -> np.ndarray:
    if state.game_phase != C.PHASE_AIM:
        return A.empty_action()
    aim_t, pow_t = _desired_aim_power(_target_cup(state), rng, skill)
    a = A.empty_action()
    if state.aim_x < aim_t - _AIM_EPS:
        a[A.AIM_RIGHT] = 1.0
    elif state.aim_x > aim_t + _AIM_EPS:
        a[A.AIM_LEFT] = 1.0
    if state.power < pow_t - _POW_EPS:
        a[A.POWER_UP] = 1.0
    elif state.power > pow_t + _POW_EPS:
        a[A.POWER_DOWN] = 1.0
    if abs(state.aim_x - aim_t) <= _AIM_EPS and abs(state.power - pow_t) <= _POW_EPS:
        a[A.THROW] = 1.0
    return a
