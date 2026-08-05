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


def _desired_aim_power(target: np.ndarray, rng: np.random.Generator, skill: float):
    dx = float(target[0] - C.THROW_ORIGIN[0])
    dy = float(target[1] - C.THROW_ORIGIN[1])
    aim_angle = float(np.arctan2(dx, dy))
    aim_x = float(np.clip(aim_angle / C.MAX_AIM_ANGLE, -1.0, 1.0))
    R = float(np.hypot(dx, dy))
    speed = float(np.sqrt(R * C.GRAVITY / max(1e-3, np.sin(2 * C.LAUNCH_ELEV))))
    power = float(np.clip((speed - C.POWER_MIN) / (C.POWER_MAX - C.POWER_MIN), 0.0, 1.0))
    # aiming error for variety
    err = (1.0 - skill)
    aim_x = float(np.clip(aim_x + rng.uniform(-1, 1) * err * 0.5, -1.0, 1.0))
    power = float(np.clip(power + rng.uniform(-1, 1) * err * 0.12, 0.0, 1.0))
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
