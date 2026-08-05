"""Discrete thrower actions (5 bits): aim left/right, power up/down, throw."""

from __future__ import annotations

import numpy as np

ACTION_DIM: int = 5
AIM_LEFT: int = 0
AIM_RIGHT: int = 1
POWER_UP: int = 2
POWER_DOWN: int = 3
THROW: int = 4
ACTION_NAMES: tuple[str, ...] = ("aim_left", "aim_right", "power_up", "power_down", "throw")


def empty_action() -> np.ndarray:
    return np.zeros(ACTION_DIM, dtype=np.float32)


def make_action(aim_left=False, aim_right=False, power_up=False,
                power_down=False, throw=False) -> np.ndarray:
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    a[AIM_LEFT] = float(aim_left)
    a[AIM_RIGHT] = float(aim_right)
    a[POWER_UP] = float(power_up)
    a[POWER_DOWN] = float(power_down)
    a[THROW] = float(throw)
    return a


def pressed(a: np.ndarray, idx: int) -> bool:
    return bool(a[idx] > 0.5)
