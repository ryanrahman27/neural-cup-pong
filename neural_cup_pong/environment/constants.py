"""Geometry, physics, and rendering constants for Neural Cup Pong.

Fixed-camera 2.5D: a table in table-space (x across, y depth, z height) projected
to a small 2D frame. Aim left/right + charge power, then throw a ballistic ball
at a triangular rack of cups. Original art; deterministic.
"""

from __future__ import annotations

import numpy as np

# --- Timing ------------------------------------------------------------------
SIM_HZ: int = 60
DT: float = 1.0 / SIM_HZ
SIM_STEPS_PER_OBS: int = 3          # -> 20 Hz observations
OBS_HZ: float = SIM_HZ / SIM_STEPS_PER_OBS

# --- Table (table-space units) -----------------------------------------------
TABLE_W: float = 60.0               # x: left-right
TABLE_D: float = 100.0              # y: depth (near thrower -> far cups)
GRAVITY: float = 200.0              # z units/s^2

# --- Throw / aim -------------------------------------------------------------
THROW_ORIGIN = np.array([TABLE_W / 2.0, 6.0, 9.0], dtype=np.float32)
LAUNCH_ELEV: float = 0.95           # rad (~54deg) fixed arc
MAX_AIM_ANGLE: float = 0.35         # rad lateral aim range
POWER_MIN: float = 96.0             # launch speed at power=0
POWER_MAX: float = 162.0            # launch speed at power=1
AIM_RATE: float = 1.4               # aim_x units/s while held (aim_x in [-1,1])
POWER_RATE: float = 0.9             # power/s while held (power in [0,1])

# --- Ball / cups -------------------------------------------------------------
BALL_R: float = 1.6
CUP_R: float = 4.0                  # mouth radius
CUP_H: float = 9.0                  # cup height
CUP_RIM_Z: float = 8.0              # sink/rim-contact test height
RESTITUTION: float = 0.45           # table bounce on a miss

# Rim bounce: ball catches the rim ring (not clean through the hole) -> deflect.
RIM_RESTITUTION: float = 0.5        # vertical bounce energy kept
RIM_POP: float = 9.0                # extra upward pop on a rim hit
RIM_KICK: float = 15.0              # outward horizontal impulse
RIM_HDAMP: float = 0.55             # horizontal damping on a rim hit

NUM_CUPS: int = 6                   # triangular rack (3 + 2 + 1)


def cup_layout() -> np.ndarray:
    """Return (NUM_CUPS, 2) cup centers (x, y), triangle pointing at the thrower."""
    cx = TABLE_W / 2.0
    rows = [
        (90.0, [cx - 9.0, cx, cx + 9.0]),   # back row (farthest)
        (82.0, [cx - 4.5, cx + 4.5]),
        (74.0, [cx]),                        # apex (nearest)
    ]
    pts = []
    for y, xs in rows:
        for x in xs:
            pts.append([x, y])
    return np.array(pts, dtype=np.float32)


# --- Game --------------------------------------------------------------------
RESULT_STEPS: int = 26              # frozen frames after a throw lands
MAX_THROWS: int = 30                # safety cap (win = clear the rack)
MAX_FLIGHT_STEPS: int = 300         # force a miss if a flight rattles too long (~5s)

# --- Phases ------------------------------------------------------------------
PHASE_AIM: int = 0
PHASE_FLIGHT: int = 1
PHASE_RESULT: int = 2
PHASE_GAME_OVER: int = 3

# --- Rendering ---------------------------------------------------------------
OBS_W: int = 128
OBS_H: int = 96
DISPLAY_SCALE: int = 6              # 128x96 -> 768x576

COLOR_BG = (16, 20, 30)
COLOR_TABLE_NEAR = (40, 96, 70)     # felt-ish green
COLOR_TABLE_FAR = (26, 66, 50)
COLOR_TABLE_LINE = (210, 220, 210)
COLOR_CUP = (216, 64, 52)           # red cups
COLOR_CUP_DARK = (150, 38, 30)
COLOR_CUP_RIM = (240, 120, 108)
COLOR_CUP_INNER = (30, 16, 14)
COLOR_BALL = (244, 244, 236)
COLOR_BALL_SHADOW = (8, 12, 10)
COLOR_RETICLE = (250, 232, 96)
COLOR_HUD = (236, 236, 236)
