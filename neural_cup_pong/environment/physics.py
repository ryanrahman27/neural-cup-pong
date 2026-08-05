"""Aim adjustment, ballistic launch, and flight resolution (sink vs miss).

Deterministic. Flight ends at the first ground contact: the ball either drops
into a present cup (sink) or hits the table / goes out of bounds (miss).
"""

from __future__ import annotations

import numpy as np

from . import actions as A
from . import constants as C
from .state import GameState, EVENT_NAMES

_EV = {n: i for i, n in enumerate(EVENT_NAMES)}


def update_aim(state: GameState, action: np.ndarray) -> None:
    daim = (float(A.pressed(action, A.AIM_RIGHT)) - float(A.pressed(action, A.AIM_LEFT)))
    dpow = (float(A.pressed(action, A.POWER_UP)) - float(A.pressed(action, A.POWER_DOWN)))
    state.aim_x = float(np.clip(state.aim_x + daim * C.AIM_RATE * C.DT, -1.0, 1.0))
    state.power = float(np.clip(state.power + dpow * C.POWER_RATE * C.DT, 0.0, 1.0))


def launch(state: GameState) -> None:
    angle = state.aim_x * C.MAX_AIM_ANGLE
    speed = C.POWER_MIN + (C.POWER_MAX - C.POWER_MIN) * state.power
    hs = speed * float(np.cos(C.LAUNCH_ELEV))
    vz = speed * float(np.sin(C.LAUNCH_ELEV))
    state.ball_position[:] = C.THROW_ORIGIN
    state.ball_velocity[:] = [hs * float(np.sin(angle)), hs * float(np.cos(angle)), vz]
    state.game_phase = C.PHASE_FLIGHT


def integrate_flight(state: GameState, events: np.ndarray) -> None:
    bp, bv = state.ball_position, state.ball_velocity
    bv[2] -= C.GRAVITY * C.DT
    bp += bv * C.DT

    # out of bounds -> miss
    if bp[0] < 0 or bp[0] > C.TABLE_W or bp[1] < 0 or bp[1] > C.TABLE_D:
        _end_flight(state, events, sunk_cup=-1)
        return

    # descending through the cup rim -> sink test
    if bv[2] < 0 and bp[2] <= C.CUP_RIM_Z:
        cups = C.cup_layout()
        for c in range(C.NUM_CUPS):
            if state.cups_present[c] and float(np.hypot(bp[0] - cups[c, 0], bp[1] - cups[c, 1])) <= C.CUP_R:
                _end_flight(state, events, sunk_cup=c)
                return

    # hit the table -> miss
    if bp[2] <= C.BALL_R:
        bp[2] = C.BALL_R
        events[_EV["table_bounce"]] = 1
        _end_flight(state, events, sunk_cup=-1)


def _end_flight(state: GameState, events: np.ndarray, sunk_cup: int) -> None:
    if sunk_cup >= 0:
        state.cups_present[sunk_cup] = 0
        state.score += 1
        events[_EV["cup_sunk"]] = 1
        if int(state.cups_present.sum()) == 0:
            events[_EV["rack_cleared"]] = 1
    else:
        events[_EV["miss"]] = 1
    state.game_phase = C.PHASE_RESULT
    state.result_timer = C.RESULT_STEPS
    state.ball_velocity[:] = 0.0
