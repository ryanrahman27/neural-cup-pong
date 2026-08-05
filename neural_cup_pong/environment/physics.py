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
    state.flight_steps = 0
    state.game_phase = C.PHASE_FLIGHT


def integrate_flight(state: GameState, events: np.ndarray) -> None:
    state.flight_steps += 1
    if state.flight_steps > C.MAX_FLIGHT_STEPS:   # rattled too long -> call it a miss
        _end_flight(state, events, sunk_cup=-1)
        return
    bp, bv = state.ball_position, state.ball_velocity
    bv[2] -= C.GRAVITY * C.DT
    bp += bv * C.DT

    # out of bounds -> miss
    if bp[0] < 0 or bp[0] > C.TABLE_W or bp[1] < 0 or bp[1] > C.TABLE_D:
        _end_flight(state, events, sunk_cup=-1)
        return

    # descending near rim height -> clean make / rim clip / past
    if bv[2] < 0 and bp[2] <= C.CUP_RIM_Z:
        cups = C.cup_layout()
        for c in range(C.NUM_CUPS):
            if not state.cups_present[c]:
                continue
            dx = float(bp[0] - cups[c, 0])
            dy = float(bp[1] - cups[c, 1])
            d = float(np.hypot(dx, dy))
            if d <= C.CUP_R - C.BALL_R:          # fits through the hole -> make
                _end_flight(state, events, sunk_cup=c)
                return
            if d <= C.CUP_R + C.BALL_R:          # caught the rim -> bounce out, stay live
                _rim_bounce(state, events, float(cups[c, 0]), float(cups[c, 1]), dx, dy, d)
                return

    # hit the table -> miss
    if bp[2] <= C.BALL_R:
        bp[2] = C.BALL_R
        events[_EV["table_bounce"]] = 1
        _end_flight(state, events, sunk_cup=-1)


def _rim_bounce(state, events, cx, cy, dx, dy, d) -> None:
    """Deflect the ball off a cup rim: pop up + kick outward, keep it in flight."""
    bp, bv = state.ball_position, state.ball_velocity
    nx, ny = (1.0, 0.0) if d < 1e-4 else (dx / d, dy / d)
    # bounce up
    bv[2] = abs(bv[2]) * C.RIM_RESTITUTION + C.RIM_POP
    # reflect any inward horizontal motion, damp, then add an outward kick
    v_in = bv[0] * nx + bv[1] * ny
    if v_in < 0:
        bv[0] -= (1.0 + C.RIM_RESTITUTION) * v_in * nx
        bv[1] -= (1.0 + C.RIM_RESTITUTION) * v_in * ny
    bv[0] = bv[0] * C.RIM_HDAMP + nx * C.RIM_KICK
    bv[1] = bv[1] * C.RIM_HDAMP + ny * C.RIM_KICK
    # seat the ball just outside the rim, above rim height, so it can't re-trigger
    bp[0] = cx + nx * (C.CUP_R + C.BALL_R + 0.1)
    bp[1] = cy + ny * (C.CUP_R + C.BALL_R + 0.1)
    bp[2] = C.CUP_RIM_Z + 0.2
    events[_EV["rim_bounce"]] = 1


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
