"""Phase machine: AIM -> FLIGHT -> RESULT -> (AIM | GAME_OVER)."""

from __future__ import annotations

import numpy as np

from . import actions as A
from . import constants as C
from . import physics
from .state import GameState, EVENT_NAMES

_EV = {n: i for i, n in enumerate(EVENT_NAMES)}


def build_initial() -> GameState:
    return GameState(
        ball_position=C.THROW_ORIGIN.copy(),
        ball_velocity=np.zeros(3, dtype=np.float32),
        aim_x=0.0,
        power=0.5,
        cups_present=np.ones(C.NUM_CUPS, dtype=np.int32),
        score=0,
        throws_used=0,
        game_phase=C.PHASE_AIM,
        result_timer=0,
    )


def handle_aim(state: GameState, action: np.ndarray, events: np.ndarray) -> None:
    if state.game_phase != C.PHASE_AIM:
        return
    physics.update_aim(state, action)
    if A.pressed(action, A.THROW):
        state.throws_used += 1
        events[_EV["throw_released"]] = 1
        physics.launch(state)


def advance_result(state: GameState, events: np.ndarray) -> None:
    if state.game_phase != C.PHASE_RESULT:
        return
    state.result_timer -= 1
    if state.result_timer > 0:
        return
    cleared = int(state.cups_present.sum()) == 0
    if cleared or state.throws_used >= C.MAX_THROWS:
        state.game_phase = C.PHASE_GAME_OVER
        events[_EV["game_over"]] = 1
    else:
        state.ball_position[:] = C.THROW_ORIGIN
        state.ball_velocity[:] = 0.0
        state.game_phase = C.PHASE_AIM
