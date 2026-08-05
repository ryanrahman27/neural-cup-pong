import numpy as np

from neural_cup_pong.environment import actions as A
from neural_cup_pong.environment import constants as C
from neural_cup_pong.environment import physics, rules
from neural_cup_pong.environment.rules import build_initial
from neural_cup_pong.environment.state import GameState, empty_events, EVENT_NAMES, EVENT_DIM

_EV = {n: i for i, n in enumerate(EVENT_NAMES)}


def test_state_vector_and_copy():
    st = build_initial()
    v = st.to_vector()
    assert v.shape == (GameState.vector_length(),) and v.dtype == np.float32
    clone = st.copy()
    clone.cups_present[0] = 0
    clone.score = 5
    assert st.cups_present[0] == 1 and st.score == 0


def test_events_shape():
    ev = empty_events()
    assert ev.shape == (EVENT_DIM,) and not ev.any()


def test_aim_updates_clamp():
    st = build_initial()
    for _ in range(200):
        physics.update_aim(st, A.make_action(aim_right=True, power_up=True))
    assert st.aim_x <= 1.0 + 1e-6 and st.power <= 1.0 + 1e-6
    assert st.aim_x > 0.9 and st.power > 0.9


def test_throw_launches():
    st = build_initial()
    ev = empty_events()
    rules.handle_aim(st, A.make_action(throw=True), ev)
    assert st.game_phase == C.PHASE_FLIGHT
    assert st.ball_velocity[2] > 0 and st.throws_used == 1
    assert ev[_EV["throw_released"]] == 1


def test_sink_removes_cup_and_scores():
    st = build_initial()
    st.game_phase = C.PHASE_FLIGHT
    cups = C.cup_layout()
    st.ball_position[:] = [cups[0, 0], cups[0, 1], C.CUP_RIM_Z + 0.4]
    st.ball_velocity[:] = [0.0, 0.0, -30.0]
    ev = empty_events()
    physics.integrate_flight(st, ev)
    assert st.cups_present[0] == 0 and st.score == 1
    assert ev[_EV["cup_sunk"]] == 1 and st.game_phase == C.PHASE_RESULT


def test_miss_on_table():
    st = build_initial()
    st.game_phase = C.PHASE_FLIGHT
    st.ball_position[:] = [5.0, 20.0, C.BALL_R + 0.2]   # nowhere near a cup
    st.ball_velocity[:] = [0.0, 0.0, -30.0]
    ev = empty_events()
    physics.integrate_flight(st, ev)
    assert ev[_EV["miss"]] == 1 and st.game_phase == C.PHASE_RESULT
    assert st.score == 0


def test_rack_cleared_game_over():
    st = build_initial()
    st.cups_present[:] = 0
    st.game_phase = C.PHASE_RESULT
    st.result_timer = 1
    ev = empty_events()
    rules.advance_result(st, ev)
    assert st.game_phase == C.PHASE_GAME_OVER and ev[_EV["game_over"]] == 1


def test_result_returns_to_aim():
    st = build_initial()
    st.game_phase = C.PHASE_RESULT
    st.result_timer = 1
    ev = empty_events()
    rules.advance_result(st, ev)
    assert st.game_phase == C.PHASE_AIM
    assert np.allclose(st.ball_position, C.THROW_ORIGIN)
