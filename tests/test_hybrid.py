"""Hybrid world model: the ball flight uses exact ballistics, so a throw aimed
at a cup sinks it exactly like the engine (unlike the GRU's own ~8u-off flight).
This path bypasses the GRU during FLIGHT, so it holds for an untrained model."""

import numpy as np
import torch

from neural_cup_pong.data import collect_dataset
from neural_cup_pong.data.dataset import TrajectoryDataset
from neural_cup_pong.environment import bots, constants as C, physics as ph
from neural_cup_pong.environment import actions as A
from neural_cup_pong.environment.game import NeuralCupPongEnv
from neural_cup_pong.environment.state import GameState, empty_events
from neural_cup_pong.models.dynamics_gru import build_model
from neural_cup_pong.models.hybrid import HybridWorldModel
from neural_cup_pong.models.normalizer import fit_normalizer


def _model(tmp_path):
    collect_dataset(str(tmp_path), num_episodes=2, base_seed=1, max_steps=200, verbose=False)
    ds = TrajectoryDataset(str(tmp_path), window=8, with_frames=False, preload=True)
    return build_model(fit_normalizer(ds), hidden=64)


def _flight_state_aimed_at(cup_idx):
    """Build a just-launched FLIGHT state aimed at a specific cup (via the
    same landing grid the scripted bot uses)."""
    target = C.cup_layout()[cup_idx]
    aims, pows, land = bots._grid()
    d = np.hypot(land[..., 0] - target[0], land[..., 1] - target[1])
    i, j = np.unravel_index(int(d.argmin()), d.shape)
    gs = GameState(
        ball_position=np.array(C.THROW_ORIGIN, dtype=np.float32),
        ball_velocity=np.zeros(3, dtype=np.float32),
        aim_x=float(aims[i]), power=float(pows[j]),
        cups_present=np.ones(C.NUM_CUPS, dtype=np.float32),
        score=0, throws_used=1, game_phase=C.PHASE_FLIGHT, result_timer=0)
    ph.launch(gs)                      # seed ballistic velocity from aim/power
    return gs


def test_hybrid_flight_sinks_like_engine(tmp_path):
    model = _model(tmp_path)
    hybrid = HybridWorldModel(model, "cpu")

    for cup_idx in range(C.NUM_CUPS):
        gs = _flight_state_aimed_at(cup_idx)

        # engine reference: integrate the same launched ball to completion
        eng = gs.copy()
        for _ in range(C.MAX_FLIGHT_STEPS + 5):
            if eng.game_phase != C.PHASE_FLIGHT:
                break
            ph.integrate_flight(eng, empty_events())

        # hybrid: step the flight with empty actions (GRU untouched during FLIGHT)
        hybrid.reset(gs.to_vector())
        noop = A.empty_action()
        from neural_cup_pong.environment.state import from_vector
        for _ in range(C.MAX_FLIGHT_STEPS + 5):
            s, _ = hybrid.step(noop)
            if from_vector(s).game_phase != C.PHASE_FLIGHT:
                break

        hyb_cups = from_vector(hybrid.state).cups_present.sum()
        assert hyb_cups == eng.cups_present.sum(), \
            f"cup {cup_idx}: hybrid left {hyb_cups}, engine left {eng.cups_present.sum()}"
        # a well-aimed throw should actually remove a cup
        assert hyb_cups == C.NUM_CUPS - 1, f"cup {cup_idx}: expected a sink, got {hyb_cups} left"
