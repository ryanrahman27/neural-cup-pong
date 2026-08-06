"""Hybrid world model: neural control + exact ballistic flight.

The structured GRU tracks aim/power (its control is very accurate) and the phase
machine, but its predicted ball *lands ~8u off* — too imprecise to sink a 3.3u
cup. So the ball's flight/sink/miss use the exact, deterministic ballistics
seeded from the GRU-tracked aim/power at release. Throws land where aimed and
cups sink like the engine, while control stays neural and Phase-5 renders it.
"""

from __future__ import annotations

import numpy as np
import torch

from ..environment import actions as A
from ..environment import constants as C
from ..environment import physics as ph
from ..environment import rules
from ..environment.state import empty_events, from_vector


class HybridWorldModel:
    def __init__(self, gru, device):
        self.gru = gru
        self.dev = device
        self.state = None
        self.h = None

    def reset(self, state_vec):
        self.state = np.asarray(state_vec, dtype=np.float32).copy()
        self.h = None
        return self.state

    @torch.no_grad()
    def step(self, action):
        gs = from_vector(self.state)
        phase = gs.game_phase
        events = empty_events()

        # advance the GRU (keeps its hidden state synced to the true trajectory)
        # and read its aim/power control prediction
        gnext_t, self.h = self.gru.predict_step(
            torch.tensor(self.state, dtype=torch.float32, device=self.dev)[None],
            torch.tensor(action, dtype=torch.float32, device=self.dev)[None], self.h)
        gnext = from_vector(gnext_t[0].cpu().numpy())

        if phase == C.PHASE_AIM:
            gs.aim_x = float(np.clip(gnext.aim_x, -1.0, 1.0))   # neural control
            gs.power = float(np.clip(gnext.power, 0.0, 1.0))
            gs.ball_position[:] = C.THROW_ORIGIN
            gs.ball_velocity[:] = 0.0
            if A.pressed(action, A.THROW):
                gs.throws_used += 1
                events[0] = 1
                ph.launch(gs)                                   # exact ballistic launch
        elif phase == C.PHASE_FLIGHT:
            ph.integrate_flight(gs, events)                     # exact arc + sink/miss
        elif phase == C.PHASE_RESULT:
            rules.advance_result(gs, events)
        # GAME_OVER: frozen

        self.state = gs.to_vector()
        return self.state, events
