"""``NeuralCupPongEnv`` — deterministic fixed-camera 2.5D cup-pong.

    obs, state = env.reset(seed=0)
    obs, state, reward, terminated, truncated, info = env.step(action)   # action: (5,)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import actions as A
from . import constants as C
from . import physics
from . import rules
from .renderer import Renderer
from .state import GameState, empty_events


@dataclass
class StepInfo:
    events: np.ndarray
    score: int
    throws_used: int
    game_phase: int


class NeuralCupPongEnv:
    def __init__(self, obs_width: int = C.OBS_W, obs_height: int = C.OBS_H,
                 sim_steps_per_obs: int = C.SIM_STEPS_PER_OBS) -> None:
        self.sim_steps_per_obs = sim_steps_per_obs
        self._renderer = Renderer(obs_width, obs_height)
        self.state: GameState | None = None
        self._rng = np.random.default_rng(0)
        self._seed = 0

    def reset(self, seed: int = 0):
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self.state = rules.build_initial()
        obs = self._renderer.render(self.state)
        return obs, self.state.copy()

    def step(self, action: np.ndarray):
        assert self.state is not None, "call reset() first"
        action = np.asarray(action, dtype=np.float32).reshape(A.ACTION_DIM)
        events = empty_events()
        prev_score = self.state.score

        for _ in range(self.sim_steps_per_obs):
            st = self.state
            if st.game_phase == C.PHASE_AIM:
                rules.handle_aim(st, action, events)
            elif st.game_phase == C.PHASE_FLIGHT:
                physics.integrate_flight(st, events)
            elif st.game_phase == C.PHASE_RESULT:
                rules.advance_result(st, events)
            else:  # GAME_OVER
                break

        self.state.step_index += 1
        obs = self._renderer.render(self.state)
        reward = float(self.state.score - prev_score)
        terminated = self.state.game_phase == C.PHASE_GAME_OVER
        info = StepInfo(events, self.state.score, self.state.throws_used, self.state.game_phase)
        return obs, self.state.copy(), reward, terminated, False, info

    def render(self) -> np.ndarray:
        assert self.state is not None
        return self._renderer.render(self.state)

    @property
    def rng(self):
        return self._rng
