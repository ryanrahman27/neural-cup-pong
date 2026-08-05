"""Structured state + flat serialization for the Phase 3 dynamics model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants as C


@dataclass
class GameState:
    ball_position: np.ndarray     # (3,) x, y, z
    ball_velocity: np.ndarray     # (3,)
    aim_x: float                  # [-1, 1] lateral aim
    power: float                  # [0, 1] launch power
    cups_present: np.ndarray      # (NUM_CUPS,) 1 present / 0 sunk
    score: int                    # cups sunk
    throws_used: int
    game_phase: int               # PHASE_* constant
    result_timer: int             # frames left in the frozen result phase
    flight_steps: int = 0         # physics steps in the current flight (safety cap)
    step_index: int = 0
    rng_draw: float = 0.0

    @staticmethod
    def vector_length() -> int:
        return 3 + 3 + 1 + 1 + C.NUM_CUPS + 1 + 1 + 4 + 1

    def to_vector(self) -> np.ndarray:
        phase = np.zeros(4, dtype=np.float32)
        phase[self.game_phase] = 1.0
        return np.concatenate([
            self.ball_position.reshape(-1),
            self.ball_velocity.reshape(-1),
            np.array([self.aim_x, self.power], dtype=np.float32),
            self.cups_present.astype(np.float32).reshape(-1),
            np.array([self.score, self.throws_used], dtype=np.float32),
            phase,
            np.array([self.result_timer], dtype=np.float32),
        ]).astype(np.float32)

    def copy(self) -> "GameState":
        return GameState(
            ball_position=self.ball_position.copy(),
            ball_velocity=self.ball_velocity.copy(),
            aim_x=float(self.aim_x),
            power=float(self.power),
            cups_present=self.cups_present.copy(),
            score=self.score,
            throws_used=self.throws_used,
            game_phase=self.game_phase,
            result_timer=self.result_timer,
            flight_steps=self.flight_steps,
            step_index=self.step_index,
            rng_draw=self.rng_draw,
        )


EVENT_NAMES: tuple[str, ...] = (
    "throw_released",  # 0
    "cup_sunk",        # 1
    "miss",            # 2
    "table_bounce",    # 3
    "rim_bounce",      # 4
    "rack_cleared",    # 5
    "game_over",       # 6
)
EVENT_DIM: int = len(EVENT_NAMES)


def empty_events() -> np.ndarray:
    return np.zeros(EVENT_DIM, dtype=np.int32)
