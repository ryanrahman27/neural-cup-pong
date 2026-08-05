"""On-disk trajectory schema (one compressed .npz per episode).

Aligned arrays of length T (one row per ~20 Hz observation tick):
    frames  : uint8   [T, H, W, 3]        rendered RGB observation at tick t
    actions : float32 [T, ACTION_DIM]     action applied at tick t -> t+1
    states  : float32 [T, STATE_DIM]      GameState vector at tick t
    events  : int32   [T, EVENT_DIM]      events during t -> t+1
    valid   : int8    [T]                 1 real transition, 0 padded terminal row

Contract: (states[t], actions[t]) -> states[t+1]. Plus scalar metadata
(episode_id, seed, policy, dims, obs_hz).
"""

from __future__ import annotations

from dataclasses import dataclass

FRAMES_KEY = "frames"
ACTIONS_KEY = "actions"
STATES_KEY = "states"
EVENTS_KEY = "events"
VALID_KEY = "valid"
REQUIRED_ARRAYS = (FRAMES_KEY, ACTIONS_KEY, STATES_KEY, EVENTS_KEY, VALID_KEY)
EPISODE_GLOB = "episode_*.npz"


@dataclass(frozen=True)
class TrajectoryMeta:
    episode_id: int
    seed: int
    policy: str
    action_dim: int
    state_dim: int
    event_dim: int
    obs_h: int
    obs_w: int
    obs_hz: float
    length: int


def episode_filename(episode_id: int) -> str:
    return f"episode_{episode_id:06d}.npz"
