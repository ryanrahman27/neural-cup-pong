"""Synchronized trajectory recorder: accumulate rows, then save one .npz."""

from __future__ import annotations

import os

import numpy as np

from ..environment.actions import ACTION_DIM
from ..environment.state import GameState, EVENT_DIM
from . import schema


class TrajectoryRecorder:
    def __init__(self, out_dir: str, episode_id: int, seed: int, policy: str) -> None:
        self.out_dir = out_dir
        self.episode_id = int(episode_id)
        self.seed = int(seed)
        self.policy = str(policy)
        self._frames: list[np.ndarray] = []
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._events: list[np.ndarray] = []
        self._finished = False

    def record(self, frame, state: GameState, action, events) -> None:
        assert not self._finished, "record() after finish()"
        self._frames.append(np.ascontiguousarray(frame, dtype=np.uint8))
        self._states.append(state.to_vector().astype(np.float32))
        self._actions.append(np.asarray(action, dtype=np.float32).reshape(ACTION_DIM))
        self._events.append(np.asarray(events, dtype=np.int32).reshape(EVENT_DIM))

    def finish(self, frame, state: GameState) -> None:
        if self._finished:
            return
        self._frames.append(np.ascontiguousarray(frame, dtype=np.uint8))
        self._states.append(state.to_vector().astype(np.float32))
        self._actions.append(np.zeros(ACTION_DIM, dtype=np.float32))
        self._events.append(np.zeros(EVENT_DIM, dtype=np.int32))
        self._finished = True

    def __len__(self) -> int:
        return len(self._frames)

    def save(self) -> str:
        if not self._finished:
            raise RuntimeError("call finish() before save()")
        if len(self._frames) < 2:
            raise ValueError("episode too short to save")
        os.makedirs(self.out_dir, exist_ok=True)
        frames = np.stack(self._frames)
        states = np.stack(self._states)
        actions = np.stack(self._actions)
        events = np.stack(self._events)
        valid = np.ones(frames.shape[0], dtype=np.int8)
        valid[-1] = 0
        path = os.path.join(self.out_dir, schema.episode_filename(self.episode_id))
        np.savez_compressed(
            path,
            **{schema.FRAMES_KEY: frames, schema.ACTIONS_KEY: actions,
               schema.STATES_KEY: states, schema.EVENTS_KEY: events, schema.VALID_KEY: valid},
            episode_id=np.int64(self.episode_id),
            seed=np.int64(self.seed),
            policy=np.array(self.policy),
            action_dim=np.int64(ACTION_DIM),
            state_dim=np.int64(states.shape[1]),
            event_dim=np.int64(EVENT_DIM),
            obs_h=np.int64(frames.shape[1]),
            obs_w=np.int64(frames.shape[2]),
            obs_hz=np.float32(__import__("neural_cup_pong.environment.constants",
                                         fromlist=["OBS_HZ"]).OBS_HZ),
        )
        return path
