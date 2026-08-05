"""Read recorded trajectories and sample fixed-length training windows."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

from . import schema
from .schema import TrajectoryMeta


@dataclass
class Episode:
    frames: np.ndarray
    actions: np.ndarray
    states: np.ndarray
    events: np.ndarray
    valid: np.ndarray
    meta: TrajectoryMeta


def load_episode(path: str, with_frames: bool = True) -> Episode:
    with np.load(path, allow_pickle=False) as z:
        states = z[schema.STATES_KEY]
        meta = TrajectoryMeta(
            episode_id=int(z["episode_id"]), seed=int(z["seed"]), policy=str(z["policy"]),
            action_dim=int(z["action_dim"]), state_dim=int(z["state_dim"]),
            event_dim=int(z["event_dim"]), obs_h=int(z["obs_h"]), obs_w=int(z["obs_w"]),
            obs_hz=float(z["obs_hz"]), length=int(states.shape[0]),
        )
        return Episode(
            frames=z[schema.FRAMES_KEY] if with_frames else np.empty((0,), np.uint8),
            actions=z[schema.ACTIONS_KEY], states=states,
            events=z[schema.EVENTS_KEY], valid=z[schema.VALID_KEY], meta=meta,
        )


@dataclass
class Window:
    states: np.ndarray   # [L+1, STATE_DIM]
    actions: np.ndarray  # [L, ACTION_DIM]
    events: np.ndarray   # [L, EVENT_DIM]
    frames: np.ndarray   # [L+1, H, W, 3] or empty


class TrajectoryDataset:
    def __init__(self, data_dir, window=8, with_frames=False, preload=False):
        self.data_dir = data_dir
        self.window = int(window)
        self.with_frames = with_frames
        self.paths = sorted(glob.glob(os.path.join(data_dir, schema.EPISODE_GLOB)))
        if not self.paths:
            raise FileNotFoundError(f"no episodes in {data_dir}")
        self._cache: dict[int, Episode] = {}
        self._preload = preload
        if preload:
            for i in range(len(self.paths)):
                self._cache[i] = load_episode(self.paths[i], with_frames)
        self._index: list[tuple[int, int]] = []
        for ei, path in enumerate(self.paths):
            ep = self._cache.get(ei) or load_episode(path, with_frames=False)
            T, L = ep.states.shape[0], self.window
            for s in range(0, T - L):
                if ep.valid[s:s + L].all():
                    self._index.append((ei, s))
            if not preload:
                self._cache.pop(ei, None)

    def __len__(self):
        return len(self._index)

    def _episode(self, ei):
        ep = self._cache.get(ei)
        if ep is None:
            ep = load_episode(self.paths[ei], self.with_frames)
            if self._preload:
                self._cache[ei] = ep
            else:
                self._cache.clear()
                self._cache[ei] = ep
        return ep

    def __getitem__(self, i):
        ei, s = self._index[i]
        ep = self._episode(ei)
        L = self.window
        frames = ep.frames[s:s + L + 1] if (self.with_frames and ep.frames.size) else np.empty((0,), np.uint8)
        return Window(states=ep.states[s:s + L + 1], actions=ep.actions[s:s + L],
                      events=ep.events[s:s + L], frames=frames)

    def iter_episodes(self):
        for ei in range(len(self.paths)):
            yield self._episode(ei)

    def torch_dataset(self):
        import torch
        from torch.utils.data import Dataset
        outer = self

        class _TW(Dataset):
            def __len__(self):
                return len(outer)

            def __getitem__(self, i):
                w = outer[i]
                item = {"states": torch.from_numpy(w.states.copy()),
                        "actions": torch.from_numpy(w.actions.copy()),
                        "events": torch.from_numpy(w.events.copy()).float()}
                if w.frames.size:
                    item["frames"] = torch.from_numpy(w.frames.copy())
                return item

        return _TW()
