"""Neural Cup Pong data package: recording, reading, validating trajectories."""

from __future__ import annotations

from .collect import collect_dataset, collect_episode
from .dataset import Episode, TrajectoryDataset, Window, load_episode
from .recorder import TrajectoryRecorder
from .validation import validate_dir, validate_episode, validate_path

__all__ = [
    "TrajectoryRecorder", "collect_episode", "collect_dataset",
    "load_episode", "Episode", "Window", "TrajectoryDataset",
    "validate_episode", "validate_path", "validate_dir",
]
