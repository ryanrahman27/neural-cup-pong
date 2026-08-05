"""Trajectory validation: shape/dtype/finite/valid-mask + game-logic invariants.

`validate_episode` returns a list of problems (empty = clean). Key world-model
invariants: score never decreases, cups never reappear, score matches
cups-sunk, and a cup only disappears on a tick where `cup_sunk` fired.
"""

from __future__ import annotations

import glob
import os

import numpy as np

from ..environment import constants as C
from ..environment.state import EVENT_NAMES, decode_vector
from . import schema
from .dataset import Episode, load_episode

_CUP_SUNK = EVENT_NAMES.index("cup_sunk")


def validate_episode(ep: Episode) -> list[str]:
    problems: list[str] = []
    T = ep.states.shape[0]

    if ep.actions.shape[0] != T or ep.events.shape[0] != T or ep.valid.shape[0] != T:
        problems.append("array length mismatch")
        return problems
    if ep.frames.size and ep.frames.shape[0] != T:
        problems.append(f"frames length {ep.frames.shape[0]} != states {T}")
    if ep.states.shape[1] != ep.meta.state_dim:
        problems.append("state_dim mismatch")
    if ep.states.dtype != np.float32:
        problems.append(f"states dtype {ep.states.dtype} != float32")
    if ep.frames.size and ep.frames.dtype != np.uint8:
        problems.append("frames dtype != uint8")
    if not np.isfinite(ep.states).all():
        problems.append("non-finite states")
    if not np.isfinite(ep.actions).all():
        problems.append("non-finite actions")
    if ep.actions.size and (ep.actions.min() < -1e-6 or ep.actions.max() > 1 + 1e-6):
        problems.append("actions outside [0,1]")
    if ep.valid[-1] != 0:
        problems.append("valid[-1] should be 0")
    if T >= 2 and not ep.valid[:-1].all():
        problems.append("valid mask has holes")

    decoded = [decode_vector(ep.states[t]) for t in range(T)]
    score = np.array([d["score"] for d in decoded])
    cups = np.stack([d["cups_present"] for d in decoded])  # [T, NUM_CUPS]

    if (np.diff(score) < 0).any():
        problems.append("score decreased")
    if (np.diff(cups, axis=0) > 0).any():
        problems.append("a sunk cup reappeared")
    # score should equal cups removed
    if not np.all(score == (C.NUM_CUPS - cups.sum(axis=1)).round()):
        problems.append("score != cups-sunk count")
    # a cup only disappears on a cup_sunk tick
    for t in range(T - 1):
        if (cups[t + 1] < cups[t]).any() and not ep.events[t, _CUP_SUNK]:
            problems.append(f"cup vanished at tick {t} without a cup_sunk event")
            break

    return problems


def validate_path(path: str) -> list[str]:
    return validate_episode(load_episode(path, with_frames=True))


def validate_dir(data_dir: str, verbose: bool = True) -> dict:
    paths = sorted(glob.glob(os.path.join(data_dir, schema.EPISODE_GLOB)))
    report = {"num_episodes": len(paths), "clean": 0, "problem_episodes": {}}
    for p in paths:
        probs = validate_path(p)
        if probs:
            report["problem_episodes"][os.path.basename(p)] = probs
        else:
            report["clean"] += 1
    if verbose:
        print(f"Validated {len(paths)} episodes: {report['clean']} clean, "
              f"{len(report['problem_episodes'])} with problems")
        for name, probs in list(report["problem_episodes"].items())[:10]:
            print(f"  {name}: {probs}")
    return report
