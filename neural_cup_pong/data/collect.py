"""Episode collection: drive the engine with exploration policies and record."""

from __future__ import annotations

import json
import os
import time

import numpy as np

from ..environment import actions as A
from ..environment.game import NeuralCupPongEnv
from ..environment.state import EVENT_NAMES
from . import policies as P
from .recorder import TrajectoryRecorder


def collect_episode(out_dir, episode_id, seed, policy_name, max_steps=2000,
                    obs_width=None, obs_height=None):
    from ..environment import constants as C

    env = NeuralCupPongEnv(obs_width=obs_width or C.OBS_W, obs_height=obs_height or C.OBS_H)
    obs, state = env.reset(seed=seed)
    policy = P.POLICY_REGISTRY[policy_name]()
    policy_rng = np.random.default_rng(seed * 2 + 1)

    rec = TrajectoryRecorder(out_dir, episode_id, seed, policy_name)
    event_totals = np.zeros(len(EVENT_NAMES), dtype=np.int64)

    terminated = False
    for t in range(max_steps):
        action = policy(state, policy_rng, t)
        next_obs, next_state, r, terminated, truncated, info = env.step(action)
        rec.record(obs, state, action, info.events)
        event_totals += info.events.astype(np.int64)
        obs, state = next_obs, next_state
        if terminated:
            break

    rec.finish(obs, state)
    path = rec.save()
    stats = {
        "episode_id": episode_id, "seed": seed, "policy": policy_name,
        "length": len(rec), "score": int(state.score), "throws": int(state.throws_used),
        "terminated": bool(terminated),
        "events": {n: int(v) for n, v in zip(EVENT_NAMES, event_totals) if v},
    }
    return path, stats


def collect_dataset(out_dir, num_episodes, base_seed=1000, max_steps=2000,
                    mixture=None, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    picker = np.random.default_rng(base_seed)
    episodes, event_grand, policy_counts = [], {}, {}
    t0 = time.perf_counter()

    for i in range(num_episodes):
        seed = base_seed + i
        policy_name = P.sample_policy(picker, mixture)
        _, stats = collect_episode(out_dir, i, seed, policy_name, max_steps)
        episodes.append(stats)
        policy_counts[policy_name] = policy_counts.get(policy_name, 0) + 1
        for k, v in stats["events"].items():
            event_grand[k] = event_grand.get(k, 0) + v
        if verbose and (i + 1) % max(1, num_episodes // 20) == 0:
            frames = sum(e["length"] for e in episodes)
            print(f"  [{i+1}/{num_episodes}] {policy_name:<10} score {stats['score']} "
                  f"throws {stats['throws']} len {stats['length']} (frames {frames})")

    total_frames = sum(e["length"] for e in episodes)
    dt = time.perf_counter() - t0
    manifest = {
        "num_episodes": num_episodes, "total_frames": total_frames,
        "base_seed": base_seed, "max_steps": max_steps,
        "policy_counts": policy_counts, "event_totals": event_grand,
        "seconds": round(dt, 2),
        "frames_per_second": round(total_frames / dt, 1) if dt else 0.0,
        "episodes": episodes,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    if verbose:
        print(f"\nCollected {num_episodes} episodes, {total_frames} frames in {dt:.1f}s")
        print(f"Policies: {policy_counts}")
        print(f"Events:   {event_grand}")
    return manifest
