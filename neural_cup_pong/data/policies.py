"""Exploration policies for dataset collection.

The world model needs throws landing all over (makes, rim clips, misses, table
bounces, out-of-bounds), so the collector drives the thrower with a mixture:
competent scripted play, a target-explorer that samples the whole aim x power
space, pure random, button-mash, and corner aiming. Each policy is a stateful
callable built from a factory (so it can re-sample a target per throw).
"""

from __future__ import annotations

import numpy as np

from ..environment import actions as A
from ..environment import constants as C
from ..environment.bots import scripted_thrower

_AIM_EPS, _POW_EPS = 0.05, 0.04


def _toward(state, aim_t, pow_t):
    a = A.empty_action()
    if state.aim_x < aim_t - _AIM_EPS:
        a[A.AIM_RIGHT] = 1.0
    elif state.aim_x > aim_t + _AIM_EPS:
        a[A.AIM_LEFT] = 1.0
    if state.power < pow_t - _POW_EPS:
        a[A.POWER_UP] = 1.0
    elif state.power > pow_t + _POW_EPS:
        a[A.POWER_DOWN] = 1.0
    if abs(state.aim_x - aim_t) <= _AIM_EPS and abs(state.power - pow_t) <= _POW_EPS:
        a[A.THROW] = 1.0
    return a


def make_competent(skill=0.85):
    def policy(state, rng, t):
        return scripted_thrower(state, rng, skill=skill)
    return policy


def make_explore(pow_lo=0.15, pow_hi=1.0):
    st = {"last": -1, "aim": 0.0, "pow": 0.5}

    def policy(state, rng, t):
        if state.game_phase != C.PHASE_AIM:
            return A.empty_action()
        if state.throws_used != st["last"]:      # new throw -> new target
            st["last"] = state.throws_used
            st["aim"] = float(rng.uniform(-1.0, 1.0))
            st["pow"] = float(rng.uniform(pow_lo, pow_hi))
        return _toward(state, st["aim"], st["pow"])
    return policy


def make_random():
    def policy(state, rng, t):
        a = A.empty_action()
        a[A.AIM_LEFT:A.POWER_DOWN + 1] = (rng.random(4) < 0.45).astype(np.float32)
        a[A.THROW] = float(rng.random() < 0.04)
        return a
    return policy


def make_mash():
    def policy(state, rng, t):
        a = (rng.random(A.ACTION_DIM) < 0.5).astype(np.float32)
        a[A.THROW] = float(rng.random() < 0.15)
        return a
    return policy


def make_corner():
    st = {"last": -1, "side": 1.0, "pow": 0.5}

    def policy(state, rng, t):
        if state.game_phase != C.PHASE_AIM:
            return A.empty_action()
        if state.throws_used != st["last"]:
            st["last"] = state.throws_used
            st["side"] = 1.0 if rng.random() < 0.5 else -1.0
            st["pow"] = float(rng.uniform(0.2, 1.0))
        return _toward(state, st["side"] * 0.95, st["pow"])
    return policy


POLICY_REGISTRY = {
    "competent": make_competent,
    "sharp": lambda: make_competent(skill=0.97),
    "explore": make_explore,
    "random": make_random,
    "mash": make_mash,
    "corner": make_corner,
}

DEFAULT_MIXTURE = {
    "competent": 0.24, "sharp": 0.14, "explore": 0.34,
    "random": 0.12, "mash": 0.06, "corner": 0.10,
}


def sample_policy(rng: np.random.Generator, mixture: dict | None = None) -> str:
    mixture = mixture or DEFAULT_MIXTURE
    names = list(mixture)
    w = np.array([mixture[n] for n in names], dtype=np.float64)
    w /= w.sum()
    return str(rng.choice(names, p=w))
