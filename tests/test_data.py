import numpy as np

from neural_cup_pong.data import (
    TrajectoryDataset, collect_dataset, collect_episode, load_episode,
    validate_dir, validate_episode,
)
from neural_cup_pong.environment import constants as C
from neural_cup_pong.environment.actions import ACTION_DIM
from neural_cup_pong.environment.rules import build_initial
from neural_cup_pong.environment.state import GameState, decode_vector


def test_decode_vector_roundtrips():
    st = build_initial()
    st.cups_present[0] = 0
    st.score = 1
    st.throws_used = 3
    d = decode_vector(st.to_vector())
    assert d["cups_left"] == C.NUM_CUPS - 1
    assert d["score"] == 1 and d["throws_used"] == 3
    assert d["game_phase"] == C.PHASE_AIM


def test_collect_episode_roundtrip(tmp_path):
    path, stats = collect_episode(str(tmp_path), 0, 0, "explore", max_steps=400)
    ep = load_episode(path, with_frames=True)
    T = ep.states.shape[0]
    assert ep.frames.shape == (T, C.OBS_H, C.OBS_W, 3)
    assert ep.actions.shape == (T, ACTION_DIM)
    assert ep.states.shape == (T, GameState.vector_length())
    assert ep.valid[-1] == 0 and ep.valid[:-1].all()
    assert stats["length"] == T


def test_collected_episodes_validate_clean(tmp_path):
    for pol in ["competent", "explore", "random", "mash", "corner"]:
        path, _ = collect_episode(str(tmp_path), 0, 7, pol, max_steps=500)
        problems = validate_episode(load_episode(path, with_frames=True))
        assert problems == [], f"{pol}: {problems}"


def test_dataset_window_shapes(tmp_path):
    collect_dataset(str(tmp_path), num_episodes=3, base_seed=10, max_steps=400, verbose=False)
    ds = TrajectoryDataset(str(tmp_path), window=8, with_frames=True)
    assert len(ds) > 0
    w = ds[0]
    assert w.states.shape == (9, GameState.vector_length())
    assert w.actions.shape == (8, ACTION_DIM)
    assert w.frames.shape == (9, C.OBS_H, C.OBS_W, 3)


def test_manifest_and_validation(tmp_path):
    m = collect_dataset(str(tmp_path), num_episodes=5, base_seed=20, max_steps=500, verbose=False)
    assert m["num_episodes"] == 5 and m["total_frames"] > 0
    assert sum(m["policy_counts"].values()) == 5
    report = validate_dir(str(tmp_path), verbose=False)
    assert report["problem_episodes"] == {} and report["clean"] == 5


def test_collection_reproducible(tmp_path):
    p1, _ = collect_episode(str(tmp_path / "a"), 0, 42, "explore", max_steps=400)
    p2, _ = collect_episode(str(tmp_path / "b"), 0, 42, "explore", max_steps=400)
    e1, e2 = load_episode(p1), load_episode(p2)
    assert np.array_equal(e1.states, e2.states)
    assert np.array_equal(e1.actions, e2.actions)
