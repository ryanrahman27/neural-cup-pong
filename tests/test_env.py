import numpy as np

from neural_cup_pong.environment import NeuralCupPongEnv, actions as A
from neural_cup_pong.environment import constants as C
from neural_cup_pong.environment.bots import scripted_thrower


def test_reset_and_step_shapes():
    env = NeuralCupPongEnv()
    obs, st = env.reset(seed=0)
    assert obs.shape == (C.OBS_H, C.OBS_W, 3) and obs.dtype == np.uint8
    obs, st, r, term, trunc, info = env.step(A.empty_action())
    assert isinstance(r, float) and isinstance(term, bool)
    assert info.events.shape[0] > 0


def test_aim_controllability():
    env = NeuralCupPongEnv()
    _, s0 = env.reset(seed=1)
    a0 = float(s0.aim_x)
    st = s0
    for _ in range(6):
        _, st, *_ = env.step(A.make_action(aim_right=True))
    assert float(st.aim_x) > a0 + 0.05


def test_same_seed_identical():
    def run(seed):
        env = NeuralCupPongEnv(); env.reset(seed=seed)
        vecs = []
        for _ in range(120):
            act = scripted_thrower(env.state, env.rng)
            _, st, *_ = env.step(act)
            vecs.append(st.to_vector())
        return np.stack(vecs)
    assert np.array_equal(run(2), run(2))


def test_scripted_full_game_terminates():
    env = NeuralCupPongEnv(); env.reset(seed=4)
    term = False
    st = None
    for _ in range(6000):
        act = scripted_thrower(env.state, env.rng)
        _, st, r, term, trunc, info = env.step(act)
        if term:
            break
    assert term
    assert st.score == C.NUM_CUPS or st.throws_used >= C.MAX_THROWS


def test_render_deterministic():
    env = NeuralCupPongEnv()
    env.reset(seed=7); f1 = env.render()
    env.reset(seed=7); f2 = env.render()
    assert np.array_equal(f1, f2)
