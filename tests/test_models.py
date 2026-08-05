import numpy as np
import torch

from neural_cup_pong.data import collect_dataset
from neural_cup_pong.data.dataset import TrajectoryDataset
from neural_cup_pong.environment import constants as C
from neural_cup_pong.models import layout as L
from neural_cup_pong.models import projection
from neural_cup_pong.models.dynamics_gru import build_model
from neural_cup_pong.models.normalizer import fit_normalizer


def _small_dataset(tmp_path, n=3, max_steps=300):
    collect_dataset(str(tmp_path), num_episodes=n, base_seed=1, max_steps=max_steps, verbose=False)
    return TrajectoryDataset(str(tmp_path), window=8, with_frames=False, preload=True)


def test_model_builds_small_and_forward_shapes(tmp_path):
    ds = _small_dataset(tmp_path)
    model = build_model(fit_normalizer(ds), hidden=192)
    assert model.param_count() < 500_000
    S = torch.zeros(4, 8, L.STATE_DIM)
    A = torch.zeros(4, 8, L.ACTION_DIM)
    heads, h = model(S, A)
    assert heads["cont"].shape == (4, 8, L.CONT_DIM)
    assert heads["cups"].shape == (4, 8, 6)
    assert heads["phase"].shape == (4, 8, 4)


def test_normalizer_cont_roundtrip(tmp_path):
    ds = _small_dataset(tmp_path)
    norm = fit_normalizer(ds)
    ep = next(ds.iter_episodes())
    cur = torch.tensor(ep.states[:-1]); nxt = torch.tensor(ep.states[1:])
    tgt = norm.norm_cont_target(cur, nxt)
    recon = norm.apply_cont(cur, tgt)
    # continuous fields reconstruct (cups/phase come from other heads)
    assert torch.allclose(recon[:, L.POS], nxt[:, L.POS], atol=1e-3)
    assert torch.allclose(recon[:, L.VEL], nxt[:, L.VEL], atol=1e-2)
    assert torch.allclose(recon[:, L.AIM:L.POWER + 1], nxt[:, L.AIM:L.POWER + 1], atol=1e-3)


def test_snap_enforces_legal_state():
    B = 32
    prev = torch.zeros(B, L.STATE_DIM)
    prev[:, L.CUPS] = 1.0
    prev[:, 16] = 1.0                      # phase = AIM
    cont_next = torch.randn(B, L.STATE_DIM) * 5
    cups_logits = torch.randn(B, 6)
    phase_logits = torch.randn(B, 4)
    nxt = projection.snap_batch(prev, cont_next, cups_logits, phase_logits)
    # cups monotone non-increasing, score derived, phase one-hot & legal
    assert (nxt[:, L.CUPS] <= prev[:, L.CUPS] + 1e-6).all()
    assert torch.allclose(nxt[:, L.SCORE], C.NUM_CUPS - nxt[:, L.CUPS].sum(-1))
    assert torch.allclose(nxt[:, L.PHASE].sum(-1), torch.ones(B))
    nph = nxt[:, L.PHASE].argmax(-1)
    assert ((nph == C.PHASE_AIM) | (nph == C.PHASE_FLIGHT)).all()   # legal from AIM
    assert (nxt[:, L.AIM] <= 1).all() and (nxt[:, L.AIM] >= -1).all()
    assert (nxt[:, L.POWER] >= 0).all() and (nxt[:, L.POWER] <= 1).all()


def test_rollout_stays_legal(tmp_path):
    ds = _small_dataset(tmp_path)
    model = build_model(fit_normalizer(ds), hidden=192)
    ep = next(ds.iter_episodes())
    warm = torch.tensor(ep.states[:4])[None]
    warm_a = torch.tensor(ep.actions[:4])[None]
    roll_a = torch.tensor(ep.actions[4:40])[None]
    preds = model.rollout(warm, warm_a, roll_a)[0].numpy()
    # even an untrained model must emit LEGAL states (snap guarantees it)
    cups = preds[:, L.CUPS]
    assert (np.diff(cups, axis=0) <= 1e-6).all()          # no cup reappears
    scores = C.NUM_CUPS - cups.sum(1)
    assert (np.diff(scores) >= -1e-6).all()               # score non-decreasing


def test_train_smoke(tmp_path):
    from neural_cup_pong.training.train import TrainConfig, main
    collect_dataset(str(tmp_path / "d"), num_episodes=3, base_seed=1, max_steps=300, verbose=False)
    cfg = TrainConfig(data_dir=str(tmp_path / "d"), ckpt=str(tmp_path / "ck"),
                      window=8, batch=8, tf_epochs=1, ss_epochs=1, ss_horizon=4,
                      steps_per_epoch=3)
    model = main(cfg)
    assert (tmp_path / "ck.pt").exists() and (tmp_path / "ck.norm.npz").exists()
