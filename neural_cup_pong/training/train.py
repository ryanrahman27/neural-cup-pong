"""Train the structured-dynamics GRU (Phase 3).

Curriculum: teacher-forced multi-step (fast, nails local dynamics) then
scheduled sampling that feeds the model its own SNAP-projected predictions
(fights exposure bias for stable autoregressive rollout). Trains on GPU if
available. Saves ``<ckpt>.pt`` + ``<ckpt>.norm.npz`` + ``<ckpt>.json``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch

from ..data.dataset import TrajectoryDataset
from ..models import layout as L
from ..models import projection
from ..models.dynamics_gru import build_model
from ..models.normalizer import fit_normalizer
from .losses import DynamicsLoss


@dataclass
class TrainConfig:
    data_dir: str = "data/cup_v1"
    ckpt: str = "checkpoints/phase3_gru"
    window: int = 40           # must be >= ss_burn + ss_horizon
    hidden: int = 192
    batch: int = 256
    lr: float = 1e-3
    tf_epochs: int = 8
    ss_epochs: int = 6
    ss_burn: int = 4
    ss_horizon: int = 24
    ss_tf_start: float = 1.0
    ss_tf_end: float = 0.4     # gentle free-run; harder schedules destabilized
    steps_per_epoch: int = 200
    seed: int = 0


def _load_tensors(ds: TrajectoryDataset, device):
    eps = []
    for ep in ds.iter_episodes():
        T = ep.states.shape[0]
        eps.append({
            "states": torch.tensor(ep.states, dtype=torch.float32, device=device),
            "actions": torch.tensor(ep.actions, dtype=torch.float32, device=device),
            "events": torch.tensor(ep.events, dtype=torch.float32, device=device),
            "max_start": T - 1 - ds.window,
        })
    return [e for e in eps if e["max_start"] >= 0]


def _sample_batch(eps, window, batch, rng, event_pool=None, event_frac=0.0):
    B = batch
    S = torch.empty(B, window + 1, L.STATE_DIM, device=eps[0]["states"].device)
    A = torch.empty(B, window, L.ACTION_DIM, device=S.device)
    E = torch.empty(B, window, L.EVENT_DIM, device=S.device)
    for b in range(B):
        if event_pool and rng.random() < event_frac:      # oversample sink windows
            ei_b, s = event_pool[int(rng.integers(0, len(event_pool)))]
        else:
            ei_b = int(rng.integers(0, len(eps)))
            s = int(rng.integers(0, eps[ei_b]["max_start"] + 1))
        e = eps[ei_b]
        S[b] = e["states"][s:s + window + 1]
        A[b] = e["actions"][s:s + window]
        E[b] = e["events"][s:s + window]
    return S, A, E


def main(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = TrajectoryDataset(cfg.data_dir, window=cfg.window, with_frames=False, preload=True)
    norm = fit_normalizer(ds)
    model = build_model(norm, hidden=cfg.hidden).to(device)
    loss_fn = DynamicsLoss(norm).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    eps = _load_tensors(ds, device)
    # pool of window starts straddling a cup_sunk tick (rare -> oversample so the
    # model actually learns to fire sinks instead of collapsing to "never change")
    event_pool = []
    for ei, e in enumerate(eps):
        for t in torch.where(e["events"][:, 1] > 0)[0].tolist():
            event_pool.append((ei, int(np.clip(t - cfg.window // 2, 0, e["max_start"]))))
    print(f"device={device}  params={model.param_count()}  episodes={len(eps)}  "
          f"windows~{sum(e['max_start']+1 for e in eps)}  sink_windows={len(event_pool)}")

    def teacher_forced_epoch():
        model.train(); tot = 0.0
        for _ in range(cfg.steps_per_epoch):
            S, A, E = _sample_batch(eps, cfg.window, cfg.batch, rng, event_pool, 0.5)
            heads, _ = model(S[:, :cfg.window], A)
            loss, _ = loss_fn(heads, S[:, :cfg.window], S[:, 1:], E)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tot += float(loss)
        return tot / cfg.steps_per_epoch

    def scheduled_epoch(tf_prob):
        model.train(); tot = 0.0
        W, burn = cfg.window, cfg.ss_burn
        H = min(cfg.ss_horizon, W - burn)          # must fit inside the window
        for _ in range(cfg.steps_per_epoch):
            S, A, E = _sample_batch(eps, W, cfg.batch, rng, event_pool, 0.5)
            _, h = model(S[:, :burn], A[:, :burn])         # warm hidden state (GT)
            cur = S[:, burn]
            losses = []
            for t in range(burn, burn + H):
                heads, h = model.forward_step(cur, A[:, t], h)
                loss, _ = loss_fn({k: v for k, v in heads.items()},
                                  cur, S[:, t + 1], E[:, t])
                losses.append(loss)
                with torch.no_grad():
                    cont_next = norm.apply_cont(cur, heads["cont"])
                    pred = projection.snap_batch(cur, cont_next, heads["cups"], heads["phase"])
                use_gt = (torch.rand(cur.shape[0], 1, device=device) < tf_prob).float()
                cur = (use_gt * S[:, t + 1] + (1 - use_gt) * pred).detach()
            loss = torch.stack(losses).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tot += float(loss)
        return tot / cfg.steps_per_epoch

    # fixed validation batch for a cheap free-run rollout proxy (keep-best)
    val_S, val_A, _ = _sample_batch(eps, cfg.window, min(128, cfg.batch), np.random.default_rng(cfg.seed + 777))
    best = {"proxy": float("inf"), "state": None}

    def rollout_proxy():
        model.eval()
        with torch.no_grad():
            preds = model.rollout(val_S[:, :cfg.ss_burn], val_A[:, :cfg.ss_burn], val_A[:, cfg.ss_burn:cfg.window])
            gt = val_S[:, cfg.ss_burn + 1:cfg.window + 1]
            n = min(preds.shape[1], gt.shape[1])
            err = (preds[:, :n, L.POS] - gt[:, :n, L.POS]).pow(2).sum(-1).sqrt().mean()
        model.train()
        return float(err)

    def track_best():
        p = rollout_proxy()
        if p < best["proxy"]:
            best["proxy"] = p
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return p

    t0 = time.time()
    for e in range(cfg.tf_epochs):
        loss = teacher_forced_epoch()
        print(f"[TF {e+1}/{cfg.tf_epochs}] loss={loss:.4f}  proxy={track_best():.3f}  ({time.time()-t0:.0f}s)")
    for e in range(cfg.ss_epochs):
        frac = e / max(1, cfg.ss_epochs - 1)
        tf_prob = cfg.ss_tf_start + (cfg.ss_tf_end - cfg.ss_tf_start) * frac
        loss = scheduled_epoch(tf_prob)
        print(f"[SS {e+1}/{cfg.ss_epochs} tf={tf_prob:.2f}] loss={loss:.4f}  proxy={track_best():.3f}  ({time.time()-t0:.0f}s)")

    if best["state"] is not None:
        model.load_state_dict(best["state"])
        print(f"restored best checkpoint (rollout proxy={best['proxy']:.3f})")

    os.makedirs(os.path.dirname(os.path.abspath(cfg.ckpt)), exist_ok=True)
    torch.save({"model": model.state_dict(), "hidden": cfg.hidden}, cfg.ckpt + ".pt")
    norm.save(cfg.ckpt + ".norm.npz")
    with open(cfg.ckpt + ".json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    print(f"saved {cfg.ckpt}.pt (+ .norm.npz, .json)")
    return model
