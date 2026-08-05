"""Train the Phase-5 state-grounded decoder (Stage 1: state-only, no latent).

Learns Decoder(state, hints(state)) -> frame from the dataset's exact states +
frames. At Phase-6 time the same decoder is fed the Phase-3 GRU's predicted
state. Trains on GPU. Saves ``<ckpt>.pt`` + ``<ckpt>.json``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch

from ..data.dataset import TrajectoryDataset
from ..models.visual.autoencoder import Decoder
from ..models.visual.geometry import geometry_hints
from .visual_losses import VisualLoss


@dataclass
class TrainVisualConfig:
    data_dir: str = "data/cup_v2"
    ckpt: str = "checkpoints/phase5_decoder"
    n_frames: int = 60000
    val_frames: int = 4000
    batch: int = 48
    lr: float = 1.5e-3
    epochs: int = 10
    seed: int = 0


def load_frame_pairs(data_dir, n_frames, seed):
    ds = TrajectoryDataset(data_dir, window=1, with_frames=True, preload=False)
    rng = np.random.default_rng(seed)
    n_ep = len(ds.paths)
    per_ep = max(1, n_frames // n_ep + 1)
    frames, states = [], []
    for ep in ds.iter_episodes():
        T = ep.frames.shape[0]
        idx = rng.integers(0, T, size=min(per_ep, T))
        frames.append(ep.frames[idx])
        states.append(ep.states[idx])
    frames = np.concatenate(frames)[:n_frames]
    states = np.concatenate(states)[:n_frames]
    perm = rng.permutation(len(frames))
    return frames[perm], states[perm]


def main(cfg: TrainVisualConfig):
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames, states = load_frame_pairs(cfg.data_dir, cfg.n_frames + cfg.val_frames, cfg.seed)
    v = cfg.val_frames
    tr_f, tr_s = frames[v:], states[v:]
    va_f = torch.tensor(frames[:v], device=device)
    va_s = torch.tensor(states[:v], device=device)
    N = len(tr_f)
    print(f"device={device}  train_frames={N}  val={v}")

    model = Decoder().to(device)
    print(f"decoder params={model.param_count()}")
    loss_fn = VisualLoss().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed + 1)

    def to_gpu(f_np, s_np):
        f = torch.tensor(f_np, device=device).permute(0, 3, 1, 2).float() / 255.0
        s = torch.tensor(s_np, device=device)
        return f, s

    @torch.no_grad()
    def val_loss():
        model.eval(); tot = 0.0; nb = 0
        for i in range(0, v, cfg.batch):
            f = va_f[i:i + cfg.batch].permute(0, 3, 1, 2).float() / 255.0
            s = va_s[i:i + cfg.batch]
            pred = model(s, geometry_hints(s))
            tot += float((pred - f).abs().mean()); nb += 1
        model.train(); return tot / max(1, nb)

    best = {"l1": float("inf"), "state": None}
    steps = N // cfg.batch
    t0 = time.time()
    for e in range(cfg.epochs):
        order = rng.permutation(N)
        run = 0.0
        for b in range(steps):
            sel = order[b * cfg.batch:(b + 1) * cfg.batch]
            f, s = to_gpu(tr_f[sel], tr_s[sel])
            hn = geometry_hints(s)                    # computed once, reused in loss
            pred = model(s, hn)
            loss, _ = loss_fn(pred, f, s, hn)
            opt.zero_grad(); loss.backward(); opt.step()
            run += float(loss)
        vl = val_loss()
        if vl < best["l1"]:
            best["l1"] = vl
            best["state"] = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        print(f"[{e+1}/{cfg.epochs}] train={run/steps:.4f}  val_L1={vl:.4f}  ({time.time()-t0:.0f}s)")

    if best["state"] is not None:
        model.load_state_dict(best["state"])
        print(f"restored best (val_L1={best['l1']:.4f})")
    os.makedirs(os.path.dirname(os.path.abspath(cfg.ckpt)), exist_ok=True)
    torch.save({"decoder": model.state_dict()}, cfg.ckpt + ".pt")
    json.dump(asdict(cfg), open(cfg.ckpt + ".json", "w"), indent=2)
    print(f"saved {cfg.ckpt}.pt")
    return model
