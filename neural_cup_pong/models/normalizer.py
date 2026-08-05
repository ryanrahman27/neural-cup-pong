"""Input/target normalization with frozen stats (single source of truth).

Two normalizations, computed once over the training set and stored in a sidecar
``.norm.npz`` so training and Phase-4 rollout use identical scaling:

* INPUT — what the GRU consumes (positions affine to ~[-1,1], velocities
  z-scored, aim/power/cups/counts scaled, phase pass-through).
* TARGET — the continuous head: deltas z-scored for pos/aim/power/counts/timer,
  absolute z-scored for velocity.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from . import layout as L

# fixed affine centers/scales for position (TABLE_W=60, TABLE_D=100, z_max~45)
_POS_C = np.array([30.0, 50.0, 22.5], dtype=np.float32)
_POS_S = np.array([30.0, 50.0, 22.5], dtype=np.float32)
_EPS = 1e-3


class Normalizer(nn.Module):
    def __init__(self, vel_mean, vel_std, cont_mean, cont_std):
        super().__init__()
        self.register_buffer("pos_c", torch.tensor(_POS_C))
        self.register_buffer("pos_s", torch.tensor(_POS_S))
        self.register_buffer("vel_mean", torch.as_tensor(vel_mean, dtype=torch.float32))
        self.register_buffer("vel_std", torch.as_tensor(vel_std, dtype=torch.float32))
        self.register_buffer("cont_mean", torch.as_tensor(cont_mean, dtype=torch.float32))
        self.register_buffer("cont_std", torch.as_tensor(cont_std, dtype=torch.float32))

    # --- input ---------------------------------------------------------------
    def normalize_input(self, s: torch.Tensor) -> torch.Tensor:
        """Normalize a state (or batch of states), last dim = STATE_DIM."""
        out = s.clone().float()
        out[..., L.POS] = (s[..., L.POS] - self.pos_c) / self.pos_s
        out[..., L.VEL] = (s[..., L.VEL] - self.vel_mean) / self.vel_std
        out[..., L.POWER] = (s[..., L.POWER] - 0.5) * 2.0
        out[..., L.CUPS] = s[..., L.CUPS] * 2.0 - 1.0
        out[..., L.SCORE] = s[..., L.SCORE] / 6.0
        out[..., L.THROWS] = s[..., L.THROWS] / 30.0
        out[..., L.TIMER] = s[..., L.TIMER] / 26.0
        # aim (idx 6) and phase one-hot (16:20) pass through unchanged
        return out

    # --- continuous target <-> raw ------------------------------------------
    def norm_cont_target(self, cur: torch.Tensor, nxt: torch.Tensor) -> torch.Tensor:
        """Build the normalized 11-dim continuous target from (cur, next) states."""
        raw = torch.zeros(*cur.shape[:-1], L.CONT_DIM, device=cur.device)
        raw[..., L.H_POS] = nxt[..., L.POS] - cur[..., L.POS]
        raw[..., L.H_VEL] = nxt[..., L.VEL]                       # absolute
        raw[..., L.H_AIM] = nxt[..., L.AIM] - cur[..., L.AIM]
        raw[..., L.H_POWER] = nxt[..., L.POWER] - cur[..., L.POWER]
        raw[..., L.H_SCORE] = nxt[..., L.SCORE] - cur[..., L.SCORE]
        raw[..., L.H_THROWS] = nxt[..., L.THROWS] - cur[..., L.THROWS]
        raw[..., L.H_TIMER] = nxt[..., L.TIMER] - cur[..., L.TIMER]
        return (raw - self.cont_mean) / self.cont_std

    def apply_cont(self, cur: torch.Tensor, cont_pred_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize a predicted continuous head and produce next raw fields.

        Returns a full STATE_DIM tensor with pos/vel/aim/power/score/throws/timer
        filled from the head (cups/phase come from the classifier heads / snap).
        """
        raw = cont_pred_norm * self.cont_std + self.cont_mean
        nxt = cur.clone().float()
        nxt[..., L.POS] = cur[..., L.POS] + raw[..., L.H_POS]
        nxt[..., L.VEL] = raw[..., L.H_VEL]                       # absolute
        nxt[..., L.AIM] = cur[..., L.AIM] + raw[..., L.H_AIM]
        nxt[..., L.POWER] = cur[..., L.POWER] + raw[..., L.H_POWER]
        nxt[..., L.SCORE] = cur[..., L.SCORE] + raw[..., L.H_SCORE]
        nxt[..., L.THROWS] = cur[..., L.THROWS] + raw[..., L.H_THROWS]
        nxt[..., L.TIMER] = cur[..., L.TIMER] + raw[..., L.H_TIMER]
        return nxt

    # --- persistence ---------------------------------------------------------
    def save(self, path: str) -> None:
        np.savez(path, vel_mean=self.vel_mean.cpu().numpy(), vel_std=self.vel_std.cpu().numpy(),
                 cont_mean=self.cont_mean.cpu().numpy(), cont_std=self.cont_std.cpu().numpy())

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        z = np.load(path)
        return cls(z["vel_mean"], z["vel_std"], z["cont_mean"], z["cont_std"])


def fit_normalizer(dataset) -> Normalizer:
    """Compute normalization stats over all valid transitions of a dataset."""
    vels, deltas = [], []
    for ep in dataset.iter_episodes():
        s = ep.states.astype(np.float32)
        valid = ep.valid.astype(bool)
        vels.append(s[:, L.VEL])
        idx = np.where(valid[:-1])[0]
        if len(idx) == 0:
            continue
        cur, nxt = s[idx], s[idx + 1]
        d = np.zeros((len(idx), L.CONT_DIM), dtype=np.float32)
        d[:, L.H_POS] = nxt[:, L.POS] - cur[:, L.POS]
        d[:, L.H_VEL] = nxt[:, L.VEL]                              # absolute
        d[:, L.H_AIM] = nxt[:, L.AIM] - cur[:, L.AIM]
        d[:, L.H_POWER] = nxt[:, L.POWER] - cur[:, L.POWER]
        d[:, L.H_SCORE] = nxt[:, L.SCORE] - cur[:, L.SCORE]
        d[:, L.H_THROWS] = nxt[:, L.THROWS] - cur[:, L.THROWS]
        d[:, L.H_TIMER] = nxt[:, L.TIMER] - cur[:, L.TIMER]
        deltas.append(d)
    vels = np.concatenate(vels, 0)
    deltas = np.concatenate(deltas, 0)
    vel_mean, vel_std = vels.mean(0), vels.std(0) + _EPS
    cont_mean = deltas.mean(0)
    cont_std = deltas.std(0) + _EPS
    # velocity entries of the continuous head use the absolute vel stats
    cont_mean[L.H_VEL] = vel_mean
    cont_std[L.H_VEL] = vel_std
    return Normalizer(vel_mean, vel_std, cont_mean, cont_std)
