"""Action-conditioned GRU dynamics model (~230K params).

Predicts the next structured state from the current state + action, with a
multi-head decoder: continuous head (delta for pos/aim/power/counts/timer,
absolute for velocity), cups bitmask logits, phase logits, and an auxiliary
event head. Rollout feeds each prediction through ``projection.snap`` so states
stay legal. GRU hidden state carries history -> O(1) per inference step.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import layout as L
from . import projection
from .normalizer import Normalizer


class PongDynamicsGRU(nn.Module):
    def __init__(self, normalizer: Normalizer, hidden: int = 192, layers: int = 1):
        super().__init__()
        self.norm = normalizer
        self.hidden = hidden
        self.enc = nn.Linear(L.STATE_DIM + L.ACTION_DIM, hidden)
        self.act = nn.SiLU()
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True)
        self.h_cont = nn.Linear(hidden, L.CONT_DIM)
        self.h_cups = nn.Linear(hidden, 6)
        self.h_phase = nn.Linear(hidden, 4)
        self.h_event = nn.Linear(hidden, L.EVENT_DIM)

    def _heads(self, z):
        return {"cont": self.h_cont(z), "cups": self.h_cups(z),
                "phase": self.h_phase(z), "event": self.h_event(z)}

    def forward(self, states, actions, h=None):
        """states [B,L,21], actions [B,L,5] -> heads (each [B,L,*]), h."""
        x = torch.cat([self.norm.normalize_input(states), actions], dim=-1)
        z, h = self.gru(self.act(self.enc(x)), h)
        return self._heads(z), h

    def forward_step(self, state, action, h=None):
        """state [B,21], action [B,5] -> heads (each [B,*]), h."""
        heads, h = self.forward(state.unsqueeze(1), action.unsqueeze(1), h)
        return {k: v[:, 0] for k, v in heads.items()}, h

    @torch.no_grad()
    def predict_step(self, state, action, h=None):
        """Raw state [B,21] + action [B,5] -> snapped next raw state [B,21], h."""
        heads, h = self.forward_step(state, action, h)
        cont_next = self.norm.apply_cont(state, heads["cont"])
        nxt = projection.snap_batch(state, cont_next, heads["cups"], heads["phase"])
        return nxt, h

    @torch.no_grad()
    def rollout(self, warm_states, warm_actions, roll_actions, h=None):
        """Warm the hidden state on ground-truth (warm_states [B,W,21],
        warm_actions [B,W,5]), then free-run over roll_actions [B,T,5].
        Returns predicted raw states [B,T,21]."""
        if warm_states.shape[1] > 0:
            _, h = self.forward(warm_states, warm_actions, h)
            cur = warm_states[:, -1]
        else:
            cur = warm_states.new_zeros(warm_actions.shape[0], L.STATE_DIM)
        preds = []
        for t in range(roll_actions.shape[1]):
            cur, h = self.predict_step(cur, roll_actions[:, t], h)
            preds.append(cur)
        return torch.stack(preds, dim=1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(normalizer: Normalizer, hidden: int = 192, layers: int = 1) -> PongDynamicsGRU:
    m = PongDynamicsGRU(normalizer, hidden=hidden, layers=layers)
    n = m.param_count()
    assert n < 500_000, f"model too big: {n} params"
    return m
