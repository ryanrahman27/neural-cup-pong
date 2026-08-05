"""Hard projection / snap operator: re-quantize a predicted next-state so it is
always a *legal* game state. Applied identically in scheduled-sampling training
feedback and at Phase-4 inference (single source of truth).

Enforces: legal phase transitions (automaton), monotone cups (present -> sunk
only), score = NUM_CUPS - cups_left, monotone rounded throw counter, bounded
aim/power/z/timer, and phase-conditioned parking (ball at the throw origin +
zero velocity outside FLIGHT).
"""

from __future__ import annotations

import torch

from ..environment import constants as C
from . import layout as L

# legal[prev_phase, next_phase]
_LEGAL = torch.tensor([
    [1, 1, 0, 0],   # AIM -> AIM, FLIGHT
    [0, 1, 1, 0],   # FLIGHT -> FLIGHT, RESULT
    [1, 0, 1, 1],   # RESULT -> AIM, RESULT, GAME_OVER
    [0, 0, 0, 1],   # GAME_OVER -> GAME_OVER
], dtype=torch.float32)
_ORIGIN = torch.tensor(C.THROW_ORIGIN, dtype=torch.float32)


def snap_batch(prev: torch.Tensor, cont_next: torch.Tensor,
               cups_logits: torch.Tensor, phase_logits: torch.Tensor) -> torch.Tensor:
    """prev, cont_next: [B, STATE_DIM]; cups_logits [B,6]; phase_logits [B,4].
    Returns a legal snapped next state [B, STATE_DIM]."""
    dev = prev.device
    nxt = cont_next.clone()
    legal = _LEGAL.to(dev)
    origin = _ORIGIN.to(dev)

    prev_phase = prev[:, L.PHASE].argmax(-1)
    mask = legal[prev_phase]                                  # [B,4]
    next_phase = phase_logits.masked_fill(mask == 0, -1e9).argmax(-1)  # [B]

    # continuous clamps
    nxt[:, L.AIM] = nxt[:, L.AIM].clamp(-1.0, 1.0)
    nxt[:, L.POWER] = nxt[:, L.POWER].clamp(0.0, 1.0)
    nxt[:, 2] = nxt[:, 2].clamp(min=0.0)                      # ball z >= 0

    # cups: threshold + monotone (a present cup may sink, never reappear)
    cups_pred = (torch.sigmoid(cups_logits) > 0.5).float()
    nxt_cups = torch.minimum(prev[:, L.CUPS], cups_pred)
    nxt[:, L.CUPS] = nxt_cups
    # score derived from cups; throws monotone non-decreasing, rounded
    nxt[:, L.SCORE] = C.NUM_CUPS - nxt_cups.sum(-1)
    nxt[:, L.THROWS] = torch.maximum(nxt[:, L.THROWS].round(), prev[:, L.THROWS])

    # phase one-hot
    ph = torch.zeros_like(nxt[:, L.PHASE])
    ph.scatter_(1, next_phase[:, None], 1.0)
    nxt[:, L.PHASE] = ph

    # timer only lives in RESULT
    not_result = next_phase != C.PHASE_RESULT
    nxt[not_result, L.TIMER] = 0.0
    nxt[:, L.TIMER] = nxt[:, L.TIMER].round().clamp(0.0, float(C.RESULT_STEPS))

    # phase-conditioned parking
    is_aim = next_phase == C.PHASE_AIM
    not_flight = next_phase != C.PHASE_FLIGHT
    nxt[is_aim, 0:3] = origin                                 # ball parked at origin during AIM
    nxt[not_flight, 3:6] = 0.0                                # zero velocity outside FLIGHT
    return nxt


def snap(prev: torch.Tensor, cont_next: torch.Tensor,
         cups_logits: torch.Tensor, phase_logits: torch.Tensor) -> torch.Tensor:
    """Single-state convenience wrapper (1-D tensors)."""
    out = snap_batch(prev[None], cont_next[None], cups_logits[None], phase_logits[None])
    return out[0]
