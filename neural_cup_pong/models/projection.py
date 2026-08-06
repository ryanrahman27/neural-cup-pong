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
_CUPS = torch.tensor(C.cup_layout(), dtype=torch.float32)   # [6,2]


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

    # --- rule-based flight resolution (from the model's PREDICTED ball) --------
    # The cups head under-fires the rare 1-tick sink flip, so instead resolve the
    # sink with the known game rule applied to the predicted ball position (same
    # spirit as snap's other legality rules). The model still predicts the flight.
    cups = _CUPS.to(dev)
    prev_cups = prev[:, L.CUPS]
    in_flight = prev_phase == C.PHASE_FLIGHT
    bz, bvz = nxt[:, 2], nxt[:, 5]
    dists = torch.cdist(nxt[:, 0:2], cups) + (1.0 - prev_cups) * 1e9   # mask absent cups
    mind, minc = dists.min(-1)                                # [B]
    sink = in_flight & (bvz < 0) & (bz <= C.CUP_RIM_Z) & (mind <= C.SINK_RADIUS)
    hit_table = in_flight & (bz <= C.BALL_R)                  # miss -> ends flight too
    nxt_cups = prev_cups.clone()
    if sink.any():
        rows = torch.where(sink)[0]
        nxt_cups[rows, minc[rows]] = 0.0
    nxt[:, L.CUPS] = nxt_cups
    nxt[:, L.SCORE] = C.NUM_CUPS - nxt_cups.sum(-1)
    nxt[:, L.THROWS] = torch.maximum(nxt[:, L.THROWS].round(), prev[:, L.THROWS])
    # a sink or a table-hit ends the throw -> force RESULT
    end_flight = sink | hit_table
    next_phase = torch.where(end_flight, torch.full_like(next_phase, C.PHASE_RESULT), next_phase)

    # phase one-hot
    ph = torch.zeros_like(nxt[:, L.PHASE])
    ph.scatter_(1, next_phase[:, None], 1.0)
    nxt[:, L.PHASE] = ph

    # timer only lives in RESULT (fresh RESULT_STEPS when a flight just ended)
    not_result = next_phase != C.PHASE_RESULT
    nxt[not_result, L.TIMER] = 0.0
    nxt[:, L.TIMER] = nxt[:, L.TIMER].round().clamp(0.0, float(C.RESULT_STEPS))
    nxt[end_flight, L.TIMER] = float(C.RESULT_STEPS)

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
