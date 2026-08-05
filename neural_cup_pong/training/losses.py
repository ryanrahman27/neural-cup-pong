"""Per-field weighted dynamics loss with motion/transition masking.

Continuous head: Huber in normalized space (higher weight on position, which
Phase 4 draws). Cups: BCE. Phase: cross-entropy. Events: auxiliary BCE. A
motion mask upweights moving/flight frames and a transition mask upweights the
sparse throw/sink/land ticks, so gradient goes where rollouts actually break.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models import layout as L

# per-continuous-field base weights: pos(3), vel(3), aim, power, score, throws, timer
_CONT_W = torch.tensor([2., 2., 2., 1., 1., 1., 1., 1., 0.5, 0.5, 0.5])
# events that mark a discrete transition tick (throw/sink/miss/table/rim)
_TRANS_EVENTS = [0, 1, 2, 3, 4]


class DynamicsLoss(nn.Module):
    def __init__(self, normalizer, w_cups=3.0, w_phase=3.0, w_event=0.5):
        super().__init__()
        self.norm = normalizer
        self.register_buffer("cont_w", _CONT_W)
        self.w_cups, self.w_phase, self.w_event = w_cups, w_phase, w_event

    def forward(self, heads, cur, nxt, events):
        """heads: dict of [...,*]; cur/nxt: [...,21]; events: [...,7]. Scalar loss."""
        cont_target = self.norm.norm_cont_target(cur, nxt)                 # [...,11]
        huber = F.smooth_l1_loss(heads["cont"], cont_target, reduction="none")

        is_moving = (nxt[..., L.VEL].pow(2).sum(-1) > 1e-3).float()        # [...]
        is_trans = (events[..., _TRANS_EVENTS].sum(-1) > 0).float()        # [...]
        trans_w = 1.0 + 2.0 * is_trans
        motion_w = 1.0 + 4.0 * is_moving

        w = self.cont_w.to(huber) * trans_w.unsqueeze(-1)                  # [...,11]
        w[..., 0:6] = w[..., 0:6] * motion_w.unsqueeze(-1)
        cont_loss = (huber * w).mean()

        cups_bce = F.binary_cross_entropy_with_logits(
            heads["cups"], nxt[..., L.CUPS], reduction="none").mean(-1)     # [...]
        cups_loss = (cups_bce * trans_w).mean()

        phase_tgt = nxt[..., L.PHASE].argmax(-1)
        ce = F.cross_entropy(heads["phase"].reshape(-1, 4), phase_tgt.reshape(-1),
                             reduction="none").reshape(phase_tgt.shape)
        phase_loss = (ce * trans_w).mean()

        event_loss = F.binary_cross_entropy_with_logits(heads["event"], events.float())

        total = cont_loss + self.w_cups * cups_loss + self.w_phase * phase_loss \
            + self.w_event * event_loss
        return total, {"cont": float(cont_loss.detach()), "cups": float(cups_loss.detach()),
                       "phase": float(phase_loss.detach()), "event": float(event_loss.detach()),
                       "total": float(total.detach())}
