"""Shared field layout for the dynamics model (state vector + continuous head).

State vector (STATE_DIM=21): pos[0:3] vel[3:6] aim[6] power[7] cups[8:14]
score[14] throws[15] phase[16:20] result_timer[20].

Continuous regression head (11 values): pos_delta[0:3], vel_absolute[3:6],
aim_delta[6], power_delta[7], score_delta[8], throws_delta[9], timer_delta[10].
Velocity is the only DIRECT (absolute) field; everything else in the head is a
delta (next - current).
"""

from __future__ import annotations

STATE_DIM = 21
ACTION_DIM = 5
EVENT_DIM = 7
CONT_DIM = 11

# state-vector slices
POS = slice(0, 3)
VEL = slice(3, 6)
AIM = 6
POWER = 7
CUPS = slice(8, 14)
SCORE = 14
THROWS = 15
PHASE = slice(16, 20)
TIMER = 20

# continuous-head slices
H_POS = slice(0, 3)
H_VEL = slice(3, 6)
H_AIM = 6
H_POWER = 7
H_SCORE = 8
H_THROWS = 9
H_TIMER = 10

# which state indices each continuous-head entry maps to (for delta add-back)
DELTA_STATE_IDX = [0, 1, 2, None, None, None, 6, 7, 14, 15, 20]  # None = velocity (direct)
