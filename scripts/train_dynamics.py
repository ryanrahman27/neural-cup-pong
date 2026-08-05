"""Train the Phase-3 structured-dynamics GRU.

    python scripts/train_dynamics.py --data data/cup_v1 --ckpt checkpoints/phase3_gru
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/cup_v1")
    p.add_argument("--ckpt", default="checkpoints/phase3_gru")
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tf-epochs", type=int, default=8)
    p.add_argument("--ss-epochs", type=int, default=6)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    from neural_cup_pong.training.train import TrainConfig, main as train_main
    cfg = TrainConfig(data_dir=a.data, ckpt=a.ckpt, window=a.window, hidden=a.hidden,
                      batch=a.batch, lr=a.lr, tf_epochs=a.tf_epochs, ss_epochs=a.ss_epochs,
                      steps_per_epoch=a.steps_per_epoch, seed=a.seed)
    train_main(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
