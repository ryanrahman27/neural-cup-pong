"""Train the Phase-5 state-grounded decoder (Stage 1).

    python scripts/train_visual.py --data data/cup_v2 --ckpt checkpoints/phase5_decoder
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
    p.add_argument("--data", default="data/cup_v2")
    p.add_argument("--ckpt", default="checkpoints/phase5_decoder")
    p.add_argument("--n-frames", type=int, default=60000)
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    from neural_cup_pong.training.train_visual import TrainVisualConfig, main as tmain
    tmain(TrainVisualConfig(data_dir=a.data, ckpt=a.ckpt, n_frames=a.n_frames,
                            batch=a.batch, epochs=a.epochs, lr=a.lr, seed=a.seed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
