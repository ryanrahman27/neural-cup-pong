"""Evaluate a trained Phase-3 dynamics model.

    python scripts/eval_dynamics.py --ckpt checkpoints/phase3_gru
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="checkpoints/phase3_gru")
    p.add_argument("--seeds", type=int, default=16, help="number of held-out eval seeds")
    p.add_argument("--out", default=None, help="optional metrics.json output path")
    a = p.parse_args(argv)

    from neural_cup_pong.eval.evaluate import evaluate
    m = evaluate(a.ckpt, seeds=range(5000, 5000 + a.seeds))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(m, open(a.out, "w"), indent=2)
        print(f"\nmetrics -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
