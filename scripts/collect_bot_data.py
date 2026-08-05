"""Collect a bot/exploration dataset of Neural Cup Pong trajectories.

    python scripts/collect_bot_data.py --episodes 200 --out data/cup_v1 --validate
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out", type=str, default="data/cup_v1")
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)

    from neural_cup_pong.data.collect import collect_dataset
    from neural_cup_pong.data.validation import validate_dir

    collect_dataset(args.out, args.episodes, base_seed=args.base_seed, max_steps=args.max_steps)
    print(f"\nManifest -> {os.path.join(args.out, 'manifest.json')}")
    if args.validate:
        print("\nValidating...")
        report = validate_dir(args.out)
        if report["problem_episodes"]:
            print("WARNING: problems found.")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
