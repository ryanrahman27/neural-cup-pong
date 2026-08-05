"""Inspect a Neural Cup Pong dataset: stats, validation, and a contact sheet.

    python scripts/inspect_dataset.py --dir data/cup_v1 --episode 0 --sheet out/ep0.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--sheet", type=str, default=None)
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args(argv)

    import numpy as np

    from neural_cup_pong.data import schema
    from neural_cup_pong.data.dataset import load_episode
    from neural_cup_pong.data.validation import validate_dir

    paths = sorted(glob.glob(os.path.join(args.dir, schema.EPISODE_GLOB)))
    if not paths:
        print(f"No episodes in {args.dir}")
        return 1

    mpath = os.path.join(args.dir, "manifest.json")
    if os.path.exists(mpath):
        m = json.load(open(mpath))
        print(f"Episodes {m['num_episodes']}  frames {m['total_frames']}")
        print(f"Policies: {m['policy_counts']}")
        print(f"Events:   {m['event_totals']}")
    lengths = [load_episode(p, with_frames=False).states.shape[0] for p in paths]
    print(f"Length: min={min(lengths)} max={max(lengths)} mean={np.mean(lengths):.1f}")

    if not args.no_validate:
        print()
        validate_dir(args.dir)

    if args.episode is not None and args.sheet:
        from neural_cup_pong.data.replay import export_contact_sheet
        out = export_contact_sheet(paths[args.episode], args.sheet)
        print(f"contact sheet -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
