"""Keyboard-playable cup pong (Phase 1).

Controls
--------
    Aim ...... Left / Right  (or A / D)
    Power .... Up / Down      (or W / S)
    Throw .... Space
    Reset .... R
    Quit ..... Esc
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Play Neural Cup Pong (deterministic engine).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=int, default=None)
    args = parser.parse_args(argv)

    os.environ.pop("SDL_VIDEODRIVER", None)
    import pygame

    from neural_cup_pong.environment import actions as A
    from neural_cup_pong.environment import constants as C
    from neural_cup_pong.environment.game import NeuralCupPongEnv

    scale = args.scale or C.DISPLAY_SCALE
    disp_w, disp_h = C.OBS_W * scale, C.OBS_H * scale
    pygame.init()
    pygame.display.set_caption("NEURAL CUP PONG — engine mode")
    screen = pygame.display.set_mode((disp_w, disp_h))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", max(11, scale * 3))

    env = NeuralCupPongEnv()
    env.reset(seed=args.seed)
    world_dt = 1.0 / C.OBS_HZ
    accumulator = 0.0
    running = True

    while running:
        accumulator += clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset(seed=args.seed)

        while accumulator >= world_dt:
            keys = pygame.key.get_pressed()
            act = A.make_action(
                aim_left=keys[pygame.K_LEFT] or keys[pygame.K_a],
                aim_right=keys[pygame.K_RIGHT] or keys[pygame.K_d],
                power_up=keys[pygame.K_UP] or keys[pygame.K_w],
                power_down=keys[pygame.K_DOWN] or keys[pygame.K_s],
                throw=keys[pygame.K_SPACE],
            )
            _, _, _, term, _, _ = env.step(act)
            accumulator -= world_dt
            if term:
                break

        frame = env.render()
        surf = pygame.transform.scale(
            pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2))), (disp_w, disp_h))
        screen.blit(surf, (0, 0))
        strip = pygame.Surface((disp_w, 20), pygame.SRCALPHA)
        strip.fill((0, 0, 0, 150))
        screen.blit(strip, (0, disp_h - 20))
        label = font.render(
            f"ENGINE MODE | Arrows aim/power  Space throw  R reset  Esc quit | {clock.get_fps():3.0f}fps",
            True, (210, 210, 210))
        screen.blit(label, (6, disp_h - 18))
        pygame.display.flip()

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
