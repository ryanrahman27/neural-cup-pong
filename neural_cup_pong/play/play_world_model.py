"""Phase 4: playable learned simulator (engine-off).

Same controls as the engine, but you can toggle who advances the world:

    F1  DETERMINISTIC ENGINE   (the real physics)
    F2  NEURAL WORLD MODEL     (engine OFF; the trained GRU predicts each next
                                structured state, drawn by the real renderer)

Controls: Left/Right aim, Up/Down power, Space throw, R reset, Esc quit.
The key message when in F2: GAME ENGINE ACTIVE: NO.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Play Neural Cup Pong (engine vs neural world model).")
    p.add_argument("--ckpt", default="checkpoints/phase3_gru")
    p.add_argument("--visual-ckpt", default="checkpoints/phase5_decoder")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scale", type=int, default=None)
    p.add_argument("--start-neural", action="store_true")
    args = p.parse_args(argv)

    os.environ.pop("SDL_VIDEODRIVER", None)
    import pygame
    import torch

    from neural_cup_pong.environment import actions as A
    from neural_cup_pong.environment import constants as C
    from neural_cup_pong.environment import state as St
    from neural_cup_pong.environment.game import NeuralCupPongEnv
    from neural_cup_pong.eval.evaluate import load_model

    model, device = load_model(args.ckpt)

    # optional Phase-5 neural decoder for the F3 fully-generated mode
    decoder = None
    try:
        import os as _os
        if _os.path.exists(args.visual_ckpt + ".pt"):
            from neural_cup_pong.models.visual.autoencoder import Decoder
            from neural_cup_pong.models.visual.geometry import geometry_hints as _ghints
            decoder = Decoder().to(device)
            decoder.load_state_dict(torch.load(args.visual_ckpt + ".pt", map_location=device)["decoder"])
            decoder.eval()
            print("F3 available: neural decoder loaded (engine + renderer OFF)")
    except Exception as e:
        print(f"F3 unavailable ({e})")

    scale = args.scale or C.DISPLAY_SCALE
    disp_w, disp_h = C.OBS_W * scale, C.OBS_H * scale
    pygame.init()
    pygame.display.set_caption("NEURAL CUP PONG — engine vs neural world model")
    screen = pygame.display.set_mode((disp_w, disp_h))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", max(11, scale * 3))

    env = NeuralCupPongEnv()
    env.reset(seed=args.seed)
    mode = "neural" if args.start_neural else "engine"
    lvec = env.state.to_vector()          # learned-mode state vector
    h = None

    def enter_neural():
        nonlocal lvec, h
        lvec = env.state.to_vector()       # seed the learned sim from the current state
        h = None

    if mode == "neural":
        enter_neural()

    world_dt = 1.0 / C.OBS_HZ
    acc = 0.0
    running = True

    while running:
        acc += clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset(seed=args.seed); enter_neural()
                elif event.key == pygame.K_F1:
                    mode = "engine"
                elif event.key == pygame.K_F2:
                    mode = "neural"; enter_neural()
                elif event.key == pygame.K_F3 and decoder is not None:
                    mode = "generated"; enter_neural()   # engine AND renderer OFF

        while acc >= world_dt:
            keys = pygame.key.get_pressed()
            act = A.make_action(
                aim_left=keys[pygame.K_LEFT] or keys[pygame.K_a],
                aim_right=keys[pygame.K_RIGHT] or keys[pygame.K_d],
                power_up=keys[pygame.K_UP] or keys[pygame.K_w],
                power_down=keys[pygame.K_DOWN] or keys[pygame.K_s],
                throw=keys[pygame.K_SPACE],
            )
            if mode == "engine":
                env.step(act)
            else:
                with torch.no_grad():
                    lv = torch.tensor(lvec, dtype=torch.float32, device=device)[None]
                    a = torch.tensor(act, dtype=torch.float32, device=device)[None]
                    nv, h = model.predict_step(lv, a, h)
                lvec = nv[0].cpu().numpy()
            acc -= world_dt

        if mode == "generated":                 # engine OFF and renderer OFF
            with torch.no_grad():
                lv = torch.tensor(lvec, dtype=torch.float32, device=device)[None]
                gen = decoder(lv, _ghints(lv))[0]
                frame = (gen.permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        else:
            cur_state = env.state if mode == "engine" else St.from_vector(lvec)
            frame = env._renderer.render(cur_state)
        surf = pygame.transform.scale(
            pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2))), (disp_w, disp_h))
        screen.blit(surf, (0, 0))
        _overlay(screen, font, mode, clock.get_fps())
        pygame.display.flip()

    pygame.quit()
    return 0


def _overlay(screen, font, mode, fps) -> None:
    import pygame

    w, h = screen.get_width(), screen.get_height()
    strip = pygame.Surface((w, 42), pygame.SRCALPHA)
    strip.fill((0, 0, 0, 170))
    screen.blit(strip, (0, h - 42))
    info = {
        "engine":    ("YES", "WORLD: DETERMINISTIC ENGINE",           (120, 230, 140)),
        "neural":    ("NO",  "WORLD: NEURAL STATE (233K) + renderer",  (250, 200, 90)),
        "generated": ("NO",  "WORLD: NEURAL STATE + NEURAL DECODER (engine + renderer OFF)", (250, 110, 90)),
    }[mode]
    l1 = font.render(f"GAME ENGINE ACTIVE: {info[0]}   |   {info[1]}", True, info[2])
    l2 = font.render("F1 engine   F2 neural-state   F3 fully-generated   |   "
                     f"Arrows aim/power  Space throw  R reset  Esc quit   |   {fps:3.0f}fps",
                     True, (210, 210, 210))
    screen.blit(l1, (6, h - 40))
    screen.blit(l2, (6, h - 20))


if __name__ == "__main__":
    sys.exit(main())
