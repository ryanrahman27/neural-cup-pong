"""Phase-3 evaluation: one-step error, autoregressive rollout, controllability,
and engine-agreement — on fresh held-out seeds disjoint from training.
"""

from __future__ import annotations

import numpy as np
import torch

from ..data import policies as P
from ..environment import constants as C
from ..environment.game import NeuralCupPongEnv
from ..environment.state import decode_vector
from ..models import layout as L
from ..models import projection
from ..models.dynamics_gru import PongDynamicsGRU
from ..models.normalizer import Normalizer


def load_model(ckpt: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    norm = Normalizer.load(ckpt + ".norm.npz")
    blob = torch.load(ckpt + ".pt", map_location=device)
    model = PongDynamicsGRU(norm, hidden=blob.get("hidden", 192)).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, device


def gen_trajectory(seed: int, policy_name: str, max_steps: int = 1600):
    env = NeuralCupPongEnv()
    _, state = env.reset(seed=seed)
    policy = P.POLICY_REGISTRY[policy_name]()
    prng = np.random.default_rng(seed * 2 + 1)
    states = [state.to_vector()]
    actions, events = [], []
    for t in range(max_steps):
        a = policy(state, prng, t)
        _, state, _, term, _, info = env.step(a)
        actions.append(a.astype(np.float32))
        events.append(info.events.astype(np.float32))
        states.append(state.to_vector())
        if term:
            break
    return np.stack(states), np.stack(actions), np.stack(events)


def _diverged(pred_vec, gt_vec):
    pd, gd = decode_vector(pred_vec), decode_vector(gt_vec)
    if pd["game_phase"] != gd["game_phase"]:
        return True
    if pd["cups_left"] != gd["cups_left"]:
        return True
    return float(np.linalg.norm(pred_vec[L.POS] - gt_vec[L.POS])) > C.CUP_R


def _invariants_ok(seq):
    scores = np.array([decode_vector(s)["score"] for s in seq])
    cups = np.stack([s[L.CUPS] for s in seq])
    if (np.diff(scores) < 0).any():
        return False
    if (np.diff(cups, axis=0) > 0).any():                # a cup reappeared
        return False
    if not np.all(scores == (C.NUM_CUPS - cups.sum(1)).round()):
        return False
    return True


@torch.no_grad()
def evaluate(ckpt: str, seeds=range(5000, 5016), policies=("competent", "explore"),
             burn: int = 4, horizons=(5, 10, 30, 100), verbose=True):
    model, device = load_model(ckpt)
    one_pos, one_aim, one_pow, one_cup, one_phase, one_thr = [], [], [], [], [], []
    roll_pos = {h: [] for h in horizons}
    phase_acc = {h: [] for h in horizons}
    divergence, inv_ok, score_mae, cause_match = [], [], [], []

    for seed in seeds:
        for pol in policies:
            S, A, E = gen_trajectory(seed, pol)
            T = A.shape[0]
            St = torch.tensor(S, device=device)
            At = torch.tensor(A, device=device)

            # one-step (teacher-forced, hidden carried along GT)
            heads, h = model(St[:T].unsqueeze(0), At.unsqueeze(0))
            cont_next = model.norm.apply_cont(St[:T].unsqueeze(0), heads["cont"])
            pred1 = projection.snap_batch(
                St[:T].unsqueeze(0).reshape(-1, L.STATE_DIM),
                cont_next.reshape(-1, L.STATE_DIM),
                heads["cups"].reshape(-1, 6), heads["phase"].reshape(-1, 4)).reshape(T, L.STATE_DIM)
            gt_next = St[1:T + 1]
            ph_in = St[:T, L.PHASE].argmax(-1)
            one_pos.append(float((pred1[:, L.POS] - gt_next[:, L.POS]).pow(2).sum(-1).sqrt().mean()))
            aim_mask = ph_in == C.PHASE_AIM
            if aim_mask.any():
                one_aim.append(float((pred1[aim_mask, L.AIM] - gt_next[aim_mask, L.AIM]).abs().mean()))
                one_pow.append(float((pred1[aim_mask, L.POWER] - gt_next[aim_mask, L.POWER]).abs().mean()))
            one_cup.append(float((pred1[:, L.CUPS].round() == gt_next[:, L.CUPS]).float().mean()))
            one_phase.append(float((pred1[:, L.PHASE].argmax(-1) == gt_next[:, L.PHASE].argmax(-1)).float().mean()))
            one_thr.append(float((pred1[:, L.THROWS].round() == gt_next[:, L.THROWS].round()).float().mean()))

            # free-run rollout with GT actions
            if T <= burn + 2:
                continue
            preds = model.rollout(St[:burn].unsqueeze(0), At[:burn].unsqueeze(0),
                                  At[burn:].unsqueeze(0))[0]           # [T-burn, 21]
            gt_roll = St[burn + 1:T + 1]
            n = min(preds.shape[0], gt_roll.shape[0])
            preds_np = preds[:n].cpu().numpy()
            gt_np = gt_roll[:n].cpu().numpy()
            for hh in horizons:
                if n >= hh:
                    roll_pos[hh].append(float(np.linalg.norm(preds_np[hh - 1, L.POS] - gt_np[hh - 1, L.POS])))
                    phase_acc[hh].append(float(np.mean([
                        decode_vector(preds_np[i])["game_phase"] == decode_vector(gt_np[i])["game_phase"]
                        for i in range(hh)])))
            div = n
            for i in range(n):
                if _diverged(preds_np[i], gt_np[i]):
                    div = i
                    break
            divergence.append(div)
            inv_ok.append(_invariants_ok(preds_np))
            score_mae.append(abs(decode_vector(preds_np[-1])["score"] - decode_vector(gt_np[-1])["score"]))
            cause_match.append(decode_vector(preds_np[-1])["game_phase"] == decode_vector(gt_np[-1])["game_phase"])

    ctrl = controllability(model, device)
    m = {
        "one_step": {
            "pos_rmse": float(np.mean(one_pos)),
            "aim_mae": float(np.mean(one_aim)), "power_mae": float(np.mean(one_pow)),
            "cups_acc": float(np.mean(one_cup)), "phase_acc": float(np.mean(one_phase)),
            "throws_exact": float(np.mean(one_thr)),
        },
        "rollout_pos_rmse": {h: float(np.mean(roll_pos[h])) if roll_pos[h] else None for h in horizons},
        "phase_timeline_acc": {h: float(np.mean(phase_acc[h])) if phase_acc[h] else None for h in horizons},
        "steps_until_divergence_median": float(np.median(divergence)) if divergence else None,
        "invariant_ok_rate": float(np.mean(inv_ok)) if inv_ok else None,
        "engine_score_mae": float(np.mean(score_mae)) if score_mae else None,
        "controllability": ctrl,
    }
    if verbose:
        _print(m)
    return m


@torch.no_grad()
def controllability(model, device, n: int = 64):
    """Matched counterfactuals from AIM states: does the right button win?"""
    from ..environment import actions as A
    aim_ok = pow_ok = throw_ok = 0
    env = NeuralCupPongEnv()
    trials = 0
    for seed in range(9000, 9000 + n):
        _, st = env.reset(seed=seed)
        for _ in range(int(env.rng.integers(1, 20))):   # random AIM state
            env.step(A.make_action(aim_right=bool(env.rng.integers(0, 2)),
                                    power_up=bool(env.rng.integers(0, 2))))
        s = torch.tensor(env.state.to_vector(), device=device)
        if env.state.game_phase != C.PHASE_AIM:
            continue
        trials += 1

        def daim(act):
            h, _ = model.forward_step(s.unsqueeze(0), torch.tensor(act, device=device).unsqueeze(0))
            nxt = model.norm.apply_cont(s.unsqueeze(0), h["cont"])[0]
            return float(nxt[L.AIM] - s[L.AIM]), float(nxt[L.POWER] - s[L.POWER]), h["phase"][0]

        aR = daim(A.make_action(aim_right=True))[0]
        aL = daim(A.make_action(aim_left=True))[0]
        aim_ok += aR > aL
        pU = daim(A.make_action(power_up=True))[1]
        pD = daim(A.make_action(power_down=True))[1]
        pow_ok += pU > pD
        ph = daim(A.make_action(throw=True))[2]
        throw_ok += int(ph.argmax().item()) == C.PHASE_FLIGHT
    trials = max(1, trials)
    return {"aim_dir_rate": aim_ok / trials, "power_dir_rate": pow_ok / trials,
            "throw_launch_rate": throw_ok / trials, "trials": trials}


def _print(m):
    o = m["one_step"]
    print("\n=== ONE-STEP (teacher-forced) ===")
    print(f"  pos_rmse={o['pos_rmse']:.3f}u  aim_mae={o['aim_mae']:.4f}  power_mae={o['power_mae']:.4f}")
    print(f"  cups_acc={o['cups_acc']:.4f}  phase_acc={o['phase_acc']:.4f}  throws_exact={o['throws_exact']:.4f}")
    print("=== ROLLOUT (free-run, GT actions) ===")
    print(f"  pos_rmse@H: " + "  ".join(f"H{h}={v:.2f}" if v is not None else f"H{h}=NA"
                                        for h, v in m["rollout_pos_rmse"].items()))
    print(f"  phase_acc@H: " + "  ".join(f"H{h}={v:.3f}" if v is not None else f"H{h}=NA"
                                         for h, v in m["phase_timeline_acc"].items()))
    print(f"  steps_until_divergence(median)={m['steps_until_divergence_median']}")
    print(f"  invariant_ok_rate={m['invariant_ok_rate']}  engine_score_mae={m['engine_score_mae']:.3f}")
    c = m["controllability"]
    print("=== CONTROLLABILITY ===")
    print(f"  aim_dir={c['aim_dir_rate']:.3f}  power_dir={c['power_dir_rate']:.3f}  "
          f"throw_launch={c['throw_launch_rate']:.3f}  (n={c['trials']})")
