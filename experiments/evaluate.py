"""Step 3 of a run: score the trained model on clips it has never seen.

* One-step error ("onestep_error"): give the model the true state(s), let it
  predict one step ahead, and divide the squared error by how much the state
  typically changes in one step. 1.0 = no better than guessing the average
  change; 0.01 = the error is 1% of a typical change.

* Rollout ("rollout_error@h", "valid_steps"): start from the true state and let
  the model feed on its own predictions for 100 steps (1 second). Error after h
  steps is in units of the state's spread (0.5 = clearly off). valid_steps =
  steps until the error first passes 0.5.

* Shuffle test ("shuffle_<group>"): scramble one input group (gravity, damping,
  tokens) between test examples and see how much the one-step error grows.
  About 0 = the model ignores that input; 25 = the error grew 26x.

* Train/test gap ("gap"): test error / training error. Much larger than 1 =
  the model memorised the training clips.

* Floor check (ground world only): how often a rod is more than 5 cm inside
  the floor, in the predicted and the true rollouts, and the one-step error
  split into steps with and without floor contact.
"""
from __future__ import annotations

import numpy as np
import torch

from experiments.config import MAX_CONTEXT
from experiments.data import PreparedData
from experiments.model import DynamicsMLP, predict
from experiments.systems import SimEnv

VALID_THRESHOLD = 0.5      # rollout error at which a prediction counts as "off"
REPORT_HORIZONS = (10, 25, 50, 100)
MAX_EVAL_EXAMPLES = 10_000
CHUNK = 4096
TOUCHING = 0.005           # lowest rod point below 5 mm = touching the floor
INSIDE_FLOOR = -0.05       # lowest rod point below -5 cm = clearly through the floor


def _one_step_sq_errors(data: PreparedData, model: DynamicsMLP, split: str, n_idx, t_idx, noise,
                        shuffle_columns: slice | None = None, perm=None) -> torch.Tensor:
    """Squared error of the next raw observation for every example: [B, D].

    With `shuffle_columns`, those condition columns are taken from the examples `perm` points to instead.
    """
    s = data.split[split]
    out = []
    for start in range(0, len(n_idx), CHUNK):
        sl = slice(start, start + CHUNK)
        n, t = n_idx[sl], t_idx[sl]
        context, cond, _ = data.batch(split, n, t)
        if shuffle_columns is not None:
            cond = cond.clone()
            cond[:, shuffle_columns] = s.cond_n[n_idx[perm[sl]], t_idx[perm[sl]]][:, shuffle_columns]
        pred = data.to_next_obs(predict(model, context, cond, noise[sl], data.cfg.inference_steps), s.obs[n, t])
        out.append((pred - s.obs[n, t + 1]) ** 2)
    return torch.cat(out)


def _error(sq: torch.Tensor, data: PreparedData) -> float:
    """Mean squared error divided by the typical one-step change, averaged over dimensions."""
    return (sq.mean(0) / data.delta_var).mean().item()


def _example_cap(data: PreparedData, flat_cap: int) -> int:
    """How many examples to score. The per-unit network runs once per unit, so for it the cap
    shrinks with the number of units (but never below 1/8 of the flat cap); 1,250 examples of
    256 pendulums are still 320,000 pendulum predictions."""
    if data.cfg.arch != "units":
        return flat_cap
    return max(flat_cap // 8, flat_cap * 8 // data.cfg.count)


def _eval_examples(data: PreparedData, split: str, seed: int):
    """The (clip, step) pairs used for one-step scoring; a fixed random subset if there are many."""
    n_idx, t_idx = data.examples(split, t_min=MAX_CONTEXT - 1)
    cap = _example_cap(data, MAX_EVAL_EXAMPLES)
    if len(n_idx) > cap:
        g = torch.Generator(device="cpu").manual_seed(seed)
        keep = torch.randperm(len(n_idx), generator=g)[:cap].to(data.device)
        n_idx, t_idx = n_idx[keep], t_idx[keep]
    return n_idx, t_idx


def _rollout(data: PreparedData, model: DynamicsMLP, split: str, seed: int) -> dict:
    """Let the model run on its own predictions for `rollout_horizon` steps, on every clip at once."""
    cfg = data.cfg
    s = data.split[split]
    n_traj = s.obs.shape[0]
    t0, horizon, K = cfg.rollout_start, cfg.rollout_horizon, cfg.context_len
    truth = s.obs[:, t0 + 1: t0 + horizon + 1]                 # [N, H, D]
    history_n = s.obs_n[:, t0 - K + 1: t0 + 1].clone()          # last K true states, scaled
    current = s.obs[:, t0].clone()
    mean = data.obs_norm.mean.to(current.device)
    g = torch.Generator(device="cpu").manual_seed(seed)

    preds = []
    for h in range(horizon):
        noise = torch.randn(n_traj, data.obs_dim, generator=g).to(data.device)
        pred_n = predict(model, history_n.reshape(n_traj, -1), s.cond_n[:, t0 + h], noise, cfg.inference_steps)
        nxt = data.to_next_obs(pred_n, current)
        # If a rollout explodes, keep the numbers finite so the scores stay defined.
        nxt = torch.clamp(torch.nan_to_num(nxt, nan=0.0), mean - 50 * data.state_std, mean + 50 * data.state_std)
        preds.append(nxt)
        current = nxt
        history_n = torch.cat([history_n[:, 1:], data.obs_norm.encode(nxt)[:, None]], dim=1)  # slide window
    pred = torch.stack(preds, dim=1)

    def nrmse(x):   # [N, H]: error per clip and step, in units of the state's spread
        return torch.sqrt((((x - truth) / data.state_std) ** 2).mean(-1))

    error = nrmse(pred)
    copy_error = nrmse(s.obs[:, t0:t0 + 1].expand_as(truth))  # baseline: nothing moves after t0
    off = error > VALID_THRESHOLD
    first_off = torch.where(off.any(1), off.float().argmax(1), torch.full((n_traj,), horizon, device=error.device))
    return {"error": error.cpu().numpy(), "copy_error": copy_error.cpu().numpy(),
            "valid_steps": first_off.float().cpu().numpy(),
            "pred": pred.cpu().numpy(), "truth": truth.cpu().numpy()}


def _lowest(data: PreparedData, split: str, states: np.ndarray, clips: np.ndarray) -> np.ndarray:
    """Lowest rod point of each state. When rod lengths/masses vary, each clip uses its own rods."""
    cfg = data.cfg
    params = data.split[split].params
    if not params:
        return SimEnv(cfg.system, cfg.count, cfg.links).lowest_point(states)
    out = np.empty(len(states))
    for n in np.unique(clips):
        sel = clips == n
        env = SimEnv(cfg.system, cfg.count, cfg.links, lengths=params["length"][n] if "length" in params else None,
                     masses=params["mass"][n] if "mass" in params else None)
        out[sel] = env.lowest_point(states[sel])
    return out


def _floor_check(data: PreparedData, roll: dict, sq: torch.Tensor, n_idx, t_idx) -> tuple[dict, dict]:
    n_traj, horizon, _ = roll["pred"].shape
    clips = np.repeat(np.arange(n_traj), horizon)
    low_pred = _lowest(data, "test", roll["pred"].reshape(n_traj * horizon, -1), clips).reshape(n_traj, horizon)
    low_true = _lowest(data, "test", roll["truth"].reshape(n_traj * horizon, -1), clips).reshape(n_traj, horizon)
    # One-step examples: was a rod touching the floor at the start or the end of the step?
    obs = data.split["test"].obs
    ex_clips = n_idx.cpu().numpy()
    low_t = _lowest(data, "test", obs[n_idx, t_idx].cpu().numpy(), ex_clips)
    low_t1 = _lowest(data, "test", obs[n_idx, t_idx + 1].cpu().numpy(), ex_clips)
    contact = torch.from_numpy(np.minimum(low_t, low_t1) < TOUCHING).to(sq.device)
    metrics = {
        "floor_contact_share": float(contact.float().mean()),
        "onestep_error_contact": _error(sq[contact], data) if contact.any() else float("nan"),
        "onestep_error_free": _error(sq[~contact], data) if (~contact).any() else float("nan"),
        "inside_floor_true": float((low_true < INSIDE_FLOOR).mean()),
        "inside_floor_pred": float((low_pred < INSIDE_FLOOR).mean()),
        "deepest_true_cm": float(-100 * low_true.min(1).mean()),
        "deepest_pred_cm": float(-100 * low_pred.min(1).mean()),
    }
    curves = {"inside_floor_pred": (low_pred < INSIDE_FLOOR).mean(0),
              "inside_floor_true": (low_true < INSIDE_FLOOR).mean(0),
              "lowest_pred_first": low_pred[0], "lowest_true_first": low_true[0]}
    return metrics, curves


TRACK_EXAMPLES = 500       # examples per set used to follow progress during training


class TrainingTracker:
    """Measures the model on fixed training and test examples during training (see model.train).

    For each checkpoint epoch it records
    * loss (train / test): the training objective on fixed examples with fixed noise, so the two
      are directly comparable and do not jump around from batch to batch;
    * 1-step error (train / test), as defined at the top of this file, on TRACK_EXAMPLES examples.
    It uses its own random generators, so tracking does not change the training run.
    """

    def __init__(self, data: PreparedData):
        self.data = data
        cfg = data.cfg
        self.history: dict[str, list[float]] = {k: [] for k in
                                                ("epoch", "train_loss", "test_loss", "train_onestep", "test_onestep")}
        self.sets = {}
        for i, split in enumerate(("train", "test")):
            n_idx, t_idx = data.examples(split, t_min=MAX_CONTEXT - 1)
            g = torch.Generator(device="cpu").manual_seed(cfg.seed + 21 + i)
            keep = torch.randperm(len(n_idx), generator=g)[:_example_cap(data, TRACK_EXAMPLES)].to(data.device)
            n_idx, t_idx = n_idx[keep], t_idx[keep]
            k = len(n_idx)
            self.sets[split] = dict(
                n=n_idx, t=t_idx,
                noise=torch.randn(k, data.obs_dim, generator=g).to(data.device),        # for 1-step predictions
                x0=torch.randn(k, data.obs_dim, generator=g).to(data.device),           # for the loss
                tau=torch.rand(k, 1, generator=g).to(data.device))

    @torch.no_grad()
    def __call__(self, epoch: int, model) -> None:
        was_training = model.training
        model.eval()
        self.history["epoch"].append(epoch)
        for split, s in self.sets.items():
            context, cond, target = self.data.batch(split, s["n"], s["t"])
            if model.flow:
                x0 = s["x0"] if self.data.cfg.noise_scale == 1.0 else s["x0"] * self.data.cfg.noise_scale
                x_tau = (1 - s["tau"]) * x0 + s["tau"] * target
                loss = torch.mean((model(context, cond, tau=s["tau"], x_tau=x_tau) - (target - x0)) ** 2)
            else:
                loss = torch.mean((model(context, cond) - target) ** 2)
            self.history[f"{split}_loss"].append(loss.item())
            sq = _one_step_sq_errors(self.data, model, split, s["n"], s["t"], s["noise"])
            self.history[f"{split}_onestep"].append(_error(sq, self.data))
        model.train(was_training)


def evaluate(data: PreparedData, model: DynamicsMLP) -> tuple[dict, dict]:
    """All scores (`metrics`) and per-step curves (`curves`) for every non-training set.

    The main test set has plain names; extra sets get a prefix, e.g. "g=20/onestep_error".
    """
    cfg = data.cfg
    model.eval()
    metrics: dict[str, float] = {}
    curves: dict[str, np.ndarray] = {}

    for split in data.split:
        if split == "train":
            continue
        p = "" if split == "test" else f"{split}/"

        # --- one-step error (fixed noise, so shuffled and unshuffled scores are comparable)
        n_idx, t_idx = _eval_examples(data, split, cfg.seed)
        g = torch.Generator(device="cpu").manual_seed(cfg.seed + 7)
        noise = torch.randn(len(n_idx), data.obs_dim, generator=g).to(data.device)
        sq = _one_step_sq_errors(data, model, split, n_idx, t_idx, noise)
        metrics[f"{p}onestep_error"] = base = _error(sq, data)
        obs = data.split[split].obs
        metrics[f"{p}copy_onestep_error"] = _error((obs[n_idx, t_idx + 1] - obs[n_idx, t_idx]) ** 2, data)

        roll = _rollout(data, model, split, cfg.seed + 5)
        for h in REPORT_HORIZONS:
            if h <= cfg.rollout_horizon:
                metrics[f"{p}rollout_error@{h}"] = float(roll["error"][:, h - 1].mean())
                metrics[f"{p}copy_rollout_error@{h}"] = float(roll["copy_error"][:, h - 1].mean())
        metrics[f"{p}valid_steps"] = float(roll["valid_steps"].mean())
        curves[f"{split}/rollout_error"] = roll["error"].mean(0)
        curves[f"{split}/copy_rollout_error"] = roll["copy_error"].mean(0)
        curves[f"{split}/pred_first"] = roll["pred"][0]     # first clip of every set, for videos
        curves[f"{split}/truth_first"] = roll["truth"][0]
        if split != "test":
            continue

        # --- shuffle test, one input group at a time
        perm = torch.randperm(len(n_idx), generator=torch.Generator().manual_seed(cfg.seed + 3)).to(data.device)
        for name, columns in data.groups.items():
            shuffled = _error(_one_step_sq_errors(data, model, split, n_idx, t_idx, noise, columns, perm), data)
            metrics[f"shuffle_{name}"] = shuffled / base - 1.0
        # --- same one-step error on training examples, for the train/test gap
        tr_n, tr_t = _eval_examples(data, "train", cfg.seed)
        tr_noise = torch.randn(len(tr_n), data.obs_dim, generator=g).to(data.device)
        metrics["train_onestep_error"] = train = _error(_one_step_sq_errors(data, model, "train", tr_n, tr_t, tr_noise), data)
        metrics["gap"] = base / max(train, 1e-12)
        # --- floor check
        if cfg.system == "ground" and cfg.obs == "full":
            floor_metrics, floor_curves = _floor_check(data, roll, sq, n_idx, t_idx)
            metrics.update(floor_metrics)
            curves.update({f"test/{k}": v for k, v in floor_curves.items()})
    return metrics, curves
