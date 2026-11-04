"""Step 1 of a run: get the data and turn it into model inputs.

1. `load_datasets` simulates clips ("trajectories") in MuJoCo, or loads them
   from the cache if an identical set was simulated before. Every clip gets its
   own random gravity and damping.
2. `PreparedData` scales everything to roughly mean 0 / std 1 (networks learn
   much better that way) and cuts the clips into training examples:

       input  = the last K snapshots (+ the facts the model is told, e.g. gravity)
       answer = the next snapshot, or the change to it ("delta")
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from experiments.model import Normalizer
from experiments.config import CONDITION_GROUPS, RunConfig
from experiments.systems import SimEnv


# ----------------------------------------------------------------------------- simulate

def collect_trajectories(system, count, links, n_traj, steps, gravity_range, damping_range,
                         seed_parts) -> dict[str, np.ndarray]:
    """Simulate `n_traj` clips of `steps` steps. Returns states [n_traj, steps+1, D]."""
    env = SimEnv(system, count, links)
    rng = np.random.default_rng(np.random.SeedSequence(list(seed_parts)))

    # One random value per clip, fixed for the whole clip.
    gravity = rng.uniform(*gravity_range, size=n_traj)
    damping = rng.uniform(*damping_range, size=n_traj)

    states = np.empty((n_traj, steps + 1, env.state_dim), dtype=np.float32)
    for n in range(n_traj):
        env.set_physics(gravity[n], damping[n])
        for _attempt in range(20):                     # redraw the start if the physics blew up
            states[n, 0] = env.reset(rng)
            for t in range(steps):
                states[n, t + 1] = env.step()
            if not env.unstable():
                break
        else:
            raise RuntimeError(f"{system}: simulation unstable for 20 starts in a row.")
    return {
        "states": states,
        "gravity": gravity.astype(np.float32),
        "damping": damping.astype(np.float32),
        "nq": np.array(env.nq),                        # number of position values (rest are speeds)
    }


def load_or_collect(cache_dir: Path, **kwargs) -> dict[str, np.ndarray]:
    """Same as collect_trajectories, but reuse a saved file if the exact same data was made before.

    This is why variants with the same world and data settings (and one seed) train on identical clips.
    """
    key = json.dumps(kwargs, sort_keys=True, default=list)
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    path = cache_dir / f"{kwargs['system']}_c{kwargs['count']}_l{kwargs['links']}_{digest}.npz"
    if path.exists():
        with np.load(path) as saved:
            return {k: saved[k] for k in saved.files}
    data = collect_trajectories(**kwargs)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(tmp, **data)
    os.replace(tmp, path)   # rename at the end, so parallel workers never read half-written files
    return data


def load_datasets(cfg: RunConfig, cache_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    """'train', 'test' (same gravity range, new clips) and optional 'g=<value>' test sets."""
    common = dict(system=cfg.system, count=cfg.count, links=cfg.links, steps=cfg.steps,
                  damping_range=tuple(cfg.damping_range))
    sets = {
        "train": load_or_collect(cache_dir, n_traj=cfg.n_train_traj, gravity_range=tuple(cfg.gravity_range),
                                 seed_parts=(cfg.seed, 0), **common),
        "test": load_or_collect(cache_dir, n_traj=cfg.n_test_traj, gravity_range=tuple(cfg.gravity_range),
                                seed_parts=(cfg.seed, 1), **common),
    }
    for i, g in enumerate(cfg.eval_gravities):
        sets[f"g={g:g}"] = load_or_collect(cache_dir, n_traj=cfg.n_test_traj, gravity_range=(g, g),
                                           seed_parts=(cfg.seed, 2, i), **common)
    return sets


# ----------------------------------------------------------------------------- model inputs

def observations(cfg: RunConfig, data: dict) -> np.ndarray:
    """What the model sees of each snapshot: everything, or only the positions (first nq values)."""
    states = data["states"]
    return states[..., : int(data["nq"])] if cfg.obs == "positions" else states


def condition_features(cfg: RunConfig, data: dict, token_seed: tuple[int, ...]):
    """The extra numbers fed to the model next to the snapshots.

    Returns an array [n_traj, T+1, C] and, for each group ("gravity", "tokens", ...),
    which of the C columns it occupies (used later to shuffle one group at a time).
    """
    n_traj, length = data["states"].shape[:2]
    per_clip = {
        "gravity": data["gravity"][:, None] / 9.81,    # 1 number, 1.0 = Earth
        "damping": data["damping"][:, None],           # 1 number
    }
    columns, groups, start = [], {}, 0

    def add(name: str, values: np.ndarray) -> None:
        nonlocal start
        columns.append(values)
        groups[name] = slice(start, start + values.shape[-1])
        start += values.shape[-1]

    for name in CONDITION_GROUPS:
        if name in cfg.condition:
            values = per_clip[name]                    # same value at every step of a clip
            add(name, np.broadcast_to(values[:, None, :], (n_traj, length, values.shape[1])))

    if cfg.nuisance_tokens:                            # meaningless random numbers
        rng = np.random.default_rng(np.random.SeedSequence(list(token_seed)))
        k = cfg.nuisance_tokens
        if cfg.nuisance_mode == "clip":                # one draw per clip (acts like a clip ID)
            tokens = np.broadcast_to(rng.uniform(-1, 1, size=(n_traj, 1, k)), (n_traj, length, k))
        else:                                          # fresh draw at every step (pure noise)
            tokens = rng.uniform(-1, 1, size=(n_traj, length, k))
        add("tokens", tokens)

    if not columns:
        return np.zeros((n_traj, length, 0), dtype=np.float32), groups
    return np.concatenate(columns, axis=-1).astype(np.float32), groups


@dataclass
class Split:
    """One data set (train, test, g=...) as tensors."""
    obs: torch.Tensor      # [n_traj, T+1, D] raw observations
    obs_n: torch.Tensor    # same, scaled
    cond_n: torch.Tensor   # [n_traj, T+1, C] scaled condition features


class PreparedData:
    """All data sets of one run, scaled with statistics of the training set."""

    def __init__(self, cfg: RunConfig, sets: dict, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.nq = int(sets["train"]["nq"])

        # Scalers for snapshots and for one-step changes, fitted on training data only.
        train_obs = observations(cfg, sets["train"])
        self.obs_dim = train_obs.shape[-1]
        flat = torch.from_numpy(train_obs.reshape(-1, self.obs_dim))
        self.obs_norm = Normalizer(flat)
        deltas = torch.from_numpy((train_obs[:, 1:] - train_obs[:, :-1]).reshape(-1, self.obs_dim))
        self.delta_norm = Normalizer(deltas)
        # Used by the scores: typical size of a one-step change, and of the state itself.
        self.delta_var = deltas.var(dim=0).clamp_min(1e-12).to(device)
        self.state_std = self.obs_norm.std.squeeze(0).to(device)

        # Same for the condition features (columns that never vary are left unscaled).
        train_cond, self.groups = condition_features(cfg, sets["train"], (cfg.seed, 11))
        self.cond_dim = train_cond.shape[-1]
        n_rows = train_cond.shape[0] * train_cond.shape[1]   # explicit: reshape(-1, 0) fails when C = 0
        cond_flat = torch.from_numpy(train_cond.reshape(n_rows, self.cond_dim))
        self.cond_mean = cond_flat.mean(0) if self.cond_dim else torch.zeros(0)
        std = cond_flat.std(0) if self.cond_dim else torch.zeros(0)
        self.cond_std = torch.where(std < 1e-6, torch.ones_like(std), std)

        self.split: dict[str, Split] = {}
        for i, (name, data) in enumerate(sets.items()):
            cond = train_cond if name == "train" else condition_features(cfg, data, (cfg.seed, 12, i))[0]
            obs = torch.from_numpy(observations(cfg, data))
            self.split[name] = Split(
                obs=obs.to(device),
                obs_n=self.obs_norm.encode(obs).to(device),
                cond_n=((torch.from_numpy(cond) - self.cond_mean) / self.cond_std).to(device),
            )
        # Offsets -K+1 ... 0: which past steps form the context window.
        self.offsets = torch.arange(-cfg.context_len + 1, 1, device=device)

    def examples(self, split: str, t_min: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Every (clip, step) pair with step >= t_min that still has a next step."""
        n_traj, length = self.split[split].obs.shape[:2]
        n_idx, t_idx = torch.meshgrid(torch.arange(n_traj, device=self.device),
                                      torch.arange(t_min, length - 1, device=self.device), indexing="ij")
        return n_idx.reshape(-1), t_idx.reshape(-1)

    def batch(self, split: str, n_idx: torch.Tensor, t_idx: torch.Tensor):
        """Model input and answer for the examples (clip n_idx, step t_idx).

        context = scaled snapshots t-K+1 ... t, flattened to one vector
        cond    = scaled condition features at step t
        target  = scaled snapshot t+1, or the scaled change from t to t+1
        """
        s = self.split[split]
        window = s.obs_n[n_idx[:, None], t_idx[:, None] + self.offsets]   # [B, K, D]
        context = window.reshape(len(n_idx), -1)                          # [B, K*D]
        cond = s.cond_n[n_idx, t_idx]
        if self.cfg.target == "state":
            target = s.obs_n[n_idx, t_idx + 1]
        else:
            target = self.delta_norm.encode(s.obs[n_idx, t_idx + 1] - s.obs[n_idx, t_idx])
        return context, cond, target

    def to_next_obs(self, pred_n: torch.Tensor, current_obs: torch.Tensor) -> torch.Tensor:
        """Turn a scaled model output back into the next raw observation."""
        if self.cfg.target == "state":
            return self.obs_norm.decode(pred_n)
        return current_obs + self.delta_norm.decode(pred_n)
