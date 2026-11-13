"""One complete run, start to finish. This is the file to read first.

    data  -> load_datasets   (data.py)      simulate clips in MuJoCo, or load them from the cache
          -> PreparedData    (data.py)      scale numbers, cut clips into (input, answer) examples
    model -> train           (model.py)     fit the network
    score -> evaluate        (evaluate.py)  one-step error, rollouts, shuffle test, train/test gap
    save  -> metrics.json + curves.npz in the run's folder
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.config import RunConfig
from experiments.data import PreparedData, load_datasets
from experiments.evaluate import TrainingTracker, evaluate
from experiments.model import train
from experiments.systems import SimEnv
from PIL import Image, ImageDraw, ImageFont


def run_experiment(cfg: RunConfig, out_dir: Path, cache_dir: Path, device: str = "cpu",
                   save_model: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # 1. data
    sets = load_datasets(cfg, cache_dir)
    t_data = time.time()
    data = PreparedData(cfg, sets, torch.device(device))

    # 2. model (progress on fixed train/test examples is recorded along the way)
    tracker = TrainingTracker(data)
    model, losses = train(cfg, data, on_checkpoint=tracker)
    t_train = time.time()

    # 3. score
    metrics, curves = evaluate(data, model)
    curves["train_loss"] = np.asarray(losses)
    curves.update({f"track/{k}": np.asarray(v) for k, v in tracker.history.items()})
    n_examples = int(data.examples("train", cfg.context_len - 1)[0].numel())
    updates_per_epoch = -(-n_examples // cfg.batch_size)

    # 4. save
    result = {
        **cfg.to_dict(),
        "run_id": cfg.run_id,
        "state_dim": int(sets["train"]["states"].shape[-1]),
        "obs_dim": data.obs_dim,
        "cond_dim": data.cond_dim,
        "n_params": sum(p.numel() for p in model.parameters()),
        "train_transitions": n_examples,
        "updates_per_epoch": updates_per_epoch,
        "updates": updates_per_epoch * cfg.epochs,              # total weight updates (training iterations)
        "examples_seen": n_examples * cfg.epochs,
        "final_train_loss": losses[-1],
        "final_test_loss": tracker.history["test_loss"][-1],
        **metrics,
        "seconds_data": t_data - t_start,
        "seconds_train": t_train - t_data,
        "seconds_eval": time.time() - t_train,
    }
    np.savez_compressed(out_dir / "curves.npz", **curves)
    if save_model:
        torch.save(model.state_dict(), out_dir / "model.pt")
    # metrics.json is written last: if it exists, the run finished (run.py skips finished runs).
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(result, f, indent=2, default=float)
    return result


def render_video(cfg: RunConfig, run_dir: Path, path: Path, fps: int = 25, split: str = "test",
                 cache_dir: Path | None = None) -> None:
    """Side-by-side video of the true (left) and predicted (right) rollout of the first clip of `split`.

    The rollout is 100 steps = 1 s of simulated time, shown at 25 frames/s (4x slow motion).
    `split` is "test" or an extra test set such as "g=25".
    """
    import mediapy

    with np.load(run_dir / "curves.npz") as c:
        truth, pred = c[f"{split}/truth_first"], c[f"{split}/pred_first"]
    if cfg.obs == "positions":
        # Positions-only models predict angles only; speeds do not change the picture.
        truth, pred = (np.concatenate([x, np.zeros_like(x)], axis=1) for x in (truth, pred))
    model_label = "Flow-matching model" if cfg.model == "fm" else "Regression model"
    if cfg.target == "state":
        model_label += " (next state)"
    shape = {}
    if cfg.length_range is not None or cfg.mass_range is not None:
        # Every clip has its own rods: draw the ones of the clip shown (first clip of `split`).
        clip = load_datasets(cfg, cache_dir or run_dir.parent.parent / "datasets")[split]
        shape = {k: clip[k][0] for k in ("length", "mass") if k in clip}
    env = SimEnv(cfg.system, cfg.count, cfg.links, img_size=(640, 400),
                 lengths=shape.get("length"), masses=shape.get("mass"))
    font = ImageFont.load_default(size=22)
    path.parent.mkdir(parents=True, exist_ok=True)
    with mediapy.VideoWriter(str(path), shape=(400, 1280), fps=fps) as writer:
        for step in range(len(truth)):
            panels = []
            for state, label in ((truth[step], "MuJoCo (truth)"), (pred[step], model_label)):
                image = Image.fromarray(env.render(state))
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, image.width, 34), fill=(24, 32, 38))
                draw.text((14, 17), label, fill=(255, 255, 255), anchor="lm", font=font)
                draw.text((image.width - 14, 17), f"t = {step * 0.01:.2f} s", fill=(190, 200, 205), anchor="rm", font=font)
                panels.append(np.asarray(image))
            writer.add_image(np.concatenate(panels, axis=1))
