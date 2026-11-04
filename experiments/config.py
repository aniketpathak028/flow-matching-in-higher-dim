"""The settings card for one run: simulate data, train one model, score it.

Everything that can differ between runs is a field here, so two runs with equal
settings give equal results. plan.py lists the settings of every experiment.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

# Facts about each clip the model can be told about (see data.py).
CONDITION_GROUPS = ("gravity", "damping")

# Largest history length. One-step scoring always starts at this step, so models
# that look back 1 or 5 steps are graded on exactly the same transitions.
MAX_CONTEXT = 5


@dataclass(frozen=True)
class RunConfig:
    name: str                         # variant name, e.g. "hang_P8"; the run folder is <name>__seed<k>
    seed: int                         # new seed = new clips + new network starting weights

    # --- which world (see systems.py)
    system: str = "pendulum"          # pendulum | ground | spring_chain | cartpole | balls
    count: int = 8                    # independent copies (pendulums, carts, balls...)
    links: int = 1                    # joints per copy (1 = single pendulum, 8 = 8-link chain)

    # --- what the model sees and predicts
    obs: str = "full"                 # "full" = positions + speeds, "positions" = positions only
    context_len: int = 1              # number of past states the model sees (K)
    target: str = "delta"             # "delta" = predict the change, "state" = predict the next state
    model: str = "fm"                 # "fm" = flow matching, "regression" = plain network
    condition: tuple[str, ...] = ("gravity",)   # which of CONDITION_GROUPS the model is told
    nuisance_tokens: int = 0          # meaningless random numbers added to the input
    nuisance_mode: str = "clip"       # "clip" = drawn once per clip, "step" = redrawn every step

    # --- data
    gravity_range: tuple[float, float] = (4.0, 15.0)   # each clip gets a random gravity in this range
    damping_range: tuple[float, float] = (0.0, 0.0)    # ... and a random joint damping in this range
    eval_gravities: tuple[float, ...] = ()             # extra test sets at these exact gravities
    n_train_traj: int = 200
    n_test_traj: int = 50
    steps: int = 200                  # steps per clip (0.01 s each)

    # --- training
    epochs: int = 60
    batch_size: int = 512
    lr: float = 2e-3
    hidden: int = 256
    blocks: int = 3

    # --- scoring
    inference_steps: int = 20         # Euler steps from noise to prediction (flow matching)
    rollout_start: int = 10           # rollouts start from the true state at this step...
    rollout_horizon: int = 100        # ...and run on the model's own predictions for this many steps

    def __post_init__(self):
        checks = [
            (self.obs in ("full", "positions"), "obs must be 'full' or 'positions'."),
            (self.target in ("state", "delta"), "target must be 'state' or 'delta'."),
            (self.model in ("fm", "regression"), "model must be 'fm' or 'regression'."),
            (self.nuisance_mode in ("clip", "step"), "nuisance_mode must be 'clip' or 'step'."),
            (set(self.condition) <= set(CONDITION_GROUPS), f"condition must be a subset of {CONDITION_GROUPS}."),
            (1 <= self.context_len <= MAX_CONTEXT, f"context_len must be in [1, {MAX_CONTEXT}]."),
            (self.rollout_start >= self.context_len - 1, "rollout_start must leave room for the context window."),
            (self.rollout_start + self.rollout_horizon <= self.steps, "steps must cover the rollout."),
        ]
        for ok, message in checks:
            if not ok:
                raise ValueError(message)

    @property
    def run_id(self) -> str:
        return f"{self.name}__seed{self.seed}"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
