"""The settings card for one run: simulate data, train one model, score it.

Everything that can differ between runs is a field here, so two runs with equal
settings give equal results. plan.py lists the settings of every experiment.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

# Facts about each clip the model can be told about (see data.py).
# "length" and "mass" are per unit (one number per pendulum), the others one number per clip.
CONDITION_GROUPS = ("gravity", "damping", "length", "mass")
UNIT_GROUPS = ("length", "mass")

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
    length_range: tuple[float, float] | None = None    # pendulum worlds: each unit gets its own rod length (m)
    mass_range: tuple[float, float] | None = None      # ... and its own mass, as a factor of the normal mass
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
    arch: str = "flat"                # "flat" = one network on the whole state vector,
                                      # "units" = one shared network applied to each unit (pendulum / chain)

    # --- scoring
    inference_steps: int = 20         # Euler steps from noise to prediction (flow matching)
    noise_scale: float = 1.0          # flow matching starts from noise ~ N(0, noise_scale^2) (in scaled units)
    input_noise: float = 0.0          # training only: Gaussian noise added to the input states (fraction of their spread)
    rollout_start: int = 10           # rollouts start from the true state at this step...
    rollout_horizon: int = 100        # ...and run on the model's own predictions for this many steps

    def __post_init__(self):
        checks = [
            (self.obs in ("full", "positions"), "obs must be 'full' or 'positions'."),
            (self.target in ("state", "delta"), "target must be 'state' or 'delta'."),
            (self.model in ("fm", "regression"), "model must be 'fm' or 'regression'."),
            (self.nuisance_mode in ("clip", "step"), "nuisance_mode must be 'clip' or 'step'."),
            (self.arch in ("flat", "units"), "arch must be 'flat' or 'units'."),
            (self.arch == "flat" or (self.obs == "full" and self.system in ("pendulum", "ground")),
             "arch='units' needs full observations of a pendulum world."),
            ((self.length_range is None and self.mass_range is None) or self.system in ("pendulum", "ground"),
             "length_range / mass_range only exist for pendulum worlds."),
            (not ({"length", "mass"} & set(self.condition)) or
             all(getattr(self, f"{g}_range") is not None for g in ("length", "mass") if g in self.condition),
             "telling the model length/mass needs length_range/mass_range."),
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
