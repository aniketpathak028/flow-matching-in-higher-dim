"""The experiment plan: every variant that is trained, and the 10 experiments that compare them.

A variant is a name plus the settings that differ from RunConfig's defaults
(8 hanging single pendulums, gravity told, flow matching, predict the change).
Each variant is trained once per seed. Experiments only *compare* variants, so a
variant used by several experiments (e.g. "hang_P8") is trained only once.

Within one seed, variants of the same world and data settings train on the same
clips (the data cache makes sure of that), so differences come from the setting
being compared, not from luck in the data.
"""
from __future__ import annotations

from dataclasses import dataclass

from experiments.config import RunConfig

GRAVITY_TESTS = (2.0, 4.0, 9.81, 15.0, 20.0, 25.0)   # 2, 20 and 25 lie outside the 4-15 training range

VARIANTS: dict[str, dict] = {
    # --- hanging pendulums: more pendulums (P) vs one longer chain (L), D = 2 x joints
    "hang_P1": dict(count=1),
    "hang_P2": dict(count=2),
    "hang_P4": dict(count=4),
    "hang_P8": dict(eval_gravities=GRAVITY_TESTS),     # the default setup, D = 16
    "hang_L2": dict(count=1, links=2),
    "hang_L4": dict(count=1, links=4),
    "hang_L8": dict(count=1, links=8),
    # --- the same on the ground (rods start upright and fall onto a solid floor)
    "ground_P1": dict(system="ground", count=1),
    "ground_P2": dict(system="ground", count=2),
    "ground_P4": dict(system="ground", count=4),
    "ground_P8": dict(system="ground"),
    "ground_L2": dict(system="ground", count=1, links=2),
    "ground_L4": dict(system="ground", count=1, links=4),
    "ground_L8": dict(system="ground", count=1, links=8),
    # --- design choices
    "hang_P8_state": dict(target="state"),
    "hang_P8_regression": dict(model="regression"),
    "hang_P8_regression_state": dict(model="regression", target="state"),
    # --- gravity
    "g_hidden": dict(condition=(), eval_gravities=GRAVITY_TESTS),
    "g_fixed_hidden": dict(condition=(), gravity_range=(9.81, 9.81), eval_gravities=GRAVITY_TESTS),
    "g_wide_told": dict(gravity_range=(2.0, 25.0), eval_gravities=GRAVITY_TESTS),
    # --- damping (joint friction); 0.15 brakes a swinging rod within about a second
    "damp_hidden": dict(damping_range=(0.0, 0.15)),
    "damp_told": dict(damping_range=(0.0, 0.15), condition=("gravity", "damping")),
    "damp_fixed_hidden": dict(damping_range=(0.075, 0.075)),
    # --- context: K past states; "pos" = the model sees positions only, no speeds
    **{f"ctx_full_told_K{k}": dict(context_len=k) for k in (2, 3, 5)},
    **{f"ctx_full_hidden_K{k}": dict(context_len=k, condition=()) for k in (2, 3, 5)},
    **{f"ctx_pos_told_K{k}": dict(context_len=k, obs="positions") for k in (1, 2, 3, 5)},
    # --- nuisance tokens: meaningless random numbers added to the input
    "tok8_step": dict(nuisance_tokens=8, nuisance_mode="step"),
    "tok8_clip": dict(nuisance_tokens=8, nuisance_mode="clip"),
    "tok32_clip": dict(nuisance_tokens=32, nuisance_mode="clip"),
    "tok32_clip_hidden": dict(nuisance_tokens=32, nuisance_mode="clip", condition=()),
    # --- other worlds at the same D = 16
    "spring_D16": dict(system="spring_chain", count=4, links=2),
    "cartpole_D16": dict(system="cartpole", count=4),
    "balls_D16": dict(system="balls", count=4),
}


@dataclass(frozen=True)
class Experiment:
    key: str
    title: str
    question: str
    variants: dict[str, str]      # label shown in the report -> variant name


EXPERIMENTS = [
    Experiment("setup", "Hanging vs ground pendulum",
               "How much harder is the original ground setup than a freely hanging pendulum?",
               {"hanging": "hang_P8", "ground": "ground_P8"}),
    Experiment("collisions", "Does the model learn the floor?",
               "In the ground setup, do predicted rods respect the floor, and are contact steps the hard part?",
               {"8 pendulums": "ground_P8", "chain of 2": "ground_L2", "chain of 4": "ground_L4",
                "chain of 8": "ground_L8"}),
    Experiment("size", "More pendulums vs longer chains",
               "Is it the size of the state (D) or the coupling between joints that makes prediction hard?",
               {f"{setup} · {label}": f"{prefix}_{suffix}"
                for setup, prefix in (("hanging", "hang"), ("ground", "ground"))
                for label, suffix in (("1 pendulum", "P1"), ("2 pendulums", "P2"), ("4 pendulums", "P4"),
                                      ("8 pendulums", "P8"), ("chain of 2", "L2"), ("chain of 4", "L4"),
                                      ("chain of 8", "L8"))}),
    Experiment("design", "Flow matching vs regression, change vs next state",
               "Which model and which prediction target work best on the hanging setup?",
               {"flow matching · change": "hang_P8", "flow matching · next state": "hang_P8_state",
                "regression · change": "hang_P8_regression", "regression · next state": "hang_P8_regression_state"}),
    Experiment("gravity", "Telling the model gravity",
               "When does it matter whether the model is told the gravity of a clip?",
               {"gravity fixed, not told": "g_fixed_hidden", "gravity varies, not told": "g_hidden",
                "gravity varies, told": "hang_P8"}),
    Experiment("extrapolation", "Gravities outside the training range",
               "Does a model trained on gravity 4-15 work at gravity 2, 20 or 25?",
               {"trained 4-15, told": "hang_P8", "trained 4-15, not told": "g_hidden",
                "trained 2-25, told": "g_wide_told", "trained at 9.81 only": "g_fixed_hidden"}),
    Experiment("damping", "Joint damping",
               "Does friction that varies between clips hurt, and does telling the model fix it?",
               {"no damping": "hang_P8", "same damping every clip, not told": "damp_fixed_hidden",
                "damping varies, not told": "damp_hidden", "damping varies, told": "damp_told"}),
    Experiment("context", "Past states as context",
               "Do 2-5 past states help, and when?",
               {**{f"full · told · K={k}": ("hang_P8" if k == 1 else f"ctx_full_told_K{k}") for k in (1, 2, 3, 5)},
                **{f"full · not told · K={k}": ("g_hidden" if k == 1 else f"ctx_full_hidden_K{k}") for k in (1, 2, 3, 5)},
                **{f"positions · told · K={k}": f"ctx_pos_told_K{k}" for k in (1, 2, 3, 5)}}),
    Experiment("nuisance", "Random nuisance tokens",
               "Does the model ignore meaningless random inputs, or learn from them?",
               {"no tokens": "hang_P8", "8 tokens, new every step": "tok8_step",
                "8 tokens, fixed per clip": "tok8_clip", "32 tokens, fixed per clip": "tok32_clip",
                "no tokens, gravity not told": "g_hidden", "32 tokens per clip, gravity not told": "tok32_clip_hidden"}),
    Experiment("systems", "Other worlds",
               "How do springs, cart-poles and bouncing balls compare to the hanging pendulum at the same D?",
               {"spring chains": "spring_D16", "hanging pendulums": "hang_P8", "cart-poles": "cartpole_D16",
                "bouncing balls": "balls_D16"}),
]

# Variants that get a side-by-side video (seed 0).
VIDEOS = ("hang_P8", "ground_P8", "hang_L8", "ground_L8", "spring_D16", "cartpole_D16", "balls_D16")


def build_runs(seeds: list[int], only: list[str] | None = None, overrides: dict | None = None,
               quick: bool = False) -> list[RunConfig]:
    """One RunConfig per (variant, seed) used by the selected experiments (default: all)."""
    chosen = [e for e in EXPERIMENTS if not only or e.key in only]
    if only and len(chosen) != len(only):
        raise ValueError(f"Unknown experiment in {only}. Choose from {[e.key for e in EXPERIMENTS]}.")
    names = list(dict.fromkeys(name for e in chosen for name in e.variants.values()))
    configs = []
    for name in names:
        params = {**VARIANTS[name], **(overrides or {})}
        if quick:
            params.update(epochs=2, n_train_traj=20, n_test_traj=8, steps=60, rollout_horizon=40)
        configs += [RunConfig(name=name, seed=seed, **params) for seed in seeds]
    return configs
