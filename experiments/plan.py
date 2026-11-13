"""The experiment plan: every variant that is trained, and the 14 experiments that compare them.

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

# Week 3
LONG = 300                                   # epochs: 300 x 79 = 23,700 weight updates (was 60 x 79 = 4,740)
UNIT_COUNTS = (8, 16, 32, 64, 128, 256, 512)  # separate pendulums: D = 16 ... 1024
CHAIN_COUNTS = (2, 4, 8, 16, 32, 64, 128)    # 4-link chains:      D = 16 ... 1024
LENGTHS = (0.2, 0.45)                        # rod length in metres (normal: 0.3)
MASSES = (0.5, 2.0)                          # mass as a factor of the normal rod mass
PARAM_SETTINGS = {                           # what the model is told besides gravity
    "hidden": dict(),
    "length": dict(condition=("gravity", "length")),
    "told": dict(condition=("gravity", "length", "mass")),
    "hidden_K3": dict(context_len=3),        # not told, but sees 3 past states and can work it out
}

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
    # --- scaling the state: D = 16, 32, 64, 128 (hang_P8 and dim_L4x2 are the D = 16 points)
    **{f"dim_P{p}": dict(count=p) for p in (16, 32, 64)},
    **{f"dim_L4x{c}": dict(count=c, links=4) for c in (2, 4, 8, 16)},

    # ===== week 3: longer training is the new standard (300 epochs instead of 60)
    # --- flat network (one vector of D numbers), D = 16 ... 512
    **{f"flat_P{p}": dict(count=p, epochs=LONG) for p in UNIT_COUNTS},
    **{f"flat_L4x{c}": dict(count=c, links=4, epochs=LONG) for c in CHAIN_COUNTS},
    # --- the same with one shared network per unit (pendulum or 4-link chain)
    **{f"units_P{p}": dict(count=p, epochs=LONG, arch="units") for p in UNIT_COUNTS},
    **{f"units_L4x{c}": dict(count=c, links=4, epochs=LONG, arch="units") for c in CHAIN_COUNTS},
    # --- a flat network wide enough for D = 512 (a 256-wide last layer can only make 256 independent outputs)
    "wide_P256": dict(count=256, epochs=LONG, hidden=1024),
    "wide_L4x64": dict(count=64, links=4, epochs=LONG, hidden=1024),
    "wide_P512": dict(count=512, epochs=LONG, hidden=1024),
    # --- training length: 1000 epochs (the 60 and 300 epoch points are dim_*/hang_P8 and flat_*)
    "flat_P8_e1000": dict(epochs=1000),
    "flat_P64_e1000": dict(count=64, epochs=1000),
    "flat_L4x16_e1000": dict(count=16, links=4, epochs=1000),
    "flat_P512_e60": dict(count=512, epochs=60),
    "flat_P512_e1000": dict(count=512, epochs=1000),
    # --- training length, flat vs per unit: chains at D = 256 and 1024, and the per-unit network at 60 / 1000 epochs
    **{f"flat_L4x{c}_e{e}": dict(count=c, links=4, epochs=e) for c in (32, 128) for e in (60, 1000)},
    **{f"units_P{p}_e{e}": dict(count=p, epochs=e, arch="units") for p in (8, 64, 512) for e in (60, 1000)},
    **{f"units_L4x{c}_e{e}": dict(count=c, links=4, epochs=e, arch="units") for c in (32, 128) for e in (60, 1000)},
    # --- physical parameters: every pendulum gets its own rod length and mass
    **{f"par_P{p}_{label}": dict(count=p, epochs=LONG, length_range=LENGTHS, mass_range=MASSES, **extra)
       for p in (8, 64) for label, extra in PARAM_SETTINGS.items()},
    "par_P64_units_told": dict(count=64, epochs=LONG, length_range=LENGTHS, mass_range=MASSES, arch="units",
                               condition=("gravity", "length", "mass")),
    "par_P64_units_hidden_K3": dict(count=64, epochs=LONG, length_range=LENGTHS, mass_range=MASSES, arch="units",
                                    context_len=3),
    # --- ground pendulums with rods of different length and mass: does telling them help with the floor?
    "gnd_P8": dict(system="ground", epochs=LONG),
    # --- noise levels (8 hanging pendulums, D = 16): size of the flow-matching start noise, and
    # noise added to the input states during training (noisy measurements)
    **{f"fmnoise_{s:g}": dict(epochs=LONG, noise_scale=s) for s in (0.1, 0.3, 3.0)},
    **{f"innoise_{s:g}": dict(epochs=LONG, input_noise=s) for s in (0.003, 0.03, 0.3)},
    "gnd_P8_par_hidden": dict(system="ground", epochs=LONG, length_range=LENGTHS, mass_range=MASSES),
    "gnd_P8_par_told": dict(system="ground", epochs=LONG, length_range=LENGTHS, mass_range=MASSES,
                            condition=("gravity", "length", "mass")),
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
    Experiment("dimension", "Scaling the state: 16 to 128 dimensions",
               "How does the model cope as the state grows from D = 16 to 128, with independent joints and with coupled ones?",
               {**{f"separate pendulums · D={2 * p}": ("hang_P8" if p == 8 else f"dim_P{p}") for p in (8, 16, 32, 64)},
                **{f"4-link chains · D={8 * c}": f"dim_L4x{c}" for c in (2, 4, 8, 16)}}),
]

DIMS = (16, 32, 64, 128, 256, 512, 1024)
EXPERIMENTS += [
    Experiment("scaling", "Scaling to 1024 dimensions: one big network vs one network per unit",
               "How far can the state grow before the model breaks, and does cutting it into units help?",
               {**{f"flat · separate pendulums · D={d}": f"flat_P{d // 2}" for d in DIMS},
                **{f"flat · 4-link chains · D={d}": f"flat_L4x{d // 8}" for d in DIMS},
                **{f"per unit · separate pendulums · D={d}": f"units_P{d // 2}" for d in DIMS},
                **{f"per unit · 4-link chains · D={d}": f"units_L4x{d // 8}" for d in DIMS},
                "flat, 4x wider · separate pendulums · D=512": "wide_P256",
                "flat, 4x wider · 4-link chains · D=512": "wide_L4x64",
                "flat, 4x wider · separate pendulums · D=1024": "wide_P512"}),
    Experiment("training", "Training longer: 60, 300 and 1000 epochs",
               "Does more training help, and when does the model start to memorise?",
               {**{f"8 pendulums (D=16) · {e} epochs": n for e, n in
                   ((60, "hang_P8"), (300, "flat_P8"), (1000, "flat_P8_e1000"))},
                **{f"64 pendulums (D=128) · {e} epochs": n for e, n in
                   ((60, "dim_P64"), (300, "flat_P64"), (1000, "flat_P64_e1000"))},
                **{f"16 chains of 4 (D=128) · {e} epochs": n for e, n in
                   ((60, "dim_L4x16"), (300, "flat_L4x16"), (1000, "flat_L4x16_e1000"))},
                **{f"512 pendulums (D=1024) · {e} epochs": n for e, n in
                   ((60, "flat_P512_e60"), (300, "flat_P512"), (1000, "flat_P512_e1000"))}}),
    Experiment("training_arch", "Training longer, flat vs per unit",
               "Does the per-unit network also memorise when trained longer, or only the flat one?",
               {f"{setup} · {arch} · {e} epochs": name
                for setup, flat, units in (
                    ("8 pendulums (D=16)", ("hang_P8", "flat_P8", "flat_P8_e1000"), "units_P8"),
                    ("64 pendulums (D=128)", ("dim_P64", "flat_P64", "flat_P64_e1000"), "units_P64"),
                    ("512 pendulums (D=1024)", ("flat_P512_e60", "flat_P512", "flat_P512_e1000"), "units_P512"),
                    ("32 chains of 4 (D=256)", ("flat_L4x32_e60", "flat_L4x32", "flat_L4x32_e1000"), "units_L4x32"),
                    ("128 chains of 4 (D=1024)", ("flat_L4x128_e60", "flat_L4x128", "flat_L4x128_e1000"), "units_L4x128"))
                for arch, names in (("flat", flat), ("per unit", (units + "_e60", units, units + "_e1000")))
                for e, name in zip((60, 300, 1000), names)}),
    Experiment("parameters", "Telling the model each pendulum's length and mass",
               "Which physical parameters does the model need to be told, and which can it work out itself?",
               {**{f"{p} pendulums (D={2 * p}) · {label}": name for p in (8, 64) for label, name in (
                   ("all rods equal", f"flat_P{p}"),
                   ("lengths & masses vary, not told", f"par_P{p}_hidden"),
                   ("told lengths", f"par_P{p}_length"),
                   ("told lengths & masses", f"par_P{p}_told"),
                   ("not told, 3 past states", f"par_P{p}_hidden_K3"))},
                "64 pendulums (D=128) · per unit, told lengths & masses": "par_P64_units_told",
                "64 pendulums (D=128) · per unit, not told, 3 past states": "par_P64_units_hidden_K3"}),
    Experiment("noise", "Noise levels",
               "How do the size of the flow-matching start noise and noise on the input states affect the model?",
               {**{f"start noise {s:g}": n for s, n in ((0.1, "fmnoise_0.1"), (0.3, "fmnoise_0.3"), (1.0, "flat_P8"), (3.0, "fmnoise_3"))},
                **{f"input noise {s:g}": n for s, n in ((0.0, "flat_P8"), (0.003, "innoise_0.003"), (0.03, "innoise_0.03"), (0.3, "innoise_0.3"))}}),
    Experiment("ground_params", "Ground pendulums: telling the model rod lengths and masses",
               "With rods of different length on the floor, does telling the model the lengths and masses help it respect the floor?",
               {"equal rods, 60 epochs (week 2)": "ground_P8", "equal rods": "gnd_P8",
                "different rods, not told": "gnd_P8_par_hidden", "different rods, told": "gnd_P8_par_told"}),
]

# Side-by-side videos (seed 0). "variant" shows its first test clip; "variant@g=25" shows the
# first clip of an extra test set (see eval_gravities).
VIDEOS = ("gnd_P8_par_hidden", "gnd_P8_par_told", "hang_P8", "ground_P8", "hang_L8", "ground_L8", "spring_D16", "cartpole_D16", "balls_D16",
          "hang_P8_state", "hang_P8_regression", "hang_P8_regression_state",
          "g_hidden", "g_fixed_hidden",
          "hang_P8@g=25", "g_wide_told@g=25", "g_fixed_hidden@g=25",
          "ctx_pos_told_K1", "ctx_pos_told_K2", "ctx_full_hidden_K5",
          "dim_P64", "dim_L4x16",
          "flat_L4x16", "units_L4x16", "flat_P256", "units_P256", "par_P8_hidden", "par_P8_told",
          "flat_P512", "units_P512", "flat_L4x128", "units_L4x128")


def video_jobs() -> list[tuple[str, str, str]]:
    """(variant, test set, file name) for every entry of VIDEOS, e.g. ("hang_P8", "g=25", "hang_P8__g25.mp4")."""
    jobs = []
    for entry in VIDEOS:
        name, _, split = entry.partition("@")
        suffix = "__" + split.replace("=", "").replace(".", "_") if split else ""
        jobs.append((name, split or "test", f"{name}{suffix}.mp4"))
    return jobs


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
