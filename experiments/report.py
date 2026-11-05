"""Collect finished runs into figures, tables and a summary.

Nothing here trains or simulates. It reads results/runs/*/metrics.json and
curves.npz, averages over seeds (mean and 95% confidence interval) and writes:

    results/REPORT.md          one section per experiment: question, figure, table
    results/figures/<key>.png  one figure per experiment
    results/summary.json       the same numbers, for the presentation page

    python -m experiments.report --out results
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.plan import EXPERIMENTS, Experiment

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}   # two-sided 95% Student-t, by degrees of freedom
plt.rcParams.update({
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "font.size": 10, "legend.frameon": False, "axes.titlesize": 11,
    "axes.titleweight": "bold",
})

COLUMNS = [("onestep_error", "1-step error"), ("rollout_error@25", "rollout error @25"),
           ("valid_steps", "valid steps"), ("gap", "test/train gap")]
EXTRA_COLUMNS = {
    "setup": [("floor_contact_share", "steps touching floor")],
    "collisions": [("onestep_error_free", "1-step, no contact"), ("onestep_error_contact", "1-step, contact"),
                   ("inside_floor_true", ">5 cm in floor (truth)"), ("inside_floor_pred", ">5 cm in floor (model)"),
                   ("deepest_pred_cm", "deepest, model (cm)")],
    "gravity": [("shuffle_gravity", "shuffle gravity")],
    "extrapolation": [(f"g={g:g}/onestep_error", f"1-step @ g={g:g}") for g in (2, 9.81, 20, 25)],
    "damping": [("shuffle_damping", "shuffle damping")],
    "context": [("shuffle_gravity", "shuffle gravity")],
    "nuisance": [("shuffle_tokens", "shuffle tokens"), ("shuffle_gravity", "shuffle gravity")],
    "systems": [("copy_onestep_error", "copy baseline 1-step")],
}
CURVES = ("test/rollout_error", "test/copy_rollout_error", "test/inside_floor_pred", "test/inside_floor_true")


# ----------------------------------------------------------------------------- loading

def load(out_root: Path) -> dict[str, dict]:
    """Per variant: settings of its first seed, every metric as [mean, ci95, n], mean curves."""
    runs = defaultdict(list)
    for path in sorted((out_root / "runs").glob("*/metrics.json")):
        run = json.loads(path.read_text())
        runs[run["name"]].append((run, path.parent))
    table = {}
    for name, items in runs.items():
        first = items[0][0]
        entry = {"settings": {k: v for k, v in first.items() if not isinstance(v, float)}, "metrics": {}, "curves": {}}
        for key, value in first.items():
            if isinstance(value, float):
                values = np.array([r.get(key, np.nan) for r, _ in items], dtype=float)
                n = len(values)
                std = values.std(ddof=1) if n > 1 else 0.0
                entry["metrics"][key] = [float(values.mean()), float(T95.get(n - 1, 1.96) * std / np.sqrt(n)), n]
        for curve in CURVES:
            stack = []
            for _, run_dir in items:
                with np.load(run_dir / "curves.npz") as c:
                    if curve in c:
                        stack.append(c[curve])
            if stack:
                entry["curves"][curve.split("/", 1)[1]] = np.mean(stack, axis=0).round(5).tolist()
        table[name] = entry
    return table


def value(table: dict, name: str, key: str) -> tuple[float, float]:
    m = table.get(name, {}).get("metrics", {}).get(key)
    return (m[0], m[1]) if m else (np.nan, 0.0)


# ----------------------------------------------------------------------------- figures

def _bars(ax, table, exp: Experiment, key: str, title: str, log: bool = True) -> None:
    labels = list(exp.variants)
    means, cis = zip(*(value(table, exp.variants[label], key) for label in labels))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, means, xerr=cis, color=SERIES[0], height=0.6, capsize=3, error_kw={"elinewidth": 1})
    ax.set_yticks(y, labels)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    if log:
        ax.set_xscale("log")


def _lines(ax, table, groups: dict[str, list[tuple[float, str]]], key: str, title: str, xlabel: str,
           log: bool = True) -> None:
    for color, (label, points) in zip(SERIES, groups.items()):
        xs = [x for x, _ in points]
        means, cis = zip(*(value(table, name, key) for _, name in points))
        ax.errorbar(xs, means, yerr=cis, label=label, color=color, marker="o", capsize=3, elinewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if log:
        ax.set_yscale("log")


def _rollout_curves(ax, table, exp: Experiment) -> None:
    for color, (label, name) in zip(SERIES, exp.variants.items()):
        curves = table.get(name, {}).get("curves", {})
        if "rollout_error" in curves:
            steps = np.arange(1, len(curves["rollout_error"]) + 1)
            ax.plot(steps, curves["rollout_error"], color=color, label=label)
            ax.plot(steps, curves["copy_rollout_error"], color=color, linestyle=":", linewidth=1)
    ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1)
    ax.set(yscale="log", xlabel="rollout step (0.01 s each)", title="Rollout error (dotted: 'nothing moves')")
    ax.legend(fontsize=8)


def figure(table: dict, exp: Experiment, path: Path) -> None:
    key = exp.key
    if key == "size":
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        groups = {}
        for setup, prefix in (("hanging", "hang"), ("ground", "ground")):
            groups[f"{setup}, more pendulums"] = [(2 * p, f"{prefix}_P{p}") for p in (1, 2, 4, 8)]
            groups[f"{setup}, longer chain"] = [(2, f"{prefix}_P1")] + [(2 * l, f"{prefix}_L{l}") for l in (2, 4, 8)]
        _lines(axes[0], table, groups, "onestep_error", "1-step error", "state dimension D")
        _lines(axes[1], table, groups, "valid_steps", "Valid rollout steps (of 100)", "state dimension D", log=False)
        for ax in axes:
            ax.set_xscale("log", base=2)
        axes[0].legend(fontsize=8)
    elif key == "dimension":
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        groups = defaultdict(list)
        for label, name in exp.variants.items():
            series, d = label.rsplit(" · D=", 1)
            groups[series].append((int(d), name))
        _lines(axes[0], table, groups, "onestep_error", "1-step error", "state dimension D")
        _lines(axes[1], table, groups, "valid_steps", "Valid rollout steps (of 100)", "state dimension D", log=False)
        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_xticks([16, 32, 64, 128], ["16", "32", "64", "128"])
        axes[0].legend(fontsize=8)
    elif key == "context":
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        groups = defaultdict(list)
        for label, name in exp.variants.items():
            series, k = label.rsplit(" · K=", 1)
            groups[series].append((int(k), name))
        _lines(axes[0], table, groups, "onestep_error", "1-step error", "past states K")
        _lines(axes[1], table, groups, "valid_steps", "Valid rollout steps (of 100)", "past states K", log=False)
        axes[0].legend(fontsize=8)
    elif key == "extrapolation":
        fig, ax = plt.subplots(figsize=(7, 4))
        gs = [2.0, 4.0, 9.81, 15.0, 20.0, 25.0]
        for color, (label, name) in zip(SERIES, exp.variants.items()):
            means, cis = zip(*(value(table, name, f"g={g:g}/onestep_error") for g in gs))
            ax.errorbar(gs, means, yerr=cis, label=label, color=color, marker="o", capsize=3, elinewidth=1)
        ax.axvspan(4, 15, color=GRID, alpha=0.7, linewidth=0, label="training range 4-15")
        ax.set(yscale="log", xlabel="test gravity (m/s²)", title="1-step error at fixed test gravities")
        ax.legend(fontsize=8)
    elif key == "collisions":
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        labels = list(exp.variants)
        x = np.arange(len(labels))
        for i, (metric, label) in enumerate([("onestep_error_free", "no floor contact"),
                                              ("onestep_error_contact", "floor contact")]):
            means, cis = zip(*(value(table, exp.variants[l], metric) for l in labels))
            axes[0].bar(x + (i - 0.5) * 0.38, means, 0.36, yerr=cis, color=SERIES[i], label=label, capsize=3)
        axes[0].set(yscale="log", title="1-step error by step type")
        for i, (metric, label) in enumerate([("inside_floor_true", "truth"), ("inside_floor_pred", "model")]):
            means, cis = zip(*(value(table, exp.variants[l], metric) for l in labels))
            axes[1].bar(x + (i - 0.5) * 0.38, np.array(means) * 100, 0.36, yerr=np.array(cis) * 100,
                        color=SERIES[i + 2], label=label, capsize=3)
        axes[1].set(title="Rollout states with a rod >5 cm inside the floor (%)")
        for ax in axes:
            ax.set_xticks(x, labels)
            ax.grid(axis="x", visible=False)
            ax.legend(fontsize=8)
    elif key in ("setup", "systems"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        _bars(axes[0], table, exp, "onestep_error", "1-step error")
        _rollout_curves(axes[1], table, exp)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(exp.variants) + 1.6))
        _bars(axes[0], table, exp, "onestep_error", "1-step error")
        if key == "nuisance":
            _bars(axes[1], table, exp, "gap", "Test/train gap (≫1 = memorised)")
        else:
            _bars(axes[1], table, exp, "valid_steps", "Valid rollout steps (of 100)", log=False)
            axes[1].set_yticklabels([])
    fig.suptitle(exp.title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- markdown

def _fmt(m: tuple[float, float]) -> str:
    mean, ci = m
    if not np.isfinite(mean):
        return "–"
    spec = ".2e" if 0 < abs(mean) < 0.01 else (".3f" if abs(mean) < 10 else ".1f")
    return f"{mean:{spec}} ± {ci:{spec}}"


def section(table: dict, exp: Experiment) -> str:
    columns = COLUMNS + EXTRA_COLUMNS.get(exp.key, [])
    lines = [f"## {exp.title}", "", f"*{exp.question}*", "", f"![{exp.key}](figures/{exp.key}.png)", "",
             "| variant | run | D | " + " | ".join(t for _, t in columns) + " |",
             "|---|---|---|" + "---|" * len(columns)]
    for label, name in exp.variants.items():
        dim = table.get(name, {}).get("settings", {}).get("state_dim", "–")
        lines.append(f"| {label} | `{name}` | {dim} | " + " | ".join(_fmt(value(table, name, k)) for k, _ in columns) + " |")
    return "\n".join(lines) + "\n"


def build_report(out_root: Path) -> Path:
    table = load(out_root)
    (out_root / "figures").mkdir(parents=True, exist_ok=True)
    sections = []
    for exp in EXPERIMENTS:
        if not any(name in table for name in exp.variants.values()):
            continue
        figure(table, exp, out_root / "figures" / f"{exp.key}.png")
        sections.append(section(table, exp))
    header = (
        "# Flow-matching experiment report\n\n"
        "Mean ± 95% confidence interval over seeds (each seed = new clips + new network start).\n\n"
        "* **1-step error**: squared one-step error ÷ variance of the true one-step change "
        "(1.0 = no better than guessing the average change).\n"
        "* **rollout error @h**: error after h steps on the model's own predictions, in units of the state's spread "
        "(0.5 = clearly off).\n"
        "* **valid steps**: steps until the rollout error first passes 0.5 (max 100).\n"
        "* **test/train gap**: test 1-step error ÷ training 1-step error (≫1 = memorised).\n"
        "* **shuffle X**: relative growth of the 1-step error when X is scrambled between test examples "
        "(≈0 = X is ignored).\n\n")
    path = out_root / "REPORT.md"
    path.write_text(header + "\n".join(sections))
    summary = {"experiments": [dict(key=e.key, title=e.title, question=e.question, variants=e.variants)
                               for e in EXPERIMENTS],
               "variants": table}
    (out_root / "summary.json").write_text(json.dumps(summary, default=float))
    print(f"Report written to '{path}'.")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build figures and REPORT.md from finished runs.")
    parser.add_argument("--out", default="results")
    build_report(Path(parser.parse_args().out))
