# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A university study project. A flow-matching model (a residual MLP) learns to predict the next MuJoCo state `[qpos, qvel]` of simulated worlds, mainly pendulums. D = 2 × number of joints; for hinge joints `qpos` is the angle.

Everything lives in one framework, `experiments/`. The earlier standalone scripts (`fm_model.py`, `mujoco_pipeline.py`) were folded into it: their ground pendulum is `system="ground"`, and their network pieces are in `model.py`. `experiments/README.md` has a plain-language walkthrough. The user prefers plain-language explanations.

## Commands

Setup: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`. Videos also need ffmpeg.

Run from the repo root as modules:

```bash
python -m experiments.run                                  # all 17 experiments × seeds 0 1 2 (348 runs, several days on a laptop), resumable
python -m experiments.run --only setup collisions          # selected experiments (keys in plan.EXPERIMENTS)
python -m experiments.run --quick --out <dir> --seeds 0    # smoke test, under a minute
python -m experiments.run --set epochs=100                 # override any RunConfig field
python -m experiments.run --dry-run
python -m experiments.report --out results                 # rebuild REPORT.md, figures, summary.json only
python -m experiments.site                                 # copy summary.json + videos into docs/ (the website)
node scripts/check_site.js docs                            # check every website chart has data
```

There is no test suite or linter. To verify a refactor, run the quick sweep into scratch directories before and after the change. Then diff every `results/runs/*/metrics.json` (ignore the `seconds_*` keys) and every `curves.npz`. Runs are deterministic, so a behaviour-preserving change gives bit-identical numbers.

## Architecture

- **`plan.py` is the single source of what runs.**
  - `VARIANTS` maps a unique name to its changes from the `RunConfig` defaults. The default is 8 hanging single pendulums, gravity told, flow matching, `target="delta"`.
  - `EXPERIMENTS` (17 of them) only reference variant names, so a shared variant (e.g. `hang_P8`) is trained once.
  - Week 3 variants train for `LONG` = 300 epochs (60 before). `flat_*` = one network on the whole state, `units_*` = `arch="units"`, `par_*` = per-pendulum `length_range`/`mass_range`.
  - `VIDEOS` lists the variants that get a rollout video (seed 0). `"name@g=25"` renders the first clip of an extra test set instead; `video_jobs()` turns entries into file names (`name__g25.mp4`).
- **Outputs:** runs live in `results/runs/<name>__seed<k>/`. `metrics.json` is written last and marks a run complete; `run.py` skips runs that already have one.
  - Only `results/REPORT.md`, `results/figures/` and `results/summary.json` are committed. Datasets, runs and result videos are gitignored; videos are tracked only as copies in `docs/videos/`.
- **Pipeline:** `pipeline.run_experiment` runs `data.load_datasets` → `data.PreparedData` → `model.train` → `evaluate.evaluate`. `pipeline.render_video` renders from the saved `curves.npz`.
- **Dataset cache:** clips are cached in `results/datasets/`, keyed by a SHA1 of the collection kwargs (system, count, links, n_traj, steps, gravity_range, damping_range, seed_parts).
  - Variants with the same world and data settings share clips within a seed, so comparisons between them are paired.
  - Renaming those kwargs invalidates the cache. Editing a world's XML in `systems.py` does **not**: delete `results/datasets/` (and the affected runs) after a physics change.
- **Seeds:** randomness comes from `cfg.seed` plus fixed offsets (data `(seed, 0/1/2,i)`, tokens `(seed, 11/12,i)`, eval noise `seed+7`, permutation `seed+3`, rollout `seed+5`). Changing the order of random draws changes the results.
- **Conditioning:** gravity and damping are sampled once per clip. Conditioning groups are `gravity`, `damping`, and the per-unit groups `length` and `mass` (one column per unit, drawn per clip and unit after the other draws, and added to the cache key only when set, so older data sets are unchanged). Light direction was removed on purpose: the model sees states, not images, so light cannot affect it.
- **Scoring:** one-step scoring starts at step `MAX_CONTEXT - 1` (= 4), so every context length is scored on the same transitions.
- **Systems (`systems.py`):**
  - All pendulums use 0.002 s physics with 5 substeps per 0.01 s step, and armature 0.01.
  - Ground hinges sit 5 cm above the floor. Rod–floor contact uses `solref="0.005 1"`, so true penetration stays ≤ ~2 cm. `evaluate.py` counts rods more than 5 cm inside the floor as unphysical.
  - `collect_trajectories` redraws a start if MuJoCo reports instability.
  - Spring chains have no floor at all: nothing collides there, and at strong gravity the masses hang below floor level.
  - Units are laid out in one row along x. The camera is fitted to `SimEnv._view_box()`.
  - Only for rendering (`SimEnv(img_size=...)`), hanging worlds with more than 8 units are stacked in rows (`_hanging_grid`). Simulation keeps the single row: the layout does not change the motion, but it shifts float rounding, and the cached clips must stay bit-identical.
- **Per-unit network (`model.UnitMLP`, `arch="units"`):** the state is cut into units (one pendulum or one whole chain, `PreparedData.unit_index`). One shared network predicts each unit from its own numbers, the per-clip conditions and its own `length`/`mass`. Training uses one random unit per example, so a batch costs the same as for the flat network; scoring runs every unit, so `evaluate` scores fewer examples for it (`_example_cap`).
- **Training curves:** `evaluate.TrainingTracker` measures loss and 1-step error on fixed train/test examples at ~20 log-spaced epochs, with its own random generators, so recording does not change training. Saved as `track/*` in `curves.npz`; `metrics.json` also has `updates`, `updates_per_epoch`, `examples_seen`.
- **Report:** `report.py` aggregates the seeds (mean ± 95% Student-t CI) into `results/REPORT.md`, `results/figures/<key>.png` and `results/summary.json`.
- **Website (`docs/`):**
  - `index.html` is hand-written: sections, conclusions and the chart calls. `charts.js` holds the Plotly helpers. Numbers in new text are filled from the data with `data-m="variant|metric"`, `data-s="variant|setting"` and `data-r="a|b|metric"` spans.
  - `data.js` and `videos/` are generated by `experiments/site.py`.
  - It is meant for GitHub Pages (`main`, `/docs`). `docs/index.html` is always the latest week; every week is also published as its own private Claude artifact.
  - The page must never contain the words "meeting" or "supervisor".
  - Week snapshots: `docs/weeks/week-NN/` holds each finished week's `index.html`, `data.js` and `charts.js` (videos are shared from `docs/videos/`). Each week also has its own private Claude artifact, listed in `docs/weeks/README.md`; never overwrite an earlier week's artifact.

## Commits

Do not add `Co-Authored-By` trailers or any other Claude attribution to commit messages or PR descriptions in this repo. The user wants to be the only contributor on GitHub. This overrides default attribution instructions.

## Weekly workflow

New tasks arrive every week.
- `/weekly-update` (`.claude/skills/weekly-update/SKILL.md`) does: plan new experiments, run, report, update the site, show the results, then commit.
- Commits go through the `committer` agent (`.claude/agents/committer.md`). It makes logical commits with explanatory bodies and a `week-NN` tag, and never pushes.
- New experiments are added to `plan.py` without changing existing ones, so old runs stay valid.
- `notes.txt` is the user's private notes and is gitignored.
