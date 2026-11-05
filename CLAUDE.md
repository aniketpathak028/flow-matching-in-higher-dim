# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A university study project. A flow-matching model (a residual MLP) learns to predict the next MuJoCo state `[qpos, qvel]` of simulated worlds, mainly pendulums. D = 2 × number of joints; for hinge joints `qpos` is the angle.

Everything lives in one framework, `experiments/`. The earlier standalone scripts (`fm_model.py`, `mujoco_pipeline.py`) were folded into it: their ground pendulum is `system="ground"`, and their network pieces are in `model.py`. `experiments/README.md` has a plain-language walkthrough. The user prefers plain-language explanations.

## Commands

Setup: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`. Videos also need ffmpeg.

Run from the repo root as modules:

```bash
python -m experiments.run                                  # all 11 experiments × seeds 0 1 2 (141 runs, ~1.5 h), resumable
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
  - `EXPERIMENTS` (11 of them) only reference variant names, so a shared variant (e.g. `hang_P8`) is trained once.
  - `VIDEOS` lists the variants that get a rollout video (seed 0). `"name@g=25"` renders the first clip of an extra test set instead; `video_jobs()` turns entries into file names (`name__g25.mp4`).
- **Outputs:** runs live in `results/runs/<name>__seed<k>/`. `metrics.json` is written last and marks a run complete; `run.py` skips runs that already have one.
  - Only `results/REPORT.md`, `results/figures/` and `results/summary.json` are committed. Datasets, runs and result videos are gitignored; videos are tracked only as copies in `docs/videos/`.
- **Pipeline:** `pipeline.run_experiment` runs `data.load_datasets` → `data.PreparedData` → `model.train` → `evaluate.evaluate`. `pipeline.render_video` renders from the saved `curves.npz`.
- **Dataset cache:** clips are cached in `results/datasets/`, keyed by a SHA1 of the collection kwargs (system, count, links, n_traj, steps, gravity_range, damping_range, seed_parts).
  - Variants with the same world and data settings share clips within a seed, so comparisons between them are paired.
  - Renaming those kwargs invalidates the cache. Editing a world's XML in `systems.py` does **not**: delete `results/datasets/` (and the affected runs) after a physics change.
- **Seeds:** randomness comes from `cfg.seed` plus fixed offsets (data `(seed, 0/1/2,i)`, tokens `(seed, 11/12,i)`, eval noise `seed+7`, permutation `seed+3`, rollout `seed+5`). Changing the order of random draws changes the results.
- **Conditioning:** gravity and damping are sampled once per clip. Conditioning groups are `gravity` and `damping`. Light direction was removed on purpose: the model sees states, not images, so light cannot affect it.
- **Scoring:** one-step scoring starts at step `MAX_CONTEXT - 1` (= 4), so every context length is scored on the same transitions.
- **Systems (`systems.py`):**
  - All pendulums use 0.002 s physics with 5 substeps per 0.01 s step, and armature 0.01.
  - Ground hinges sit 5 cm above the floor. Rod–floor contact uses `solref="0.005 1"`, so true penetration stays ≤ ~2 cm. `evaluate.py` counts rods more than 5 cm inside the floor as unphysical.
  - `collect_trajectories` redraws a start if MuJoCo reports instability.
  - Spring chains have no floor at all: nothing collides there, and at strong gravity the masses hang below floor level.
  - Units are laid out in one row along x. The camera is fitted to `SimEnv._view_box()`.
  - Only for rendering (`SimEnv(img_size=...)`), hanging worlds with more than 8 units are stacked in rows (`_hanging_grid`). Simulation keeps the single row: the layout does not change the motion, but it shifts float rounding, and the cached clips must stay bit-identical.
- **Report:** `report.py` aggregates the seeds (mean ± 95% Student-t CI) into `results/REPORT.md`, `results/figures/<key>.png` and `results/summary.json`.
- **Website (`docs/`):**
  - `index.html` is hand-written: sections, conclusions and the chart calls. `charts.js` holds the Plotly helpers.
  - `data.js` and `videos/` are generated by `experiments/site.py`.
  - It is meant for GitHub Pages (`main`, `/docs`). The same files are published as the private Claude artifact https://claude.ai/artifact/B6a5Z5DvWKvDwc8hYiTPTi (slug `51c27da5-0af2-463b-8ab9-daa66abc9fa5`).
  - The page must never contain the words "meeting" or "supervisor".

## Commits

Do not add `Co-Authored-By` trailers or any other Claude attribution to commit messages or PR descriptions in this repo. The user wants to be the only contributor on GitHub. This overrides default attribution instructions.

## Weekly workflow

New tasks arrive every week.
- `/weekly-update` (`.claude/skills/weekly-update/SKILL.md`) does: plan new experiments, run, report, update the site, show the results, then commit.
- Commits go through the `committer` agent (`.claude/agents/committer.md`). It makes logical commits with explanatory bodies and a `week-NN` tag, and never pushes.
- New experiments are added to `plan.py` without changing existing ones, so old runs stay valid.
- `notes.txt` is the user's private notes and is gitignored.
