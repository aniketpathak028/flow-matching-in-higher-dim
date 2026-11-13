---
name: weekly-update
description: Run one week of the study project. Turn this week's tasks into experiments, run them, update the report and website, show the results, then commit. Use when the user gives the new week's tasks or types /weekly-update.
---

# Weekly update

The user gives this week's tasks, usually from the person they report to. Do the steps in order. Explain in plain language: the user prefers it, and long explanations belong on the page.

## 0. Get oriented

- Find the week number: the latest `week-NN` tag + 1 (`git tag -l 'week-*'`). Ask only if there are no tags.
- `git status`: if there are uncommitted changes from earlier work, ask before mixing them in.
- Read `CLAUDE.md`, `experiments/plan.py`, and the sections of `docs/index.html` relevant to the tasks.

## 1. Design the experiments

- Turn each task into one or a few **meaningful** comparisons. Prefer few variants with clear differences over sweeps; the user dislikes exhaustive grids.
- Add new variants to `VARIANTS` and new `Experiment(...)` entries to `EXPERIMENTS` in `experiments/plan.py`. **Never change or remove existing ones**: earlier weeks must stay reproducible and their runs stay valid.
- Reuse existing variants as baselines (e.g. `hang_P8`). That way a comparison is paired and costs nothing extra.
- New abilities go in the framework (`systems.py`, `data.py`, `evaluate.py`, …), behind a `RunConfig` field whose default keeps old behaviour.
  - Check: the dry run must still say all old runs are done.
- If you change a world's XML in `systems.py`, cached clips are **not** invalidated. Delete the affected `results/datasets/` files and `results/runs/` folders, then rerun those variants.
- If a task does not make sense (e.g. something that cannot affect the model), say why on the page instead of running it. Light direction is an example.
- Briefly tell the user the planned experiments before a long run (more than ~15 minutes).

## 2. Run

```bash
.venv/bin/python -m experiments.run --dry-run --only <new keys>
.venv/bin/python -m experiments.run --quick --seeds 0 --no-videos --out <scratchpad>/quick --only <new keys>
.venv/bin/python -m experiments.run --only <new keys>          # in the background; finished runs are skipped
```

- Add new video variants to `plan.VIDEOS` if a new behaviour is worth watching.
- Then rebuild:

```bash
.venv/bin/python -m experiments.report --out results
.venv/bin/python -m experiments.site
```

## 3. Update the website (`docs/index.html`)

- Per new experiment, add a section in the same style as the existing ones:
  - question;
  - what was varied;
  - charts via `Charts.*` in the inline script;
  - a video if relevant;
  - a written conclusion;
  - caveats.
- Every number comes from `window.DATA`. Never type numbers into the HTML that the data can supply.
- Update the summary table at the end and add a **"Week N"** line to the changelog (create the section if it is missing): date plus what was added.
- **Never use the words "meeting" or "supervisor"** anywhere on the page.
- Verify: `node scripts/check_site.js docs` (must say "0 with problems"). Also `grep -ci "meeting\|supervisor" docs/index.html` must print 0.

## 4. Show the user

In chat, give:
- what was run;
- a short table of the headline numbers;
- the conclusions;
- anything surprising or anything that went wrong.

Then save the week:
- copy `docs/index.html`, `data.js` and `charts.js` into `docs/weeks/week-NN/`, with video paths changed to `../../videos/`;
- publish a **new** private artifact titled "Week N" from `docs/` (never overwrite an earlier week's artifact);
- add its link to `docs/weeks/README.md`.

**Wait for the user's OK** before committing.

## 5. Commit

- Hand off to the `committer` agent with the week number and a one-line summary of the week.
- Show its commit list.
- Push (`git push origin main --tags`) only when the user says so.
