# `experiments/` in plain words

## What is being learned

A network looks at a **state** of a simulated world (every joint's position
and speed; for a hinge the position is its angle) and guesses the state
0.01 s later. If it does that well, you can chain the guesses and the network
plays the physics forward on its own. That is a **rollout**.

## Files, in reading order

| File | Role |
|---|---|
| `pipeline.py` | one run start to finish: data → train → score → save (read this first) |
| `config.py` | `RunConfig`, the settings card of one run, one comment per field |
| `data.py` | simulate clips (or load them from the cache), scale them, cut them into (input, answer) pairs |
| `model.py` | the residual MLP, flow-matching training and prediction |
| `evaluate.py` | one-step error, rollouts, shuffle test, train/test gap, floor check |
| `plan.py` | every variant and the 10 experiments that compare them |
| `systems.py` | the MuJoCo worlds: hanging and ground pendulums, springs, cart-poles, balls |
| `run.py` | command line: trains missing runs in parallel, renders videos, builds the report |
| `report.py` | averages seeds into figures, `REPORT.md` and `summary.json` |

## How one run flows

```
RunConfig
  ├─ data.load_datasets     200 training + 50 test clips of 200 steps (2 s), cached
  │                         each clip: random start, random gravity in 4–15 (and damping if enabled)
  ├─ data.PreparedData      scale to mean 0 / std 1; input = last K states (+ gravity, damping, tokens)
  │                         answer = change to the next state
  ├─ model.train            flow matching, 60 epochs
  ├─ evaluate.evaluate      on the 50 unseen test clips
  └─ save                   results/runs/<variant>__seed<k>/metrics.json + curves.npz
```

## Flow matching in one paragraph

To predict, start from random noise and move it 20 small steps in the
direction the network points to; the end point is the predicted change.
Training teaches the network those directions: take the true answer `x1` and
noise `x0`, pick a point `x_tau = (1 - tau) * x0 + tau * x1` on the straight
line between them, and train the network to output `x1 - x0` there.

## The scores

| Name | Plain meaning | Good value |
|---|---|---|
| 1-step error | squared error of one guess ÷ size of a typical one-step change | ≪ 1 (1 = useless) |
| rollout error @h | error after h steps on the model's own guesses, in units of the state's spread | < 0.5 |
| valid steps | steps before the rollout error passes 0.5 (max 100) | 100 |
| test/train gap | test error ÷ training error | ≈ 1 (≫ 1 = memorised) |
| shuffle X | how much the error grows when X is scrambled between examples | ≈ 0 = X is ignored |
| floor check | share of rollout states with a rod more than 5 cm inside the floor (ground only) | as low as the truth |

## Two worlds that matter most

* **Hanging** (`pendulum`): rods hang from a raised mount and swing freely. Nothing touches anything.
* **Ground** (`ground`): the original setup. Rods are hinged 5 cm above a solid
  floor and start upright with a random push, so they fall and hit the floor.
  The floor is drawn see-through in videos, so a predicted rod that sinks into it stays visible.

Light direction is no longer part of the setup. It only changes how a picture
is rendered, and the model sees physical states, never pictures, so it cannot
matter.
