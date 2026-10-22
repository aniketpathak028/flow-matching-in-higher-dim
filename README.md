# MuJoCo Flow Matching

This project uses MuJoCo to simulate robot arms with different numbers of links. It trains a flow-matching model to predict the next arm state, then compares its predictions with a MuJoCo simulation.

Each state contains the arm's joint positions and velocities. The experiment can compare several arm sizes, such as 1, 2, 3, 5, and 10 links.

## Setup

Create and activate a Python virtual environment, then install the project requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run an experiment

Run the default comparison for 1 through 5 links:

```bash
python fm_model.py
```

Choose link counts and training settings with command-line options:

```bash
python fm_model.py \
  --link-counts 2 3 5 10 \
  --num-trajectories 200 \
  --steps-per-trajectory 100 \
  --epochs 100 \
  --rollout-horizon 200 \
  --inference-steps 40
```

The program collects simulation transitions for each link count, trains a model, and predicts a rollout with Euler integration. The rollout is compared with a ground-truth MuJoCo simulation.

## Results

Results are written to `flow_matching_results/` by default:

- `comparison_summary.csv` contains training loss and rollout MSE/RMSE for each link count.
- Each `n_links_N/` folder contains that experiment's dataset, trained model weights, training-loss CSV, and comparison video.

Use a new output directory with `--output-dir` to keep a fresh experiment separate. Existing datasets in an experiment folder are reused if their saved collection settings match the requested settings.
