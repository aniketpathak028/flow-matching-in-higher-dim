import argparse
import csv
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mediapy as media
from PIL import Image, ImageDraw
from mujoco_pipeline import MuJoCoEnvironment

# sinusoidal Positional Embedding for Flow Time tau
class SinusoidalPosEmb(nn.Module):
    """Embeds scalar flow time tau [0,1] into continuous high-dimensional features."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

# state normalizer class
class Normalizer:
    """Standardizes state vectors to zero mean and unit variance."""
    def __init__(self, tensor: torch.Tensor):
        self.mean = tensor.mean(dim=0, keepdim=True)
        self.std = tensor.std(dim=0, keepdim=True) + 1e-7

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)


# residual mlp flownet 
class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.act = nn.SiLU() # why SiLU() and why adding a Res Block?

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))

# flow model that predicts the vector field -> dir and the rate of change 
class StateFlowResNet(nn.Module):
    """Vector field network v_theta(tau, x_tau, context) for Flow Matching."""
    def __init__(self, state_dim: int, hidden_dim: int = 256, time_embed_dim: int = 64):
        super().__init__()
        self.time_emb = SinusoidalPosEmb(time_embed_dim)
        
        # input features: time_embed_dim + state_dim (x_tau) + state_dim (context s_t)
        in_dim = time_embed_dim + state_dim + state_dim
        
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(3)])
        self.out_proj = nn.Linear(hidden_dim, state_dim)

    def forward(self, tau: torch.Tensor, x_tau: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(tau)
        inputs = torch.cat([t_emb, x_tau, context], dim=-1)
        
        h = torch.relu(self.input_proj(inputs))
        for block in self.blocks:
            h = block(h)
        return self.out_proj(h)


def collect_and_save_dataset(
    env: MuJoCoEnvironment, 
    num_trajectories: int = 200, 
    steps_per_traj: int = 100,
    save_path: str = "mujoco_dataset.pt"
):
    """Collects simulation trajectory transitions (s_t, s_{t+1}) and saves to disk."""
    s_t_list, s_next_list = [], []
    
    for _ in range(num_trajectories):
        env.data.qpos[:] = 0.0
        env.data.qvel[:] = np.random.uniform(-2.0, 2.0, size=env.nv)
        env.set_state(env.get_state())
        
        for _ in range(steps_per_traj):
            s_t = env.get_state()
            s_next = env.step_simulation()
            
            s_t_list.append(s_t)
            s_next_list.append(s_next)

    s_t_tensor = torch.from_numpy(np.array(s_t_list, dtype=np.float32))
    s_next_tensor = torch.from_numpy(np.array(s_next_list, dtype=np.float32))
    
    torch.save(
        {
            "s_t": s_t_tensor,
            "s_next": s_next_tensor,
            "num_trajectories": num_trajectories,
            "steps_per_traj": steps_per_traj,
        },
        save_path,
    )
    print(f"Collected and saved dataset to '{save_path}' ({len(s_t_tensor)} transition samples).")
    return s_t_tensor, s_next_tensor


# ODE sampler using Euler integration for the learned vector field
@torch.no_grad()
def sample_next_state_euler(
    model: nn.Module, 
    s_t_norm: torch.Tensor, 
    normalizer: Normalizer,
    num_steps: int = 40
) -> torch.Tensor:
    """Integrates the learned vector field from tau=0 to tau=1 using Euler's method."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")

    model.eval()
    batch_size, state_dim = s_t_norm.shape
    device = s_t_norm.device
    
    x = torch.randn(batch_size, state_dim, device=device)
    d_tau = 1.0 / num_steps

    for step in range(num_steps):
        tau_val = step * d_tau
        tau = torch.full((batch_size, 1), tau_val, device=device)
        velocity = model(tau, x, context=s_t_norm)
        x = x + d_tau * velocity

    return normalizer.decode(x)


def render_comparison_video(
    env: MuJoCoEnvironment,
    gt_trajectory: np.ndarray,
    pred_trajectory: np.ndarray,
    output_filename: str,
    fps: int = 30,
) -> bool:
    """Render labeled ground-truth and flow-matching trajectories side by side."""
    print(f"\nRendering offscreen video frames ({len(gt_trajectory)} frames)...")
    combined_frames = []

    for step in range(len(gt_trajectory)):
        gt_frame = env.render_rgb(gt_trajectory[step])
        pred_frame = env.render_rgb(pred_trajectory[step])
        gt_image = Image.fromarray(gt_frame)
        pred_image = Image.fromarray(pred_frame)
        gt_draw = ImageDraw.Draw(gt_image)
        pred_draw = ImageDraw.Draw(pred_image)
        label_height = 36
        gt_draw.rectangle((0, 0, gt_image.width, label_height), fill=(32, 32, 32))
        pred_draw.rectangle((0, 0, pred_image.width, label_height), fill=(32, 32, 32))
        gt_draw.text(
            (gt_image.width // 2, label_height // 2),
            "Ground Truth",
            fill=(255, 255, 255),
            anchor="mm",
        )
        pred_draw.text(
            (pred_image.width // 2, label_height // 2),
            "Flow Matching",
            fill=(255, 255, 255),
            anchor="mm",
        )
        combined_frames.append(
            np.concatenate([np.asarray(gt_image), np.asarray(pred_image)], axis=1)
        )

    try:
        media.write_video(output_filename, combined_frames, fps=fps)
        print(f"Successfully saved comparison video to '{output_filename}'.")
        return True
    except RuntimeError as error:
        print(f"\n[Warning] Could not render MP4 video: {error}")
        print("Tip: Install ffmpeg using 'brew install ffmpeg' or 'pip install imageio-ffmpeg'.")
        return False


def train_model(
    s_t_raw: torch.Tensor,
    s_next_raw: torch.Tensor,
    state_dim: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[StateFlowResNet, Normalizer, list[float]]:
    """Train one flow model and return its normalizer and per-epoch losses."""
    normalizer = Normalizer(s_t_raw)
    s_t_norm = normalizer.encode(s_t_raw)
    s_next_norm = normalizer.encode(s_next_raw)

    dataset = TensorDataset(s_t_norm, s_next_norm)
    loader_generator = torch.Generator().manual_seed(seed)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )

    model = StateFlowResNet(state_dim=state_dim, hidden_dim=256).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    epoch_losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_s_t, batch_s_next in dataloader:
            batch_s_t, batch_s_next = batch_s_t.to(device), batch_s_next.to(device)
            b_size = batch_s_t.shape[0]

            tau = torch.rand(b_size, 1, device=device)
            x_0 = torch.randn(b_size, state_dim, device=device)
            x_1 = batch_s_next

            x_tau = (1 - tau) * x_0 + tau * x_1
            target_velocity = x_1 - x_0
            pred_velocity = model(tau, x_tau, context=batch_s_t)

            loss = torch.mean((pred_velocity - target_velocity) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * b_size

        scheduler.step()
        avg_loss = total_loss / len(dataset)
        epoch_losses.append(avg_loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} | Normalized Loss: {avg_loss:.6f}")

    return model, normalizer, epoch_losses


def run_experiment(
    num_links: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, int | float | str | bool]:
    """Train and evaluate one link-count configuration and save its artifacts."""
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    experiment_dir = Path(args.output_dir) / f"n_links_{num_links}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = experiment_dir / "dataset.pt"
    weights_path = experiment_dir / "flow_model.pt"

    env = MuJoCoEnvironment(
        num_links=num_links,
        img_width=args.image_size,
        img_height=args.image_size,
    )
    state_dim = env.state_dim

    if dataset_path.exists():
        print(f"Loading dataset from '{dataset_path}'...")
        saved_data = torch.load(dataset_path, map_location="cpu")
        s_t_raw, s_next_raw = saved_data["s_t"], saved_data["s_next"]
        expected_shape = (state_dim,)
        if s_t_raw.ndim != 2 or s_t_raw.shape[1:] != expected_shape:
            raise ValueError(
                f"Dataset '{dataset_path}' has state shape {tuple(s_t_raw.shape)}; "
                f"expected [samples, {state_dim}]. Remove or replace the incompatible dataset."
            )
        if s_next_raw.shape != s_t_raw.shape:
            raise ValueError(f"Dataset '{dataset_path}' has mismatched transition tensor shapes.")
        saved_trajectories = saved_data.get("num_trajectories")
        saved_steps = saved_data.get("steps_per_traj")
        if saved_trajectories is not None and saved_trajectories != args.num_trajectories:
            raise ValueError(
                f"Dataset '{dataset_path}' was collected with {saved_trajectories} trajectories, "
                f"not the requested {args.num_trajectories}. Use a new output directory or "
                "remove this dataset to recollect it."
            )
        if saved_steps is not None and saved_steps != args.steps_per_trajectory:
            raise ValueError(
                f"Dataset '{dataset_path}' has {saved_steps} steps per trajectory, "
                f"not the requested {args.steps_per_trajectory}. Use a new output directory or "
                "remove this dataset to recollect it."
            )
    else:
        print(f"Collecting simulation transitions for {num_links} links...")
        s_t_raw, s_next_raw = collect_and_save_dataset(
            env,
            num_trajectories=args.num_trajectories,
            steps_per_traj=args.steps_per_trajectory,
            save_path=str(dataset_path),
        )

    model, normalizer, epoch_losses = train_model(
        s_t_raw=s_t_raw,
        s_next_raw=s_next_raw,
        state_dim=state_dim,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=seed,
    )
    torch.save(model.state_dict(), weights_path)

    with (experiment_dir / "training_loss.csv").open("w", newline="") as loss_file:
        writer = csv.writer(loss_file)
        writer.writerow(["epoch", "normalized_training_loss"])
        writer.writerows(enumerate(epoch_losses, start=1))

    initial_state = np.zeros(state_dim, dtype=np.float64)
    initial_state[env.nq:] = np.random.default_rng(seed).uniform(-2.0, 2.0, size=env.nv)
    env.set_state(initial_state)

    ground_truth = [env.get_state()]
    for _ in range(args.rollout_horizon):
        ground_truth.append(env.step_simulation())
    ground_truth = np.asarray(ground_truth)

    torch.manual_seed(seed + 1)
    prediction = [ground_truth[0]]
    current_state = ground_truth[0]
    for _ in range(args.rollout_horizon):
        state_tensor = torch.from_numpy(current_state[None, :]).float().to(device)
        state_context = normalizer.encode(state_tensor)
        next_state = sample_next_state_euler(
            model,
            state_context,
            normalizer,
            num_steps=args.inference_steps,
        )
        current_state = next_state.cpu().numpy()[0]
        prediction.append(current_state)
        env.set_state(current_state)

    prediction = np.asarray(prediction)
    squared_errors = (prediction[1:] - ground_truth[1:]) ** 2
    position_mse = float(np.mean(squared_errors[:, :env.nq]))
    velocity_mse = float(np.mean(squared_errors[:, env.nq:]))
    rollout_mse = float(np.mean(squared_errors))
    video_path = experiment_dir / "ground_truth_vs_flow_matching.mp4"
    video_saved = render_comparison_video(
        env,
        ground_truth,
        prediction,
        output_filename=str(video_path),
        fps=args.fps,
    )

    result: dict[str, int | float | str | bool] = {
        "num_links": num_links,
        "state_dim": state_dim,
        "transitions": len(s_t_raw),
        "num_trajectories": args.num_trajectories,
        "steps_per_trajectory": args.steps_per_trajectory,
        "epochs": args.epochs,
        "final_training_loss": epoch_losses[-1],
        "rollout_mse": rollout_mse,
        "rollout_rmse": float(np.sqrt(rollout_mse)),
        "position_mse": position_mse,
        "velocity_mse": velocity_mse,
        "inference_method": "Euler",
        "inference_steps": args.inference_steps,
        "video": str(video_path),
        "video_saved": video_saved,
    }
    print(
        f"{num_links} links | final train loss={epoch_losses[-1]:.6f} | "
        f"rollout MSE={rollout_mse:.6f} | RMSE={np.sqrt(rollout_mse):.6f}"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare flow-matching models across arm link counts."
    )
    parser.add_argument("--link-counts", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--num-trajectories", type=int, default=200)
    parser.add_argument("--steps-per-trajectory", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rollout-horizon", type=int, default=100)
    parser.add_argument("--inference-steps", type=int, default=40)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="flow_matching_results")
    args = parser.parse_args()

    positive_values = {
        "link counts": args.link_counts,
        "num-trajectories": [args.num_trajectories],
        "steps-per-trajectory": [args.steps_per_trajectory],
        "epochs": [args.epochs],
        "batch-size": [args.batch_size],
        "rollout-horizon": [args.rollout_horizon],
        "inference-steps": [args.inference_steps],
        "image-size": [args.image_size],
        "fps": [args.fps],
    }
    for name, values in positive_values.items():
        if any(value <= 0 for value in values):
            parser.error(f"{name} must contain only positive integers.")
    if len(set(args.link_counts)) != len(args.link_counts):
        parser.error("--link-counts must not contain duplicates.")
    return args


def main():
    args = parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Compute Device: {device}")
    print(f"Comparing link counts: {args.link_counts}")

    results = [
        run_experiment(num_links, args, device)
        for num_links in args.link_counts
    ]
    summary_path = Path(args.output_dir) / "comparison_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nComparison summary saved to '{summary_path}'.")


if __name__ == "__main__":
    main()