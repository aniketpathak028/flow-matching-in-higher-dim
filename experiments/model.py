"""Step 2 of a run: the network and how it is trained.

* Flow matching (model="fm"). To make a prediction, start from random noise x
  and nudge it 20 times in the direction the network points to; after the last
  nudge x is the predicted answer. Training: take the true answer, mix it with
  noise (`tau` = how much answer, 0..1) and teach the network which way points
  from the noise to the answer.

* Plain regression (model="regression"): the network outputs the answer in one
  go. Used only as a comparison.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from experiments.config import RunConfig

if TYPE_CHECKING:
    from experiments.data import PreparedData


class Normalizer:
    """Rescales vectors to mean 0 / std 1 per dimension, using statistics of the data it was fitted on."""

    def __init__(self, tensor: torch.Tensor):
        self.mean = tensor.mean(dim=0, keepdim=True)
        self.std = tensor.std(dim=0, keepdim=True) + 1e-7

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)


class SinusoidalPosEmb(nn.Module):
    """Turns the flow time tau (one number in [0, 1]) into `dim` sine/cosine features."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=x.device) * -(math.log(10000) / (half - 1)))
        emb = x * freqs[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ResBlock(nn.Module):
    """Two linear layers plus a skip connection: output = SiLU(x + f(x))."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU(),
                                 nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class DynamicsMLP(nn.Module):
    """input -> Linear -> ReLU -> `blocks` x ResBlock -> Linear -> output."""

    def __init__(self, out_dim: int, context_dim: int, cond_dim: int,
                 hidden: int, blocks: int, flow: bool, time_embed_dim: int = 64):
        super().__init__()
        self.flow = flow
        self.time_emb = SinusoidalPosEmb(time_embed_dim) if flow else None
        # Flow matching also gets the current noisy guess x_tau and the time tau.
        in_dim = context_dim + cond_dim + (time_embed_dim + out_dim if flow else 0)
        self.input_proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(blocks)])
        self.out_proj = nn.Linear(hidden, out_dim)

    def forward(self, context, cond, tau=None, x_tau=None):
        parts = [context, cond]
        if self.flow:
            parts = [self.time_emb(tau), x_tau] + parts
        h = torch.relu(self.input_proj(torch.cat(parts, dim=-1)))
        for block in self.blocks:
            h = block(h)
        return self.out_proj(h)


@torch.no_grad()
def predict(model: DynamicsMLP, context, cond, noise, steps: int):
    """Scaled prediction. Flow matching: walk from `noise` to the answer in `steps` Euler steps."""
    if not model.flow:
        return model(context, cond)
    x = noise.clone()
    d_tau = 1.0 / steps
    for i in range(steps):
        tau = torch.full((x.shape[0], 1), i * d_tau, device=x.device)
        x = x + d_tau * model(context, cond, tau=tau, x_tau=x)
    return x


def train(cfg: RunConfig, data: "PreparedData") -> tuple[DynamicsMLP, list[float]]:
    """Train one network. Returns it and the average loss of every epoch."""
    device = data.device
    torch.manual_seed(cfg.seed)
    model = DynamicsMLP(
        out_dim=data.obs_dim,
        context_dim=data.obs_dim * cfg.context_len,
        cond_dim=data.cond_dim,
        hidden=cfg.hidden,
        blocks=cfg.blocks,
        flow=cfg.model == "fm",
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    n_idx, t_idx = data.examples("train", t_min=cfg.context_len - 1)
    shuffler = torch.Generator(device="cpu").manual_seed(cfg.seed)
    losses = []
    for _ in range(cfg.epochs):
        model.train()
        order = torch.randperm(len(n_idx), generator=shuffler).to(device)
        total = 0.0
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start:start + cfg.batch_size]
            context, cond, target = data.batch("train", n_idx[idx], t_idx[idx])
            if model.flow:
                tau = torch.rand(len(idx), 1, device=device)       # how far along: 0 = noise, 1 = answer
                x0 = torch.randn_like(target)                       # random starting noise
                x_tau = (1 - tau) * x0 + tau * target               # a point on the straight line noise -> answer
                velocity = model(context, cond, tau=tau, x_tau=x_tau)
                loss = torch.mean((velocity - (target - x0)) ** 2)  # should point from noise to answer
            else:
                loss = torch.mean((model(context, cond) - target) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        scheduler.step()
        losses.append(total / len(order))
    return model, losses
