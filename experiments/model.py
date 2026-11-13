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


class UnitMLP(nn.Module):
    """The same small network applied to every unit (one pendulum, or one whole chain).

    Instead of one big vector of D numbers, the state is cut into `count` pieces of
    `u` = 2 x links numbers (the angles and speeds of one unit). Each piece goes through
    the same network, together with the facts about the whole clip (gravity) and about
    that unit (its rod length and mass, if told). Joints inside a chain stay together,
    because they push on each other; separate units never touch, so nothing passes
    between them. The input/output interface is the same as DynamicsMLP's.
    """

    def __init__(self, unit_index: torch.Tensor, context_len: int, global_cols: torch.Tensor,
                 unit_cols: torch.Tensor, hidden: int, blocks: int, flow: bool, time_embed_dim: int = 64):
        super().__init__()
        self.flow = flow
        self.register_buffer("unit_index", unit_index)     # [count, u]: which state entries form each unit
        self.register_buffer("global_cols", global_cols)   # condition columns shared by all units
        self.register_buffer("unit_cols", unit_cols)       # [count, n]: condition columns of each unit
        self.count, self.u = unit_index.shape
        self.state_dim = int(unit_index.numel())
        self.context_len = context_len
        self.time_emb = SinusoidalPosEmb(time_embed_dim) if flow else None
        in_dim = (context_len * self.u + len(global_cols) + unit_cols.shape[1]
                  + (time_embed_dim + self.u if flow else 0))
        self.input_proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(blocks)])
        self.out_proj = nn.Linear(hidden, self.u)

    def forward(self, context, cond, tau=None, x_tau=None, units=None):
        """`units` = None: predict every unit, x_tau and output are [B, D].
        `units` = [B] unit numbers: predict only that unit per example, x_tau and output are [B, u]."""
        B = context.shape[0]
        ctx = context.view(B, self.context_len, -1)[:, :, self.unit_index]        # [B, K, count, u]
        ctx = ctx.permute(0, 2, 1, 3).reshape(B, self.count, -1)                   # [B, count, K*u]
        per_unit = cond[:, self.unit_cols]                                          # [B, count, n]
        shared = cond[:, self.global_cols]                                          # [B, Cg]
        if units is not None:
            rows = torch.arange(B, device=context.device)
            ctx, per_unit = ctx[rows, units][:, None], per_unit[rows, units][:, None]   # one unit each
            x_units = x_tau[:, None] if self.flow else None
        else:
            x_units = x_tau[:, self.unit_index] if self.flow else None             # [B, count, u]
        n = ctx.shape[1]
        parts = [ctx, per_unit, shared[:, None].expand(B, n, -1)]
        if self.flow:
            parts = [self.time_emb(tau)[:, None].expand(B, n, -1), x_units] + parts
        h = torch.relu(self.input_proj(torch.cat(parts, dim=-1)))
        for block in self.blocks:
            h = block(h)
        out = self.out_proj(h)                                                       # [B, n, u]
        if units is not None:
            return out[:, 0]
        full = out.new_empty(B, self.state_dim)
        full[:, self.unit_index.reshape(-1)] = out.reshape(B, -1)
        return full


def build_model(cfg: RunConfig, data: "PreparedData") -> nn.Module:
    flow = cfg.model == "fm"
    if cfg.arch == "units":
        global_cols, unit_cols = data.unit_layout()
        model = UnitMLP(data.unit_index(), cfg.context_len, global_cols, unit_cols, cfg.hidden, cfg.blocks, flow)
    else:
        model = DynamicsMLP(out_dim=data.obs_dim, context_dim=data.obs_dim * cfg.context_len, cond_dim=data.cond_dim,
                            hidden=cfg.hidden, blocks=cfg.blocks, flow=flow)
    model.noise_scale = cfg.noise_scale    # used by predict(): prediction starts from noise of this size
    return model


def checkpoint_epochs(epochs: int, points: int = 20) -> list[int]:
    """About `points` epochs, spread evenly on a log scale from 1 to `epochs`, at which progress is measured."""
    import numpy as np
    return sorted({int(round(e)) for e in np.geomspace(1, epochs, points)} | {epochs})


@torch.no_grad()
def predict(model: DynamicsMLP, context, cond, noise, steps: int):
    """Scaled prediction. Flow matching: walk from `noise` to the answer in `steps` Euler steps."""
    if not model.flow:
        return model(context, cond)
    x = noise.clone() if getattr(model, "noise_scale", 1.0) == 1.0 else noise * model.noise_scale
    d_tau = 1.0 / steps
    for i in range(steps):
        tau = torch.full((x.shape[0], 1), i * d_tau, device=x.device)
        x = x + d_tau * model(context, cond, tau=tau, x_tau=x)
    return x


def train(cfg: RunConfig, data: "PreparedData", on_checkpoint=None) -> tuple[nn.Module, list[float]]:
    """Train one network. Returns it and the average loss of every epoch.

    One epoch = one pass over all training examples in batches of `batch_size`, i.e.
    ceil(examples / batch_size) weight updates. `on_checkpoint(epoch, model)` is called after
    the epochs listed by checkpoint_epochs, to record progress (it must not use torch's global
    random numbers, so recording does not change the training itself).
    """
    device = data.device
    torch.manual_seed(cfg.seed)
    model = build_model(cfg, data).to(device)
    checkpoints = set(checkpoint_epochs(cfg.epochs)) if on_checkpoint else set()
    units = cfg.arch == "units"
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    n_idx, t_idx = data.examples("train", t_min=cfg.context_len - 1)
    shuffler = torch.Generator(device="cpu").manual_seed(cfg.seed)
    losses = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(len(n_idx), generator=shuffler).to(device)
        total = 0.0
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start:start + cfg.batch_size]
            context, cond, target = data.batch("train", n_idx[idx], t_idx[idx])
            if cfg.input_noise > 0:     # noisy measurements of the input states (training only)
                context = context + cfg.input_noise * torch.randn_like(context)
            if units:
                # Every example trains one randomly chosen unit, so a batch has as many rows
                # (and costs as much) as for the flat network.
                which = torch.randint(model.count, (len(idx),), device=device)
                target = torch.gather(target, 1, model.unit_index[which])
                if model.flow:
                    tau = torch.rand(len(idx), 1, device=device)
                    x0 = torch.randn_like(target)
                    if cfg.noise_scale != 1.0:
                        x0 = x0 * cfg.noise_scale
                    x_tau = (1 - tau) * x0 + tau * target
                    velocity = model(context, cond, tau=tau, x_tau=x_tau, units=which)
                    loss = torch.mean((velocity - (target - x0)) ** 2)
                else:
                    loss = torch.mean((model(context, cond, units=which) - target) ** 2)
            elif model.flow:
                tau = torch.rand(len(idx), 1, device=device)       # how far along: 0 = noise, 1 = answer
                x0 = torch.randn_like(target)                       # random starting noise
                if cfg.noise_scale != 1.0:
                    x0 = x0 * cfg.noise_scale
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
        if epoch in checkpoints:
            on_checkpoint(epoch, model)
    return model, losses
