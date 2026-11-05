"""Command line entry point: train every run of the plan, render videos, build the report.

  1. plan.py lists the runs (variant x seed) needed by the chosen experiments.
  2. Runs whose folder already has a metrics.json are skipped, so an interrupted
     sweep continues where it stopped (--force reruns them).
  3. The rest are spread over --workers processes; each calls
     pipeline.run_experiment (data -> train -> score -> save).
  4. Videos are rendered for plan.VIDEOS (seed 0), then report.py builds
     results/REPORT.md, the figures and results/summary.json.

Examples
--------
    python -m experiments.run                                  # everything, seeds 0 1 2
    python -m experiments.run --only setup collisions          # selected experiments
    python -m experiments.run --quick --out /tmp/q --seeds 0   # tiny smoke test
    python -m experiments.run --set epochs=100                 # override a RunConfig field everywhere
    python -m experiments.run --dry-run                        # list runs, train nothing
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import multiprocessing as mp
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from experiments.config import RunConfig
from experiments.plan import EXPERIMENTS, build_runs, video_jobs


def _worker_init(threads: int) -> None:
    # Limit torch threads per worker so parallel workers don't fight over CPU cores.
    import torch
    torch.set_num_threads(threads)


def _train_one(cfg: RunConfig, out_root: str, device: str, save_model: bool) -> tuple[str, str | None]:
    from experiments.pipeline import run_experiment
    try:
        run_experiment(cfg, Path(out_root) / "runs" / cfg.run_id, Path(out_root) / "datasets",
                       device=device, save_model=save_model)
        return cfg.run_id, None
    except Exception:  # keep the sweep going; report failures at the end
        return cfg.run_id, traceback.format_exc()


def _parse_overrides(pairs: list[str]) -> dict:
    """Turn ["epochs=100", "target=state"] into {"epochs": 100, "target": "state"}."""
    fields = {f.name for f in dataclasses.fields(RunConfig)}
    overrides = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        if key not in fields:
            raise SystemExit(f"--set: unknown RunConfig field '{key}'.")
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        overrides[key] = tuple(value) if isinstance(value, list) else value
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", metavar="EXPERIMENT", help=f"any of {[e.key for e in EXPERIMENTS]}")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", default="results")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2, help="torch threads per worker")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--set", nargs="*", default=[], metavar="FIELD=VALUE")
    parser.add_argument("--quick", action="store_true", help="tiny sizes for a smoke test")
    parser.add_argument("--force", action="store_true", help="rerun finished runs")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out)
    configs = build_runs(args.seeds, args.only, _parse_overrides(args.set), args.quick)
    pending = [c for c in configs if args.force or not (out_root / "runs" / c.run_id / "metrics.json").exists()]
    print(f"{len(configs)} runs; {len(configs) - len(pending)} already done, {len(pending)} to train.")
    if args.dry_run:
        for cfg in pending:
            print("  ", cfg.run_id)
        return

    # Biggest runs first so the pool finishes evenly.
    pending.sort(key=lambda c: -(c.count * c.links * c.context_len))
    failures, started = [], time.time()
    if pending:
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"),
                                 initializer=_worker_init, initargs=(args.threads,)) as pool:
            futures = [pool.submit(_train_one, cfg, str(out_root), args.device, args.save_models) for cfg in pending]
            for done, future in enumerate(as_completed(futures), start=1):
                run_id, error = future.result()
                minutes = (time.time() - started) / 60
                print(f"[{done}/{len(futures)}] {'FAILED' if error else 'ok':6s} {run_id}  "
                      f"({minutes:.1f} min, ~{minutes / done * (len(futures) - done):.1f} min left)", flush=True)
                if error:
                    failures.append((run_id, error))
    for run_id, error in failures:
        print(f"\n--- {run_id} failed ---\n{error}")

    if not args.no_videos:
        from experiments.pipeline import render_video
        seed = min(args.seeds)
        for cfg in configs:
            run_dir = out_root / "runs" / cfg.run_id
            for name, split, file in video_jobs():
                video = out_root / "videos" / file
                if name == cfg.name and cfg.seed == seed and (run_dir / "metrics.json").exists() \
                        and (args.force or not video.exists()):
                    render_video(cfg, run_dir, video, split=split)
                    print(f"video: {video}")

    from experiments.report import build_report
    build_report(out_root)


if __name__ == "__main__":
    main()
