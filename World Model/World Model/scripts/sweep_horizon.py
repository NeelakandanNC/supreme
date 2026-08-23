#!/usr/bin/env python3
"""Memory-horizon sweep: the study's main figure.

Trains the same model on ``TMaze-v0`` at a range of corridor lengths and reports
how far the decision accuracy, the probe retention and the memory-critical NLL
hold up as the required memory horizon grows.

This is the plot that carries the argument: two memory layers that tie on
average NLL will separate here, at the corridor length where one still knows
the cue and the other does not.

    python scripts/sweep_horizon.py --backbones lstm,gru --corridors 6,16,32,64
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wmcore.bench.suite import run_benchmarks
from wmcore.config import Config
from wmcore.train.encode_latents import encode_latents
from wmcore.train.train_memory import train_memory
from wmcore.train.train_vae import train_vae
from wmcore.utils import get_logger

log = get_logger("sweep")


def corridor_config(base: Config, corridor: int, backbone: str, out_dir: str) -> Config:
    cfg = Config.from_dict(copy.deepcopy(base.to_dict()))
    cfg.env.kwargs = {**base.env.kwargs, "corridor_length": corridor}
    episode = corridor + base.env.kwargs.get("cue_steps", 2) + 1 + \
        base.env.kwargs.get("feedback_steps", 2)
    cfg.env.max_episode_steps = episode
    cfg.data.steps_per_rollout = episode
    # Windows must span cue -> decision, so they are full episodes.  Anything
    # shorter removes the dependency from the training signal entirely; see
    # wmcore.train.train_memory.check_window_spans_the_dependency.
    cfg.data.sequence_length = episode
    cfg.data.eval_sequence_length = episode
    cfg.bench.probe_delays = tuple(sorted({0, 1, 2, 4, 8, 16, 32, 64, 128, corridor}
                                          & set(range(episode))))
    cfg.bench.horizons = tuple(h for h in (1, 2, 5, 10, 20, 50, 100) if h < episode - 8)
    cfg.name = f"{backbone}_c{corridor}"
    cfg.memory.backbone = backbone
    cfg.out_dir = out_dir
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="configs/tmaze.yaml")
    ap.add_argument("--backbones", default="lstm")
    ap.add_argument("--corridors", default="6,16,32,64")
    ap.add_argument("--out", default="runs_sweep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollouts", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    base = Config.load(args.config)
    base.seed = args.seed
    if args.rollouts:
        base.data.n_rollouts = args.rollouts
    if args.epochs:
        base.memory.epochs = args.epochs

    rows = []
    for corridor in [int(c) for c in args.corridors.split(",")]:
        # V is trained once per corridor length (the pixel statistics change with
        # episode length) and then shared by every backbone at that length.
        vae_cfg = corridor_config(base, corridor, base.memory.backbone, args.out)
        vae_ckpt = train_vae(vae_cfg)
        encode_latents(vae_cfg)

        for backbone in args.backbones.split(","):
            cfg = corridor_config(base, corridor, backbone, args.out)
            cfg.vision.checkpoint = str(vae_ckpt)
            log.info("=== corridor %d | backbone %s ===", corridor, backbone)
            train_memory(cfg)
            res = run_benchmarks(cfg, include_control=False)
            rows.append({
                "corridor": corridor,
                "backbone": backbone,
                "decision_accuracy": res["decision"]["accuracy"],
                "nll_memory_critical": res["teacher_forced"].get("nll_memory_critical"),
                "nll_1step": res["teacher_forced"]["nll_1step"],
                "effective_memory_horizon": res["probes"]["effective_memory_horizon"],
                "probe_at_corridor": res["probes"]["probe_h"].get(str(corridor), {}).get("accuracy"),
            })
            log.info("  -> decision acc %.3f | probe@%d %s | nll@crit %.3f",
                     rows[-1]["decision_accuracy"], corridor,
                     rows[-1]["probe_at_corridor"], rows[-1]["nll_memory_critical"])

    out = Path(args.out) / "horizon_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n| corridor | backbone | decision acc | probe@corridor | NLL@critical |")
    print("|---|---|---|---|---|")
    for r in rows:
        pa = r["probe_at_corridor"]
        print(f"| {r['corridor']} | {r['backbone']} | {r['decision_accuracy']:.3f} | "
              f"{pa if pa is None else f'{pa:.3f}'} | {r['nll_memory_critical']:.3f} |")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
