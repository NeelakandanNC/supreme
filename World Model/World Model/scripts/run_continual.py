#!/usr/bin/env python3
"""Continual learning, run in a regime where the metric is meaningful.

The earlier attempt used corridor 16 with the LSTM core, which never learns any
single task there -- so its forgetting matrix was just its output bias reflected
across the task variants, and the benchmark correctly flagged
``informative: false``.

Forgetting can only be measured against something learned. This script therefore
runs the protocol at **corridor 6**, the shortest setting at which all three
cores solve the single-task version (see docs/findings.md §1), so every model
starts each task from genuine competence and any drop is genuine forgetting.

    python scripts/run_continual.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wmcore.bench.continual import run_continual
from wmcore.config import Config
from wmcore.utils import get_logger

log = get_logger("continual")

CORRIDOR = 6
BACKBONES = ("lstm", "gru", "supreme")
SEEDS = (0, 1, 2)


def main() -> None:
    out_dir = Path("runs_continual")
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for seed in SEEDS:
        for backbone in BACKBONES:
            cfg = Config.load("configs/tmaze.yaml", [
                "out_dir=runs_continual",
                f"name={backbone}",
                f"memory.backbone={backbone}",
                f"seed={seed}",
                f"env.kwargs={{'corridor_length': {CORRIDOR}, 'n_cues': 2, "
                "'distractors': 3, 'cue_steps': 2, 'feedback_steps': 2}}",
                f"env.max_episode_steps={CORRIDOR + 5}",
                f"data.steps_per_rollout={CORRIDOR + 5}",
                f"data.sequence_length={CORRIDOR + 5}",
                f"data.eval_sequence_length={CORRIDOR + 5}",
                "data.n_rollouts=2000",
                "bench.continual_tasks=3",
                "bench.continual_epochs_per_task=60",
            ])
            log.info("=== continual | %s | seed %d ===", backbone, seed)
            res = run_continual(cfg)
            res["backbone"] = backbone
            res["seed"] = seed
            res["corridor"] = CORRIDOR
            results.append(res)
            (out_dir / "continual.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8")

    print("\n| backbone | seed | learned acc | final acc | BWT (acc) | forgetting T0 | informative |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['backbone']} | {r['seed']} | {r['learned_accuracy_mean']:.3f} | "
              f"{r['final_accuracy_mean']:.3f} | {r['backward_transfer_accuracy']:+.3f} | "
              f"{r['forgetting_task0_accuracy']:+.3f} | {r['informative']} |")


if __name__ == "__main__":
    main()
