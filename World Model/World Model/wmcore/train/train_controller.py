"""Stage 4 -- train C with an evolution strategy.

Two modes, both driven by the same code:

``controller.train_in_dream = False``
    candidates are scored in the real environment (Ha & Schmidhuber, Sec. 3-4);
``controller.train_in_dream = True``
    candidates are scored inside M's imagination and only the *final* policy
    touches the real environment (their Sec. 5).  This is the strongest
    end-to-end test of a memory layer -- see :mod:`wmcore.dream.dream_env`.

Parallelism
-----------
Candidates are evaluated in ``spawn``ed workers pinned to **CPU**, not MPS.
Six processes each opening a Metal context costs more in driver overhead and
memory than the models save in compute: V is 4.3 M parameters and M is under
half a million, so a single-frame forward pass is ~1 ms on an M4 performance
core.  Keeping workers on CPU also leaves the GPU free and keeps peak unified
memory well inside the 10 GB budget.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from wmcore.config import Config
from wmcore.controller.agent import WorldModelAgent
from wmcore.controller.es import make_strategy
from wmcore.controller.linear import LinearController
from wmcore.data.collect import collect
from wmcore.envs import make_env_from_config
from wmcore.utils import JSONLLogger, Stopwatch, get_logger, pick_device, seed_everything
from wmcore.utils.parallel import ProcessMapper

log = get_logger(__name__)

# Per-process cache, populated once by the pool initializer.
_WORKER: dict = {}


def controller_path(cfg: Config) -> Path:
    return cfg.run_dir / "controller.npz"


# ---------------------------------------------------------------- worker --
def _init_worker(cfg_dict: dict) -> None:
    """Load V and M once per process."""
    from wmcore.train.train_memory import load_memory
    from wmcore.train.train_vae import load_vae

    torch.set_num_threads(1)  # six workers each spawning BLAS threads thrashes the cache
    cfg = Config.from_dict(cfg_dict)
    device = torch.device("cpu")

    _WORKER["cfg"] = cfg
    _WORKER["device"] = device
    _WORKER["vae"] = load_vae(cfg, device)
    _WORKER["memory"] = load_memory(cfg, device)
    _WORKER["env"] = make_env_from_config(cfg.env)

    if cfg.controller.train_in_dream:
        from wmcore.data.collect import dataset_dir
        from wmcore.data.store import RolloutStore
        from wmcore.dream import DreamEnv, collect_initial_latents

        store = RolloutStore(dataset_dir(cfg))
        _WORKER["dream"] = DreamEnv(
            _WORKER["memory"], collect_initial_latents(store),
            _WORKER["env"].action_space, device,
            temperature=cfg.memory.temperature, horizon=cfg.controller.dream_horizon,
        )


def _make_agent(params: np.ndarray) -> WorldModelAgent:
    cfg: Config = _WORKER["cfg"]
    memory = _WORKER["memory"]
    dim = (memory.controller_input_dim if cfg.controller.hidden_input else memory.latent_dim)
    controller = LinearController(dim, _WORKER["env"].action_space).set_params(params)
    return WorldModelAgent(_WORKER["vae"], memory, controller, _WORKER["device"],
                           use_hidden=cfg.controller.hidden_input)


def _evaluate_candidate(job: tuple) -> tuple[int, float, float]:
    """Score one candidate.  Returns ``(index, mean_return, mean_accuracy)``."""
    idx, params, seeds = job
    cfg: Config = _WORKER["cfg"]
    agent = _make_agent(np.asarray(params))

    returns, accs = [], []
    if cfg.controller.train_in_dream:
        dream = _WORKER["dream"]
        for s in seeds:
            total = _dream_rollout(agent, dream, int(s), cfg.controller.dream_horizon)
            returns.append(total)
    else:
        env = _WORKER["env"]
        for s in seeds:
            res = agent.rollout(env, seed=int(s), max_steps=cfg.env.max_episode_steps)
            returns.append(res.total_reward)
            acc = res.accuracy
            if not np.isnan(acc):
                accs.append(acc)
    return idx, float(np.mean(returns)), float(np.mean(accs)) if accs else float("nan")


@torch.no_grad()
def _dream_rollout(agent: WorldModelAgent, dream, seed: int, horizon: int) -> float:
    """Roll the controller out inside M's latent imagination."""
    dream.reset(seed=seed)
    total = 0.0
    for _ in range(horizon):
        action = agent.controller.act(dream.state_features())
        _, reward, terminated, truncated, _ = dream.step(action)
        total += reward
        if terminated or truncated:
            break
    return total


# --------------------------------------------------------------- driver --
def train_controller(cfg: Config, *, force: bool = False) -> Path:
    out_path = controller_path(cfg)
    if out_path.exists() and not force:
        log.info("C already trained -> %s", out_path)
        return out_path

    seed_everything(cfg.seed)
    collect(cfg)  # ensure dataset/latents exist for dream seeding

    device = pick_device(cfg.device)
    from wmcore.train.train_memory import load_memory

    memory = load_memory(cfg, device)
    env = make_env_from_config(cfg.env)
    dim = memory.controller_input_dim if cfg.controller.hidden_input else memory.latent_dim
    template = LinearController(dim, env.action_space)
    env.close()

    log.info("C: %d parameters (input %d = z%s) | %s | pop=%d gens=%d",
             template.n_params, dim,
             "+h" if cfg.controller.hidden_input else " only",
             "dream" if cfg.controller.train_in_dream else "real env",
             cfg.controller.population, cfg.controller.generations)

    strategy = make_strategy(cfg.controller.algo, template.n_params,
                             population=cfg.controller.population,
                             sigma=cfg.controller.sigma_init, seed=cfg.seed)

    n_workers = max(1, min(cfg.controller.n_workers, os.cpu_count() or 1))
    watch = Stopwatch("controller")
    rng = np.random.default_rng(cfg.seed)
    best_params, best_fitness = template.get_params(), -np.inf

    with ProcessMapper(n_workers, initializer=_init_worker, initargs=(cfg.to_dict(),),
                       label="CMA-ES evaluation") as mapper, \
            JSONLLogger(cfg.run_dir / "metrics.jsonl",
                        extra={"stage": "controller", "seed": cfg.seed,
                               "backbone": cfg.memory.backbone}) as jlog:
        for gen in range(cfg.controller.generations):
            solutions = strategy.ask()
            # All candidates in a generation face the *same* episode seeds:
            # common random numbers remove most of the between-candidate
            # variance, which is worth several generations of budget.
            seeds = rng.integers(0, 2**31 - 1, size=cfg.controller.episodes_per_candidate)
            jobs = [(i, sol, seeds) for i, sol in enumerate(solutions)]
            results = mapper.map(_evaluate_candidate, jobs)

            fitness = np.zeros(len(solutions))
            accs = np.full(len(solutions), np.nan)
            for i, f, a in results:
                fitness[i], accs[i] = f, a
            strategy.tell(solutions, fitness)

            if fitness.max() > best_fitness:
                best_fitness = float(fitness.max())
                best_params = solutions[int(np.argmax(fitness))].copy()

            watch.lap(f"gen{gen}")
            jlog.write(generation=gen, fitness_mean=float(fitness.mean()),
                       fitness_max=float(fitness.max()),
                       fitness_std=float(fitness.std()),
                       accuracy_mean=float(np.nanmean(accs)) if not np.all(np.isnan(accs)) else None)
            if gen % 5 == 0 or gen == cfg.controller.generations - 1:
                # Accuracy is undefined when training in the dream (no ground-truth
                # decision to score against) and for continuous-action envs.
                acc = float(np.nanmean(accs)) if not np.all(np.isnan(accs)) else float("nan")
                log.info("gen %3d | fitness mean %8.2f max %8.2f | acc %.3f | %.1fs",
                         gen, fitness.mean(), fitness.max(), acc, watch.elapsed)

    final = strategy.best()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, params=final, best_params=best_params,
             best_fitness=best_fitness, input_dim=dim,
             hidden_input=cfg.controller.hidden_input,
             trained_in_dream=cfg.controller.train_in_dream,
             timing=np.array(watch.elapsed))
    log.info("saved C -> %s (best training fitness %.2f)", out_path, best_fitness)
    return out_path


def evaluate_controller(cfg: Config, *, episodes: int | None = None,
                        params_key: str = "params") -> dict:
    """Score the trained controller in the **real** environment.

    Always real, even when C was trained in the dream: the dream score is a
    property of the model, the real score is the result.
    """
    from wmcore.train.train_memory import load_memory
    from wmcore.train.train_vae import load_vae

    episodes = episodes or cfg.controller.eval_episodes
    device = torch.device("cpu")
    vae, memory = load_vae(cfg, device), load_memory(cfg, device)
    env = make_env_from_config(cfg.env)
    data = np.load(controller_path(cfg))
    dim = int(data["input_dim"])
    controller = LinearController(dim, env.action_space).set_params(data[params_key])
    agent = WorldModelAgent(vae, memory, controller, device,
                            use_hidden=bool(data["hidden_input"]))

    returns, accs = [], []
    for i in range(episodes):
        res = agent.rollout(env, seed=10_000 + i, max_steps=cfg.env.max_episode_steps)
        returns.append(res.total_reward)
        if not np.isnan(res.accuracy):
            accs.append(res.accuracy)
    env.close()

    out = {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "return_sem": float(np.std(returns) / np.sqrt(len(returns))),
        "episodes": episodes,
        "trained_in_dream": bool(data["trained_in_dream"]),
    }
    if accs:
        out["decision_accuracy"] = float(np.mean(accs))
    return out
