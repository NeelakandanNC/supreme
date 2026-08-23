"""Command line interface.

    python -m wmcore.cli <command> --config configs/<file>.yaml [-o key=value ...]

Commands
--------
``collect``     roll out the environment and build the frame store
``vae``         train V
``latents``     freeze V and precompute latents
``memory``      train M
``controller``  train C with CMA-ES
``bench``       run the memory benchmark suite -> bench.json
``report``      aggregate bench.json files -> REPORT.md (+ figures)
``study``       the whole A/B: shared data + shared V, then both memory
                backbones, then the report
``info``        environment / device / registry sanity check

Every command takes ``-o dotted.key=value`` overrides, so a sweep is a shell
loop rather than a directory of near-identical YAML files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the project root importable when invoked as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wmcore.config import Config, assert_comparable  # noqa: E402
from wmcore.utils import get_logger  # noqa: E402

log = get_logger("wmcore.cli")


def _base_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", "-c", default=None, help="path to a YAML/JSON config")
    p.add_argument("--override", "-o", action="append", default=[],
                   metavar="KEY=VALUE", help="dotted-path config override")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--force", action="store_true", help="recompute even if artefacts exist")
    return p


def _load(args) -> Config:
    cfg = Config.load(args.config, args.override) if args.config else Config()
    if args.config is None:
        for ov in args.override:
            cfg.apply_override(ov)
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.save()
    return cfg


# ------------------------------------------------------------- commands --
def cmd_collect(args) -> None:
    from wmcore.data.collect import collect, summarise

    cfg = _load(args)
    store = collect(cfg, force=args.force)
    print(json.dumps(summarise(store), indent=2))


def cmd_vae(args) -> None:
    from wmcore.train.train_vae import train_vae

    print(train_vae(_load(args), force=args.force))


def cmd_latents(args) -> None:
    from wmcore.train.encode_latents import encode_latents

    print(encode_latents(_load(args), force=args.force))


def cmd_memory(args) -> None:
    from wmcore.train.train_memory import train_memory

    print(train_memory(_load(args), force=args.force))


def cmd_controller(args) -> None:
    from wmcore.train.train_controller import evaluate_controller, train_controller

    cfg = _load(args)
    train_controller(cfg, force=args.force)
    print(json.dumps(evaluate_controller(cfg), indent=2))


def cmd_bench(args) -> None:
    from wmcore.bench.suite import run_benchmarks

    cfg = _load(args)
    results = run_benchmarks(cfg, include_control=not args.no_control,
                             include_continual=args.continual)
    print(json.dumps(results["teacher_forced"], indent=2))
    print(json.dumps(results["probes"]["effective_memory_horizon"], indent=2))


def cmd_report(args) -> None:
    from wmcore.bench.report import plot_curves, write_report

    path = write_report(args.root, args.out)
    figs = plot_curves(args.root)
    log.info("wrote %s", path)
    for f in figs:
        log.info("wrote %s", f)
    print(path.read_text(encoding="utf-8"))


def cmd_study(args) -> None:
    """The full apples-to-apples A/B.

    Order matters and is enforced here:
      1. collect the dataset **once**;
      2. train V **once** and freeze it;
      3. for each backbone: train M, train C, benchmark -- everything else equal;
      4. verify the configs differ only in the backbone;
      5. write the report.
    """
    from wmcore.bench.report import plot_curves, write_report
    from wmcore.bench.suite import run_benchmarks
    from wmcore.data.collect import collect, summarise
    from wmcore.train.encode_latents import encode_latents
    from wmcore.train.train_controller import train_controller
    from wmcore.train.train_memory import train_memory
    from wmcore.train.train_vae import train_vae, vae_checkpoint_path

    base = _load(args)
    backbones = args.backbones.split(",")

    log.info("=== shared stage: data ===")
    store = collect(base, force=args.force)
    log.info("dataset: %s", json.dumps(summarise(store)))

    log.info("=== shared stage: V ===")
    vae_ckpt = train_vae(base, force=args.force)
    encode_latents(base, force=args.force)

    configs = []
    for backbone in backbones:
        cfg = Config.from_dict(json.loads(json.dumps(base.to_dict())))
        cfg.name = backbone
        cfg.memory.backbone = backbone
        cfg.vision.checkpoint = str(vae_ckpt)
        cfg.save()
        configs.append(cfg)

    for a, b in zip(configs, configs[1:]):
        offenders = assert_comparable(a, b, strict=False)
        offenders = [o for o in offenders if o not in {"vision.checkpoint"}]
        if offenders:
            raise SystemExit(f"configs are not comparable: {offenders}")
    log.info("comparability check passed: the only difference is memory.backbone")

    for cfg in configs:
        log.info("=== %s: M ===", cfg.name)
        train_memory(cfg, force=args.force)
        if not args.no_control:
            log.info("=== %s: C ===", cfg.name)
            train_controller(cfg, force=args.force)
        log.info("=== %s: benchmark ===", cfg.name)
        run_benchmarks(cfg, include_control=not args.no_control,
                       include_continual=args.continual)

    root = Path(base.out_dir)
    path = write_report(root)
    plot_curves(root)
    print(path.read_text(encoding="utf-8"))


def cmd_info(args) -> None:
    import torch

    from wmcore.envs import registered_envs
    from wmcore.memory import available_backbones
    from wmcore.utils.device import device_info, pick_device

    cfg = _load(args)
    device = pick_device(cfg.device)
    print(json.dumps({
        "device": device_info(device),
        "torch_mps_available": torch.backends.mps.is_available(),
        "environments": registered_envs(),
        "memory_backbones": available_backbones(),
        "run_dir": str(cfg.run_dir),
    }, indent=2))


def main(argv: list[str] | None = None) -> None:
    base = _base_parser()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("collect", parents=[base]).set_defaults(func=cmd_collect)
    sub.add_parser("vae", parents=[base]).set_defaults(func=cmd_vae)
    sub.add_parser("latents", parents=[base]).set_defaults(func=cmd_latents)
    sub.add_parser("memory", parents=[base]).set_defaults(func=cmd_memory)
    sub.add_parser("controller", parents=[base]).set_defaults(func=cmd_controller)

    p_bench = sub.add_parser("bench", parents=[base])
    p_bench.add_argument("--no-control", action="store_true")
    p_bench.add_argument("--continual", action="store_true")
    p_bench.set_defaults(func=cmd_bench)

    p_report = sub.add_parser("report")
    p_report.add_argument("root", nargs="?", default="runs")
    p_report.add_argument("--out", default=None)
    p_report.set_defaults(func=cmd_report)

    p_study = sub.add_parser("study", parents=[base])
    p_study.add_argument("--backbones", default="lstm",
                         help="comma-separated backbone keys, e.g. lstm,supreme")
    p_study.add_argument("--no-control", action="store_true")
    p_study.add_argument("--continual", action="store_true")
    p_study.set_defaults(func=cmd_study)

    sub.add_parser("info", parents=[base]).set_defaults(func=cmd_info)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
