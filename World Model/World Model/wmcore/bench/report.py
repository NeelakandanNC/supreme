"""Turn a directory of ``bench.json`` files into the tables a paper needs.

Aggregates over seeds (mean +/- standard error), pairs models up, and emits
Markdown plus optional matplotlib figures.  Deliberately dumb: all of the
science already happened, this file only formats it.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_results(root: str | Path) -> list[dict]:
    """Load every ``bench.json`` under ``root``."""
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(Path(root).rglob("bench.json"))]


def group_by_model(results: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        grouped[r["meta"]["name"]].append(r)
    return dict(grouped)


def _get(d: dict, path: str, default=float("nan")):
    cur = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _agg(runs: list[dict], path: str) -> tuple[float, float, int]:
    vals = [float(_get(r, path)) for r in runs]
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), float(np.std(vals) / max(1, np.sqrt(len(vals)))), len(vals)


#: (column label, dotted path into bench.json, "lower"|"higher" is better)
#: Horizon-dependent rows are appended dynamically by `_headline_metrics`, since
#: the available horizons come from the config and differ between environments.
HEADLINE_METRICS = [
    ("Decision accuracy (model)", "decision.accuracy",                       "higher"),
    ("1-step NLL",              "teacher_forced.nll_1step",                 "lower"),
    ("NLL @ memory-critical",   "teacher_forced.nll_memory_critical",       "lower"),
    ("NLL @ long eval length",  "length_extrapolation.nll_1step",           "lower"),
    ("__HORIZONS__",            "",                                          ""),
    ("Drift slope",             "imagination.drift_slope",                  "lower"),
    ("Probe AUC (state)",       "probes.auc_h",                             "higher"),
    ("Probe AUC (obs control)", "probes.auc_z",                             "higher"),
    ("Effective horizon",       "probes.effective_memory_horizon",          "higher"),
    ("Return (real env)",       "control.return_mean",                      "higher"),
    ("Decision accuracy (agent)", "control.decision_accuracy",              "higher"),
    ("Forgetting on T0 (acc)",  "continual.forgetting_task0_accuracy",      "lower"),
    ("Backbone params",         "cost.params_backbone",                     "lower"),
    ("Step latency (ms)",       "cost.inference_step_ms",                   "lower"),
    ("Train pass (ms)",         "cost.train_pass_ms",                       "lower"),
]


def _headline_metrics(results: list[dict]) -> list[tuple[str, str, str]]:
    """Expand the horizon placeholder using whichever horizons were actually run."""
    horizons = sorted({int(h) for r in results
                       for h in (_get(r, "imagination.mse_at_horizon", {}) or {})})
    picks = [h for h in horizons if h in (10, 20, 50, 100, 200)] or horizons[-2:]
    rows: list[tuple[str, str, str]] = []
    for label, path, direction in HEADLINE_METRICS:
        if label == "__HORIZONS__":
            rows += [(f"Open-loop MSE at step {h}", f"imagination.mse_at_horizon.{h}", "lower")
                     for h in picks]
        else:
            rows.append((label, path, direction))
    return rows


def markdown_table(results: list[dict]) -> str:
    """Main results table: one column per model, mean +/- s.e.m. over seeds."""
    grouped = group_by_model(results)
    names = sorted(grouped)
    if not names:
        return "_no results found_\n"

    lines = ["| Metric | " + " | ".join(names) + " | Better |",
             "|---|" + "---|" * (len(names) + 1)]
    for label, path, direction in _headline_metrics(results):
        cells = []
        stats = []
        for name in names:
            mean, sem, n = _agg(grouped[name], path)
            stats.append(mean)
            cells.append("--" if np.isnan(mean)
                         else (f"{mean:.4g}" if n <= 1 else f"{mean:.4g} ± {sem:.2g}"))
        if len(names) == 2 and not any(np.isnan(s) for s in stats):
            win = 0 if ((stats[0] < stats[1]) == (direction == "lower")) else 1
            cells[win] = f"**{cells[win]}**"
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {direction} |")

    lines.append("")
    lines.append(f"_Seeds per model: " +
                 ", ".join(f"{n}={len(grouped[n])}" for n in names) + "._")
    return "\n".join(lines) + "\n"


def curve_table(results: list[dict], section: str, key: str) -> str:
    """Per-horizon (or per-delay) breakdown, for the appendix."""
    grouped = group_by_model(results)
    names = sorted(grouped)
    xs = sorted({int(k) for n in names for r in grouped[n]
                 for k in _get(r, f"{section}.{key}", {}) or {}})
    if not xs:
        return ""
    lines = [f"| {key} | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for x in xs:
        cells = []
        for n in names:
            mean, sem, cnt = _agg(grouped[n], f"{section}.{key}.{x}")
            cells.append("--" if np.isnan(mean) else f"{mean:.4g}")
        lines.append(f"| {x} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def retention_table(results: list[dict]) -> str:
    grouped = group_by_model(results)
    names = sorted(grouped)
    delays = sorted({int(d) for n in names for r in grouped[n]
                     for d in _get(r, "probes.probe_h", {}) or {}})
    if not delays:
        return ""
    head = ["| delay |"]
    for n in names:
        head.append(f" {n} (state) | {n} (obs) |")
    lines = ["".join(head), "|---|" + "---|" * (2 * len(names))]
    for d in delays:
        cells = []
        for n in names:
            h, _, _ = _agg(grouped[n], f"probes.probe_h.{d}.accuracy")
            z, _, _ = _agg(grouped[n], f"probes.probe_z.{d}.accuracy")
            cells += [f"{h:.3f}" if not np.isnan(h) else "--",
                      f"{z:.3f}" if not np.isnan(z) else "--"]
        lines.append(f"| {d} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _provenance(results: list[dict]) -> str:
    """Settings the numbers were produced under.

    Printed at the top of every report because two of them -- the environment
    kwargs and `critical_loss_weight` -- change what the numbers mean, and a
    table pasted into a paper without them is not interpretable.
    """
    if not results:
        return ""
    meta = results[0]["meta"]
    seeds = sorted({r["meta"]["seed"] for r in results})
    dev = meta.get("device", {})
    return (
        "**Settings.** "
        f"env `{meta['env']}` {meta.get('env_kwargs', {})}; "
        f"train window {meta.get('train_seq_len')}; "
        f"seeds {seeds}; "
        f"device `{dev.get('device')}` on {dev.get('machine')}, "
        f"torch {dev.get('torch')}.\n\n"
        "Note the `memory.critical_loss_weight` used for these runs (see each "
        "run's `config.json`, and `docs/findings.md` §1 for why it matters): at "
        "weight 1.0 every memory layer scores at chance on the decision metric."
    )


def write_report(root: str | Path, out_path: str | Path | None = None) -> Path:
    root = Path(root)
    results = load_results(root)
    out_path = Path(out_path) if out_path else root / "REPORT.md"

    env = results[0]["meta"]["env"] if results else "?"
    body = [
        f"# Memory-layer comparison -- `{env}`",
        "",
        "Every number below comes from runs whose configs differ **only** in "
        "`memory.backbone` (verified by `wmcore.config.assert_comparable`). "
        "V, C, the dataset, the MDN head, the optimiser and the seeds are shared.",
        "",
        _provenance(results),
        "",
        "## Headline results",
        "",
        markdown_table(results),
        "",
        "## Open-loop imagination error by horizon",
        "",
        curve_table(results, "imagination", "mse_at_horizon"),
        "",
        "## Cue retention by delay (linear probe accuracy)",
        "",
        "`state` probes the recurrent features h_t; `obs` probes the latent z_t "
        "alone and is the control -- once the cue leaves the screen it falls to "
        "chance, so any gap above it is memory.",
        "",
        retention_table(results),
        "",
        "## Cost",
        "",
        "Measured on the machine named in each run's `meta.device`.",
        "",
    ]
    out_path.write_text("\n".join(body), encoding="utf-8")
    return out_path


def plot_curves(root: str | Path, out_dir: str | Path | None = None) -> list[Path]:
    """Optional figures.  Silently no-ops without matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    root = Path(root)
    out_dir = Path(out_dir) if out_dir else root
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped = group_by_model(load_results(root))
    written: list[Path] = []

    # Open-loop divergence.
    fig, ax = plt.subplots(figsize=(5, 3.4), dpi=150)
    for name, runs in sorted(grouped.items()):
        lengths = {len(r["imagination"]["mse_step"]) for r in runs}
        n = min(lengths)
        steps = np.array([r["imagination"]["mse_step"][:n] for r in runs], dtype=float)
        # Trim trailing all-NaN columns: a horizon longer than the episodes can
        # supply is not a data point, and plotting it produces a warning and a
        # misleading gap.
        keep = np.isfinite(steps).any(axis=0)
        if not keep.any():
            continue
        last = int(np.max(np.nonzero(keep))) + 1
        steps = steps[:, :last]
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(steps, axis=0)
            sem = np.nanstd(steps, axis=0) / max(1, np.sqrt(len(runs)))
        x = np.arange(1, len(mean) + 1)
        ax.plot(x, mean, label=name)
        ax.fill_between(x, mean - sem, mean + sem, alpha=0.2)
    ax.set(xlabel="imagination horizon (steps)", ylabel="latent MSE", yscale="log")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    p = out_dir / "open_loop_divergence.png"; fig.savefig(p); plt.close(fig)
    written.append(p)

    # Retention curves.
    fig, ax = plt.subplots(figsize=(5, 3.4), dpi=150)
    for name, runs in sorted(grouped.items()):
        delays = sorted({int(d) for r in runs for d in r["probes"]["probe_h"]})
        mh = [np.nanmean([r["probes"]["probe_h"].get(str(d), {}).get("accuracy", np.nan)
                          for r in runs]) for d in delays]
        mz = [np.nanmean([r["probes"]["probe_z"].get(str(d), {}).get("accuracy", np.nan)
                          for r in runs]) for d in delays]
        ax.plot(delays, mh, marker="o", label=f"{name} (state)")
        ax.plot(delays, mz, marker=".", ls="--", alpha=0.5, label=f"{name} (obs)")
    ax.set(xlabel="delay since cue (steps)", ylabel="linear probe accuracy", ylim=(0, 1.02))
    ax.legend(fontsize=7); ax.grid(alpha=0.3); fig.tight_layout()
    p = out_dir / "retention.png"; fig.savefig(p); plt.close(fig)
    written.append(p)
    return written
