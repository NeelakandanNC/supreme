#!/usr/bin/env python3
"""Generate the LaTeX results tables from ``bench.json`` artefacts.

Written to ``../../results_tables.tex`` -- next to ``supreme.tex`` -- and pulled
in with ``\\input``.  The point is that no number in the paper is ever typed by
hand: the tables are a pure function of the run artefacts, so the document cannot
drift from the data, and re-running this after new seeds land updates the paper.

Every cell is mean +/- standard error over seeds, with the seed count printed
underneath so a single-seed entry is never mistaken for a converged one.

    python scripts/make_tables.py [--runs runs_seeds] [--out ../../results_tables.tex]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Presentation order and display names.  Controls first, then the model under
# test, then its ablations -- so a reader meets the baselines before the claim.
ORDER = [
    ("lstm", r"\code{lstm} (MDN-RNN)"),
    ("gru", r"\code{gru}"),
    ("attention", r"\code{attention}"),
    ("supreme", r"\textbf{\code{supreme}} (Hope)"),
    ("supreme-titans", r"\code{supreme-titans}"),
    ("supreme-cms", r"\code{supreme-cms}"),
    ("hope-attention", r"\code{hope-attention}"),
]

METRICS = [
    ("Decision accuracy $\\uparrow$", "decision.accuracy", 3),
    ("NLL @ memory-critical $\\downarrow$", "teacher_forced.nll_memory_critical", 2),
    ("NLL, all steps $\\downarrow$", "teacher_forced.nll_1step", 2),
    ("Effective memory horizon $\\uparrow$", "probes.effective_memory_horizon", 1),
    ("Probe AUC, state $\\uparrow$", "probes.auc_h", 3),
    ("Return, real env $\\uparrow$", "control.return_mean", 3),
    ("Backbone params", "cost.params_backbone", 0),
]


def get(d: dict, path: str):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur if isinstance(cur, (int, float)) else None


def load(runs_root: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for bj in sorted(runs_root.rglob("bench.json")):
        d = json.loads(bj.read_text())
        out[d["meta"]["backbone"]].append(d)
    return dict(out)


def cell(runs: list[dict], path: str, places: int) -> str:
    vals = [v for v in (get(r, path) for r in runs) if v is not None]
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return "--"
    m = float(np.mean(vals))
    if len(vals) == 1:
        return f"{m:,.0f}" if places == 0 else f"{m:.{places}f}"
    sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
    if places == 0:
        return f"{m:,.0f}"
    return f"${m:.{places}f}\\!\\pm\\!{sem:.{places}f}$"


def results_table(groups: dict[str, list[dict]], caption: str, label: str) -> str:
    present = [(k, name) for k, name in ORDER if k in groups]
    if not present:
        return f"% no runs found for {label}\n"
    lines = [
        r"\begin{table}[h]", r"\centering", r"\small",
        r"\begin{tabular}{l" + "r" * len(present) + "}", r"\toprule",
        "Metric & " + " & ".join(n for _, n in present) + r" \\", r"\midrule",
    ]
    for label_m, path, places in METRICS:
        lines.append(label_m + " & "
                     + " & ".join(cell(groups[k], path, places) for k, _ in present)
                     + r" \\")
    lines += [
        r"\midrule",
        "Seeds & " + " & ".join(str(len(groups[k])) for k, _ in present) + r" \\",
        r"\bottomrule", r"\end{tabular}",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}", "",
    ]
    return "\n".join(lines)


def retention_table(groups: dict[str, list[dict]]) -> str:
    present = [(k, n) for k, n in ORDER if k in groups]
    delays = sorted({int(d) for k, _ in present for r in groups[k]
                     for d in r.get("probes", {}).get("probe_h", {})})
    if not delays:
        return ""
    lines = [
        r"\begin{table}[h]", r"\centering", r"\small",
        r"\begin{tabular}{l" + "c" * len(delays) + "}", r"\toprule",
        "Delay since cue & " + " & ".join(str(d) for d in delays) + r" \\", r"\midrule",
    ]
    for k, name in present:
        vals = []
        for d in delays:
            a = [r["probes"]["probe_h"].get(str(d), {}).get("accuracy")
                 for r in groups[k]]
            a = [x for x in a if x is not None and not np.isnan(x)]
            vals.append(f"{np.mean(a):.2f}" if a else "--")
        lines.append(f"{name} & " + " & ".join(vals) + r" \\")
    # the control: identical for every model, so shown once
    ctrl = []
    any_runs = groups[present[0][0]]
    for d in delays:
        a = [r["probes"]["probe_z"].get(str(d), {}).get("accuracy") for r in any_runs]
        a = [x for x in a if x is not None and not np.isnan(x)]
        ctrl.append(f"{np.mean(a):.2f}" if a else "--")
    lines += [
        r"\midrule",
        r"\emph{observation control} & " + " & ".join(ctrl) + r" \\",
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Linear probe accuracy of the cue from the recurrent state, by "
        r"delay. The observation control probes $z_t$ alone: it is at ceiling "
        r"while the cue is on screen and at chance from the next step onward, so "
        r"any value above it is memory and nothing else.}",
        r"\label{tab:retention}", r"\end{table}", "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_seeds")
    ap.add_argument("--carracing", default="runs_carracing")
    ap.add_argument("--out", default="../../results_tables.tex")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    parts = [
        "% Auto-generated by scripts/make_tables.py -- do not edit by hand.",
        "% Regenerate after new runs land; every number is a function of bench.json.",
        "",
    ]

    main_groups = load(root / args.runs)
    if main_groups:
        parts.append(results_table(
            main_groups,
            r"\code{TMaze-v0}, corridor 32 (a 32-step required memory horizon). "
            r"Mean $\pm$ standard error over seeds. All columns share the same "
            r"data, the same frozen $V$, the same MDN head, loss, optimiser and "
            r"seeds; only the memory backbone differs.",
            "tab:headline"))
        parts.append(retention_table(main_groups))

    car = load(root / args.carracing)
    if car:
        parts.append(results_table(
            car,
            r"\code{CarRacing-v0} \cite{ha2018world}. Reported for external "
            r"comparability rather than as a discriminating benchmark: the task "
            r"is close to fully observable from a single frame, so a memory layer "
            r"has little to do.",
            "tab:carracing"))

    if len(parts) == 3:
        parts.append("% no completed runs yet")

    out = (root / args.out).resolve()
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out}  ({len(main_groups)} backbones in {args.runs}, "
          f"{len(car)} in {args.carracing})")


if __name__ == "__main__":
    main()
