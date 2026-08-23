#!/usr/bin/env bash
# CarRacing: the reference environment from Ha & Schmidhuber (2018).
#
# Not the discriminating benchmark -- CarRacing is close to fully observable
# from a single frame, so a memory layer has little to do and CMA-ES variance
# across seeds exceeds any plausible effect. It is here for external
# comparability, because reviewers expect the original paper's environment to
# appear somewhere.
#
# Run in two phases so the expensive controller search is optional.
set -uo pipefail
cd "$(dirname "$0")/.."

# Activate the project virtualenv. Runner scripts must not depend on the caller
# having done this: a bare `python` here resolves to the system interpreter,
# which has no torch, and with `set -e` absent every stage then "succeeds"
# instantly while doing nothing. That failure mode cost a full queue run.
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
python - <<'PYCHECK' || { echo "FATAL: project venv not active (torch missing)"; exit 1; }
import torch, sys           # noqa: F401
PYCHECK

# Refuse to start a second copy: two queues on one GPU thrash and interleave
# their logs into something unreadable.
LOCK="$(pwd)/.$(basename "$0").lock"
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "FATAL: $(basename "$0") already running as PID $(cat "$LOCK")"; exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

# Abort the queue on a failed stage rather than reporting DONE for a no-op.
run() { echo "\$ $*" >> "$LOG"; "$@" >> "$LOG" 2>&1 || { echo "FATAL: stage failed: $*" | tee -a "$LOG"; exit 1; }; }
LOG=runs_carracing.log
say() { echo "=== $(date '+%H:%M:%S')  $*" | tee -a "$LOG"; }

say "CarRacing: data + V + M + benchmark (no controller)"
run python -u wm.py study -c configs/carracing.yaml \
    --backbones lstm,gru,attention,supreme --no-control \
    -o out_dir=runs_carracing 
run python -u wm.py report runs_carracing 
say "DONE: CarRacing -> runs_carracing/REPORT.md"

if [ "${WITH_CONTROL:-0}" = "1" ]; then
  say "CarRacing: CMA-ES controller (slow)"
  for B in lstm supreme; do
    run python -u wm.py controller -c configs/carracing.yaml \
        -o out_dir=runs_carracing -o name="$B" -o memory.backbone="$B" \
        -o controller.generations=25 
  done
  run python -u wm.py report runs_carracing 
  say "DONE: CarRacing control"
fi
