#!/usr/bin/env bash
# Priority queue: headline -> ablations -> CarRacing.
#
# SequenceRecall, continual learning and the corridor sweep are deliberately
# deferred: they broaden the argument, but the headline comparison and its
# ablations are what decide whether there is an argument at all, and CarRacing is
# the only environment in the study that we did not design ourselves.
#
# Safe to re-run: every stage skips work whose checkpoints already exist.
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

LOG=runs_priority.log

# Abort the queue on a failed stage rather than reporting DONE for a no-op.
run() { echo "\$ $*" >> "$LOG"; "$@" >> "$LOG" 2>&1 || { echo "FATAL: stage failed: $*" | tee -a "$LOG"; exit 1; }; }
say() { echo "=== $(date '+%H:%M:%S')  $*" | tee -a "$LOG"; }

# Do not compete with whatever is already training.
while pgrep -f "wm\.py (study|controller)" > /dev/null; do sleep 30; done
say "starting priority queue"

# --- 1. HEADLINE: corridor 32, 3 seeds, with the CMA-ES controller ----------
for SEED in 0 1 2; do
  say "headline  seed $SEED  (lstm, gru, supreme)"
  run python -u wm.py study -c configs/tmaze_hard.yaml \
      --backbones lstm,gru,supreme --seed "$SEED" \
      -o out_dir=runs_seeds -o memory.epochs=150 
done
for SEED in 0 1 2; do
  say "headline  seed $SEED  (attention control)"
  run python -u wm.py study -c configs/tmaze_hard.yaml --backbones attention \
      --seed "$SEED" -o out_dir=runs_seeds -o memory.epochs=150 
done
run python -u wm.py report runs_seeds 
run python -u scripts/make_tables.py 
say "DONE headline -> runs_seeds/REPORT.md"

# --- ABLATIONS: SKIPPED --------------------------------------------------
# supreme-titans / supreme-cms / hope-attention isolate which half of Hope does
# the work. Deliberately not run here: the headline comparison and the external
# environment decide whether there is a result at all, and those come first.
# Run separately with:  ./scripts/run_ablations.sh

# --- 2. CARRACING: the external reference environment ------------------------
say "CarRacing"
run ./scripts/run_carracing.sh
run python -u scripts/make_tables.py 
say "DONE CarRacing -> runs_carracing/REPORT.md"

say "PRIORITY QUEUE COMPLETE"
