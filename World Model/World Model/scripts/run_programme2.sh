#!/usr/bin/env bash
# Stage 2 of the benchmark programme: the gaps identified in review.
#
# Runs after scripts/run_programme.sh. Kept separate rather than appended so the
# first script's queue is not disturbed mid-flight.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=runs_programme2.log
say() { echo "=== $(date '+%H:%M:%S')  $*" | tee -a "$LOG"; }

# Wait for stage 1 to finish before competing for the GPU.
while pgrep -f "run_programme.sh" > /dev/null; do sleep 60; done
say "stage 1 finished; starting stage 2"

# --- A. attention control at the headline operating point, 3 seeds ----------
for SEED in 0 1 2; do
  say "corridor-32 attention  seed $SEED"
  python -u wm.py study -c configs/tmaze_hard.yaml --backbones attention \
      --seed "$SEED" -o out_dir=runs_seeds -o memory.epochs=150 >> "$LOG" 2>&1
done
python -u wm.py report runs_seeds >> "$LOG" 2>&1
say "DONE: attention control -> runs_seeds/"

# --- B. ablation set, re-run post stability-fix, at corridor 32, 3 seeds ----
# supreme-titans / supreme-cms / hope-attention isolate the two halves of Hope.
for SEED in 0 1 2; do
  say "corridor-32 ablations  seed $SEED"
  python -u wm.py study -c configs/tmaze_hard.yaml \
      --backbones supreme-titans,supreme-cms,hope-attention \
      --seed "$SEED" -o out_dir=runs_seeds -o memory.epochs=150 \
      --no-control >> "$LOG" 2>&1
done
python -u wm.py report runs_seeds >> "$LOG" 2>&1
say "DONE: ablations -> runs_seeds/"

# --- C. critical_loss_weight = 1 ablation AT the headline operating point ----
say "critical_loss_weight = 1 ablation"
python -u wm.py study -c configs/tmaze_hard.yaml \
    --backbones lstm,gru,attention,supreme -o out_dir=runs_w1 \
    -o memory.critical_loss_weight=1 -o memory.epochs=150 \
    --no-control >> "$LOG" 2>&1
python -u wm.py report runs_w1 >> "$LOG" 2>&1
say "DONE: weight-1 ablation -> runs_w1/REPORT.md"

# --- D. CarRacing, the external reference environment ------------------------
say "CarRacing"
./scripts/run_carracing.sh >> "$LOG" 2>&1
say "DONE: CarRacing -> runs_carracing/REPORT.md"

say "STAGE 2 COMPLETE"
