#!/usr/bin/env bash
# The full benchmark programme, run in order, one heavy job at a time.
#
# Sequential on purpose: the M4 has 10 cores and ~10 GB usable, and CMA-ES
# already uses 6 worker processes. Two studies at once thrash both.
#
# Each stage is skipped if its artefacts already exist, so the script is safe to
# re-run after an interruption.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=runs_programme.log
say() { echo "=== $(date '+%H:%M:%S')  $*" | tee -a "$LOG"; }

# --- 1 + 3: multi-seed corridor 32, WITH the CMA-ES controller ---------------
for SEED in 0 1 2; do
  say "corridor-32  seed $SEED"
  python -u wm.py study -c configs/tmaze_hard.yaml \
      --backbones lstm,gru,supreme --seed "$SEED" \
      -o out_dir=runs_seeds -o memory.epochs=150 >> "$LOG" 2>&1
done
python -u wm.py report runs_seeds >> "$LOG" 2>&1
say "DONE: multi-seed corridor 32 -> runs_seeds/REPORT.md"

# --- 4: SequenceRecall (the capacity axis) -----------------------------------
say "SequenceRecall"
python -u wm.py study -c configs/recall.yaml \
    --backbones lstm,gru,supreme -o out_dir=runs_recall >> "$LOG" 2>&1
python -u wm.py report runs_recall >> "$LOG" 2>&1
say "DONE: SequenceRecall -> runs_recall/REPORT.md"

# --- 5: continual learning, in a regime every model can actually learn -------
say "continual learning"
python -u scripts/run_continual.py >> "$LOG" 2>&1
say "DONE: continual -> runs_continual/continual.json"

# --- 2: the corridor sweep figure -------------------------------------------
say "corridor sweep"
python -u scripts/sweep_horizon.py --backbones lstm,gru,supreme \
    --corridors 6,16,32,48 --rollouts 1200 --epochs 150 \
    --out runs_sweep >> "$LOG" 2>&1
say "DONE: sweep -> runs_sweep/horizon_sweep.json"

say "PROGRAMME COMPLETE"
