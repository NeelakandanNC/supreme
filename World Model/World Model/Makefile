# Convenience targets. All of them assume `source .venv/bin/activate`.
PY ?= python
CFG ?= configs/tmaze.yaml
SEEDS ?= 0 1 2

.PHONY: help env info test smoke data vae memory controller bench study sweep sweep-horizon report clean-runs

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

env:            ## create the virtualenv and install dependencies
	./setup_env.sh

info:           ## print device, registered environments and memory backbones
	$(PY) wm.py info

test:           ## correctness tests -- run these before trusting any number
	$(PY) tests/test_core.py
	$(PY) tests/test_pipeline.py

smoke:          ## end-to-end pipeline on a tiny config (~2 min) -- run after any change
	$(PY) wm.py study -c configs/tmaze_fast.yaml --backbones lstm,gru --no-control

sweep-horizon:  ## the main figure: quality vs. required memory horizon
	$(PY) scripts/sweep_horizon.py --backbones $(or $(BACKBONES),lstm) --corridors 6,16,32,64

data:           ## collect rollouts
	$(PY) wm.py collect -c $(CFG)

vae:            ## train V
	$(PY) wm.py vae -c $(CFG)

memory:         ## train M
	$(PY) wm.py memory -c $(CFG)

controller:     ## train C with CMA-ES
	$(PY) wm.py controller -c $(CFG)

bench:          ## run the memory benchmark suite
	$(PY) wm.py bench -c $(CFG)

study:          ## full A/B for one seed (set BACKBONES=lstm,supreme once Supreme exists)
	$(PY) wm.py study -c $(CFG) --backbones $(or $(BACKBONES),lstm)

sweep:          ## full A/B across $(SEEDS) -- this is what the paper numbers come from
	@for s in $(SEEDS); do \
	  echo "=== seed $$s ==="; \
	  $(PY) wm.py study -c $(CFG) --backbones $(or $(BACKBONES),lstm) --seed $$s || exit 1; \
	done
	$(PY) wm.py report runs

report:         ## regenerate REPORT.md and figures from existing bench.json files
	$(PY) wm.py report runs

clean-runs:     ## delete checkpoints and metrics but KEEP collected data
	find runs -mindepth 1 -maxdepth 1 ! -name data -exec rm -rf {} +
