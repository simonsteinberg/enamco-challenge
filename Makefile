.PHONY: setup train predict validate all clean

# Override paths via the command line if you want to point at the
# eval_sample/ data instead of the production data, e.g.
#   make all EVAL_TARGETS=eval_sample/eval_targets.csv \
#            HISTORIC=eval_sample/historic.csv

EVAL_TARGETS ?= data/eval_targets.csv
HISTORIC     ?= data/historic.csv
PREDICTIONS  ?= predictions.csv

setup:
	uv sync

train:
	uv run python -m challenge.train --historic $(HISTORIC)

predict:
	uv run python -m challenge.predict \
		--historic $(HISTORIC) \
		--eval-targets $(EVAL_TARGETS) \
		--output $(PREDICTIONS)

# Sanity-check that predictions.csv has the right shape for us to consume.
# This does NOT score your model -- it only confirms the file is well-formed.
validate:
	uv run python -m challenge.format $(PREDICTIONS)

all: setup train predict validate

clean:
	rm -f $(PREDICTIONS)
	rm -rf artifacts/*
	@touch artifacts/.gitkeep
