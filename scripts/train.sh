#!/bin/bash
set -e

python -m pipeline.run_pipeline --config configs/config_edge.yaml "$@"
