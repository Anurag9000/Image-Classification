#!/bin/bash
set -e

python -m pipeline.run_pipeline --config configs/garbage_edge.yaml "$@"
