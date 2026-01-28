#!/bin/bash
set -e


# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Optimization for fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo -e "${GREEN}==> Starting NVIDIA RTX 3050 Training Pipeline Setup${NC}"


# Auto-detect Python interpreter (Virtual Env preferred)
if [ -f ".venv/bin/python3" ]; then
    PYTHON_CMD=".venv/bin/python3"
    echo -e "${GREEN}Using Virtual Environment: .venv${NC}"
else
    PYTHON_CMD="python3"
    echo -e "${GREEN}Using System Python${NC}"
fi

# 1. Verify GPU
echo -e "\n${GREEN}[1/3] Checking GPU Availability...${NC}"
if $PYTHON_CMD check_gpu.py; then
    echo -e "${GREEN}GPU Check Passed!${NC}"
else
    echo -e "${RED}GPU Check Failed! Please ensure CUDA is set up correctly according to INSTALL.md${NC}"
    exit 1
fi

# 2. Check Data
DATA_PATH="./data/Dataset_Final"
echo -e "\n${GREEN}[2/3] Checking Dataset Path ($DATA_PATH)...${NC}"
if [ -d "$DATA_PATH" ]; then
    echo -e "${GREEN}Dataset directory found.${NC}"
else
    echo -e "${RED}Dataset not found at $DATA_PATH!${NC}"
    echo "Please ensure you have unzipped the dataset into ./data/Dataset_Final"
    exit 1
fi

# 3. Run Pipeline
echo -e "\n${GREEN}[3/3] Starting Training Pipeline (SupCon -> ArcFace)...${NC}"
echo "Config: configs/config_tiny.yaml"
echo "Log file will be generated in ./logs/"

# Run the pipeline module
# Phases: supcon (Pretraining) -> arcface (Classification)
$PYTHON_CMD -m pipeline.run_pipeline --config configs/config_tiny.yaml --phases supcon arcface


echo -e "\n${GREEN}==> Pipeline Completed Successfully!${NC}"
