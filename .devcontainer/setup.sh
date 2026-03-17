#!/bin/bash
set -e
set -o pipefail

# Install project + training dependencies
# gdal/geopandas come from the base image (conda), the rest via pip
uv pip install --system -e ".[user-guide]"
uv pip install --system \
  "deepforest>=2.1.0" \
  "wandb" \
  "simple-parsing" \
  "rich" \
  "pytorch-lightning>=2.6.1" \
  "scipy" \
  "torch" \
  "torchvision"

# install node + codex
export NVM_DIR="$HOME/.nvm"
if [ ! -d "$NVM_DIR" ]; then
  curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
fi

# shellcheck disable=SC1090
. "$NVM_DIR/nvm.sh"
nvm install --lts
nvm use --lts

npm i -g @openai/codex

# install claude code
curl -fsSL https://claude.ai/install.sh | bash

# Download TreeAI dataset if not present
DATASET_DIR="/mnt/new-pvc/datasets/treeai/12_RGB_ObjDet_640_fL"
if [ ! -d "$DATASET_DIR" ]; then
  echo "TreeAI dataset not found at $DATASET_DIR"
  echo "Please download from https://zenodo.org/records/15351054 and extract to $DATASET_DIR"
fi

echo "Done!"