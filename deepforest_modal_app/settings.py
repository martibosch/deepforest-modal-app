"""Settings."""
# TODO: use shared config data class? e.g., see
# https://modal.com/docs/examples/diffusers_lora_finetune

# GPU type
GPU_TYPE = "H100"

# volumes
MODELS_VOLUME_NAME = "models"
MODELS_DIR = "/models"

DATA_VOLUME_NAME = "data"
DATA_DIR = "/data"

# app
APP_NAME = "deepforest"
