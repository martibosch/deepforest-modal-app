"""Settings."""
# TODO: use shared config data class? e.g., see
# https://modal.com/docs/examples/diffusers_lora_finetune

# extra pip requirements for the image
PIP_EXTRA_REQS = ["comet-ml==3.49.11"]

# CPU cores
# see https://modal.com/docs/guide/resources#cpu-cores
CPU = 8.0

# GPU type
GPU_TYPE = "H100"

# volumes
MODELS_VOLUME_NAME = "models"
MODELS_DIR = "/models"

DATA_VOLUME_NAME = "data"
DATA_DIR = "/data"

# app
APP_NAME = "deepforest"
TIMEOUT = 60 * 60

# inference args
DEFAULT_IMG_EXT = ".jpeg"
# use these defaults (from `deepforest.main.predict_tile`)
DEFAULT_PATCH_SIZE = 400
DEFAULT_PATCH_OVERLAP = 0.05

# model args
DEFAULT_CREATE_TRAINER_KWARGS = {"max_epochs": 20}
# crop model args
DEFAULT_CROP_MODEL_KWARGS = {"batch_size": 8, "num_workers": 8, "lr": 0.0001}
