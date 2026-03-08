"""Settings."""

import os

# TODO: use shared config data class? e.g., see
# https://modal.com/docs/examples/diffusers_lora_finetune

# extra pip requirements for the image
DEFAULT_PIP_EXTRA_REQS = ["comet-ml==3.56.0"]


def _parse_pip_extra_reqs(value: str | None) -> list[str]:
    if not value:
        return []
    return [req.strip() for req in value.split(",") if req.strip()]


_env_pip_extra_reqs = _parse_pip_extra_reqs(os.getenv("DEEPFOREST_PIP_EXTRA_REQS"))
PIP_EXTRA_REQS = _env_pip_extra_reqs or DEFAULT_PIP_EXTRA_REQS

# CPU cores
# see https://modal.com/docs/guide/resources#cpu-cores
CPU = 8.0

# GPU types - training benefits from high-memory GPUs; inference can use cheaper ones
TRAIN_GPU_TYPE = "H100"
INFERENCE_GPU_TYPE = "A10G"

# volumes
MODELS_VOLUME_NAME = "deepforest-models"
MODELS_DIR = "/models"

DATA_VOLUME_NAME = "deepforest-data"
DATA_DIR = "/data"

# app
APP_NAME = "deepforest"
TIMEOUT = 60 * 60

# inference args
DEFAULT_IMG_EXT = ".jpeg"
# use these to predict entire tiles by default
DEFAULT_PATCH_SIZE = 2000
DEFAULT_PATCH_OVERLAP = 0

# model args
DEFAULT_MODEL_NAME = "weecology/deepforest-tree"
DEFAULT_MODEL_REVISION = "main"

# crown model args
DEFAULT_CROWN_CONFIG_ARGS = {
    "batch_size": 4,
    "workers": int(CPU) // 2,
    "train": {
        "lr": 1e-4,
        "preload_images": True,
        "scheduler": {"type": "ReduceLROnPlateau"},
    },
    "validation": {
        "preload_images": True,
        "val_accuracy_interval": 5,
    },
}
DEFAULT_CREATE_TRAINER_KWARGS = {
    "max_epochs": 50,
    "precision": "bf16-mixed",
    "accumulate_grad_batches": 4,
}

# crop model args
DEFAULT_CROP_MODEL_KWARGS = {
    "config_args": {
        "cropmodel": {
            "batch_size": 256,
            "num_workers": int(CPU) // 2,
            "resize": [128, 128],
            "lr": 1e-4,
            "train": {"preload_images": True},
            "validation": {"preload_images": True},
        },
    },
}
DEFAULT_CROP_MODEL_CREATE_TRAINER_KWARGS = {
    "max_epochs": 100,
    "precision": "bf16-mixed",
}
