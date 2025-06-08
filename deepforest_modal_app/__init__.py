"""Modal app for DeepForest model training and inference."""

import json
import os
from collections.abc import Mapping
from os import path
from typing import Optional

import modal
from grpclib import GRPCError

# type annotations
# type hint for path-like objects
PathType = str | os.PathLike
# type hint for keyword arguments
KwargsType = Mapping | None

# TODO: use shared config data class? e.g., see
# https://modal.com/docs/examples/diffusers_lora_finetune


GPU_TYPE = "H100"

# create Modal image with required dependencies
MODELS_DIR = "/cache"
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

# path to base project directory
# REMOTE_IMAGES_DIR = path.join("/root/images")
DATA_DIR = "/data"
data_volume = modal.Volume.from_name("data", create_if_missing=True)
# with data_volume.batch_upload() as batch:
#     batch.put_directory(LOCAL_DATA_DIR, ".")

app = modal.App(name="deepforest")
image = (
    modal.Image.micromamba("3.11")
    .micromamba_install(
        "geopandas=1.0.1",
        "opencv=4.11.0",
        channels=["conda-forge"],
    )
    .pip_install(
        "albumentations==1.4.24",
        "deepforest==1.5.2",
    )
    .env({"HF_HUB_CACHE": MODELS_DIR})
    # .add_local_dir(LOCAL_DATA_DIR, remote_path=DATA_DIR)
)

with image.imports():
    import tempfile
    import time

    import geopandas as gpd
    import pandas as pd
    from deepforest import main as deepforest_main


@app.cls(
    image=image, gpu=GPU_TYPE, volumes={MODELS_DIR: cache_volume, DATA_DIR: data_volume}
)
class DeepForestApp:
    """DeepForest app.

    Parameters
    ----------
    model_name : str
        Name of the model to load from Hugging Face Hub. Ignored if
        `checkpoint_filename` is provided.
    model_revision : str
        Revision of the model to load from Hugging Face Hub. Ignored if
        `checkpoint_filename` is provided.
    checkpoint_filename : str
        Name of the checkpoint file to load from the cache volume.
    config_filepath : str
        Path to the JSON file with model configuration. If not provided,
        the model will be configured to use all available GPUs and 4 workers.
    """

    model_name: str = modal.parameter(default="weecology/deepforest-tree")
    model_revision: str = modal.parameter(default="main")
    checkpoint_filename: str = modal.parameter(default="")
    config_filepath: str = modal.parameter(default="")

    @modal.enter()
    def load_model(self) -> None:
        """Load the model from Hugging Face Hub or a checkpoint.

        This method is run at container startup only.
        """
        if self.checkpoint_filename != "":
            # TODO: how does this affect build time?
            checkpoint_filepath = path.join(MODELS_DIR, self.checkpoint_filename)
            print(f"Loading model from checkpoint: {checkpoint_filepath}")

            model = deepforest_main.deepforest.load_from_checkpoint(
                checkpoint_filepath,
            )
        else:
            # load the default release checkpoint
            model = deepforest_main.deepforest()
            model.load_model(model_name=self.model_name)
        if self.config_filepath == "":
            # by default, use all available GPUs and 4 workers
            self.config_dict = {"gpus": -1, "workers": 4}
        else:
            # load the config from a JSON file
            with open(self.config_filepath, encoding="utf-8") as src:
                self.config_dict = json.load(src)
        for key, value in self.config_dict.items():
            model.config[key] = value
        self.model = model

    @modal.method()
    def retrain_crown_model(
        self,
        train_df: pd.DataFrame | gpd.GeoDataFrame,
        remote_img_dir: PathType,
        *,
        checkpoint_filename: PathType | None = None,
        test_df: pd.DataFrame | gpd.GeoDataFrame | None = None,
        train_config: Mapping | None = None,
        validation_config: Mapping | None = None,
        dst_filename: str | None = None,
        **create_trainer_kwargs: KwargsType,
    ) -> None:  # deepforest_main.deepforest:
        """Retrain the DeepForest model with the provided training data.

        Parameters
        ----------
        train_df : pd.DataFrame or gpd.GeoDataFrame
            Training data, as pandas or geopandas data frame with bounding box
            annotations.
        remote_img_dir : path-like
            Path to the remote directory with images, relative to the data volume root.
        checkpoint_filename : path-like, optional
            Name of the checkpoint file to load from the cache volume. If not provided,
            the model loaded at container startup will be used.
        test_df : pd.DataFrame or gpd.GeoDataFrame, optional
            Test data to use for validation during training. If not provided, training
            will be performed without validation.
        train_config, validation_config : dict-like, optional
            Configuration for the training and validation, passed to the model's
            `config` attribute under the keys "train" and "validation" respectively.
        dst_filename : str, optional
            Name of the file to save the retrained model to. If not provided, a file
            name will be generated based on the current timestamp.

        """

        def save_annot_df(annot_df, dst_filepath):
            """Save the annotated data frame."""
            # we are just using a function to DRY any eventual required preprocessing
            annot_df.to_csv(dst_filepath)
            return dst_filepath

        if checkpoint_filename is not None:
            checkpoint_filepath = path.join(MODELS_DIR, checkpoint_filename)
            print(f"Loading model from checkpoint: {checkpoint_filepath}")
            model = deepforest_main.deepforest.load_from_checkpoint(
                checkpoint_filepath,
            )
        else:
            model = self.model

        # prepend volume path to the remote image directory
        remote_img_dir = path.join(DATA_DIR, remote_img_dir)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # save training data to a temporary file
            train_df_filepath = path.join(tmp_dir, "train.csv")
            save_annot_df(train_df, train_df_filepath)
            model.config["train"]["csv_file"] = train_df_filepath
            model.config["train"]["root_dir"] = remote_img_dir
            if test_df is not None:
                # save training data to a temporary file
                test_df_filepath = path.join(tmp_dir, "test.csv")
                save_annot_df(test_df, test_df_filepath)
                model.config["validation"]["root_dir"] = remote_img_dir
                model.config["validation"]["csv_file"] = test_df_filepath

            if train_config is None:
                # use all gpus by default
                train_config = {"gpus": -1}
            for key, value in train_config.items():
                model.config["train"][key] = value

            if validation_config is None:
                validation_config = {}
            for key, value in validation_config.items():
                model.config["validation"][key] = value

            model.create_trainer(**create_trainer_kwargs)
            start_time = time.time()
            model.trainer.fit(model)
        print(f"--- Model retrained in {(time.time() - start_time):.2f} seconds ---")

        # TODO: replace model attribute with the trained model?
        # self.model = model
        if dst_filename is None:
            dst_filename = f"deepforest-retrained-{time.strftime('%Y%m%d_%H%M%S')}.pl"
        dst_filepath = path.join(MODELS_DIR, dst_filename)
        # model.save_model(dst_filepath)
        model.trainer.save_checkpoint(dst_filepath)
        print(f"Saved checkpoint to {dst_filepath}")
        # return model

    @modal.method()
    def predict(
        self,
        img_filename: str,
        remote_img_dir: str,
        *,
        checkpoint_filename: str | None = None,
        patch_size: int = 400,
        patch_overlap: float = 0.05,
        iou_threshold: float = 0.15,
    ) -> gpd.GeoDataFrame:
        """Predict tree crown bounding boxes using the a DeepForest-like model.

        Parameters
        ----------
        img_filename : str
            File name of the image to predict on.
        remote_img_dir : str
            Path to the remote directory with images, relative to the data volume root.
        checkpoint_filename : str, optional
            Name of the checkpoint file to load from the cache volume. If not provided,
            the model loaded at container startup will be used.
        patch_size : int, optional
            Size of the window to use for prediction, in pixels. Default is 400.
        patch_overlap : float, optional
            Overlap between windows, as a fraction of the patch size. Default is 0.05.
        iou_threshold : float, optional
            Minimum Intersection over Union (IoU) overlap threshold for among
            predictions between windows to be suppressed. Default is 0.15.

        Returns
        -------
        gpd.GeoDataFrame
            Predicted bounding boxes with tree crown annotations.
        """
        if checkpoint_filename is not None:
            checkpoint_filepath = path.join(MODELS_DIR, checkpoint_filename)
            print(f"Loading model from checkpoint: {checkpoint_filepath}")
            model = deepforest_main.deepforest.load_from_checkpoint(checkpoint_filepath)
        else:
            model = self.model
        print(
            f"Predicting on {img_filename} with patch size {patch_size}, "
            f"overlap {patch_overlap}, and IOU threshold {iou_threshold}."
        )
        return model.predict_tile(
            path.join(DATA_DIR, remote_img_dir, img_filename),
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            iou_threshold=iou_threshold,
        )


@app.local_entrypoint()
def upload_file(
    local_filepath: str,
    *,
    remote_dir: Optional[str] = None,
    remote_filepath: Optional[str] = None,
) -> None:
    """Upload a local file to the data volume.

    Parameters
    ----------
    local_filepath : str
        Path to the local file to upload.
    remote_dir : str, optional
        Directory in the data volume where the file should be uploaded. If not provided,
        the directory name of the latest component of `local_filepath` will be used.
        Ignored if `remote_filepath` is provided.
    remote_filepath : str, optional
        Full path in the data volume where the file should be uploaded. If not provided,
        it will be constructed from `remote_dir` and the base name of `local_filepath`.
    """
    if remote_filepath is None:
        if remote_dir is None:
            raise ValueError(
                "Either `remote_dir` or `remote_filepath` must be provided."
            )
        remote_filepath = path.join(remote_dir, path.basename(local_filepath))
    with data_volume.batch_upload() as batch:
        batch.put_file(local_filepath, remote_filepath)


@app.local_entrypoint()
def ensure_imgs(
    imgs_filepath: str, local_img_dir: str, *, remote_img_dir: Optional[str] = None
) -> None:
    """Ensure that images exist in the data volume, otherwise upload them.

    Parameters
    ----------
    imgs_filepath : str
        Path to a JSON file containing a list of image filenames to ensure.
    local_img_dir : str
        Path to the local directory containing the images.
    remote_img_dir : str, optional
        Directory in the data volume where the images should be uploaded. If not
        provided, it will be set to the base name of `local_img_dir`.
    """
    if remote_img_dir is None:
        # use the directory name of `local_img_dir`
        remote_img_dir = path.basename(local_img_dir)

    with open(imgs_filepath, encoding="utf-8") as src:
        img_filenames = json.load(src)
    if not img_filenames:
        raise ValueError("No image filenames provided in the arguments.")

    # data_volume.reload()  # fetch latest changes
    try:
        remote_img_filenames = [
            path.basename(remote_img_filepath.path)
            for remote_img_filepath in data_volume.listdir(remote_img_dir)
        ]
    except GRPCError:
        # the directory does not exist yet, we have to upload all images
        remote_img_filenames = []
    imgs_to_upload = []
    for img_filename in img_filenames:
        if img_filename not in remote_img_filenames:
            imgs_to_upload.append(img_filename)

    if imgs_to_upload:
        print(
            f"Uploading {len(imgs_to_upload)} images from {local_img_dir} to"
            f" {remote_img_dir}."
        )
        with data_volume.batch_upload() as batch:
            for img_filename in imgs_to_upload:
                src_filepath = path.join(local_img_dir, img_filename)
                batch.put_file(src_filepath, path.join(remote_img_dir, img_filename))


# @app.local_entrypoint()
# def retrain_crown_model(
#     train_df_filepath: str,
#     local_img_dir: str,
#     dst_filepath: str,
#     *,
#     checkpoint_filepath: Optional[str] = None,
#     remote_img_dir: Optional[str] = None,
#     remote_models_dir: Optional[str] = None,
#     test_df_filepath: Optional[str] = None,
#     train_config_filepath: Optional[str] = None,
#     create_trainer_kwargs_filepath: Optional[str] = None,
# ) -> None:
#     # TODO: use "upload_file" instead? the advantage of the current approach below is
#     # that it allows passing the data frame directly when using the app, e.g., in a
#     # Jupyter notebook
#     train_df = pd.read_csv(train_df_filepath)
#     if test_df_filepath is not None:
#         test_df = pd.read_csv(test_df_filepath)
#     else:
#         test_df = None

#     # TODO: use model_id approach instead like in most modal examples?
#     if checkpoint_filepath is not None:
#         if remote_models_dir is None:
#             remote_models_dir = MODELS_DIR
#         upload_file(checkpoint_filepath, remote_dir=remote_models_dir)

#     if remote_img_dir is None:
#         # use the directory name of `local_img_dir`
#         remote_img_dir = path.basename(local_img_dir)

#     # TODO: allow passing a logger via the CLI
#     # https://deepforest.readthedocs.io/en/v1.5.0/user_guide/11_training.html#loggers
#     if train_config_filepath is not None:
#         with open(train_config_filepath, encoding="utf-8") as src:
#             train_config = json.load(src)
#     else:
#         train_config = {}
#     if create_trainer_kwargs_filepath is not None:
#         with open(create_trainer_kwargs_filepath, encoding="utf-8") as src:
#             create_trainer_kwargs = json.load(src)
#     else:
#         create_trainer_kwargs = {}

#     # create app
#     deepforest_app = DeepForestApp()

#     # retrain model
#     model = deepforest_app.retrain_crown_model.remote(
#         train_df,
#         remote_img_dir,
#         checkpoint_filepath=checkpoint_filepath,
#         test_df=test_df,
#         train_config=train_config,
#         **create_trainer_kwargs,
#     )

#     # save retrained model
#     model.save_model(dst_filepath)
#     print(f"Saved retrained model to {dst_filepath}")


# @app.local_entrypoint()
# def predict(
#     args_filepath: str,
#     local_img_dir: str,
#     dst_filepath: str,
#     *,
#     remote_img_dir: Optional[str] = None,
# ) -> None:
#     if remote_img_dir is None:
#         # use the directory name of `local_img_dir`
#         remote_img_dir = path.basename(local_img_dir)

#     deepforest_app = DeepForestApp()

#     with open(args_filepath) as src:
#         args_dict = json.load(src)
#     patch_sizes = args_dict.get("patch_sizes", [800])
#     patch_overlap = args_dict.get("patch_overlap", 0.1)
#     iou_threshold = args_dict.get("iou_threshold", 0.15)
#     img_filenames = args_dict.get("img_filenames", [])
#     if not img_filenames:
#         raise ValueError("No image filenames provided in the arguments.")

#     # upload images to modal volume if needed
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         imgs_filepath = path.join(tmp_dir, "imgs.json")
#         with open(imgs_filepath, "w", encoding="utf-8") as dst:
#             json.dump(list(img_filenames), dst)

#         ensure_imgs(imgs_filepath, local_img_dir, remote_img_dir=remote_img_dir)

#     pred_gdf = pd.concat(
#         [
#             deepforest_app.predict.remote(
#                 img_filename,
#                 remote_img_dir,
#                 patch_size=patch_size,
#                 patch_overlap=patch_overlap,
#                 iou_threshold=iou_threshold,
#             ).assign(**{"patch_size": patch_size})
#             for img_filename in img_filenames
#             for patch_size in patch_sizes
#         ],
#         ignore_index=True,
#     )

#     # ACHTUNG: this requires geopandas in the local environment
#     pred_gdf.to_file(dst_filepath)
#     print(f"Saved {len(pred_gdf.index)} predictions to {dst_filepath}")

#     pred_gdf = pd.concat(
#         [
#             deepforest_app.predict.remote(
#                 img_filename,
#                 remote_img_dir,
#                 patch_size=patch_size,
#                 patch_overlap=patch_overlap,
#                 iou_threshold=iou_threshold,
#             ).assign(**{"patch_size": patch_size})
#             for img_filename in img_filenames
#             for patch_size in patch_sizes
#         ],
#         ignore_index=True,
#     )

#     # ACHTUNG: this requires geopandas in the local environment
#     pred_gdf.to_file(dst_filepath)
#     print(f"Saved {len(pred_gdf.index)} predictions to {dst_filepath}")
