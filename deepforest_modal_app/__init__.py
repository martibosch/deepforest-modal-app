"""Modal app for DeepForest model training and inference."""

import glob
import json
import os
from collections.abc import Mapping
from os import path
from typing import Optional

import modal
import rasterio as rio
from grpclib import GRPCError

from deepforest_modal_app import settings

# type annotations
# type hint for path-like objects
PathType = str | os.PathLike
# type hint for keyword arguments
KwargsType = Mapping | None


# volume to store models, i.e., (i) HuggingFace Hub cache, (ii) PyTorch hub cache and
# (iii) our deepforest checkpoints
models_volume = modal.Volume.from_name(
    settings.MODELS_VOLUME_NAME, create_if_missing=True
)

# volume to store data (images, annotations, etc.)
# REMOTE_IMAGES_DIR = path.join("/root/images")
data_volume = modal.Volume.from_name(settings.DATA_VOLUME_NAME, create_if_missing=True)
# with data_volume.batch_upload() as batch:
#     batch.put_directory(LOCAL_DATA_DIR, ".")


# create Modal image with required dependencies
app = modal.App(name=settings.APP_NAME)
image = (
    modal.Image.micromamba("3.11")
    .micromamba_install(
        "geopandas=1.0.1",
        "opencv=4.11.0",
        channels=["conda-forge"],
    )
    .pip_install(
        "albumentations==1.4.24", "deepforest==1.5.2", *settings.PIP_EXTRA_REQS
    )
    .env(
        {
            "HF_HUB_CACHE": path.join(settings.MODELS_DIR, "hf_hub_cache"),
            "TORCH_HOME": path.join(settings.MODELS_DIR, "torch"),
        }
    )
    # .add_local_dir(LOCAL_DATA_DIR, remote_path=DATA_DIR)
)

with image.imports():
    import tempfile
    import time

    import geopandas as gpd
    import pandas as pd
    import torch
    from deepforest import main as deepforest_main
    from deepforest import model as deepforest_model


@app.cls(
    image=image,
    gpu=settings.GPU_TYPE,
    volumes={
        settings.MODELS_DIR: models_volume,
        settings.DATA_DIR: data_volume,
    },
    timeout=settings.TIMEOUT,
)
class DeepForestApp:
    """DeepForest app.

    Parameters
    ----------
    model_name : str
        Name of the model to load from Hugging Face Hub. Ignored if
        `checkpoint_filepath` is provided.
    model_revision : str
        Revision of the model to load from Hugging Face Hub. Ignored if
        `checkpoint_filepath` is provided.
    checkpoint_filepath : str
        Path to the checkpoint file to load from the model volume (relative to the
        volume's root).
    config_filepath : str
        Path to the JSON file with model configuration. If not provided,
        the model will be configured to use all available GPUs and 4 workers.
    torch_seed : int
        Seed for PyTorch random number generator, used for reproducibility.
    """

    model_name: str = modal.parameter(default="weecology/deepforest-tree")
    model_revision: str = modal.parameter(default="main")
    checkpoint_filepath: str = modal.parameter(default="")
    config_filepath: str = modal.parameter(default="")
    torch_seed: int = modal.parameter(default=0)

    @modal.enter()
    def load_model(self) -> None:
        """Load the model from Hugging Face Hub or a checkpoint.

        This method is run at container startup only.
        """
        # set the random seed for reproducibility
        _ = torch.manual_seed(self.torch_seed)
        if self.checkpoint_filepath != "":
            # TODO: how does this affect build time?
            checkpoint_filepath = path.join(
                settings.MODELS_DIR, self.checkpoint_filepath
            )
            model = deepforest_main.deepforest.load_from_checkpoint(
                checkpoint_filepath,
            )
            print(f"Loaded model from checkpoint: {checkpoint_filepath}")
        else:
            # load the default release checkpoint
            model = deepforest_main.deepforest()
            model.load_model(model_name=self.model_name, revision=self.model_revision)
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
        checkpoint_filepath: PathType | None = None,
        test_df: pd.DataFrame | gpd.GeoDataFrame | None = None,
        train_config: KwargsType = None,
        validation_config: KwargsType = None,
        dst_filepath: str | None = None,
        retrain_if_exists: bool = False,
        **create_trainer_kwargs: KwargsType,
    ) -> None:  # deepforest_main.deepforest:
        """Retrain the DeepForest model with the provided training data.

        Parameters
        ----------
        train_df : pd.DataFrame or gpd.GeoDataFrame
            Training data, as pandas or geopandas data frame with bounding box
            annotations.
        remote_img_dir : path-like
            Path to the remote directory with images, relative to the data volume's
            root.
        checkpoint_filepath : path-like
            Path to the checkpoint file to load from the model volume (relative to the
            volume's root). If not provided, the model loaded at container startup will
            be used.
        test_df : pd.DataFrame or gpd.GeoDataFrame, optional
            Test data to use for validation during training. If not provided, training
            will be performed without validation.
        train_config, validation_config : dict-like, optional
            Configuration for the training and validation, passed to the model's
            `config` attribute under the keys "train" and "validation" respectively.
        dst_filepath : path-like, optional
            Path to the file to save the retrained model to (relative to the model
            volume's root). If not provided, a file name will be generated based on the
            current timestamp.
        retrain_if_exists : bool, default False
            If True, the model will be retrained even if a checkpoint with the file name
            provided as `dst_filepath` already exists and subsequently overwritten.
            If False, no retraining will be done if the checkpoint already exists.
        **create_trainer_kwargs : dict-like
            Additional keyword arguments to pass to the model's `create_trainer` method.
            If none provided, the value from `settings.DEFAULT_CREATE_TRAINER_KWARGS`
            will be used.
        """
        if not retrain_if_exists and dst_filepath is not None:
            # check if the checkpoint file already exists
            _dst_filepath = path.join(settings.MODELS_DIR, dst_filepath)
            if path.exists(_dst_filepath):
                print(
                    f"Checkpoint {_dst_filepath} already exists, skipping"
                    " retraining. Use `retrain_if_exists=True` to overwrite."
                )
                return

        def save_annot_df(annot_df, dst_filepath):
            """Save the annotated data frame."""
            # we are just using a function to DRY any eventual required preprocessing
            annot_df.to_csv(dst_filepath)
            return dst_filepath

        if checkpoint_filepath is not None:
            _checkpoint_filepath = path.join(settings.MODELS_DIR, checkpoint_filepath)
            model = deepforest_main.deepforest.load_from_checkpoint(
                _checkpoint_filepath,
            )
            print(f"Loaded model from checkpoint: {_checkpoint_filepath}")
        else:
            model = self.model

        # pass configuration to the model
        if train_config is None:
            # use all gpus by default
            train_config = {"gpus": -1}
        for key, value in train_config.items():
            model.config["train"][key] = value

        if validation_config is None:
            validation_config = {}
        for key, value in validation_config.items():
            model.config["validation"][key] = value

        # prepend volume path to the remote image directory
        remote_img_dir = path.join(settings.DATA_DIR, remote_img_dir)

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
            if not create_trainer_kwargs:
                create_trainer_kwargs = settings.DEFAULT_CREATE_TRAINER_KWARGS

            model.create_trainer(**create_trainer_kwargs)
            start_time = time.time()
            model.trainer.fit(model)
        print(f"Model retrained in {(time.time() - start_time):.2f} seconds.")

        # TODO: replace model attribute with the trained model?
        # self.model = model
        if dst_filepath is None:
            dst_filepath = f"deepforest-retrained-{time.strftime('%Y%m%d_%H%M%S')}.pl"
        _dst_filepath = path.join(settings.MODELS_DIR, dst_filepath)
        # model.save_model(dst_filepath)
        model.trainer.save_checkpoint(_dst_filepath)
        print(f"Saved checkpoint to {_dst_filepath}")
        # return model

    @modal.method()
    def predict(
        self,
        img_filename: str,
        remote_img_dir: str,
        *,
        checkpoint_filepath: str | None = None,
        crop_model_filepath: str | None = None,
        crop_model_num_classes: int | None = None,
        iou_threshold: float = 0.15,
        patch_size: int | None = None,
        patch_overlap: float | None = None,
    ) -> gpd.GeoDataFrame:
        """Predict tree crown bounding boxes using the a DeepForest-like model.

        Parameters
        ----------
        img_filename : str
            File name of the image to predict on.
        remote_img_dir : str
            Path to the remote directory with images, relative to the data volume's
            root.
        checkpoint_filepath : path-like
            Path to the checkpoint file to load from the model volume (relative to the
            volume's root). If not provided, the model loaded at container startup will
            be used.
        crop_model_filepath : path-like, optional
            Path to the checkpoint of the model to classify the cropped images, i.e.,
            species detection for the tree bounding boxes (relative to the model
            volume's root). If not provided, no classification will be performed.
        crop_model_num_classes : int, optional
            Number of classes for the crop model. Required if `crop_model_filepath` is
            provided, ignored otherwise.
        iou_threshold : float, optional
            Minimum Intersection over Union (IoU) overlap threshold for among
            predictions between windows to be suppressed. Default is 0.15.
        patch_size : int, optional
            Size of the window to use for prediction, in pixels. If not provided, the
            behaviour depends on how the image size compares to
            `settings.MAX_IMAGE_SIZE`, i.e., if smaller, the whole image will be
            used for prediction; whereas if larger, the image will be split into
            (square) tiles of size `settings.MAX_IMAGE_SIZE` for prediction.
        patch_overlap : float, optional
            Overlap between windows, as a fraction of the patch size. Ignored if the
            image is not split into tiles (depending on the image size and the provided
            `patch_size`, see the description of the `patch_size` argument above).

        Returns
        -------
        gpd.GeoDataFrame
            Predicted bounding boxes with tree crown annotations.
        """
        if checkpoint_filepath is not None:
            _checkpoint_filepath = path.join(settings.MODELS_DIR, checkpoint_filepath)
            model = deepforest_main.deepforest.load_from_checkpoint(
                _checkpoint_filepath
            )
            print(f"Loaded model from checkpoint: {_checkpoint_filepath}")
        else:
            model = self.model
        if crop_model_filepath is not None:
            _crop_model_filepath = path.join(settings.MODELS_DIR, crop_model_filepath)
            crop_model = deepforest_model.CropModel.load_from_checkpoint(
                _crop_model_filepath, num_classes=crop_model_num_classes
            )
            print(f"Loaded crop model from checkpoint: {_crop_model_filepath}")
        else:
            crop_model = None

        img_filepath = path.join(settings.DATA_DIR, remote_img_dir, img_filename)
        log_msg = f"Predicting on {img_filename} with"
        if patch_size is None:
            # if no `patch_size` is provided, check if the image is large
            with rio.open(img_filepath) as src:
                max_size = max(src.width, src.height)
                if max_size > settings.MAX_IMG_SIZE:
                    # the image is large, use default patch size
                    patch_size = settings.DEFAULT_PATCH_SIZE
                    if patch_overlap is None:
                        # use default overlap if not provided
                        patch_overlap = settings.DEFAULT_PATCH_OVERLAP
                    log_msg += f" patch size {patch_size}, overlap {patch_overlap} and "
                else:
                    # the image is small, use the whole image
                    patch_size = max_size
                    patch_overlap = 0
        else:
            # a patch size is provided, use it
            if patch_overlap is None:
                # use default overlap if not provided
                patch_overlap = settings.DEFAULT_PATCH_OVERLAP
            log_msg += f" patch size {patch_size}, overlap {patch_overlap} and "

        log_msg = f" IOU threshold {iou_threshold}."
        print(log_msg)
        return model.predict_tile(
            img_filepath,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            iou_threshold=iou_threshold,
            crop_model=crop_model,
        )

    @modal.method()
    def train_crop_model(
        self,
        train_df: pd.DataFrame | gpd.GeoDataFrame,
        remote_img_dir: PathType,
        *,
        test_df: pd.DataFrame | gpd.GeoDataFrame | None = None,
        dst_filepath: str | None = None,
        retrain_if_exists: bool = False,
        crop_model_kwargs: KwargsType = None,
        create_trainer_kwargs: KwargsType = None,
    ) -> None:
        """Train a crop model.

        Parameters
        ----------
        train_df : pd.DataFrame or gpd.GeoDataFrame
            Training data, as pandas or geopandas data frame with multi-class (under the
            "label" column) bounding box annotations.
        remote_img_dir : PathType
            Path to the remote directory with images, relative to the data volume root.
        test_df : pd.DataFrame or gpd.GeoDataFrame, optional
            Test data to use for validation during training. If not provided, training
            will be performed without validation.
        dst_filepath : path-like, optional
            Path to the file to save the retrained model to (relative to the model
            volume's root). If not provided, a file name will be generated based on the
            current timestamp.
        retrain_if_exists : bool, default False
            If True, the model will be retrained even if a checkpoint with the file name
            provided as `dst_filepath` already exists and subsequently overwritten.
            If False, no retraining will be done if the checkpoint already exists.
        crop_model_kwargs, create_trainer_kwargs : dict-like
            Keyword arguments to pass to the model's initialization and `create_trainer`
            methods respectively. If none provided, the values from
            `settings.DEFAULT_CROP_MODEL_KWARGS` and
            `settings.DEFAULT_CREATE_TRAINER_KWARGS` will be used.
        """
        if not retrain_if_exists and dst_filepath is not None:
            # check if the checkpoint file already exists
            _dst_filepath = path.join(settings.MODELS_DIR, dst_filepath)
            if path.exists(_dst_filepath):
                print(
                    f"Checkpoint {_dst_filepath} already exists, skipping"
                    " retraining. Use `retrain_if_exists=True` to overwrite."
                )
                return

        # TODO: do NOT uncumment the code below until the following commit is released
        # github.com/weecology/DeepForest/b99d068be36da4d995931d7d4905bce251530a0f
        # provide a label dict containing all the keys from both the train and test
        # data frames
        # labels = set(train_df["label"].unique())
        # if test_df is not None:
        #     labels.update(test_df["label"].unique())
        # label_dict = {label: i for i, label in enumerate(sorted(labels))}
        # crop_model = deepforest_model.CropModel(
        #     num_classes=len(label_dict), label_dict=label_dict
        # )
        # ACHTUNG: in the meantime (see TODO above), use all the data for training,
        # namely train and test data frames to ensure that we have all the labels in the
        # label dict
        # TODO: how to handle the case where not all labels are on the training set?
        # e.g., raise a ValueError?
        train_df = pd.concat([train_df, test_df]) if test_df is not None else train_df
        if crop_model_kwargs is None:
            crop_model_kwargs = settings.DEFAULT_CROP_MODEL_KWARGS
        crop_model = deepforest_model.CropModel(
            num_classes=train_df["label"].nunique(), **crop_model_kwargs
        )
        # create trainer
        if create_trainer_kwargs is None:
            create_trainer_kwargs = settings.DEFAULT_CREATE_TRAINER_KWARGS
        crop_model.create_trainer(**create_trainer_kwargs)

        def write_crops(annot_df, dst_dir):
            crop_model.write_crops(
                path.join(settings.DATA_DIR, remote_img_dir),
                annot_df["image_path"].values,
                annot_df[["xmin", "ymin", "xmax", "ymax"]].values,
                annot_df["label"].values.astype(str),
                dst_dir,
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            train_dir = path.join(tmp_dir, "train")
            test_dir = path.join(tmp_dir, "test")
            for _dir in [train_dir, test_dir]:
                os.makedirs(_dir, exist_ok=True)
            write_crops(train_df, train_dir)
            if test_df is not None:
                write_crops(test_df, test_dir)
            crop_model.load_from_disk(train_dir=train_dir, val_dir=test_dir)
            crop_model.trainer.fit(crop_model)

        # self.model = model
        if dst_filepath is None:
            dst_filepath = f"crop-{time.strftime('%Y%m%d_%H%M%S')}.pl"
        _dst_filepath = path.join(settings.MODELS_DIR, dst_filepath)
        # model.save_model(dst_filepath)
        crop_model.trainer.save_checkpoint(_dst_filepath)
        print(f"Saved checkpoint to {dst_filepath}")


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
    local_img_dir: str,
    *,
    imgs_filepath: Optional[str] = None,
    remote_img_dir: Optional[str] = None,
) -> None:
    """Ensure that images exist in the data volume, otherwise upload them.

    Parameters
    ----------
    local_img_dir : str
        Path to the local directory containing the images. If `imgs_filepath` is not
        provided, all files in this directory will be uploaded (if not already in the
        data volume). Otherwise only the images listed in `imgs_filepath` will be
        uploaded.
    imgs_filepath : str, optional
        Path to a JSON file containing a list of image filenames to ensure. If not
        provided, all files in `local_img_dir` will be uploaded (if not already in
        the data volume).
    remote_img_dir : str, optional
        Directory in the data volume where the images should be uploaded. If not
        provided, it will be set to the base name of `local_img_dir`.
    """
    if remote_img_dir is None:
        # use the directory name of `local_img_dir`
        remote_img_dir = path.basename(local_img_dir)

    if imgs_filepath is None:
        img_filenames = [
            path.basename(img_filepath)
            for img_filepath in glob.glob(path.join(local_img_dir, "*"))
        ]
    else:
        with open(imgs_filepath, encoding="utf-8") as src:
            img_filenames = json.load(src)
    if len(img_filenames) == 0:
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
        print("Upload completed.")
