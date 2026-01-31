"""Modal app for DeepForest model training and inference."""

import copy
import glob
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from os import path
from typing import Optional

import modal
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
huggingface_secret = modal.Secret.from_name(
    "huggingface-secret", required_keys=["HF_TOKEN"]
)

# create Modal image with required dependencies
app = modal.App(name=settings.APP_NAME)
image = (
    modal.Image.micromamba("3.11")
    .micromamba_install(
        "geopandas=1.0.1",
        "opencv=4.11.0",
        channels=["conda-forge"],
    )
    .uv_pip_install(
        # "deepforest>=2.0.0",
        "https://github.com/martibosch/DeepForest/archive/"
        "58a9b39e4ba55c89b90ff7146e64892cb769c35f"
        # "https://github.com/weecology/DeepForest/archive/"
        # "e44b2681bfe7fc20b8702304d364024bcde01c7c",
        # "ad614add382c1cd5213a016d3d671a9c324deedc",
        # "7fff5fed131aa048a79dfdf8ecc42abf973468a5"
        # "927e614b8cca14a599488e03b5d284ec656250ed"
        ".zip",
        *settings.PIP_EXTRA_REQS,
    )
    .env(
        {
            "HF_HUB_CACHE": path.join(settings.MODELS_DIR, "hf_hub_cache"),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # turn on faster downloads from HF
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
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
    from deepforest.datasets.training import create_aligned_image_folders
    from shapely import geometry
    from torchvision import transforms


def _load_model(
    *,
    checkpoint_filepath: PathType | None = None,
    config_filepath: PathType | None = None,
    config_args: Mapping | None = None,
    model_name: str | None = None,
    model_revision: str | None = None,
):
    """Load a model.

    Parameters
    ----------
    checkpoint_filepath : path-like, optional
        Path to the checkpoint file to load from the model volume (relative to the
        volume's root). If None, the model from `model_name` and `model_revision` are
        loaded from Hugging Face Hub.
    config_filepath : path-like, optional
        Path to the JSON file with model configuration. If None, the defaults from
        DeepForest is used.
    config_args : mapping, optional
        Mapping of configuration overrides to the given file.
    model_name : str, optional
        Name of the model to load from Hugging Face Hub. Ignored if
        `checkpoint_filepath` is provided. If None, the default from
        `settings.DEFAULT_MODEL_NAME` is used.
    model_revision : str, optional
        Revision of the model to load from Hugging Face Hub. Ignored if
        `checkpoint_filepath` is provided. If None, the default from
        `settings.DEFAULT_MODEL_REVISION` is used.

    Returns
    -------
    deepforest_main.deepforest
        The loaded DeepForest model.
    """
    # process config path first
    if config_filepath is None:
        _config_filepath = None
    else:
        _config_filepath = path.join(settings.DATA_DIR, config_filepath)

    if checkpoint_filepath is not None:
        checkpoint_filepath = path.join(settings.MODELS_DIR, checkpoint_filepath)
        load_kwargs = {}
        if _config_filepath is not None:
            load_kwargs["config"] = _config_filepath
        if config_args is None:
            _config_args = {}
        else:
            _config_args = copy.deepcopy(config_args)
        # prevent metric initialization with the tmp no longer existing validation csv
        # file
        val_config = {}
        val_config["csv_file"] = None
        val_config["root_dir"] = None
        _config_args["validation"] = val_config
        load_kwargs["config_args"] = _config_args
        model = deepforest_main.deepforest.load_from_checkpoint(
            checkpoint_filepath, **load_kwargs
        )
        print(f"Loaded model from checkpoint: {checkpoint_filepath}")
    else:
        # init model
        model = deepforest_main.deepforest(
            config=_config_filepath, config_args=config_args
        )
        # load the default release checkpoint
        model.load_model(model_name=model_name, revision=model_revision)

    # return the model
    return model


@app.cls(
    image=image,
    cpu=settings.CPU,
    gpu=settings.GPU_TYPE,
    volumes={
        settings.MODELS_DIR: models_volume,
        settings.DATA_DIR: data_volume,
    },
    secrets=[huggingface_secret],
    timeout=settings.TIMEOUT,
)
class DeepForestApp:
    """DeepForest app.

    Parameters
    ----------
    torch_seed : int
        Seed for PyTorch random number generator, used for reproducibility.
    """

    torch_seed: int = modal.parameter(default=0)

    @modal.enter()
    def set_seed(self) -> None:
        """Set a random seed for reproducibility.

        This method is run at container startup only.
        """
        # set the random seed for reproducibility
        _ = torch.manual_seed(self.torch_seed)

    @modal.method()
    def retrain_crown_model(
        self,
        train_df: pd.DataFrame | gpd.GeoDataFrame,
        remote_img_dir: PathType,
        *,
        test_df: pd.DataFrame | gpd.GeoDataFrame | None = None,
        dst_filepath: str | None = None,
        retrain_if_exists: bool = False,
        checkpoint_filepath: PathType | None = None,
        config_filepath: PathType | None = None,
        config_args: Mapping | None = None,
        model_name: str | None = None,
        model_revision: str | None = None,
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
        checkpoint_filepath : path-like, optional
            Path to the checkpoint file to load from the model volume (relative to the
            volume's root). If None, the model from `model_name` and `model_revision`
            are loaded from Hugging Face Hub.
        config_filepath : path-like, optional
            Path to the JSON file with model configuration. If None, the defaults from
            DeepForest is used.
        config_args : mapping, optional
            Mapping of configuration overrides to the given file.
        model_name : str, optional
            Name of the model to load from Hugging Face Hub. Ignored if
            `checkpoint_filepath` is provided. If None, the default from
            `settings.DEFAULT_MODEL_NAME` is used.
        model_revision : str, optional
            Revision of the model to load from Hugging Face Hub. Ignored if
            `checkpoint_filepath` is provided. If None, the default from
            `settings.DEFAULT_MODEL_REVISION` is used.
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

        model = _load_model(
            checkpoint_filepath=checkpoint_filepath,
            config_filepath=config_filepath,
            config_args=config_args,
            model_name=model_name,
            model_revision=model_revision,
        )

        # pass configuration to the model
        # if train_config is None:
        #     train_config = {}
        # for key, value in train_config.items():
        #     model.config["train"][key] = value

        # if validation_config is None:
        #     validation_config = {}
        # for key, value in validation_config.items():
        #     model.config["validation"][key] = value

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
    def checkpoint_to_hf_hub(
        self,
        checkpoint_filepath: PathType,
        repo_id: str,
        *,
        model_card_kwargs: KwargsType = None,
        **push_to_hub_kwargs,
    ) -> None:
        """
        Push a DeepForest model checkpoint to the Hugging Face Hub.

        Parameters
        ----------
        repo_id : str
            The repository in HuggingFace Hub to which the model should be pushed.
        checkpoint_filepath : path-like, optional
            Path to the checkpoint file to load from the model volume (relative to the
            volume's root). If not provided, the model loaded at container startup will
            be used.
        model_card_kwargs : dict, optional
            Additional keyword arguments passed as the `model_card_kwargs` keyword
            argument in the model's `save_pretrained` method.
        **push_to_hub_kwargs
            Additional keyword arguments passed to the model's `save_pretrained` method.
        """
        _checkpoint_filepath = path.join(settings.MODELS_DIR, checkpoint_filepath)
        model = deepforest_main.deepforest.load_from_checkpoint(
            _checkpoint_filepath,
        )
        print(f"Loaded model from checkpoint: {_checkpoint_filepath}")
        _push_to_hub_kwargs = {}
        token = _push_to_hub_kwargs.get("token", os.environ["HF_TOKEN"])
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(
                save_directory=tmp_dir,
                repo_id=repo_id,
                push_to_hub=True,
                model_card_kwargs=model_card_kwargs,
                token=token,
                **_push_to_hub_kwargs,
            )
        print(f"Pushed model from {_checkpoint_filepath} to {repo_id}")

    @modal.method()
    def predict(
        self,
        remote_img_dir: str,
        *,
        img_filenames: str | Sequence[str] | None = None,
        img_ext: str | None = None,
        checkpoint_filepath: str | None = None,
        config_filepath: PathType | None = None,
        config_args: Mapping | None = None,
        model_name: str | None = None,
        model_revision: str | None = None,
        crop_model_filepath: str | None = None,
        iou_threshold: float = 0.15,
        patch_size: int | None = None,
        patch_overlap: float | None = None,
        dataloader_strategy: str = "batch",
    ) -> gpd.GeoDataFrame:
        """Predict tree crown bounding boxes using the a DeepForest-like model.

        Parameters
        ----------
        remote_img_dir : str
            Path to the remote directory with images, relative to the data volume's
            root.
        img_filenames : str or list-like of str, optional
            Images to predict on, relative to `remote_img_dir`. If not provided,
            all images in `remote_img_dir` with the specified `img_ext` will be used.
        img_ext : str, optional
            Image file extension to use when searching for images in `remote_img_dir`.
            Ignored if `img_filenames` is provided. If not provided, the value from
            `settings.IMG_EXT` will be used.
        checkpoint_filepath : path-like, optional
            Path to the checkpoint file to load from the model volume (relative to the
            volume's root). If None, the model from `model_name` and `model_revision`
            are loaded from Hugging Face Hub.
        config_filepath : path-like, optional
            Path to the JSON file with model configuration. If None, the defaults from
            DeepForest is used.
        config_args : mapping, optional
            Mapping of configuration overrides to the given file.
        model_name : str, optional
            Name of the model to load from Hugging Face Hub. Ignored if
            `checkpoint_filepath` is provided. If None, the default from
            `settings.DEFAULT_MODEL_NAME` is used.
        model_revision : str, optional
            Revision of the model to load from Hugging Face Hub. Ignored if
            `checkpoint_filepath` is provided. If None, the default from
            `settings.DEFAULT_MODEL_REVISION` is used.
        crop_model_filepath : path-like, optional
            Path to the checkpoint of the model to classify the cropped images, i.e.,
            species detection for the tree bounding boxes (relative to the model
            volume's root). If not provided, no classification will be performed.
        iou_threshold : float, optional
            Minimum Intersection over Union (IoU) overlap threshold for among
            predictions between windows to be suppressed. Default is 0.15.
        patch_size : int, optional
            Size of the square patches to split the image into for prediction. If not
            provided, the default value from `settings.PATCH_SIZE` will be used. If
            the image is smaller than `patch_size`, it will be processed as a whole.
        patch_overlap : float, optional
            Overlap between windows, as a fraction of the patch size. Ignored if the
            image is not split into tiles (depending on the image size and the provided
            `patch_size`, see the description of the `patch_size` argument above).
        dataloader_strategy : {"single", "batch", "window"}, optional
            Strategy to use for creating the dataloader for prediction. Options are:
            - "single": loads the entire image into CPU memory and passes individual
              windows to GPU.
            - "batch": loads the entire image into GPU memory and creates views of the
              image as batches. Requires the entire tile to fit into GPU memory. CPU
              parallelization is possible for loading images.
            - "window": loads only the desired window of the image from the raster
              dataset. Most memory efficient option, but cannot parallelize across
              windows due to Python’s Global Interpreter Lock, workers must be set to 0.

        Returns
        -------
        gpd.GeoDataFrame
            Predicted bounding boxes with tree crown annotations.
        """
        model = _load_model(
            checkpoint_filepath=checkpoint_filepath,
            config_filepath=config_filepath,
            config_args=config_args,
            model_name=model_name,
            model_revision=model_revision,
        )
        if crop_model_filepath is not None:
            _crop_model_filepath = path.join(settings.MODELS_DIR, crop_model_filepath)
            crop_model = deepforest_model.CropModel.load_from_checkpoint(
                _crop_model_filepath,
            )
            print(f"Loaded crop model from checkpoint: {_crop_model_filepath}")
        else:
            crop_model = None

        if isinstance(img_filenames, str):
            img_filenames = [img_filenames]
        if len(img_filenames) == 0:
            if img_ext is None:
                img_ext = settings.DEFAULT_IMG_EXT
            img_filenames = glob.glob(
                path.join(
                    settings.DATA_DIR,
                    remote_img_dir,
                    f"*.{img_ext}",
                )
            )

        img_filepaths = [
            path.join(settings.DATA_DIR, remote_img_dir, img_filename)
            for img_filename in img_filenames
        ]
        log_msg = (
            f"Predicting on {len(img_filepaths)} images at {remote_img_dir} with"
            f" patch size {patch_size}, overlap {patch_overlap} and"
            f" IOU threshold {iou_threshold}."
        )
        print(log_msg)
        # for some reason, the line below returns nan geometries, likely related to
        # https://github.com/weecology/DeepForest/issues/1149#event-19952397024
        # in the meantime, ensure the geometry column manually
        # return model.predict_tile(
        #     img_filepaths,
        #     patch_size=patch_size,
        #     patch_overlap=patch_overlap,
        #     iou_threshold=iou_threshold,
        #     crop_model=crop_model,
        #     dataloader_strategy=dataloader_strategy,
        # )
        pred_gdf = model.predict_tile(
            img_filepaths,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            iou_threshold=iou_threshold,
            crop_model=crop_model,
            dataloader_strategy=dataloader_strategy,
        )
        return gpd.GeoDataFrame(
            pred_gdf,
            geometry=pred_gdf.apply(
                lambda x: geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis="columns"
            ),
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
        crop_augmentations: Sequence[Mapping[str, object]] | None = None,
        crops_dir: str | None = None,
        write_crops_max_workers: int | None = None,
        write_crops_chunk_size: int | None = 512,
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
        crop_augmentations : list of dict, optional
            Torchvision augmentation configs for crop classification training. Each
            entry should be a mapping with a "name" key (e.g., "RandomRotation") and
            optional "kwargs". If provided, custom transforms are used instead of the
            default HorizontalFlip-only augmentation.
        crops_dir : str, optional
            Directory (relative to the data volume root) where crop images will be
            written. If not provided, a temporary directory will be used.
        write_crops_max_workers : int, optional
            Maximum number of worker threads used to write crops per label. If not
            provided, a conservative default based on CPU count is used.
        write_crops_chunk_size : int, optional
            Chunk size for writing crops per label. Smaller values reduce long blocking
            calls and can help avoid heartbeat timeouts.
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

        # TODO: how to handle the case where not all labels are on the training set?
        # e.g., raise a ValueError?
        # train_df = pd.concat([train_df, test_df]) if test_df is not None else train_df
        if crop_model_kwargs is None:
            crop_model_kwargs = settings.DEFAULT_CROP_MODEL_KWARGS
        crop_model = deepforest_model.CropModel(**crop_model_kwargs)
        # create trainer
        if create_trainer_kwargs is None:
            create_trainer_kwargs = settings.DEFAULT_CROP_MODEL_CREATE_TRAINER_KWARGS
        crop_model.create_trainer(**create_trainer_kwargs)

        def _build_crop_transforms(augmentations):
            resize_dims = crop_model.config["cropmodel"].get("resize", [224, 224])
            supported = {
                "RandomResizedCrop": transforms.RandomResizedCrop,
                "RandomHorizontalFlip": transforms.RandomHorizontalFlip,
                "RandomVerticalFlip": transforms.RandomVerticalFlip,
                "RandomRotation": transforms.RandomRotation,
                "ColorJitter": transforms.ColorJitter,
                "RandomAffine": transforms.RandomAffine,
                "GaussianBlur": transforms.GaussianBlur,
            }
            aug_transforms = []
            aug_names = []
            for aug in augmentations or []:
                if not isinstance(aug, Mapping):
                    raise ValueError(
                        "Each crop augmentation must be a mapping with a 'name' key."
                    )
                name = aug.get("name")
                if name not in supported:
                    raise ValueError(
                        f"Unsupported crop augmentation '{name}'. Supported: "
                        f"{', '.join(sorted(supported))}."
                    )
                kwargs = dict(aug.get("kwargs") or {})
                if name == "RandomResizedCrop" and "size" not in kwargs:
                    kwargs["size"] = resize_dims
                aug_transforms.append(supported[name](**kwargs))
                aug_names.append(name)
            if "RandomResizedCrop" not in aug_names:
                aug_transforms.insert(0, transforms.Resize(resize_dims))
            aug_transforms.extend([transforms.ToTensor(), crop_model.normalize()])
            return transforms.Compose(aug_transforms)

        def _iter_chunks(df, chunk_size):
            for start in range(0, len(df), chunk_size):
                yield df.iloc[start : start + chunk_size]

        def _write_crops_for_label(label_df, dst_dir):
            label = label_df["_label_str"].iloc[0]
            total = len(label_df)
            if not write_crops_chunk_size or total <= write_crops_chunk_size:
                crop_model.write_crops(
                    path.join(settings.DATA_DIR, remote_img_dir),
                    label_df["image_path"].values,
                    label_df[["xmin", "ymin", "xmax", "ymax"]].values,
                    label_df["_label_str"].values,
                    dst_dir,
                )
                print(
                    f"Wrote {total} crops for label '{label}' to {dst_dir}",
                    flush=True,
                )
                return
            written = 0
            for chunk_df in _iter_chunks(label_df, write_crops_chunk_size):
                crop_model.write_crops(
                    path.join(settings.DATA_DIR, remote_img_dir),
                    chunk_df["image_path"].values,
                    chunk_df[["xmin", "ymin", "xmax", "ymax"]].values,
                    chunk_df["_label_str"].values,
                    dst_dir,
                )
                written += len(chunk_df)
                print(
                    f"Wrote {written}/{total} crops for label '{label}' to {dst_dir}",
                    flush=True,
                )

        def write_crops(annot_df, dst_dir):
            labeled_df = annot_df.assign(_label_str=annot_df["label"].astype(str))
            label_groups = labeled_df.groupby("_label_str", sort=False)
            if write_crops_max_workers is None:
                cpu_workers = max(1, (os.cpu_count() or 1) // 2)
                max_workers = min(label_groups.ngroups, cpu_workers)
            else:
                max_workers = max(1, write_crops_max_workers)
                max_workers = min(max_workers, label_groups.ngroups)
            print(
                f"Writing crop images using {max_workers} workers "
                f"(chunk_size={write_crops_chunk_size})"
            )
            if max_workers <= 1:
                for _, label_df in label_groups:
                    _write_crops_for_label(label_df, dst_dir)
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(_write_crops_for_label, label_df, dst_dir)
                        for _, label_df in label_groups
                    ]
                    for future in futures:
                        future.result()
            print(f"Written {len(annot_df)} crop images at {dst_dir}")

        if crops_dir is None:
            tmp_ctx = tempfile.TemporaryDirectory()
            base_dir = tmp_ctx.name
        else:
            _crops_dir = path.normpath(crops_dir)
            if path.isabs(_crops_dir) or _crops_dir.startswith(".."):
                raise ValueError("`crops_dir` must be relative to the data volume.")
            tmp_ctx = nullcontext()
            base_dir = path.join(settings.DATA_DIR, _crops_dir)

        with tmp_ctx:
            train_dir = path.join(base_dir, "train")
            test_dir = path.join(base_dir, "test")
            for _dir in [train_dir, test_dir]:
                os.makedirs(_dir, exist_ok=True)
            write_crops(train_df, train_dir)
            if test_df is not None:
                write_crops(test_df, test_dir)
            if crop_augmentations:
                train_tf = _build_crop_transforms(crop_augmentations)
                val_tf = _build_crop_transforms(None)
                crop_model.train_ds, crop_model.val_ds = create_aligned_image_folders(
                    train_dir,
                    test_dir,
                    transform_train=train_tf,
                    transform_val=val_tf,
                )
                crop_model.label_dict = crop_model.train_ds.class_to_idx
                crop_model.numeric_to_label_dict = {
                    v: k for k, v in crop_model.label_dict.items()
                }
                crop_model.num_classes = len(crop_model.label_dict)
                if crop_model.model is None:
                    crop_model.create_model(crop_model.num_classes)
            else:
                crop_model.load_from_disk(train_dir=train_dir, val_dir=test_dir)
            start_time = time.time()
            print("Started crop model training with config: ", crop_model.config)
            crop_model.trainer.fit(crop_model)
        print(f"Crop model retrained in {(time.time() - start_time):.2f} seconds.")

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


# @app.function(volumes={settings.MODELS_DIR: models_volume})
# def _checkpoint_to_hf_hub(
#     checkpoint_filepath: PathType,
#     repo_id: str,
#     *,
#     model_cards_kwargs: KwargsType = None,
#     **push_to_hub_kwargs,
# ) -> str:
#     """
#     Push a DeepForest model checkpoint to the Hugging Face Hub.

#     Parameters
#     ----------
#     repo_id : str
#         The repository in HuggingFace Hub to which the model should be pushed.
#     model_cards_kwargs : dict, optional
#         Additional keyword arguments passed as the `model_cards_kwargs` keyword
#         argument in the model's `save_pretrained` method.
#     **push_to_hub_kwargs
#         Additional keyword arguments passed to the model's `save_pretrained` method.

#     Returns
#     -------
#     commit_url: str
#         The commit URL.
#     """
#     _checkpoint_filepath = path.join(settings.MODELS_DIR, checkpoint_filepath)
#     model = deepforest_main.deepforest.load_from_checkpoint(
#         _checkpoint_filepath,
#     )
#     print(f"Loaded model from checkpoint: {_checkpoint_filepath}")
#     _push_to_hub_kwargs = {}
#     token = _push_to_hub_kwargs.get("token", os.environ["HF_TOKEN"])
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         commit_url = model.save_pretrained(
#             save_directory=tmp_dir,
#             push_to_hub=True,
#             repo_id=repo_id,
#             model_cards_kwargs=model_cards_kwargs,
#             token=token,
#             **_push_to_hub_kwargs,
#         )
#     print(f"Pushed model from {_checkpoint_filepath} to {repo_id}")
#     return commit_url


# @app.local_entrypoint()
# def checkpoint_to_hf_hub(checkpoint_filepath: str, repo_id: str) -> str:
#     """
#     Push a DeepForest model checkpoint to the Hugging Face Hub.

#     Parameters
#     ----------
#     checkpoint_filepath : path-like
#         Path to the checkpoint file to load from the model volume (relative to the
#         volume's root).
#     repo_id : str
#         The repository in HuggingFace Hub to which the model should be pushed.

#     Returns
#     -------
#     commit_url: str
#         The commit URL.
#     """
#     return _checkpoint_to_hf_hub.remote(
#         checkpoint_filepath,
#         repo_id,
#     )
