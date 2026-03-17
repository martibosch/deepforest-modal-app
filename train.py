"""Train/fine-tune a DeepForest tree crown detection model locally with wandb logging."""

import tempfile
import time
from dataclasses import dataclass, field
from os import path

import geopandas as gpd
import numpy as np
import pandas as pd
import simple_parsing as sp
import torch
import wandb
from deepforest import main as deepforest_main
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import WandbLogger
from rich.console import Console
from rich.panel import Panel

from deepforest_modal_app import eval_utils

console = Console()


@dataclass
class Args:
    """DeepForest crown detection training configuration."""

    # data
    base_dir: str = "/mnt/new-pvc/datasets/treeai/12_RGB_ObjDet_640_fL"  # path to TreeAI dataset
    target_species: list[str] = field(
        default_factory=lambda: ["picea abies", "pinus sylvestris", "fagus sylvatica"]
    )

    # model
    model_name: str = "weecology/deepforest-tree"  # HF model to fine-tune
    model_revision: str = "main"  # HF model revision
    checkpoint: str | None = None  # resume from local checkpoint

    # training
    max_epochs: int = 50
    batch_size: int = 4
    lr: float = 1e-4
    workers: int = 4
    precision: str = "32"
    accumulate_grad_batches: int = 4
    val_accuracy_interval: int = 5
    seed: int = 0

    # augmentations
    use_augmentations: bool = True

    # early stopping
    patience: int = 10
    min_delta: float = 0.002

    # eval
    iou_threshold: float = 0.4  # IoU threshold for evaluation matching

    # output
    output_dir: str = "checkpoints"  # where to save checkpoints
    project: str = "deepforest-treeai"  # wandb project name


def load_treeai_data(
    base_dir: str, target_species: list[str]
) -> tuple[gpd.GeoDataFrame, str, gpd.GeoDataFrame, str, dict]:
    """Load TreeAI train/val splits using docs/treeai_utils."""
    import sys

    sys.path.insert(0, "docs")
    import treeai_utils

    train_gdf, train_img_dir, label_dict = treeai_utils.get_annot_gdf(
        base_dir, which="train", species=target_species
    )
    val_gdf, val_img_dir, _ = treeai_utils.get_annot_gdf(
        base_dir, which="val", species=target_species
    )
    train_gdf = treeai_utils.ensure_gt_1px(train_gdf)
    val_gdf = treeai_utils.ensure_gt_1px(val_gdf)
    return train_gdf, train_img_dir, val_gdf, val_img_dir, label_dict


def get_augmentations() -> list[dict]:
    return [
        {"HorizontalFlip": {"p": 0.5}},
        {"VerticalFlip": {"p": 0.5}},
        {"Rotate": {"degrees": 45, "p": 0.5}},
        {"RandomBrightnessContrast": {"brightness": 0.2, "contrast": 0.2, "p": 0.5}},
        {"HueSaturationValue": {"hue": 0.1, "saturation": 0.1, "p": 0.5}},
        {"GaussNoise": {"std": 0.05, "p": 0.2}},
        {"GaussianBlur": {"kernel_size": (3, 3), "sigma": (0.1, 1.5), "p": 0.2}},
    ]


def evaluate_model(
    model: deepforest_main.deepforest,
    val_gdf: gpd.GeoDataFrame,
    val_img_dir: str,
    iou_threshold: float,
) -> dict:
    """Run prediction on val images and compute box precision/recall/F1."""
    val_img_filenames = val_gdf["image_path"].unique()
    crown_annot_gdf = val_gdf.assign(label="Tree")

    # predict on each tile
    all_preds = []
    img_paths = [path.join(val_img_dir, f) for f in val_img_filenames]
    pred_gdf = model.predict_tile(img_paths)
    if pred_gdf is not None and len(pred_gdf) > 0:
        all_preds.append(pred_gdf)

    if not all_preds:
        return {"box_precision": 0.0, "box_recall": 0.0, "box_f1": 0.0}

    pred_gdf = gpd.GeoDataFrame(pd.concat(all_preds, ignore_index=True))
    if "geometry" not in pred_gdf.columns:
        from shapely import geometry

        pred_gdf = gpd.GeoDataFrame(
            pred_gdf,
            geometry=pred_gdf.apply(
                lambda x: geometry.box(x.xmin, x.ymin, x.xmax, x.ymax),
                axis="columns",
            ),
        )

    results = eval_utils.evaluate_geometry(
        pred_gdf, crown_annot_gdf, iou_threshold=iou_threshold
    )

    box_precision = float(results["box_precision"])
    box_recall = float(results["box_recall"])
    box_f1 = (
        0.0
        if (box_precision + box_recall) == 0
        else 2 * box_precision * box_recall / (box_precision + box_recall)
    )
    return {
        "box_precision": box_precision,
        "box_recall": box_recall,
        "box_f1": box_f1,
        "n_predictions": len(pred_gdf),
        "n_ground_truth": len(crown_annot_gdf),
        "n_images": len(val_img_filenames),
    }


def main():
    args = sp.parse(Args)
    torch.manual_seed(args.seed)

    console.rule("[bold]DeepForest Training")
    console.print(Panel(str(args), title="Configuration"))

    # --- data ---
    console.rule("Loading data")
    train_gdf, train_img_dir, val_gdf, val_img_dir, label_dict = load_treeai_data(
        args.base_dir, args.target_species
    )
    # crown detection: all labels → "Tree"
    train_crown = train_gdf.assign(label="Tree")
    val_crown = val_gdf.assign(label="Tree")

    console.print(
        f"Train: {len(train_crown)} boxes across "
        f"{train_crown['image_path'].nunique()} images"
    )
    console.print(
        f"Val:   {len(val_crown)} boxes across "
        f"{val_crown['image_path'].nunique()} images"
    )

    # --- model ---
    console.rule("Loading model")
    config_args = {
        "batch_size": args.batch_size,
        "workers": args.workers,
        "train": {
            "lr": args.lr,
            "preload_images": True,
        },
        "validation": {
            "preload_images": True,
            "val_accuracy_interval": args.val_accuracy_interval,
        },
    }
    if args.use_augmentations:
        config_args["train"]["augmentations"] = get_augmentations()

    if args.checkpoint:
        model = deepforest_main.deepforest.load_from_checkpoint(
            args.checkpoint, config_args=config_args
        )
        console.print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        model = deepforest_main.deepforest(config_args=config_args)
        model.load_model(model_name=args.model_name, revision=args.model_revision)
        console.print(f"Loaded pre-trained: {args.model_name}")

    # --- wandb ---
    console.rule("Initializing wandb")
    wandb.init(project=args.project, config=vars(args))
    wandb_logger = WandbLogger(experiment=wandb.run)

    # --- evaluate before training ---
    console.rule("Pre-training evaluation")
    pre_metrics = evaluate_model(model, val_crown, val_img_dir, args.iou_threshold)
    console.print(Panel(
        f"Precision: {pre_metrics['box_precision']:.4f}\n"
        f"Recall:    {pre_metrics['box_recall']:.4f}\n"
        f"F1:        {pre_metrics['box_f1']:.4f}",
        title="Pre-training metrics",
    ))
    wandb.log({"pre_train/" + k: v for k, v in pre_metrics.items()})

    # --- train ---
    console.rule("Training")
    import os

    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        monitor="box_recall",
        mode="max",
        save_top_k=2,
        filename="deepforest-{epoch:02d}-{box_recall:.3f}",
        every_n_epochs=args.val_accuracy_interval,
    )
    early_stop_callback = callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=args.min_delta,
        patience=args.patience,
        verbose=True,
        mode="min",
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        train_csv = path.join(tmp_dir, "train.csv")
        val_csv = path.join(tmp_dir, "val.csv")
        train_crown.to_csv(train_csv, index=False)
        val_crown.to_csv(val_csv, index=False)

        model.config["train"]["csv_file"] = train_csv
        model.config["train"]["root_dir"] = train_img_dir
        model.config["validation"]["csv_file"] = val_csv
        model.config["validation"]["root_dir"] = val_img_dir

        model.create_trainer(
            max_epochs=args.max_epochs,
            precision=args.precision,
            accumulate_grad_batches=args.accumulate_grad_batches,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, early_stop_callback],
        )

        start_time = time.time()
        model.trainer.fit(model)
        train_duration = time.time() - start_time

    console.print(f"Training completed in {train_duration:.1f}s")

    # save final checkpoint
    final_ckpt = path.join(args.output_dir, "deepforest-final.pl")
    model.trainer.save_checkpoint(final_ckpt)
    console.print(f"Saved final checkpoint: {final_ckpt}")

    # --- evaluate after training ---
    console.rule("Post-training evaluation")
    post_metrics = evaluate_model(model, val_crown, val_img_dir, args.iou_threshold)
    console.print(Panel(
        f"Precision: {post_metrics['box_precision']:.4f}\n"
        f"Recall:    {post_metrics['box_recall']:.4f}\n"
        f"F1:        {post_metrics['box_f1']:.4f}",
        title="Post-training metrics",
    ))
    wandb.log({"post_train/" + k: v for k, v in post_metrics.items()})

    # --- summary ---
    console.rule("Summary")
    delta_f1 = post_metrics["box_f1"] - pre_metrics["box_f1"]
    delta_symbol = "+" if delta_f1 >= 0 else ""
    summary = (
        f"Pre-train  F1: {pre_metrics['box_f1']:.4f}\n"
        f"Post-train F1: {post_metrics['box_f1']:.4f}\n"
        f"Delta:         {delta_symbol}{delta_f1:.4f}\n"
        f"Training time: {train_duration:.1f}s\n"
        f"Best checkpoint: {checkpoint_callback.best_model_path or final_ckpt}"
    )
    console.print(Panel(summary, title="Results"))

    wandb.summary.update({
        "train_duration_s": train_duration,
        "best_checkpoint": checkpoint_callback.best_model_path or final_ckpt,
        **{f"final/{k}": v for k, v in post_metrics.items()},
    })
    wandb.finish()


if __name__ == "__main__":
    main()
