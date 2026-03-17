"""Train/fine-tune a DeepForest tree crown detection model locally with wandb logging."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

from deepforest_modal_app import eval_utils, plot_utils

console = Console()


@dataclass
class Args:
    """DeepForest crown detection training configuration."""

    # data
    base_dir: str = "/mnt/new-pvc/datasets/treeai/12_RGB_ObjDet_640_fL"  # path to TreeAI dataset
    target_species: list[str] = field(
        default_factory=list  # empty = use all species
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
    score_threshold: float = 0.25  # minimum confidence score for predictions
    patch_size: int = 640  # predict_tile patch size (=image size = no tiling)
    nms_threshold: float = 0.15  # NMS threshold for predict_tile

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

    species_filter = target_species if target_species else None
    train_gdf, train_img_dir, label_dict = treeai_utils.get_annot_gdf(
        base_dir, which="train", species=species_filter
    )
    val_gdf, val_img_dir, _ = treeai_utils.get_annot_gdf(
        base_dir, which="val", species=species_filter
    )
    train_gdf = treeai_utils.ensure_gt_1px(train_gdf)
    val_gdf = treeai_utils.ensure_gt_1px(val_gdf)
    return train_gdf, train_img_dir, val_gdf, val_img_dir, label_dict


def get_augmentations() -> list[dict]:
    return [
        {"HorizontalFlip": {"p": 0.5}},
        {"VerticalFlip": {"p": 0.5}},
        {"Rotate": {"degrees": 45, "p": 0.5}},
        {"RandomResizedCrop": {"size": [640, 640], "scale": [0.5, 1.0], "ratio": [1.0, 1.0], "p": 0.5}},
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
    score_threshold: float = 0.0,
    patch_size: int = 640,
    nms_threshold: float = 0.15,
) -> tuple[dict, gpd.GeoDataFrame | None]:
    """Run prediction on val images and compute box precision/recall/F1.

    Returns a (metrics dict, pred_gdf) tuple; pred_gdf is None when the model
    makes no predictions.
    """
    val_img_filenames = val_gdf["image_path"].unique()
    crown_annot_gdf = val_gdf.assign(label="Tree")

    # predict on each tile
    all_preds = []
    img_paths = [path.join(val_img_dir, f) for f in val_img_filenames]
    pred_gdf = model.predict_tile(img_paths, patch_size=patch_size, iou_threshold=nms_threshold)
    if pred_gdf is not None and len(pred_gdf) > 0:
        all_preds.append(pred_gdf)

    if not all_preds:
        return {"box_precision": 0.0, "box_recall": 0.0, "box_f1": 0.0}, None

    pred_gdf = gpd.GeoDataFrame(pd.concat(all_preds, ignore_index=True))
    if score_threshold > 0.0 and "score" in pred_gdf.columns:
        pred_gdf = pred_gdf[pred_gdf["score"] >= score_threshold].reset_index(drop=True)
    if len(pred_gdf) == 0:
        return {"box_precision": 0.0, "box_recall": 0.0, "box_f1": 0.0}, None
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
    }, pred_gdf


def log_visualizations(
    pred_gdf: gpd.GeoDataFrame | None,
    val_gdf: gpd.GeoDataFrame,
    val_img_dir: str,
    iou_threshold: float,
    n_images: int = 8,
    seed: int = 0,
) -> None:
    """Log side-by-side annotation vs prediction plots to a wandb.Table.

    Images are chosen deterministically using *seed* so the same tiles are
    shown across all experiments, making runs directly comparable.
    """
    crown_annot_gdf = val_gdf.assign(label="Tree")
    all_img_filenames = val_gdf["image_path"].unique()

    rng = np.random.default_rng(seed)
    selected = rng.choice(all_img_filenames, size=min(n_images, len(all_img_filenames)), replace=False)

    # empty GeoDataFrame used when the model produced no predictions
    empty_pred = gpd.GeoDataFrame(columns=["image_path", "geometry", "label"])

    table = wandb.Table(
        columns=["image_path", "visualization", "n_ground_truth", "n_predictions",
                 "box_precision", "box_recall", "box_f1"]
    )

    for img_filename in selected:
        img_annot = crown_annot_gdf[crown_annot_gdf["image_path"] == img_filename]
        img_pred = (
            pred_gdf[pred_gdf["image_path"] == img_filename]
            if pred_gdf is not None and len(pred_gdf) > 0
            else empty_pred
        )

        # per-image precision / recall / F1
        if len(img_pred) > 0 and len(img_annot) > 0:
            img_results = eval_utils.evaluate_geometry(
                img_pred, img_annot, iou_threshold=iou_threshold
            )
            bp = float(img_results["box_precision"])
            br = float(img_results["box_recall"])
        else:
            bp = br = 0.0
        bf1 = 0.0 if (bp + br) == 0 else 2 * bp * br / (bp + br)

        fig = plot_utils.plot_annot_vs_pred(
            img_pred,
            img_annot,
            val_img_dir,
            plot_pred_kwargs={"legend": False, "column": None},
            plot_annot_kwargs={"legend": False, "column": None},
        )
        table.add_data(
            img_filename,
            wandb.Image(fig),
            len(img_annot),
            len(img_pred),
            round(bp, 4),
            round(br, 4),
            round(bf1, 4),
        )
        plt.close(fig)

    wandb.log({"post_train/visualizations": table})
    console.print(f"Logged {len(selected)} visualization(s) to wandb")


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
    pre_metrics, _ = evaluate_model(model, val_crown, val_img_dir, args.iou_threshold, args.score_threshold, args.patch_size, args.nms_threshold)
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
    post_metrics, post_pred_gdf = evaluate_model(model, val_crown, val_img_dir, args.iou_threshold, args.score_threshold, args.patch_size, args.nms_threshold)
    console.print(Panel(
        f"Precision: {post_metrics['box_precision']:.4f}\n"
        f"Recall:    {post_metrics['box_recall']:.4f}\n"
        f"F1:        {post_metrics['box_f1']:.4f}",
        title="Post-training metrics",
    ))
    wandb.log({"post_train/" + k: v for k, v in post_metrics.items()})

    # --- visualizations ---
    console.rule("Logging visualizations")
    log_visualizations(
        post_pred_gdf, val_crown, val_img_dir,
        iou_threshold=args.iou_threshold,
        n_images=8,
        seed=args.seed,
    )

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
