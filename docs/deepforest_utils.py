"""DeepForest utils."""

import os
from collections.abc import Mapping
from os import path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from deepforest import evaluate as evaluate_iou
from rasterio import plot

# for type annotations
PathDType = str | os.PathLike
KwargsDType = Mapping | None

# constants
DEFAULT_IOU_THRESHOLD = 0.4


# eval
def multiscale_eval_df(
    predictions: gpd.GeoDataFrame,
    ground_df: gpd.GeoDataFrame,
    tile_dir: PathDType,
    *,
    label_dict: dict = None,
    iou_threshold: float | None = None,
    compute_f1: bool = False,
) -> pd.DataFrame:
    """Compute IoU, recall and precision for multiscale (patch size) predictions.

    Parameters
    ----------
    predictions, ground_df : geopandas.GeoDataFrame
        Predictions and annotations geo-data frames respectively. The prediction data
        frame must have a `patch_size` column.
    tile_dir : path-like
        Path to the directory containing the tiles.
    label_dict : dict, optional
        Label dictionary mapping the class names to the integer ids. If None, it will be
        inferred from the `label` column of `pred_df` and `annot_df`.
    iou_threshold : float, optional
        IoU threshold to use.
    compute_f1 : bool, optional
        Whether to compute the F1 score. Defaults to False.

    Returns
    -------
    eval_df : pd.DataFrame
        Evaluation data frame with the patch size, IoU, recall, precision and optionally
        F1 score columns.

    """
    if iou_threshold is None:
        iou_threshold = DEFAULT_IOU_THRESHOLD
    if label_dict is None:
        label_dict = {
            label: i
            for i, label in enumerate(
                set(predictions["label"]).union(ground_df["label"])
            )
        }
    # numeric_to_label_dict = {val: label for label, val in label_dict.items()}

    def _compute_metrics(_predictions):
        eval_dict = evaluate_iou.__evaluate_wrapper__(
            predictions=_predictions,
            ground_df=ground_df,
            iou_threshold=iou_threshold,
            label_dict=label_dict,
        )

        return [
            eval_dict["results"]["IoU"].mean(),
            eval_dict["box_recall"],
            eval_dict["box_precision"],
        ], eval_dict["class_recall"]

    box_evals = []
    class_eval_dfs = []
    for patch_size, _predictions in predictions.groupby("patch_size"):
        box_eval, class_eval_df = _compute_metrics(_predictions)
        box_evals.append(box_eval)
        class_eval_dfs.append(class_eval_df.assign(patch_size=patch_size))

    # eval_df = predictions.groupby("patch_size").apply(_compute_metrics)
    box_eval_df = pd.DataFrame(
        box_evals,
        columns=["IoU", "box_recall", "box_precision"],
    ).assign(
        **{
            "patch_size": predictions["patch_size"].unique(),
        }
    )
    class_eval_df = pd.concat(class_eval_dfs, ignore_index=True)

    def _compute_f1(eval_df, prefix):
        return (
            2
            * eval_df[f"{prefix}precision"]
            * eval_df[f"{prefix}recall"]
            / (eval_df[f"{prefix}precision"] + eval_df[f"{prefix}recall"])
        ).fillna(0)

    if compute_f1:
        box_eval_df["F1"] = _compute_f1(box_eval_df, "box_")
    if len(label_dict) > 1:
        if compute_f1:
            class_eval_df["F1"] = _compute_f1(class_eval_df, "")
        return box_eval_df, class_eval_df
    else:
        return box_eval_df


# plotting


def plot_img_and_gdf(src, gdf, *, ax=None, **plot_kwargs) -> plt.Axes:
    """Small helper to plot an image and bounding box geo-data frame over it."""
    if ax is None:
        _, ax = plt.subplots()
    plot.show(src, with_bounds=False, ax=ax)
    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()
    gdf.assign(**{"geometry": gdf.boundary}).plot(ax=ax, **plot_kwargs)
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)

    return ax


def plot_annot_vs_pred(
    pred_gdf: gpd.GeoDataFrame,
    annot_gdf: gpd.GeoDataFrame,
    tile_dir: PathDType,
    *,
    figwidth: float | None = None,
    figheight: float | None = None,
    col_wrap: int | None = 3,
    pred_title: str = "predictions",
    legend: bool = True,
    label_dict: Mapping | None = None,
    plot_pred_kwargs: KwargsDType = None,
    plot_annot_kwargs: KwargsDType = None,
) -> plt.Figure:
    """Plot annotations and predictions side by side for each image.

    Parameters
    ----------
    pred_gdf, annot_gdf : geopandas.GeoDataFrame
        Predictions and annotations and geo-data frames respectively.
    tile_dir : path-like
        Path to the directory containing the tiles.
    figwidth, figheight : numeric, optional
        Figure width and height. If None, the matplotlib defaults are used.
    col_wrap : int, default 3
        Number of columns to wrap the plots at. Ignored if the provided value is greater
        than the number of patch sizes.
    pred_title : str, default "predictions"
        Title for the predictions plots.
    legend : bool, default True
        Whether to show the legend on the last plot.
    label_dict : mapping, optional
        Label dictionary mapping the integer ids to class names. If None, the integer
        ids will be used as labels for the legend. Ignored if `legend` is False.
    plot_pred_kwargs, plot_annot_kwargs : mapping, optional
        Keyword arguments to pass to `geopandas.GeoDataFrame.plot` when plotting
        predictions and annotations respectively.

    Returns
    -------
    fig : matplotlib.figure.Figure

    """
    img_filenames = annot_gdf["image_path"].unique()

    if figwidth is None:
        figwidth = plt.rcParams["figure.figsize"][0]
    if figheight is None:
        figheight = plt.rcParams["figure.figsize"][1]
    if plot_annot_kwargs is None:
        _plot_annot_kwargs = {}
    else:
        _plot_annot_kwargs = plot_annot_kwargs.copy()
    if plot_pred_kwargs is None:
        _plot_pred_kwargs = {}
    else:
        _plot_pred_kwargs = plot_pred_kwargs.copy()

    if legend:
        # if there is a crop model label in th predictions, use it for the legend
        if "cropmodel_label" in pred_gdf.columns:
            pred_gdf = pred_gdf.assign(**{"label": pred_gdf["cropmodel_label"]})

        for kwargs in [_plot_annot_kwargs, _plot_pred_kwargs]:
            kwargs["legend"] = kwargs.pop("legend", True)
            kwargs["column"] = kwargs.pop("column", "label")
            kwargs["categorical"] = kwargs.pop("categorical", True)
            kwargs["legend_kwds"] = kwargs.pop(
                "legend_kwds", {"loc": "center left", "bbox_to_anchor": (1, 0.5)}
            )

        if label_dict is not None:
            pred_gdf = pred_gdf.assign(**{"label": pred_gdf["label"].map(label_dict)})
            annot_gdf = annot_gdf.assign(
                **{"label": annot_gdf["label"].map(label_dict)}
            )

    num_imgs = len(img_filenames)
    num_cols = 2
    fig, axes = plt.subplots(
        num_imgs, num_cols, figsize=(figwidth * num_cols, figheight * num_imgs)
    )
    # in case there is only one image
    if num_imgs == 1:
        axes = np.array([axes])

    for img_filename, ax_row in zip(img_filenames, axes):
        with rio.open(path.join(tile_dir, img_filename)) as src:
            for ax, gdf, title in zip(
                ax_row,
                [annot_gdf, pred_gdf],
                ["annotations", pred_title],
            ):
                plot_img_and_gdf(
                    src,
                    gdf[gdf["image_path"] == img_filename],
                    ax=ax,
                    **_plot_pred_kwargs if title == pred_title else _plot_annot_kwargs,
                )
                ax.set_title(title)

    return fig
