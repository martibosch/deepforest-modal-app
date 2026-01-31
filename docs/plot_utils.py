"""Plotting utils."""

import os
from collections.abc import Mapping
from os import path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from rasterio import plot

# for type annotations
PathDType = str | os.PathLike
KwargsDType = Mapping | None


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
