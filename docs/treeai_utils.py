"""Utils to process data from the TreeAI Global Initiative."""

import glob
import os
import warnings
from os import path

import geopandas as gpd
import pandas as pd
import rasterio as rio
from rasterio.errors import NotGeoreferencedWarning
from shapely import geometry
from tqdm import tqdm

IMG_FORMAT = "png"
ANNOT_COLUMNS = ["label", "x_center", "y_center", "width", "height"]


def ensure_gt_1px(gdf):
    """Ensure that boxes are strictly larger than 1 px in each dimension (x and y)."""
    return gdf[
        gdf["xmin"].sub(gdf["xmax"]).abs().gt(1)
        & gdf["ymin"].sub(gdf["ymax"]).abs().gt(1)
    ]


def get_annot_gdf(
    base_dir,
    *,
    which: str = "train",
    species: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Get the annotations geo-data frame for the specified split (train or validation).

    Parameters
    ----------
    base_dir : str
        Path to the 12_RGB_ObjDet_640_fL dataset directory.
    which : str, optional
        The split of the dataset to retrieve, either "train" or "val". Defaults to
        "train".
    species : list of str, optional
        If provided, only images containing at least one occurrence of the given species
        are returned. Defaults to None, which returns all images.

    Returns
    -------
    annot_gdf : geopandas.GeoDataFrame
        Annotation data frame following the DeepForest format, i.e., with label,
        bounding box and image path columns.
    label_dict: dict
        Dictionary mapping label names to integer IDs.
    """
    if not path.isdir(base_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {base_dir}. "
            "Please download and extract 12_RGB_ObjDet_640_fL into docs/treeai-data/."
        )
    label_dict = (
        pd.read_excel(path.join(base_dir, "class12_RGB_all_L.xlsx"))
        .set_index(["Sp_ID"])["Sp_Class"]
        .to_dict()
    )
    img_dir = path.join(base_dir, which, "images")
    img_filepaths = glob.glob(path.join(img_dir, f"*.{IMG_FORMAT}"))
    label_filepaths = glob.glob(
        path.join(path.join(base_dir, which, "labels"), "*.txt")
    )
    annot_filepath_ser = pd.Series(img_filepaths + label_filepaths)

    def _process_annot_filepath(annot_filepath):
        img_filename = path.basename(
            annot_filepath.replace("labels", "images").replace("txt", "png")
        )
        df = pd.read_csv(
            annot_filepath,
            names=ANNOT_COLUMNS,
            index_col=False,
            header=None,
            sep=" ",
        ).assign(**{"image_path": img_filename})
        # transform from center x, center y, width, height to xmin, ymin, xmax, ymax in
        # absolute coordinates
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rio.open(path.join(img_dir, img_filename)) as src:
                width = src.width
                height = src.height
                df = df.assign(
                    **{
                        "xmin": (df["x_center"] - df["width"] / 2) * width,
                        "ymin": (df["y_center"] - df["height"] / 2) * height,
                        "xmax": (df["x_center"] + df["width"] / 2) * width,
                        "ymax": (df["y_center"] + df["height"] / 2) * height,
                    }
                ).drop(columns=["x_center", "y_center", "width", "height"])
                # clip to image bounds
                df = df.assign(
                    **{
                        "xmin": df["xmin"].clip(lower=0, upper=width),
                        "ymin": df["ymin"].clip(lower=0, upper=height),
                        "xmax": df["xmax"].clip(lower=0, upper=width),
                        "ymax": df["ymax"].clip(lower=0, upper=height),
                    }
                )

        return df

    annot_df = pd.concat(
        [
            _process_annot_filepath(annot_filepath)
            for annot_filepath in tqdm(
                annot_filepath_ser[
                    annot_filepath_ser.str.split(os.sep).str[-2] == "labels"
                ]
            )
        ],
        ignore_index=True,
    )

    # # round bounds to integers
    # annot_df = annot_df.assign(
    #     **{
    #         col: annot_df[col].round().astype(int)
    #         for col in ["xmin", "ymin", "xmax", "ymax"]
    #     }
    # )

    # make it a geo-data frame
    annot_gdf = gpd.GeoDataFrame(
        annot_df,
        geometry=annot_df.apply(
            lambda row: geometry.box(*row[["xmin", "ymin", "xmax", "ymax"]]),
            axis="columns",
        ),
    )

    # remove bboxes with zero/or less than a pixel area - some annotations have the same
    # xmin-xmax or ymin-ymax, which raises errors when training
    # annot_gdf = annot_gdf[annot_gdf.area != 0]
    # annot_gdf = annot_gdf[annot_gdf.area.gt(1)]
    annot_gdf = annot_gdf[annot_gdf["xmax"].sub(annot_gdf["xmin"]).ge(1)]
    annot_gdf = annot_gdf[annot_gdf["ymax"].sub(annot_gdf["ymin"]).ge(1)]

    if species is not None:
        species_ids = {
            sp_id for sp_id, sp_class in label_dict.items() if sp_class in species
        }
        imgs_with_species = annot_gdf.loc[
            annot_gdf["label"].isin(species_ids), "image_path"
        ].unique()
        annot_gdf = annot_gdf[annot_gdf["image_path"].isin(imgs_with_species)]

    # return the annotations geo-data frame, the path to the images directory and the
    # label dict
    return (
        annot_gdf,
        path.join(base_dir, which, "images"),
        label_dict,
    )
