"""Utils to process data from the TreeAI Global Initiative."""

import os
import warnings
from os import path

import geopandas as gpd
import pandas as pd
import pooch
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from shapely import geometry
from tqdm.auto import tqdm

# TODO: support further datasets from TreeAI
DATASET_URL = (
    "https://zenodo.org/records/15351054/files/12_RGB_ObjDet_640_fL.zip?download=1"
)
IMG_FORMAT = ".png"
ANNOT_COLUMNS = ["label", "x_center", "y_center", "width", "height"]


def resize_img(img_filepath: str, resize_factor: float, resampling: int) -> None:
    """Resize the image at the given file path by the specified ratio."""
    with rio.open(img_filepath) as src:
        profile = src.profile
        profile.update(
            {
                "width": int(src.width * resize_factor),
                "height": int(src.height * resize_factor),
            }
        )
        dst_arr = src.read(
            out_shape=(src.count, profile["height"], profile["width"]),
            resampling=resampling,
        )
        transform = src.transform * src.transform.scale(
            (src.width / dst_arr.shape[-1]),
            (src.height / dst_arr.shape[-2]),
        )
        profile.update({"transform": transform})
    with rio.open(img_filepath, "w", **profile) as dst:
        dst.write(dst_arr)


class UnzipAndResize(pooch.Unzip):
    """Custom Pooch processor to unzip and resize images in the dataset."""

    def __init__(
        self,
        resize_factor: float,
        resampling: int = Resampling.bilinear,
        *args,
        **kwargs,
    ):
        """Initialize the processor with a resize factor."""
        super().__init__(*args, **kwargs)
        self.resize_factor = resize_factor
        self.resampling = resampling

    @property
    def suffix(self):
        """Return the suffix for the processor."""
        return f"_{str(self.resize_factor).replace('.', '_')}_resize.unzip"

    def _extract_file(self, fname, extract_dir):
        """Extract the file and resize it if it is an image."""
        super()._extract_file(fname, extract_dir)
        for member in self._all_members(fname):
            if member.endswith(IMG_FORMAT):
                resize_img(
                    path.join(extract_dir, member), self.resize_factor, self.resampling
                )


def get_annot_gdf(
    *, which: str = "train", resize_factor: float | None = None
) -> tuple[pd.DataFrame, str, dict]:
    """Get the annotations geo-data frame for the specified split (train or validation).

    Parameters
    ----------
    which : str, optional
        The split of the dataset to retrieve, either "train" or "val". Defaults to
        "train".

    Returns
    -------
    annot_gdf : geopandas.GeoDataFrame
        Annotation data frame following the DeepForest format, i.e., with label,
        bounding box and image path columns.
    img_dir : str
        Directory where the images are stored.
    label_dict: dict
        Dictionary mapping label names to integer IDs.
    """
    if resize_factor is not None:
        processor = UnzipAndResize(resize_factor=resize_factor)
    else:
        processor = pooch.Unzip()
    filepaths = pooch.retrieve(
        url=DATASET_URL,
        known_hash=None,
        processor=processor,
        progressbar=True,
    )

    base_dir = path.commonprefix(filepaths)
    img_dir = path.join(base_dir, which, "images")
    filepath_ser = pd.Series([filepath[len(base_dir) :] for filepath in filepaths])

    annot_filepath_ser = filepath_ser[filepath_ser.str.split(os.sep).str[-3] == which]

    def _process_annot_filepath(annot_filepath):
        img_filename = path.basename(
            annot_filepath.replace("labels", "images").replace("txt", "png")
        )
        df = pd.read_csv(
            path.join(base_dir, annot_filepath),
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

    # remove bboxes with zero area - some annotations have the same xmin-xmax or
    # ymin-ymax, which raises errors when training
    annot_gdf = annot_gdf[annot_gdf.area != 0]

    # return the annotations geo-data frame, the path to the images directory and the
    # label dict
    return (
        annot_gdf,
        path.join(base_dir, which, "images"),
        pd.read_excel(path.join(base_dir, "class12_RGB_all_L.xlsx"))
        .set_index(["Sp_ID"])["Sp_Class"]
        .to_dict(),
    )
