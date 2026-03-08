"""Evaluation utils.

This is essentially hard-copied from the DeepForest `evaluate` and `utilities` modules
to avoid having to install the whole library and especially its pytorch dependencies.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.optimize import linear_sum_assignment
from shapely import STRtree


# utilities
class DeepForest_DataFrame(gpd.GeoDataFrame):
    """Custom GeoDataFrame that preserves a root_dir attribute if present."""

    _metadata = ["root_dir"]

    def __init__(self, *args, **kwargs):
        root_dir = getattr(args[0], "root_dir", None) if args else None
        super().__init__(*args, **kwargs)
        if root_dir is not None:
            self.root_dir = root_dir

    @property
    def _constructor(self):
        return DeepForest_DataFrame


def determine_geometry_type(df):
    """Determine the geometry type of a prediction or annotation
    Args:
        df: a pandas dataframe
    Returns:
        geometry_type: a string of the geometry type
    """
    if type(df) in [pd.DataFrame, gpd.GeoDataFrame, DeepForest_DataFrame]:
        columns = df.columns
        if "geometry" in columns:
            df = gpd.GeoDataFrame(geometry=df["geometry"])
            geometry_type = df.geometry.type.unique()[0]
            if geometry_type == "Polygon":
                if (df.geometry.area == df.envelope.area).all():
                    return "box"
                else:
                    return "polygon"
            else:
                return "point"
        elif (
            "xmin" in columns
            and "ymin" in columns
            and "xmax" in columns
            and "ymax" in columns
        ):
            geometry_type = "box"
        elif "polygon" in columns:
            geometry_type = "polygon"
        elif "x" in columns and "y" in columns:
            geometry_type = "point"
        else:
            raise ValueError(
                f"Could not determine geometry type from columns {columns}"
            )

    elif isinstance(df, dict):
        if "boxes" in df.keys():
            geometry_type = "box"
        elif "polygon" in df.keys():
            geometry_type = "polygon"
        elif "points" in df.keys():
            geometry_type = "point"

    return geometry_type


def __pandas_to_geodataframe__(df: pd.DataFrame):
    """Create a geometry column from a pandas dataframe with coordinates".

    Args:
        df: a pandas dataframe with columns: xmin, ymin, xmax, ymax, or x, y, or polygon
    Returns:
        gdf: a geodataframe with a geometry column
    """
    # If the geometry column is present, convert to geodataframe directly
    if "geometry" in df.columns:
        if pd.api.types.infer_dtype(df["geometry"]) == "string":
            df["geometry"] = gpd.GeoSeries.from_wkt(df["geometry"])
    else:
        geom_type = determine_geometry_type(df)
        if geom_type == "box":
            df["geometry"] = df.apply(
                lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1
            )
        elif geom_type == "polygon":
            df["geometry"] = gpd.GeoSeries.from_wkt(df["polygon"])
        elif geom_type == "point":
            df["geometry"] = gpd.GeoSeries(
                [
                    shapely.geometry.Point(x, y)
                    for x, y in zip(
                        df.x.astype(float), df.y.astype(float), strict=False
                    )
                ]
            )
    gdf = gpd.GeoDataFrame(df, geometry="geometry")
    gdf = DeepForest_DataFrame(gdf)

    return gdf


# IoU
def _overlap_all(test_polys: "gpd.GeoDataFrame", truth_polys: "gpd.GeoDataFrame"):
    """Computes intersection and union areas for all polygons in the test/truth
    dataframes.

    For efficient querying, truth polygons are stored in a spatial R-Tree
    and we only compute intersections/unions for matching pairs. The output from the
    function are Numpy arrays containing the all-to-all intersection and union areas and
    the indices of intersecting ground truth and prediction polygons.

    This method works with any Shapely polygon, but may have
    reduced performance for the polygon case where bounding box intersection does
    not necessarily mean the vertices intersect. For rectangles, it's efficient
    as the an r-tree hit is usually a true intersection, depending on how touching
    edge cases are handled.

    Returns
    -------
      intersections  : (n_truth, n_pred) intersection areas
      unions : (n_truth, n_pred) union areas
      truth_ids : (n_truth,) truth index values (order matches rows of areas/unions)
      pred_ids  : (n_pred,) prediction index values (order matches cols of areas/unions)
    """
    # geometry arrays
    pred_geoms = np.asarray(test_polys.geometry.values, dtype=object)
    truth_geoms = np.asarray(truth_polys.geometry.values, dtype=object)

    pred_ids = test_polys.index.to_numpy()
    truth_ids = truth_polys.index.to_numpy()

    n_pred = pred_geoms.size
    n_truth = truth_geoms.size

    # empty cases
    if n_pred == 0 or n_truth == 0:
        return (
            np.zeros((n_truth, n_pred), dtype=float),
            np.zeros((n_truth, n_pred), dtype=float),
            truth_ids,
            pred_ids,
        )

    # spatial index on truth
    tree = STRtree(truth_geoms)
    p_idx, t_idx = tree.query(pred_geoms, predicate="intersects")  # shape (2, M)

    intersections = np.zeros((n_truth, n_pred), dtype=float)
    unions = np.zeros((n_truth, n_pred), dtype=float)

    if p_idx.size:
        inter = shapely.intersection(truth_geoms[t_idx], pred_geoms[p_idx])
        uni = shapely.union(truth_geoms[t_idx], pred_geoms[p_idx])
        intersections[t_idx, p_idx] = shapely.area(inter)
        unions[t_idx, p_idx] = shapely.area(uni)

    return intersections, unions, truth_ids, pred_ids


def match_polygons(ground_truth: "gpd.GeoDataFrame", submission: "gpd.GeoDataFrame"):
    """Find area of overlap among all sets of ground truth and prediction.

    This function performs matching between a ground truth dataset and a
    submission or prediction dataset, typically the output from a validation or
    test run. In order to compute IoU, we must know which boxes correspond
    between the datasets. This is performed by Hungarian matching, or linear
    sum assignment.

    For each ground truth polygon, we compute the IoUs of all
    overlapping polygons. Intersection areas are used as the input cost matrix for the
    assignment and the algorithm is such that at most one prediction is assigned to each
    ground truth, and each prediction is only used at most once, with the solver aiming
    to maximise the total area of intersection. The matching indices are then returned,
    along with their IoUs and scores, to be used in downstream metrics like recall and
    precision.

    No filtering on IoU or score is performed.

    Args:
        ground_truth: a projected geopandas dataframe with geometry
        submission: a projected geopandas dataframe with geometry
    Returns:
        iou_df: dataframe of IoU scores
    """
    plot_names = submission["image_path"].unique()
    if len(plot_names) > 1:
        raise ValueError(f"More than one image passed to function: {plot_names[0]}")

    # Compute truth <> prediction overlaps
    intersections, unions, truth_ids, pred_ids = _overlap_all(
        test_polys=submission, truth_polys=ground_truth
    )

    # Cost matrix is the intersection area
    matrix = intersections

    if matrix.size == 0:
        # No matches, early exit
        return pd.DataFrame(
            {
                "prediction_id": pd.Series(dtype="float64"),
                "truth_id": pd.Series(dtype=truth_ids.dtype),
                "IoU": pd.Series(dtype="float64"),
                "score": pd.Series(dtype="float64"),
                "geometry": pd.Series(dtype=object),
            }
        )

    # Linear sum assignment + match lookup
    row_ind, col_ind = linear_sum_assignment(matrix, maximize=True)
    match_for_truth = dict(zip(row_ind, col_ind, strict=False))

    # Score lookup
    pred_scores = submission["score"].to_dict() if "score" in submission.columns else {}

    # IoU matrix
    with np.errstate(divide="ignore", invalid="ignore"):
        iou_mat = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections, dtype=float),
            where=unions > 0,
        )

    # build rows for every truth element (unmatched => None, IoU 0)
    records = []
    for t_idx, truth_id in enumerate(truth_ids):
        # If we matched this truth box
        if t_idx in match_for_truth:
            # Look up matching prediction and corresponding IoU and score
            p_idx = match_for_truth[t_idx]
            matched_id = pred_ids[p_idx]
            iou = float(iou_mat[t_idx, p_idx])
            score = pred_scores.get(matched_id, None)
        else:
            matched_id = None
            iou = 0.0
            score = None
        records.append(
            {
                "prediction_id": matched_id,
                "truth_id": truth_id,
                "IoU": iou,
                "score": score,
            }
        )

    # Output dataframe
    iou_df = pd.DataFrame.from_records(records)
    iou_df = iou_df.merge(
        ground_truth.assign(truth_id=truth_ids)[["truth_id", "geometry"]],
        on="truth_id",
        how="left",
    )
    return iou_df


def match_points(
    ground_truth: "gpd.GeoDataFrame", submission: "gpd.GeoDataFrame", norm: str = "l2"
):
    """Find distance among all sets of ground truth and prediction points.

    This function performs matching between a ground truth dataset and a
    submission or prediction dataset, typically the output from a validation or
    test run. In order to compute distances, we must know which points correspond
    between the datasets. This is performed by Hungarian matching, or linear
    sum assignment.

    For each ground truth point, we compute the L2 distances of all
    overlapping points. The distance matrix is used as the input cost matrix for the
    assignment and the algorithm is such that at most one prediction is assigned to each
    ground truth, and each prediction is only used at most once, with the solver aiming
    to minimise the total distance. The matching indices are then returned, along with
    their distances and scores, to be used in downstream metrics like recall and
    precision.

    No filtering on distance or score is performed.

    Args:
        ground_truth: a projected geopandas dataframe with geometry
        submission: a projected geopandas dataframe with geometry
        norm: distance norm to use ("l1" or "l2")

    Returns
    -------
        dist_df: dataframe of distances
    """
    plot_names = submission["image_path"].unique()
    if len(plot_names) > 1:
        raise ValueError(f"More than one image passed to function: {plot_names[0]}")

    # Compute pairwise distances
    pred_geoms = np.asarray(submission.geometry.values, dtype=object)
    truth_geoms = np.asarray(ground_truth.geometry.values, dtype=object)

    pred_ids = submission.index.to_numpy()
    truth_ids = ground_truth.index.to_numpy()

    n_pred = pred_geoms.size
    n_truth = truth_geoms.size

    # empty cases
    if n_pred == 0 or n_truth == 0:
        return pd.DataFrame(
            {
                "prediction_id": pd.Series(dtype="float64"),
                "truth_id": pd.Series(dtype=truth_ids.dtype),
                "distance": pd.Series(dtype="float64"),
                "score": pd.Series(dtype="float64"),
                "geometry": pd.Series(dtype=object),
            }
        )
    distances = np.full((n_truth, n_pred), np.inf, dtype=float)
    for t_idx in range(n_truth):
        for p_idx in range(n_pred):
            if norm.lower() == "l2":
                distances[t_idx, p_idx] = truth_geoms[t_idx].distance(pred_geoms[p_idx])
            elif norm.lower() == "l1":
                diff = shapely.geometry.Point(
                    abs(truth_geoms[t_idx].x - pred_geoms[p_idx].x),
                    abs(truth_geoms[t_idx].y - pred_geoms[p_idx].y),
                )
                distances[t_idx, p_idx] = diff.x + diff.y
            else:
                raise ValueError(f"Unknown norm type: {norm}")

    # Linear sum assignment + match lookup
    row_ind, col_ind = linear_sum_assignment(distances, maximize=False)
    match_for_truth = dict(zip(row_ind, col_ind, strict=False))

    # Score lookup
    pred_scores = submission["score"].to_dict() if "score" in submission.columns else {}
    # build rows for every truth element (unmatched => None, distance inf)
    records = []
    for t_idx, truth_id in enumerate(truth_ids):
        # If we matched this truth point
        if t_idx in match_for_truth:
            # Look up matching prediction and corresponding distance and score
            p_idx = match_for_truth[t_idx]
            matched_id = pred_ids[p_idx]
            distance = float(distances[t_idx, p_idx])
            score = pred_scores.get(matched_id, None)
        else:
            matched_id = None
            distance = float("inf")
            score = None
        records.append(
            {
                "prediction_id": matched_id,
                "truth_id": truth_id,
                "distance": distance,
                "score": score,
            }
        )
    dist_df = pd.DataFrame.from_records(records)
    dist_df = dist_df.merge(
        ground_truth.assign(truth_id=truth_ids)[["truth_id", "geometry"]],
        on="truth_id",
        how="left",
    )
    return dist_df


# evaluate
def _empty_result_dataframe_(group, image_path, task="box"):
    """Create an empty result dataframe for images with no predictions."""
    result_dict = {
        "truth_id": group.index.values,
        "prediction_id": pd.Series([None] * len(group), dtype="object"),
        "geometry": group.geometry,
        "image_path": image_path,
        "match": pd.Series([False] * len(group), dtype="bool"),
        "score": pd.Series([None] * len(group), dtype="float64"),
        "predicted_label": pd.Series([None] * len(group), dtype="object"),
        "true_label": group.label,
    }

    if task == "box" or task == "polygon":
        result_dict.update(
            {
                "IoU": pd.Series([0.0] * len(group), dtype="float64"),
            }
        )

    return pd.DataFrame(result_dict)


def match_predictions(predictions, ground_df, task="box"):
    """Compute intersection-over-union matching among prediction and ground
    truth geometries for one image. The returned results are guaranteed to be
    at most one-to-one, but are not filtered for "quality" of match (i.e. IoU
    threshold).

    Args:
        predictions: a geopandas dataframe with geometry columns
        ground_df: a geopandas dataframe with geometry columns

    Returns
    -------
        result: pandas dataframe with crown ids of prediction and ground truth and the
        IoU score.
    """
    plot_names = predictions["image_path"].unique()
    if len(plot_names) > 1:
        raise ValueError(f"More than one plot passed to image crown: {plot_names}")

    # match
    if task in ["box", "polygon"]:
        # result = IoU.match_polygons(ground_df, predictions)
        result = match_polygons(ground_df, predictions)
    elif task == "point":
        # result = IoU.match_points(ground_df, predictions, norm="l2")
        result = match_points(ground_df, predictions, norm="l2")
    else:
        raise NotImplementedError(f"Geometry type {task} not implemented")

    # Map prediction/truth IDs back to their original labels from input dataframes
    pred_label_dict = predictions.label.to_dict()
    ground_label_dict = ground_df.label.to_dict()
    result["predicted_label"] = result.prediction_id.map(pred_label_dict)
    result["true_label"] = result.truth_id.map(ground_label_dict)

    return result


def compute_class_recall(results):
    """Given a set of evaluations, what proportion of predicted boxes match.

    True boxes which are not matched to predictions do not count against
    accuracy.
    """
    # Per class recall and precision
    class_recall_dict = {}
    class_precision_dict = {}
    class_size = {}

    box_results = results[results.predicted_label.notna()]
    if box_results.empty:
        print("No predictions made")
        class_recall = None
        return class_recall

    # Get all labels from both predictions and ground truth
    predicted_labels = set(box_results["predicted_label"].dropna())
    true_labels = set(box_results["true_label"].dropna())
    all_labels = predicted_labels.union(true_labels)

    for label in all_labels:
        # Recall: of all ground truth boxes with this label, how many were correctly
        # predicted?
        ground_df = box_results[box_results["true_label"] == label]
        n_ground_boxes = ground_df.shape[0]
        if n_ground_boxes > 0:
            class_recall_dict[label] = (
                sum(ground_df.true_label == ground_df.predicted_label) / n_ground_boxes
            )

        # Precision: of all predictions with this label, how many were correct?
        pred_df = box_results[box_results["predicted_label"] == label]
        n_pred_boxes = pred_df.shape[0]
        if n_pred_boxes > 0:
            class_precision_dict[label] = (
                sum(pred_df.true_label == pred_df.predicted_label) / n_pred_boxes
            )

        class_size[label] = n_ground_boxes

    # fillna(0) handles labels with no ground truth (recall=0) or no predictions
    # (precision=0)
    class_recall = (
        pd.DataFrame(
            {
                "recall": pd.Series(class_recall_dict),
                "precision": pd.Series(class_precision_dict),
                "size": pd.Series(class_size),
            }
        )
        .reset_index(names="label")
        .fillna(0)
        .sort_values("label")
    )

    return class_recall


def evaluate_geometry(
    predictions: pd.DataFrame | gpd.GeoDataFrame,
    ground_df: pd.DataFrame | gpd.GeoDataFrame,
    iou_threshold: float = 0.4,
    distance_threshold: float = 10.0,
    geometry_type: str = "box",
) -> dict:
    """Image annotated crown evaluation routine submission can be submitted as
    a .shp, existing pandas dataframe or .csv path.

    Args:
        predictions: a pandas dataframe with geometry columns. The labels in ground
        truth and predictions must match. If one is numeric, the other must be numeric.
        ground_df: a pandas dataframe with geometry columns
        iou_threshold: intersection-over-union threshold, see deepforest.evaluate
        l2_threshold: L2 distance threshold for point matching
        geometry_type: 'box', 'polygon' or 'point'

    Returns
    -------
        results: a dataframe of match bounding boxes
        box_recall: proportion of true positives of box position, regardless of class
        box_precision: proportion of predictions that are true positive, regardless of
        class
        class_recall: a pandas dataframe of class level recall and precision with class
        sizes
    """
    if geometry_type not in ["box", "polygon", "point"]:
        raise ValueError(
            f"Unknown geometry type {geometry_type}. Must be one of 'box', 'polygon' or"
            " 'point'."
        )

    # If no predictions, return 0 recall, NaN precision
    if predictions.empty:
        return {
            "results": None,
            f"{geometry_type}_recall": 0,
            f"{geometry_type}_precision": np.nan,
            "class_recall": None,
            "predictions": predictions,
            "ground_df": ground_df,
        }
    elif not isinstance(predictions, gpd.GeoDataFrame):
        predictions = __pandas_to_geodataframe__(predictions)

    # Remove empty ground truth boxes
    if geometry_type == "box":
        ground_df = ground_df[
            ~(
                (ground_df.xmin == 0)
                & (ground_df.xmax == 0)
                & (ground_df.ymin == 0)
                & (ground_df.ymax == 0)
            )
        ]
    elif geometry_type == "polygon":
        ground_df = ground_df[~ground_df.geometry.is_empty]
    elif geometry_type == "point":
        ground_df = ground_df[~((ground_df.x == 0) & (ground_df.y == 0))]

    # If all empty ground truth, return 0 recall and precision
    if ground_df.empty:
        return {
            "results": None,
            f"{geometry_type}_recall": None,
            f"{geometry_type}_precision": 0,
            "class_recall": None,
            "predictions": predictions,
            "ground_df": ground_df,
        }

    if not isinstance(ground_df, gpd.GeoDataFrame):
        ground_df = __pandas_to_geodataframe__(ground_df)

    # Pre-group predictions by image
    predictions_by_image = {
        name: group.reset_index(drop=True)
        for name, group in predictions.groupby("image_path")
    }

    # Run evaluation on all plots
    results = []
    per_image_recalls = []
    per_image_precisions = []
    for image_path, group in ground_df.groupby("image_path"):
        # Predictions for this image
        image_predictions = predictions_by_image.get(image_path, pd.DataFrame())

        # If empty, add to list without computing IoU
        if image_predictions.empty:
            # Reset index
            group = group.reset_index(drop=True)
            result = _empty_result_dataframe_(group, image_path, task=geometry_type)
            # An empty prediction set has recall of 0, precision of NA.
            per_image_recalls.append(0)
            results.append(result)
            continue
        else:
            group = group.reset_index(drop=True)
            result = match_predictions(
                predictions=image_predictions, ground_df=group, task=geometry_type
            )

        result["image_path"] = image_path

        # Determine matches based on IoU or distance thresholds
        if geometry_type == "box" or geometry_type == "polygon":
            result["match"] = result.IoU > iou_threshold
        elif geometry_type == "point":
            result["match"] = result.distance < distance_threshold

        # Convert None to False for boolean consistency
        result["match"] = result["match"].fillna(False)
        true_positive = sum(result["match"])
        recall = true_positive / result.shape[0]
        precision = true_positive / image_predictions.shape[0]

        per_image_recalls.append(recall)
        per_image_precisions.append(precision)
        results.append(result)

    # Concatenate results
    if results:
        results = pd.concat(results, ignore_index=True)
        # Convert back to GeoDataFrame if it has geometry column
        if "geometry" in results.columns:
            results = gpd.GeoDataFrame(results, geometry="geometry")
    else:
        columns = [
            "truth_id",
            "prediction_id",
            "predicted_label",
            "score",
            "match",
            "true_label",
            "geometry",
            "image_path",
        ]

        if geometry_type == "box" or geometry_type == "polygon":
            columns.append("IoU")
        elif geometry_type == "point":
            columns.append("distance")

        results = gpd.GeoDataFrame(columns=columns)

    mean_precision = np.mean(per_image_precisions)
    mean_recall = np.mean(per_image_recalls)

    # Only matching boxes are considered in class recall
    matched_results = results[results.match]
    class_recall = compute_class_recall(matched_results)

    return {
        "results": results,
        f"{geometry_type}_precision": mean_precision,
        f"{geometry_type}_recall": mean_recall,
        "class_recall": class_recall,
        "predictions": predictions,
        "ground_df": ground_df,
    }
