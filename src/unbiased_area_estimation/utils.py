import functools
import json
import os
import time

import numpy as np
import pandas as pd


def read_config(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)

    config["class_merge_dict"] = {
        int(k): int(v) for k, v in config["class_merge_dict"].items()
    }
    config["sampling"]["expected_uas"] = {
        int(k): float(v) for k, v in config["sampling"]["expected_uas"].items()
    }
    return config


# Currently stored as utils, Will be refactored the AreaEstimator


def load_sampling_design(region_name: str, sampling_design_folder: str):
    sampling_design_file = f"{region_name}_design.csv"
    sampling_design_path = os.path.join(sampling_design_folder, sampling_design_file)
    sampling_design = pd.read_csv(sampling_design_path)
    return sampling_design


def compute_unbiased_areas_from_ceo(
    annotated_samples_folder: str, annotated_column_name: str = "class_id"
):
    annotated_sample_files = [
        f for f in os.listdir(annotated_samples_folder) if f.endswith("samples.csv")
    ]
    sampling_design_files = [
        f for f in os.listdir(annotated_samples_folder) if f.endswith("design.csv")
    ]

    results = {}

    for annotated_sample_file in annotated_sample_files:
        region_name = annotated_sample_file.split("_")[0]
        sampling_desing_file = annotated_sample_file.replace(
            "samples.csv", "design.csv"
        )
        if sampling_desing_file not in sampling_design_files:
            raise ValueError(
                f"No corresponding sampling design file found for {annotated_sample_file}."
            )

        df_samples = pd.read_csv(
            os.path.join(annotated_samples_folder, annotated_sample_file)
        )
        predicted = df_samples["stratum_id"]
        true = df_samples[annotated_column_name]

        df_design = pd.read_csv(
            os.path.join(annotated_samples_folder, sampling_desing_file)
        )
        classes = df_design.index
        areas = df_design["area"]
        weights = df_design["wh"]

        weights_dict = {class_id: weight for class_id, weight in zip(classes, weights)}
        areas_dict = {class_id: area for class_id, area in zip(classes, areas)}

        classwise_metrics_df, overall_metrics_df = compute_unbiased_area_estimates(
            predicted, true, weights_dict, areas_dict
        )
        results[region_name] = {
            "classwise_metrics": classwise_metrics_df,
            "overall_metrics": overall_metrics_df,
        }

    return results


def compute_unbiased_area_estimates(predicted, annotated, wh, areas):
    classes = np.unique(annotated)
    num_classes = len(classes)
    confusion_matrix = np.zeros((num_classes, num_classes))

    # Compute confusion matrix
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            confusion_matrix[i, j] = np.sum(
                (annotated == true_class) & (predicted == pred_class)
            )

    counts_pred = np.sum(confusion_matrix, axis=1)  # Total predicted per class
    counts_true = np.sum(confusion_matrix, axis=0)  # Total actual per class
    total_positive = np.diag(confusion_matrix)  # True positives
    users_accuracy_by_class = np.diag(confusion_matrix) / counts_pred  # User's accuracy
    producers_accuracy_by_class = (
        np.diag(confusion_matrix) / counts_true
    )  # Producer's accuracy

    # Ensure wh dictionary matches class indices
    if set(classes) != set(wh.keys()):
        raise ValueError(
            f"Mismatch between confusion matrix classes {set(classes)} and weight keys {set(wh.keys())}. Not yet implemented."
        )

    weights = np.array([wh[class_id] for class_id in classes])
    areas = np.array([areas[class_id] for class_id in classes])
    weighted_confusion_matrix = confusion_matrix * weights[:, np.newaxis]
    weighted_norm_confusion_matrix = (
        weighted_confusion_matrix / counts_pred[:, np.newaxis]
    )
    oa = np.sum(np.diag(weighted_norm_confusion_matrix))

    total_pred_weight_norm = np.sum(weighted_norm_confusion_matrix, axis=1)
    total_true_weight_norm = np.sum(weighted_norm_confusion_matrix, axis=0)
    users_accuracy_by_class_norm = (
        np.diag(weighted_norm_confusion_matrix) / total_pred_weight_norm
    )
    producers_accuracy_by_class_norm = (
        np.diag(weighted_norm_confusion_matrix) / total_true_weight_norm
    )

    temp = (
        np.square(weights)
        * users_accuracy_by_class
        * (1 - users_accuracy_by_class)
        / (counts_pred - 1)
    )  # Validate why -1 for pred
    v_oa = np.sum(temp)
    se_oa = np.sqrt(v_oa)
    oa_ci = 1.96 * se_oa

    v_ua = users_accuracy_by_class * (1 - users_accuracy_by_class) / (counts_pred - 1)
    se_ua = np.sqrt(v_ua)
    ua_ci = 1.96 * se_ua

    N_j = []
    for j in range(num_classes):
        n_j = np.sum(confusion_matrix[:, j] / counts_pred * areas)
        N_j.append(n_j)

    N_j = np.array(N_j)

    xp1 = (
        np.square(areas)
        * np.square((1 - producers_accuracy_by_class_norm))
        * users_accuracy_by_class_norm
        * (1 - users_accuracy_by_class_norm)
        / (counts_pred - 1)
    )

    conf_no_diag = np.triu(confusion_matrix, k=1) + np.tril(confusion_matrix, k=-1)
    prod_a_sq = np.square(producers_accuracy_by_class_norm)
    areas_squared = np.square(areas)
    frac = conf_no_diag / counts_pred[:, np.newaxis]
    var_term = frac * (1 - frac) / (counts_pred[:, np.newaxis] - 1)
    xp2 = prod_a_sq * np.sum(areas_squared[:, np.newaxis] * var_term, axis=0)

    v_pa = (1 / np.square(N_j)) * (xp1 + xp2)
    se_pa = np.sqrt(v_pa)
    pa_ci = 1.96 * se_pa

    s_pk_arr = []
    for i in range(num_classes):
        ir = np.sum(
            (
                weights * weighted_norm_confusion_matrix[:, i]
                - np.square(weighted_norm_confusion_matrix[:, i])
            )
            / (counts_pred - 1)
        )
        s_pk_arr.append(ir)
    s_pk = np.sqrt(np.array(s_pk_arr))

    se_areas = s_pk * areas
    areas_ci = 1.96 * se_areas
    uc_ci = areas_ci / areas
    areas_se_percent = se_areas / areas

    classwise_metrics_df = pd.DataFrame(
        {
            "Class": classes,
            "Total Predicted": counts_pred,
            "Total True": counts_true,
            "Total Positive": total_positive,
            "User's Accuracy": users_accuracy_by_class,
            "Producer's Accuracy": producers_accuracy_by_class,
            "User's Accuracy Proportional": users_accuracy_by_class_norm,
            "Producer's Accuracy Proportional": producers_accuracy_by_class_norm,
            "User's Variance": v_ua,
            "Producer's Variance": v_pa,
            "User's Std Error": se_ua,
            "Producer's Std Error": se_pa,
            "User's 95% CI": ua_ci,
            "Producer's 95% CI": pa_ci,
            "Area": areas,
            "95% Area Error": areas_ci,
            "UC 95% error": uc_ci,
            "Area Std Error": se_areas,
            "Area Std Error %": areas_se_percent,
        }
    )

    overall_metrics_df = pd.DataFrame(
        {
            "Overall Accuracy": oa,
            "Overall Accuracy Variance": v_oa,
            "Overall Accuracy Std Error": se_oa,
            "Overall Accuracy 95% CI": oa_ci,
        }
    )

    return classwise_metrics_df, overall_metrics_df


def benchmark(message="Execution time"):
    """Decorator to measure the execution time of a function."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            print(
                f"{message}: Function '{func.__name__}' executed in {execution_time:.6f} seconds"
            )
            return result

        return wrapper

    return decorator
