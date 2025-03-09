from typing import Dict

import numpy as np
import pandas as pd

from unbiased_area_estimation.storage_manager import StorageManager


class UnbiasedAreaEstimator:
    def __init__(self, results_dir: str):
        self.storage_manager = StorageManager(storage_base_path=results_dir)

    def _compute_unbiased_area_estimates(
        self, predicted, annotated, strata_areas: Dict[int, float]
    ):
        classes = list(strata_areas.keys())
        num_classes = len(classes)
        confusion_matrix = np.zeros((num_classes, num_classes))

        total_area = np.sum(list(strata_areas.values()))
        wh = {class_id: area / total_area for class_id, area in strata_areas.items()}

        # Compute confusion matrix
        for i, true_class in enumerate(classes):
            for j, pred_class in enumerate(classes):
                confusion_matrix[i, j] = np.sum(
                    (annotated == true_class) & (predicted == pred_class)
                )

        counts_pred = np.sum(confusion_matrix, axis=1)  # Total predicted per class
        counts_true = np.sum(confusion_matrix, axis=0)  # Total actual per class
        total_positive = np.diag(confusion_matrix)  # True positives
        users_accuracy_by_class = (
            np.diag(confusion_matrix) / counts_pred
        )  # User's accuracy
        producers_accuracy_by_class = (
            np.diag(confusion_matrix) / counts_true
        )  # Producer's accuracy

        # Ensure wh dictionary matches class indices
        if set(classes) != set(wh.keys()):
            raise ValueError(
                f"Mismatch between confusion matrix classes {set(classes)} and weight keys {set(wh.keys())}. Not yet implemented."
            )

        weights = np.array([wh[class_id] for class_id in classes])
        areas = np.array([strata_areas[class_id] for class_id in classes])
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

        v_ua = (
            users_accuracy_by_class * (1 - users_accuracy_by_class) / (counts_pred - 1)
        )
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
                "Class": np.array(classes),
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
                "Overall Accuracy": [oa],
                "Overall Accuracy Variance": [v_oa],
                "Overall Accuracy Std Error": [se_oa],
                "Overall Accuracy 95% CI": [oa_ci],
            }
        )

        return classwise_metrics_df, overall_metrics_df

    def get_unbiased_area_estimates(self, annotated_column_name):
        regions = self.storage_manager.get_available_regions()

        results = {}

        for region_name in regions:
            sampling_design_df = self.storage_manager.load_sampling_design(
                region_name=region_name
            )
            annotated_samples_df = self.storage_manager.load_annotated_samples(
                region_name=region_name
            )

            predicted = annotated_samples_df["stratum_id"]
            true = annotated_samples_df[annotated_column_name]

            classes = sampling_design_df.index
            areas = sampling_design_df["area"]
            areas_dict = {
                class_id: float(area) for class_id, area in zip(classes, areas)
            }

            (
                classwise_metrics_df,
                overall_metrics_df,
            ) = self._compute_unbiased_area_estimates(
                predicted=predicted, annotated=true, strata_areas=areas_dict
            )

            results[region_name] = {
                "classwise_metrics": classwise_metrics_df,
                "overall_metrics": overall_metrics_df,
            }

        return results
