import numpy as np
import pandas as pd


def create_sample_allocator(sampling_params):
    if sampling_params["method"] == "proportional":
        return ProportionalAllocator(
            expected_uas=sampling_params["expected_uas"],
            target_error=sampling_params["target_error"],
        )
    elif sampling_params["method"] == "neyman":
        return NeymanAllocator(
            expected_uas=sampling_params["expected_uas"],
            target_error=sampling_params["target_error"],
        )
    else:
        raise ValueError("Invalid sampling method specified in config file.")


class SampleAllocator:
    def __init__(self, expected_uas: dict[str, float], target_error: float):
        self.expected_uas = expected_uas
        self.target_error = target_error

    def allocate(
        self,
        pixel_counts: dict[str, int],
        areas: dict[str, float],
        output_path: str = None,
    ):
        pass


class ProportionalAllocator(SampleAllocator):
    def __init__(self, expected_uas: dict[str, float], target_error: float):
        super().__init__(expected_uas, target_error)

    def allocate(
        self,
        pixel_counts: dict[str, int],
        areas: dict[str, float],
        output_path: str = None,
    ):
        print("Using Proportional Allocator to allocate number of samples to strata.")
        # TODO deal with zero counts, currently they are not being considered

        total_pixels = sum(pixel_counts.values())
        df = pd.DataFrame.from_dict(pixel_counts, orient="index", columns=["count"])
        df["ua"] = df.index.map(self.expected_uas)
        df["wh"] = df["count"] / total_pixels
        df["s"] = np.sqrt(df["ua"] * (1 - df["ua"]))
        df["nInt"] = df["wh"] * df["s"]

        n = round(np.square(df["nInt"].sum() / self.target_error))
        df["nh"] = round(df["wh"] * n)
        df["sampling_rate"] = df["nh"] / df["count"]

        df["area"] = df.index.map(areas)

        if output_path:
            df.to_csv(output_path, index=True)

        return df["nh"].to_dict()

    def compute_expected_error(self, sample_assignment: dict[str, int], nInt, wh):
        total_pixels = sum(sample_assignment.values())
        df = pd.DataFrame.from_dict(
            sample_assignment, orient="index", columns=["count"]
        )

        # Map expected UA scores
        df["ua"] = df.index.map(self.expected_uas)

        # Compute weights and standard deviation
        df["wh"] = df["count"] / total_pixels  # TODO use original wh
        df["s"] = np.sqrt(df["ua"] * (1 - df["ua"]))

        # Compute the target error using the formula
        numerator = (df["wh"] * df["s"]).sum()
        target_error = numerator / np.sqrt(nInt)

        return target_error


class NeymanAllocator(SampleAllocator):
    def __init__(self, expected_uas: dict[str, float], target_error: float):
        super().__init__(expected_uas, target_error)

    def allocate(self, pixel_counts: dict[str, int], output_path: str = None):
        pass

    def compute_expected_error(
        self,
        sample_assignment: dict[str, int],
    ):
        pass
