from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import pandas as pd


class AllocationStrategy(ABC):
    def __init__(self):
        pass

    # TODO refactor to not use weights as weights are included in detailed_design_df
    @abstractmethod
    def allocate(
        self,
        n_samples: int,
        weights: Dict[str, float],
        detailed_design_df: pd.DataFrame,
    ):
        pass

    @abstractmethod
    def get_expected_error(self):
        pass


class ProportionalAllocation(AllocationStrategy):
    def __init__(self):
        pass

    def allocate(
        self,
        n_samples: int,
        weights: Dict[str, float],
        detailed_design_df: pd.DataFrame = None,
    ):
        print("Allocating samples to classes with Proportional Allocation.")
        # Rounding with ceil to ensure that target error is at least met.
        sampling_design = {k: int(np.ceil(v * n_samples)) for k, v in weights.items()}
        return sampling_design

    def get_expected_error(self):
        pass


class NeymanAllocation(AllocationStrategy):
    def __init__(self):
        pass

    def allocate(
        self,
        n_samples: int,
        weights: Dict[str, float],
        detailed_design_df: pd.DataFrame,
    ):
        print("Allocating samples to classes with Neyman Allocation.")
        sampling_design = {}
        denominator = np.sum(detailed_design_df["s"] * detailed_design_df["count"])
        for k in weights.keys():
            N_h = detailed_design_df.loc[k, "count"]
            S_h = detailed_design_df.loc[k, "s"]
            n_samples_strata = n_samples * (N_h * S_h / denominator)

            # Rounding with ceil to ensure that target error is at least met.
            sampling_design[k] = int(np.ceil(n_samples_strata))
        return sampling_design

    def get_expected_error(self):
        pass


def get_allocator(allocator_name: str) -> AllocationStrategy:
    allocator_name = allocator_name.lower()
    allocators = {"proportional": ProportionalAllocation, "neyman": NeymanAllocation}

    if allocator_name not in allocators:
        raise IndexError(f"{allocator_name} invalid. Use one of {allocators.keys()}.")

    return allocators[allocator_name]()
