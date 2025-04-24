from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import pandas as pd


class AllocationStrategy(ABC):
    """
    Abstract base class for defining a sample allocation strategy across strata/classes.
    """

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
        """
        Allocate a number of samples to each class based on the implemented strategy.

        Parameters:
            n_samples (int): Total number of samples to allocate.
            weights (Dict[str, float]): Proportional weights for each class.
            detailed_design_df (pd.DataFrame): Additional details per class, e.g. standard deviation and counts.

        Returns:
            Dict[str, int]: Number of samples allocated per class.
        """
        pass

    @abstractmethod
    def get_expected_error(self):
        pass


class ProportionalAllocation(AllocationStrategy):
    """
    Allocates samples proportionally based on the weight of each class.
    """

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
    """
    Implements Neyman allocation which minimizes variance by considering both stratum size and variability.
    """

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
    """
    Factory method to instantiate the appropriate allocation strategy.

    Parameters:
        allocator_name (str): Name of the desired allocation strategy ('proportional' or 'neyman').

    Returns:
        AllocationStrategy: Instance of the corresponding allocation strategy class.

    Raises:
        IndexError: If an invalid allocator name is provided.
    """

    allocator_name = allocator_name.lower()
    allocators = {"proportional": ProportionalAllocation, "neyman": NeymanAllocation}

    if allocator_name not in allocators:
        raise IndexError(f"{allocator_name} invalid. Use one of {allocators.keys()}.")

    return allocators[allocator_name]()
