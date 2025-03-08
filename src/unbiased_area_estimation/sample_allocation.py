from abc import ABC, abstractmethod
from typing import Dict


class AllocationStrategy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def allocate(
        self,
    ):
        pass

    @abstractmethod
    def get_expected_error(self):
        pass


class ProportionalAllocation(AllocationStrategy):
    def __init__(self):
        pass

    def allocate(self, n_samples: int, weights: Dict[str, float]):
        print("Allocating samples to classes with Proportional Allocation.")
        sampling_design = {k: v * n_samples for k, v in weights.items()}
        return sampling_design

    def get_expected_error(self):
        pass


class NeymanAllocation(AllocationStrategy):
    def __init__(self):
        pass

    def allocate(
        self,
    ):
        pass

    def get_expected_error(self):
        pass


def get_allocator(allocator_name: str) -> AllocationStrategy:
    allocator_name = allocator_name.lower()
    allocators = {"proportional": ProportionalAllocation, "neyman": NeymanAllocation}

    if allocator_name not in allocators:
        raise IndexError(f"{allocator_name} invalid. Use one of {allocators.keys()}.")

    return allocators[allocator_name]()
