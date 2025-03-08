import json
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class SamplingConfig:
    sampling_method: str
    allocation_method: str
    expected_uas: Dict[int, float]
    target_error: float


@dataclass
class Config:
    map_path: str  # Raster map
    mask_paths: List[str]  # Vector Masks
    target_spatial_ref: str
    class_merge_map: Dict[int, int]
    class_names: Dict[int, str]
    sampling: SamplingConfig
    output_path: str
    use_cached: bool

    @staticmethod
    def load_from_json(json_path: str) -> "Config":
        """Loads configuration from a JSON file."""
        with open(json_path, "r") as file:
            data = json.load(file)

        # Convert nested dict to SamplingConfig object
        data["sampling"] = SamplingConfig(**data["sampling"])

        data["sampling"].expected_uas = {
            int(k): v for k, v in data["sampling"].expected_uas.items()
        }
        data["class_merge_map"] = {
            int(k): v for k, v in data["class_merge_map"].items()
        }
        data["class_names"] = {int(k): v for k, v in data["class_names"].items()}

        return Config(**data)
