import glob
import os
import os.path as op
from typing import Dict, List, Tuple

import mmh3
import pandas as pd


class StorageManager:
    def __init__(self, storage_base_path: str, use_cached=True):
        self.storage_base_path = storage_base_path
        self.preprocessed_data_dir = op.join(storage_base_path, "preprocessed_data")
        self.use_cached = use_cached

        os.makedirs(self.preprocessed_data_dir, exist_ok=True)

    def _hash_file_contents(self, file_path) -> str:
        with open(file_path, "rb") as f:
            file_contents = f.read()
            return str(mmh3.hash_bytes(file_contents, 0xBEFFE).hex())

    def _hash_str(self, in_string) -> str:
        return str(mmh3.hash(in_string) & 0xFFFFFFFF)

    def _hash_dict(self, in_dict) -> str:
        return str(mmh3.hash(str(in_dict)) & 0xFFFFFFFF)

    def get_reprojected_map_path(self, map_path: str, target_spatial_ref: str) -> str:
        map_hash = self._hash_file_contents(map_path)
        tsr_hash = self._hash_str(target_spatial_ref)
        return op.join(
            self.preprocessed_data_dir, f"map_{map_hash}_{tsr_hash}_reprojected.tif"
        )

    def get_merged_classes_map_path(
        self, map_path: list, class_merge_dict: Dict[int, int]
    ) -> str:
        map_hash = self._hash_file_contents(map_path)
        class_map_hash = self._hash_dict(class_merge_dict)
        return op.join(
            self.preprocessed_data_dir,
            f"map_{map_hash}_{class_map_hash}_reclassified.tif",
        )

    def get_rasterized_mask_path(
        self, mask_path: str, resolution: Tuple[float, float]
    ) -> str:
        mask_hash = self._hash_file_contents(mask_path)
        resolution_hash = self._hash_str(str(resolution))
        return op.join(
            self.preprocessed_data_dir,
            f"mask_{mask_hash}_{resolution_hash}_rasterized.tif",
        )

    def get_reprojected_mask_path(self, mask_path: str, target_spatial_ref: str) -> str:
        mask_hash = self._hash_file_contents(mask_path)
        tsr_hash = self._hash_str(target_spatial_ref)
        ext = op.splitext(mask_path)[1]
        return op.join(
            self.preprocessed_data_dir, f"mask_{mask_hash}_{tsr_hash}_reprojected{ext}"
        )

    def get_masked_map_path(self, map_path: str, mask_path: str) -> str:
        map_hash = self._hash_file_contents(map_path)
        mask_hash = self._hash_file_contents(mask_path)
        return op.join(
            self.preprocessed_data_dir, f"map_{map_hash}_{mask_hash}_masked.tif"
        )

    def exists(self, path: str) -> bool:
        if not self.use_cached:
            return False

        return op.exists(path)

    def _get_design_fname(self, region_name: str) -> str:
        return f"{region_name}_design.csv"

    def _get_samples_fname(self, region_name: str) -> str:
        return f"{region_name}_samples.csv"

    def save_samples(self, region_name: str, samples: pd.DataFrame) -> None:
        samples_file_name = self._get_samples_fname(region_name)
        out_path = op.join(self.storage_base_path, samples_file_name)
        samples.to_csv(out_path, index=False)

    def save_sampling_design(
        self, region_name: str, sampling_design: pd.DataFrame
    ) -> None:
        print("Saving sampling design...")
        design_file_name = self._get_design_fname(region_name)
        out_path = op.join(self.storage_base_path, design_file_name)
        sampling_design.to_csv(out_path, index=True, index_label="class_id")

    def load_sampling_design(self, region_name: str) -> pd.DataFrame:
        design_file_name = self._get_design_fname(region_name)
        sampling_design_path = op.join(self.storage_base_path, design_file_name)
        sampling_design = pd.read_csv(sampling_design_path, index_col="class_id")
        return sampling_design

    def load_annotated_samples(self, region_name: str) -> pd.DataFrame:
        samples_file_name = self._get_samples_fname(region_name)
        samples_path = op.join(self.storage_base_path, samples_file_name)
        samples_df = pd.read_csv(samples_path)
        return samples_df

    def get_available_regions(self) -> List[str]:
        design_files = glob.glob(
            op.join(self.storage_base_path, self._get_design_fname("*"))
        )
        region_names = ["_".join(op.basename(f).split("_")[:-1]) for f in design_files]
        return region_names
