from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from unbiased_area_estimation.config import Config
from unbiased_area_estimation.preprocess import Preprocessor
from unbiased_area_estimation.region import Region
from unbiased_area_estimation.sampling import create_sampler
from unbiased_area_estimation.storage_manager import StorageManager


class SamplingDesignPipeline:
    def __init__(
        self,
        output_path: str,
        use_cached: bool,
        sampling_method: str,
        cache_path: str = None,
    ):
        self.storage_manager = StorageManager(
            storage_base_path=output_path, cache_path=cache_path, use_cached=use_cached
        )
        self.sampler = create_sampler(sampling_method_name=sampling_method)
        self.preprocessor = Preprocessor(storage_manager=self.storage_manager)

    def preprocess(
        self,
        map_path: str,
        mask_paths: List[str],
        target_spatial_ref: str = None,
        class_merge_map: Dict[int, int] = None,
        nodata_value: int = None,
    ) -> List[Region]:
        print("Preprocessing...")

        map_path_preproc, masks = self.preprocessor.preprocess(
            map_path=map_path,
            mask_paths=mask_paths,
            target_spatial_ref=target_spatial_ref,
            class_merge_map=class_merge_map,
            nodata_value=nodata_value,
        )

        if len(masks) == 0:
            return [
                Region(
                    name=Path(map_path).stem,
                    map_path=map_path_preproc,
                    mask_path=None,
                    mask_extent=None,
                )
            ]

        regions = [
            Region(
                name=name,
                map_path=map_path_preproc,
                mask_path=mask["path"],
                mask_extent=mask["extent"],
            )
            for name, mask in masks.items()
        ]
        return regions

    def create_and_save_sampling_designs(
        self,
        regions: List[Region],
        expected_uas: Dict[int, float],
        target_error: float,
        allocation_method_name: str,
    ) -> Tuple[Dict[str, Dict[int, int]], Dict[str, pd.DataFrame]]:
        print("Creating sampling designs...")

        sampling_designs = {}
        sampling_design_details = {}

        for region in regions:
            sampling_design, detailed_design_df = self.sampler.create_design(
                region=region,
                expected_uas=expected_uas,
                target_error=target_error,
                allocation_method_name=allocation_method_name,
            )
            self.storage_manager.save_sampling_design_details(
                region.name, detailed_design_df
            )
            sampling_designs[region.name] = sampling_design
            sampling_design_details[region.name] = detailed_design_df

        return sampling_designs, sampling_design_details

    def sample_and_save(
        self,
        regions: List[Region],
        sampling_designs: Dict[str, Dict[int, int]],
        merge_regions=True,
    ) -> Dict[str, pd.DataFrame]:
        print("Sampling...")

        sample_sets_merged = None

        sample_sets = {}
        for region in regions:
            sampling_design = sampling_designs[region.name]
            sample_set = self.sampler.sample(region, sampling_design)
            sample_set["RegionName"] = region.name
            if merge_regions:
                if sample_sets_merged is None:
                    sample_sets_merged = sample_set
                else:
                    sample_sets_merged = pd.concat([sample_sets_merged, sample_set])
            else:
                self.storage_manager.save_samples(region.name, sample_set)
            sample_sets[region.name] = sample_set

        if merge_regions:
            sample_sets_merged = sample_sets_merged.sample(frac=1)
            sample_sets_merged["PLOTID"] = np.arange(0, sample_sets_merged.shape[0])
            self.storage_manager.save_samples("all_regions", sample_sets_merged)

        return sample_sets

    def get_expected_target_errors(
        self, region_names: List[str], sampling_designs: Dict[str, Dict[int, int]]
    ) -> Dict[str, float]:
        print("Calculating expected target errors...")

        expected_target_errors = {}
        for region_name in region_names:
            new_sampling_design = sampling_designs[region_name]
            detailed_original_sampling_design = (
                self.storage_manager.load_sampling_design_details(
                    region_name=region_name
                )
            )
            expected_target_error = self.sampler.get_expected_error(
                sampling_design=new_sampling_design,
                detailed_design_df=detailed_original_sampling_design,
            )
            expected_target_errors[region_name] = expected_target_error

        return expected_target_errors

    def run(self, config: Config) -> None:
        map_path = config.map_path
        mask_paths = config.mask_paths
        target_spatial_ref = config.target_spatial_ref
        class_merge_map = config.class_merge_map
        expected_uas = config.sampling.expected_uas
        target_error = config.sampling.target_error
        allocation_method_name = config.sampling.allocation_method
        nodata_value = config.nodata_value

        regions = self.preprocess(
            map_path=map_path,
            mask_paths=mask_paths,
            target_spatial_ref=target_spatial_ref,
            class_merge_map=class_merge_map,
            nodata_value=nodata_value,
        )

        (
            sampling_designs,
            sampling_design_details,
        ) = self.create_and_save_sampling_designs(
            regions=regions,
            expected_uas=expected_uas,
            target_error=target_error,
            allocation_method_name=allocation_method_name,
        )
        self.sample_and_save(regions=regions, sampling_designs=sampling_designs)

    def generate_additional_samples(
        self, region: Region, config: Config, num_samples_per_region: Dict[str, int]
    ):
        sampling_design_details = self.storage_manager.load_sampling_design_details(
            region.name
        )
        current_samples = self.storage_manager.load_samples(region_name=region.name)
        num_samples = num_samples_per_region[region.name]

        strata_weights = sampling_design_details["wh"].to_dict()
        allocation_method_name = config.sampling.allocation_method
        new_sampling_design = self.sampler.allocate(
            strata_weights=strata_weights,
            total_n_samples=num_samples,
            allocation_method_name=allocation_method_name,
            detailed_design_df=sampling_design_details,
        )
        new_samples = self.sampler.sample(
            region=region,
            sampling_design=new_sampling_design,
            shuffle=True,
            existing_samples_df=current_samples,
        )

        # Join with current samples
        merged_samples = pd.concat(current_samples, new_samples)
        return merged_samples

    def add_samples_and_save(
        self,
        regions: list[Region],
        config: Config,
        num_samples_per_region: Dict[str, int],
    ):
        results = {}
        for region in regions:
            updated_samples = self.generate_additional_samples(
                region=region,
                config=config,
                num_samples_per_region=num_samples_per_region,
            )
            results[region.name] = updated_samples

        return results
