from typing import Dict, List

from unbiased_area_estimation.config import Config
from unbiased_area_estimation.preprocess import Preprocessor
from unbiased_area_estimation.region import Region
from unbiased_area_estimation.sampling import create_sampler
from unbiased_area_estimation.storage_manager import StorageManager


class SamplingDesignPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.storage_manager = StorageManager(
            storage_base_path=config.output_path, use_cached=config.use_cached
        )
        self.sampler = create_sampler(
            sampling_method_name=config.sampling.sampling_method,
            allocation_method_name=config.sampling.allocation_method,
        )
        self.preprocessor = Preprocessor(storage_manager=self.storage_manager)

    def _preprocess(
        self,
        map_path: str,
        mask_paths: List[str],
        target_spatial_ref: str = None,
        class_merge_map: Dict[int, int] = None,
    ) -> List[Region]:
        masked_map_paths = self.preprocessor.preprocess(
            map_path=map_path,
            mask_paths=mask_paths,
            target_spatial_ref=target_spatial_ref,
            class_merge_map=class_merge_map,
        )

        regions = [
            Region(name=name, raster_path=raster_path)
            for name, raster_path in masked_map_paths.items()
        ]
        return regions

    def _sample_and_save(self, regions: List[Region]) -> None:
        for region in regions:
            sampling_config = self.config.sampling
            sampling_design, detailed_design_df = self.sampler.create_design(
                region,
                expected_uas=sampling_config.expected_uas,
                target_error=sampling_config.target_error,
            )
            areas = region.get_areas()
            print(areas)
            self.storage_manager.save_sampling_design(region.name, detailed_design_df)
            sample_set = self.sampler.sample(region, sampling_design)
            self.storage_manager.save_samples(region.name, sample_set)

    def run(self) -> None:
        map_path = self.config.map_path
        mask_paths = self.config.mask_paths
        target_spatial_ref = self.config.target_spatial_ref
        class_merge_map = self.config.class_merge_map

        regions = self._preprocess(
            map_path=map_path,
            mask_paths=mask_paths,
            target_spatial_ref=target_spatial_ref,
            class_merge_map=class_merge_map,
        )
        self._sample_and_save(regions)
