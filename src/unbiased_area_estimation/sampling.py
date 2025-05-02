from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import rasterio as rio
from osgeo import gdal, osr

from unbiased_area_estimation.region import Region
from unbiased_area_estimation.sample_allocation import get_allocator


class SamplingStrategy(ABC):
    """
    Abstract base class for all sampling strategies. Defines required methods for creating sampling designs.
    """

    def __init__(self):
        pass

    def _get_coords(self, samples, geo_transform, proj_ref, wgs84):
        """
        Converts pixel coordinates to projected and geographic coordinates.
        Used for exporting results with full spatial context.

        Parameters:
            samples (list): List of tuples (x_px, y_px, stratum_id).
            geo_transform: Affine transform for the raster.
            proj_ref: Raster spatial reference (OSR).
            wgs84: WGS84 spatial reference.

        Returns:
            pd.DataFrame: Sample points with projected and geographic coordinates.
        """

        coords = []
        for loc in samples:
            x_px = loc[0]
            y_px = loc[1]
            stratum_id = loc[2]
            y_m = geo_transform[3] + y_px * geo_transform[5]
            x_m = geo_transform[0] + x_px * geo_transform[1]
            tx = osr.CoordinateTransformation(proj_ref, wgs84)
            (lon, lat, z) = tx.TransformPoint(x_m, y_m)

            entry = {
                "x_m": x_m,
                "y_m": y_m,
                "x_px": x_px,
                "y_px": y_px,
                "LAT": lat,
                "LON": lon,
                "stratum_id": stratum_id,
                "class_id": -1,
            }
            coords.append(entry)

        coords_df = pd.DataFrame(coords)
        return coords_df

    def allocate(
        self,
        strata_weights: Dict[str, float],
        total_n_samples: int,
        allocation_method_name: str,
        detailed_design_df: pd.DataFrame,
    ):
        """
        Allocates sample counts to each stratum using a specified allocation method.

        Returns:
            Dict[str, int]: Number of samples allocated per class/stratum.
        """

        allocator = get_allocator(allocation_method_name)
        sampling_design = allocator.allocate(
            n_samples=total_n_samples,
            weights=strata_weights,
            detailed_design_df=detailed_design_df,
        )
        return sampling_design

    @abstractmethod
    def create_design(
        self,
        region: Region,
        expected_uas: Dict[str, Dict[str, float]],
        target_error: float,
    ):
        pass

    @abstractmethod
    def sample(
        self,
        region: Region,
        sampling_design: Dict[str, int],
        shuffle,
        existing_samples_df: pd.DataFrame,
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_expected_error(
        self, sampling_design: Dict[int, int], detailed_design_df: pd.DataFrame
    ):
        pass

    @abstractmethod
    def _compute_total_num_samples(
        self, region: Region, expected_uas: Dict[int, float], target_error: float
    ) -> Tuple[int, pd.DataFrame]:
        pass


class StratifiedRandomSampling(SamplingStrategy):
    """
    Stratified random sampling where samples are drawn per stratum based on weights and desired error.
    """

    def _compute_total_num_samples(
        self, region: Region, expected_uas: Dict[int, float], target_error: float
    ) -> Tuple[int, pd.DataFrame]:
        region_name = region.name
        print(f"Computing total number of samples for {region_name}...")

        if not region.pixel_counts:
            pixel_counts = region.get_pixel_counts_by_class()
        else:
            pixel_counts = region.pixel_counts

        total_pixels = sum(pixel_counts.values())
        areas = region.get_areas()
        df = pd.DataFrame.from_dict(pixel_counts, orient="index", columns=["count"])
        df["area"] = df.index.map(areas)
        df["ua"] = df.index.map(expected_uas)
        df["wh"] = df["count"] / total_pixels
        df["s"] = np.sqrt(df["ua"] * (1 - df["ua"]))
        df["nInt"] = df["wh"] * df["s"]

        n = int(np.ceil(np.square(df["nInt"].sum() / target_error)))

        return n, df

    def create_design(
        self,
        region: Region,
        expected_uas: Dict[str, Dict[str, float]],
        target_error: float,
        allocation_method_name: str,
    ):
        """
        Creates the sampling design including allocation of samples to strata.
        The sampling design that is returned holds the number of sampls to sample
        from each stratum. The detailed_design_df can be used to better understand
        the sampling design procedure.

        Returns:
            Tuple: (sampling_design: Dict[str, int], detailed_design_df: pd.DataFrame)
        """

        print("Using Statfied random sampling.")

        n_samples, detailed_design_df = self._compute_total_num_samples(
            region=region, expected_uas=expected_uas, target_error=target_error
        )

        weights = detailed_design_df["wh"].to_dict()
        sampling_design = self.allocate(
            strata_weights=weights,
            total_n_samples=n_samples,
            allocation_method_name=allocation_method_name,
            detailed_design_df=None,
        )

        return sampling_design, detailed_design_df

    def sample(
        self,
        region: Region,
        num_samples_per_stratum: Dict[str, int],
        shuffle=True,
        existing_samples_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Draws samples from the map according to the sampling design.
        Each sample has the same probability of being choosen.

        Returns:
            pd.DataFrame: Sample locations with coordinates and class info.
        """

        if existing_samples_df is None:
            samples = {}
        else:
            samples = {
                (row["x_px"], row["y_px"], row["stratum_id"]): True
                for _, row in existing_samples_df.iterrows()
            }
            previous_samples = samples.copy()

        remaining_sample_counter = num_samples_per_stratum.copy()
        shape_x, shape_y = region.get_shape()
        map_path = region.map_path
        mask_path = region.mask_path

        np.random.seed(5)

        # TODO need to find a way to better encapsulate behaviour related to region (map/mask)
        # TODO add an upper bounds of tries!
        # TODO this is not very efficient - might want to optimize this
        with rio.open(map_path) as ds:
            mask_ds = None
            if mask_path:
                mask_ds = rio.open(mask_path)

            try:
                # Iteratively sample: We can not read the complete map into memory.
                # Therefore, we only read a window at our sampled x,y coordinates
                # This will become unefficient for large sampling sizes
                # TODO use e.g reservoir sampling?
                while sum(remaining_sample_counter.values()) > 0:
                    x = np.random.randint(0, shape_x)
                    y = np.random.randint(0, shape_y)

                    if mask_ds:
                        mask_value = mask_ds.read(1, window=((y, y + 1), (x, x + 1)))
                        if mask_value[0, 0] != 1:
                            continue

                    value = ds.read(
                        1, window=((y, y + 1), (x, x + 1))
                    )  # Read a single pixel
                    sample_cls = int(value[0, 0])  # Extract the scalar value
                    if (
                        sample_cls in remaining_sample_counter
                        and remaining_sample_counter[sample_cls] > 0
                    ):
                        if (x, y, sample_cls) in samples:
                            # We already have taken this sample, so we continue
                            # Ensures probability for each sample is the same
                            continue
                        samples[(x, y, sample_cls)] = True
                        remaining_sample_counter[sample_cls] -= 1
            finally:
                if mask_ds:
                    mask_ds.close()

        if existing_samples_df is not None:
            # Remove already existing samples from the samples dict
            # Only process new samples, shuffle and create ids for them
            diff_keys = set(previous_samples.keys()) - set(samples.keys())
            samples = {k: True for k in diff_keys}

        base_ds = gdal.Open(map_path)
        geo_transform = base_ds.GetGeoTransform()
        projection = base_ds.GetProjectionRef()
        proj_ref = osr.SpatialReference()
        proj_ref.ImportFromWkt(projection)
        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)
        if int(gdal.VersionInfo()) >= 3000000:
            wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        samples_df = self._get_coords(samples.keys(), geo_transform, proj_ref, wgs84)

        if shuffle:
            samples_df = samples_df.sample(frac=1)  # shuffle the samples

        # Enumerate (shuffled) samples. If we have an existing dataframe we continue its index
        plot_id_start_idx = (
            0 if existing_samples_df is None else existing_samples_df["PLOTID"].max()
        )
        samples_df["PLOTID"] = np.arange(
            plot_id_start_idx, plot_id_start_idx + samples_df.shape[0]
        )

        base_ds = None
        return samples_df

    def get_expected_error(
        self, sampling_design: Dict[int, int], detailed_design_df: pd.DataFrame
    ):
        """
        Computes the expected standard error for a given sampling design.

        Returns:
            float: Expected standard error.
        """

        new_n_total = sum(sampling_design.values())
        expected_error = detailed_design_df["nInt"].sum() / np.sqrt(new_n_total)
        return expected_error


class SimpleRandomSampling(SamplingStrategy):
    def create_design(
        self,
        region: Region,
        expected_uas: Dict[str, Dict[str, float]],
        target_error: float,
        allocation_method_name: str,
    ):
        raise NotImplementedError()

    def sample(self, region: Region, sampling_design: Dict[str, int], shuffle=True):
        raise NotImplementedError()

    def _compute_total_num_samples(
        self, region: Region, expected_uas: Dict[int, float], target_error: float
    ) -> Tuple[int, pd.DataFrame]:
        pass

    def get_expected_error(
        self, sampling_design: Dict[str, int], detailed_design_df: pd.DataFrame
    ):
        pass


class TwoStageRandomSampling(SamplingStrategy):
    def create_design(
        self,
        region: Region,
        expected_uas: Dict[str, Dict[str, float]],
        target_error: float,
        allocation_method_name: str,
    ):
        raise NotImplementedError()

    def sample(self, region: Region, sampling_design: Dict[str, int], shuffle=True):
        raise NotImplementedError()

    def _compute_total_num_samples(
        self, region: Region, expected_uas: Dict[int, float], target_error: float
    ) -> Tuple[int, pd.DataFrame]:
        pass

    def get_expected_error(
        self, sampling_design: Dict[str, int], detailed_design_df: pd.DataFrame
    ):
        pass


def create_sampler(sampling_method_name: str) -> SamplingStrategy:
    """
    Factory function to create a sampling strategy instance based on its name.

    Parameters:
        sampling_method_name (str): One of "random", "stratified", or "twostage".

    Returns:
        SamplingStrategy: Corresponding strategy object.

    Raises:
        IndexError: If the sampling method name is invalid.
    """

    samplers = {
        "random": SimpleRandomSampling,
        "stratified": StratifiedRandomSampling,
        "twostage": TwoStageRandomSampling,
    }

    sampling_method_name = sampling_method_name.lower()
    if sampling_method_name not in samplers:
        raise IndexError(f"Invalid sampler_name provided. Use one of {samplers.keys()}")

    return samplers[sampling_method_name]()
