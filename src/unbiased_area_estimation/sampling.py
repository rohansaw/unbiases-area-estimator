from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import rasterio as rio
from osgeo import gdal, osr

from unbiased_area_estimation.region import Region
from unbiased_area_estimation.sample_allocation import get_allocator


class SamplingStrategy(ABC):
    def __init__(self, allocation_method_name: str):
        self.allocator = get_allocator(allocation_method_name)

    def _compute_total_num_samples(
        self, region: Region, expected_uas: Dict[int, float], target_error: float
    ) -> Tuple[int, pd.DataFrame]:
        region_name = region.name
        print(f"Computing total number of samples for {region_name}...")

        pixel_counts = region.get_pixel_counts_by_class()
        total_pixels = sum(pixel_counts.values())
        df = pd.DataFrame.from_dict(pixel_counts, orient="index", columns=["count"])
        df["ua"] = df.index.map(expected_uas)
        df["wh"] = df["count"] / total_pixels
        df["s"] = np.sqrt(df["ua"] * (1 - df["ua"]))
        df["nInt"] = df["wh"] * df["s"]

        n = round(np.square(df["nInt"].sum() / target_error))

        return n, df

    def _get_coords(self, samples, geo_transform, proj_ref, wgs84):
        # ToDo better naming of arguments
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

    @abstractmethod
    def create_design(
        self,
        region: Region,
        expected_uas: Dict[str, Dict[str, float]],
        target_error: float,
    ):
        pass

    @abstractmethod
    def sample(self, region: Region, sampling_design: Dict[str, int], shuffle):
        pass


class StratifiedRandomSampling(SamplingStrategy):
    def create_design(
        self,
        region: Region,
        expected_uas: Dict[str, Dict[str, float]],
        target_error: float,
    ):
        print("Using Statfied random sampling.")

        n_samples, detailed_design_df = self._compute_total_num_samples(
            region=region, expected_uas=expected_uas, target_error=target_error
        )
        weights = detailed_design_df["wh"].to_dict()
        sampling_design = self.allocator.allocate(n_samples=n_samples, weights=weights)

        return sampling_design, detailed_design_df

    def sample(
        self, region: Region, num_samples_per_stratum: Dict[str, int], shuffle=True
    ):
        samples = {}
        remaining_sample_counter = num_samples_per_stratum.copy()
        shape_x, shape_y = region.get_raster_shape()
        map_path = region.raster_path

        np.random.seed(5)

        # TODO add an upper bounds of tries?
        while sum(remaining_sample_counter.values()) > 0:
            x = np.random.randint(0, shape_x)
            y = np.random.randint(0, shape_y)

            with rio.open(map_path) as ds:
                value = ds.read(
                    1, window=((y, y + 1), (x, x + 1))
                )  # Read a single pixel
                sample_cls = int(value[0, 0])  # Extract the scalar value
                if (
                    sample_cls in remaining_sample_counter
                    and remaining_sample_counter[sample_cls] > 0
                ):
                    if (x, y, sample_cls) in samples:
                        # This would break the assumption for random sampling.
                        # We might want to think about using reservoir sampling.
                        raise Exception(
                            "The same sample was picked twice, not yet handling this case."
                        )
                    samples[(x, y, sample_cls)] = True
                    remaining_sample_counter[sample_cls] -= 1

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

        samples_df["PLOTID"] = np.arange(0, samples_df.shape[0])
        base_ds = None
        return samples_df


class SimpleRandomSampling(SamplingStrategy):
    def create_design(
        self,
        region: Region,
        expected_uas: Dict[str, Dict[str, float]],
        target_error: float,
    ):
        raise NotImplementedError()

    def sample(self, region: Region, sampling_design: Dict[str, int], shuffle=True):
        raise NotImplementedError()


class TwoStageRandomSampling(SamplingStrategy):
    def create_design(
        self,
        region: Region,
        expected_uas: Dict[str, Dict[str, float]],
        target_error: float,
    ):
        raise NotImplementedError()

    def sample(self, region: Region, sampling_design: Dict[str, int], shuffle=True):
        raise NotImplementedError()


def create_sampler(
    sampling_method_name: str, allocation_method_name: str
) -> SamplingStrategy:
    samplers = {
        "random": SimpleRandomSampling,
        "stratified": StratifiedRandomSampling,
        "twostage": TwoStageRandomSampling,
    }

    if sampling_method_name not in samplers:
        raise IndexError(f"Invalid sampler_name provided. Use one of {samplers.keys()}")

    return samplers[sampling_method_name](allocation_method_name=allocation_method_name)
