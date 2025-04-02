import functools
import time
from typing import List

import geopandas as gpd
import numpy as np
import rasterio as rio
from osgeo import gdal


def get_nodata_value(map_path: str):
    with rio.open(map_path) as src:
        return src.nodata


def get_width_height(map_path: str):
    with rio.open(map_path) as src:
        return src.width, src.height


def get_map_extent(map_path: str):
    with rio.open(map_path) as src:
        return src.bounds


def get_mask_extent(mask_path: str):
    mask = gpd.read_file(mask_path)
    return mask.total_bounds


def get_map_dtype(map_path: str):
    raster = gdal.Open(map_path, gdal.GA_ReadOnly)
    if raster is None:
        raise ValueError(f"Failed to open raster at {map_path}")

    raster_band = raster.GetRasterBand(1)
    if raster_band is None:
        raise ValueError("Raster or mask does not contain a valid band.")

    return raster_band.DataType


def get_mask_spatial_ref(mask_path: str):
    # mask is a vector file
    mask = gpd.read_file(mask_path)
    return mask.crs


def get_map_spatial_ref(map_path: str):
    with rio.open(map_path) as src:
        return src.crs


def are_crs_matching(map_path: str, mask_paths: List[str]):
    map_crs = get_map_spatial_ref(map_path)
    all_mask_crs = get_mask_spatial_ref(mask_paths)
    return all([map_crs == mask_crs for mask_crs in all_mask_crs])


def get_map_resolution(map_path: str):
    with rio.open(map_path) as src:
        return src.res


def benchmark(message="Execution time"):
    """Decorator to measure the execution time of a function."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            print(
                f"{message}: Function '{func.__name__}' executed in {execution_time:.6f} seconds"
            )
            return result

        return wrapper

    return decorator


def get_classes(map_path: str):
    unique_classes = set()
    with rio.open(map_path) as src:
        for _, window in src.block_windows(1):
            data = src.read(1, window=window)
            unique_classes.update(np.unique(data))
    return np.array(sorted(unique_classes))
