import functools
import time
from typing import List

import geopandas as gpd
import numpy as np
import rasterio as rio
from osgeo import gdal
from rasterio.windows import Window


def get_nodata_value(map_path: str):
    """
    Retrieves the 'nodata' value from a raster file.

    Parameters:
        map_path (str): Path to the raster file.

    Returns:
        float or int: The nodata value used in the raster.
    """

    with rio.open(map_path) as src:
        return src.nodata


def get_width_height(map_path: str):
    """
    Gets the dimensions (width, height) of the raster.

    Parameters:
        map_path (str): Path to the raster file.

    Returns:
        Tuple[int, int]: Width and height in pixels.
    """

    with rio.open(map_path) as src:
        return src.width, src.height


def get_map_extent(map_path: str):
    """
    Retrieves the spatial bounding box of a raster.

    Parameters:
        map_path (str): Path to the raster file.

    Returns:
        rasterio.coords.BoundingBox: Bounding box of the raster.
    """

    with rio.open(map_path) as src:
        return src.bounds


def get_mask_extent(mask_path: str):
    """
    Computes the bounding box of a vector mask.

    Parameters:
        mask_path (str): Path to the vector file (e.g., shapefile).

    Returns:
        np.ndarray: Array of [minx, miny, maxx, maxy] coordinates.
    """

    mask = gpd.read_file(mask_path)
    return mask.total_bounds


def get_map_dtype(map_path: str):
    """
    Determines the data type of the first band in a raster.

    Parameters:
        map_path (str): Path to the raster file.

    Returns:
        int: GDAL data type enum.
    """

    raster = gdal.Open(map_path, gdal.GA_ReadOnly)
    if raster is None:
        raise ValueError(f"Failed to open raster at {map_path}")

    raster_band = raster.GetRasterBand(1)
    if raster_band is None:
        raise ValueError("Raster or mask does not contain a valid band.")

    return raster_band.DataType


def get_mask_spatial_ref(mask_path: str):
    """
    Retrieves the coordinate reference system (CRS) of a vector mask.

    Parameters:
        mask_path (str): Path to the vector file.

    Returns:
        CRS: CRS object of the mask.
    """

    mask = gpd.read_file(mask_path)
    return mask.crs


def get_map_spatial_ref(map_path: str):
    """
    Retrieves the CRS of a raster.

    Parameters:
        map_path (str): Path to the raster file.

    Returns:
        CRS: CRS object of the raster.
    """

    with rio.open(map_path) as src:
        return src.crs


def are_crs_matching(map_path: str, mask_paths: List[str]):
    """
    Verifies that the CRS of a raster matches that of all masks.

    Parameters:
        map_path (str): Path to the raster.
        mask_paths (List[str]): List of vector mask paths.

    Returns:
        bool: True if all CRSs match, False otherwise.
    """

    map_crs = get_map_spatial_ref(map_path)
    all_mask_crs = get_mask_spatial_ref(mask_paths)
    return all([map_crs == mask_crs for mask_crs in all_mask_crs])


def get_map_resolution(map_path: str):
    """
    Retrieves the pixel resolution (x, y) of a raster.

    Parameters:
        map_path (str): Path to the raster.

    Returns:
        Tuple[float, float]: Pixel width and height in map units.
    """

    with rio.open(map_path) as src:
        return src.res


def get_classes(map_path: str, block_multiplier=(4, 4), max_full_read_size=1e9):
    """
    Retrieves all unique class values from a raster using either full read or block-wise strategy.
    To avoid out of memory, set the max_full_read_size lower than you available memory.

    Parameters:
        map_path (str): Path to the raster.
        block_multiplier (tuple): Multiplier for block size when reading in chunks.
        max_full_read_size (float): Threshold to decide between full read and chunked read.

    Returns:
        np.ndarray: Sorted array of unique class values.
    """

    unique_classes = set()
    with rio.open(map_path) as src:
        # Check if the file is too large to read in one go
        if src.width * src.height < max_full_read_size:
            print(f"Reading {map_path} in one go")
            unique_classes = np.unique(src.read(1))
            return np.array(sorted(unique_classes))

        # If the file is too large, read it in chunks
        if src.count > 1:
            raise ValueError("Currently only single band inputs supported.")

        block_shape = src.block_shapes[0]  # (rows, cols)
        block_h, block_w = block_shape

        if (
            block_h * block_w * block_multiplier[0] * block_multiplier[1]
            < max_full_read_size
        ):
            chunk_h = min(block_h * block_multiplier[0], src.height)
            chunk_w = min(block_w * block_multiplier[1], src.width)
        else:
            chunk_h = block_h
            chunk_w = block_w

        print(f"Reading {map_path} in chunks of {chunk_h}x{chunk_w}")
        for row_off in range(0, src.height, chunk_h):
            for col_off in range(0, src.width, chunk_w):
                win_width = min(chunk_w, src.width - col_off)
                win_height = min(chunk_h, src.height - row_off)
                window = Window(col_off, row_off, win_width, win_height)

                data = src.read(1, window=window)
                unique_classes.update(np.unique(data))

    return np.array(sorted(unique_classes))


def set_nodata_value(map_path, nodata_value, out_path, overwrite=False):
    with rio.open(map_path) as src:
        profile = src.profile.copy()
        data = src.read()

    profile.update(nodata=nodata_value, dtype=profile["dtype"])

    with rio.open(out_path, "w", **profile) as dst:
        dst.write(data)


def benchmark(message="Execution time"):
    """
    Dev-Util: Decorator to log the execution time of a function.

    Parameters:
        message (str): Custom message to prefix the timing output.

    Returns:
        Callable: Wrapped function with benchmarking.
    """

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
