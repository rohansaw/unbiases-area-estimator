import functools
import multiprocessing as mp
import os
import time
from typing import List

import geopandas as gpd
import numpy as np
import rasterio as rio
from osgeo import gdal
from rasterio.windows import Window
from tqdm import tqdm


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


def process_chunk(args):
    """Process a single chunk of the raster file to find unique values."""
    map_path, window = args
    with rio.open(map_path) as src:
        data = src.read(1, window=window)
        return set(np.unique(data))


def get_unique_classes(map_path, max_full_read_size=1e10, num_workers=None):
    """
    Get unique classes from a raster map efficiently using parallel processing.

    Parameters:
    -----------
    map_path : str
        Path to the raster file
    max_full_read_size : int
        Maximum size (width*height) to read the raster in one go
    num_workers : int, optional
        Number of workers for parallel processing. Defaults to CPU count.

    Returns:
    --------
    np.ndarray
        Sorted array of unique class values
    """
    # Determine number of workers if not specified
    if num_workers is None:
        num_workers = max(1, min(os.cpu_count() - 1, 16))

    with rio.open(map_path) as src:
        # If file is small enough, read it in one go
        if src.width * src.height < max_full_read_size:
            print(f"Reading {map_path} in one go")
            return np.array(sorted(np.unique(src.read(1))))

        # If the file is too large, read it in chunks
        if src.count > 1:
            raise ValueError("Currently only single band inputs supported.")

        block_h, block_w = src.block_shapes[0]

        # Check if the raster is stored in stripes
        is_striped = block_h == 1 or block_w == 1

        if is_striped:
            # For striped data, read along the stripes (typically rows)
            if block_h == 1:  # Horizontal stripes
                chunk_h = 8
                chunk_w = src.width
            else:  # Vertical stripes
                chunk_h = src.height
                chunk_w = 8
        else:
            # For tiled data, we use larger chunks than the native blocks
            # to reduce overhead while keeping memory usage reasonable
            chunk_h = block_h * 8
            chunk_w = block_w * 8

        print(
            f"Reading {map_path} in chunks of {chunk_h}x{chunk_w} using {num_workers} workers"
        )

        # Create windows for parallel processing
        windows = []
        for row_off in range(0, src.height, chunk_h):
            for col_off in range(0, src.width, chunk_w):
                win_width = min(chunk_w, src.width - col_off)
                win_height = min(chunk_h, src.height - row_off)
                windows.append(Window(col_off, row_off, win_width, win_height))

        # Process chunks in parallel
        args = [(map_path, window) for window in windows]
        unique_classes = set()

        # Use progress bar to track processing
        with mp.Pool(num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(process_chunk, args),
                    total=len(args),
                    desc="Processing chunks",
                )
            )

            # Combine results from all chunks
            for chunk_unique in results:
                unique_classes.update(chunk_unique)

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
