import concurrent.futures
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import rasterio as rio
from rasterio.windows import Window
from tqdm import tqdm

from unbiased_area_estimation.utils import (
    benchmark,
    get_map_resolution,
    get_width_height,
)


class Region:
    def __init__(
        self, name: str, map_path: str, mask_path: str = None, mask_extent: List = None
    ):
        """
        Initialize a Region object that represents a geographical area with an associated raster map.

        Parameters:
            name (str): Name of the region.
            map_path (str): File path to the raster map (GeoTIFF or similar).
            mask_path (str, optional): File path to a binary mask raster to constrain analysis.
            mask_extent (List, optional): Extent info for validating mask application.
        """

        self.name = name
        self.map_path = map_path
        self.mask_path = mask_path
        self.mask_extent = mask_extent
        self.pixel_counts = None

    def count_pixels_chunk(self, args: Tuple) -> Counter:
        """
        Process a single chunk of the raster file to count pixels by class.

        Parameters:
        -----------
        args : Tuple
            Contains window, use_mask, and nodata_value

        Returns:
        --------
        Counter
            Counter object with class values as keys and pixel counts as values
        """
        window, use_mask, nodata_value = args

        try:
            with rio.open(self.map_path) as src:
                raster_chunk = src.read(1, window=window)

                if use_mask:
                    with rio.open(self.mask_path) as mask_src:
                        mask_chunk = mask_src.read(1, window=window)
                        raster_chunk = raster_chunk[mask_chunk == 1]

                # Filter out nodata values
                if nodata_value is not None:
                    raster_chunk = raster_chunk[raster_chunk != nodata_value]

                # Count unique values
                return Counter(raster_chunk.flatten())

        except Exception as e:
            print(f"Error processing chunk at {window}: {e}")
            return Counter()

    @benchmark("get_pixel_counts_by_class")
    def get_pixel_counts_by_class(
        self, num_workers: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Computes the number of pixels per unique class value in the raster map, optionally within a mask.

        Returns:
            Dict[str, int]: Dictionary mapping class values to pixel counts.

        Raises:
            ValueError: If raster has unsupported data types or if mask metadata doesn't match the map.
        """

        if num_workers is None:
            num_workers = max(1, os.cpu_count() - 1)

        with rio.open(self.map_path) as src:
            nodata_value = src.nodata
            dtype = src.dtypes[0]

            # Ensure only int-based rasters are processed
            if dtype not in ["uint8", "uint16", "int16", "uint32", "int32"]:
                raise ValueError(
                    "Sorry, only handling integer-based values for the moment. Please convert to int first."
                )

            width, height = src.width, src.height
            block_h, block_w = src.block_shapes[0]

        use_mask = self.mask_path is not None and self.mask_extent is not None
        if use_mask:
            with rio.open(self.mask_path) as mask_src:
                if mask_src.width != width or mask_src.height != height:
                    raise ValueError("Mask and raster do not have the same extent.")
                if mask_src.res != src.res:
                    raise ValueError("Mask and raster do not have the same resolution.")
                if mask_src.crs != src.crs:
                    raise ValueError("Mask and raster do not have the same CRS.")

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

        print(f"Reading pixel counts with chunk size: {chunk_h}x{chunk_w}")

        windows = []
        for row_off in range(0, height, chunk_h):
            for col_off in range(0, width, chunk_w):
                win_width = min(chunk_w, width - col_off)
                win_height = min(chunk_h, height - row_off)
                windows.append(Window(col_off, row_off, win_width, win_height))

        # Prepare arguments for each worker
        args = [(window, use_mask, nodata_value) for window in windows]

        combined_counts = Counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.count_pixels_chunk, arg) for arg in args]

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Processing chunks",
            ):
                chunk_counts = future.result()
                combined_counts.update(chunk_counts)

        # Convert to dictionary of integers
        pixel_counts = {int(k): int(v) for k, v in combined_counts.items()}
        self.pixel_counts = pixel_counts

        return pixel_counts

    def get_areas(self):
        """
        Calculates the area in hectares for each class based on pixel counts and raster resolution.

        Returns:
            Dict[int, float]: Dictionary mapping class values to their respective areas in hectares.
        """

        if self.pixel_counts:
            pixel_counts = self.pixel_counts
        else:
            pixel_counts = self.get_pixel_counts_by_class()

        resolution = get_map_resolution(self.map_path)
        areas_ha = {
            k: (v * resolution[0] * resolution[1]) / 100 * 100
            for k, v in pixel_counts.items()
        }
        return areas_ha

    def get_shape(self):
        """
        Retrieves the shape (width and height in pixels) of the regions raster map.

        Returns:
            Tuple[int, int]: Width and height of the raster.
        """

        return get_width_height(self.map_path)
