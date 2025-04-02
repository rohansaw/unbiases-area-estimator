from typing import Dict, List

import numpy as np
import rasterio as rio
from rasterio.windows import Window

from unbiased_area_estimation.utils import (
    benchmark,
    get_map_resolution,
    get_width_height,
)


class Region:
    def __init__(
        self, name: str, map_path: str, mask_path: str = None, mask_extent: List = None
    ):
        self.name = name
        self.map_path = map_path
        self.mask_path = mask_path
        self.mask_extent = mask_extent
        self.pixel_counts = None

    @benchmark("get_pixel_counts_by_class")
    def get_pixel_counts_by_class(self) -> Dict[str, int]:
        with rio.open(self.map_path) as src:
            nodata_value = src.nodata
            dtype = src.dtypes[0]

            # Ensure only int-based rasters are processed
            if dtype not in ["uint8", "uint16", "int16", "uint32", "int32"]:
                raise ValueError(
                    "Sorry, only handling int-based values for the moment."
                )

            width, height = src.width, src.height
            blockxsize, blockysize = src.block_shapes[0]

            # Ensure block size is meaningful, otherwise use 512 or min size
            if blockxsize == width and blockysize == 1:  # Striped raster (inefficient)
                blockxsize, blockysize = min(512, width), min(512, height)

            print(f"Block size: {blockxsize}x{blockysize}")

            pixel_counts = {}
            start_row = 0
            start_col = 0

            # TODO - might want to set start and end column only within mask
            mask_src = None
            if self.mask_extent is not None:
                mask_src = rio.open(self.mask_path)
                if mask_src.width != width or mask_src.height != height:
                    raise ValueError("Mask and raster do not have the same extent.")
                if mask_src.res != src.res:
                    raise ValueError("Mask and raster do not have the same resolution.")
                if mask_src.crs != src.crs:
                    raise ValueError("Mask and raster do not have the same CRS.")

            try:
                # Process raster in blocks
                for row_off in range(start_row, height, blockysize):
                    for col_off in range(start_col, width, blockxsize):
                        win_height = min(blockysize, height - row_off)
                        win_width = min(blockxsize, width - col_off)
                        window = Window(col_off, row_off, win_width, win_height)
                        raster_chunk = src.read(1, window=window)

                        if mask_src is not None:
                            mask_chunk = mask_src.read(1, window=window)
                            raster_chunk = raster_chunk[mask_chunk == 1]

                        if raster_chunk is None:
                            raise ValueError("Error reading raster block.")

                        # Count unique pixel values
                        unique, counts = np.unique(raster_chunk, return_counts=True)

                        for val, count in zip(unique, counts):
                            pixel_counts[val] = pixel_counts.get(val, 0) + count

                if nodata_value is not None:
                    pixel_counts.pop(nodata_value, None)

                pixel_counts = {int(k): int(v) for k, v in pixel_counts.items()}
                self.pixel_counts = pixel_counts

            except Exception as e:
                print(f"Error processing raster: {e}")
                raise e

            finally:
                if mask_src is not None:
                    mask_src.close()

            return pixel_counts

    def get_areas(self):
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
        return get_width_height(self.map_path)
