from typing import Dict

import numpy as np
import rasterio as rio
from osgeo import gdal
from rasterio.windows import Window


class Region:
    def __init__(self, name: str, raster_path: str):
        self.name = name
        self.raster_path = raster_path
        self.pixel_counts = None

    def get_pixel_counts_by_class(self) -> Dict[str, int]:
        with rio.open(self.raster_path) as src:
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

            pixel_counts = {}

            # Process raster in blocks
            for row_off in range(0, height, blockysize):
                for col_off in range(0, width, blockxsize):
                    # Define block size ensuring it doesn't exceed raster dimensions
                    win_width = min(blockxsize, width - col_off)
                    win_height = min(blockysize, height - row_off)
                    window = Window(col_off, row_off, win_width, win_height)
                    raster_chunk = src.read(1, window=window)

                    if raster_chunk is None:
                        raise ValueError("Error reading raster block.")

                    # Count unique pixel values
                    unique, counts = np.unique(raster_chunk, return_counts=True)

                    for val, count in zip(unique, counts):
                        pixel_counts[val] = pixel_counts.get(val, 0) + count

            if nodata_value is not None:
                pixel_counts.pop(nodata_value, None)

            pixel_counts = {int(k): int(v) for k, v in pixel_counts.items()}

            print(f"Pixel counts: {pixel_counts}")
            self.pixel_counts = pixel_counts
            return pixel_counts

    def get_areas(self):
        if self.pixel_counts:
            pixel_counts = self.pixel_counts
        else:
            pixel_counts = self.get_pixel_counts_by_class()

        ds = gdal.Open(self.raster_path)
        gt = ds.GetGeoTransform()
        pixel_size = gt[1]
        areas_ha = {
            k: (v * pixel_size * pixel_size) / 100 * 100
            for k, v in pixel_counts.items()
        }
        ds = None
        return areas_ha

    def get_raster_shape(self):
        raster = gdal.Open(self.raster_path)
        if raster is None:
            raise ValueError(f"Failed to open raster at {self.raster_path}")

        x_size = raster.RasterXSize
        y_size = raster.RasterYSize

        raster = None
        return x_size, y_size
