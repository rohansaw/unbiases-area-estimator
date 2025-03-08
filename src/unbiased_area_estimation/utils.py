import rasterio as rio
from osgeo import gdal


def get_nodata_value(map_path: str):
    with rio.open(map_path) as src:
        return src.nodata


def get_width_height(map_path: str):
    with rio.open(map_path) as src:
        return src.width, src.height


def get_map_dtype(map_path: str):
    raster = gdal.Open(map_path, gdal.GA_ReadOnly)
    if raster is None:
        raise ValueError(f"Failed to open raster at {map_path}")

    raster_band = raster.GetRasterBand(1)
    if raster_band is None:
        raise ValueError("Raster or mask does not contain a valid band.")

    return raster_band.DataType
