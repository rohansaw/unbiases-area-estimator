import rasterio as rio
from rasterio.warp import calculate_default_transform
import subprocess
import tempfile
import os
import geopandas as gpd
from osgeo import gdal
import numpy as np

def reproject_raster(
    in_raster_path,
    out_raster_path,
    dst_crs,
    dst_resolution=None,
    dst_extent=None,
    dst_dtype='Byte',
    src_nodata=None,
    dst_nodata=0,
    compress="DEFLATE",
):
    print(f"Reprojecting raster {in_raster_path} to {dst_crs}")
    with rio.open(in_raster_path) as src:
        src_crs = src.crs
        src_transform = src.transform
        src_width = src.width
        src_height = src.height
        
        if dst_resolution is None:
            dst_resolution = src.res
        
        if dst_extent is None:
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src_crs, dst_crs, src_width, src_height, *src.bounds, resolution=dst_resolution[0]
            )
            dst_extent = [dst_transform.c, dst_transform.f + dst_transform.e * dst_height, 
                          dst_transform.c + dst_transform.a * dst_width, dst_transform.f]
        
        if dst_dtype is None:
            original_dtype = src.dtypes[0]
            # make first letter uppercase
            dst_dtype = original_dtype[0].upper() + original_dtype[1:]

            # if it contains 'int', make it 'Int', if it contains 'float', make it 'Float' to cover for e.g UInt16 etc
            if 'int' in dst_dtype:
                dst_dtype = dst_dtype.replace('int', 'Int')
            elif 'float' in dst_dtype:
                dst_dtype = dst_dtype.replace('float', 'Float')
        
        if src_nodata is None:
            src_nodata = src.nodata if src.nodata is not None else 0
    
    command = (
        f'gdalwarp -co "COMPRESS={compress}" '
        f'-t_srs "{dst_crs}" '
        f'-tr {dst_resolution[0]} {dst_resolution[1]} '
        f'-te {dst_extent[0]} {dst_extent[1]} {dst_extent[2]} {dst_extent[3]} '
        f'-ot {dst_dtype} '
        f'-srcnodata "{src_nodata}" '
        f'-dstnodata "{dst_nodata}" '
        f'"{in_raster_path}" "{out_raster_path}"'
    )
    
    subprocess.run(command, shell=True, check=True)

    return out_raster_path


def reproject_vector(
    in_vector_path,
    out_vector_path,
    dst_crs,
): 
    print(f"Reprojecting vector {in_vector_path} to {dst_crs}")
    command = f'ogr2ogr -t_srs "{dst_crs}" {out_vector_path} {in_vector_path}'

    subprocess.run(command, shell=True, check=True)

    return out_vector_path


def merge_classes(raster_path, out_raster_path, class_merge_dict):
    print(f"Reclassing raster {raster_path}")

    # Open the source raster
    base_ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    base_raster = np.array(base_ds.GetRasterBand(1).ReadAsArray(), dtype='uint8')
    if base_raster is None:
        raise ValueError(f"Could not open raster {raster_path}")

      
    for init_cls, dest_cls in class_merge_dict.items():
        base_raster[base_raster==init_cls] = dest_cls
            
            
    # Write output array
    driver = gdal.GetDriverByName('GTiff')
    output_dataset = driver.CreateCopy(out_raster_path, base_ds, options = ["COMPRESS=DEFLATE"])
    new_band = output_dataset.GetRasterBand(1)
    new_band.WriteArray(base_raster)
    new_band.FlushCache()
    output_dataset = None
    new_band = None
    base_raster = None
    base_ds = None

    return out_raster_path


def rasterize(
    vector_path,
    raster_outpath,
    resolution,
    dtype='Byte',
    nodata=0,
    compress='DEFLATE'
):
    print(f"Rasterizing {vector_path} to {raster_outpath}")

    gdf = gpd.read_file(vector_path)
    bounds = gdf.total_bounds
    min_x, min_y, max_x, max_y = bounds

    if min_y > max_y:
        min_y, max_y = max_y, min_y
        print('Switching min_y and max_y')

    if min_x > max_x:
        min_x, max_x = max_x, min_x
        print('Switching min_x and max_x')

    extent = (min_x, min_y, max_x, max_y)

    command = (
        f'gdal_rasterize -tr {resolution[0]} {resolution[1]} -burn 1 '
        f'-a_nodata {nodata} -te {extent[0]} {extent[1]} {extent[2]} {extent[3]} '
        f'-co "COMPRESS={compress}" -ot {dtype} "{vector_path}" "{raster_outpath}"'
    )
    
    subprocess.run(command, shell=True, check=True)
    return raster_outpath

def mask_raster_by_vector(
    raster_path,
    vector_path,
    raster_outpath,
    resolution=None,
    extent=None,
    dtype='Byte',
    src_nodata=0,
    compress='DEFLATE',
):
    print(f"Masking raster {raster_path} by vector {vector_path}")
    with rio.open(raster_path) as src:
        if extent is None:
            extent = src.bounds
        if resolution is None:
            resolution = src.res

    tmp_masked_raster_path = tempfile.mktemp(suffix='.tif')
    rasterized_vector_path = rasterize(vector_path, tmp_masked_raster_path, resolution, dtype, src_nodata, compress)
    
    print(f"Masking raster")
    command = (
        f'gdalwarp -co "COMPRESS={compress}" '
        f'-tr {resolution[0]} {resolution[1]} -te {extent[0]} {extent[1]} {extent[2]} {extent[3]} '
        f'-ot {dtype} -srcnodata "{src_nodata}" '
        f'"{rasterized_vector_path}" "{raster_outpath}"'
    )
    
    subprocess.run(command, shell=True, check=True)
        
    return raster_outpath

            