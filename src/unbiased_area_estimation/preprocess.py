import shutil
import subprocess
from typing import Dict, List

import geopandas as gpd
import rasterio as rio
from osgeo import gdal
from rasterio.warp import calculate_default_transform
from rasterio.windows import Window

from unbiased_area_estimation.storage_manager import StorageManager
from unbiased_area_estimation.utils import (
    get_map_dtype,
    get_map_extent,
    get_map_resolution,
    get_map_spatial_ref,
    get_mask_extent,
    get_mask_spatial_ref,
    get_nodata_value,
    get_width_height,
)


class Preprocessor:
    def __init__(self, storage_manager: StorageManager, block_size: int = 2048):
        self.storage_manager = storage_manager
        self.block_size = block_size

    def preprocess(
        self,
        map_path: str,
        mask_paths: List[str],
        target_spatial_ref: str = None,
        class_merge_map: Dict[int, int] = None,
    ) -> Dict[str, str]:
        if get_nodata_value(map_path) is None:
            # ToDo: In the future we should allow setting a nodata value
            print(
                f"WARNING: Map {map_path} does not have a nodata value set. Using 0 as nodata value. This might cause incorrect calculations."
            )

        if get_map_dtype(map_path) not in [
            gdal.GDT_Byte,
            gdal.GDT_UInt16,
            gdal.GDT_Int16,
            gdal.GDT_UInt32,
            gdal.GDT_Int32,
        ]:
            raise ValueError(
                "Sorry, only handling int-based map values for the moment."
            )

        if not target_spatial_ref:
            print(
                "No target spatial reference set. Reprojecting masks to match map spatial reference."
            )
            raise NotImplementedError("Not yet implemented")

        if target_spatial_ref:
            map_path = self._reproject_map(
                in_raster_path=map_path, target_spatial_ref=target_spatial_ref
            )

        if class_merge_map:
            map_path = self._merge_classes(
                in_raster_path=map_path, class_merge_map=class_merge_map
            )

        if len(mask_paths) == 0:
            return map_path, {}

        raster_mask_paths = {}
        for mask_name, mask_path in mask_paths.items():
            if target_spatial_ref:
                mask_path = self._reproject_vector_mask(
                    mask_in_path=mask_path, target_spatial_ref=target_spatial_ref
                )

            if get_mask_spatial_ref(mask_path) != get_map_spatial_ref(map_path):
                raise ValueError(
                    f"Map {map_path} and mask {mask_path} are not in the same spatial reference system."
                )

            rastered_mask_path = self._rasterize_mask(
                map_in_path=map_path, mask_in_path=mask_path
            )
            raster_mask_paths[mask_name] = {
                "path": rastered_mask_path,
                "extent": get_mask_extent(mask_path),
            }

        return map_path, raster_mask_paths

    def _rasterize_mask(self, map_in_path: str, mask_in_path: str):
        gdf = gpd.read_file(mask_in_path)
        bounds = gdf.total_bounds
        min_x, min_y, max_x, max_y = bounds

        if min_y > max_y:
            min_y, max_y = max_y, min_y
            print("Switching min_y and max_y")

        if min_x > max_x:
            min_x, max_x = max_x, min_x
            print("Switching min_x and max_x")

        map_resolution = get_map_resolution(map_in_path)
        map_extent = get_map_extent(map_in_path)

        rasterized_mask_path = self.storage_manager.get_rasterized_mask_path(
            mask_path=mask_in_path, resolution=map_resolution
        )

        if not self.storage_manager.exists(rasterized_mask_path):
            print(f"Rasterizing mask {mask_in_path}...")
            command = (
                f"gdal_rasterize -tr {map_resolution[0]} {map_resolution[1]} -burn 1 "
                f"-a_nodata 0 -te {map_extent[0]} {map_extent[1]} {map_extent[2]} {map_extent[3]} "
                f'-co "COMPRESS=DEFLATE" -ot Byte "{mask_in_path}" "{rasterized_mask_path}"'
            )

            subprocess.run(command, shell=True, check=True)

        rasterized_mask_mapaligned_path = self.storage_manager.get_masked_map_path(
            map_path=map_in_path, mask_path=rasterized_mask_path
        )

        if self.storage_manager.exists(rasterized_mask_mapaligned_path):
            print(f"Using cached rasterized mask {rasterized_mask_mapaligned_path}")
            return rasterized_mask_mapaligned_path

        src_nodata = get_nodata_value(map_in_path)
        dtype = gdal.GetDataTypeName(get_map_dtype(map_in_path))
        map_extent = get_map_extent(map_in_path)

        # Align extent and resolution with map
        command = (
            f'gdalwarp -co "COMPRESS=DEFLATE" '
            f"-tr {map_resolution[0]} {map_resolution[1]} -te {map_extent[0]} {map_extent[1]} {map_extent[2]} {map_extent[3]} "
            f'-ot {dtype} -srcnodata "{src_nodata}" '
            f'"{rasterized_mask_path}" "{rasterized_mask_mapaligned_path}"'
        )

        subprocess.run(command, shell=True, check=True)
        return rasterized_mask_mapaligned_path

    def _reproject_map(
        self,
        in_raster_path: str,
        target_spatial_ref: str,
        compress="DEFLATE",
    ):
        out_raster_path = self.storage_manager.get_reprojected_map_path(
            in_raster_path, target_spatial_ref
        )

        if self.storage_manager.exists(out_raster_path):
            print(f"Using cached reprojected raster {out_raster_path}")
            return out_raster_path

        print(f"Reprojecting raster {in_raster_path} to {target_spatial_ref}...")
        with rio.open(in_raster_path) as src:
            src_spatial_ref = src.crs
            src_width = src.width
            src_height = src.height
            src_nodata = src.nodata
            target_nodata = src.nodata
            target_resolution = src.res

            target_transform, target_width, target_height = calculate_default_transform(
                src_spatial_ref,
                target_spatial_ref,
                src_width,
                src_height,
                *src.bounds,
                resolution=target_resolution[0],
            )
            target_extent = [
                target_transform.c,
                target_transform.f + target_transform.e * target_height,
                target_transform.c + target_transform.a * target_width,
                target_transform.f,
            ]

        # Reading with gdal to avoid issues with different naming conventions
        target_dtype = gdal.GetDataTypeName(get_map_dtype(in_raster_path))

        command = (
            f'gdalwarp -co "COMPRESS={compress}" '
            f'-t_srs "{target_spatial_ref}" '
            f"-tr {target_resolution[0]} {target_resolution[1]} "
            f"-te {target_extent[0]} {target_extent[1]} {target_extent[2]} {target_extent[3]} "
            f"-ot {target_dtype} "
            f'-srcnodata "{src_nodata}" '
            f'-dstnodata "{target_nodata}" '
            f'"{in_raster_path}" "{out_raster_path}"'
        )
        subprocess.run(command, shell=True, check=True)

        return out_raster_path

    def _merge_classes(self, in_raster_path: str, class_merge_map: Dict[int, int]):
        out_raster_path = self.storage_manager.get_merged_classes_map_path(
            in_raster_path, class_merge_map
        )

        if self.storage_manager.exists(out_raster_path):
            print(f"Using cached merged raster {out_raster_path}")
            return out_raster_path

        if len(class_merge_map) == len(set(class_merge_map.values())):
            print("Class merge map is 1:1 mapping. Copying raster to cache.")
            shutil.copy(in_raster_path, out_raster_path)
            return out_raster_path

        print(f"Merging classes in raster {in_raster_path}...")
        with rio.open(in_raster_path) as src:
            profile = src.profile.copy()
            height, width = src.shape

            blocksize = min(self.block_size, width, height)

            # Ensure tiling for efficient reading later-on
            profile.update(
                tiled=True,
                blockxsize=blocksize,
                blockysize=blocksize,
                compress="DEFLATE",
            )

            # Open output raster
            with rio.open(out_raster_path, "w", **profile) as dst:
                # Iterate over chunks (row-wise and column-wise)
                for row_off in range(0, height, blocksize):
                    for col_off in range(0, width, blocksize):
                        # Define and read window
                        win_width = min(blocksize, width - col_off)
                        win_height = min(blocksize, height - row_off)
                        window = Window(col_off, row_off, win_width, win_height)
                        base_raster = src.read(1, window=window)

                        # Apply class merging transformation
                        for init_cls, dest_cls in class_merge_map.items():
                            base_raster[base_raster == init_cls] = dest_cls

                        dst.write(base_raster, 1, window=window)

        return out_raster_path

    def _reproject_vector_mask(self, mask_in_path: str, target_spatial_ref: str):
        out_mask_path = self.storage_manager.get_reprojected_mask_path(
            mask_in_path, target_spatial_ref
        )

        if self.storage_manager.exists(out_mask_path):
            print(f"Using cached reprojected mask {out_mask_path}")
            return out_mask_path

        print(f"Reprojecting mask {mask_in_path} to {target_spatial_ref}...")

        command = (
            f'ogr2ogr -t_srs "{target_spatial_ref}" {out_mask_path} {mask_in_path}'
        )

        subprocess.run(command, shell=True, check=True)
        return out_mask_path

    def _mask_map(self, map_in_path: str, mask_in_path: str, compress="DEFLATE"):
        out_map_path = self.storage_manager.get_masked_map_path(
            map_in_path, mask_in_path
        )

        if self.storage_manager.exists(out_map_path):
            print(f"Using cached masked map {out_map_path}")
            return out_map_path

        print(f"Masking map {map_in_path} with mask {mask_in_path}...")

        nodata_value = get_nodata_value(map_in_path)
        width, height = get_width_height(map_in_path)
        blocksize = min(self.block_size, width, height)

        command = (
            f'gdalwarp "{map_in_path}" "{out_map_path}" -cutline "{mask_in_path}" -crop_to_cutline '
            f'-dstnodata {nodata_value} -co "COMPRESS={compress}" -co "TILED=YES" -co "BLOCKXSIZE={blocksize}" -co "BLOCKYSIZE={blocksize}"'
        )

        subprocess.run(command, shell=True, check=True)
        return out_map_path
