import os
import tempfile
from pathlib import Path
from typing import Dict

import mmh3
import numpy as np
import pandas as pd
import rasterio as rio
from osgeo import gdal, osr

from unbiased_area_estimation.raster_handling import (
    mask_raster_by_vector,
    merge_classes,
    reproject_raster,
    reproject_vector,
)
from unbiased_area_estimation.sampling_allocation import SampleAllocator
from unbiased_area_estimation.utils import benchmark, load_sampling_design


class Region:
    def __init__(self, name, raster_path, mask_raster_path=None):
        self.name = name
        self.raster_path = raster_path
        self.mask_raster_path = mask_raster_path

    def get_pixel_counts(self):
        try:
            # Open raster
            raster = gdal.Open(self.raster_path)
            if raster is None:
                raise ValueError(f"Failed to open raster at {self.raster_path}")

            # Open mask
            mask = gdal.Open(self.mask_raster_path)
            if mask is None:
                raise ValueError(f"Failed to open mask at {self.mask_raster_path}")

            # Get raster bands
            raster_band = raster.GetRasterBand(1)
            mask_band = mask.GetRasterBand(1)

            if raster_band is None or mask_band is None:
                raise ValueError("Raster or mask does not contain a valid band.")

            x_size = raster_band.XSize  # Width
            y_size = raster_band.YSize  # Height

            pixel_counts = {}  # Dictionary to store pixel value counts

            # Read in chunks to save memory
            block_size = block_size = min(4096, x_size, y_size)
            for y in range(0, y_size, block_size):
                for x in range(0, x_size, block_size):
                    x_block = min(block_size, x_size - x)
                    y_block = min(block_size, y_size - y)

                    # Read raster and mask chunks
                    raster_chunk = raster_band.ReadAsArray(x, y, x_block, y_block)
                    mask_chunk = mask_band.ReadAsArray(x, y, x_block, y_block)

                    if raster_chunk is None or mask_chunk is None:
                        raise ValueError("Error reading raster or mask block.")

                    # Apply mask (consider only pixels where mask > 0)
                    masked_values = raster_chunk[mask_chunk > 0]

                    # Count occurrences of each value
                    unique, counts = np.unique(masked_values, return_counts=True)

                    for val, count in zip(unique, counts):
                        pixel_counts[val] = pixel_counts.get(val, 0) + count

            pixel_counts = {int(k): int(v) for k, v in pixel_counts.items()}
            print(f"Pixel counts by value within the mask: {pixel_counts}")
            raster = None
            raster_band = None
            mask_band = None
            return pixel_counts

        except Exception as e:
            print(f"Error processing raster and mask: {e}")
            return None  # Return None or handle it appropriately

        finally:
            del raster, mask  # Free memory

    def get_areas(self, pixel_counts):
        ds = gdal.Open(self.raster_path)
        gt = ds.GetGeoTransform()
        pixel_size = gt[1]
        areas_ha = {
            k: (v * pixel_size * pixel_size) / 100 * 100
            for k, v in pixel_counts.items()
        }
        ds = None
        return areas_ha

    def _get_coords(self, samples, gt, proj_ref, wgs84):
        coords = []
        for loc in samples:
            x_px = loc[0]
            y_px = loc[1]
            stratum_id = loc[2]
            y_m = gt[3] + y_px * gt[5]
            x_m = gt[0] + x_px * gt[1]
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

    def sample(self, num_samples_per_stratum, shuffle=True):
        samples = {}
        remaining_sample_counter = num_samples_per_stratum.copy()
        shape_x, shape_y = self._get_raster_shape()

        np.random.seed(5)

        # TODO add an upper bounds of tries?
        while sum(remaining_sample_counter.values()) > 0:
            x = np.random.randint(0, shape_x)
            y = np.random.randint(0, shape_y)

            if self.mask_raster_path:
                # Check if the samples pixel is within our roi, if not skip
                with rio.open(self.mask_raster_path) as ds_mask:
                    value = ds_mask.read(1, window=((y, y + 1), (x, x + 1)))
                    if value == 0:
                        continue

            with rio.open(self.raster_path) as ds:
                value = ds.read(
                    1, window=((y, y + 1), (x, x + 1))
                )  # Read a single pixel
                sample_cls = value[0, 0]  # Extract the scalar value
                if remaining_sample_counter[sample_cls] > 0:
                    if (x, y, sample_cls) in samples:
                        # This would break the assumption for random sampling.
                        # We might want to think about using reservoir sampling.
                        raise Exception(
                            "The same sample was picked twice, not yet handling this case."
                        )
                    samples[(x, y, sample_cls)] = True
                    remaining_sample_counter[sample_cls] -= 1

        base_ds = gdal.Open(self.raster_path)
        gt = base_ds.GetGeoTransform()
        projection = base_ds.GetProjectionRef()
        proj_ref = osr.SpatialReference()
        proj_ref.ImportFromWkt(projection)
        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)
        if int(gdal.VersionInfo()) >= 3000000:
            wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        samples_df = self._get_coords(samples.keys(), gt, proj_ref, wgs84)

        if shuffle:
            samples_df = samples_df.sample(frac=1)  # shuffle the samples

        samples_df["PLOTID"] = np.arange(0, samples_df.shape[0])
        base_ds = None
        return samples_df

    def _get_raster_shape(self):
        raster = gdal.Open(self.raster_path)
        if raster is None:
            raise ValueError(f"Failed to open raster at {self.raster_path}")

        x_size = raster.RasterXSize
        y_size = raster.RasterYSize

        raster = None
        return x_size, y_size


class AreaEstimator:
    def __init__(
        self,
        raster_path: str,
        mask_paths: Dict[str, str],
        sample_allocator: SampleAllocator,
        temp_dir: str = None,
        class_merge_dict: Dict[str, str] = None,
        epsg: str = None,
        output_dir: str = None,
        overwrite_existing: bool = True,
    ):
        self.sample_allocator = sample_allocator
        self.tempdir = temp_dir if temp_dir else tempfile.mkdtemp()
        self.raster_path = raster_path
        self.mask_paths = mask_paths
        self.class_merge_dict = class_merge_dict
        self.epsg = epsg
        self.regions = None
        self.output_dir = output_dir
        self.overwrite_existing = overwrite_existing

    @benchmark("Preprocessing files.")
    def preprocess_files(self):
        print("Preprocessing files. This might take a while...")
        self.regions = self._create_regions(
            self.raster_path, self.mask_paths, self.class_merge_dict, self.epsg
        )

    def _create_regions(
        self,
        raster_path: str,
        mask_paths: Dict[str, str],
        class_merge_dict: Dict[str, str],
        epsg: str,
    ):
        regions = {}
        original_name = Path(raster_path).stem

        # Reproject raster if necessary
        if epsg:
            reproject_raster_path = os.path.join(
                self.tempdir,
                Path(raster_path).stem
                + "_reprojected_EPSG"
                + str(mmh3.hash(epsg))
                + Path(raster_path).suffix,
            )
            if not os.path.exists(reproject_raster_path) or self.overwrite_existing:
                raster_path = reproject_raster(raster_path, reproject_raster_path, epsg)
            else:
                raster_path = reproject_raster_path

        # Merge classes if necessary
        if class_merge_dict:
            raster_fused_path = os.path.join(
                self.tempdir,
                Path(raster_path).stem + "_fused" + Path(raster_path).suffix,
            )

            if not os.path.exists(raster_fused_path) or self.overwrite_existing:
                raster_path = merge_classes(
                    raster_path, raster_fused_path, class_merge_dict
                )
            else:
                raster_path = raster_fused_path

        # If no masks are provided, use the entire raster
        if len(mask_paths) == 0:
            print("No masks provided. Sampling the entire raster.")
            regions[original_name] = Region(
                name=original_name, raster_path=raster_path, mask_raster_path=None
            )
            return regions

        # Reproject masks if necessary
        if len(mask_paths) > 0 and epsg:
            new_mask_paths = {}
            for mask_name, mask_path in mask_paths.items():
                new_mask_path = os.path.join(
                    self.tempdir,
                    mask_name
                    + "_reprojected_EPSG"
                    + str(mmh3.hash(epsg))
                    + Path(mask_path).suffix,
                )

                if not os.path.exists(new_mask_path) or self.overwrite_existing:
                    reproject_vector(mask_path, new_mask_path, epsg)

                new_mask_paths[mask_name] = new_mask_path

            mask_paths = new_mask_paths

        # TODO validate that masks and raster have the same crs.
        # Also validate nodata value correctly used and set in all cases

        # Create rasters masked by each vector and store them in a dictionary
        for mask_name, mask_path in mask_paths.items():
            masked_raster_out_path = os.path.join(
                self.tempdir,
                Path(raster_path).stem
                + "_"
                + Path(mask_path).stem
                + Path(raster_path).suffix,
            )

            if not os.path.exists(masked_raster_out_path) or self.overwrite_existing:
                mask_raster_by_vector(raster_path, mask_path, masked_raster_out_path)

            if mask_name in regions:
                raise ValueError(
                    f"Mask names must be unique. {mask_name} already exists."
                )

            regions[mask_name] = Region(
                name=mask_name,
                raster_path=raster_path,
                mask_raster_path=masked_raster_out_path,
            )

        return regions

    def create_sample_designs(self):
        print("Creating sample designs...")
        sample_designs = {}

        for region_name, region in self.regions.items():
            sample_design = self._create_sample_design(region)
            sample_designs[region_name] = sample_design

        return sample_designs

    def _create_sample_design(self, region: Region):
        pixel_counts = region.get_pixel_counts()
        areas = region.get_areas(pixel_counts)
        out_path = os.path.join(self.output_dir, f"{region.name}_design.csv")
        num_samples = self.sample_allocator.allocate(pixel_counts, areas, out_path)
        return num_samples

    @benchmark("All regions sampled.")
    def create_sample_sets(self, sample_designs: Dict[str, Dict[str, int]]):
        print("Creating sample sets...")
        sample_sets = {}

        for region_name, sample_design in sample_designs.items():
            region = self.regions[region_name]
            sample_set = region.sample(sample_design)
            sample_sets[region_name] = sample_set

        return sample_sets

    def get_unbiased_area_estimates(self):
        results = {}
        for region in self.regions:
            results[region.name] = region.compute_area_metrics()
        return results

    def get_expected_target_errors(self, sampling_design):
        expected_target_errors = {}
        for region_name, sample_design in sampling_design.items():
            original_sampling_design = load_sampling_design(
                region_name, self.output_dir
            )
            nInt = original_sampling_design["nInt"]
            wh = original_sampling_design["wh"]
            te = self.sample_allocator.compute_expected_error(sample_design, nInt, wh)
            expected_target_errors[region_name] = te
        return expected_target_errors

    def save_sample_sets(self, sample_sets, single_file: bool):
        for region_name, sample_set in sample_sets.items():
            sample_set["RegionName"] = self.regions[region_name].name

        if single_file:
            outfile = os.path.join(self.output_dir, "samples.csv")
            samples = pd.concat([sample_set for sample_set in sample_sets.values()])
            samples = samples.sample(frac=1)
            samples["PLOTID"] = np.arange(0, samples.shape[0])
            samples.to_csv(outfile, index=False)
            print(f"Samples saved to {outfile}")
            return

        for region_name, sample_set in sample_sets.items():
            outfile = os.path.join(self.output_dir, f"{region_name}_samples.csv")
            sample_set.to_csv(outfile, index=False)
        print(f"Samples saved to {self.output_dir}")
        return self.output_dir

    @benchmark("Complete sampling design run.")
    def run_complete_sampling_design(self, single_file: bool = False):
        self.preprocess_files()
        sample_sets = self.create_sample_sets(self.output_dir)
        self.save_sample_sets(sample_sets, self.output_dir, single_file)
