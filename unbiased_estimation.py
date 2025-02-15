from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import time

from raster_handling import merge_classes, reproject_raster, reproject_vector, mask_raster_by_vector
from sampling_allocation import SampleAllocator

from pathlib import Path
import tempfile
import os
from osgeo import gdal, osr
import rasterio as rio

def compute_area_metrics(predicted, annotated, wh, areas):
        classes = np.unique(annotated)
        num_classes = len(classes)
        confusion_matrix = np.zeros((num_classes, num_classes))
        
        # Compute confusion matrix
        for i, true_class in enumerate(classes):
            for j, pred_class in enumerate(classes):
                confusion_matrix[i, j] = np.sum((annotated == true_class) & (predicted == pred_class))
        
        counts_pred = np.sum(confusion_matrix, axis=1)  # Total predicted per class
        counts_true = np.sum(confusion_matrix, axis=0)  # Total actual per class
        total_positive = np.diag(confusion_matrix)  # True positives
        users_accuracy_by_class = np.diag(confusion_matrix) / counts_pred  # User's accuracy
        producers_accuracy_by_class = np.diag(confusion_matrix) / counts_true  # Producer's accuracy
        
        # Ensure wh dictionary matches class indices
        if set(classes) != set(wh.keys()):
            raise ValueError(f"Mismatch between confusion matrix classes {set(classes)} and weight keys {set(wh.keys())}")
        
        weights = np.array([wh[class_id] for class_id in classes])
        areas = np.array([areas[class_id] for class_id in classes])
        weighted_confusion_matrix = confusion_matrix * weights[:, np.newaxis]
        weighted_norm_confusion_matrix = weighted_confusion_matrix / counts_pred[:, np.newaxis]
        oa = np.sum(np.diag(weighted_norm_confusion_matrix))
        
        total_pred_weight_norm = np.sum(weighted_norm_confusion_matrix, axis=1)
        total_true_weight_norm = np.sum(weighted_norm_confusion_matrix, axis=0)
        users_accuracy_by_class_norm  = np.diag(weighted_norm_confusion_matrix) / total_pred_weight_norm
        producers_accuracy_by_class_norm = np.diag(weighted_norm_confusion_matrix) / total_true_weight_norm

        temp = np.square(weights) * users_accuracy_by_class * (1 - users_accuracy_by_class) / (counts_pred - 1) # Validate why -1 for pred
        v_oa = np.sum(temp)
        se_oa = np.sqrt(v_oa)
        oa_ci = 1.96 * se_oa

        v_ua = users_accuracy_by_class * (1 - users_accuracy_by_class) / (counts_pred - 1)
        se_ua = np.sqrt(v_ua)
        ua_ci = 1.96 * se_ua

        N_j = []
        for j in range(num_classes):
            n_j = np.sum(confusion_matrix[:, j] / counts_pred * areas)
            N_j.append(n_j)
        
        N_j = np.array(N_j)

        xp1 = np.square(areas) * np.square((1-producers_accuracy_by_class_norm)) * users_accuracy_by_class_norm * (1 - users_accuracy_by_class_norm) / (counts_pred -1)

        conf_no_diag = np.triu(confusion_matrix, k=1) + np.tril(confusion_matrix, k=-1)
        prod_a_sq = np.square(producers_accuracy_by_class_norm)
        areas_squared = np.square(areas)
        frac = conf_no_diag / counts_pred[:, np.newaxis]
        var_term = frac * (1 - frac) / (counts_pred[:, np.newaxis] - 1)
        xp2 = prod_a_sq * np.sum(areas_squared[:, np.newaxis] * var_term, axis=0)
        
        v_pa = (1/np.square(N_j)) * (xp1+xp2)
        se_pa = np.sqrt(v_pa)
        pa_ci = 1.96 * se_pa

        s_pk_arr = []
        for i in range(num_classes):
            ir = np.sum((weights * weighted_norm_confusion_matrix[:, i] - np.square(weighted_norm_confusion_matrix[:, i])) / (counts_pred -1))
            s_pk_arr.append(ir)
        s_pk = np.sqrt(np.array(s_pk_arr))

        se_areas = s_pk * areas
        areas_ci = 1.96 * se_areas
        uc_ci = areas_ci / areas
        areas_se_percent = se_areas / areas


        classwise_metrics_df = pd.DataFrame({
            "Class": classes,
            "Total Predicted": counts_pred,
            "Total True": counts_true,
            "Total Positive": total_positive,
            "User's Accuracy": users_accuracy_by_class,
            "Producer's Accuracy": producers_accuracy_by_class,
            "User's Accuracy Proportional": users_accuracy_by_class_norm,
            "Producer's Accuracy Proportional": producers_accuracy_by_class_norm,
            "User's Variance": v_ua,
            "Producer's Variance": v_pa,                                                                                                                            
            "User's Std Error": se_ua,                                           
            "Producer's Std Error": se_pa,
            "User's 95% CI": ua_ci,
            "Producer's 95% CI": pa_ci,
            "Area": areas,
            "95% Area Error": areas_ci,
            "UC 95% error": uc_ci,
            "Area Std Error": se_areas,
            "Area Std Error %": areas_se_percent,
        })

        overall_metrics_df = pd.DataFrame({
            "Overall Accuracy": oa,
            "Overall Accuracy Variance": v_oa,
            "Overall Accuracy Std Error": se_oa,
            "Overall Accuracy 95% CI": oa_ci
        })
        
        return classwise_metrics_df, overall_metrics_df

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
            block_size = 512
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
            print(f'Pixel counts by value within the mask: {pixel_counts}')
            return pixel_counts

        except Exception as e:
            print(f"Error processing raster and mask: {e}")
            return None  # Return None or handle it appropriately

        finally:
            del raster, mask  # Free memory
    
    def get_area(self):
        pass

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
                'x_m': x_m,
                'y_m': y_m,
                'x_px': x_px,
                'y_px': y_px,
                'lat': lat,
                'lon': lon,
                'stratum_id': stratum_id,
                'class_id': -1
            }
            coords.append(entry)
        
        coords_df = pd.DataFrame(coords)
        return coords_df
    
    def sample(self, num_samples_per_stratum, shuffle=True):
        samples = {}
        remaining_sample_counter = num_samples_per_stratum.copy()
        shape_x, shape_y = self._get_raster_shape()

        np.random.seed(5)
        

        # TODO add an upper bounds of tries
        while sum(remaining_sample_counter.values()) > 0:
            print(remaining_sample_counter)
            x = np.random.randint(0, shape_x)
            y = np.random.randint(0, shape_y)

            if self.mask_raster_path:
                # Check if the samples pixel is within our roi, if not skip
                with rio.open(self.mask_raster_path) as ds_mask:
                    value = ds_mask.read(1, window=((y, y+1), (x, x+1)))
                    if value == 0:
                        continue

            with rio.open(self.raster_path) as ds:
                value = ds.read(1, window=((y, y+1), (x, x+1)))  # Read a single pixel
                sample_cls = value[0, 0]  # Extract the scalar value
                if remaining_sample_counter[sample_cls] > 0:
                    if (x, y, sample_cls) in samples:
                        raise Exception('The same sample was picked twice, not yet handling this case.')
                    samples[(x, y, sample_cls)] = True
                    remaining_sample_counter[sample_cls] -= 1
        
        base_ds = gdal.Open(self.raster_path)
        gt = base_ds.GetGeoTransform()
        projection = base_ds.GetProjectionRef()
        proj_ref = osr.SpatialReference()
        proj_ref.ImportFromWkt(projection)
        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)
        if (int(gdal.VersionInfo()) >= 3000000):
            wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        samples_df = self._get_coords(samples.keys(), gt, proj_ref, wgs84)

        if shuffle:
            samples_df = samples_df.sample(frac=1) # shuffle the samples
        
        samples_df['sample_id'] = np.arange(0, samples_df.shape[0])
        return samples_df


    def sample_memory_unefficient(self, num_samples_per_stratum, shuffle=True):
        
        base_ds = gdal.Open(self.raster_path)
        base_raster = np.array(base_ds.GetRasterBand(1).ReadAsArray(), dtype='uint8')

        if self.mask_raster_path:
            mask_ds = gdal.Open(self.mask_raster_path)
            mask_raster = np.array(mask_ds.GetRasterBand(1).ReadAsArray(), dtype='uint8')


        # Setting up transformation to lat/lon
        gt = base_ds.GetGeoTransform()
        projection = base_ds.GetProjectionRef()
        proj_ref = osr.SpatialReference()
        proj_ref.ImportFromWkt(projection)
        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)
        if (int(gdal.VersionInfo()) >= 3000000):
            wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        
        np.random.seed(5)

        samples = []
        for stratum_id, n_samples in num_samples_per_stratum.items():

            if self.mask_raster_path:
                locations = np.where(mask_raster & (base_raster==stratum_id))
            else:
                locations = np.where(base_raster==stratum_id)
            

            # Sample random points
            samples_idx = np.random.choice(np.arange(0,locations[0].shape[0]), size = n_samples, replace = False)
            coords = self._get_coords(samples_idx, locations, gt, proj_ref, wgs84, stratum_id, base_raster)
            samples.append(coords)

        samples_df = pd.concat(samples)

        if shuffle:
            samples_df = samples_df.sample(frac=1) # shuffle the samples
        
        samples_df['sample_id'] = np.arange(0, samples_df.shape[0])
        return samples_dfs

    def _get_raster_shape(self):
        raster = gdal.Open(self.raster_path)
        if raster is None:
            raise ValueError(f"Failed to open raster at {self.raster_path}")

        x_size = raster.RasterXSize
        y_size = raster.RasterYSize

        return x_size, y_size


class AreaEstimator:
    def __init__(self, raster_path: str, mask_paths: list[str], sample_allocator: SampleAllocator, temp_dir: str = None, class_merge_dict: Dict[str, str] = None , epsg: str = None, ):
        self.sample_allocator = sample_allocator
        self.tempdir = temp_dir if temp_dir else tempfile.mkdtemp()
        self.raster_path = raster_path
        self.mask_paths = mask_paths
        self.class_merge_dict = class_merge_dict
        self.epsg = epsg
        self.regions = None

        #TODO ensure region_names are unique

    def preprocess_files(self):
        print('Preprocessing files. This might take a while...')
        t = time.time()
        self.regions = self._create_regions(self.raster_path, self.mask_paths, self.class_merge_dict, self.epsg)
        print(f'Preprocessing done in {time.time() - t} seconds')

    def _create_regions(self, raster_path: str, mask_paths: List[str], class_merge_dict: Dict[str, str], epsg: str):
        raster_name = Path(raster_path).stem
        regions = {}

        if epsg:
            reproject_raster_path = os.path.join(self.tempdir, raster_name + '_reprojected' + Path(raster_path).suffix)
            raster_path = reproject_raster(raster_path, reproject_raster_path, epsg)

        if class_merge_dict:
            raster_fused_path = os.path.join(self.tempdir, raster_name + '_fused' + Path(raster_path).suffix)
            raster_path = merge_classes(raster_path, raster_fused_path, class_merge_dict)

        if mask_paths and epsg:
            new_mask_paths = [os.path.join(self.tempdir, Path(mask_path).stem + '_reprojected' + Path(mask_path).suffix) for mask_path in mask_paths]
            mask_paths = [reproject_vector(mask_path, reproj_mask_path, epsg) for mask_path, reproj_mask_path in zip(mask_paths, new_mask_paths)]

        # TODO validate that masks and raster have the same crs. Also validate nodata value correctly used and set in all cases

        if len(mask_paths) == 0:
            regions[raster_name] = raster_path
       
        for mask_path in mask_paths:
            region_name = raster_name + '_' + Path(mask_path).stem
            masked_raster_out_path = os.path.join(self.tempdir, region_name + '_masked' + Path(raster_path).suffix)
            masked_raster_path = mask_raster_by_vector(raster_path, mask_path, masked_raster_out_path)
            if region_name in regions:
                raise ValueError(f"Mask names must be unique. {region_name} already exists.")
            regions[region_name] = Region(region_name, raster_path, masked_raster_path)
        
        return regions
    
    def create_sample_designs(self):
        print('Creating sample design...')
        sample_designs = {}

        for region_name, region in self.regions.items():
            sample_design = self._create_sample_design(region)
            sample_designs[region_name] = sample_design

        return sample_designs
    
    def _create_sample_design(self, region: Region):
        pixel_counts = region.get_pixel_counts()
        num_samples = self.sample_allocator.allocate(pixel_counts)
        return num_samples
       
    def create_sample_sets(self, sample_designs: Dict[str, Dict[str, int]]):
        print('Creating sample sets...')
        sample_sets = {}
        t = time.time()

        for region_name, sample_design in sample_designs.items():
            region = self.regions[region_name]
            sample_set = region.sample(sample_design)
            sample_sets[region_name] = sample_set

        print(f'Sample sets created in {time.time() - t} seconds')
        return sample_sets
    
    def update_confusion_matrix(self, region_name: str, confusion_matrix: pd.DataFrame):
        region = self.regions[region_name]
        region.confusion_matrix = confusion_matrix

    def get_unbiased_area_estimates(self):
        results = {}
        for region in self.regions:
            results[region.name] = region.compute_area_metrics()
        return results
    
    def save_sample_sets(self, sample_sets, output_dir: str, single_file: bool):
        if single_file:
            outfile = os.path.join(output_dir, 'samples.csv')
            samples = pd.concat([sample_set for sample_set in sample_sets.values()])
            samples = samples.sample(frac=1)
            samples.to_csv(outfile, index=False)
            return

        for region_name, sample_set in sample_sets.items():
            outfile = os.path.join(output_dir, f'{region_name}_samples.csv')
            sample_set.to_csv(outfile, index=False)
    
    def run_sampling_design(self, output_dir: str, single_file: bool = False):
        self.preprocess_files()
        sample_sets = self.create_sample_sets()
        self.save_sample_sets(sample_sets, output_dir, single_file)

