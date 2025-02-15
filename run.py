import click
import json

from sampling_allocation import ProportionalAllocator, NeymanAllocator
from unbiased_estimation import AreaEstimator

def read_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)

    config['class_merge_dict'] = {int(k): int(v) for k, v in config['class_merge_dict'].items()}
    config['sampling']['expected_uas'] = {int(k): float(v) for k, v in config['sampling']['expected_uas'].items()}
    return config

def create_sample_allocator(sampling_params):
    if sampling_params['method'] == 'proportional':
        return ProportionalAllocator(
            expected_uas=sampling_params['expected_uas'],
            target_error=sampling_params['target_error']
        )
    elif sampling_params['method'] == 'neyman':
        return NeymanAllocator(
            expected_uas=sampling_params['expected_uas'],
            target_error=sampling_params['target_error']
        )
    else:
        raise ValueError('Invalid sampling method specified in config file.')

@click.command()
@click.argument('config_file')
def main(config_file):
    # IMPORTANT - THIS IS A WIP THAT WILL LEAD TO WRONG RESULTS IF 
    # INCORRECT NODATA VALUES ARE USED. TODO CHECK RASTER MASKIN ZERO COUNTING 

    config = read_config(config_file)
    sample_allocator = create_sample_allocator(config['sampling'])

    area_estimator = AreaEstimator(
        raster_path=config['raster_path'],
        mask_paths=config['mask_paths'],
        class_merge_dict=config['class_merge_dict'],
        epsg=config['epsg'],
        sample_allocator=sample_allocator
    )

    area_estimator.prepare_raster()
    sampling_designs = area_estimator.create_sample_designs()

    sample_sets = area_estimator.create_sample_sets(sampling_designs)
    area_estimator.save_sample_sets(sample_sets, config['output_dir'], False)
    

if __name__ == '__main__':
    main()