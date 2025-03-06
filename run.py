import click

from unbiased_area_estimation.sampling_allocation import create_sample_allocator
from unbiased_area_estimation.unbiased_estimation import AreaEstimator
from unbiased_area_estimation.utils import read_config


@click.command()
@click.argument("config_file")
def main(config_file):
    # IMPORTANT - THIS IS A WIP THAT WILL LEAD TO WRONG RESULTS IF INCORRECT NODATA VALUES ARE USED.
    # TODO CHECK RASTER MASKING ZERO COUNTING

    config = read_config(config_file)
    sample_allocator = create_sample_allocator(config["sampling"])

    area_estimator = AreaEstimator(
        raster_path=config["raster_path"],
        mask_paths=config["mask_paths"],
        class_merge_dict=config["class_merge_dict"],
        epsg=config["epsg"],
        sample_allocator=sample_allocator,
        temp_dir=config["output_dir"],
        output_dir=config["output_dir"],
    )

    # Preprocesses data, creates a sample allocation and saves the samples in the output directory
    area_estimator.run_complete_sampling_design(single_file=True)


if __name__ == "__main__":
    main()
