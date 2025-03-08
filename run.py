import click

from unbiased_area_estimation.config import Config
from unbiased_area_estimation.sampling_design import SamplingDesignPipeline


@click.command()
@click.argument(
    "config_fpath", type=click.Path(file_okay=True, dir_okay=False), required=True
)
def main(config_fpath):
    # IMPORTANT - THIS IS A WIP THAT WILL LEAD TO WRONG RESULTS IF INCORRECT NODATA VALUES ARE USED.
    # TODO CHECK RASTER MASKING ZERO COUNTING

    config = Config.load_from_json(json_path=config_fpath)

    sampling_design_pipeline = SamplingDesignPipeline(config=config)
    sampling_design_pipeline.run()


if __name__ == "__main__":
    main()
