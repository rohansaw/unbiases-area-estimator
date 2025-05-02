import click

from unbiased_area_estimation.config import Config
from unbiased_area_estimation.sampling_design import SamplingDesignPipeline


@click.command()
@click.argument(
    "config_fpath", type=click.Path(file_okay=True, dir_okay=False), required=True
)
def main(config_fpath):
    """
    Runs the complete sampling design creation pipeline based on an input configuration.
    This method does not enable to interactively adapt proposed sampling designs.
    The sampling design and sample set will be saved under the out_path specified in
    the configuration.

    Parameters:
        config_fpath (str): The path to the configuration file.

    Returns: None
    """
    config = Config.load_from_json(json_path=config_fpath)

    # Initialize Workflow Orchestrator
    sampling_design_pipeline = SamplingDesignPipeline(
        output_path=config.output_path,
        use_cached=config.use_cached,
        sampling_method=config.sampling.sampling_method,
        cache_path=config.cache_path,
    )

    # Run preprocessing, sampling design creation and saving
    sampling_design_pipeline.run(config=config)


if __name__ == "__main__":
    main()
