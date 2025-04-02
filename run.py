import os.path as op

import click

from unbiased_area_estimation.config import Config
from unbiased_area_estimation.sampling_design import SamplingDesignPipeline


@click.command()
@click.argument(
    "config_fpath", type=click.Path(file_okay=True, dir_okay=False), required=True
)
def main(config_fpath):
    config = Config.load_from_json(json_path=config_fpath)

    # Initialize Workflow Orchestrator
    sampling_design_pipeline = SamplingDesignPipeline(
        output_path=config.output_path,
        use_cached=config.use_cached,
        sampling_method=config.sampling.sampling_method,
        cache_path=op.join(config.output_path, "preprocessed_data"),
    )

    # Run preprocessing, sampling design creation and saving
    sampling_design_pipeline.run(config=config)


if __name__ == "__main__":
    main()
