import argparse

from ..refinement_config import RocketRefinmentConfig
from .run_losslab_refine import run_panddamap_refinement


def run_refinement(config: RocketRefinmentConfig | str) -> RocketRefinmentConfig:
    if isinstance(config, str):
        config = RocketRefinmentConfig.from_yaml_file(config)

    return run_panddamap_refinement(config)


def cli_runrefine():
    parser = argparse.ArgumentParser(description="Run ROCKET refinement")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    args = parser.parse_args()
    run_refinement(args.config)
