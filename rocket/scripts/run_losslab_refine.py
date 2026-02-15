"""PanddaMap refinement using LossLab with ROCKET models."""

from __future__ import annotations

import argparse
import os
import sys

import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from loguru import logger

from rocket import panddamap_pipeline as pipeline
from rocket.refinement_config import RocketRefinmentConfig


def run_panddamap_refinement(
    config: RocketRefinmentConfig | str,
    starting_bias: torch.Tensor | None = None,
    starting_weights: torch.Tensor | None = None,
    target_map_override=None,
) -> RocketRefinmentConfig:
    if isinstance(config, str):
        config = RocketRefinmentConfig.from_yaml_file(config)

    seed_value = pipeline.resolve_seed(config)
    pipeline.set_deterministic(seed_value)
    logger.info("Deterministic seeding enabled (seed={})", seed_value)
    inputs = pipeline.load_inputs(config, target_map_override=target_map_override)
    engine, output_dir, run_note = pipeline.build_engine(config, inputs)
    results, _, _ = pipeline.run_engine_with_predictor(
        config,
        inputs,
        engine,
        starting_bias=starting_bias,
        starting_weights=starting_weights,
        save_best_biases=True,
    )
    pipeline.log_results(results, output_dir, run_note)

    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PanddaMap refinement using ROCKET config",
    )
    parser.add_argument("config", type=str, help="Path to YAML config")
    args = parser.parse_args()
    config = RocketRefinmentConfig.from_yaml_file(args.config)
    logger.remove()
    log_level = "DEBUG" if config.execution.verbose else "INFO"
    logger.add(sys.stderr, level=log_level)
    starting_bias = None
    starting_weights = None
    target_map_override = None
    if config.panddamap.preprocess_target_map:
        logger.info("Preprocessing target map (denoise + ligand mask)...")
        target_map_override = pipeline.preprocess_target_map(config)
    if config.panddamap.mse_prepass.enabled:
        logger.info("Running MSE prepass refinement...")
        starting_bias, starting_weights = pipeline.run_mseloss_refinement(
            config,
            writeout=config.panddamap.mse_prepass.save_biases,
            target_map_override=target_map_override,
        )
    run_panddamap_refinement(
        config,
        starting_bias=starting_bias,
        starting_weights=starting_weights,
        target_map_override=target_map_override,
    )


if __name__ == "__main__":
    main()
