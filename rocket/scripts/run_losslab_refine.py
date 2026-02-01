"""PanddaMap refinement using LossLab with ROCKET models."""

from __future__ import annotations

import argparse
import os

import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from loguru import logger

from rocket import io as rk_io
from rocket import panddamap_pipeline as pipeline
from rocket.refinement_config import RocketRefinmentConfig


def run_panddamap_refinement(
    config: RocketRefinmentConfig | str,
    starting_bias: torch.Tensor | None = None,
    starting_weights: torch.Tensor | None = None,
) -> RocketRefinmentConfig:
    if isinstance(config, str):
        config = RocketRefinmentConfig.from_yaml_file(config)

    seed_value = pipeline.resolve_seed(config)
    pipeline.set_deterministic(seed_value)
    logger.info("Deterministic seeding enabled (seed={})", seed_value)
    inputs = pipeline.load_inputs(config)
    engine, output_dir, run_note = pipeline.build_engine(config, inputs)
    model = pipeline.build_model(inputs.device)
    device_features, feature_key, features_at_it_start, optimizer = (
        pipeline.build_features_and_optimizer(
            config,
            inputs,
            starting_bias=starting_bias,
            starting_weights=starting_weights,
        )
    )
    predictor = pipeline.build_predictor(
        model,
        device_features,
        features_at_it_start,
        inputs.input_pdb,
        feature_key,
        bias=True,
    )
    best_state_callback = rk_io.make_best_biases_saver(device_features, output_dir)
    results = engine.run(
        reference_coordinates=inputs.reference_coords,
        prediction_callback=predictor,
        optimizer=optimizer,
        best_state_callback=best_state_callback,
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
    starting_bias = None
    starting_weights = None
    if config.panddamap.run_mse_prepass:
        logger.info("Running MSE prepass refinement...")
        starting_bias, starting_weights = pipeline.run_mseloss_refinement(
            config,
            writeout=config.panddamap.save_mse_biases,
        )
        print("$$$$$$ starting_bias mean:", starting_bias.mean().item())
        print("$$$$ starting_weights mean:", starting_weights.mean().item())
    run_panddamap_refinement(
        config,
        starting_bias=starting_bias,
        starting_weights=starting_weights,
    )


if __name__ == "__main__":
    main()
