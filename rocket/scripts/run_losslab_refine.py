"""PanddaMap refinement using LossLab with ROCKET models."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import gemmi
import numpy as np
import SFC_Torch as sfc
import torch
from loguru import logger
from LossLab import RealSpaceLoss, RefinementConfig, RefinementEngine
from openfold.config import model_config
from SFC_Torch import PDBParser

import rocket
from rocket import refinement_utils as rkrf_utils
from rocket import utils as rk_utils
from rocket.losslab_predictor import OpenFoldPredictor, PredictorConfig
from rocket.refinement_config import RocketRefinmentConfig


def _set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _resolve_input_pdb(config: RocketRefinmentConfig) -> Path:
    if config.paths.input_pdb:
        return Path(config.paths.input_pdb)
    input_dir = Path(config.paths.input_dir or config.paths.path)
    return input_dir / f"{config.paths.file_id}-pred-aligned.pdb"


def _resolve_input_fasta(config: RocketRefinmentConfig) -> Path | None:
    if config.paths.input_fasta:
        return Path(config.paths.input_fasta)
    input_dir = Path(config.paths.path)
    for pattern in ["*.fasta", "*.fa", "*.faa"]:
        matches = sorted(input_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _resolve_alignment_dir(config: RocketRefinmentConfig) -> Path | None:
    if config.paths.alignment_dir:
        return Path(config.paths.alignment_dir)
    input_dir = Path(config.paths.path)
    for candidate in ["alignments", "alignment", "msa"]:
        dir_path = input_dir / candidate
        if dir_path.exists() and dir_path.is_dir():
            return dir_path
    return None


def _resolve_target_map(config: RocketRefinmentConfig) -> Path:
    if config.paths.target_map:
        return Path(config.paths.target_map)

    input_dir = Path(config.paths.input_dir or config.paths.path)

    search_roots = [
        input_dir,
    ]
    patterns = [
        f"*{config.paths.file_id}*masked*.ccp4",
        f"*{config.paths.file_id}*masked*.map",
        "*masked*.ccp4",
        "*masked*.map",
        "*likelihood_weighted*.map",
    ]
    for root in search_roots:
        for pattern in patterns:
            matches = sorted(root.glob(pattern))
            if matches:
                return matches[0]
    raise FileNotFoundError(
        "Target map not found. Set paths.target_map in the YAML config."
    )


def _resolve_starting_path(path_value: str | None, base_dir: Path) -> str | None:
    if not path_value:
        return None
    if os.path.isabs(path_value):
        return path_value
    return str(base_dir / path_value)


def run_panddamap_refinement(
    config: RocketRefinmentConfig | str,
) -> RocketRefinmentConfig:
    if isinstance(config, str):
        config = RocketRefinmentConfig.from_yaml_file(config)

    seed_value = getattr(config.execution, "seed", None)
    if seed_value is None:
        seed_value = 1
    _set_deterministic(int(seed_value))
    logger.info("Deterministic seeding enabled (seed=%s)", seed_value)

    device = f"cuda:{config.execution.cuda_device}"
    base_dir = Path(config.paths.path)
    input_dir = Path(config.paths.input_dir or config.paths.path)

    target_map_path = _resolve_target_map(config)
    target_map = gemmi.read_ccp4_map(str(target_map_path))
    target_map.setup(0.0)

    input_pdb_path = _resolve_input_pdb(config)
    input_pdb = PDBParser(str(input_pdb_path))
    input_pdb.set_spacegroup("P 1")
    input_pdb.set_unitcell(target_map.grid.unit_cell)

    structure_factor_calc = sfc.SFcalculator(
        input_pdb,
        dmin=1.8,
        mode="xray",
        device=device,
    )
    structure_factor_calc.inspect_data()
    structure_factor_calc.gridsize = [
        target_map.grid.nu,
        target_map.grid.nv,
        target_map.grid.nw,
    ]

    loss_fn = RealSpaceLoss(
        target_map=target_map,
        pdb_obj=input_pdb,
        device=device,
        loss_type=config.panddamap.loss_type,
        mask_center=None,
        mask_radius=None,
    )

    output_dir = (
        Path(config.panddamap.output_dir)
        if Path(config.panddamap.output_dir).is_absolute()
        else base_dir / config.panddamap.output_dir
    )
    run_note = config.note or config.panddamap.run_note or "panddamap"

    losslab_config = RefinementConfig(
        num_iterations=config.algorithm.iterations,
        num_runs=config.execution.num_of_runs,
        learning_rate_additive=config.algorithm.optimization.additive_learning_rate,
        learning_rate_multiplicative=(
            config.algorithm.optimization.multiplicative_learning_rate
        ),
        loss_type=config.panddamap.loss_type,
        output_dir=str(output_dir),
        run_note=run_note,
        save_every_n_iterations=config.panddamap.save_every_n_iterations,
        early_stopping_patience=config.panddamap.early_stopping_patience,
        save_best_pdb=config.panddamap.save_best_pdb,
        save_trajectory_pdb=config.panddamap.save_trajectory_pdb,
        save_trajectory_interval=config.panddamap.save_trajectory_interval,
        use_wandb=config.panddamap.use_wandb,
        wandb_entity=config.panddamap.wandb_entity,
        wandb_project=config.panddamap.wandb_project,
        wandb_name=config.panddamap.wandb_name,
        wandb_tags=config.panddamap.wandb_tags,
        wandb_notes=config.panddamap.wandb_notes,
    )

    if config.algorithm.bias_version != 3:
        raise ValueError("Only bias_version=3 is supported.")

    engine = RefinementEngine(
        config=losslab_config,
        loss_function=loss_fn,
        structure_factor_calculator=structure_factor_calc,
        pdb_template=str(input_pdb_path),
    )

    def _save_best_biases(*, run_id: str, iteration: int, loss: float) -> None:
        output_dir = Path(engine.output_dir)
        bias_tensor = device_features["msa_feat_bias"].detach().cpu().clone()
        weight_tensor = device_features["msa_feat_weights"].detach().cpu().clone()
        bias_path = output_dir / "best_msa_bias.pt"
        weights_path = output_dir / "best_feat_weights.pt"
        torch.save(bias_tensor, bias_path)
        torch.save(weight_tensor, weights_path)

    reference_coords = torch.tensor(
        input_pdb.atom_pos,
        device=device,
        dtype=torch.float32,
    )

    preset = "model_1_ptm"
    af_model = rocket.MSABiasAFv3(
        model_config(preset, train=True),
        preset,
    ).to(device)
    af_model.freeze()
    afconfig = model_config(preset)
    afconfig.data.common.max_recycling_iters = config.init_recycling
    del afconfig.data.common.masked_msa
    afconfig.data.common.resample_msa_in_recycling = False
    from openfold.data import data_pipeline, feature_pipeline

    feature_processor = feature_pipeline.FeaturePipeline(afconfig.data)
    data_processor = data_pipeline.DataPipeline(template_featurizer=None)

    fasta_path = _resolve_input_fasta(config)
    alignment_dir = _resolve_alignment_dir(config)

    if fasta_path and alignment_dir:
        print("I AM IN 1")
        print(fasta_path)
        print(alignment_dir)
        fullmsa_feature_dict = rkrf_utils.generate_feature_dict(
            fasta_path,
            alignment_dir,
            data_processor,
        )
        msa_processed_feature_dict = feature_processor.process_features(
            fullmsa_feature_dict, mode="predict"
        )
        device_features = rk_utils.move_tensors_to_device(
            msa_processed_feature_dict, device=device
        )
        print("shape of msa_feat:", device_features["msa_feat"].shape)
        logger.info("Feature keys: {}", sorted(device_features.keys()))
        if "msa_feat" in device_features:
            logger.info(
                "msa_feat shape: {}",
                tuple(device_features["msa_feat"].shape),
            )
        else:
            logger.warning("No msa_feat found in device_features.")
        feature_key = "msa_feat"
        features_at_it_start = device_features[feature_key].detach().clone()
        print("features at it start mean:", features_at_it_start.mean())
    else:
        template_pdb_name = (
            Path(config.paths.template_pdb).name
            if config.paths.template_pdb
            else input_pdb_path.name
        )
        print("I AM IN 2")
        device_features, feature_key, features_at_it_start = (
            rkrf_utils.init_processed_dict(
                bias_version=config.algorithm.bias_version,
                path=str(input_dir),
                device=device,
                template_pdb=template_pdb_name,
                target_seq=None,
                PRESET=preset,
                processed_feats_path=config.paths.msa_feat_init_path,
            )
        )
        print("shape of msa_feat:", device_features["msa_feat"].shape)
        logger.info("Feature keys: {}", sorted(device_features.keys()))
        print("features at it start mean:", features_at_it_start.mean())

    device_features, optimizer, _ = rkrf_utils.init_bias(
        device_processed_features=device_features,
        bias_version=config.algorithm.bias_version,
        device=device,
        lr_a=config.algorithm.optimization.additive_learning_rate,
        lr_m=config.algorithm.optimization.multiplicative_learning_rate,
        weight_decay=config.algorithm.optimization.weight_decay,
        starting_bias=_resolve_starting_path(config.paths.starting_bias, base_dir),
        starting_weights=_resolve_starting_path(
            config.paths.starting_weights, base_dir
        ),
    )

    print("$$$$$$ FIRST TIME GETTING TO PREDICTOR $$$$$$")
    predictor = OpenFoldPredictor(
        af_model,
        device_features,
        features_at_it_start,
        input_pdb,
        config=PredictorConfig(feature_key=feature_key),
        bias=True,
    )
    print("$$$$$$ RIGHT AFTER GETTING TO PREDICTOR $$$$$$")

    results = engine.run(
        reference_coordinates=reference_coords,
        prediction_callback=predictor,
        optimizer=optimizer,
        best_state_callback=_save_best_biases,
    )

    logger.info("Refinement complete.")
    logger.info("Best loss: {:.6f}", results["loss"])
    logger.info("Best run: {}", results["run_id"])
    logger.info("Best iteration: {}", results["iteration"])
    logger.info("Output directory: {}", output_dir / run_note)

    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PanddaMap refinement using ROCKET config",
    )
    parser.add_argument("config", type=str, help="Path to YAML config")
    args = parser.parse_args()
    run_panddamap_refinement(args.config)


if __name__ == "__main__":
    main()
