"""PanddaMap refinement using LossLab with ROCKET models."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import gemmi
import SFC_Torch as sfc
import torch
from loguru import logger
from LossLab import RealSpaceLoss, RefinementConfig, RefinementEngine
from openfold.config import model_config
from SFC_Torch import PDBParser

import rocket
from rocket import refinement_utils as rkrf_utils
from rocket.losslab_predictor import OpenFoldPredictor, PredictorConfig
from rocket.refinement_config import RocketRefinmentConfig


def _resolve_input_pdb(config: RocketRefinmentConfig) -> Path:
    if config.paths.input_pdb:
        return Path(config.paths.input_pdb)
    input_dir = Path(config.paths.input_dir or config.paths.path)
    return input_dir / f"{config.paths.file_id}-pred-aligned.pdb"


def _resolve_input_fasta(config: RocketRefinmentConfig) -> Path | None:
    if config.paths.input_fasta:
        return Path(config.paths.input_fasta)
    input_dir = Path(config.paths.input_dir or config.paths.path)
    for pattern in ["*.fasta", "*.fa", "*.faa"]:
        matches = sorted(input_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _resolve_alignment_dir(config: RocketRefinmentConfig) -> Path | None:
    if config.paths.alignment_dir:
        return Path(config.paths.alignment_dir)
    input_dir = Path(config.paths.input_dir or config.paths.path)
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
    run_note = config.panddamap.run_note or config.note or "panddamap"

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

    engine = RefinementEngine(
        config=losslab_config,
        loss_function=loss_fn,
        structure_factor_calculator=structure_factor_calc,
        pdb_template=str(input_pdb_path),
    )

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

    fasta_path = _resolve_input_fasta(config)
    alignment_dir = _resolve_alignment_dir(config)

    if fasta_path and alignment_dir:
        if config.algorithm.bias_version == 4:
            raise ValueError("bias_version=4 requires template-based features.")
        device_features = rkrf_utils.build_processed_features_from_alignment(
            fasta_path=str(fasta_path),
            alignment_dir=str(alignment_dir),
            preset=preset,
            device=device,
            max_recycling_iters=config.algorithm.init_recycling,
            template_mmcif_dir=config.paths.template_mmcif_dir,
            kalign_binary_path=config.paths.kalign_binary_path,
            max_template_date=config.paths.max_template_date,
            template_max_hits=config.paths.template_max_hits,
            template_release_dates_path=config.paths.template_release_dates_path,
            template_obsolete_pdbs_path=config.paths.template_obsolete_pdbs_path,
        )
        feature_key = "msa_feat"
        features_at_it_start = device_features[feature_key].detach().clone()
    else:
        template_pdb_name = (
            Path(config.paths.template_pdb).name
            if config.paths.template_pdb
            else input_pdb_path.name
        )
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

    predictor = OpenFoldPredictor(
        af_model,
        device_features,
        features_at_it_start,
        input_pdb,
        config=PredictorConfig(feature_key=feature_key),
    )

    results = engine.run(
        reference_coordinates=reference_coords,
        prediction_callback=predictor,
        optimizer=optimizer,
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
