"""Pipeline helpers for PanddaMap LossLab refinement."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path

import gemmi
import numpy as np
import SFC_Torch as sfc
import torch
from loguru import logger
from LossLab import RealSpaceLoss, RefinementConfig, RefinementEngine
from LossLab.losses.mse import MSECoordinatesLoss
from LossLab.refinement.trajectory import TrajectoryWriter
from LossLab.utils.map_utils import (
    denoise_and_mask_ccp4_map,
    parse_pdb_coords,
)
from openfold.config import model_config

import rocket
from rocket import io as rk_io
from rocket import refinement_utils as rkrf_utils
from rocket import utils as rk_utils
from rocket.losslab_predictor import OpenFoldPredictor, PredictorConfig
from rocket.refinement_config import RocketRefinmentConfig


@dataclass
class RefinementInputs:
    device: str
    base_dir: Path
    input_dir: Path
    target_map_path: Path
    target_map: object
    reference_pdb_path: Path
    reference_pdb: object
    reference_coords: torch.Tensor


def resolve_seed(config: RocketRefinmentConfig, default: int = 1) -> int:
    return int(getattr(config.execution, "seed", default))


def set_deterministic(seed: int) -> None:
    import os
    import random
    import warnings

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    warnings.filterwarnings(
        "ignore",
        message=("Deterministic behavior was enabled.*CUBLAS_WORKSPACE_CONFIG.*"),
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=("torch.utils.checkpoint: please pass in use_reentrant=.*"),
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="None of the inputs have requires_grad=True.*",
        category=UserWarning,
    )


def preprocess_target_map(config: RocketRefinmentConfig) -> gemmi.Ccp4Map:
    target_map_path = rk_io.resolve_target_map(config)
    target_map = rk_io.load_target_map(target_map_path)
    if not config.panddamap.preprocess_target_map:
        return target_map
    if not config.paths.ligand_pdb:
        raise ValueError("paths.ligand_pdb is required for preprocessing")
    ligand_coords = parse_pdb_coords(config.paths.ligand_pdb)
    return denoise_and_mask_ccp4_map(
        target_map,
        ligand_coords,
        high_res_limit=config.panddamap.denoise_high_res_limit,
        mask_radius=config.panddamap.ligand_mask_radius,
        tv_denoise=config.panddamap.tv_denoise,
    )


def load_inputs(
    config: RocketRefinmentConfig,
    target_map_override: gemmi.Ccp4Map | None = None,
) -> RefinementInputs:
    device = f"cuda:{config.execution.cuda_device}"
    base_dir = Path(config.paths.path)
    input_dir = Path(config.paths.input_dir or config.paths.path)

    target_map_path = rk_io.resolve_target_map(config)
    target_map = target_map_override or rk_io.load_target_map(target_map_path)

    reference_pdb_path = rk_io.resolve_input_pdb(config)
    reference_pdb = rk_io.load_input_pdb(reference_pdb_path, target_map)

    reference_coords = torch.tensor(
        reference_pdb.atom_pos,
        device=device,
        dtype=torch.float32,
    )

    return RefinementInputs(
        device=device,
        base_dir=base_dir,
        input_dir=input_dir,
        target_map_path=target_map_path,
        target_map=target_map,
        reference_pdb_path=reference_pdb_path,
        reference_pdb=reference_pdb,
        reference_coords=reference_coords,
    )


def build_structure_factor_calculator(
    moving_pdb,
    target_map,
    device: str,
    dmin: float,
) -> sfc.SFcalculator:
    structure_factor_calc = sfc.SFcalculator(
        moving_pdb,
        dmin=dmin,
        mode="xray",
        device=device,
    )
    structure_factor_calc.inspect_data()
    structure_factor_calc.gridsize = [
        target_map.grid.nu,
        target_map.grid.nv,
        target_map.grid.nw,
    ]
    return structure_factor_calc


def build_loss_function(
    target_map,
    moving_pdb,
    device: str,
    loss_type: str,
    mask_center: np.ndarray | None = None,
    mask_radius: float | None = None,
) -> RealSpaceLoss:
    return RealSpaceLoss(
        target_map=target_map,
        pdb_obj=moving_pdb,
        device=device,
        loss_type=loss_type,
        mask_center=mask_center,
        mask_radius=mask_radius,
    )


def build_engine(
    config: RocketRefinmentConfig,
    inputs: RefinementInputs,
    moving_pdb=None,
) -> tuple[RefinementEngine, Path, str]:
    output_dir, run_note = rk_io.get_output_dir_and_note(config, inputs.base_dir)

    base_tags = list(config.panddamap.wandb_tags or [])
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
        wandb_tags=base_tags + ["realspace"],
        wandb_notes=config.panddamap.wandb_notes,
    )

    if config.algorithm.bias_version != 3:
        raise ValueError("Only bias_version=3 is supported.")

    if moving_pdb is None:
        moving_pdb = inputs.reference_pdb

    if config.data.min_resolution is None:
        raise ValueError("data.min_resolution is required")
    dmin = config.data.min_resolution
    structure_factor_calc = build_structure_factor_calculator(
        moving_pdb,
        inputs.target_map,
        inputs.device,
        dmin=dmin,
    )
    mask_center = None
    if config.panddamap.ligand_centroid:
        mask_center = np.array(config.panddamap.ligand_centroid, dtype=float)
    else:
        logger.warning("panddamap.ligand_centroid is null; mask center disabled")
    mask_radius = config.panddamap.pandda_map_radius
    if mask_radius is None:
        logger.warning("panddamap.pandda_map_radius is null; defaulting to 15.0")
        mask_radius = 15.0
    loss_fn = build_loss_function(
        inputs.target_map,
        moving_pdb,
        inputs.device,
        config.panddamap.loss_type,
        mask_center=mask_center,
        mask_radius=mask_radius,
    )
    engine = RefinementEngine(
        config=losslab_config,
        loss_function=loss_fn,
        structure_factor_calculator=structure_factor_calc,
        pdb_template=str(inputs.reference_pdb_path),
    )
    return engine, output_dir, run_note


def build_model(device: str, preset: str = "model_1_ptm"):
    af_model = rocket.MSABiasAFv3(
        model_config(preset, train=True),
        preset,
    ).to(device)
    af_model.freeze()
    return af_model


def build_features_and_optimizer(
    config: RocketRefinmentConfig,
    inputs: RefinementInputs,
    preset: str = "model_1_ptm",
    starting_bias: torch.Tensor | str | None = None,
    starting_weights: torch.Tensor | str | None = None,
):
    fasta_path = rk_io.resolve_input_fasta(config)
    alignment_dir = rk_io.resolve_alignment_dir(config)
    raw_feature_dict = None
    feature_processor = None
    processed_features_unbiased = None

    if fasta_path and alignment_dir:
        device_features, raw_feature_dict, feature_processor = (
            rkrf_utils.build_processed_features_from_alignment(
                fasta_path=str(fasta_path),
                alignment_dir=str(alignment_dir),
                preset=preset,
                device=inputs.device,
                max_recycling_iters=config.algorithm.init_recycling,
            )
        )
        processed_features_unbiased = {
            k: (v.detach().clone() if torch.is_tensor(v) else np.array(v))
            for k, v in device_features.items()
        }
        feature_key = "msa_feat"
        features_at_it_start = device_features[feature_key].detach().clone()
    else:
        device_features, feature_key, features_at_it_start = (
            rkrf_utils.init_processed_dict(
                bias_version=config.algorithm.bias_version,
                path=str(inputs.input_dir),
                device=inputs.device,
                target_seq=None,
                PRESET=preset,
                processed_feats_path=config.paths.msa_feat_init_path,
            )
        )
        processed_features_unbiased = {
            k: (v.detach().clone() if torch.is_tensor(v) else np.array(v))
            for k, v in device_features.items()
        }

    resolved_bias = (
        starting_bias
        if starting_bias is not None
        else rk_io.resolve_starting_path(config.paths.starting_bias, inputs.base_dir)
    )
    resolved_weights = (
        starting_weights
        if starting_weights is not None
        else rk_io.resolve_starting_path(config.paths.starting_weights, inputs.base_dir)
    )

    device_features, optimizer, _ = rkrf_utils.init_bias(
        device_processed_features=device_features,
        bias_version=config.algorithm.bias_version,
        device=inputs.device,
        lr_a=config.algorithm.optimization.additive_learning_rate,
        lr_m=config.algorithm.optimization.multiplicative_learning_rate,
        weight_decay=config.algorithm.optimization.weight_decay,
        starting_bias=resolved_bias,
        starting_weights=resolved_weights,
    )

    return (
        device_features,
        feature_key,
        features_at_it_start,
        optimizer,
        processed_features_unbiased,
        raw_feature_dict,
        feature_processor,
    )


def build_predictor(
    model,
    device_features,
    features_at_it_start,
    moving_pdb,
    feature_key: str,
    bias: bool = True,
) -> OpenFoldPredictor:
    return OpenFoldPredictor(
        model,
        device_features,
        features_at_it_start,
        moving_pdb,
        config=PredictorConfig(feature_key=feature_key),
        bias=bias,
    )


def _update_engine_with_moving_pdb(
    engine: RefinementEngine,
    moving_pdb,
    target_map,
    device: str,
    dmin: float,
) -> None:
    engine.sfc = build_structure_factor_calculator(
        moving_pdb,
        target_map,
        device,
        dmin=dmin,
    )
    if hasattr(engine.loss_fn, "pdb_obj"):
        engine.loss_fn.pdb_obj = moving_pdb
        if hasattr(engine.loss_fn, "set_pdb_obj"):
            engine.loss_fn.set_pdb_obj(moving_pdb)
        with contextlib.suppress(Exception):
            engine.loss_fn.alignment_indices = np.arange(len(moving_pdb.atom_pos))


def run_engine_with_predictor(
    config: RocketRefinmentConfig,
    inputs: RefinementInputs,
    engine: RefinementEngine,
    starting_bias: torch.Tensor | str | None = None,
    starting_weights: torch.Tensor | str | None = None,
    save_best_biases: bool = False,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    model = build_model(inputs.device)
    (
        device_features,
        feature_key,
        features_at_it_start,
        optimizer,
        processed_features_unbiased,
        raw_feature_dict,
        feature_processor,
    ) = build_features_and_optimizer(
        config,
        inputs,
        starting_bias=starting_bias,
        starting_weights=starting_weights,
    )
    predictor = build_predictor(
        model,
        device_features,
        features_at_it_start,
        inputs.reference_pdb,
        feature_key,
        bias=True,
    )

    predictor()
    if raw_feature_dict is None or feature_processor is None:
        raise ValueError(
            "raw_feature_dict and feature_processor are required to write "
            "the initial PDB; provide fasta/alignment inputs."
        )
    moving_pdb_path = Path(engine.output_dir) / "initial_prediction_unrelaxed.pdb"
    rkrf_utils.init_processed_dict(
        bias_version=config.algorithm.bias_version,
        path=str(inputs.input_dir),
        device=inputs.device,
        PRESET="model_1_ptm",
        output_pdb_path=moving_pdb_path,
        outputs=getattr(predictor, "last_outputs", None),
        raw_feature_dict=raw_feature_dict,
        feature_processor=feature_processor,
        processed_feature_dict_override=processed_features_unbiased,
        multimer_ri_gap=200,
        subtract_plddt=False,
        write_pdb_only=True,
    )
    moving_pdb = rk_io.load_input_pdb(moving_pdb_path, inputs.target_map)
    if engine.config.save_best_pdb or engine.config.save_trajectory_pdb:
        engine.trajectory_writer = TrajectoryWriter(
            output_dir=engine.output_dir,
            pdb_template_path=moving_pdb_path,
            save_interval=engine.config.save_trajectory_interval,
            wandb_logger=engine.wandb_logger,
        )
    if config.data.min_resolution is None:
        raise ValueError("data.min_resolution is required")
    dmin = config.data.min_resolution
    _update_engine_with_moving_pdb(
        engine,
        moving_pdb,
        inputs.target_map,
        inputs.device,
        dmin=dmin,
    )
    predictor.pdb_obj = moving_pdb

    if isinstance(engine.loss_fn, MSECoordinatesLoss):
        engine.loss_fn.set_moving_pdb(moving_pdb)

    best_state: dict[str, torch.Tensor | None] = {
        "bias": None,
        "weights": None,
    }

    save_callback = (
        rk_io.make_best_biases_saver(device_features, Path(engine.output_dir))
        if save_best_biases
        else None
    )

    def _best_state_callback(*, run_id: str, iteration: int, loss: float) -> None:
        best_state["bias"] = device_features["msa_feat_bias"].detach().clone()
        best_state["weights"] = device_features["msa_feat_weights"].detach().clone()
        if save_callback is not None:
            save_callback(run_id=run_id, iteration=iteration, loss=loss)

    moving_reference_coords = torch.tensor(
        inputs.reference_pdb.atom_pos,
        device=inputs.device,
        dtype=torch.float32,
    )
    try:
        common_ref_idx, common_mov_idx = rkrf_utils.get_common_ca_ind(
            inputs.reference_pdb, moving_pdb
        )
        engine.alignment_indices_reference = np.array(common_ref_idx)
        engine.alignment_indices_moving = np.array(common_mov_idx)
    except Exception as exc:
        logger.warning("Failed to compute common CA indices: {}", exc)

    def _predict_with_pseudob():
        prediction = predictor()
        if prediction is None or prediction.shape[-1] < 4:
            return prediction
        coords = prediction[:, :3]
        conf = prediction[:, 3]
        pseudo_b = rk_utils.plddt2pseudoB_pt(conf)
        return torch.cat([coords, pseudo_b.unsqueeze(-1)], dim=-1)

    results = engine.run(
        reference_coordinates=moving_reference_coords,
        prediction_callback=_predict_with_pseudob,
        optimizer=optimizer,
        best_state_callback=_best_state_callback,
    )

    bias_state = best_state.get("bias")
    weights_state = best_state.get("weights")

    bias_tensor = (
        bias_state
        if bias_state is not None
        else device_features["msa_feat_bias"].detach().clone()
    )
    weights_tensor = (
        weights_state
        if weights_state is not None
        else device_features["msa_feat_weights"].detach().clone()
    )
    return results, bias_tensor, weights_tensor


def log_results(results: dict, output_dir: Path, run_note: str) -> None:
    logger.info("Refinement complete.")
    logger.info("Best loss: {:.6f}", results["loss"])
    logger.info("Best run: {}", results["run_id"])
    logger.info("Best iteration: {}", results["iteration"])
    logger.info("Output directory: {}", output_dir / run_note)


def run_mseloss_refinement(
    config: RocketRefinmentConfig,
    writeout: bool = False,
    target_map_override: gemmi.Ccp4Map | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    seed_value = resolve_seed(config)
    set_deterministic(seed_value)
    logger.info("Deterministic seeding enabled (seed={})", seed_value)

    inputs = load_inputs(config, target_map_override=target_map_override)
    output_dir = (
        Path(config.panddamap.output_dir)
        if Path(config.panddamap.output_dir).is_absolute()
        else inputs.base_dir / config.panddamap.output_dir
    )
    run_note = f"{config.note or config.panddamap.run_note or 'panddamap'}_mse"

    base_tags = list(config.panddamap.wandb_tags or [])
    losslab_config = RefinementConfig(
        num_iterations=config.algorithm.iterations,
        num_runs=config.execution.num_of_runs,
        learning_rate_additive=config.algorithm.optimization.additive_learning_rate,
        learning_rate_multiplicative=(
            config.algorithm.optimization.multiplicative_learning_rate
        ),
        loss_type="mse",
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
        wandb_tags=base_tags + ["mse"],
        wandb_notes=config.panddamap.wandb_notes,
    )

    loss_fn = MSECoordinatesLoss(
        reference_coordinates=inputs.reference_coords,
        device=inputs.device,
        reference_pdb=inputs.reference_pdb,
        moving_pdb=inputs.reference_pdb,
        selection=config.panddamap.mse_selection,
    )

    engine = RefinementEngine(
        config=losslab_config,
        loss_function=loss_fn,
        structure_factor_calculator=None,
        pdb_template=str(inputs.reference_pdb_path),
    )

    results, bias_tensor, weights_tensor = run_engine_with_predictor(
        config,
        inputs,
        engine,
        save_best_biases=writeout,
    )
    log_results(results, output_dir, run_note)
    return bias_tensor, weights_tensor
