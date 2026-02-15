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
from LossLab.utils.geometry import iterative_kabsch_alignment, kabsch_align
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


@dataclass
class MaskState:
    mask_center: torch.Tensor | None
    atom_grad_mask: torch.Tensor | None
    mse_atom_mask: torch.Tensor | None


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


def build_first_n_residue_atom_mask(
    moving_pdb,
    first_n_residues: int | None,
    device: str,
) -> torch.Tensor | None:
    if not first_n_residues or first_n_residues <= 0:
        return None
    if not hasattr(moving_pdb, "cra_name"):
        logger.warning("moving_pdb has no cra_name; cannot build MSE mask.")
        return None

    mask_values: list[float] = []
    for cra_name in moving_pdb.cra_name:
        try:
            parts = str(cra_name).split("-", 3)
            resid = int(parts[1])
        except Exception:
            logger.warning("Failed to parse residue id from cra_name=%s", cra_name)
            return None
        mask_values.append(1.0 if resid < first_n_residues else 0.0)

    mask = torch.tensor(mask_values, device=device, dtype=torch.float32).unsqueeze(-1)
    logger.info(
        "Applying MSE first-N residue mask ({} of {} atoms within residues < {}).",
        int(mask.sum().item()),
        int(mask.numel()),
        first_n_residues,
    )
    return mask


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
    reference_coords: torch.Tensor | None = None,
    reference_pdb=None,
    mse_selection: str | None = None,
    mask_center: np.ndarray | None = None,
    mask_radius: float | None = None,
) -> RealSpaceLoss | MSECoordinatesLoss:
    if loss_type == "mse":
        if reference_coords is None or reference_pdb is None:
            raise ValueError("reference_coords and reference_pdb are required for MSE")
        return MSECoordinatesLoss(
            reference_coordinates=reference_coords,
            device=device,
            reference_pdb=reference_pdb,
            moving_pdb=moving_pdb or reference_pdb,
            selection=mse_selection or "BB",
        )
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
    structure_factor_calc = None
    if config.panddamap.loss_type != "mse":
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
        reference_coords=inputs.reference_coords,
        reference_pdb=inputs.reference_pdb,
        mse_selection=config.panddamap.mse_prepass.selection,
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


def build_model(
    device: str,
    preset: str = "model_1_ptm",
    use_deepspeed_evo_attention: bool = True,
):
    af_model = rocket.MSABiasAFv3(
        model_config(preset, train=True),
        preset,
        use_deepspeed_evo_attention=use_deepspeed_evo_attention,
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
    offload_activations: bool = False,
    bias: bool = True,
) -> OpenFoldPredictor:
    return OpenFoldPredictor(
        model,
        device_features,
        features_at_it_start,
        moving_pdb,
        config=PredictorConfig(
            feature_key=feature_key,
            offload_activations=offload_activations,
        ),
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


def _is_mse_loss(engine: RefinementEngine) -> bool:
    return isinstance(engine.loss_fn, MSECoordinatesLoss)


def _build_predictor_context(
    config: RocketRefinmentConfig,
    inputs: RefinementInputs,
    starting_bias: torch.Tensor | str | None,
    starting_weights: torch.Tensor | str | None,
):
    model = build_model(
        inputs.device,
        use_deepspeed_evo_attention=config.algorithm.use_deepspeed_evo_attention,
    )
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
        offload_activations=config.algorithm.optimization.offload_activations,
        bias=True,
    )
    return (
        model,
        predictor,
        optimizer,
        device_features,
        processed_features_unbiased,
        raw_feature_dict,
        feature_processor,
    )


def _write_initial_prediction(
    config: RocketRefinmentConfig,
    inputs: RefinementInputs,
    engine: RefinementEngine,
    predictor: OpenFoldPredictor,
    processed_features_unbiased,
    raw_feature_dict,
    feature_processor,
):
    predictor(map_to_pdb=False)
    prediction_outputs = getattr(predictor, "last_outputs", None)
    if prediction_outputs is None:
        raise ValueError(
            "Predictor produced no outputs; cannot write "
            "initial_prediction_unrelaxed.pdb."
        )
    if raw_feature_dict is None or feature_processor is None:
        raise ValueError(
            "raw_feature_dict and feature_processor are required to "
            "write the initial PDB; provide fasta/alignment inputs."
        )
    moving_pdb_path = Path(engine.output_dir) / "initial_prediction_unrelaxed.pdb"
    rkrf_utils.init_processed_dict(
        bias_version=config.algorithm.bias_version,
        path=str(inputs.input_dir),
        device=inputs.device,
        PRESET="model_1_ptm",
        output_pdb_path=moving_pdb_path,
        outputs=prediction_outputs,
        raw_feature_dict=raw_feature_dict,
        feature_processor=feature_processor,
        processed_feature_dict_override=processed_features_unbiased,
        multimer_ri_gap=200,
        subtract_plddt=False,
        write_pdb_only=True,
    )
    moving_pdb = rk_io.load_input_pdb(moving_pdb_path, inputs.target_map)
    return moving_pdb, moving_pdb_path


def _configure_engine(
    config: RocketRefinmentConfig,
    inputs: RefinementInputs,
    engine: RefinementEngine,
    moving_pdb,
):
    if engine.config.save_best_pdb or engine.config.save_trajectory_pdb:
        engine.trajectory_writer = TrajectoryWriter(
            output_dir=engine.output_dir,
            pdb_template_path=str(
                Path(engine.output_dir) / "initial_prediction_unrelaxed.pdb"
            ),
            save_interval=engine.config.save_trajectory_interval,
            wandb_logger=engine.wandb_logger,
        )
    if config.data.min_resolution is None:
        raise ValueError("data.min_resolution is required")
    dmin = config.data.min_resolution
    if not _is_mse_loss(engine):
        _update_engine_with_moving_pdb(
            engine,
            moving_pdb,
            inputs.target_map,
            inputs.device,
            dmin=dmin,
        )
    else:
        engine.sfc = None
        engine.loss_fn.set_moving_pdb(moving_pdb)
    return dmin


def _write_atom_grad_mask_ccp4(
    atom_positions: np.ndarray,
    atom_grad_mask: np.ndarray,
    target_map: gemmi.Ccp4Map,
    output_path: Path,
) -> None:
    """Write atom gradient mask as CCP4 map for visualization."""
    try:
        grid = gemmi.FloatGrid(
            target_map.grid.nu,
            target_map.grid.nv,
            target_map.grid.nw,
        )
        grid.set_unit_cell(target_map.grid.unit_cell)
        grid.spacegroup = target_map.grid.spacegroup

        # Set grid values based on proximity to masked atoms
        for _i, (pos, mask_val) in enumerate(
            zip(atom_positions, atom_grad_mask, strict=False)
        ):
            if mask_val > 0.5:  # Atom is masked (receives gradients)
                frac = grid.unit_cell.fractionalize(gemmi.Position(*pos))
                grid.set_value(
                    int(frac.x * grid.nu),
                    int(frac.y * grid.nv),
                    int(frac.z * grid.nw),
                    1.0,
                )

        # Smooth the mask slightly for visualization
        try:
            from scipy.ndimage import gaussian_filter

            grid_array = np.array(grid, copy=False)
            grid_array[:] = gaussian_filter(grid_array, sigma=1.5)
        except ImportError:
            logger.warning("scipy not available; mask will not be smoothed")

        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = grid
        ccp4.update_ccp4_header(2, True)
        ccp4.write_ccp4_map(str(output_path))
        logger.info("Wrote atom gradient mask to {}", output_path)
    except Exception as exc:
        logger.warning("Failed to write atom gradient mask CCP4: {}", exc)


def _compute_mask_state(
    config: RocketRefinmentConfig,
    inputs: RefinementInputs,
    engine: RefinementEngine,
    moving_pdb,
    moving_reference_coords: torch.Tensor,
):
    atom_grad_mask = None
    mask_center_tensor = None
    if config.panddamap.ligand_centroid and config.panddamap.pandda_map_radius:
        ligand_center = torch.tensor(
            config.panddamap.ligand_centroid,
            device=inputs.device,
            dtype=torch.float32,
        )
        mask_center_tensor = ligand_center
        atom_pos_source = (
            moving_pdb.atom_pos_orth
            if hasattr(moving_pdb, "atom_pos_orth")
            else moving_pdb.atom_pos
        )
        atom_pos = torch.tensor(
            atom_pos_source,
            device=inputs.device,
            dtype=torch.float32,
        )
        try:
            common_ref_idx, common_mov_idx = rkrf_utils.get_common_ca_ind(
                inputs.reference_pdb, moving_pdb
            )
            engine.alignment_indices_reference = np.array(common_ref_idx)
            engine.alignment_indices_moving = np.array(common_mov_idx)
            atom_pos = kabsch_align(
                atom_pos,
                moving_reference_coords,
                indices_moving=np.array(common_mov_idx),
                indices_reference=np.array(common_ref_idx),
            )
            P = atom_pos[common_mov_idx]
            Q = moving_reference_coords[common_ref_idx]
            R, t, _ = iterative_kabsch_alignment(P, Q, torch_backend=True, max_iters=5)
            mask_center_tensor = ligand_center @ R + t
        except Exception as exc:
            logger.warning("Failed to compute Kabsch alignment for mask: {}", exc)

        distances = torch.norm(atom_pos - mask_center_tensor, dim=-1)
        atom_grad_mask = (distances <= config.panddamap.pandda_map_radius).float()
        atom_grad_mask = atom_grad_mask.unsqueeze(-1)
        logger.info(
            "Applying atom gradient mask ({} of {} atoms within radius).",
            int(atom_grad_mask.sum().item()),
            int(atom_grad_mask.numel()),
        )
        logger.info(
            "Gradient mask centroid={}, radius={}",
            mask_center_tensor.tolist() if mask_center_tensor is not None else None,
            config.panddamap.pandda_map_radius,
        )

        # Write mask as CCP4 for visualization
        output_dir = Path(engine.output_dir)
        _write_atom_grad_mask_ccp4(
            atom_pos.detach().cpu().numpy(),
            atom_grad_mask.detach().cpu().numpy().squeeze(-1),
            inputs.target_map,
            output_dir / "atom_grad_mask.ccp4",
        )

        if isinstance(engine.loss_fn, RealSpaceLoss):
            engine.loss_fn.set_mask(
                mask_center_tensor.detach().cpu().numpy(),
                float(config.panddamap.pandda_map_radius),
            )

    mse_atom_mask = None
    if _is_mse_loss(engine):
        mse_atom_mask = build_first_n_residue_atom_mask(
            moving_pdb,
            config.panddamap.mse_prepass.first_n_residues,
            inputs.device,
        )

    return MaskState(
        mask_center=mask_center_tensor,
        atom_grad_mask=atom_grad_mask,
        mse_atom_mask=mse_atom_mask,
    )


def run_engine_with_predictor(
    config: RocketRefinmentConfig,
    inputs: RefinementInputs,
    engine: RefinementEngine,
    starting_bias: torch.Tensor | str | None = None,
    starting_weights: torch.Tensor | str | None = None,
    save_best_biases: bool = False,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    (
        _model,
        predictor,
        optimizer,
        device_features,
        processed_features_unbiased,
        raw_feature_dict,
        feature_processor,
    ) = _build_predictor_context(
        config,
        inputs,
        starting_bias,
        starting_weights,
    )

    moving_pdb, moving_pdb_path = _write_initial_prediction(
        config,
        inputs,
        engine,
        predictor,
        processed_features_unbiased,
        raw_feature_dict,
        feature_processor,
    )
    _configure_engine(config, inputs, engine, moving_pdb)
    predictor.pdb_obj = moving_pdb

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
    mask_state = _compute_mask_state(
        config,
        inputs,
        engine,
        moving_pdb,
        moving_reference_coords,
    )

    def _predict_with_pseudob():
        prediction = predictor()
        if prediction is None or prediction.shape[-1] < 4:
            return prediction
        coords = prediction[:, :3]
        conf = prediction[:, 3]
        if (
            mask_state.atom_grad_mask is not None
            and mask_state.atom_grad_mask.shape[0] == coords.shape[0]
        ):
            coords = coords * mask_state.atom_grad_mask + coords.detach() * (
                1.0 - mask_state.atom_grad_mask
            )
        if (
            mask_state.mse_atom_mask is not None
            and mask_state.mse_atom_mask.shape[0] == coords.shape[0]
        ):
            coords = coords * mask_state.mse_atom_mask + coords.detach() * (
                1.0 - mask_state.mse_atom_mask
            )
        pseudo_b = rk_utils.plddt2pseudoB_pt(conf)
        return torch.cat([coords, pseudo_b.unsqueeze(-1)], dim=-1)

    if torch.cuda.is_available() and str(inputs.device).startswith("cuda"):
        with contextlib.suppress(Exception):
            torch.cuda.reset_peak_memory_stats()

    results = engine.run(
        reference_coordinates=moving_reference_coords,
        prediction_callback=_predict_with_pseudob,
        optimizer=optimizer,
        best_state_callback=_best_state_callback,
    )

    if torch.cuda.is_available() and str(inputs.device).startswith("cuda"):
        try:
            peak_mem = torch.cuda.max_memory_allocated()
            logger.info("Peak CUDA memory allocated: {:.2f} GB", peak_mem / 1e9)
        except Exception as exc:
            logger.warning("Failed to read peak CUDA memory: {}", exc)

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
        selection=config.panddamap.mse_prepass.selection,
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
