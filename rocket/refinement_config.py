"""
This module contains the configuration classes for the ROCKET refinement pipeline.
The configuration is stored in a human-readable YAML file and
can be loaded into a RocketRefinmentConfig object.
"""

import os
import uuid
from enum import Enum
from typing import Any, ClassVar

import yaml
from pydantic import BaseModel, Field, field_validator


# Custom StrEnum implementation for Python < 3.11
class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return str(self)


class DATAMODE(StrEnum):
    PANDDAMAP = "panddamap"


# Path and file configuration
class PathConfig(BaseModel):
    path: str = ""
    input_dir: str = ""
    file_id: str = ""
    input_pdb: str = ""
    input_fasta: str | None = None
    alignment_dir: str | None = None
    template_mmcif_dir: str | None = None
    template_release_dates_path: str | None = None
    template_obsolete_pdbs_path: str | None = None
    kalign_binary_path: str | None = None
    max_template_date: str | None = None
    template_max_hits: int | None = None
    template_pdb: str | None = None
    target_map: str | None = None
    input_msa: str | None = None
    ligand_pdb: str | None = None
    sub_msa_path: str | None = None
    sub_delmat_path: str | None = None
    msa_feat_init_path: str | None = None
    starting_bias: str | None = None
    starting_weights: str | None = None
    uuid_hex: str | None = None


# Hardware and execution configuration
class ExecutionConfig(BaseModel):
    seed: int = 1
    cuda_device: int = 0
    num_of_runs: int = 1
    verbose: bool = False


# Optimization parameters
class OptimizationParams(BaseModel):
    additive_learning_rate: float = 1e-3
    multiplicative_learning_rate: float = 1e-3
    weight_decay: float | None = 0.0001
    batch_sub_ratio: float = 0.7
    number_of_batches: int = 1
    rbr_opt_algorithm: str = "lbfgs"
    rbr_lbfgs_learning_rate: float = 150.0
    smooth_stage_epochs: int | None = 50
    phase2_final_lr: float = 1e-3
    l2_weight: float = 1e-7


# Feature flags
class FeatureFlags(BaseModel):
    solvent: bool = True
    sfc_scale: bool = True
    refine_sigmaA: bool = True
    additional_chain: bool = False
    total_chain_copy: float = 1.0
    bias_from_fullmsa: bool = False
    chimera_profile: bool = False


# Algorithm parameters
class AlgorithmConfig(BaseModel):
    bias_version: int = 3
    iterations: int = 100
    init_recycling: int = 1
    domain_segs: list[int] | None = None
    optimization: OptimizationParams = Field(default_factory=OptimizationParams)
    features: FeatureFlags = Field(default_factory=FeatureFlags)


# Data-specific configuration
class DataConfig(BaseModel):
    datamode: DATAMODE = "panddamap"
    free_flag: str = "R-free-flags"
    testset_value: int = 1
    min_resolution: float | None = None
    max_resolution: float | None = None
    voxel_spacing: float = 4.5
    msa_subratio: float | None = None
    w_plddt: float = 0.0
    downsample_ratio: int | None = None

    @field_validator("datamode", mode="before")
    @classmethod
    def validate_datamode(cls, v: str) -> "DATAMODE":
        """
        Validates and converts the input to a DATAMODE enum member.

        Raises:
            ValueError: If `v` is not a valid DATAMODE value.
        """
        if isinstance(v, str):
            try:
                return DATAMODE(v)
            except ValueError as err:
                valid_values = [e.value for e in DATAMODE]
                raise ValueError(
                    f"Invalid datamode: {v}. Must be one of: {valid_values}"
                ) from err
        return v

    model_config = {"use_enum_values": True}


class PanddaMapConfig(BaseModel):
    loss_type: str = "l2"
    output_dir: str = "panddamap_outputs"
    run_note: str = "panddamap"
    run_mse_prepass: bool = True
    save_mse_biases: bool = False
    preprocess_target_map: bool = False
    tv_denoise: bool = False
    ligand_mask_radius: float = 2.5
    denoise_high_res_limit: float = 1.8
    save_every_n_iterations: int = 50
    early_stopping_patience: int = 150
    save_best_pdb: bool = True
    save_trajectory_pdb: bool = True
    save_trajectory_interval: int = 1
    use_wandb: bool = False
    wandb_entity: str | None = Field(
        default_factory=lambda: os.environ.get("WANDB_ENTITY")
    )
    wandb_project: str | None = Field(
        default_factory=lambda: os.environ.get("WANDB_PROJECT")
    )
    wandb_name: str | None = None
    wandb_tags: list[str] = Field(default_factory=list)
    wandb_notes: str | None = None


# Main configuration class
class RocketRefinmentConfig(BaseModel):
    # Metadata
    note: str = ""

    # Nested configurations
    paths: PathConfig = Field(default_factory=PathConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    panddamap: PanddaMapConfig = Field(default_factory=PanddaMapConfig)

    model_config = {"use_enum_values": True}

    # Mapping for flat to nested structure conversion
    _flat_to_nested_map: ClassVar[dict[str, str]] = {
        # Paths
        "path": "paths.path",
        "input_dir": "paths.input_dir",
        "file_id": "paths.file_id",
        "template_pdb": "paths.template_pdb",
        "input_msa": "paths.input_msa",
        "input_pdb": "paths.input_pdb",
        "input_fasta": "paths.input_fasta",
        "alignment_dir": "paths.alignment_dir",
        "template_mmcif_dir": "paths.template_mmcif_dir",
        "template_release_dates_path": "paths.template_release_dates_path",
        "template_obsolete_pdbs_path": "paths.template_obsolete_pdbs_path",
        "kalign_binary_path": "paths.kalign_binary_path",
        "max_template_date": "paths.max_template_date",
        "template_max_hits": "paths.template_max_hits",
        "target_map": "paths.target_map",
        "sub_msa_path": "paths.sub_msa_path",
        "sub_delmat_path": "paths.sub_delmat_path",
        "msa_feat_init_path": "paths.msa_feat_init_path",
        "starting_bias": "paths.starting_bias",
        "starting_weights": "paths.starting_weights",
        "ligand_pdb": "paths.ligand_pdb",
        "uuid_hex": "paths.uuid_hex",
        # Execution
        "cuda_device": "execution.cuda_device",
        "num_of_runs": "execution.num_of_runs",
        "verbose": "execution.verbose",
        # Algorithm
        "bias_version": "algorithm.bias_version",
        "iterations": "algorithm.iterations",
        "init_recycling": "algorithm.init_recycling",
        "domain_segs": "algorithm.domain_segs",
        # Optimization
        "additive_learning_rate": "algorithm.optimization.additive_learning_rate",
        "multiplicative_learning_rate": "algorithm.optimization.multiplicative_learning_rate",  # noqa: E501
        "weight_decay": "algorithm.optimization.weight_decay",
        "batch_sub_ratio": "algorithm.optimization.batch_sub_ratio",
        "number_of_batches": "algorithm.optimization.number_of_batches",
        "rbr_opt_algorithm": "algorithm.optimization.rbr_opt_algorithm",
        "rbr_lbfgs_learning_rate": "algorithm.optimization.rbr_lbfgs_learning_rate",
        "smooth_stage_epochs": "algorithm.optimization.smooth_stage_epochs",
        "phase2_final_lr": "algorithm.optimization.phase2_final_lr",
        "l2_weight": "algorithm.optimization.l2_weight",
        # Features
        "solvent": "algorithm.features.solvent",
        "sfc_scale": "algorithm.features.sfc_scale",
        "refine_sigmaA": "algorithm.features.refine_sigmaA",
        "additional_chain": "algorithm.features.additional_chain",
        "total_chain_copy": "algorithm.features.total_chain_copy",
        "bias_from_fullmsa": "algorithm.features.bias_from_fullmsa",
        "chimera_profile": "algorithm.features.chimera_profile",
        # Data
        "datamode": "data.datamode",
        "free_flag": "data.free_flag",
        "testset_value": "data.testset_value",
        "min_resolution": "data.min_resolution",
        "max_resolution": "data.max_resolution",
        "voxel_spacing": "data.voxel_spacing",
        "msa_subratio": "data.msa_subratio",
        "w_plddt": "data.w_plddt",
        "downsample_ratio": "data.downsample_ratio",
        # Metadata
        "note": "note",
        "run_mse_prepass": "panddamap.run_mse_prepass",
        "save_mse_biases": "panddamap.save_mse_biases",
        "preprocess_target_map": "panddamap.preprocess_target_map",
        "tv_denoise": "panddamap.tv_denoise",
        "ligand_mask_radius": "panddamap.ligand_mask_radius",
        "denoise_high_res_limit": "panddamap.denoise_high_res_limit",
    }

    # Helper methods for backward compatibility
    def __getattr__(self, name):
        """Allow access to nested attributes directly for backward compatibility"""
        if name in self._flat_to_nested_map:
            path = self._flat_to_nested_map[name].split(".")
            value = self
            for part in path:
                value = getattr(value, part)
            return value
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def to_yaml_file(self, file_path: str) -> None:
        """Save configuration to YAML with fields in specific order"""
        # Convert model to dict
        model_dict = self.model_dump()

        # Create an ordered dictionary with the desired field order
        ordered_dict = {}
        # Define the order of top-level fields
        field_order = ["note", "data", "paths", "execution", "algorithm", "panddamap"]

        # Add fields in the specified order
        for field in field_order:
            if field in model_dict:
                ordered_dict[field] = model_dict[field]

        # Add any remaining fields that weren't in our order list
        for key, value in model_dict.items():
            if key not in ordered_dict:
                ordered_dict[key] = value

        # Write to file
        with open(file_path, "w") as file:
            yaml.dump(ordered_dict, file, sort_keys=False)

    def to_flat_yaml_file(self, file_path: str) -> None:
        """Save configuration in the old flat format for backward compatibility"""
        flat_dict = self.to_flat_dict()
        with open(file_path, "w") as file:
            yaml.dump(flat_dict, file)

    def to_flat_dict(self) -> dict[str, Any]:
        """Convert the nested structure to a flat dictionary"""
        result = {}
        for flat_key, nested_path in self._flat_to_nested_map.items():
            path_parts = nested_path.split(".")
            value = self
            for part in path_parts:
                value = getattr(value, part)
            result[flat_key] = value
        return result

    @classmethod
    def from_yaml_file(cls, file_path: str):
        with open(file_path) as file:
            payload = yaml.safe_load(file)

        # Try to determine if this is a flat or nested format
        if any(key in payload for key in ["paths", "algorithm", "data", "execution"]):
            # This appears to be a nested format
            return cls.model_validate(payload)
        else:
            # This appears to be a flat format
            return cls.from_flat_dict(payload)

    @classmethod
    def from_flat_dict(cls, flat_dict: dict[str, Any]):
        """Create an instance from a flat dictionary (old format)"""
        # Initialize nested dictionaries
        nested_dict = {
            "paths": {},
            "execution": {},
            "algorithm": {"optimization": {}, "features": {}},
            "data": {},
            "note": flat_dict.get("note", ""),
        }

        # Map flat keys to nested structure
        for flat_key, value in flat_dict.items():
            if flat_key in cls._flat_to_nested_map:
                nested_path = cls._flat_to_nested_map[flat_key].split(".")

                # Navigate to the correct nested dictionary
                target_dict = nested_dict
                for part in nested_path[:-1]:
                    target_dict = target_dict[part]

                # Set the value in the nested structure
                target_dict[nested_path[-1]] = value

        # Create the instance
        return cls.model_validate(nested_dict)


def gen_config(
    working_dir: str,
    input_dir: str,
    file_id: str,
    datamode: DATAMODE | None = DATAMODE.PANDDAMAP,
):
    config = RocketRefinmentConfig(
        note="panddamap_<your_note_here>",
        paths=PathConfig(
            path=working_dir,
            input_dir=input_dir,
            file_id=file_id,
            uuid_hex=uuid.uuid4().hex[:10],
        ),
        execution=ExecutionConfig(
            cuda_device=0,
            num_of_runs=1,
            verbose=False,
        ),
        algorithm=AlgorithmConfig(
            iterations=100,
            optimization=OptimizationParams(
                additive_learning_rate=1e-3,
                multiplicative_learning_rate=1e-3,
                l2_weight=1e-7,
                smooth_stage_epochs=50,
            ),
        ),
        data=DataConfig(
            datamode=datamode,
            min_resolution=3.0,
        ),
    )
    return config
