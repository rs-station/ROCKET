"""Input/output helpers for refinement workflows."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import gemmi
import torch
from SFC_Torch import PDBParser

from rocket.refinement_config import RocketRefinmentConfig


def resolve_input_pdb(config: RocketRefinmentConfig) -> Path:
    if config.paths.input_pdb:
        return Path(config.paths.input_pdb)
    input_dir = Path(config.paths.input_dir or config.paths.path)
    return input_dir / f"{config.paths.file_id}-pred-aligned.pdb"


def resolve_input_fasta(config: RocketRefinmentConfig) -> Path | None:
    if config.paths.input_fasta:
        return Path(config.paths.input_fasta)
    input_dir = Path(config.paths.input_dir or config.paths.path)
    for pattern in ["*.fasta", "*.fa", "*.faa"]:
        matches = sorted(input_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def resolve_alignment_dir(config: RocketRefinmentConfig) -> Path | None:
    if config.paths.alignment_dir:
        return Path(config.paths.alignment_dir)
    input_dir = Path(config.paths.input_dir or config.paths.path)
    for candidate in ["alignments", "alignment", "msa"]:
        dir_path = input_dir / candidate
        if dir_path.exists() and dir_path.is_dir():
            return dir_path
    return None


def resolve_target_map(config: RocketRefinmentConfig) -> Path:
    if config.paths.target_map:
        return Path(config.paths.target_map)

    input_dir = Path(config.paths.input_dir or config.paths.path)

    search_roots = [input_dir]
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


def resolve_starting_path(path_value: str | None, base_dir: Path) -> str | None:
    if not path_value:
        return None
    if os.path.isabs(path_value):
        return path_value
    return str(base_dir / path_value)


def load_target_map(target_map_path: Path) -> gemmi.Ccp4Map:
    target_map = gemmi.read_ccp4_map(str(target_map_path))
    target_map.setup(0.0)
    return target_map


def load_input_pdb(input_pdb_path: Path, target_map: gemmi.Ccp4Map) -> PDBParser:
    input_pdb = PDBParser(str(input_pdb_path))
    input_pdb.set_spacegroup("P 1")
    input_pdb.set_unitcell(target_map.grid.unit_cell)
    return input_pdb


def get_output_dir_and_note(
    config: RocketRefinmentConfig,
    base_dir: Path,
) -> tuple[Path, str]:
    output_dir = (
        Path(config.panddamap.output_dir)
        if Path(config.panddamap.output_dir).is_absolute()
        else base_dir / config.panddamap.output_dir
    )
    run_note = config.note or config.panddamap.run_note or "panddamap"
    return output_dir, run_note


def make_best_biases_saver(
    device_features: dict[str, torch.Tensor],
    output_dir: Path,
) -> Callable[..., None]:
    def _save_best_biases(*, run_id: str, iteration: int, loss: float) -> None:
        bias_tensor = device_features["msa_feat_bias"].detach().cpu().clone()
        weight_tensor = device_features["msa_feat_weights"].detach().cpu().clone()
        bias_path = output_dir / "best_msa_bias.pt"
        weights_path = output_dir / "best_feat_weights.pt"
        torch.save(bias_tensor, bias_path)
        torch.save(weight_tensor, weights_path)

    return _save_best_biases
