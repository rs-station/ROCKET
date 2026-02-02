import argparse
from pathlib import Path

from loguru import logger

from ..refinement_config import DATAMODE, gen_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a PanddaMap ROCKET config",
    )
    parser.add_argument(
        "--datamode",
        type=str,
        choices=["panddamap"],
        default="panddamap",
        help="Data mode",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input folder containing .pdb/.ccp4/.pickle",
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        required=True,
        help="Output working directory",
    )
    parser.add_argument("--file-id", type=str, required=True, help="File ID")
    return parser.parse_args()


def _auto_find(input_dir: Path, patterns: list[str]) -> str | None:
    for pattern in patterns:
        matches = sorted(input_dir.glob(pattern))
        if matches:
            return str(matches[0])
    return None


def _auto_find_dir(input_dir: Path, candidates: list[str]) -> str | None:
    for name in candidates:
        candidate = input_dir / name
        if candidate.exists() and candidate.is_dir():
            return str(candidate)
    return None


def cli_runconfig():
    args = parse_args()
    input_dir = Path(args.input_dir)
    working_dir = Path(args.working_dir)

    config = gen_config(
        working_dir=str(working_dir),
        input_dir=str(input_dir),
        file_id=args.file_id,
        datamode=DATAMODE(args.datamode),
    )

    config.paths.input_pdb = (
        _auto_find(
            input_dir,
            ["*pred-aligned*.pdb", "*.pdb"],
        )
        or ""
    )
    config.paths.target_map = _auto_find(
        input_dir,
        ["*masked*.ccp4", "*masked*.map", "*.ccp4", "*.map"],
    )
    config.paths.input_fasta = _auto_find(
        input_dir,
        ["*.fasta", "*.fa", "*.faa"],
    )
    config.paths.alignment_dir = _auto_find_dir(
        input_dir,
        ["alignments", "alignment", "msa"],
    )
    config.paths.msa_feat_init_path = _auto_find(
        input_dir,
        ["*processed*.pickle", "*.pickle"],
    )

    output_path = Path(working_dir) / "ROCKET_config.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config.to_yaml_file(str(output_path))
    logger.info("Generated config at: %s", output_path)


if __name__ == "__main__":
    cli_runconfig()
