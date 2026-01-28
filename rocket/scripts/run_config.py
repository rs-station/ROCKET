import argparse
import os

from loguru import logger

from ..refinement_config import DATAMODE, RUNMODE, RocketRefinmentConfig, gen_config


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ROCKET configuration files")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["phase1", "phase2", "both"],
        default="both",
        help="Execution mode: phase1, phase2, or both (default: both)",
    )
    parser.add_argument(
        "--datamode",
        type=str,
        choices=["xray", "cryoem", "panddamap"],
        help="Data mode",
    )
    parser.add_argument("--working-dir", type=str, help="Working directory path")
    parser.add_argument("--file-id", type=str, help="File ID to use")
    parser.add_argument(
        "--pre-phase1-config",
        type=str,
        help="Path to pre-phase1 configuration file (required for phase2 mode)",
    )

    args = parser.parse_args()

    # Convert string mode to RUNMODE enum
    run_mode = RUNMODE(args.mode)

    # Convert string datamode to DATAMODE enum if provided
    data_mode = None
    if args.datamode:
        data_mode = DATAMODE(args.datamode)

    # Load pre_phase1_config if provided and mode is phase2
    pre_phase1_config_obj = None
    if args.pre_phase1_config:
        if run_mode == RUNMODE.PHASE2 and not os.path.exists(args.pre_phase1_config):
            parser.error(f"Pre-phase1 config file not found: {args.pre_phase1_config}")
        # Load the config file
        pre_phase1_config_obj = RocketRefinmentConfig.from_yaml_file(
            args.pre_phase1_config
        )  # noqa: E501
    elif run_mode == RUNMODE.PHASE2:
        parser.error("--pre-phase1-config is required when mode is phase2")

    return {
        "mode": run_mode,
        "datamode": data_mode,
        "working_dir": args.working_dir,
        "file_id": args.file_id,
        "pre_phase1_config": pre_phase1_config_obj,
    }


def cli_runconfig():
    args = parse_args()

    result = gen_config(
        mode=args.get("mode"),
        datamode=args.get("datamode"),
        working_dir=args.get("working_dir"),
        file_id=args.get("file_id"),
        pre_phase1_config=args.get("pre_phase1_config"),
    )

    if args.get("mode") == RUNMODE.BOTH:
        phase1_config, phase2_config = result
        logger.info(
            f"Generated Phase 1 config at: {os.path.join(args.get('working_dir'), 'ROCKET_config_phase1.yaml')}"  # noqa: E501
        )
        logger.info(
            f"Generated Phase 2 config at: {os.path.join(args.get('working_dir'), 'ROCKET_config_phase2.yaml')}"  # noqa: E501
        )
    elif args.get("mode") == RUNMODE.PHASE1:
        logger.info(
            f"Generated Phase 1 config at: {os.path.join(args.get('working_dir'), 'ROCKET_config_phase1.yaml')}"  # noqa: E501
        )
    else:  # PHASE2
        logger.info(
            f"Generated Phase 2 config at: {os.path.join(result.working_dir, 'ROCKET_config_phase2.yaml')}"  # noqa: E501
        )


if __name__ == "__main__":
    cli_runconfig()
