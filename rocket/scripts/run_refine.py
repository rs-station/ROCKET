import argparse

from ..refinement_config import RocketRefinmentConfig
from .run_losslab_refine import (
    main as _losslab_main,
)
from .run_losslab_refine import (
    run_mse_only,
    run_panddamap_refinement,
)


def run_refinement(config: RocketRefinmentConfig | str) -> RocketRefinmentConfig:
    if isinstance(config, str):
        config = RocketRefinmentConfig.from_yaml_file(config)

    return run_panddamap_refinement(config)


def cli_runrefine():
    """Entry point for ``rk.refine``.

    Delegates to :func:`run_losslab_refine.main` so that map
    preprocessing and the MSE prepass are executed when enabled in the
    config (previously these steps were skipped).
    """
    _losslab_main()


def cli_refine_mse():
    """Entry point for ``rk.refine_mse``.

    Runs **only** the MSE coordinate loss on backbone atoms, with
    weighted Kabsch alignment.  No realspace map refinement is performed.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run MSE backbone-coordinate refinement only (no realspace map loss)"
        ),
    )
    parser.add_argument("config", type=str, help="Path to YAML config")
    parser.add_argument(
        "--selection",
        type=str,
        default="BB",
        choices=["BB", "CA", "ALL"],
        help="Atom selection for MSE loss (default: BB)",
    )
    args = parser.parse_args()
    run_mse_only(args.config, selection_override=args.selection)
