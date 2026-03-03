"""Example script demonstrating GPU memory profiling for ROCKET refinement."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from rocket.memory_profiler import MemoryProfiler, MemorySnapshotProfiler
from rocket.run_losslab_refine import run_panddamap_refinement


def example_memory_snapshot(config_path: str, output_dir: str = "./profiles"):
    """
    Profile memory usage with detailed snapshots.

    This approach captures detailed memory allocation history with stack traces.
    Use this when debugging out-of-memory errors.

    To visualize the snapshot:
    1. Upload the .pickle file to https://pytorch.org/memory_viz
    2. Or use: python torch/cuda/_memory_viz.py trace_plot snapshot.pickle \
       -o snapshot.html
    """
    with MemorySnapshotProfiler(output_dir):
        run_panddamap_refinement(config_path)


def example_memory_profile(config_path: str, output_dir: str = "./profiles"):
    """
    Profile memory usage with PyTorch profiler.

    Note: For detailed memory visualization, use MemorySnapshotProfiler instead,
    which is faster and provides stack traces via pytorch.org/memory_viz.
    This mode is kept for compatibility but HTML export is disabled due to performance.
    """
    with MemoryProfiler(output_dir, device="cuda:0"):
        run_panddamap_refinement(config_path)


def example_memory_profile_with_steps(config_path: str, output_dir: str = "./profiles"):
    """
    Profile memory with manual step tracking for granular monitoring.

    Use prof.step() to mark iteration boundaries for better visualization.
    """
    profiler = MemoryProfiler(output_dir, device="cuda:0")

    with profiler:
        # training loop
        config = run_panddamap_refinement(config_path)
        profiler.step()  # Mark iteration end

    return config


def example_compare_memory_snapshots(config_path: str, output_dir: str = "./profiles"):
    """
    Capture multiple snapshots to compare memory usage at different stages.

    This is useful for identifying memory issues between iterations.
    """
    print("Running first memory snapshot (baseline)...")
    example_memory_snapshot(config_path, f"{output_dir}/snapshot_1")

    # Clear GPU cache between snapshots
    torch.cuda.empty_cache()

    print("\nRunning second memory snapshot (for comparison)...")
    example_memory_snapshot(config_path, f"{output_dir}/snapshot_2")

    print("\nCompare the snapshots at https://pytorch.org/memory_viz")
    print("Both .pickle files can be uploaded there for side-by-side analysis")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPU memory profiling examples")
    parser.add_argument("config", type=str, help="Path to ROCKET YAML config")
    parser.add_argument(
        "--mode",
        choices=["snapshot", "profile", "profile-steps", "compare"],
        default="profile",
        help="Profiling mode to use",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./memory_profiles",
        help="Directory to save profiling outputs",
    )

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == "snapshot":
        print("Starting memory snapshot profiling...")
        example_memory_snapshot(args.config, args.output_dir)
    elif args.mode == "profile":
        print("Starting memory profiling with timeline...")
        example_memory_profile(args.config, args.output_dir)
    elif args.mode == "profile-steps":
        print("Starting memory profiling with step tracking...")
        example_memory_profile_with_steps(args.config, args.output_dir)
    elif args.mode == "compare":
        print("Starting comparative memory snapshots...")
        example_compare_memory_snapshots(args.config, args.output_dir)

    print("\nProfiling complete!")
    print(f"Results saved to: {args.output_dir}")
    print(
        "View memory timeline in browser or upload .pickle to https://pytorch.org/memory_viz"
    )
