"""GPU Memory profiling utilities for ROCKET refinement."""

import logging
import socket
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from rocket.refinement_config import RocketRefinmentConfig

logger = logging.getLogger(__name__)

TIME_FORMAT_STR = "%b_%d_%H_%M_%S"
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100000


def get_device_from_config(config: "RocketRefinmentConfig") -> str:
    """Extract CUDA device string from ROCKET config.

    Args:
        config: RocketRefinmentConfig object

    Returns:
        Device string in format "cuda:N"
    """
    cuda_device = config.execution.cuda_device
    return f"cuda:{cuda_device}"


class MemoryProfiler:
    def __init__(
        self,
        output_dir: str | Path = ".",
        device: str | None = None,
        with_stack: bool = True,
        record_shapes: bool = True,
    ):
        """
        Initialize memory profiler.

        Args:
            output_dir: Directory to save profiling outputs
            device: CUDA device to profile (e.g., "cuda:0"). Defaults to "cuda:0"
            with_stack: Include stack traces in profiling
            record_shapes: Record tensor shapes
        """
        # Default to cuda:0 if not specified
        if device is None:
            device = "cuda:0"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.with_stack = with_stack
        self.record_shapes = record_shapes
        self.profiler = None
        self._get_file_prefix()

    def _get_file_prefix(self):
        """Generate file prefix with hostname and timestamp."""
        host_name = socket.gethostname()
        timestamp = datetime.now().strftime(TIME_FORMAT_STR)
        self.file_prefix = self.output_dir / f"{host_name}_{timestamp}"

    def __enter__(self):
        """Start profiling."""
        if not torch.cuda.is_available():
            logger.warning("CUDA unavailable. Memory profiling disabled.")
            return self

        logger.info("Starting GPU memory profiler")
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=5, repeat=1),
            record_shapes=self.record_shapes,
            profile_memory=True,
            with_stack=self.with_stack,
        )
        self.profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling and save outputs."""
        if self.profiler is None:
            return

        logger.info("Stopping GPU memory profiler")
        self.profiler.__exit__(exc_type, exc_val, exc_tb)
        logger.info(
            "Memory profiler completed. Use MemorySnapshotProfiler "
            "for detailed visualization."
        )

    def step(self):
        """Signal a profiling step."""
        if self.profiler is not None:
            self.profiler.step()


class MemorySnapshotProfiler:
    """Context manager for GPU memory snapshot profiling."""

    def __init__(self, output_dir: str | Path = "."):
        """
        Initialize memory snapshot profiler.

        Args:
            output_dir: Directory to save snapshot files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._get_file_prefix()

    def _get_file_prefix(self):
        """Generate file prefix with hostname and timestamp."""
        host_name = socket.gethostname()
        timestamp = datetime.now().strftime(TIME_FORMAT_STR)
        self.file_prefix = self.output_dir / f"{host_name}_{timestamp}"

    def __enter__(self):
        """Start recording memory history."""
        if not torch.cuda.is_available():
            logger.warning("CUDA unavailable. Memory snapshot profiling disabled.")
            return self

        logger.info("Starting GPU memory snapshot recording")
        torch.cuda.memory._record_memory_history(
            max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop recording and save snapshot."""
        if not torch.cuda.is_available():
            return

        try:
            snapshot_file = f"{self.file_prefix}.pickle"
            logger.info(f"Saving memory snapshot to {snapshot_file}")
            torch.cuda.memory._dump_snapshot(snapshot_file)
            logger.info(
                "Snapshot saved successfully. Upload to "
                "https://pytorch.org/memory_viz to visualize."
            )
        except Exception as e:
            logger.error(f"Failed to capture memory snapshot: {e}")

        # Stop recording
        logger.info("Stopping GPU memory snapshot recording")
        torch.cuda.memory._record_memory_history(enabled=None)


def memory_snapshot(output_dir: str | Path = "."):
    """Decorator for memory snapshot profiling of a function."""

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            with MemorySnapshotProfiler(output_dir):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def memory_profile(output_dir: str | Path = "."):
    """Decorator for memory profiling of a function."""

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            with MemoryProfiler(output_dir):
                return func(*args, **kwargs)

        return wrapper

    return decorator
