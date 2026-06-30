import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evalscope.perf.arguments import Arguments

_MAX_AUTO_DATASET_WORKERS = 32
_MIN_AUTO_ITEMS_PER_WORKER = 128


def available_cpu_count() -> int:
    """Return the CPU count available to this process."""
    if hasattr(os, 'sched_getaffinity'):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except Exception:
            pass
    return max(1, os.cpu_count() or 1)


def resolve_dataset_generation_workers(
    args: 'Arguments',
    total_count: int,
    supports_parallel_generation: bool,
) -> int:
    """Resolve worker count for CPU-bound dataset/request generation."""
    if total_count <= 1 or not supports_parallel_generation:
        return 1

    configured_workers = args.num_workers
    if configured_workers == 1:
        return 1
    if configured_workers > 1:
        return min(configured_workers, total_count)

    amortized_workers = max(1, total_count // _MIN_AUTO_ITEMS_PER_WORKER)
    return min(available_cpu_count(), total_count, _MAX_AUTO_DATASET_WORKERS, amortized_workers)
