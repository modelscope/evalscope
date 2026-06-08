from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional

from evalscope.api.dataset import Dataset, DatasetDict, MemoryDataset, Sample


def stable_sample_key(sample: Sample) -> str:
    """Return a deterministic key for a sample.

    Prefer explicit sample id or metadata id. Fall back to a hash of prompt + target.
    This keeps pruning stable across machines and model runs.
    """
    if sample.id is not None:
        return str(sample.id)

    metadata_id = sample.metadata.get("id")
    if metadata_id is not None:
        return str(metadata_id)

    raw = f"{sample.input}|{sample.target}|{sample.metadata}"
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()


def prune_dataset_dict(
    dataset_dict: DatasetDict,
    *,
    strategy: str,
    prune_ratio: float,
    seed: int = 42,
    bucket_fn: Optional[Callable[[Sample], str]] = None,
) -> DatasetDict:
    """Prune every subset in a DatasetDict using a deterministic strategy.

    Args:
        dataset_dict: EvalScope dataset dictionary.
        strategy: Name of pruning strategy.
        prune_ratio: Fraction of samples to keep, in (0, 1].
        seed: Deterministic seed.
        bucket_fn: Optional benchmark-specific bucketing function.

    Returns:
        DatasetDict with pruned MemoryDataset subsets.
    """
    if not strategy or strategy in {"none", "full"}:
        return dataset_dict

    if prune_ratio <= 0 or prune_ratio > 1:
        raise ValueError(f"prune_ratio must be in (0, 1], got {prune_ratio}")

    pruned = {}
    for subset_name, dataset in dataset_dict.items():
        samples = list(dataset)
        keep_n = max(1, math.ceil(len(samples) * prune_ratio))

        if strategy == "coverage":
            selected = coverage_sample(samples, keep_n=keep_n, seed=seed, bucket_fn=bucket_fn)
        elif strategy == "stable_head":
            selected = stable_head_sample(samples, keep_n=keep_n)
        else:
            raise ValueError(
                f"Unknown pruning_strategy={strategy!r}. "
                "Supported: coverage, stable_head, none"
            )

        pruned[subset_name] = MemoryDataset(
            samples=selected,
            name=getattr(dataset, "name", subset_name),
            location=getattr(dataset, "location", None),
            shuffled=False,
        )
        pruned[subset_name].reindex()

    return DatasetDict(pruned)


def stable_head_sample(samples: List[Sample], *, keep_n: int) -> List[Sample]:
    """Deterministic smoke-test strategy. Not the final submission strategy."""
    return samples[:keep_n]


def coverage_sample(
    samples: List[Sample],
    *,
    keep_n: int,
    seed: int,
    bucket_fn: Optional[Callable[[Sample], str]],
) -> List[Sample]:
    """Coverage-preserving deterministic sampler.

    This is not uniform random. It first groups samples into metadata buckets,
    then allocates coverage across buckets, then uses stable hashing inside each
    bucket so selection is deterministic and reproducible.
    """
    if keep_n >= len(samples):
        return samples

    if bucket_fn is None:
        bucket_fn = default_bucket

    buckets: Dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        buckets[bucket_fn(sample)].append(sample)

    rng = random.Random(seed)

    # Stable shuffle within each bucket.
    for bucket_samples in buckets.values():
        bucket_samples.sort(key=lambda s: stable_sample_key(s))
        rng.shuffle(bucket_samples)

    # First pass: keep one from each bucket when possible.
    selected: List[Sample] = []
    bucket_names = sorted(buckets.keys())

    for bucket in bucket_names:
        if len(selected) >= keep_n:
            break
        if buckets[bucket]:
            selected.append(buckets[bucket].pop(0))

    # Second pass: fill remaining slots round-robin by bucket size.
    while len(selected) < keep_n:
        non_empty = [b for b in bucket_names if buckets[b]]
        if not non_empty:
            break

        non_empty.sort(key=lambda b: len(buckets[b]), reverse=True)
        for bucket in non_empty:
            if len(selected) >= keep_n:
                break
            selected.append(buckets[bucket].pop(0))

    # Preserve original dataset order for easier result diffing.
    selected_keys = {stable_sample_key(s) for s in selected}
    return [s for s in samples if stable_sample_key(s) in selected_keys]


def default_bucket(sample: Sample) -> str:
    metadata = sample.metadata or {}

    parts = [
        str(metadata.get("question_type", "na")),
        str(metadata.get("img_type", "na")),
        str(metadata.get("topic_difficulty", "na")),
        str(metadata.get("subfield", "na")),
    ]

    input_tokens = metadata.get("input_tokens")
    if isinstance(input_tokens, (int, float)):
        if input_tokens < 8000:
            parts.append("ctx:short")
        elif input_tokens < 32000:
            parts.append("ctx:medium")
        else:
            parts.append("ctx:long")

    return "|".join(parts)