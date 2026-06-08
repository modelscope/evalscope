from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def load_statistics(path: str | Path) -> Dict[int, dict]:
    stats = {}
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            stats[int(row["index"])] = row

    return stats


def select_indices_from_statistics(
    stats_path: str | Path,
    *,
    prune_ratio: float,
    seed: int = 42,
    min_keep: int = 1,
) -> List[int]:
    stats = load_statistics(stats_path)
    rows = list(stats.values())

    keep_n = max(min_keep, math.ceil(len(rows) * prune_ratio))
    if keep_n >= len(rows):
        return sorted(int(row["index"]) for row in rows)

    buckets = defaultdict(list)
    for row in rows:
        key = (
            row.get("difficulty_bucket", "unknown"),
            row.get("context_bucket", "none"),
        )
        buckets[key].append(row)

    selected = []

    # Preserve original full-dataset bucket proportions.
    allocations = {}
    for key, bucket_rows in buckets.items():
        allocations[key] = max(1, round(len(bucket_rows) / len(rows) * keep_n))

    # Adjust allocation to exact keep_n.
    while sum(allocations.values()) > keep_n:
        largest = max(allocations, key=lambda k: allocations[k])
        if allocations[largest] > 1:
            allocations[largest] -= 1
        else:
            break

    while sum(allocations.values()) < keep_n:
        largest_remaining = max(
            allocations,
            key=lambda k: len(buckets[k]) - allocations[k],
        )
        allocations[largest_remaining] += 1

    for key, bucket_rows in buckets.items():
        bucket_rows = sorted(
            bucket_rows,
            key=lambda r: (
                -float(r.get("disagreement", 0.0)),
                int(r["index"]),
            ),
        )
        selected.extend(bucket_rows[: allocations[key]])

    return sorted(int(row["index"]) for row in selected[:keep_n])
