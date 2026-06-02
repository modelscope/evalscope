# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Benchmark Pruning Modules

This package provides hybrid cognitive pruning for benchmark compression,
combining Scales++ cognitive heuristics with IRT-based discrimination analysis.
"""

from .index_loader import load_pruned_indices, load_pruning_metadata

__all__ = [
    'load_pruned_indices',
    'load_pruning_metadata',
]
