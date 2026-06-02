# Copyright (c) Alibaba, Inc. and its affiliates.
# flake8: noqa: E501
"""
AA-LCR Pruned Benchmark Adapter

Pruned version of AA-LCR using hybrid cognitive pruning:
- 50 samples (50% reduction from original 100)
- Maintains difficulty distribution (21% easy, 71% medium, 8% hard)
- MAE < 7% (achieved 6.0%)
- Strategy: 70% cognitive centrality + 30% IRT quality
"""

from typing import List

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.aa_lcr.aa_lcr_adapter import AALCRAdapter
from evalscope.benchmarks.pruners.index_loader import load_pruned_indices
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='aa_lcr_pruned',
        pretty_name='AA-LCR (Pruned)',
        tags=[Tags.KNOWLEDGE, Tags.REASONING, Tags.LONG_CONTEXT],
        description="""
## Overview

AA-LCR Pruned is a compressed version of the AA-LCR benchmark, reduced from 100 to 50 samples using hybrid cognitive pruning (Scales++ + IRT). The pruned set maintains the difficulty distribution and achieves <7% MAE.

## Pruning Strategy

- **Method**: Hybrid cognitive pruning (Scales++ cognitive heuristics + 2PL IRT)
- **Reduction**: 50% (100 → 50 samples)
- **Selection**: 70% cognitive centrality + 30% IRT quality
- **IRT Filter**: α > 1.0 (loose, accounting for noisy LLM judge)
- **Clustering**: 12 clusters via UMAP (12D→3D) + K-means
- **Global Stratification**: Matches original difficulty distribution

## Validation Results

- **Target MAE**: <7%
- **Achieved MAE**: 6.0%
- **Per-model MAE**: gpt-oss-120b: 6.0%, kimi-k2.5: 5.8%, minimax-m2.5: 6.2%

## Key Innovation

Zero LLM cost for cognitive heuristics (vs. original Scales++ using GPT-4o embeddings for clustering).

## Task Description

Same as full AA-LCR benchmark - long-context retrieval and reasoning across multiple documents.

## Evaluation Notes

- Uses the same LLM judge as full AA-LCR
- Evaluates on 50 pre-selected samples
- Maintains original difficulty distribution
- Suitable for quick model capability assessment
""",
        dataset_id='evalscope/AA-LCR',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=AALCRAdapter.__dict__.get('prompt_template') or None,
        extra_params={
            'text_dir': {
                'type': 'str | null',
                'description': 'Local directory containing extracted AA-LCR text files; if null will auto-download & extract.',
                'value': None
            }
        }
    )
)
class AALCRPrunedAdapter(AALCRAdapter):
    """
    Pruned adapter for AA-LCR benchmark.

    Inherits all functionality from AALCRAdapter but loads only the 50 pre-selected
    samples identified through hybrid cognitive pruning.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pruned_indices = None

    def load(self) -> List:
        """Load full dataset then filter to pruned samples."""
        # Load full dataset using parent's load method
        full_dataset = super().load()

        # Load pre-computed pruned indices
        if self._pruned_indices is None:
            self._pruned_indices = load_pruned_indices('aa_lcr')

        # Filter dataset to only include pruned samples
        pruned_dataset = [full_dataset[i] for i in self._pruned_indices if i < len(full_dataset)]

        logger.info(
            f"Loaded AA-LCR pruned dataset: {len(pruned_dataset)} samples "
            f"(50% reduction from {len(full_dataset)} original samples)"
        )

        return pruned_dataset
