# flake8: noqa: E501
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.benchmarks.pruning.dataset_filter import filter_dataset_by_indices
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.benchmarks.aa_lcr.aa_lcr_adapter import AALCRAdapter
from evalscope.benchmarks.pruning.statistics_selector import select_indices_from_statistics


@register_benchmark(
    BenchmarkMeta(
        name='aa_lcr_pruned',
        pretty_name='AA-LCR-Pruned',
        tags=[Tags.KNOWLEDGE, Tags.REASONING, Tags.LONG_CONTEXT],
        description='Pruned AA-LCR using disagreement-aware coverage sampling with context-length preservation.',
        dataset_id='evalscope/AA-LCR',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=AALCRAdapter.__module__,
        extra_params={
            'text_dir': {
                'type': 'str | null',
                'description': 'Local directory containing extracted AA-LCR text files.',
                'value': None
            },
            'pruning_strategy': {
                'type': 'str',
                'description': 'Supported: disagreement_coverage, none.',
                'value': 'disagreement_coverage'
            },
            'prune_ratio': {
                'type': 'float',
                'description': 'Fraction of samples to keep.',
                'value': 0.25
            },
            'prune_seed': {
                'type': 'int',
                'description': 'Deterministic seed for pruning.',
                'value': 42
            },
            'statistics_path': {
                'type': 'str',
                'description': 'Path to precomputed AA-LCR statistics JSONL.',
                'value': 'analysis/aalcr_statistics.jsonl'
            }
        },
    )
)
class AALCRPrunedAdapter(AALCRAdapter):
    """Registered pruned AA-LCR adapter."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pruning_strategy = self.extra_params.get('pruning_strategy', 'disagreement_coverage')
        self.prune_ratio = float(self.extra_params.get('prune_ratio', 0.25))
        self.prune_seed = int(self.extra_params.get('prune_seed', 42))
        self.statistics_path = self.extra_params.get('statistics_path', 'analysis/aalcr_statistics.jsonl')

    def load_dataset(self):
        dataset = super().load_dataset()

        if self.pruning_strategy in (None, '', 'none', 'full'):
            return dataset

        keep_indices = set(
            select_indices_from_statistics(
                self.statistics_path,
                prune_ratio=self.prune_ratio,
                seed=self.prune_seed,
            )
        )

        dataset = filter_dataset_by_indices(dataset, keep_indices)
        self.test_dataset = dataset
        return dataset

