# flake8: noqa: E501
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import LiveCodeBenchAdapter
from evalscope.benchmarks.pruning.statistics_selector import select_indices_from_statistics
from evalscope.benchmarks.pruning.dataset_filter import filter_dataset_by_indices


@register_benchmark(
    BenchmarkMeta(
        name='live_code_bench_pruned',
        pretty_name='Live-Code-Bench-Pruned',
        tags=[Tags.CODING],
        description='Pruned LiveCodeBench using disagreement-aware coverage sampling.',
        dataset_id='evalscope/livecodebench_code_generation_lite_parquet',
        subset_list=['v5'],
        metric_list=['acc'],
        aggregation='mean_and_pass_at_k',
        eval_split='test',
        prompt_template='### Question:\n{question_content}\n\n{format_prompt} ### Answer: (use the provided format with backticks)\n\n',
        review_timeout=6,
        extra_params={
            'start_date': {'type': 'str | null', 'description': 'Filter start date.', 'value': None},
            'end_date': {'type': 'str | null', 'description': 'Filter end date.', 'value': None},
            'debug': {'type': 'bool', 'description': 'Enable debug logging.', 'value': False},
            'pruning_strategy': {
                'type': 'str',
                'description': 'Supported: disagreement_coverage, none.',
                'value': 'disagreement_coverage'
            },
            'prune_ratio': {
                'type': 'float',
                'description': 'Fraction of samples to keep.',
                'value': 0.1
            },
            'prune_seed': {
                'type': 'int',
                'description': 'Deterministic seed for pruning.',
                'value': 42
            },
            'statistics_path': {
                'type': 'str',
                'description': 'Path to precomputed LCB statistics JSONL.',
                'value': 'analysis/lcb_statistics.jsonl'
            }
        },
        sandbox_config={
            'image': 'python:3.11-slim',
            'tools_config': {'shell_executor': {}, 'python_executor': {}}
        },
    )
)
class LiveCodeBenchPrunedAdapter(LiveCodeBenchAdapter):
    """Registered pruned LiveCodeBench adapter."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pruning_strategy = self.extra_params.get('pruning_strategy', 'disagreement_coverage')
        self.prune_ratio = float(self.extra_params.get('prune_ratio', 0.1))
        self.prune_seed = int(self.extra_params.get('prune_seed', 42))
        self.statistics_path = self.extra_params.get('statistics_path', 'analysis/lcb_statistics.jsonl')

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
        pruned = {}

        for subset_name, subset_data in dataset.items():
            selected = []
            for position, sample in enumerate(subset_data):
                sample_index = sample.id if sample.id is not None else position
                if int(sample_index) in keep_indices:
                    selected.append(sample)

            pruned_subset = MemoryDataset(
                samples=selected,
                name=getattr(subset_data, 'name', subset_name),
                location=getattr(subset_data, 'location', None),
                shuffled=False,
            )
            pruned_subset.reindex()
            pruned[subset_name] = pruned_subset

        return DatasetDict(pruned)
