# flake8: noqa: E501
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.benchmarks.pruning.dataset_filter import filter_dataset_by_metadata_id
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.benchmarks.mmmu.mmmu_adapter import MMMUAdapter, SUBSET_LIST
from evalscope.benchmarks.pruning.mmmu_probe_selector import select_mmmu_probe


@register_benchmark(
    BenchmarkMeta(
        name='mmmu_encoder_probe',
        pretty_name='MMMU-Encoder-Probe',
        tags=[Tags.MULTI_MODAL, Tags.KNOWLEDGE, Tags.QA],
        description='Pruned MMMU probe focused on image-encoder stress cases: tables, diagrams, charts, maps, medical images, chemical structures, and dense visual reasoning.',
        dataset_id='AI-ModelScope/MMMU',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        eval_split='validation',
        prompt_template=MMMUAdapter._benchmark_meta.prompt_template if hasattr(MMMUAdapter, "_benchmark_meta") else None,
        extra_params={
            'probe_size': {
                'type': 'int',
                'description': 'Number of MMMU probe samples to keep.',
                'value': 50
            },
            'statistics_path': {
                'type': 'str',
                'description': 'Path to precomputed MMMU statistics JSONL.',
                'value': 'analysis/mmmu_statistics.jsonl'
            }
        },
    )
)
class MMMUEncoderProbeAdapter(MMMUAdapter):
    """Registered MMMU image-encoder probe adapter."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.probe_size = int(self.extra_params.get('probe_size', 50))
        self.statistics_path = self.extra_params.get('statistics_path', 'analysis/mmmu_statistics.jsonl')

    def load_dataset(self):
        dataset = super().load_dataset()

        keep_ids = set(
            select_mmmu_probe(
                self.statistics_path,
                probe_size=self.probe_size,
            )
        )

        dataset = filter_dataset_by_metadata_id(dataset, keep_ids)
        self.test_dataset = dataset
        return dataset
