# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, List, Tuple

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.dataset.dataset import DatasetDict, MemoryDataset
from evalscope.api.metric.scorer import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.report.report import Report, Subset
from evalscope.utils.logger import get_logger

logger = get_logger()

SUBSET_LIST = [
    'en', 'zh', 'ar', 'bn', 'de', 'es', 'fr', 'id', 'it', 'ja', 'ko', 'ms', 'pt', 'ru', 'sw', 'te', 'th', 'vi'
]
LEVEL_LIST = ['low', 'medium', 'high', 'top']


@register_benchmark(
    BenchmarkMeta(
        name='poly_math',
        pretty_name='PolyMath',
        tags=[Tags.MATH, Tags.REASONING, Tags.MULTI_LINGUAL],
        description=
        'PolyMath is a multilingual mathematical reasoning benchmark covering 18 languages and 4 easy-to-hard difficulty levels, with 9,000 high-quality problem samples. Our benchmark ensures difficulty comprehensiveness, language diversity, and high-quality translation, making it a highly discriminative multilingual mathematical benchmark in the era of reasoning LLMs.',  # noqa: E501
        dataset_id='evalscope/PolyMath',
        subset_list=SUBSET_LIST,
        metric_list=[{
            'acc': {
                'numeric': True
            }
        }],
        eval_split='test',
        prompt_template='{question}',
    )
)
class PolyMathAdapter(DefaultDataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load(self) -> Tuple[DatasetDict, None]:
        """Load all difficulty levels, rename subsets with their level suffix, and merge them."""
        # Need to load all levels to get the full dataset
        dataset_list: List[Dict[str, MemoryDataset]] = []
        original_split = getattr(self, 'eval_split', None)
        try:
            for level in LEVEL_LIST:
                self.eval_split = level
                cur_level_dataset_dict, _ = super().load()
                # Build a renamed mapping without mutating during iteration
                renamed: Dict[str, MemoryDataset] = {
                    f'{subset}-{level}': ds
                    for subset, ds in cur_level_dataset_dict.items()
                }
                dataset_list.append(renamed)
        finally:
            # Restore original split to avoid side effects
            if original_split is not None:
                self.eval_split = original_split
        # Merge all levels into one dataset
        return DatasetDict.from_dataset_dicts(dataset_list), None

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a single record into a Sample with language-specific instruction."""
        from .utils.instruction import query_dic

        # e.g., 'high-en-1'
        question_id = record['id']
        level, language, index = question_id.split('-')
        # Get the instruction for the specific language
        instruction = query_dic[language]

        return Sample(
            input=record['question'] + '\n' + instruction,
            target=record['answer'],
            metadata={
                'level': level,
                'language': language,
                'index': index,
            },
        )

    def extract_answer(self, prediction: str, task_state):
        from evalscope.metrics.math_parser import extract_answer

        return extract_answer(prediction)

    def _on_generate_report_end(self, report: Report, output_dir, **kwargs):
        """
        Finalize the report generation process. Calculate the difficulty-weighted accuracy (DW-ACC)
        per language and the overall DW-ACC, and append as a new category to each metric.
        """
        from evalscope.report import Category, Metric

        WEIGHT_DENOMINATOR = 15  # 1 + 2 + 4 + 8 for ['low','medium','high','top']

        for metric in report.metrics:
            # Collect all subsets by name for easy lookup (e.g., "en-low")
            subset_dict: Dict[str, Subset] = {}
            for category in metric.categories:
                for subset in category.subsets:
                    subset_dict[subset.name] = subset

            # Compute per-language DW-ACC
            dw_subsets: List[Subset] = []
            for language in SUBSET_LIST:
                weighted_sum = 0.0
                total_num = 0
                for i, level in enumerate(LEVEL_LIST):
                    s = subset_dict.get(f'{language}-{level}')
                    if s is not None:
                        weighted_sum += (2**i) * s.score
                        total_num += s.num
                # missing subsets contribute 0 by design
                if total_num == 0:
                    continue
                dw_acc = weighted_sum / WEIGHT_DENOMINATOR
                dw_subsets.append(Subset(name=language, score=dw_acc, num=total_num))

            # Overall DW-ACC: unweighted average across languages
            if dw_subsets:
                overall_score = sum(s.score for s in dw_subsets) / len(dw_subsets)
                overall_num = sum(s.num for s in dw_subsets)
                dw_subsets.append(Subset(name='OVERALL', score=overall_score, num=overall_num))

        # Append DW-ACC metric
        if dw_subsets:
            report.metrics.append(Metric(name='DW-ACC', categories=[Category(name='-', subsets=dw_subsets)]))
