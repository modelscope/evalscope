# TableBench adapter: table question answering over fact checking, numerical reasoning and data analysis.
#
# Paper: https://arxiv.org/abs/2408.09174 (AAAI 2025)
# Dataset: https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableBench (Apache-2.0)
# Official evaluation: https://github.com/TableBench/TableBench

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import DatasetDict, DatasetHub, Sample, build_dataset_from_records
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

from .utils import metric_for, parse_final_answer, score_answer

SUBSET_LIST = ['FactChecking', 'NumericalReasoning', 'DataAnalysis']
INSTRUCTION_TYPES = ['TCoT', 'SCoT', 'DP']
# Pinned revision of the 2025-04-18 cleaned release on Hugging Face.
DATASET_REVISION = 'a23c244f9ccae1ea238d614fe7620984707e411a'

DESCRIPTION = """
## Overview

TableBench evaluates table question answering in realistic scenarios: 886 questions over real tables in
18 subcategories of four types (fact checking, numerical reasoning, data analysis and visualization).
This adapter covers the 836 text-answer questions of the first three types.

## Task Description

- **Task Type**: Table question answering
- **Input**: The official instruction with the table in JSON format and the question
- **Output**: Reasoning (depending on the instruction type), then `Final Answer: ...`
- **Domain**: Tables from finance, sports, demographics, science and more

## Key Features

- Fact checking (96), numerical reasoning (397: aggregation, arithmetic, comparison, counting, ranking,
  multi-hop, time-based, domain-specific) and data analysis (343: correlation, trend forecasting,
  statistical, impact, anomaly, causal, descriptive analysis)
- Uses the cleaned 2025-04-18 release of the test set and the refined official metrics
- Three official prompting styles: `TCoT` (textual chain-of-thought, default), `SCoT` (symbolic CoT)
  and `DP` (direct prompting)

## Evaluation Notes

- `mix_metric` (primary) is the official Mix_Metric: each question is scored with the metric the official
  script assigns to its subtype, then the scores are averaged:
  - EM for fact checking, numerical reasoning and impact analysis
  - EM with 10% numeric tolerance for correlation analysis, trend forecasting and statistical analysis
  - ROUGE-L for the other data-analysis subtypes
- The answer is the text after the first `Final Answer: ` on that line, as in the official parser
- As in the official code, prediction and reference are normalized (lowercase, punctuation and articles
  removed) before scoring, so decimal points, signs and percent signs are not compared
- Choose the prompting style with `extra_params={'instruction_type': 'DP'}`
- Visualization (50 chart-generation questions) needs execution of model-written plotting code and is
  not included yet
- [Paper](https://arxiv.org/abs/2408.09174) | [Code](https://github.com/TableBench/TableBench) |
  [Dataset](https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableBench) |
  [Leaderboard](https://tablebench.github.io/)
"""


@register_benchmark(
    BenchmarkMeta(
        name='tablebench',
        pretty_name='TableBench',
        dataset_id='Multilingual-Multimodal-NLP/TableBench',
        hf_dataset_id='Multilingual-Multimodal-NLP/TableBench',
        hf_dataset_revision=DATASET_REVISION,
        paper_url='https://arxiv.org/abs/2408.09174',
        tags=[Tags.REASONING, Tags.QA],
        description=DESCRIPTION,
        subset_list=SUBSET_LIST,
        eval_split='test',
        few_shot_mode='disabled',
        metric_list=['mix_metric'],
        prompt_template='{question}',
        evaluation_version='v1.0',
        extra_params={
            'instruction_type': {
                'type': 'str',
                'description': 'Official prompting style: TCoT (textual chain-of-thought), SCoT (symbolic CoT) or DP '
                '(direct answer).',
                'value': 'TCoT',
                'choices': INSTRUCTION_TYPES,
            },
        },
    )
)
class TableBenchAdapter(DefaultDataAdapter):
    """Data adapter for TableBench (text-answer types), scored with the official Mix_Metric."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.instruction_type = self.extra_params.get('instruction_type', 'TCoT')

    def load(self) -> Tuple[DatasetDict, None]:
        # Each prompting style ships as its own JSONL file. The dataset card groups the four files and
        # TableBench.jsonl into one config whose `table` column types differ, which `datasets` cannot
        # cast, so the file of the selected style is read directly.
        records = self._read_records(f'TableBench_{self.instruction_type}.jsonl')
        subsets = {}
        for subset in self.subset_list:
            subsets[subset] = build_dataset_from_records(
                records=[r for r in records if r['qtype'] == subset],
                sample_fields=self.record_to_sample,
                name=f'tablebench_{self.instruction_type}_{subset}',
                location=self.dataset_id,
                limit=self.limit,
                repeats=self.repeats,
                shuffle=self.shuffle,
                seed=self.seed,
            )
        return DatasetDict(subsets), None

    def _read_records(self, file_name: str) -> List[Dict[str, Any]]:
        if Path(self.dataset_id).is_dir():
            path = Path(self.dataset_id) / file_name
        else:
            path = DatasetHub(
                data_id_or_path=self.dataset_id,
                data_source=self.dataset_hub,
                revision=self.dataset_revision,
                force_redownload=self.force_redownload,
                cache_dir=self.dataset_dir,
            ).download_file(file_name)
        with open(path, encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(
            input=record['instruction'],
            target=record['answer'],
            subset_key=record['qtype'],
            metadata={
                'id': record['id'],
                'qtype': record['qtype'],
                'qsubtype': record['qsubtype'],
                'instruction_type': record['instruction_type'],
                'metric': metric_for(record['qtype'], record['qsubtype']),
            },
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        return parse_final_answer(prediction)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        meta = task_state.metadata
        return Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value={'mix_metric': score_answer(filtered_prediction, reference, meta['qtype'], meta['qsubtype'])},
            main_score_name='mix_metric',
            metadata={'metric': meta['metric'], 'parsed': bool(filtered_prediction)},
        )
