# flake8: noqa: E501
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger
from evalscope.utils.multi_choices import MultipleChoiceTemplate

logger = get_logger()

DESCRIPTION = """
## Overview

KINA (Knowledge Index of Noah's Ark) is a high-density multidisciplinary knowledge benchmark \
for evaluating whether large language models can solve expert-level questions across 261 fine-grained \
disciplines. It is the first benchmark to incorporate disciplinary representativeness as a core \
design principle.

## Task Description

- **Task Type**: Multiple-Choice Question Answering (MCQ)
- **Input**: A discipline-specific question with up to 10 lettered options (A–J)
- **Output**: A single correct answer letter (A–J)
- **Domains**: 261 disciplines spanning Agronomy, Medicine, Engineering, Humanities, Natural Sciences, and more

## Key Features

- 899 test questions covering 261 fine-grained disciplines
- Each question has a unique correct answer among up to 10 options (A–J)
- Includes per-option explanations for training / analysis (not shown to the model)
- Designed to test deep domain knowledge, not retrieval or commonsense reasoning
- Introduced at 2077AI with a focus on disciplinary representativeness

## Evaluation Notes

- Default evaluation uses the **test** split (899 samples)
- Primary metric: **Accuracy** (acc) — Pass@1 for single-inference mode
- 0-shot Chain-of-Thought (CoT) evaluation, answer extracted from ``ANSWER: [LETTER]`` marker
- Discipline metadata is stored per-sample and available in review output; no per-discipline subset grouping
- [GitHub](https://github.com/weihao1115/KINA-Benchmark)
"""


@register_benchmark(
    BenchmarkMeta(
        name='kina',
        pretty_name='KINA',
        dataset_id='evalscope/KINA',
        tags=[Tags.KNOWLEDGE, Tags.MULTIPLE_CHOICE],
        description=DESCRIPTION,
        paper_url='https://www.2077ai.com/kina',
        metric_list=['acc'],
        eval_split='test',
        prompt_template=MultipleChoiceTemplate.SINGLE_ANSWER_COT,
    )
)
class KINAAdapter(MultiChoiceAdapter):
    """Data adapter for evalscope/KINA.

    Each question may have up to 10 options (A–J).  Uses the standard
    MultiChoiceAdapter with CoT prompting and letter-based answer extraction.
    Discipline metadata is stored per-sample for reference but does not drive subset grouping.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a KINA record to a Sample.

        ``options`` is a list of ``{key, answer, explanation}`` dicts.
        ``correct_answer`` is a letter string such as ``'B'``.
        """
        question: str = record.get('question', '')
        options: List[Dict[str, Any]] = record.get('options', [])
        correct_answer: str = str(record.get('correct_answer', '')).strip().upper()

        # Extract ordered choice texts in key order (A, B, C, ...)
        choices = [opt['answer'].strip() for opt in sorted(options, key=lambda x: x['key'])]

        # Discipline stored in metadata for reference
        discipline: str = record.get('discipline', '')

        return Sample(
            input=question,
            choices=choices,
            target=correct_answer,
            metadata={
                'index': record.get('index'),
                'discipline': discipline,
            },
        )
