# flake8: noqa: E501
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, VisionLanguageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, ContentImage, ContentText
from evalscope.api.metric import AggScore, MetricSelector, SampleScore
from evalscope.api.metric.scorer import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags

from .utils import normalize_answer, score_answer

SUBSET_LIST = [
    'main',
    'identification',
    'withtitle',
    'original',
    'remove_background_q1q2',
    'remove_background_q3',
]

DESCRIPTION = """
## Overview

VLMs Are Biased (VLMBias) evaluates whether vision-language models answer objective visual questions from the image or fall back to memorized prior knowledge. It uses counterfactual images whose visible properties conflict with familiar concepts, such as an Adidas-style logo with four stripes or an animal with an unusual number of legs.

## Task Description

- **Task Type**: Free-form visual question answering for counting and identification
- **Input**: A counterfactual or control image paired with a counting, binary identification, or short-answer question
- **Output**: A number, `Yes`/`No`, or a short identity enclosed in curly brackets
- **Domain**: Animals, logos, flags, chess pieces, game boards, optical illusions, and patterned grids

## Key Features

- The primary `main` split contains 2,784 objective visual questions over 1,392 counterfactual images at 384, 768, and 1152 pixel resolutions
- Five official analysis splits cover binary identification, in-image title injection, original unmodified controls, and background-removed variants
- Each counterfactual record provides both the visually correct `ground_truth` and the prior-knowledge `expected_bias`
- The benchmark exposes seven topics and nineteen sub-topics for detailed analysis without creating synthetic EvalScope subsets
- The default data is **v2**: optical-illusion target features are red and named in the questions, IDs are globally unique, and the truncated identification image is repaired
- Both hubs are pinned to matching v2 snapshots; sample metadata preserves the v2 `ID`, the original `legacy_ID`, and the selected data version

## Evaluation Notes

- The dataset prompt is used verbatim, including its required curly-bracket answer format
- Primary metric: **Accuracy** (`acc`), using the official case-insensitive comparison after stripping outer braces; if exact text matching fails, digit sequences are compared
- Secondary metric: **Bias Ratio** (`bias_ratio`, lower is better), the fraction of predictions matching `expected_bias` under the same normalization
- Accuracy is also reported by topic, matching the official lmms-eval integration
- `bias_ratio` is omitted for the `original` split because those control records do not define `expected_bias`
- The six official dataset splits are evaluated as separate EvalScope subsets by default; select only `main` for the primary counting evaluation
- Default v2 runs use evaluation version **v1.1**. For the paper's v1 data, explicitly set `dataset_args={'vlms_are_biased': {'default_subset': 'v1-legacy', 'subset_list': ['main']}}`; these runs are labeled **v1 legacy** and use evaluation version **v1.0**. The legacy `identification` split retains the upstream truncated-image defect
- v1 and v2 results are not directly interchangeable. Start a new output directory after changing data versions; do not use `rerun_review` to reuse predictions for changed images or prompts. Historical unpinned runs require inspection of their saved inputs/IDs to establish the data version
- Generation should be deterministic and concise; the official lmms-eval setup uses `temperature=0` and at most 32 new tokens
- [Paper](https://arxiv.org/abs/2505.23941) | [GitHub](https://github.com/anvo25/vlms-are-biased) | [Project page](https://vlmsarebiased.github.io/)
"""


@register_benchmark(
    BenchmarkMeta(
        name='vlms_are_biased',
        pretty_name='VLMs Are Biased',
        dataset_id='evalscope/vlms-are-biased',
        dataset_revision='2e92cd147b735190b8f6a35bccf1b750bce3c399',
        hf_dataset_id='anvo25/vlms-are-biased',
        hf_dataset_revision='a6ebd68831ee7299d85631c929fea5016c03863f',
        subset_list=SUBSET_LIST,
        tags=[Tags.MULTI_MODAL, Tags.QA, Tags.REASONING],
        description=DESCRIPTION,
        paper_url='https://arxiv.org/abs/2505.23941',
        metric_list=['acc', 'bias_ratio'],
        primary_metric=MetricSelector(name='accuracy', dimensions={'scope': 'overall'}),
        eval_split='main',
        evaluation_version='v1.1',
    )
)
class VLMsAreBiasedAdapter(VisionLanguageAdapter):
    """Data adapter for the six official VLMs Are Biased dataset splits."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.split_as_subset = True
        self.data_version = 'v1-legacy' if self.default_subset == 'v1-legacy' else 'v2'
        if self.data_version == 'v1-legacy':
            self.benchmark_meta.evaluation_version = 'v1.0'
            self.benchmark_meta.pretty_name = 'VLMs Are Biased (v1 legacy)'

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert one visual question into an EvalScope sample."""
        if self.data_version == 'v2' and not record.get('legacy_ID'):
            raise ValueError(
                'VLMs Are Biased v2 requires legacy_ID. Select default_subset="v1-legacy" '
                'explicitly when evaluating v1 data.'
            )
        image = self._image_bytes_to_base64(record['image']['bytes'], guess_mimetype=True)
        expected_bias = record.get('expected_bias')
        return Sample(
            input=[
                ChatMessageUser(
                    content=[
                        ContentImage(image=image),
                        ContentText(text=record['prompt']),
                    ]
                )
            ],
            target=str(record['ground_truth']).strip(),
            metadata={
                'id': record['ID'],
                'legacy_id': record.get('legacy_ID'),
                'data_version': self.data_version,
                'topic': record['topic'],
                'sub_topic': record['sub_topic'],
                'type_of_question': record['type_of_question'],
                'expected_bias': expected_bias,
                'with_title': record['with_title'],
                'pixel': record['pixel'],
            },
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Normalize the model reply according to the official scorer."""
        return normalize_answer(prediction)

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        """Score visual accuracy and prior-knowledge bias alignment."""
        score = Score(extracted_prediction=filtered_prediction, prediction=original_prediction)
        score.value = score_answer(
            prediction=filtered_prediction,
            ground_truth=reference,
            expected_bias=(task_state.metadata or {}).get('expected_bias'),
        )
        score.main_score_name = 'acc'
        return score

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """Aggregate official overall metrics and per-topic accuracy."""
        aggregate_scores = super().aggregate_scores(sample_scores)
        for aggregate_score in aggregate_scores:
            aggregate_score.dimensions = {'scope': 'overall'}
        topics = sorted(
            {topic for sample_score in sample_scores if (topic := (sample_score.sample_metadata or {}).get('topic'))}
        )
        for topic in topics:
            topic_scores = [
                sample_score
                for sample_score in sample_scores
                if (sample_score.sample_metadata or {}).get('topic') == topic
            ]
            accuracy = sum(float(sample_score.score.value['acc']) for sample_score in topic_scores) / len(topic_scores)
            aggregate_scores.append(
                AggScore(
                    metric_name='acc',
                    aggregation='mean',
                    dimensions={'topic': topic},
                    score=accuracy,
                    num=len(topic_scores),
                    ids=[sample_score.sample_id for sample_score in topic_scores],
                )
            )
        return aggregate_scores
