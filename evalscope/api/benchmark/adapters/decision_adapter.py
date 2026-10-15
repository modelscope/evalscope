"""Native single-question datasets using the existing inference and reporting pipeline."""

import json
from typing import Any

from evalscope.api.dataset import DatasetDict, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageSystem, ChatMessageUser
from evalscope.api.metric import AggScore, MetricUnavailableError, SampleScore, Score
from evalscope.api.metric.decision import decision_score, read_decision_answer
from evalscope.api.model import Model
from evalscope.api.model.decision import DECISION_EVAL_TYPES, DecisionInput
from evalscope.api.registry import get_aggregation

from .default_data_adapter import DefaultDataAdapter


class DecisionAdapter(DefaultDataAdapter):
    """Read one typed decision per record without exposing gold metadata to the provider."""

    doc_eval_type = 'systemone_api'

    probability_sum_tolerance: float | None = None
    normalization_threshold: float = 0.0
    noul_use_argmax: bool = False

    def load_dataset(self) -> DatasetDict:
        """Reject unsupported chat execution before downloading the decision dataset."""
        if self._task_config is not None and self.eval_type not in DECISION_EVAL_TYPES:
            self.validate_choice_config()
        return super().load_dataset()

    def validate_choice_config(self) -> None:
        """Require a decision API and deterministic grading."""
        if self.eval_type not in DECISION_EVAL_TYPES:
            raise ValueError('Typed decision datasets require systemone_api or autotrust_api.')
        if self.few_shot_num or self.filters or self.use_llm_judge or self._task_config.agent_config is not None:
            raise ValueError('Typed decision datasets require zero-shot rule grading without agents or filters.')

    def _on_inference_start(self, model: Model, sample: Sample) -> None:
        """Allow dataset inspection independently, but reject incompatible inference APIs."""
        self.validate_choice_config()

    def record_to_sample(self, record: dict[str, Any]) -> Sample:
        if 'record_json' in record:
            record = json.loads(record['record_json'])
        question = record['question']
        request = DecisionInput(
            kind=question['type'],
            state=record['state'],
            instructions=question['instructions'],
            criteria=question.get('criteria'),
        )
        gold = record['expected']
        if request.kind == 'noul':
            if str(gold).lower() not in ('yes', 'true', 'no', 'false'):
                raise ValueError('Noul gold must be a boolean or yes/no label.')
            gold = 'yes' if gold is True or str(gold).lower() in ('yes', 'true') else 'no'
        if not record.get('continuous') and str(gold) not in request.labels:
            raise ValueError(f'Gold answer is not an allowed label: {record.get("id")}')
        if record.get('continuous') and (request.kind != 'score' or not 0 <= float(gold) <= len(request.labels) - 1):
            raise ValueError('Continuous gold must be inside a Score scale.')
        messages = []
        if self.system_prompt:
            messages.append(ChatMessageSystem(content=self.system_prompt))
        messages.append(
            ChatMessageUser(
                content=json.dumps({'state': request.state, 'question': question}, ensure_ascii=False),
                internal={'systemone': request.model_dump()},
            )
        )
        return Sample(
            input=messages,
            target=str(gold),
            metadata={
                'source_id': record.get('id'),
                'decision_gold': gold,
                'decision_kind': request.kind,
                'source_group': record.get('group'),
                'family': record.get('family', ''),
                'tier': record.get('tier', ''),
                'language': record.get('language', ''),
                'continuous': record.get('continuous', False),
                'decision_labels': request.labels,
                'fixed_classes': record.get('fixed_classes', False),
            },
        )

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        return prediction

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        answer = read_decision_answer(task_state)
        deviation = abs(sum(answer.probabilities.values()) - 1)
        if self.probability_sum_tolerance is not None and deviation > self.probability_sum_tolerance:
            raise MetricUnavailableError("Distribution exceeds this benchmark's probability sum tolerance.")
        continuous = task_state.metadata.get('continuous', False)
        result = decision_score(
            answer,
            float(reference) if continuous else reference,
            task_state.metadata['decision_labels'],
            continuous=continuous,
            threshold=self.extra_params.get('decision_threshold', 0.5),
            family=task_state.metadata.get('family', ''),
            tier=task_state.metadata.get('tier', ''),
            fixed_classes=task_state.metadata.get('fixed_classes', False),
            normalize=deviation > self.normalization_threshold,
            noul_use_argmax=self.noul_use_argmax,
        )
        result.prediction = original_prediction
        return result

    def aggregate_scores(self, sample_scores: list[SampleScore]) -> list[AggScore]:
        """Use the registered probability-aware aggregation implementation."""
        return get_aggregation('decision')()(sample_scores)
