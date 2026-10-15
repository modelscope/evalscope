from typing import List

from evalscope.api.dataset.dataset import Sample
from evalscope.api.evaluator import Choices, Target, TaskState
from evalscope.api.messages import ChatMessage, ChatMessageSystem, ChatMessageUser
from evalscope.api.metric import AggScore, MetricSelector, SampleScore, Score
from evalscope.api.metric.decision import decision_score, read_decision_answer
from evalscope.api.model.decision import DECISION_EVAL_TYPES
from evalscope.api.registry import get_aggregation
from evalscope.utils.multi_choices import (
    FEW_SHOT_TEMPLATE,
    MultipleChoiceTemplate,
    answer_character,
    answer_index,
    answer_options,
    format_example,
    parse_answers,
    parse_answers_zh,
    prompt,
    valid_template,
)

from .default_data_adapter import DefaultDataAdapter


class MultiChoiceAdapter(DefaultDataAdapter):
    """
    Adapter for multi-choice benchmarks.
    This adapter formats the input for multi-choice questions and handles few-shot examples.
    """

    decision_noul_positive_choice: str | None = None
    """Choice text that represents yes for an audited Noul conversion."""
    decision_fixed_classes: bool = False
    """Whether every sample uses the same semantic class inventory."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self._task_config is not None and self.eval_type in DECISION_EVAL_TYPES:
            self._benchmark_meta.primary_metric = MetricSelector(name='accuracy', aggregation='mean')
        elif self.extra_params.get('decision_kind', 'choice') != 'choice':
            raise ValueError('Noul conversion requires systemone_api or autotrust_api.')
        self.multiple_correct: bool = False
        """Whether the benchmark allows multiple correct answers."""

    def validate_choice_config(self) -> None:
        """Validate the benchmark's single-choice execution contract."""
        if not self._benchmark_meta.supports_choice:
            raise ValueError(f'Benchmark {self.name!r} has no audited text Choice conversion.')
        if self._task_config.agent_config is not None:
            raise ValueError('System One Choice evaluation does not support agent_config.')
        if self._task_config.judge.models or self._task_config.judge.strategy not in ('auto', 'rule'):
            raise ValueError('System One Choice evaluation uses deterministic scoring, without an LLM judge.')
        overrides = self._task_config.dataset_args.get(self.name, {})
        incompatible = {'prompt_template', 'few_shot_prompt_template', 'filters', 'query_template'} & overrides.keys()
        if incompatible:
            raise ValueError(
                f'System One cannot use {", ".join(sorted(incompatible))}; configure choice_instructions instead.'
            )
        if self.filters:
            raise ValueError('System One Choice evaluation does not support output filters.')
        if self.multiple_correct:
            raise ValueError('System One Choice evaluation requires a single correct answer.')
        if self.extra_params.get('use_cot'):
            raise ValueError('System One Choice does not generate chain-of-thought text; disable use_cot.')
        if not 0 <= self.extra_params.get('decision_threshold', 0.5) <= 1:
            raise ValueError('Decision threshold must be between zero and one.')

    def choice_examples(self, subset: str) -> List[str]:
        """Reuse the selected demonstration samples or benchmark-specific fixed examples."""
        if self.few_shot_num == 0:
            return []
        if self.fewshot_dataset is None:
            raise ValueError(f'{self.name} needs a choice_examples() hook for fixed few-shot examples.')
        examples = self.fewshot_dataset.get(subset)
        if examples is None:
            examples = next(iter(self.fewshot_dataset.values()))
        if len(examples) < self.few_shot_num:
            raise ValueError(f'{self.name} has fewer examples than few_shot_num={self.few_shot_num}.')
        return [self.sample_to_fewshot(example) for example in examples[: self.few_shot_num]]

    def build_systemone_messages(self, sample: Sample, subset: str) -> List[ChatMessage]:
        """Prepare chat messages with the MCQ data needed by the System One provider."""
        if not isinstance(sample.input, str) or sample.tools:
            raise ValueError('System One Choice requires raw text input without tools.')
        criteria = {answer_character(i): value for i, value in enumerate(sample.choices or [])}
        target = Target(sample.target)
        if not 2 <= len(criteria) <= 255 or len(target) != 1 or target.single() not in criteria:
            raise ValueError('System One Choice requires 2-255 options and one matching target label.')
        kind = self.extra_params.get('decision_kind', 'choice')
        if kind != 'choice' and (kind != 'noul' or self.decision_noul_positive_choice is None):
            raise ValueError(f'Benchmark {self.name!r} has no audited Noul conversion.')
        instructions = self._benchmark_meta.choice_instructions or 'Which option correctly answers `question`?'
        state = {'question': sample.input}
        examples = self.choice_examples(subset)
        if examples:
            state['examples'] = examples
            instructions += '\nUse `examples` as demonstrations of the task.'
        messages: List[ChatMessage] = []
        if self.system_prompt:
            messages.append(ChatMessageSystem(content=self.system_prompt))
        messages.append(
            ChatMessageUser(
                content=f'{sample.input}\n\n{answer_options(sample.choices or [])}',
                internal={
                    'systemone': {
                        'kind': kind,
                        'state': state,
                        'instructions': instructions,
                        'criteria': criteria if kind == 'choice' else None,
                        'answer_prefix': '答案：' if '答案：' in (self.prompt_template or '') else 'ANSWER: ',
                    }
                },
            )
        )
        return messages

    def _post_process_samples(self) -> None:
        if self._task_config is None or self.eval_type not in DECISION_EVAL_TYPES:
            super()._post_process_samples()
            return
        for subset, dataset in self.test_dataset.items():
            for sample in dataset:
                sample.input = self.build_systemone_messages(sample, subset)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """Retain categorical probabilities on decision protocols and preserve ordinary MCQ grading."""
        if self._task_config is None or self.eval_type not in DECISION_EVAL_TYPES:
            return super().match_score(original_prediction, filtered_prediction, reference, task_state)
        answer = read_decision_answer(task_state, allow_legacy_choice=True)
        labels = list(answer.probabilities)
        gold = reference
        if answer.type == 'noul':
            labels = ['no', 'yes']
            choice = task_state.choices[answer_index(reference)].value
            gold = 'yes' if choice == self.decision_noul_positive_choice else 'no'
        score = decision_score(
            answer,
            gold,
            labels,
            threshold=self.extra_params.get('decision_threshold', 0.5),
            fixed_classes=self.decision_fixed_classes,
            class_labels={answer_character(i): choice.value for i, choice in enumerate(task_state.choices)}
            if self.decision_fixed_classes
            else None,
        )
        score.prediction = original_prediction
        return score

    def aggregate_scores(self, sample_scores: list[SampleScore]) -> list[AggScore]:
        """Compute cross-sample metrics only when a decision service supplied distributions."""
        if self._task_config is not None and self.eval_type in DECISION_EVAL_TYPES:
            return get_aggregation('decision')()(sample_scores)
        return super().aggregate_scores(sample_scores)

    def format_prompt_template(self, sample: Sample) -> str:
        """
        Format the basic prompt template with the sample data.

        Args:
            sample (Sample): The sample object containing the prompt data

        Returns:
            str: The formatted prompt ready for model input
        """
        assert valid_template(self.prompt_template), 'Prompt template is not valid'

        return prompt(
            question=sample.input,
            choices=Choices(sample.choices),
            template=self.prompt_template,
        )

    def format_fewshot_template(self, fewshot: str, sample: Sample) -> str:
        """
        Format the few-shot template with demonstrations and the main prompt.

        Args:
            fewshot (str): The formatted few-shot demonstration examples
            sample (Sample): The sample object containing the prompt data

        Returns:
            str: The complete formatted input with few-shot context
        """

        few_shot_prompt_template = self.few_shot_prompt_template or (FEW_SHOT_TEMPLATE + self.prompt_template)

        assert valid_template(few_shot_prompt_template), 'Few-shot prompt template is not valid'

        return prompt(
            question=sample.input, choices=Choices(sample.choices), template=few_shot_prompt_template, fewshot=fewshot
        )

    def sample_to_fewshot(self, sample: Sample) -> str:
        """
        Convert a sample to a few-shot formatted string.

        Args:
            sample (Sample): The sample object to format

        Returns:
            str: The formatted few-shot example string
        """
        return format_example(question=sample.input, choices=Choices(sample.choices), answer=Target(sample.target))

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        if self.prompt_template in [
            MultipleChoiceTemplate.CHINESE_SINGLE_ANSWER_TEMPLATE,
            MultipleChoiceTemplate.CHINESE_SINGLE_ANSWER_TEMPLATE_COT,
            MultipleChoiceTemplate.CHINESE_MULTIPLE_ANSWER_TEMPLATE,
            MultipleChoiceTemplate.CHINESE_MULTIPLE_ANSWER_TEMPLATE_COT,
        ]:
            # For Chinese templates, use the Chinese-format extractor ('答案：...')
            answers = parse_answers_zh(task_state, multiple_correct=self.multiple_correct, completion=prediction)
        else:
            answers = parse_answers(task_state, multiple_correct=self.multiple_correct, completion=prediction)
        return ''.join(sorted(list(answers)))
