from typing import Any, Dict, List

from evalscope.api.dataset.dataset import Sample
from evalscope.api.evaluator import Choices, Target, TaskState
from evalscope.api.messages import ChatMessage, ChatMessageSystem, ChatMessageUser
from evalscope.utils.multi_choices import (
    FEW_SHOT_TEMPLATE,
    MultipleChoiceTemplate,
    answer_character,
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.multiple_correct: bool = False
        """Whether the benchmark allows multiple correct answers."""

    def validate_choice_config(self) -> None:
        """Validate the benchmark's single-choice execution contract."""
        super().validate_choice_config()
        if self.multiple_correct:
            raise ValueError('System One Choice evaluation requires a single correct answer.')
        if self.extra_params.get('use_cot'):
            raise ValueError('System One Choice does not generate chain-of-thought text; disable use_cot.')

    def choice_context(self, sample: Sample) -> Dict[str, Any]:
        """Return audited auxiliary context without copying gold-bearing metadata."""
        subject = sample.metadata.get('subject')
        return {'subject': str(subject).replace('_', ' ')} if subject else {}

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
        instructions = self._benchmark_meta.choice_instructions or 'Which option correctly answers `question`?'
        state = {'question': sample.input, **self.choice_context(sample)}
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
                        'state': state,
                        'instructions': instructions,
                        'criteria': criteria,
                        'answer_prefix': '答案：' if '答案：' in (self.prompt_template or '') else 'ANSWER: ',
                    }
                },
            )
        )
        return messages

    def _post_process_samples(self) -> None:
        if self._task_config is None or self.eval_type != 'systemone_api':
            super()._post_process_samples()
            return
        for subset, dataset in self.test_dataset.items():
            for sample in dataset:
                sample.input = self.build_systemone_messages(sample, subset)

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
