from evalscope.api.dataset.dataset import Sample
from evalscope.api.evaluator import Choices, Target
from evalscope.utils.multi_choices import FEW_SHOT_TEMPLATE, format_example, parse_answers, prompt, valid_template
from .default_data_adapter import DefaultDataAdapter


class MultiChoiceAdapter(DefaultDataAdapter):
    """
    Adapter for multi-choice benchmarks.
    This adapter formats the input for multi-choice questions and handles few-shot examples.
    """

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

    def extract_answer(self, prediction, task_state) -> str:
        answers = parse_answers(task_state, multiple_correct=False)
        return ''.join(list(answers))
