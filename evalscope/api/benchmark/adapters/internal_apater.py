from evalscope.api.dataset.dataset import Sample
from evalscope.api.evaluator import Target, TaskState
from evalscope.utils.multi_choices import (
    WOChoiceMultipleChoiceTemplate,
    format_example,
    parse_answers,
    parse_answers_zh,
)
from .default_data_adapter import DefaultDataAdapter

class WOChoiceMultiChoiceAdapter(DefaultDataAdapter):
    """
    Apapter for multi-choice benchmarks whose choices are not separated from questions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # whether the benchmark allows multiple correct answers
        self.multiple_correct: bool =  False

    def sample_to_fewshot(self, sample: Sample) -> str:
        return format_example(question=sample.input, answer=Target(sample.target))

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        if self.prompt_template in [
            WOChoiceMultipleChoiceTemplate.CHINESE_SINGLE_ANSWER_TEMPLATE_COT,
            WOChoiceMultipleChoiceTemplate.CHINESE_SINGLE_ANSWER_TEMPLATE
        ]:
            # For Chinese COT template, we use a different extraction method
            answers = parse_answers_zh(task_state, multiple_correct=self.multiple_correct)
        else:
            answers = parse_answers(task_state, multiple_correct=self.multiple_correct)
        return ''.join(sorted(list(answers)))
