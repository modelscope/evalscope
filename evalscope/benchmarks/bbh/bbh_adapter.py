# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import re
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

# BBH multiple choice subset list
MULTIPLE_CHOICE = 'multiple_choice'
MULTIPLE_CHOICE_LIST = [
    'temporal_sequences',
    'disambiguation_qa',
    'date_understanding',
    'tracking_shuffled_objects_three_objects',
    'penguins_in_a_table',
    'geometric_shapes',
    'snarks',
    'ruin_names',
    'tracking_shuffled_objects_seven_objects',
    'tracking_shuffled_objects_five_objects',
    'logical_deduction_three_objects',
    'hyperbaton',
    'logical_deduction_five_objects',
    'logical_deduction_seven_objects',
    'movie_recommendation',
    'salient_translation_error_detection',
    'reasoning_about_colored_objects',
]

# The free form subset list of BBH dataset
FREE_FORM = 'free_form'
FREE_FORM_LIST = [
    'multistep_arithmetic_two',
    'navigate',
    'dyck_languages',
    'word_sorting',
    'sports_understanding',
    'boolean_expressions',
    'object_counting',
    'formal_fallacies',
    'causal_judgement',
    'web_of_lies',
]

# BBH sub-task type
TASK_TYPE = 'task_type'
SUBSET_LIST = MULTIPLE_CHOICE_LIST + FREE_FORM_LIST

PROMPT_TEMPLATE = """
Q: {question}
A: Let's think step by step. Put your final answer in the format of "So the answer is [ANSWER]" (without quotes and markdown) where [ANSWER] is the answer to the problem.
""".lstrip()  # noqa: E501

FEWSHOT_TEMPLATE = """
{fewshot}

""".lstrip() + PROMPT_TEMPLATE


@register_benchmark(
    BenchmarkMeta(
        name='bbh',
        pretty_name='BBH',
        dataset_id='evalscope/bbh',
        tags=[Tags.REASONING],
        description=
        'The BBH (Big Bench Hard) benchmark is a collection of challenging tasks designed to evaluate the reasoning capabilities of AI models. It includes both free-form and multiple-choice tasks, covering a wide range of reasoning skills.',  # noqa: E501
        subset_list=SUBSET_LIST,
        few_shot_num=3,
        train_split=None,
        eval_split='test',
        metric_list=['acc'],
        prompt_template=PROMPT_TEMPLATE,
        few_shot_prompt_template=FEWSHOT_TEMPLATE,
    )
)
class BBHAdapter(DefaultDataAdapter):
    """
    Adapter for BBH free-form and multiple-choices sub-tasks.
    """

    def __init__(self, **kwargs):
        few_shot_num = kwargs.get('few_shot_num', 3)

        if few_shot_num != 3 and few_shot_num != 0:
            logger.error(
                f'BBH uses 3-shot examples with CoT or 0-shot by system, but got {few_shot_num}. '
                f'Use 3-shot by default.'
            )
            kwargs['few_shot_num'] = 3

        super().__init__(**kwargs)

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        input = record['input']
        target = record['target'].replace('(', '').replace(')', '').strip()  # Clean up the target answer

        # Determine task type based on subset name
        task_type = None
        subset_name = self.current_subset_name
        if subset_name in MULTIPLE_CHOICE_LIST:
            task_type = MULTIPLE_CHOICE
        elif subset_name in FREE_FORM_LIST:
            task_type = FREE_FORM

        metadata = {TASK_TYPE: task_type}

        return Sample(input=input, target=target, metadata=metadata, subset_key=subset_name)

    def format_fewshot_template(self, fewshot: str, sample: Sample) -> str:
        # Load CoT prompts from file for BBH
        subset_name = sample.subset_key
        if subset_name:
            cot_file_path = os.path.join(os.path.dirname(__file__), 'cot_prompts', f'{subset_name}.txt')
            if os.path.exists(cot_file_path):
                with open(cot_file_path, 'r', encoding='utf-8') as f:
                    fewshot = f.read().strip()
        return self.few_shot_prompt_template.format(
            fewshot=fewshot,
            question=sample.input,
        )

    def extract_answer(self, prediction: str, task_state: TaskState):
        task_type = task_state.metadata.get(TASK_TYPE)

        if task_type == MULTIPLE_CHOICE:
            return self._extract_mc_answer(prediction)
        elif task_type == FREE_FORM:
            return self._extract_ff_answer(prediction)
        else:
            return prediction.strip()

    @classmethod
    def _extract_mc_answer(cls, ans: str) -> str:
        """
        Extract normalized answer for BBH multiple-choice tasks.
        Handles formats like:
        - "answer is (A)"
        - "The answer is A."
        - Extra text after answer.
        Always uses the *last* occurrence of "answer is".
        """
        ans = ans.strip()

        parts = ans.split('So the answer is ')
        if len(parts) > 1:
            ans = parts[-1].strip()
        ans = ans.split('\n')[0].strip()

        # Remove trailing period
        if ans.endswith('.'):
            ans = ans[:-1].strip()

        # Capture uppercase letter inside parentheses (A) (B) ...
        match = re.search(r'\(([A-Z])\)', ans)
        if match:
            return match.group(1)

        # Capture single uppercase letter
        match = re.search(r'\b([A-Z])\b', ans)
        if match:
            return match.group(1)

        return ans

    @classmethod
    def _extract_ff_answer(cls, ans: str):
        """
        Extract the normalized answer for BBH free-form tasks.
        Handles patterns like:
        - "answer is XXX."
        - "The answer is **valid**."
        - Extra trailing dots / line breaks.
        - Bold-marked answers (**xxx**).
        Always uses the *last* occurrence of "answer is".
        """
        ans = ans.strip()

        parts = ans.split('So the answer is ')
        if len(parts) > 1:
            ans = parts[-1].strip()
        ans = ans.split('\n')[0].strip()

        # Remove trailing period
        if ans.endswith('.'):
            ans = ans[:-1].strip()

        # If answer is in bold (**xxx**), prefer the content inside
        match = re.search(r'\*\*(.*?)\*\*', ans)
        if match:
            ans = match.group(1).strip()

        return ans
