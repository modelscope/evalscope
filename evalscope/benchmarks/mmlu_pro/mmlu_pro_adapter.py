from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, MultiChoiceAdapter
from evalscope.api.dataset import Sample
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

# Based on the prompt provided here:
# https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/mmlu_pro
SYSTEM_W_EXAMPLES_PROMPT_TEMPLATE = """
The following are multiple choice questions (with answers) about {subject}. Think step by step and then finish your answer with 'ANSWER: [LETTER]' (without quotes) where [LETTER] is the correct letter choice.

{examples}
""".lstrip()  # noqa: E501

# Based on MultipleChoiceTemplate.SINGLE_ANSWER provided in the multiple choice solver:
# https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/src/inspect_ai/solver/_multiple_choice.py
USER_PROMPT_TEMPLATE = """Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

Question:
{question}
Options:
{choices}
""".lstrip()  # noqa: E501

SUBSET_LIST = [
    'computer science', 'math', 'chemistry', 'engineering', 'law', 'biology', 'health', 'physics', 'business',
    'philosophy', 'economics', 'other', 'psychology', 'history'
]


@register_benchmark(
    BenchmarkMeta(
        name='mmlu_pro',
        pretty_name='MMLU-Pro',
        tags=[Tags.MULTIPLE_CHOICE, Tags.KNOWLEDGE],
        description=
        'MMLU-Pro is a benchmark for evaluating language models on multiple-choice questions across various subjects. It includes questions from different domains, where the model must select the correct answer from given options.',  # noqa: E501
        dataset_id='TIGER-Lab/MMLU-Pro',
        subset_list=SUBSET_LIST,
        metric_list=['acc'],
        few_shot_num=5,
        train_split='validation',
        eval_split='test',
        prompt_template=USER_PROMPT_TEMPLATE,
        few_shot_prompt_template=SYSTEM_W_EXAMPLES_PROMPT_TEMPLATE + USER_PROMPT_TEMPLATE,
    )
)
class MMLUProAdapter(MultiChoiceAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.reformat_subset = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return Sample(
            input=record['question'],
            choices=record['options'],
            target=record['answer'],
            subset_key=record['category'].lower(),
            metadata={
                'cot_content': record['cot_content'],
                'subject': record['category'].lower(),
                'question_id': record['question_id'],
            },
        )

    def sample_to_fewshot(self, sample: Sample) -> str:
        q_str = f"""Question:\n{str(sample.input)}"""
        options = sample.choices if sample.choices is not None else []
        opt_str_list = []
        for i, opt in enumerate(options):
            opt_str_list.append(f"""{chr(65 + i)} {opt}""")
        opt_str = '\n'.join(opt_str_list)
        opt_str = f"""Options:\n{opt_str}"""
        ans_str = sample.metadata['cot_content'] if sample.metadata is not None else ''
        ans_str = ans_str.replace('The answer is', 'ANSWER:')
        ans_opt = ans_str.split('ANSWER:')[-1].split('.')[0].strip().strip('(').strip(')')
        ans_str = ans_str.replace(f'ANSWER: ({ans_opt})', f'ANSWER: {ans_opt}')
        final_str = '\n'.join([q_str, opt_str, ans_str])

        return final_str

    def format_fewshot_template(self, fewshot, sample):
        fewshot_str = SYSTEM_W_EXAMPLES_PROMPT_TEMPLATE.format(
            subject=sample.metadata['subject'],
            examples=fewshot,
        )
        prompt_str = self.format_prompt_template(sample)
        return fewshot_str + '\n' + prompt_str
