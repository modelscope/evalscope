import re
from collections import defaultdict
from typing import Any, List

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.metrics import LLMJudge, Metric, mean, metric_registry
from evalscope.utils.logger import get_logger

# flake8: noqa

logger = get_logger()

GRADER_SYSTEM_PROMPT = """You are a highly efficient assistant, who evaluates and selects the best large language model (LLMs) based on the quality of their responses to a given instruction. This process will be used to create a leaderboard reflecting the most accurate and human-preferred answers."""

GRADER_TEMPLATE = """
I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

## Instruction

{{
    "instruction": "{instruction}"
}}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{{
    {{
        "model_identifier": "m",
        "output": "{output_1}"
    }},
    {{
        "model_identifier": "M",
        "output": "{output_2}"
    }}
}}

## Task

Evaluate the models based on the quality and relevance of their outputs, and select the model that generated the best output. Answer by providing the model identifier of the best model. We will use your output as the name of the best model, so make sure your output only contains one of the following model identifiers and nothing else (no quotes, no spaces, no new lines, ...): m or M.

## Best Model Identifier
""".strip()  # noqa: E501


@Benchmark.register(
    name='alpaca_eval',
    pretty_name='AlpacaEval2.0',
    dataset_id='AI-ModelScope/alpaca_eval',
    subset_list=['alpaca_eval_gpt4_baseline'],
    metric_list=['winrate'],
    few_shot_num=0,
    train_split=None,
    eval_split='eval')
class AlpacaEvalAdapter(DataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # register metrics
        metric_registry.register(Metric(name='winrate', object=mean))

        # whether to use LLM as a judge
        self.llm_as_a_judge = True

    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> dict:
        question = input_d['instruction']
        return self.gen_prompt_data(question)

    def get_gold_answer(self, input_d: dict) -> str:
        return input_d['output']

    def parse_pred_result(self, result: str, raw_input_d: dict = None, **kwargs) -> str:
        return result.strip()

    def match(self, gold: str, pred: str):
        # simple match
        logger.warning(f'Please use LLMJudge to match the result for {self.name}')
        return None

    def llm_match(self, gold: Any, pred: Any, judge: LLMJudge, **kwargs) -> bool:
        raw_input = kwargs.get('raw_input', None)
        instruction = raw_input['instruction']
        # gold is baseline answer 'm', pred is model answer 'M'
        prompt = GRADER_TEMPLATE.format(instruction=instruction, output_1=gold, output_2=pred)
        # get grading response
        grading_response = judge(prompt, system_prompt=GRADER_SYSTEM_PROMPT)
        # parse grading response
        match = re.search(r'(m|M)', grading_response)
        res = match.group(0) if match else None
        if res:
            return res == 'M'
        else:
            logger.info(f'Failed to parse grading response: {prompt=}\n {grading_response=}')
            return None

    def compute_metric(self, review_res_list: List[bool], **kwargs) -> List[dict]:
        # zip dict answers
        res_list = [res for res in review_res_list if res is not None]

        return super().compute_metric(res_list, **kwargs)
