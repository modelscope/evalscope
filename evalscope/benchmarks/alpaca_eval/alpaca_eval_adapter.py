import re
from collections import defaultdict
from typing import Any, List

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.metrics import Metric, mean, metric_registry
from evalscope.metrics.llm_judge import LLMJudge
from evalscope.utils.logger import get_logger

# flake8: noqa

logger = get_logger()

GRADER_TEMPLATE = """
I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

## Instruction

{
    "instruction": "{question}",
}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{
    {
        "model_identifier": "m",
        "output": "{prediction}"
    },
    {
        "model_identifier": "M",
        "output": "{prediction2}"
    }
}

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
        metric_registry.register(Metric(name='is_correct', object=mean))
        metric_registry.register(Metric(name='is_incorrect', object=mean))
        metric_registry.register(Metric(name='is_not_attempted', object=mean))

        # whether to use LLM as a judge
        self.llm_as_a_judge = True

    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> dict:
        question = input_d['problem']
        return self.gen_prompt_data(question)

    def get_gold_answer(self, input_d: dict) -> str:
        return input_d['answer']

    def parse_pred_result(self, result: str, raw_input_d: dict = None, **kwargs) -> str:
        return result.strip()

    def match(self, gold: str, pred: str) -> float:
        # simple match
        logger.warning(f'Please use LLMJudge to match the result for SimpleQA')
        is_correct = 1 if gold.lower().strip() == pred.lower().strip() else 0
        is_incorrect = not is_correct
        is_not_attempted = 0
        return {
            'is_correct': is_correct,
            'is_incorrect': is_incorrect,
            'is_not_attempted': is_not_attempted,
        }

    def llm_match(self, gold: Any, pred: Any, judge: LLMJudge, **kwargs) -> dict:
        raw_input = kwargs.get('raw_input', None)
        question = raw_input['problem']
        # get grading response
        prompt = GRADER_TEMPLATE.format(question=question, target=gold, predicted_answer=pred)
        grading_response = judge(prompt)
        # parse grading response
        match = re.search(r'(A|B|C)', grading_response)
        res = match.group(0) if match else 'C'
        return {
            'is_correct': 1 if res == 'A' else 0,
            'is_incorrect': 1 if res == 'B' else 0,
            'is_not_attempted': 1 if res == 'C' else 0,
        }

    def compute_metric(self, review_res_list: List[dict], **kwargs) -> List[dict]:
        """
        compute weighted mean of the bleu score of all samples

        Args:
            review_res_list: [{'is_correct': 1, 'is_incorrect': 0, 'is_not_attempted': 0}, ...]
        """
        # zip dict answers
        res_dict = defaultdict(list)
        for res in review_res_list:
            for key, value in res.items():
                res_dict[key].append(value)

        return super().compute_metric(res_dict, **kwargs)
