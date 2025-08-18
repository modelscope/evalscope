import re
from typing import Any, Dict

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

GRADER_SYSTEM_PROMPT = """You are a highly efficient assistant, who evaluates and selects the best large language model (LLMs) based on the quality of their responses to a given instruction. This process will be used to create a leaderboard reflecting the most accurate and human-preferred answers."""  # noqa: E501

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


@register_benchmark(
    BenchmarkMeta(
        name='alpaca_eval',
        pretty_name='AlpacaEval2.0',
        tags=[Tags.INSTRUCTION_FOLLOWING, Tags.ARENA],
        description='Alpaca Eval 2.0 is an enhanced framework for evaluating instruction-following language models, '
        'featuring an improved auto-annotator, updated baselines, and continuous preference calculation to '
        'provide more accurate and cost-effective model assessments. '
        'Currently not support `length-controlled winrate`; the official Judge model is `gpt-4-1106-preview`, while the baseline model is `gpt-4-turbo`.',  # noqa: E501
        dataset_id='AI-ModelScope/alpaca_eval',
        subset_list=['alpaca_eval_gpt4_baseline'],
        metric_list=['winrate'],
        few_shot_num=0,
        train_split=None,
        eval_split='eval',
        prompt_template='{question}'
    )
)
class AlpacaEvalAdapter(DefaultDataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._use_llm_judge = True  # Use LLM as a judge by default

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """
        Convert a data record to a Sample object.

        Args:
            record (Dict[str, Any]): Input data record.

        Returns:
            Sample: Sample object with input, target, and metadata.
        """
        instruction = record['instruction']
        baseline_output = record['output']  # baseline model output

        return Sample(
            input=instruction,
            target=baseline_output,
            metadata={
                'generator': record.get('generator', 'unknown'),
                'dataset': record.get('dataset', 'unknown')
            }
        )

    def llm_match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        instruction = task_state.input_text

        # Request judge and obtain score
        # reference is baseline answer 'm', filtered_prediction is model answer 'M'
        prompt = GRADER_TEMPLATE.format(instruction=instruction, output_1=reference, output_2=filtered_prediction)
        judge_response = self.llm_judge.judge(prompt, system_prompt=GRADER_SYSTEM_PROMPT)

        # parse grading response
        match = re.search(r'(m|M)', judge_response)
        res = match.group(0) if match else None

        if res:
            winrate = 1 if res == 'M' else 0
        else:
            logger.info(f'Failed to parse grading response: {prompt=}\n {judge_response=}')
            winrate = 0

        # Set score based on the match result
        score.value = {'winrate': winrate}
        score.explanation = f'LLM judge: {judge_response}'
        score.metadata = {
            'source': 'llm_judge',
            'judge_strategy': self.judge_strategy,
            'model': self.llm_judge.model_id
        }
        score.main_score_name = 'winrate'
        return score
