import json
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric.scorer import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

logger = get_logger()

# 定义提示模板
PROMPT_TEMPLATE = """
# Example Puzzle

There are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. \
Each house is occupied by a different person. \
Each house has a unique attribute for each of the following characteristics:
 - Each person has a unique name: `Peter`, `Eric`, `Arnold`.
 - Each person has a unique favorite drink: `tea`, `water`, `milk`

## Clues for the Example Puzzle

1. Peter is in the second house.
2. Arnold is directly left of the one who only drinks water.
3. The one who only drinks water is directly left of the person who likes milk.

## Answer to the Example Puzzle

{{
    "reasoning": "Given Clue 1, we know Peter is in House 2. According to Clue 2, \
Arnold is directly left of the one who only drinks water. \
The person in House 3 cannot be on the left of anyone, so Arnold must be in House 1. \
Thus, Peter drinks water, and Eric lives in House 3. \
Then, according to Clue 3, Eric drinks milk. Therefore, Arnold drinks tea.",
    "solution": {{
        "House 1": {{
            "Name": "Arnold",
            "Drink": "tea"
        }},
        "House 2": {{
            "Name": "Peter",
            "Drink": "water"
        }},
        "House 3": {{
            "Name": "Eric",
            "Drink": "milk"
        }}
    }}
}}

# Puzzle to Solve

{question}


# Instruction

Now please solve the above puzzle. Present your reasoning and solution in the following json format:

{json_template}

""".lstrip()


# 注册基准评测
@register_benchmark(
    BenchmarkMeta(
        name='zebralogicbench',
        pretty_name='ZebraLogicBench',
        dataset_id='allenai/ZebraLogicBench-private',
        tags=[Tags.REASONING],
        description='ZebraLogic, a comprehensive evaluation framework for '
        'assessing LLM reasoning performance on logic grid puzzles '
        'derived from constraint satisfaction problems (CSPs).',
        subset_list=['grid_mode'],
        few_shot_num=0,
        eval_split='test',
        prompt_template=PROMPT_TEMPLATE,
        metric_list=[
            'puzzle_acc', 'cell_acc', 'easy_puzzle_acc', 'hard_puzzle_acc', 'small_puzzle_acc', 'medium_puzzle_acc',
            'large_puzzle_acc', 'xl_puzzle_acc', 'avg_reason_lens', 'no_answer_num'
        ]
    )
)
class ZebraLogicBenchAdapter(DefaultDataAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert one dataset record to the Sample format used by evaluator."""
        puzzle = record['puzzle']
        solution = json.dumps(record['solution'])
        created_at = record['created_at']
        return Sample(input=puzzle, target=solution, metadata={'created_at': created_at})

    def format_prompt_template(self, sample: Sample) -> str:
        """
        Format the basic prompt template with the sample data.

        This method applies the prompt template to format the input text
        for models when no few-shot examples are used.
        """
        target = json.loads(sample.target)
        num_houses = len(target['rows'])
        columns = target['header']
        assert columns[0] == 'House'
        json_template = {'reasoning': '___', 'solution': {}}
        for i in range(num_houses):
            json_template['solution'][f'House {i + 1}'] = {columns[j]: '___' for j in range(1, len(columns))}
        json_str = json.dumps(json_template, indent=4)
        return self.prompt_template.format(question=sample.input, json_template=json_str)

    def extract_answer(self, prediction: str, task_state: TaskState) -> str:
        """Extract the last complete JSON object from model output; return empty string if not found."""

        def extract_last_complete_json(s: str) -> str:
            stack: List[int] = []
            last_json_start: int | None = None
            last_json_str: str | None = None

            for i, char in enumerate(s):
                if char == '{':
                    if not stack:
                        last_json_start = i
                    stack.append(i)
                elif char == '}':
                    if stack:
                        stack.pop()
                        if not stack and last_json_start is not None:
                            # Found a complete JSON object
                            candidate = s[last_json_start:i + 1]
                            last_json_str = candidate
                            last_json_start = None

            if last_json_str:
                try:
                    # Normalize JSON to a compact string
                    return json.dumps(json.loads(last_json_str))
                except json.JSONDecodeError:
                    return ''
            return ''

        return extract_last_complete_json(prediction)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """Compute ZebraLogicBench per-sample scores using provided utils."""
        from evalscope.benchmarks.zebralogicbench.utils import process_results

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )
        try:
            results: Dict[str,
                          float] = {} if not filtered_prediction else process_results(filtered_prediction, reference)
            score.value.update(results)
            score.main_score_name = 'puzzle_acc'
        except Exception as e:
            logger.error(f'Error calculating zebralogicbench metrics: {e}')
            score.value = {}
        return score

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """Aggregate metrics across all samples."""
        total_puzzle_num = len(sample_scores)
        total_cell_num = sum(ss.score.value.get('Cell Num', 0.0) for ss in sample_scores)
        no_answer_num = sum(ss.score.value.get('No answer', 0.0) for ss in sample_scores)

        total_easy_puzzle_num = sum(ss.score.value.get('Easy Puzzle Num', 0.0) for ss in sample_scores)
        total_hard_puzzle_num = sum(ss.score.value.get('Hard Puzzle Num', 0.0) for ss in sample_scores)

        total_small_puzzle_num = sum(ss.score.value.get('Small Puzzle Num', 0.0) for ss in sample_scores)
        total_medium_puzzle_num = sum(ss.score.value.get('Medium Puzzle Num', 0.0) for ss in sample_scores)
        total_large_puzzle_num = sum(ss.score.value.get('Large Puzzle Num', 0.0) for ss in sample_scores)
        total_xl_puzzle_num = sum(ss.score.value.get('XL Puzzle Num', 0.0) for ss in sample_scores)

        puzzle_acc = (
            sum(ss.score.value.get('Solved Puzzle', 0.0)
                for ss in sample_scores) / total_puzzle_num if total_puzzle_num > 0 else 0.0
        )
        cell_acc = (
            sum(ss.score.value.get('Solved Cell', 0.0)
                for ss in sample_scores) / total_cell_num if total_cell_num > 0 else 0.0
        )

        easy_puzzle_acc = (
            sum(ss.score.value.get('Solved Easy Puzzle', 0.0)
                for ss in sample_scores) / total_easy_puzzle_num if total_easy_puzzle_num > 0 else 0.0
        )
        hard_puzzle_acc = (
            sum(ss.score.value.get('Solved Hard Puzzle', 0.0)
                for ss in sample_scores) / total_hard_puzzle_num if total_hard_puzzle_num > 0 else 0.0
        )

        small_puzzle_acc = (
            sum(ss.score.value.get('Solved Small Puzzle', 0.0)
                for ss in sample_scores) / total_small_puzzle_num if total_small_puzzle_num > 0 else 0.0
        )
        medium_puzzle_acc = (
            sum(ss.score.value.get('Solved Medium Puzzle', 0.0)
                for ss in sample_scores) / total_medium_puzzle_num if total_medium_puzzle_num > 0 else 0.0
        )
        large_puzzle_acc = (
            sum(ss.score.value.get('Solved Large Puzzle', 0.0)
                for ss in sample_scores) / total_large_puzzle_num if total_large_puzzle_num > 0 else 0.0
        )
        xl_puzzle_acc = (
            sum(ss.score.value.get('Solved XL Puzzle', 0.0)
                for ss in sample_scores) / total_xl_puzzle_num if total_xl_puzzle_num > 0 else 0.0
        )

        avg_reason_lens = (
            sum(ss.score.value.get('Reason Lens', 0.0)
                for ss in sample_scores) / total_puzzle_num if total_puzzle_num > 0 else 0.0
        )

        agg_scores = [
            AggScore(metric_name='puzzle_acc', score=puzzle_acc, num=total_puzzle_num, metadata={'type': 'puzzle_acc'}),
            AggScore(metric_name='cell_acc', score=cell_acc, num=total_cell_num, metadata={'type': 'cell_acc'}),
            AggScore(
                metric_name='easy_puzzle_acc',
                score=easy_puzzle_acc,
                num=total_easy_puzzle_num,
                metadata={'type': 'easy_puzzle_acc'}
            ),
            AggScore(
                metric_name='hard_puzzle_acc',
                score=hard_puzzle_acc,
                num=total_hard_puzzle_num,
                metadata={'type': 'hard_puzzle_acc'}
            ),
            AggScore(
                metric_name='small_puzzle_acc',
                score=small_puzzle_acc,
                num=total_small_puzzle_num,
                metadata={'type': 'small_puzzle_acc'}
            ),
            AggScore(
                metric_name='medium_puzzle_acc',
                score=medium_puzzle_acc,
                num=total_medium_puzzle_num,
                metadata={'type': 'medium_puzzle_acc'}
            ),
            AggScore(
                metric_name='large_puzzle_acc',
                score=large_puzzle_acc,
                num=total_large_puzzle_num,
                metadata={'type': 'large_puzzle_acc'}
            ),
            AggScore(
                metric_name='xl_puzzle_acc',
                score=xl_puzzle_acc,
                num=total_xl_puzzle_num,
                metadata={'type': 'xl_puzzle_acc'}
            ),
            AggScore(
                metric_name='avg_reason_lens',
                score=avg_reason_lens,
                num=total_puzzle_num,
                metadata={'type': 'avg_reason_lens'}
            ),
            AggScore(
                metric_name='no_answer_num', score=no_answer_num, num=no_answer_num, metadata={'type': 'no_answer_num'}
            )
        ]
        return agg_scores
