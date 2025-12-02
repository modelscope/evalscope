from typing import Any, Dict, List, Set, Tuple
from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.api.metric import Score
from evalscope.api.metric.scorer import AggScore, SampleScore, Score
from evalscope.utils.logger import get_logger

logger = get_logger()

# 定义提示模板
PROMPT_TEMPLATE = """
# Example Puzzle 

There are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:
 - Each person has a unique name: `Peter`, `Eric`, `Arnold`.
 - Each person has a unique favorite drink: `tea`, `water`, `milk`

## Clues for the Example Puzzle

1. Peter is in the second house.
2. Arnold is directly left of the one who only drinks water.
3. The one who only drinks water is directly left of the person who likes milk.

## Answer to the Example Puzzle

{{
    "reasoning": "Given Clue 1, we know Peter is in House 2. According to Clue 2, Arnold is directly left of the one who only drinks water. The person in House 3 cannot be on the left of anyone, so Arnold must be in House 1. Thus, Peter drinks water, and Eric lives in House 3. Then, according to Clue 3, Eric drinks milk. Therefore, Arnold drinks tea.",
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
        description='ZebraLogic, a comprehensive evaluation framework for assessing LLM reasoning performance on logic grid puzzles derived from constraint satisfaction problems (CSPs).',
        subset_list=['grid_mode'],
        few_shot_num=0,
        eval_split='test',
        prompt_template=PROMPT_TEMPLATE,
    )
)
class ZebraLogicBenchAdapter(DefaultDataAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """将原始数据记录转换为Sample对象"""
        import json
        id = record['id']
        size = record['size']
        puzzle = record['puzzle']
        solution = json.dumps(record['solution'])
        created_at = record['created_at']


        return Sample(
            input=puzzle,
            target=solution,
            metadata={'created_at': created_at}
        )

    def format_prompt_template(self, sample: Sample) -> str:
        """
        Format the basic prompt template with the sample data.

        This method applies the prompt template to format the input text
        for models when no few-shot examples are used.

        Args:
            sample (Sample): The sample object containing the prompt data

        Returns:
            str: The formatted prompt ready for model input
        """
        import json
        target = json.loads(sample.target)
        num_houses = len(target["rows"])
        columns = target["header"]
        assert columns[0] == "House"
        json_template = {"reasoning": "___", "solution": {}}
        for i in range(num_houses):
            json_template["solution"][f'House {i + 1}'] = {columns[j]: "___" for j in range(1, len(columns))}
        json_str = json.dumps(json_template, indent=4)
        return self.prompt_template.format(question=sample.input, json_template = json_str)


    def extract_answer(self, prediction: str, task_state: TaskState):
        def extract_last_complete_json(s):
            # Stack to keep track of opening and closing braces
            stack = []
            last_json_start = None
            last_json_str = None

            for i, char in enumerate(s):
                if char == '{':
                    stack.append(i)
                    if last_json_start is None:
                        last_json_start = i
                elif char == '}':
                    if stack:
                        start = stack.pop()
                        if not stack:
                            # Complete JSON object found
                            last_json_str = s[last_json_start:i + 1]
                            last_json_start = None

            # Load the last JSON object
            if last_json_str:
                try:
                    return last_json_str.replace("\n", "")
                except json.JSONDecodeError:
                    pass

            return None

        res = extract_last_complete_json(prediction)
        return res

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """
        Calculate evaluation scores by comparing prediction with reference.

        This method computes scores using all configured metrics and creates
        a comprehensive Score object with detailed evaluation results.

        Args:
            original_prediction (str): The original, unfiltered model prediction
            filtered_prediction (str): The filtered and processed prediction
            reference (str): The ground truth reference answer
            task_state (TaskState): The complete task state for context

        Returns:
            Score: Object containing all calculated metric scores and metadata
        """
        # Initialize the score object with prediction details

        from evalscope.benchmarks.zebralogicbench.utils import process_results

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )
        # metadata = task_state.metadata
        try:
            # Process results using the existing ifeval utility
            results = process_results(filtered_prediction, reference)
            score.value.update(results)

            # Set main score name
            score.main_score_name = 'Puzzle Acc'

        except Exception as e:
            logger.error(f'Error calculating zebralogicbench metrics: {e}')
            score.value = {}

        return score


    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        """
        Aggregate metrics across all samples using seqeval.
        """

        Total_Puzzle_Num = len(sample_scores)
        Total_Cell_Num = sum(ss.score.value.get('Cell Num', 0.0) for ss in sample_scores)

        Total_Easy_Puzzle_Num = sum(ss.score.value.get('Easy Puzzle Num', 0.0) for ss in sample_scores)
        Total_Hard_Puzzle_Num = sum(ss.score.value.get('Hard Puzzle Num', 0.0) for ss in sample_scores)

        Total_Small_Puzzle_Num = sum(ss.score.value.get('Small Puzzle Num', 0.0) for ss in sample_scores)
        Total_Medium_Puzzle_Num = sum(ss.score.value.get('Medium Puzzle Num', 0.0) for ss in sample_scores)
        Total_Large_Puzzle_Num = sum(ss.score.value.get('Large Puzzle Num', 0.0) for ss in sample_scores)
        Total_XL_Puzzle_Num = sum(ss.score.value.get('XL Puzzle Num', 0.0) for ss in sample_scores)

        puzzle_acc = sum(ss.score.value.get('Solved Puzzle', 0.0) for ss in sample_scores) / Total_Puzzle_Num
        cell_acc = sum(ss.score.value.get('Solved Cell', 0.0) for ss in sample_scores) / Total_Cell_Num

        easy_puzzle_acc = sum(ss.score.value.get('Solved Easy Puzzle', 0.0) for ss in sample_scores) / Total_Easy_Puzzle_Num
        hard_puzzle_acc = sum(ss.score.value.get('Solved Hard Puzzle', 0.0) for ss in sample_scores) / Total_Hard_Puzzle_Num

        small_puzzle_acc = sum(ss.score.value.get('Solved Small Puzzle', 0.0) for ss in sample_scores) / Total_Small_Puzzle_Num
        medium_puzzle_acc = sum(ss.score.value.get('Solved Medium Puzzle', 0.0) for ss in sample_scores) / Total_Medium_Puzzle_Num
        large_puzzle_acc = sum(ss.score.value.get('Solved Large Puzzle', 0.0) for ss in sample_scores) / Total_Large_Puzzle_Num
        xl_puzzle_acc = sum(ss.score.value.get('Solved XL Puzzle', 0.0) for ss in sample_scores) / Total_XL_Puzzle_Num

        avg_reason_lens = sum(ss.score.value.get('Reason Lens', 0.0) for ss in sample_scores) / Total_Puzzle_Num



        agg_scores = [
            AggScore(metric_name='puzzle_acc', score=puzzle_acc, num=Total_Puzzle_Num, metadata={'type': 'puzzle_acc'}),
            AggScore(metric_name='cell_acc', score=cell_acc, num=Total_Cell_Num, metadata={'type': 'cell_acc'}),
            AggScore(metric_name='easy_puzzle_acc', score=easy_puzzle_acc, num=Total_Easy_Puzzle_Num, metadata={'type': 'easy_puzzle_acc'}),
            AggScore(metric_name='hard_puzzle_acc', score=hard_puzzle_acc, num=Total_Hard_Puzzle_Num, metadata={'type': 'hard_puzzle_acc'}),
            AggScore(metric_name='small_puzzle_acc', score=small_puzzle_acc, num=Total_Small_Puzzle_Num, metadata={'type': 'small_puzzle_acc'}),
            AggScore(metric_name='medium_puzzle_acc', score=medium_puzzle_acc, num=Total_Medium_Puzzle_Num, metadata={'type': 'medium_puzzle_acc'}),
            AggScore(metric_name='large_puzzle_acc', score=large_puzzle_acc, num=Total_Large_Puzzle_Num, metadata={'type': 'large_puzzle_acc'}),
            AggScore(metric_name='xl_puzzle_acc', score=xl_puzzle_acc, num=Total_XL_Puzzle_Num, metadata={'type': 'xl_puzzle_acc'}),
            AggScore(metric_name='avg_reason_lens', score=avg_reason_lens, num=Total_Puzzle_Num, metadata={'type': 'avg_reason_lens'})
        ]

        return agg_scores