from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.metrics import extract_answer, math_equal, strip_answer_string
from evalscope.utils.logger import get_logger

# flake8: noqa

logger = get_logger()


@Benchmark.register(
    name='aime25',
    pretty_name='AIME-2025',
    tags=['Mathematics'],
    description=
    'The AIME 2025 benchmark is based on problems from the American Invitational Mathematics Examination, a prestigious high school mathematics competition. This benchmark tests a model’s ability to solve challenging mathematics problems by generating step-by-step solutions and providing the correct final answer.',
    dataset_id='opencompass/AIME2025',
    subset_list=['AIME2025-I', 'AIME2025-II'],
    metric_list=['AveragePass@1'],
    few_shot_num=0,
    train_split=None,
    eval_split='test',  # Only train set is available
    prompt_template='{query}\nPlease reason step by step, and put your final answer within \\boxed{{}}.',
)
class AIME25Adapter(DataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def gen_prompt(self, input_d: dict, few_shot_list: list, **kwargs) -> dict:
        """
        Generate the prompt for the model input.
        """
        problem = input_d['question']
        full_prompt = self.prompt_template.format(query=problem)

        return self.gen_prompt_data(full_prompt)

    def get_gold_answer(self, input_d: dict) -> str:
        # Extract the gold answer from the input dict.
        return strip_answer_string(input_d['answer'])

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = 'checkpoint') -> str:
        """
        Parse the model output to get the answer. Could be the best choice index.
        """
        # Note: Use same extraction method for both of checkpoint/service/custom
        result = strip_answer_string(extract_answer(result))
        return result

    def match(self, gold: str, pred: str) -> float:
        res = math_equal(pred, gold)
        return 1.0 if res else 0.0
