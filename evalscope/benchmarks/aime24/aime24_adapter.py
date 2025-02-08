from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.metrics import Pass1
from evalscope.metrics.math_accuracy import is_equiv, last_boxed_only_string, remove_boxed
from evalscope.models import ChatGenerationModelAdapter
from evalscope.utils.logger import get_logger

# flake8: noqa

logger = get_logger()


@Benchmark.register(
    name='aime24',
    dataset_id='AI-ModelScope/AIME_2024',
    model_adapter=ChatGenerationModelAdapter,
    subset_list=['default'],
    metric_list=[Pass1],
    few_shot_num=0,
    train_split=None,
    eval_split='train',  # Only train set is available
    prompt_template='',
)
class AIME24Adapter(DataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def gen_prompt(self, input_d: dict, few_shot_list: list, **kwargs) -> dict:
        """
        Generate the prompt for the model input.

        Args:
            input_d: raw input dict.
                {"problem": "How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?", "level": "Level 3", "type": "Algebra", "solution": "The denominator of the rational function factors into $x^2+x-6=(x-2)(x+3)$. Since the numerator is always nonzero, there is a vertical asymptote whenever the denominator is $0$, which occurs for $x = 2$ and $x = -3$.  Therefore, the graph has $\\boxed{2}$ vertical asymptotes."}

            few_shot_list:  few shot list. Each item is a raw input dict.
            **kwargs:

        Returns:
            {'data': [prompt]}
        """
        prompt = input_d['Problem']
        full_prompt = f'Problem: {prompt}\nMark your solution with \\boxed\nAnswer:'

        return {'data': [full_prompt], 'system_prompt': self.prompt_template}

    def get_gold_answer(self, input_d: dict) -> str:
        # Extract the gold answer from the input dict.
        return input_d['Answer']

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = 'checkpoint') -> str:
        """
        Parse the model output to get the answer. Could be the best choice index.

        Args:
            result: Predicted answer from the model. Usually a string for chat.
            raw_input_d (dict): The raw input. Depending on the dataset.
            eval_type: 'checkpoint' or 'service' or `custom`

        Returns:
            The parsed answer. Depending on the dataset. Usually a string for chat.
        """
        # Note: Use same extraction method for both of checkpoint/service/custom
        try:
            result = remove_boxed(last_boxed_only_string(result))
        except Exception:
            return None
        return result

    def match(self, gold: str, pred: str) -> float:
        res = 0
        if is_equiv(pred, gold):
            res = 1

        return res
