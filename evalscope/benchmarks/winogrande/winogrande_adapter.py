from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.constants import EvalType, OutputType
from evalscope.metrics import exact_match
from evalscope.utils.utils import ResponseParser


@Benchmark.register(
    name='winogrande',
    pretty_name='Winogrande',
    dataset_id='AI-ModelScope/winogrande_val',
    model_adapter=OutputType.GENERATION,
    output_types=[OutputType.MULTIPLE_CHOICE, OutputType.GENERATION],
    metric_list=['AverageAccuracy'],
    few_shot_num=0,
    train_split=None,
    eval_split='validation',
    prompt_template='Question: {query}\nA. {option1}\nB. {option2}\nAnswer:',  # noqa: E501
)
class WinograndeAdapter(DataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.choices = ['A', 'B']

    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> dict:
        """
        Generate model prompt from input data.
        """
        prompt = self.prompt_template.format(
            query=input_d['sentence'],
            option1=input_d['option1'],
            option2=input_d['option2'],
        )
        return self.gen_prompt_data(prompt)

    def get_gold_answer(self, input_d: dict) -> str:
        """
        Parse the raw input labels (gold).
        """
        answer_index = int(input_d['answer']) - 1
        return self.choices[answer_index]

    def parse_pred_result(self, result: str, raw_input_d: dict = None, eval_type: str = EvalType.CHECKPOINT) -> str:
        """
        Parse the predicted result and extract proper answer.
        """
        if self.model_adapter == OutputType.MULTIPLE_CHOICE:
            return result
        else:
            return ResponseParser.parse_first_option_with_choices(result, self.choices)

    def match(self, gold: str, pred: str) -> float:
        """
        Match the gold answer and the predicted answer.
        """
        return exact_match(gold=gold, pred=pred)
