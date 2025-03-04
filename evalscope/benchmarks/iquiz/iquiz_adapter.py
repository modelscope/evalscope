from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.constants import EvalType, OutputType
from evalscope.metrics import exact_match
from evalscope.utils.utils import ResponseParser


@Benchmark.register(
    name='iquiz',
    pretty_name='IQuiz',
    dataset_id='AI-ModelScope/IQuiz',
    model_adapter=OutputType.GENERATION,
    output_types=[OutputType.MULTIPLE_CHOICE, OutputType.GENERATION],
    subset_list=['IQ', 'EQ'],
    metric_list=['AverageAccuracy'],
    few_shot_num=0,
    train_split=None,
    eval_split='test',
    system_prompt='你是一个高智商和高情商的专家，你被要求回答一个选择题，并选出一个正确的选项，解释原因，最终输出格式为：`答案是(选项)`。',  # noqa: E501
)
class IQuizAdapter(DataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.choices = ['A', 'B', 'C', 'D', 'E']

    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> dict:
        """
        Generate model prompt from input data.
        example:
        {
            "question":"天气预报说本周星期三会下雨，昨天果然下雨了，今天星期几？",
            "choices":["星期一","星期二","星期三","星期四"],
            "answer":"D",
            "level":1
        }
        """
        prompt = f"问题: {input_d['question']}\n"
        prompt += self.__form_options(input_d['choices'])
        return self.gen_prompt_data(prompt)

    def __form_options(self, options: list):
        option_str = '选项:\n'
        for opt, choice in zip(options, self.choices):
            option_str += f'({choice}): {opt}' + '\n'
        return option_str

    def get_gold_answer(self, input_d: dict) -> str:
        """
        Parse the raw input labels (gold).
        """
        return input_d['answer']

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
