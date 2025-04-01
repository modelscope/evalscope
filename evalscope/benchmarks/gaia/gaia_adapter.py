import os
from lark import v_args

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.constants import EvalType, OutputType
from evalscope.metrics import exact_match
from evalscope.utils.utils import ResponseParser


@Benchmark.register(
    name='gaia',
    pretty_name='GAIA',
    dataset_id='AI-ModelScope/GAIA',
    model_adapter=OutputType.GENERATION,
    subset_list=['Level 1', 'Level 2', 'Level 3'],
    metric_list=['AverageAccuracy'],
    few_shot_num=0,
    train_split=None,
    eval_split='validation',
    system_prompt='你是一个高智商和高情商的专家，你被要求回答一个选择题，并选出一个正确的选项，解释原因，最终输出格式为：`答案是(选项)`。',  # noqa: E501
)
class GAIAAdapter(DataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, **kwargs) -> dict:
        if not os.path.exists(self.dataset_id):
            # Download the dataset snapshot first
            from modelscope import dataset_snapshot_download

            self.dataset_id = dataset_snapshot_download(self.dataset_id)
        # Note: need trust_remote_code=True to load the python script
        dataset_dict = super().load(subset_list=['2023_all'], trust_remote_code=True, **kwargs)
        return self.reformat_subset(dataset_dict, subset_key='Level', format='Level {}')

    def gen_prompt(self, input_d: dict, few_shot_list: list, **kwargs) -> dict:
        """
        Generate the prompt for the model input.
        """
        question = input_d['Question']

        return self.gen_prompt_data(question)

    def get_gold_answer(self, input_d: dict) -> str:
        # Extract the gold answer from the input dict.
        return input_d['Final answer']
