import json
import os

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.metrics.math_parser import (
    extract_answer,
    math_equal,
    strip_answer_string,
)
from evalscope.utils.logger import get_logger

# flake8: noqa

logger = get_logger()


@Benchmark.register(
    name="aime24",
    pretty_name="AIME-2024",
    dataset_id="HuggingFaceH4/aime_2024",
    subset_list=["default"],
    metric_list=["AveragePass@1"],
    few_shot_num=0,
    train_split=None,
    eval_split="train",  # Only train set is available
    prompt_template="{query}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
)
class AIME24Adapter(DataAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_from_disk(
        self, dataset_name_or_path, subset_list, work_dir, **kwargs
    ) -> dict:
        data_dict = {}
        subset_list = ["test"]
        for subset_name in subset_list:
            if os.path.exists(dataset_name_or_path):
                file_path = os.path.join(dataset_name_or_path, f"{subset_name}.jsonl")
            else:
                file_path = os.path.join(
                    work_dir, dataset_name_or_path, f"{subset_name}.jsonl"
                )
            if os.path.exists(file_path):
                with open(file_path, encoding="utf-8") as f:
                    rows = []
                    for line in f.readlines():
                        item = json.loads(line)
                        item["question"] = item["problem"]
                        rows.append(item)

                    data_dict[subset_name] = {self.eval_split: rows}

        return data_dict

    def gen_prompt(self, input_d: dict, few_shot_list: list, **kwargs) -> dict:
        """
        Generate the prompt for the model input.
        """
        problem = input_d["problem"]
        full_prompt = self.prompt_template.format(query=problem)

        return self.gen_prompt_data(full_prompt)

    def get_gold_answer(self, input_d: dict) -> str:
        # Extract the gold answer from the input dict.
        return strip_answer_string(input_d["answer"])

    def parse_pred_result(
        self, result: str, raw_input_d: dict = None, eval_type: str = "checkpoint"
    ) -> str:
        """
        Parse the model output to get the answer. Could be the best choice index.
        """
        # Note: Use same extraction method for both of checkpoint/service/custom
        result = strip_answer_string(extract_answer(result))
        return result

    def match(self, gold: str, pred: str) -> float:
        return math_equal(pred, gold)
