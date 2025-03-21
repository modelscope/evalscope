# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os.path
from collections import defaultdict
from typing import List, Optional
from evalscope.constants import DEFAULT_DATASET_CACHE_DIR

from evalscope.benchmarks import Benchmark, DataAdapter
from evalscope.metrics import (
    bleu_ngram_one_sample,
    compute_rouge_score_one_sample_zh,
    mean,
)
from evalscope.utils.io_utils import jsonl_to_list
from evalscope.utils.logger import get_logger

logger = get_logger()


@Benchmark.register(
    name="general_qa",
    dataset_id="general_qa",
    subset_list=["default"],
    metric_list=["AverageBLEU", "AverageRouge"],
    few_shot_num=0,
    train_split=None,
    eval_split="test",
    prompt_template="请回答问题\n{query}",
)
class GeneralQAAdapter(DataAdapter):
    # TODO: set few_shot_num

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def load(
        self,
        dataset_name_or_path: str = None,
        subset_list: list = None,
        work_dir: Optional[str] = DEFAULT_DATASET_CACHE_DIR,
        **kwargs,
    ) -> dict:
        """
        Load local Q&A dataset from local disk. With follow directory structure:
        |- custom_eval
            |- text
                |- example1.jsonl
                |- example2.jsonl
        and the dataset_args should be
        dataset_args = {
            'general_qa': {
                "local_path": "custom_eval/qa",  # Custom dataset path
                "subset_list": [
                    "example1",       # Evaluation dataset name, corresponding to * in the above *.jsonl
                    "example2",
                ]
            }
        """

        dataset_name_or_path = os.path.expanduser(
            dataset_name_or_path or self.dataset_id
        )
        subset_list = subset_list or self.subset_list
        task_cfg = kwargs.get("task_cfg")
        data_dict = {}
        for subset_name in subset_list:
            data_dict[subset_name] = {}
            file_path = os.path.join(
                task_cfg.get("dataset_args").get("general_qa").get("dataset_id"),
                f"{subset_name}.jsonl",
            )
            print(file_path)
            if os.path.exists(file_path):
                data_dict[subset_name][self.eval_split] = jsonl_to_list(file_path)
        print(json.dumps(data_dict, ensure_ascii=False))

        return data_dict

    def gen_prompt(
        self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs
    ) -> dict:
        """
        Args:
            input_d:
                format1: {'history': [['q1', 'a1'], ['q2', 'a2']], 'question': '', 'answer': ''}
                format2: {'history': [['q1', 'a1'], ['q2', 'a2']], 'query': '', 'response': ''}

        Returns:
            {'data': [prompt]}

        """
        # prompt = f"'<|im_start|>user\n{input_d['input']}<|im_end|>\n<|im_start|>assistant\n'"
        history = input_d.get(
            "history", []
        )  # history: [['q1', 'a1'], ['q2', 'a2'], ...]
        if len(history) > 0:
            logger.warning(
                "The history is not included in the prompt for GeneralQA. \
                           To be supported in the future."
            )

        query = input_d.get("question", "") or input_d.get("query", "")
        prompt = self.prompt_template.format(query=query)
        return self.gen_prompt_data(prompt)

    def get_gold_answer(self, input_d: dict) -> str:
        """
        Args:
            input_d: {'history': [], 'question': '', 'answer': ''}

        Returns:
            gold_answer: str

        """
        return input_d.get("answer", "") or input_d.get("response", "")

    def parse_pred_result(
        self, result: str, raw_input_d: dict = None, eval_type: str = "checkpoint"
    ) -> str:
        """
        Args:
            result: str

        Returns:
            pred_result: str

        """
        return result

    def match(self, gold: str, pred: str) -> dict:
        """
        Args:
            gold: str
            pred: str

        Returns:
            bleu_score: dict

        """
        res = dict()
        if "AverageRouge" in self.metric_list:
            rouge_dict = compute_rouge_score_one_sample_zh([pred], [gold])
            res.update(rouge_dict)
        if "AverageBLEU" in self.metric_list:
            bleu_dict = bleu_ngram_one_sample(pred, gold)
            res.update(bleu_dict)
        return res

    def compute_metric(self, review_res_list: List[dict], **kwargs) -> List[dict]:
        """
        compute weighted mean of the bleu score of all samples

        Args:
            review_res_list: [score1, score2, ...]

        Returns:
            avg_res: List[dict]

        """
        items = defaultdict(list)
        for scores in review_res_list:
            for k, v in scores.items():
                items[k].append(v)
        # items = [(score, 1.0) for score in review_res_list]
        return [
            {"metric_name": k, "score": mean(v), "num": len(v)}
            for k, v in items.items()
        ]
