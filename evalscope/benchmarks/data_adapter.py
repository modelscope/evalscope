# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, List, Optional, Union

from evalscope.benchmarks.utils import PromptData
from evalscope.constants import DEFAULT_DATASET_CACHE_DIR, AnswerKeys, EvalType, HubType
from evalscope.metrics.named_metrics import metric_registry
from evalscope.report import Report, ReportGenerator
from evalscope.utils.logger import get_logger

logger = get_logger()


class DataAdapter(ABC):

    def __init__(self,
                 name: str,
                 dataset_id: str,
                 model_adapter: str,
                 subset_list: list,
                 metric_list: List[str],
                 few_shot_num: Optional[int] = 0,
                 train_split: Optional[str] = None,
                 eval_split: Optional[str] = None,
                 prompt_template: Optional[str] = None,
                 system_prompt: Optional[str] = None,
                 query_template: Optional[str] = None,
                 pretty_name: Optional[str] = None,
                 **kwargs):
        """
        Data Adapter for the benchmark. You need to implement the following methods:
            - gen_prompt
            - get_gold_answer
            - parse_pred_result
            - match
        Args:
            name: str, the name of the benchmark.
            dataset_id: str, the dataset id on ModelScope or local path for the benchmark.
            subset_list: list of subset names for the dataset.
            metric_list: list, the metric list to evaluate the model on specific benchmark.
            few_shot_num: int, number of few-shot examples. Default: 0
            train_split: str, usually for few-shot examples. e.g. 'train'
            eval_split: str, the target eval split name. e.g. 'test'
            prompt_template: str, the prompt template for the benchmark,
                e.g. for ARC, it is `The following are multiple choice questions, please output correct answer in
                    the form of A or B or C or D, do not output explanation:`
        """
        self.name = name
        self.dataset_id = dataset_id
        self.model_adapter = model_adapter
        self.subset_list = subset_list
        self.metric_list = metric_list
        self.few_shot_num = few_shot_num
        self.train_split = train_split
        self.eval_split = eval_split
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt
        self.query_template = query_template
        self.pretty_name = pretty_name
        self.config_kwargs = kwargs
        self.category_map = kwargs.get('category_map', {})
        self.choices = kwargs.get('choices', None)

    def load(self,
             dataset_name_or_path: str = None,
             subset_list: list = None,
             work_dir: Optional[str] = DEFAULT_DATASET_CACHE_DIR,
             **kwargs) -> dict:
        """
        Load the dataset. Remote and local datasets are supported.
        You can rewrite this method to support your own local dataset, just follow the format of the output.

        Returns: {'subset_name': {'train': train_dataset, 'test': test_dataset}}
            train_dataset, test_dataset: Iterable dataset, object each item of which is a dict.

        """
        dataset_name_or_path = os.path.expanduser(dataset_name_or_path or self.dataset_id)
        subset_list = subset_list or self.subset_list

        # Try to load dataset from local disk
        if os.path.exists(dataset_name_or_path):
            data_dict = self.load_from_disk(dataset_name_or_path, subset_list, work_dir, **kwargs)
        else:
            data_dict = self.load_from_hub(dataset_name_or_path, subset_list, work_dir, **kwargs)
        if len(data_dict) == 0 or len(next(iter(data_dict.values()))) == 0:
            raise ValueError(f'Local dataset is empty: {dataset_name_or_path}')
        return data_dict

    def load_from_hub(self, dataset_name_or_path: str, subset_list: list, work_dir: str, **kwargs) -> dict:
        from modelscope.msdatasets import MsDataset

        datasets_hub: str = kwargs.pop('datasets_hub', HubType.MODELSCOPE)
        split_as_subset: bool = kwargs.pop('split_as_subset', False)
        # Load dataset from remote
        logger.info(f'Loading dataset : > dataset_name: {dataset_name_or_path} > subsets: {subset_list}')

        data_dict = {}
        split_list = [split for split in [self.train_split, self.eval_split] if split is not None]
        if len(split_list) == 0:
            logger.error(f'Got empty split list: {split_list}')

        if split_as_subset:
            for sub_name in subset_list:
                data_dict[sub_name] = {}
                # e.g. train: few-shot, test: target dataset to evaluate
                for split in split_list:
                    dataset = MsDataset.load(
                        dataset_name=dataset_name_or_path,
                        split=sub_name,  # load subset from split
                        cache_dir=work_dir,
                        hub=datasets_hub,
                        **kwargs)
                    data_dict[sub_name].update({split: dataset})
        else:
            for sub_name in subset_list:
                data_dict[sub_name] = {}
                # e.g. train: few-shot, test: target dataset to evaluate
                for split in split_list:
                    dataset = MsDataset.load(
                        dataset_name=dataset_name_or_path,
                        subset_name=sub_name,
                        split=split,
                        cache_dir=work_dir,
                        hub=datasets_hub,
                        **kwargs)
                    data_dict[sub_name].update({split: dataset})

        return data_dict

    def load_from_disk(self, dataset_name_or_path, subset_list, work_dir, **kwargs) -> dict:
        """
        Load the dataset from local disk.
        If you want to support local dataset, please rewrite this method in xxx_data_adapter.
        Use modelscope.msdatasets.MsDataset.load to load the dataset from local by default.
        """
        return self.load_from_hub(dataset_name_or_path, subset_list, work_dir, **kwargs)

    def reformat_subset(self, data_dict: dict, subset_key: str, format: str = '{}') -> dict:
        """
        Reformat the dataset subset with subset_key and format.
        """
        res_dict: dict = defaultdict(lambda: defaultdict(list), {key: defaultdict(list) for key in self.subset_list})

        for sub_name, sub_data_dict in data_dict.items():
            for split in [self.train_split, self.eval_split]:
                if split is None:
                    continue
                for sample_d in sub_data_dict[split]:
                    new_subset_name = format.format(sample_d[subset_key])
                    if new_subset_name not in self.subset_list:
                        continue
                    res_dict[new_subset_name][split].append(sample_d)
        return res_dict

    def gen_prompts(self, data_dict: dict) -> dict:
        """
        Generate dataset prompts from raw input, unify the prompt format for different datasets.

        Args:
            data_dict:  Refer to the output of load method: evalscope.benchmarks.benchmark.Benchmark.load

        Returns:
            {'subset_name': [prompt_d_1, prompt_d_2, ...]}
            prompt_d_i (dict): refer to the output of gen_prompt method.

        e.g. train -- few-shot data, test -- target dataset to evaluate.
        """
        res_dict: dict = {}

        if self.few_shot_num and self.few_shot_num < 0:
            raise ValueError(f'Invalid shot_num: {self.few_shot_num} for few-shot evaluation.')

        logger.info(f'Use default settings: '
                    f'> few_shot_num: {self.few_shot_num}, '
                    f'> few_shot_split: {self.train_split}, '
                    f'> target_eval_split: {self.eval_split}')

        for sub_name, sub_data_dict in data_dict.items():
            few_shot_data = []
            if self.train_split and self.few_shot_num and self.few_shot_num > 0:
                few_shot_random: bool = self.config_kwargs.get('few_shot_random', True)
                few_shot_data = self.get_fewshot_examples([item for item in sub_data_dict[self.train_split]],
                                                          self.few_shot_num,
                                                          few_shot_random=few_shot_random)

            res_dict[sub_name] = []
            for sample_d in sub_data_dict[self.eval_split]:
                prompt_d = self.gen_prompt(input_d=sample_d, subset_name=sub_name, few_shot_list=few_shot_data)
                prompt_d[AnswerKeys.RAW_INPUT] = sample_d
                res_dict[sub_name].append(prompt_d)

        return res_dict

    def get_fewshot_examples(self, data_list: list, k: int, few_shot_random: bool = True):

        if k > len(data_list):
            k = len(data_list)
        if few_shot_random:
            return random.sample(data_list, k)
        else:
            return data_list[:k]

    def compute_metric(self, review_res_list: Union[dict, list], **kwargs) -> List[dict]:
        """
        Compute evaluation result by specific metrics.

        Args:
            review_res_list: list, the review result list, each item of which is match result for gold and pred.

        Returns:
            Metric results. e.g. [{'metric_name': 'AverageAccuracy', 'score': 0.3389, 'num': 100}]
        """
        if len(self.metric_list) == 0:
            raise ValueError('No metric list found for the benchmark.')

        res_list = []
        for metric_str in self.metric_list:
            metric = metric_registry.get(metric_str)
            metric_name = metric.name
            metric_func = metric.object
            if isinstance(review_res_list, dict):
                review_res = review_res_list.get(metric_name, [])
            else:
                review_res = review_res_list
            res_list.append({'metric_name': metric_name, 'score': metric_func(review_res), 'num': len(review_res)})
        return res_list

    def gen_report(self, subset_score_map: dict, report_name: str = None, **kwargs) -> Report:
        """
        Generate report for the evaluation results for all subsets.

        Args:
            subset_score_map: The subset-score map.
                e.g. {subset_name: [{'metric_name': 'AverageAccuracy', 'score': 0.3389, 'num': 100}]}

            report_name: str, the user-defined report name. Default: None

        Returns: The evaluation report.

        Here is a format example for gsm8k:
        {
            "name": "qwen2.5_gsm8k",
            "metrics": [
                {
                    "name": "AverageAccuracy",
                    "categories": [
                        {
                            "name": "default",
                            "subsets": [
                                {
                                    "name": "main",
                                    "score": 0.0,
                                    "num": 2
                                }
                            ],
                            "num": 2,
                            "score": 0.0,
                            "macro_score": 0.0
                        }
                    ],
                    "num": 2,
                    "score": 0.0,
                    "macro_score": 0.0
                }
            ],
            "dataset_name": "gsm8k",
            "model_name": "qwen2.5"
        }
        """  # noqa: E501
        kwargs['category_map'] = self.category_map
        kwargs['metric_list'] = self.metric_list
        return ReportGenerator.gen_report(subset_score_map, report_name, **kwargs)

    def gen_prompt_data(self, prompt: str, **kwargs) -> dict:
        if not isinstance(prompt, list):
            prompt = [prompt]
        prompt_data = PromptData(data=prompt, multi_choices=self.choices, system_prompt=self.system_prompt)
        return prompt_data.to_dict()

    def gen_prompt(self, input_d: dict, subset_name: str, few_shot_list: list, **kwargs) -> Any:
        """
        Generate model prompt from raw input, unify the prompt format for different datasets.
        The input format is compatible with OpenAI Chat Completions APIs.

        Args:
            input_d (Any): The raw input. Depending on the dataset.
            subset_name (str): The subset name.
            few_shot_list (list): The few-shot examples.

        Returns:
            For class ChatGenerationModelAdapter, the output format is:
                {'data': [full_prompt], 'system_prompt': (str, optional)},  -- full_prompt: str, the constructed prompt for each sample from dataset.
            For class MultiChoiceModelAdapter, the output format is:
                {'data': [full_prompt], 'multi_choices': self.choices}  -- full_prompt: str, the constructed prompt for each sample from dataset.
            For class ContinuationEvalModelAdapter, the output format is:
                {'data': ctx_continuation_pair_list, 'multi_choices': self.choices} -- ctx_continuation_pair_list: list, the context-continuation pair list.
        """  # noqa: E501
        raise NotImplementedError

    @abstractmethod
    def get_gold_answer(self, input_d: Any) -> Any:
        """
        Parse the raw input labels (gold).

        Args:
            input_d: input raw data. Depending on the dataset.

        Returns:
            The parsed input. e.g. gold answer ... Depending on the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def parse_pred_result(self, result: Any, raw_input_d: dict = None, eval_type: str = EvalType.CHECKPOINT) -> Any:
        """
        Parse the predicted result and extract proper answer.

        Args:
            result: Predicted answer from the model. Usually a string for chat.
            raw_input_d: The raw input. Depending on the dataset.
            eval_type: 'checkpoint' or 'service' or `custom`, default: 'checkpoint'

        Returns:
            The parsed answer. Depending on the dataset. Usually a string for chat.
        """
        raise NotImplementedError

    @abstractmethod
    def match(self, gold: Any, pred: Any) -> Any:
        """
        Match the gold answer and the predicted answer.

        Args:
            gold (Any): The golden answer. Usually a string for chat/multiple-choice-questions.
                        e.g. 'A', extracted from get_gold_answer method.
            pred (Any): The predicted answer. Usually a string for chat/multiple-choice-questions.
                        e.g. 'B', extracted from parse_pred_result method.

        Returns:
            The match result. Usually a score (float) for chat/multiple-choice-questions.
        """
        raise NotImplementedError
