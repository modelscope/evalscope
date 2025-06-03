# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

from evalscope.benchmarks.utils import PromptData, load_file_with_extension, preprocess_decorator
from evalscope.constants import DEFAULT_DATASET_CACHE_DIR, AnswerKeys, EvalType, HubType
from evalscope.metrics import LLMJudge, metric_registry
from evalscope.report import Report, ReportGenerator
from evalscope.utils.logger import get_logger

logger = get_logger()


class DataAdapter(ABC):
    """
    Data Adapter for the benchmark. You need to implement the following methods:
        - gen_prompt
        - get_gold_answer
        - parse_pred_result
        - match
    """

    def __init__(self,
                 name: str,
                 dataset_id: str,
                 model_adapter: str,
                 subset_list: list,
                 metric_list: List[str],
                 llm_as_a_judge: bool = False,
                 output_types: Optional[List[str]] = None,
                 few_shot_num: Optional[int] = 0,
                 train_split: Optional[str] = None,
                 eval_split: Optional[str] = None,
                 prompt_template: Optional[str] = None,
                 system_prompt: Optional[str] = None,
                 query_template: Optional[str] = None,
                 pretty_name: Optional[str] = None,
                 description: Optional[str] = None,
                 **kwargs):
        """
        Args:
            name: str, the name of the benchmark.
            dataset_id: str, the dataset id on ModelScope or local path for the benchmark.
            model_adapter: str, the model adapter to use for the benchmark.
            subset_list: list of subset names for the dataset.
            metric_list: list, the metric list to evaluate the model on specific benchmark.
            llm_as_a_judge: bool, whether to use LLM as a judge to evaluate the predicted answer against the gold answer.
            output_types: list, the output types of the model adapter. Default: [model_adapter]
            few_shot_num: int, number of few-shot examples. Default: 0
            train_split: str, usually for few-shot examples. e.g. 'train'
            eval_split: str, the target eval split name. e.g. 'test'
            prompt_template: str, the prompt template for the benchmark,
                e.g. for ARC, it is `The following are multiple choice questions, please output correct answer in
                    the form of A or B or C or D, do not output explanation:`
            system_prompt: str, the system prompt for the benchmark, e.g. 'You are a helpful assistant.'
            query_template: str, the query template for the benchmark, e.g. 'Please answer the following question: {}'
            pretty_name: str, the pretty name of the benchmark, e.g. 'ARC Challenge Set'.
            description: str, the description of the benchmark,
                e.g. 'ARC Challenge Set is a benchmark for evaluating reasoning abilities of models on science questions.'
        """  # noqa: E501
        self.name = name
        self.dataset_id = dataset_id
        self.model_adapter = model_adapter
        self.subset_list = subset_list
        self.metric_list = metric_list
        self.llm_as_a_judge = llm_as_a_judge
        self.output_types = output_types or [model_adapter]
        self.few_shot_num = few_shot_num
        self.train_split = train_split
        self.eval_split = eval_split
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt
        self.query_template = query_template
        self.pretty_name = pretty_name
        self.description = description
        self.config_kwargs = kwargs
        self.category_map = kwargs.get('category_map', {})
        self.choices = kwargs.get('choices', None)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # find and decorate parse_pred_result method
        if hasattr(cls, 'parse_pred_result'):
            original_method = cls.parse_pred_result
            cls.parse_pred_result = preprocess_decorator(original_method)

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
            logger.info(f'Loading dataset from local disk: {dataset_name_or_path}')
            trust_remote_code = kwargs.pop('trust_remote_code', False)
            data_dict = self.load_from_disk(
                dataset_name_or_path, subset_list, work_dir, trust_remote_code=trust_remote_code, **kwargs)
        else:
            logger.info(f'Loading dataset from hub: {dataset_name_or_path}')
            trust_remote_code = kwargs.pop('trust_remote_code', True)
            data_dict = self.load_from_hub(
                dataset_name_or_path, subset_list, work_dir, trust_remote_code=trust_remote_code, **kwargs)
        if len(data_dict) == 0:
            raise ValueError(f'Dataset is empty: {dataset_name_or_path}')
        return data_dict

    def load_from_hub(self, dataset_name_or_path: str, subset_list: list, work_dir: str, **kwargs) -> dict:
        from modelscope.msdatasets import MsDataset

        datasets_hub: str = kwargs.pop('datasets_hub', HubType.MODELSCOPE)
        split_as_subset: bool = kwargs.pop('split_as_subset', False)
        # Load dataset from remote
        logger.info(f'Loading dataset: dataset_name: {dataset_name_or_path} > subsets: {subset_list}')

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

    def load_with_snapshot(self,
                           file_structure: Dict[str, List[str]],
                           dataset_name_or_path: str = None,
                           subset_list: list = None,
                           work_dir: Optional[str] = DEFAULT_DATASET_CACHE_DIR,
                           **kwargs) -> dict:
        """
        For datasets that cannot be correctly loaded using MsDataset, utilize snapshot downloading to load the data.
        This feature supports both remote and local datasets.

        Args:
            file_structure: dict, the file structure of the dataset, e.g. {'subset_name': ['file1.jsonl', 'file2.jsonl']}.
            dataset_name_or_path: str, the dataset id on ModelScope or local path for the benchmark.
            subset_list: list of subset names for the dataset.
            work_dir: str, the working directory to store the dataset.
        Returns: {'subset_name': {'eval': eval_dataset}}
        """  # noqa: E501
        dataset_name_or_path = os.path.expanduser(dataset_name_or_path or self.dataset_id)
        subset_list = subset_list or self.subset_list

        # Try to load dataset from local disk
        if os.path.exists(dataset_name_or_path):
            logger.info(f'Loading dataset from {dataset_name_or_path}')
            dataset_path = dataset_name_or_path
        else:
            from modelscope import dataset_snapshot_download

            # Load dataset from remote
            logger.info(f'Loading dataset from modelscope: > dataset_name: {dataset_name_or_path}')
            # flatten file structure
            file_names = [file for sub_files in file_structure.values() for file in sub_files]
            # download dataset snapshot
            dataset_path = dataset_snapshot_download(
                dataset_name_or_path, cache_dir=work_dir, allow_file_pattern=file_names)
        # read and process files
        data_dict = defaultdict(dict)
        for sub_name in subset_list:
            file_paths = [os.path.join(dataset_path, file_name) for file_name in file_structure[sub_name]]
            # not train split, only eval split
            data_dict[sub_name][self.eval_split] = load_file_with_extension(file_paths)

        return data_dict

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

        logger.info(f'Use settings: '
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

    def compute_dict_metric(self, review_res_list: Union[List[dict], List[List[dict]]],
                            **kwargs) -> Dict[str, List[float]]:
        """
        compute weighted mean of score of all samples

        Args:
            review_res_list: [score1, score2, ...]

        Returns:
            avg_res: Dict[str, List[float]]

        """
        if len(review_res_list) > 0 and isinstance(review_res_list[0], list):
            review_res_list = [item for sublist in review_res_list for item in sublist]

        items = defaultdict(list)
        for scores in review_res_list:
            if isinstance(scores, dict):
                for k, v in scores.items():
                    items[k].append(v)
            else:
                items['AverageAccuracy'].append(scores)
        return items

    def gen_report(self, subset_score_map: dict, model_name: str, **kwargs) -> Report:
        """
        Generate report for the evaluation results for all subsets.

        Args:
            subset_score_map: The subset-score map.
                e.g. {subset_name: [{'metric_name': 'AverageAccuracy', 'score': 0.3389, 'num': 100}]}

            model_name: The evaluation model name.

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
        return ReportGenerator.gen_report(subset_score_map, model_name, data_adapter=self, **kwargs)

    def post_process_report(self, report: Report, **kwargs):
        """
        Post-process the report after generation. Draw a chart, save to file, etc.
        This method can be overridden to customize the report format or content.

        Args:
            report (Report): The generated report.
        """
        pass

    def gen_prompt_data(self,
                        prompt: str,
                        system_prompt: Optional[str] = None,
                        choices: Optional[List[str]] = None,
                        index: Optional[Union[int, str]] = None,
                        id: Optional[Union[int, str]] = None,
                        messages: Optional[List[dict]] = None,
                        **kwargs) -> dict:
        """
        Generates a dictionary representation of prompt data for evaluation or inference.

        Args:
            prompt (str): The main prompt or input text. Can also be a list of prompts.
            system_prompt (Optional[str], optional): An optional system-level prompt to provide context or instructions. Defaults to None.
            choices (Optional[List[str]], optional): A list of possible choices for multi-choice tasks.
                If not provided, uses self.choices. Defaults to None.
            index (Optional[Union[int, str]], optional): An optional index or identifier for the prompt.
                Defaults to 0 if not provided. Defaults to None.
            id (Optional[Union[int, str]], optional): An optional unique identifier for the prompt data. Defaults to None.
            messages (Optional[List[dict]], optional): An optional list of message dictionaries, typically for chat-based prompts. Defaults to None.
                If messages is provided, it will be used as the prompt data instead of the prompt string.

        Returns:
            dict: A dictionary representation of the prompt data, suitable for further processing or model input.
        """  # noqa: E501
        data = [prompt] if not isinstance(prompt, list) else prompt
        prompt_data = PromptData(
            data=data,
            multi_choices=choices or self.choices,
            system_prompt=system_prompt or self.system_prompt,
            index=index or 0,
            id=id,
            messages=messages)
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

    def llm_match(self, gold: Any, pred: Any, judge: Optional[LLMJudge] = None, **kwargs) -> float:
        """
        Use LLM as a judge to evaluate the predicted answer against the gold answer.

        Args:
            gold (Any): The golden answer.
            pred (Any): The predicted answer.

        Returns:
            The match result as a float score between 0 and 1.
        """
        # Default judge handling
        if judge is None:
            logger.warning('No judge LLM provided, please specify a judge LLM in the config.')
            return 0

        # Extract question from raw_input if available
        raw_input = kwargs.get('raw_input', {})
        question_keys = ['question', 'Question', 'prompt', 'Prompt', 'query', 'Query', 'problem', 'Problem']
        # Find the first non-empty question key in raw_input
        question = next((raw_input.get(key) for key in question_keys if raw_input.get(key)), None)

        # Request judge and obtain score
        prompt = judge.build_prompt(pred, gold, question)
        score = judge(prompt)
        return judge.get_score(score)
