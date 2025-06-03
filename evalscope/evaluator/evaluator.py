# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import os
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Union

from evalscope.benchmarks import DataAdapter
from evalscope.config import TaskConfig
from evalscope.constants import AnswerKeys, DumpMode, EvalStage, EvalType, JudgeStrategy, ReviewKeys
from evalscope.models import BaseModelAdapter
from evalscope.report import Report, gen_report_table
from evalscope.utils import dict_torch_dtype_to_str, gen_hash
from evalscope.utils.io_utils import OutputsStructure, dump_jsonl_data, jsonl_to_list
from evalscope.utils.logger import get_logger

logger = get_logger()


class Evaluator(object):
    """
    The evaluator for model on datasets.

    Args:
        dataset_name_or_path: str, the dataset name or path.
                if the dataset is a local path, e.g. /path/to/your_dataset_name,
                then the task name will be the basename of the path, which is `your_dataset_name`.
        data_adapter: DataAdapter, the data adapter for the dataset.
        model_adapter: BaseModelAdapter, the model adapter for the model.
        outputs: OutputsStructure, the outputs dir. Default: None
        task_cfg: TaskConfig, the overall task config. Default: None
        **kwargs: kwargs.
    """

    def __init__(self,
                 data_adapter: DataAdapter,
                 model_adapter: BaseModelAdapter,
                 outputs: OutputsStructure = None,
                 task_cfg: TaskConfig = None,
                 **kwargs):

        self.dataset_name = data_adapter.name
        self.dataset_name_or_path = os.path.expanduser(data_adapter.dataset_id)
        self.model_name = task_cfg.model_id

        self.data_adapter = data_adapter
        self.model_adapter = model_adapter
        self.model_cfg = model_adapter.model_cfg
        self.eval_type = task_cfg.eval_type
        self.dataset_hub = task_cfg.dataset_hub
        self.stage = task_cfg.stage
        self.use_cache = task_cfg.use_cache
        self.task_cfg = task_cfg
        # Deal with the output paths
        self.outputs_structure = outputs
        self.kwargs = kwargs

        self._init_judge()

    def _init_judge(self):
        if self.task_cfg.judge_strategy == JudgeStrategy.RULE:
            self.judge = None
        else:
            from evalscope.metrics import LLMJudge
            self.judge = LLMJudge(**self.task_cfg.judge_model_args)

    def load_dataset(self):
        dataset = self.data_adapter.load(
            work_dir=os.path.expanduser(self.task_cfg.dataset_dir), datasets_hub=self.dataset_hub, **self.kwargs)

        # Get prompts from dataset
        prompts = self.data_adapter.gen_prompts(data_dict=dataset)

        # Limit and index prompts
        limited_prompts = defaultdict(list)
        for subset_name, prompts_list in prompts.items():
            # If limit is None, use all prompts
            if self.task_cfg.limit is None:
                limit = len(prompts_list)
            else:
                if isinstance(self.task_cfg.limit, int):
                    limit = self.task_cfg.limit
                elif isinstance(self.task_cfg.limit, float):
                    limit = int(len(prompts_list) * self.task_cfg.limit)
            # Limit the number of prompts
            for index, prompt in enumerate(prompts_list[:min(limit, len(prompts_list))]):
                prompt[AnswerKeys.INDEX] = index
                limited_prompts[subset_name].append(prompt)

        return limited_prompts

    def _generate_answer_id(self, model_cfg, input_d, infer_cfg):
        model_cfg_str = json.dumps(OrderedDict(sorted(dict_torch_dtype_to_str(model_cfg).items())), ensure_ascii=False)
        input_prompt_str = json.dumps(OrderedDict(sorted(dict_torch_dtype_to_str(input_d).items())), ensure_ascii=False)
        infer_cfg_str = json.dumps(OrderedDict(sorted(dict_torch_dtype_to_str(infer_cfg).items())), ensure_ascii=False)
        return 'answer-' + gen_hash(model_cfg_str + input_prompt_str + infer_cfg_str)

    def _process_answer(self, answer_d, input_d, subset_name, answer_id):
        answer_d[AnswerKeys.MODEL_SPEC] = self.model_adapter.model_cfg
        answer_d[AnswerKeys.ANSWER_ID] = answer_id
        answer_d[AnswerKeys.SUBSET_NAME] = subset_name
        answer_d[AnswerKeys.RAW_INPUT] = input_d[AnswerKeys.RAW_INPUT]
        answer_d[AnswerKeys.INDEX] = input_d[AnswerKeys.INDEX]
        return answer_d

    def _get_answer(self, input_prompts, subset_name, infer_cfg) -> List[dict]:
        answers_list = []
        try:
            # get answer from model
            answer_ds: List[dict] = self.model_adapter.predict(inputs=input_prompts, infer_cfg=infer_cfg)
        except Exception as e:
            logger.error(f'Failed to get answer for {input_prompts}, due to {e}')
            # if ignore_errors is True, continue to next input
            if self.task_cfg.ignore_errors:
                logger.warning('`ignore_errors` is set to True. Dropping this prompt and continuing with evaluation.')
                return answers_list
            else:
                raise e
        # process answer
        for answer_d, input_prompt in zip(answer_ds, input_prompts):
            answer_id = self._generate_answer_id(self.model_adapter.model_cfg, input_prompt, infer_cfg)
            processed_answer = self._process_answer(answer_d, input_prompt, subset_name, answer_id)
            answers_list.append(processed_answer)
        return answers_list

    @staticmethod
    def filter_answer(use_cache, prompts_list, pred_file_path) -> dict:
        # Filter prompts that have been answered
        answers_list = []
        if not use_cache or not os.path.exists(pred_file_path):
            return answers_list, prompts_list

        def get_answered_indices(answers_list: List[Dict]) -> List[int]:
            indices = [answer.get(AnswerKeys.INDEX) for answer in answers_list]

            if all(index is None for index in indices):
                return list(range(len(answers_list)))

            return [index for index in indices if index is not None]

        answers_list = jsonl_to_list(pred_file_path)
        answered_indices = set(get_answered_indices(answers_list))
        logger.info(f'Reusing predictions from {pred_file_path}, got {len(answered_indices)} answers.')

        prompts = [prompt for i, prompt in enumerate(prompts_list) if i not in answered_indices]
        return answers_list, prompts

    def get_answers(self, subset_name: str, prompts_list: List[dict], infer_cfg: dict = None, **kwargs) -> list:
        """
        Get answers from model inference.
        It is required to rewrite this method to support your own evaluator.

        Args:
            subset_name: subset name for benchmark.
            prompts_list: prompts list.
            infer_cfg: model inference config.
                Attributes:
                    do_sample: bool, whether to use sampling.
                    top_k: int, the number of highest probability vocabulary tokens to keep for top-k-filtering.
                    top_p: float, if set to float < 1, only the most probable tokens with probabilities to add.
                    temperature: float, the value used to module the next token probabilities.
                    num_beams: int, number of beams for beam search. 1 means no beam search.
                    max_length: int, the max length of the sequence to be generated.
                    max_new_tokens: int, the max number of new tokens to be generated.
                    repetition_penalty: float, the parameter for repetition penalty. 1.0 means no penalty.
            **kwargs: kwargs.

        Returns: The list of answers.
        """
        assert self.data_adapter is not None, 'data_adapter must be provided when calling func get_answers() !'
        assert self.model_adapter is not None, 'model must be provided when calling func get_answers() !'
        assert len(prompts_list) > 0, 'prompts_list must not be empty when calling func get_answers() !'

        pred_file_name = self.dataset_name + '_' + subset_name + '.jsonl'
        pred_file_path = os.path.join(self.outputs_structure.predictions_dir, self.model_name, pred_file_name)
        os.makedirs(os.path.dirname(pred_file_path), exist_ok=True)

        answers_list, prompts_list = Evaluator.filter_answer(self.use_cache, prompts_list, pred_file_path)

        eval_batch_size = self.task_cfg.eval_batch_size
        if self.task_cfg.eval_type == EvalType.SERVICE:
            with tqdm(total=len(prompts_list), desc=f'Predicting({subset_name}): ') as pbar:
                with ThreadPoolExecutor(max_workers=eval_batch_size) as executor:
                    futures = []
                    for input_prompt in prompts_list:
                        futures.append(executor.submit(self._get_answer, [input_prompt], subset_name, infer_cfg))
                    for future in as_completed(futures):
                        answer_ds: List[dict] = future.result()
                        answers_list.extend(answer_ds)
                        dump_jsonl_data(answer_ds, pred_file_path, dump_mode=DumpMode.APPEND)
                        pbar.update(len(answer_ds))
        else:
            batch_prompts_list = [
                prompts_list[i:i + eval_batch_size] for i in range(0, len(prompts_list), eval_batch_size)
            ]
            with tqdm(total=len(prompts_list), desc=f'Predicting({subset_name}): ') as pbar:
                for batch_prompts in batch_prompts_list:
                    answer_ds: List[dict] = self._get_answer(
                        input_prompts=batch_prompts, subset_name=subset_name, infer_cfg=infer_cfg)
                    answers_list.extend(answer_ds)
                    dump_jsonl_data(answer_ds, pred_file_path, dump_mode=DumpMode.APPEND)
                    pbar.update(len(batch_prompts))

        logger.info(f'Dump predictions to {pred_file_path}.')
        return answers_list

    def _get_review(self, answer_d: dict, review_id: str = None, reviewer_spec: dict = None) -> dict:

        if reviewer_spec is None:
            reviewer_spec = {}

        review_res = deepcopy(answer_d)
        if AnswerKeys.CHOICES not in review_res:
            review_res[AnswerKeys.CHOICES] = []
            review_res[ReviewKeys.REVIEWED] = True
            review_res[ReviewKeys.REVIEW_ID] = None
            review_res[ReviewKeys.REVIEWER_SPEC] = reviewer_spec
            review_res[ReviewKeys.REVIEW_TIME] = time.time()
            logger.warning(f'No choices found for answer dict: {review_res}')
            return review_res

        rev_choices = []
        for choice in review_res[AnswerKeys.CHOICES]:
            raw_input_d: dict = review_res[AnswerKeys.RAW_INPUT]
            answer_content = choice[ReviewKeys.MESSAGE][ReviewKeys.CONTENT]
            gold_content = self.data_adapter.get_gold_answer(raw_input_d)

            # Get review result based on judge strategy
            use_llm = (
                self.task_cfg.judge_strategy == JudgeStrategy.LLM
                or (self.task_cfg.judge_strategy == JudgeStrategy.AUTO and self.data_adapter.llm_as_a_judge))

            if use_llm:
                # Use LLM as judge
                assert self.judge is not None, f'Judge model is required for LLM judging {self.data_adapter.name}'
                review_result = self.data_adapter.llm_match(
                    gold_content, answer_content, self.judge, raw_input=raw_input_d)
                pred = answer_content
            else:
                # Use rule-based judging
                pred_content = self.data_adapter.parse_pred_result(
                    result=answer_content, raw_input_d=raw_input_d, eval_type=self.eval_type)
                review_result = self.data_adapter.match(gold_content, pred_content)

                # For LLM_RECALL strategy, use LLM to re-judge if rule-based result is not good
                if (self.task_cfg.judge_strategy == JudgeStrategy.LLM_RECALL
                        and isinstance(review_result, (bool, int, float)) and not bool(review_result)):
                    assert self.judge is not None, f'Judge model is required for LLM_RECALL strategy {self.data_adapter.name}'  # noqa: E501
                    review_result = self.data_adapter.llm_match(
                        gold_content, answer_content, self.judge, raw_input=raw_input_d)
                    pred = answer_content
                else:
                    pred = pred_content

            choice[ReviewKeys.REVIEW] = {
                ReviewKeys.GOLD: gold_content if gold_content != raw_input_d else '*Same as Input*',
                ReviewKeys.PRED: pred,
                ReviewKeys.RESULT: review_result
            }
            rev_choices.append(choice)

        review_res[AnswerKeys.CHOICES] = rev_choices
        review_res[ReviewKeys.REVIEWED] = True
        review_res[ReviewKeys.REVIEW_ID] = review_id
        review_res[ReviewKeys.REVIEWER_SPEC] = reviewer_spec
        review_res[ReviewKeys.REVIEW_TIME] = time.time()

        return review_res

    def _generate_review_id(self, answer_d):
        # Gen review_id (concat: answer_id + reviewer_spec)
        answer_id = answer_d[AnswerKeys.ANSWER_ID]
        reviewer_spec = {'metric': self.data_adapter.metric_list, 'reviewer': ['Evaluator'], 'revision': ['default']}
        reviewer_spec_str = json.dumps(
            OrderedDict(sorted(dict_torch_dtype_to_str(reviewer_spec).items())), ensure_ascii=False)
        review_id = 'review-' + gen_hash(answer_id + reviewer_spec_str)
        return review_id, reviewer_spec

    def get_reviews(self, subset_name: str, answers_list: List[dict], **kwargs) -> list:
        """
        Get reviews from answers.
        It is required to rewrite this method to support your own evaluator.

        Args:
            subset_name: subset name of benchmark
            answers_list: inference results list.
            **kwargs: kwargs.

        Returns: reviews list.
        """
        reviews_list = []

        review_file_name = self.dataset_name + '_' + subset_name + '.jsonl'
        review_file_path = os.path.join(self.outputs_structure.reviews_dir, self.model_name, review_file_name)
        os.makedirs(os.path.dirname(review_file_path), exist_ok=True)

        # Load existing reviews if using cache
        existing_reviews = {}
        if self.use_cache and os.path.exists(review_file_path):
            with open(review_file_path, 'r') as f:
                for line in f:
                    review = json.loads(line.strip())
                    existing_reviews[review['index']] = review
            logger.info(f'Reusing review result from {review_file_path}, got {len(existing_reviews)} reviews.')

        def process_single_review(answer_d):
            # Check if review already exists in cache
            if self.use_cache and answer_d['index'] in existing_reviews:
                return existing_reviews[answer_d['index']]

            review_id, reviewer_spec = self._generate_review_id(answer_d)
            # Get review
            review_d = self._get_review(answer_d=answer_d, review_id=review_id, reviewer_spec=reviewer_spec)
            logger.debug(review_d)
            return review_d

        with ThreadPoolExecutor(max_workers=self.task_cfg.judge_worker_num) as executor:
            # Submit all tasks and get futures
            futures = [executor.submit(process_single_review, answer_d) for answer_d in answers_list]

            # Process completed futures with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc=f'Reviewing({subset_name}): '):
                review_d = future.result()
                reviews_list.append(review_d)
                # Dump new reviews only if not using cache or review is new
                if not self.use_cache or review_d['index'] not in existing_reviews:
                    dump_jsonl_data(review_d, review_file_path, dump_mode=DumpMode.APPEND)

        return reviews_list

    def compute_metrics(self, reviews_list: List[dict]) -> List[dict]:
        """
        To compute metrics from reviews_list for each subset.
        It is required to rewrite this method to support your own evaluator.

        Args:
            reviews_list: reviews list.

        Returns:
            The metric result. Depends on the metric function in data_adapter.
        """
        # Get max choices
        choices_lengths = [
            len(review_d[AnswerKeys.CHOICES]) for review_d in reviews_list if review_d.get(ReviewKeys.REVIEWED)
        ]
        if choices_lengths:
            max_choices = max(choices_lengths)
        else:
            max_choices = 0

        # Get review result
        review_res_list = []
        for review_d in reviews_list:
            if not review_d[ReviewKeys.REVIEWED]:
                logger.warning(f'Review not finished for answer_id: {review_d[AnswerKeys.ANSWER_ID]}, skipping ...')
                continue

            if len(review_d[AnswerKeys.CHOICES]) == 0:
                logger.warning(f'No choices found for answer_id: {review_d[AnswerKeys.ANSWER_ID]}, skipping ...')
                continue
            elif len(review_d[AnswerKeys.CHOICES]) == 1 and max_choices == 1:
                review_res = review_d[AnswerKeys.CHOICES][0][ReviewKeys.REVIEW][ReviewKeys.RESULT]
            else:
                review_res = [choice[ReviewKeys.REVIEW][ReviewKeys.RESULT] for choice in review_d[AnswerKeys.CHOICES]]
                if len(review_d[AnswerKeys.CHOICES]) < max_choices:
                    logger.warning(
                        f'Less choices found for answer_id: {review_d[AnswerKeys.ANSWER_ID]}, '
                        f'max_choices is {max_choices}, but only {len(review_d[AnswerKeys.CHOICES])} choices found')

            review_res_list.append(review_res)

        metric_score: List[dict] = self.data_adapter.compute_metric(
            review_res_list=review_res_list, reviews_list=reviews_list)

        return metric_score

    def dump_report(self, reviews_score_all: List[dict]):
        """
        Get report for total reviews of specific dataset.
        It is required to rewrite this method to support your own evaluator.

        Args:
            reviews_score_all: reviews score list. Generated by func self.data_adapter.compute_metric().

        Returns: None
        """
        report_path = os.path.join(self.outputs_structure.reports_dir, self.model_name)
        os.makedirs(report_path, exist_ok=True)
        # Get report map
        report_map: Report = self.data_adapter.gen_report(
            subset_score_map=reviews_score_all, model_name=self.model_name)

        # Post process report
        self.data_adapter.post_process_report(report_map, report_path=report_path)

        # Make table
        try:
            report_table = gen_report_table(report_map)
            logger.info(f'{self.dataset_name_or_path} report table: \n{report_table} \n')
        except Exception:
            logger.error('Failed to generate report table.')

        # Make report analysis
        if self.task_cfg.analysis_report:
            logger.info('Generating report analysis, please wait ...')
            analysis = report_map.generate_analysis(self.task_cfg.judge_model_args)
            logger.info('Report analysis:\n%s', analysis)
        else:
            logger.info('Skipping report analysis (`analysis_report=False`).')

        # Dump report
        report_file = os.path.join(report_path, f'{self.dataset_name}.json')
        report_map.to_json(report_file)
        logger.info(f'Dump report to: {report_file} \n')

        return report_map

    def eval(self, **kwargs) -> dict:
        """
        Evaluate the model on the specific benchmark. Streaming & parallel mode is supported.
        It is required to rewrite this method to support your own evaluator.

        The evaluation process is as follows:
            1. Get the input samples from the dataset (benchmarks on the ModelScope or HuggingFace).
            2. Get the input prompts from dataset with specific data adapter.
            3. Get answers with model inference.
            4. Get reviews with metric function (or reviewers).
            5. Generate report from review results.

        Args:
            infer_cfg: The config for model inference.

        Returns:
            Dict of results. Depends on the stage of evaluation.

            stage == 'all': return the report_map
            stage == 'infer': return the answers_map
            stage == 'review': return the reviews_map
        """

        logger.info(f'Start evaluating on dataset {self.dataset_name_or_path}')

        reviews_score_all = {}  # {subset_name: (score, num)}
        stage_answers_dict = {}
        stage_reviews_dict = {}

        prompts = self.load_dataset()
        for subset_name, prompts_list in prompts.items():

            answers_list: list = self.get_answers(
                subset_name=subset_name, prompts_list=prompts_list, infer_cfg=self.task_cfg.generation_config, **kwargs)
            if self.stage == EvalStage.INFER:
                stage_answers_dict[subset_name] = answers_list
                continue

            reviews_list: list = self.get_reviews(subset_name=subset_name, answers_list=answers_list, **kwargs)

            metric_res = self.compute_metrics(reviews_list=reviews_list)
            reviews_score_all[subset_name] = metric_res
            stage_reviews_dict[subset_name] = reviews_list

        if self.stage == EvalStage.INFER:
            return stage_answers_dict

        if self.stage == EvalStage.REVIEW:
            return stage_reviews_dict

        # Generate report
        report_map = self.dump_report(reviews_score_all)

        logger.info(f'Evaluation finished on {self.dataset_name_or_path}')

        return report_map
