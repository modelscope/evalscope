import os
import time
import json
from copy import deepcopy
from collections import OrderedDict

from tqdm import tqdm
from typing import Optional, List, Any, Union

from llmuses.benchmarks import DataAdapter
from llmuses.models.model_adapter import BaseModelAdapter
from llmuses.cache import Cache, init_mem_cache
from llmuses.constants import DEFAULT_ROOT_CACHE_DIR, OutputsStructure, AnswerKeys, ReviewKeys
from llmuses.tools.combine_reports import gen_table
from llmuses.utils import gen_hash, remove_objects_in_dict, dump_jsonl_data, make_outputs_structure, make_outputs_dir, \
    jsonl_to_list
from llmuses.utils.logger import get_logger

logger = get_logger()


class EvaluateEngine(object):

    """
    The evaluate engine for model inference results on datasets
    """

    def __init__(self,
                 dataset_name_or_path: str,
                 data_adapter: DataAdapter,
                 subset_list: Optional[list] = None,
                 model_adapter: Optional[BaseModelAdapter] = None,
                 qwen_model_adapter: Optional[BaseModelAdapter] = None,
                 use_cache: bool = True,
                 mem_cache_method: str = 'ttl',
                 root_cache_dir: Optional[str] = DEFAULT_ROOT_CACHE_DIR,
                 outputs_dir: Optional[str] = '',
                 is_custom_outputs_dir: bool = False,
                 datasets_dir: Optional[str] = DEFAULT_ROOT_CACHE_DIR,
                 datasets_hub: Optional[str] = 'ModelScope',
                 stage: Optional[str] = 'all',
                 user_prompt: dict = {},
                 **kwargs):

        self.dataset_name_or_path = dataset_name_or_path
        self.root_cache_dir = os.path.expanduser(root_cache_dir)
        self.datasets_dir = os.path.expanduser(datasets_dir)
        self.kwargs = kwargs
        self.data_adapter = data_adapter
        self.model_adapter = model_adapter
        self.qwen_model_adapter = qwen_model_adapter

        self.model_cfg = self.model_adapter.model_cfg
        self.model_id = self.model_cfg['model_id']
        if self.qwen_model_adapter:
            self.qwen_model_cfg = self.qwen_model_adapter.model_cfg
            self.qwen_model_id = self.qwen_model_cfg['model_id']
        self.model_revision = self.model_cfg.get('revision', None)
        self.model_revision_str = self.model_revision if self.model_revision is not None else 'none'

        # Get default outputs_dir
        if is_custom_outputs_dir:
            logger.info(f'Deprecated: Please use the default outputs_dir.')

        outputs_dir = make_outputs_dir(work_dir=outputs_dir,
                                       model_id=self.model_id,
                                       model_revision=self.model_revision_str,
                                       dataset_id=self.dataset_name_or_path)
        qwen_outputs_dir = make_outputs_dir(work_dir=outputs_dir,
                                            model_id=self.qwen_model_id,
                                            model_revision='none',
                                            dataset_id=self.dataset_name_or_path) if self.qwen_model_adapter else ""

        self.outputs_dir = os.path.expanduser(outputs_dir)
        self.qwen_outputs_dir = os.path.expanduser(qwen_outputs_dir)

        # Deal with the output paths
        self.outputs_structure = make_outputs_structure(self.outputs_dir)
        self.qwen_outputs_structure = make_outputs_structure(self.qwen_outputs_dir) if self.qwen_model_adapter else ""

        # Load dataset
        self.dataset = self.data_adapter.load(dataset_name_or_path=dataset_name_or_path,
                                              subset_list=subset_list,
                                              work_dir=self.datasets_dir,
                                              datasets_hub=datasets_hub,
                                              **kwargs)

        # Get prompts from dataset
        self.prompts = self.data_adapter.gen_prompts(data_dict=self.dataset, user_prompt=user_prompt)
        del self.dataset

        # Init memory cache
        # TODO: refactor mem cache manager
        mem_cache_file_name = self.dataset_name_or_path.replace('/', '_') + \
            '_' + self.model_id.replace('/', '_') + \
            '_' + self.model_revision_str + \
            '_cache.pkl'
        self.mem_cache_path = os.path.join(self.root_cache_dir, 'mem_cache', mem_cache_file_name)
        self.use_cache = use_cache
        self.mem_cache_method = mem_cache_method
        self.mem_cache = None
        if self.use_cache:
            self.mem_cache = init_mem_cache(method=self.mem_cache_method, cache_file_path=self.mem_cache_path)
            logger.info(f'** Using memory cache with size: {len(self.mem_cache)}')

    def _get_review(self,
                    answer_d: dict,
                    review_id: str = None,
                    reviewer_spec: dict = None) -> dict:

        # Get review from memory cache
        if self.mem_cache is not None:
            if review_id in self.mem_cache:
                logger.info(f'** Reusing review `{review_id}` in memory cache.')
                return self.mem_cache[review_id]

        if reviewer_spec is None:
            reviewer_spec = {}

        review_res = deepcopy(answer_d)
        choices = review_res[AnswerKeys.CHOICES]
        if len(choices) == 0:
            review_res[ReviewKeys.REVIEWED] = False
            review_res[ReviewKeys.REVIEW_ID] = None
            review_res[ReviewKeys.REVIEWER_SPEC] = reviewer_spec
            review_res[ReviewKeys.REVIEW_TIME] = time.time()
            return review_res

        rev_choices = []
        for choice in choices:
            raw_input_d: dict = review_res[AnswerKeys.RAW_INPUT]
            answer_content = choice[ReviewKeys.MESSAGE][ReviewKeys.CONTENT]
            answer_content = self.data_adapter.parse_pred_result(answer_content, raw_input_d)
            gold_content = self.data_adapter.get_gold_answer(raw_input_d)

            review_result = self.data_adapter.match(gold_content, answer_content)
            choice[ReviewKeys.REVIEW] = {ReviewKeys.GOLD: gold_content,
                                         ReviewKeys.PRED: answer_content,
                                         ReviewKeys.RESULT: review_result}

            rev_choices.append(choice)

        review_res[AnswerKeys.CHOICES] = rev_choices
        review_res[ReviewKeys.REVIEWED] = True
        review_res[ReviewKeys.REVIEW_ID] = review_id
        review_res[ReviewKeys.REVIEWER_SPEC] = reviewer_spec
        review_res[ReviewKeys.REVIEW_TIME] = time.time()

        if self.mem_cache is not None:
            self.mem_cache[review_id] = review_res

        return review_res

    def get_reviews(self, subset_name: str, answers_list: List[dict], debug: bool = False, **kwargs) -> list:
        """
        Get reviews from answers.
        It is required to rewrite this method to support your own evaluator.

        Args:
            subset_name: subset name of benchmark
            answers_list: inference results list.
            debug: whether to run in debug mode.
            **kwargs: kwargs.

        Returns: reviews list.
        """
        reviews_list = []
        for answer_d in tqdm(answers_list, total=len(answers_list), desc=f'Reviewing({subset_name}): '):

            # Gen review_id (concat: answer_id + reviewer_spec)
            answer_id = answer_d[AnswerKeys.ANSWER_ID]

            reviewer_spec: dict = {'metric': [metric_d['name'] for metric_d in self.data_adapter.metric_list],
                                   'reviewer': ['Evaluator'],
                                   'revision': ['default']}
            reviewer_spec_str = json.dumps(OrderedDict(sorted(remove_objects_in_dict(reviewer_spec).items())),
                                           ensure_ascii=False)
            review_id = 'review-' + gen_hash(answer_id + reviewer_spec_str)

            # Get review
            review_d = self._get_review(answer_d=answer_d, review_id=review_id, reviewer_spec=reviewer_spec)

            if debug:
                logger.debug(review_d)

            reviews_list.append(review_d)

        # Dump reviews
        review_dir: str = self.outputs_structure.get(OutputsStructure.REVIEWS_DIR)
        review_file_name: str = self.dataset_name_or_path.replace('/', '_') + '_' + subset_name + '.jsonl'
        os.makedirs(review_dir, exist_ok=True)
        dump_jsonl_data(reviews_list, os.path.join(review_dir, review_file_name))

        return reviews_list

    def compute_metrics(self, reviews_list: List[dict]) -> Any:
        """
        To compute metrics from reviews_list for each subset.
        It is required to rewrite this method to support your own evaluator.

        Args:
            reviews_list: reviews list.

        Returns:
            The metric result. Depends on the metric function in data_adapter.
        """

        review_res_list = []
        for review_d in reviews_list:
            if not review_d[ReviewKeys.REVIEWED]:
                logger.warning(f'** Review not finished for answer_id: {review_d[AnswerKeys.ANSWER_ID]}')
                continue

            review_res = review_d[AnswerKeys.CHOICES][0][ReviewKeys.REVIEW][ReviewKeys.RESULT]
            review_res_list.append(review_res)

        metric_score: Union[float, dict] = self.data_adapter.compute_metric(review_res_list=review_res_list)

        return metric_score

    def dump_report(self, report_map: dict, qwen_report_map: dict = None, use_table: bool = True):
        """
        Get report for total reviews of specific dataset.
        It is required to rewrite this method to support your own evaluator.

        Args:
            report_map: report dict. Generated by func self.data_adapter.gen_report().
            use_table: whether to generate table for reports. Default to True.

        Returns: None
        """

        # Dump report
        report_dir: str = self.outputs_structure[OutputsStructure.REPORTS_DIR]
        report_file_name: str = self.dataset_name_or_path.replace('/', '_') + '.json'
        os.makedirs(report_dir, exist_ok=True)
        report_path: str = os.path.join(report_dir, report_file_name)
        with open(report_path, 'w') as f:
            f.write(json.dumps(report_map, ensure_ascii=False, indent=4))
        # logger.info(f'** Dump report to {report_path} \n')
        logger.info(f'** Dump report: {report_file_name} \n')

        report_list = [report_dir]

        # Dump Qwen report
        if qwen_report_map:
            qwen_report_dir: str = self.qwen_outputs_structure[OutputsStructure.REPORTS_DIR]
            os.makedirs(qwen_report_dir, exist_ok=True)
            qwen_report_path: str = os.path.join(qwen_report_dir, report_file_name)
            with open(qwen_report_path, 'w') as f:
                f.write(json.dumps(qwen_report_map, ensure_ascii=False, indent=4))
            # logger.info(f'** Dump report to {report_path} \n')
            logger.info(f'** Dump Qwen report: {report_file_name} \n')
            report_list.append(qwen_report_dir)

        if use_table:
            try:
                # Make table
                report_table: str = gen_table(report_list)
                logger.info(f'** Report table: \n {report_table} \n')
            except:
                logger.error('Failed to generate report table.')
    
    def get_answers(self, subset_name):
        pred_dir: str = self.outputs_structure.get(OutputsStructure.PREDICTIONS_DIR)
        pred_file_name: str = self.dataset_name_or_path.replace('/', '_') + '_' + subset_name + '.jsonl'
        answer_list = jsonl_to_list(os.path.join(pred_dir, pred_file_name))
        return answer_list

    def get_qwen_answers(self, subset_name):
        pred_dir: str = self.qwen_outputs_structure.get(OutputsStructure.PREDICTIONS_DIR)
        pred_file_name: str = self.dataset_name_or_path.replace('/', '_') + '_' + subset_name + '.jsonl'
        answer_list = jsonl_to_list(os.path.join(pred_dir, pred_file_name))
        return answer_list

    def save_cache(self):
        if self.mem_cache is not None:
            logger.info(f'** Saving memory cache with size: {len(self.mem_cache)}')
            Cache.save(cache=self.mem_cache, path=self.mem_cache_path)

    def clear_cache(self):
        """
        Clear memory cache.

        Returns: None
        """
        if self.mem_cache is not None:
            cache_len = len(self.mem_cache)
            self.mem_cache.clear()
            logger.info(f'** Memory cache cleared, length changed: {cache_len} -> {len(self.mem_cache)}')

    def eval(self,
             debug: bool = False,
             **kwargs):
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
            debug: Whether to run in debug mode. Default: False.

        Returns:
            None.
        """

        logger.info(f'**** Start evaluating on dataset {self.dataset_name_or_path} ****')

        reviews_map_all = {}      # {subset_name: (score, num)}
        qwen_review_map_all = {}
        for subset_name, _ in self.prompts.items():

            answers_list: list = self.get_answers(subset_name=subset_name)

            reviews_list: list = self.get_reviews(subset_name=subset_name,
                                                  answers_list=answers_list,
                                                  debug=debug,
                                                  **kwargs)

            metric_res = self.compute_metrics(reviews_list=reviews_list)
            reviews_map_all[subset_name] = (metric_res, len(reviews_list))

            if self.qwen_model_adapter:
                qwen_answers_list: list = self.get_qwen_answers(subset_name=subset_name)

                qwen_reviews_list: list = self.get_reviews(subset_name=subset_name,
                                                           answers_list=qwen_answers_list,
                                                           debug=debug,
                                                           **kwargs)

                qwen_metric_res = self.compute_metrics(reviews_list=qwen_reviews_list)
                qwen_review_map_all[subset_name] = (qwen_metric_res, len(qwen_reviews_list))

        # Generate report
        report_map: dict = self.data_adapter.gen_report(subset_score_map=reviews_map_all)
        qwen_report_map: dict = self.data_adapter.gen_report(subset_score_map=qwen_review_map_all) if self.qwen_model_adapter else None
        self.dump_report(report_map=report_map, qwen_report_map=qwen_report_map)

        self.save_cache()
        self.clear_cache()

        logger.info(f'\n**** Evaluation finished on {self.dataset_name_or_path} ****\n')