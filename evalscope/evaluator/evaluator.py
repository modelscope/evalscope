# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import time
import json
import re
from copy import deepcopy
from collections import OrderedDict

from tqdm import tqdm
from typing import Optional, List, Any, Union, Dict

from evalscope.benchmarks import DataAdapter
from evalscope.constants import DEFAULT_ROOT_CACHE_DIR, OutputsStructure, AnswerKeys, ReviewKeys, EvalStage
from evalscope.models.model_adapter import BaseModelAdapter, CustomModelAdapter
from evalscope.tools.combine_reports import gen_table
from evalscope.utils import gen_hash, dict_torch_dtype_to_str, dump_jsonl_data, process_outputs_structure, \
    normalize_score, dict_to_yaml, jsonl_to_list
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
        subset_list: list, the subset list for the dataset.
        model_adapter: BaseModelAdapter, the model adapter for the model.
        use_cache: bool, whether to use local cache. Default: True
        mem_cache_method: str, the memory cache method. Default: 'ttl' (deprecated)
        root_cache_dir: str, the root cache dir. Default: DEFAULT_ROOT_CACHE_DIR
        outputs_dir: str, the outputs dir. Default: ''
        is_custom_outputs_dir: bool, whether to use custom outputs dir. Default: False  (deprecated)
        datasets_dir: str, the datasets dir. Default: DEFAULT_ROOT_CACHE_DIR
        datasets_hub: str, the datasets hub. `Local`, `ModelScope` or `HuggingFace`. Default: 'ModelScope'
        stage: str, the stage of evaluation. `all` or `infer` or `review`. Default: 'all'
        eval_type: str, the evaluation type. `checkpoint` or `service` or `custom`. Default: 'checkpoint'
        overall_task_cfg: dict, the overall task config. Default: None
        **kwargs: kwargs.
    """

    def __init__(self,
                 dataset_name_or_path: str,
                 data_adapter: DataAdapter,
                 subset_list: Optional[list] = None,
                 model_adapter: Optional[BaseModelAdapter] = None,
                 use_cache: bool = True,
                 mem_cache_method: str = 'ttl',
                 root_cache_dir: Optional[str] = DEFAULT_ROOT_CACHE_DIR,
                 outputs_dir: Optional[str] = '',
                 is_custom_outputs_dir: bool = False,
                 datasets_dir: Optional[str] = DEFAULT_ROOT_CACHE_DIR,
                 datasets_hub: Optional[str] = 'ModelScope',
                 stage: Optional[str] = 'all',      # refer to evalscope.constants.EvalStage
                 eval_type: Optional[str] = 'checkpoint',  # `checkpoint` or `service` or `custom`
                 overall_task_cfg: Optional[dict] = None,
                 **kwargs):

        self.dataset_name_or_path = os.path.expanduser(dataset_name_or_path)
        self.custom_task_name: str = None
        if os.path.exists(self.dataset_name_or_path):
            self.custom_task_name = os.path.basename(self.dataset_name_or_path.rstrip(os.sep))

        self.root_cache_dir = os.path.expanduser(root_cache_dir)
        self.datasets_dir = os.path.expanduser(datasets_dir)
        self.kwargs = kwargs
        self.data_adapter = data_adapter
        self.model_adapter = model_adapter
        self.eval_type = eval_type
        self.stage = stage
        self.use_cache = use_cache
        self.overall_task_cfg = overall_task_cfg
        if isinstance(self.model_adapter, CustomModelAdapter):
            self.overall_task_cfg.update({'custom_config': self.model_adapter.custom_model.config})

        self.model_cfg = self.model_adapter.model_cfg
        self.model_id = self.model_cfg['model_id']
        self.model_revision = self.model_cfg.get('revision', None)
        self.model_revision_str = self.model_revision if self.model_revision is not None else 'none'

        # Get default outputs_dir
        # TODO: refactor outputs_dir, del timestamp concat
        # if not is_custom_outputs_dir:
        #     outputs_dir = make_outputs_dir(work_dir=outputs_dir,
        #                                    model_id=self.model_id,
        #                                    model_revision=self.model_revision_str)

        self.outputs_dir = os.path.expanduser(outputs_dir)

        # Deal with the output paths
        self.outputs_structure = process_outputs_structure(self.outputs_dir)

        # Load dataset
        self.dataset = self.data_adapter.load(dataset_name_or_path=dataset_name_or_path,
                                              subset_list=subset_list,
                                              work_dir=self.datasets_dir,
                                              datasets_hub=datasets_hub,
                                              **kwargs)

        # Get prompts from dataset
        self.prompts = self.data_adapter.gen_prompts(data_dict=self.dataset)
        del self.dataset

        # Init memory cache
        # TODO: refactor mem cache manager
        # mem_cache_file_name = self.dataset_name_or_path.replace('/', '_') + \
        #     '_' + self.model_id.replace('/', '_') + \
        #     '_' + self.model_revision_str + \
        #     '_cache.pkl'
        # self.mem_cache_path = os.path.join(self.root_cache_dir, 'mem_cache', mem_cache_file_name)

        # Note: mem_cache is deprecated, use `use_cache` instead
        self.mem_cache = None
        self.mem_cache_method = mem_cache_method
        # if self.use_cache:
        #     self.mem_cache = init_mem_cache(method=self.mem_cache_method, cache_file_path=self.mem_cache_path)
        #     logger.info(f'** Using memory cache with size: {len(self.mem_cache)}')

    def _pred_answer(self,
                     input_d: dict,
                     infer_cfg: dict,
                     subset_name: str,
                     answer_id: str = None) -> dict:

        # Get answer from memory cache
        if self.mem_cache is not None:
            if answer_id in self.mem_cache:
                logger.info(f'** Reusing answer `{answer_id}` in memory cache.')
                return self.mem_cache[answer_id]

        ans: dict = self.model_adapter.predict(inputs=input_d, infer_cfg=infer_cfg)
        ans[AnswerKeys.ANSWER_ID] = answer_id
        ans[AnswerKeys.SUBSET_NAME] = subset_name

        if self.mem_cache is not None:
            self.mem_cache[answer_id] = ans

        return ans

    def get_answers(self,
                    subset_name: str,
                    prompts_list: List[dict],
                    infer_cfg: dict = None,
                    debug: bool = False,
                    **kwargs) -> list:
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
            debug: whether to run in debug mode.
            **kwargs: kwargs.

        Returns: The list of answers.
        """
        assert self.data_adapter is not None, 'data_adapter must be provided when calling func get_answers() !'
        assert self.model_adapter is not None, 'model must be provided when calling func get_answers() !'

        answers_list = []
        pred_dir: str = self.outputs_structure.get(OutputsStructure.PREDICTIONS_DIR)

        if self.custom_task_name:
            pred_file_name: str = self.custom_task_name + '_' + subset_name + '.jsonl'
        else:
            pred_file_name: str = self.dataset_name_or_path.replace(os.sep, '_') + '_' + subset_name + '.jsonl'

        pred_file_path: str = os.path.join(pred_dir, pred_file_name)

        if self.use_cache and os.path.exists(pred_file_path):
            answers_list = jsonl_to_list(pred_file_path)
            logger.info(f'** Reusing predictions from {pred_file_path}, got {len(answers_list)} answers.')

            return answers_list

        if isinstance(self.model_adapter, CustomModelAdapter):
            # Batch inference for custom model

            resp_answers_list: List[Dict[str, Any]] = self.model_adapter.predict(inputs=prompts_list,
                                                                                 infer_cfg=infer_cfg)

            assert len(prompts_list) == len(resp_answers_list), \
                f'Length of prompts_list({len(prompts_list)}) != Length of resp_answers_list({len(resp_answers_list)})'

            for in_d, resp_d in zip(prompts_list, resp_answers_list):

                # Gen answer_id (concat: model_cfg + input_prompt + infer_cfg)
                model_cfg_str = json.dumps(
                    OrderedDict(sorted(dict_torch_dtype_to_str(self.model_adapter.model_cfg).items())),
                    ensure_ascii=False)
                input_prompt_str = json.dumps(OrderedDict(sorted(dict_torch_dtype_to_str(in_d).items())),
                                              ensure_ascii=False)
                infer_cfg_str = json.dumps(OrderedDict(sorted(dict_torch_dtype_to_str(infer_cfg).items())),
                                           ensure_ascii=False)
                answer_id = 'answer-' + gen_hash(model_cfg_str + input_prompt_str + infer_cfg_str)

                resp_d[AnswerKeys.MODEL_SPEC] = self.model_adapter.model_cfg
                resp_d[AnswerKeys.ANSWER_ID] = answer_id
                resp_d[AnswerKeys.SUBSET_NAME] = subset_name
                resp_d[AnswerKeys.RAW_INPUT] = in_d[AnswerKeys.RAW_INPUT]
                resp_d[AnswerKeys.ORIGIN_PROMPT] = in_d

                answers_list.append(resp_d)

        else:
            for input_prompt in tqdm(prompts_list, total=len(prompts_list), desc=f'Predicting({subset_name}): '):

                # Gen answer_id (concat: model_cfg + input_prompt + infer_cfg)
                model_cfg_str = json.dumps(
                    OrderedDict(sorted(dict_torch_dtype_to_str(self.model_adapter.model_cfg).items())),
                    ensure_ascii=False)
                input_prompt_str = json.dumps(OrderedDict(sorted(dict_torch_dtype_to_str(input_prompt).items())),
                                              ensure_ascii=False)
                infer_cfg_str = json.dumps(OrderedDict(sorted(dict_torch_dtype_to_str(infer_cfg).items())),
                                           ensure_ascii=False)
                answer_id = 'answer-' + gen_hash(model_cfg_str + input_prompt_str + infer_cfg_str)

                # Get answers
                answer_d: dict = self._pred_answer(input_d=input_prompt,
                                                   infer_cfg=infer_cfg,
                                                   subset_name=subset_name,
                                                   answer_id=answer_id)

                answer_d[AnswerKeys.MODEL_SPEC] = self.model_adapter.model_cfg
                answer_d[AnswerKeys.RAW_INPUT] = input_prompt[AnswerKeys.RAW_INPUT]
                answer_d[AnswerKeys.ORIGIN_PROMPT] = input_prompt

                if debug:
                    logger.debug(f'**input_prompt: {json.dumps(input_prompt, ensure_ascii=False)} \n')
                    logger.debug(f'**predicted ans: {json.dumps(answer_d, ensure_ascii=False)} \n')

                answers_list.append(answer_d)

        if len(answers_list) == 0:
            logger.error(f'** Got empty predictions on subset {subset_name} of dataset: {self.dataset_name_or_path}')

        # Dump answers
        os.makedirs(pred_dir, exist_ok=True)
        dump_jsonl_data(answers_list, pred_file_path)

        return answers_list

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
            answer_content = self.data_adapter.parse_pred_result(result=answer_content,
                                                                 raw_input_d=raw_input_d,
                                                                 eval_type=self.eval_type)
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

        review_dir: str = self.outputs_structure.get(OutputsStructure.REVIEWS_DIR)
        if self.custom_task_name:
            review_file_name: str = self.custom_task_name + '_' + subset_name + '.jsonl'
        else:
            review_file_name: str = self.dataset_name_or_path.replace(os.sep, '_') + '_' + subset_name + '.jsonl'
        review_file_path: str = os.path.join(review_dir, review_file_name)

        if self.use_cache and os.path.exists(review_file_path):
            logger.warning(f'** Ignore use_cache={self.use_cache}, updating the review file: {review_file_path} ...')

        for answer_d in tqdm(answers_list, total=len(answers_list), desc=f'Reviewing({subset_name}): '):

            # Gen review_id (concat: answer_id + reviewer_spec)
            answer_id = answer_d[AnswerKeys.ANSWER_ID]

            reviewer_spec: dict = {'metric': [metric_d['name'] for metric_d in self.data_adapter.metric_list],
                                   'reviewer': ['Evaluator'],
                                   'revision': ['default']}
            reviewer_spec_str = json.dumps(OrderedDict(sorted(dict_torch_dtype_to_str(reviewer_spec).items())),
                                           ensure_ascii=False)
            review_id = 'review-' + gen_hash(answer_id + reviewer_spec_str)

            # Get review
            review_d = self._get_review(answer_d=answer_d, review_id=review_id, reviewer_spec=reviewer_spec)

            if debug:
                logger.debug(review_d)

            reviews_list.append(review_d)

        # Dump reviews
        os.makedirs(review_dir, exist_ok=True)
        dump_jsonl_data(reviews_list, review_file_path)

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

    def dump_report(self, report_map: dict, use_table: bool = True):
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

        if self.custom_task_name:
            report_file_name: str = self.custom_task_name + '.json'
        else:
            report_file_name: str = self.dataset_name_or_path.replace(os.sep, '_') + '.json'

        os.makedirs(report_dir, exist_ok=True)
        report_path: str = os.path.join(report_dir, report_file_name)
        with open(report_path, 'w') as f:
            f.write(json.dumps(report_map, ensure_ascii=False, indent=4))
        # logger.info(f'** Dump report to {report_path} \n')
        logger.info(f'** Dump report: {report_file_name} \n')

        if use_table:
            try:
                # Make table
                report_table: str = gen_table([report_dir])
                logger.info(f'** Report table: \n {report_table} \n')
            except:
                logger.error('Failed to generate report table.')

    # def save_cache(self):
    #     if self.mem_cache is not None:
    #         logger.info(f'** Saving memory cache with size: {len(self.mem_cache)}')
    #         Cache.save(cache=self.mem_cache, path=self.mem_cache_path)

    # def clear_cache(self):
    #     """
    #     Clear memory cache.
    #
    #     Returns: None
    #     """
    #     if self.mem_cache is not None:
    #         cache_len = len(self.mem_cache)
    #         self.mem_cache.clear()
    #         logger.info(f'** Memory cache cleared, length changed: {cache_len} -> {len(self.mem_cache)}')

    def eval(self,
             infer_cfg: dict = None,
             debug: bool = False,
             **kwargs) -> dict:
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
            Dict of results. Depends on the stage of evaluation.

            stage == 'all': return the report_map
            stage == 'infer': return the answers_map
            stage == 'review': return the reviews_map
        """

        logger.info(f'**** Start evaluating on dataset {self.dataset_name_or_path} ****')

        reviews_score_all = {}      # {subset_name: (score, num)}
        stage_answers_dict = {}
        stage_reviews_dict = {}

        for subset_name, prompts_list in self.prompts.items():
            limit = infer_cfg.get('limit', len(prompts_list))
            prompts_list = prompts_list[:limit]

            answers_list: list = self.get_answers(subset_name=subset_name,
                                                  prompts_list=prompts_list,
                                                  infer_cfg=infer_cfg,
                                                  debug=debug,
                                                  **kwargs)
            if self.stage == EvalStage.INFER:
                stage_answers_dict[subset_name] = answers_list
                continue

            reviews_list: list = self.get_reviews(subset_name=subset_name,
                                                  answers_list=answers_list,
                                                  debug=debug,
                                                  **kwargs)

            metric_res = self.compute_metrics(reviews_list=reviews_list)
            reviews_score_all[subset_name] = (metric_res, len(reviews_list))
            stage_reviews_dict[subset_name] = reviews_list

        if self.stage == EvalStage.INFER:
            return stage_answers_dict

        if self.stage == EvalStage.REVIEW:
            return stage_reviews_dict

        # Generate report
        report_map: dict = self.data_adapter.gen_report(subset_score_map=reviews_score_all,
                                                        report_name=self.custom_task_name)
        self.dump_report(report_map=report_map)

        # Dump overall task config
        overall_task_cfg_file: str = os.path.join(self.outputs_structure.get(OutputsStructure.CONFIGS_DIR),
                                                  'task_output_config.yaml')
        overall_task_cfg_file = os.path.abspath(overall_task_cfg_file)

        # TODO: check the robustness of dump yaml
        try:
            logger.info(f'** Dump overall task config to {overall_task_cfg_file}')
            logger.info(f'** The overall task config:\n {self.overall_task_cfg}')
            if 'model' in self.overall_task_cfg and not isinstance(self.overall_task_cfg['model'], str):
                self.overall_task_cfg['model'] = None
                logger.info(f'>> Overwrite overall_task_cfg for `model` due to it is not a string')
            if 'model_args' in self.overall_task_cfg and self.overall_task_cfg.get('model_args') is not None:
                self.overall_task_cfg['model_args'].update({'precision': str(self.overall_task_cfg['model_args']['precision'])})
                logger.info(f'>> Overwrite overall_task_cfg for `model_args.precision` due to it is not a string')

            dict_to_yaml(self.overall_task_cfg, overall_task_cfg_file)
        except Exception as e:
            logger.warning(f'Failed to dump overall task config: {e}')

        # Note: deprecated
        # self.save_cache()
        # self.clear_cache()

        logger.info(f'\n**** Evaluation finished on {self.dataset_name_or_path} ****\n')

        return report_map


class HumanevalEvaluator(object):

    def __init__(self,
                 problem_file: str,
                 model_id: str,
                 model_revision: str,
                 model_adapter: BaseModelAdapter,
                 outputs_dir: Optional[str] = '',
                 is_custom_outputs_dir: bool = False,
                 k: List[int] = [1, 10, 100],
                 n_workers: int = 4,
                 timeout: float = 3.0,):
        try:
            from human_eval.evaluation import evaluate_functional_correctness
            from human_eval.data import read_problems, write_jsonl
        except ImportError:
            raise ImportError('Please install human_eval:'
                              'https://github.com/openai/human-eval/tree/master#installation , '
                              'Note that you need to enable the execution code in the human_eval/execution.py first.')

        self.problem_file = problem_file
        self.k = k
        self.num_workers = n_workers
        self.timeout = timeout
        self.model_adapter = model_adapter

        self.read_problems_func = read_problems
        self.write_jsonl_func = write_jsonl
        self.eval_func = evaluate_functional_correctness

        # {'task_id': {'task_id': '', 'prompt': '', 'entry_point': '', 'canonical_solution': '', 'test': ''}, ...}
        self.problems = self.read_problems_func(self.problem_file)

        # Get default outputs_dir
        model_revision_str: str = model_revision if model_revision is not None else 'none'
        # if not is_custom_outputs_dir:
        #     outputs_dir = make_outputs_dir(work_dir=outputs_dir,
        #                                    model_id=model_id,
        #                                    model_revision=model_revision_str)
        self.outputs_dir = os.path.expanduser(outputs_dir)

        # Deal with the output paths
        self.outputs_structure = process_outputs_structure(self.outputs_dir)

    def get_answers(self, infer_cfg: dict) -> List[dict]:
        ans_list: list = []
        system_prompt: str = 'Complete the following python code:\n'
        for task_id, data_d in tqdm(self.problems.items(), total=len(self.problems), desc='Predicting(problems)'):
            prompt: str = system_prompt + data_d['prompt']
            inputs: dict = {'data': [prompt]}
            # pred_res: dict = self.model_adapter.predict(inputs)

            pred_res: dict = self.model_adapter.predict(inputs=inputs, infer_cfg=infer_cfg)

            pred_ans: str = pred_res['choices'][0]['message']['content']
            pred_ans = self._postprocess(pred_ans)

            ans_list.append({'task_id': task_id, 'completion': pred_ans})

        return ans_list

    def eval(self, infer_cfg: dict, **kwargs):

        # predict
        ans_list: list = self.get_answers(infer_cfg)
        ans_out_file: str = os.path.join(self.outputs_structure.get(OutputsStructure.PREDICTIONS_DIR),
                                         'human_eval_predictions.jsonl')

        self.write_jsonl_func(filename=ans_out_file, data=ans_list)
        # logger.info(f'** Dump predictions to {ans_out_file} successfully.')
        logger.info('** Dump predictions successfully.')

        # evaluate  results: e.g. {'pass@1': 0.333, 'pass@10': 0.111}
        results = self.eval_func(sample_file=ans_out_file,
                                 k=self.k,
                                 n_workers=self.num_workers,
                                 timeout=self.timeout,
                                 problem_file=self.problem_file)

        # output: report
        report_map: dict = self.gen_report(results=results)
        report_dir: str = self.outputs_structure.get(OutputsStructure.REPORTS_DIR)
        report_file: str = os.path.join(report_dir, 'human_eval_report.json')

        with open(report_file, 'w') as f:
            f.write(json.dumps(report_map, ensure_ascii=False, indent=4))
        # logger.info(f'** Dump report to {report_file} \n')
        logger.info(f'** Dump report \n')

        try:
            # Make table
            report_table: str = gen_table([report_dir])
            logger.info(f'** Report table: \n {report_table} \n')
        except:
            logger.error('Failed to generate report table.')

    def gen_report(self, results: dict) -> dict:
        """
        Generate report from evaluation results.

        Returns:
            {
            "name":"ARC-Challenge",
            "metric":"WeightedAverageAccuracy",
            "score":0.3389,
            "category":[
                {
                    "name":"DEFAULT",
                    "score":0.3389,
                    "subset":[
                        {
                            "name":"ARC-Challenge",
                            "score":0.3389
                        },
                    ]
                }
            ],
            "total_num":100
        }
        """
        results = {k: normalize_score(score=v) for k, v in results.items()}

        category_d = dict(name='DEFAULT',
                          score=results,
                          subset=[])

        res_map = dict(name='HumanEval',
                       metric='pass@k',
                       score=results,
                       category=[category_d],
                       total_num=len(self.problems))

        return res_map

    @classmethod
    def _postprocess(cls, text: str) -> str:
        if '```' in text:
            blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
            if len(blocks) == 0:
                text = text.split('```')[1]  # fall back to default strategy
            else:
                text = blocks[0]  # fetch the first code block
                if not text.startswith('\n'):  # in case starting with ```python
                    text = text[max(text.find('\n') + 1, 0):]
        if text.strip().startswith('from') or text.strip().startswith('import'):
            def_idx = text.find('def')
            if def_idx != -1:
                text = text[max(text.find('\n', def_idx) + 1, 0):]
        text = text.split('\n\n')[0]
        if text.strip().startswith('def'):
            text = '\n'.join(text.split('\n')[1:])
        if not text.startswith('    '):
            if text.startswith(' '):
                text = '    ' + text.lstrip()
            else:
                text = '\n'.join(['    ' + line for line in text.split('\n')])
        return text
