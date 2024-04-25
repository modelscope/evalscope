import os
import json
from tqdm import tqdm
from collections import OrderedDict
from typing import Optional, List

from llmuses.benchmarks import DataAdapter
from llmuses.models.model_adapter import BaseModelAdapter
from llmuses.cache import Cache, init_mem_cache
from llmuses.constants import DEFAULT_ROOT_CACHE_DIR, OutputsStructure, AnswerKeys
from llmuses.utils import gen_hash, dict_torch_dtype_to_str, dump_jsonl_data, make_outputs_structure, make_outputs_dir
from llmuses.utils.logger import get_logger

logger = get_logger()


class InferenceEngine(object):

    """
    The inference engine for model on datasets.
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

    def _qwen_pred_answer(self,
                     input_d: dict,
                     infer_cfg: dict,
                     subset_name: str,
                     answer_id: str = None) -> dict:

        # Get answer from memory cache
        if self.mem_cache is not None:
            if answer_id in self.mem_cache:
                logger.info(f'** Reusing answer `{answer_id}` in memory cache.')
                return self.mem_cache[answer_id]

        ans: dict = self.qwen_model_adapter.predict(inputs=input_d, infer_cfg=infer_cfg)
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

        # Dump answers
        pred_dir: str = self.outputs_structure.get(OutputsStructure.PREDICTIONS_DIR)
        pred_file_name: str = self.dataset_name_or_path.replace('/', '_') + '_' + subset_name + '.jsonl'
        os.makedirs(pred_dir, exist_ok=True)
        dump_jsonl_data(answers_list, os.path.join(pred_dir, pred_file_name))

        return answers_list

    def get_qwen_answers(self,
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
        assert self.qwen_model_adapter is not None, 'model must be provided when calling func get_answers() !'

        answers_list = []
        for input_prompt in tqdm(prompts_list, total=len(prompts_list), desc=f'Predicting({subset_name}): '):

            # Gen answer_id (concat: model_cfg + input_prompt + infer_cfg)
            model_cfg_str = json.dumps(
                OrderedDict(sorted(dict_torch_dtype_to_str(self.qwen_model_adapter.model_cfg).items())),
                ensure_ascii=False)
            input_prompt_str = json.dumps(OrderedDict(sorted(dict_torch_dtype_to_str(input_prompt).items())),
                                          ensure_ascii=False)
            infer_cfg_str = json.dumps(OrderedDict(sorted(dict_torch_dtype_to_str(infer_cfg).items())),
                                       ensure_ascii=False)
            answer_id = 'answer-' + gen_hash(model_cfg_str + input_prompt_str + infer_cfg_str)

            # Get answers
            answer_d: dict = self._qwen_pred_answer(input_d=input_prompt,
                                               infer_cfg=infer_cfg,
                                               subset_name=subset_name,
                                               answer_id=answer_id)

            answer_d[AnswerKeys.MODEL_SPEC] = self.qwen_model_adapter.model_cfg
            answer_d[AnswerKeys.RAW_INPUT] = input_prompt[AnswerKeys.RAW_INPUT]
            answer_d[AnswerKeys.ORIGIN_PROMPT] = input_prompt

            if debug:
                logger.debug(f'**input_prompt: {json.dumps(input_prompt, ensure_ascii=False)} \n')
                logger.debug(f'**predicted ans: {json.dumps(answer_d, ensure_ascii=False)} \n')

            answers_list.append(answer_d)

        # Dump answers
        pred_dir: str = self.qwen_outputs_structure.get(OutputsStructure.PREDICTIONS_DIR)
        pred_file_name: str = self.dataset_name_or_path.replace('/', '_') + '_' + subset_name + '.jsonl'
        os.makedirs(pred_dir, exist_ok=True)
        dump_jsonl_data(answers_list, os.path.join(pred_dir, pred_file_name))

        return answers_list

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

    def infer(self,
             infer_cfg: dict = None,
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

        logger.info(f'**** Start inferencing on dataset {self.dataset_name_or_path} ****')

        for subset_name, prompts_list in self.prompts.items():
            limit = infer_cfg.get('limit', len(prompts_list))
            prompts_list = prompts_list[:limit]

            answers_list: list = self.get_answers(subset_name=subset_name,
                                                  prompts_list=prompts_list,
                                                  infer_cfg=infer_cfg,
                                                  debug=debug,
                                                  **kwargs)

            if self.qwen_model_adapter:
                qwen_answers_list: list = self.get_qwen_answers(subset_name=subset_name,
                                                                prompts_list=prompts_list,
                                                                infer_cfg=infer_cfg,
                                                                debug=debug,
                                                                **kwargs)

        self.save_cache()
        self.clear_cache()

        logger.info(f'\n**** Inference finished on {self.dataset_name_or_path} ****\n')