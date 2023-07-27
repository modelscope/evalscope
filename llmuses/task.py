# Copyright (c) Alibaba, Inc. and its affiliates.

import os
# from concurrent.futures import ProcessPoolExecutor
# from multiprocessing import Pool as ThreadPool
from typing import List, Union

from llmuses.constants import (DEFAULT_WORK_DIR, DumpMode, EvalTaskConfig,
                               TaskEnvs)
from llmuses.evaluate import Evaluate
from llmuses.predictors import Predictor
from llmuses.utils.logger import get_logger
from llmuses.utils.utils import (get_obj_from_cfg, jsonl_dump_data,
                                 jsonl_to_list, yaml_to_dict)

# from tqdm import tqdm

logger = get_logger()


class EvalTask(object):
    """
    The task class for evaluation.

    Args:
        prompts (Union[List[dict], str]):
            The list of prompts or the path of the file containing prompts.
        task_cfg (Union[dict, str]):
            The task config or the path of the file containing task config.

    Attributes:
        task_id (str): The task id.
        task_name (str): The task name.
        evaluator_obj (BaseEvalSpec): The scoring model object.
        predictor_obj (Predictor): The predictor object.
        work_dir (str): The work root dir. Default dir:   ~/maas_evals
        predicted_samples_path (str): The jsonl file path of samples
            that predicted by specific predictor.
    """

    def __init__(self, prompts: Union[List[dict], str], task_cfg: Union[dict,
                                                                        str]):
        # TODO: task calling in CLI to be added
        # TODO: output接口： {'input': {'id': 2, 'prompt': 'xxx'}, 'output': {'text': 'xxx', ...}}

        self.task_id: str = ''
        self.task_name: str = ''
        self.evaluator_obj: Evaluate = None
        self.predictor_obj: Predictor = None

        self.work_dir = os.path.expanduser(
            os.environ.get(TaskEnvs.WORK_DIR, DEFAULT_WORK_DIR))
        os.makedirs(self.work_dir, exist_ok=True)

        if isinstance(prompts, str):
            prompts = jsonl_to_list(prompts)
        self.prompts: list = prompts

        if isinstance(task_cfg, str):
            task_cfg = yaml_to_dict(task_cfg)
        logger.info(f'task_cfg={task_cfg}')

        self.task_name = task_cfg.get(EvalTaskConfig.TASK_NAME, None)
        if not self.task_name:
            logger.error('task_name must be provided in task config.')

        self.task_spec = task_cfg.get(self.task_name)
        if not self.task_spec:
            logger.error(f'The task specification must be provided '
                         f'in task config for task_name={self.task_name}.')

        self.task_id = self.task_spec.get(EvalTaskConfig.TASK_ID, None)

        self.task_dir = os.path.join(self.work_dir, 'tasks', self.task_id)
        os.makedirs(self.task_dir, exist_ok=True)
        self.predicted_samples_path = os.path.join(self.task_dir,
                                                   'predicted_samples.jsonl')
        self.eval_results_path = os.path.join(self.task_dir,
                                              'eval_results.jsonl')

        # Get predictor class and args, and create predictor object
        predictor_cfg = self.task_spec.get(EvalTaskConfig.PREDICTOR, {})
        predictor_class, predictor_args = self._parse_obj_cfg(predictor_cfg)
        if predictor_class:
            predictor_args['api_key'] = ''
            self.predictor_obj = predictor_class(**predictor_args)

        if not isinstance(self.predictor_obj, Predictor):
            logger.warning('predictor_obj should be an instance of '
                           'llmuses.predictors.Predictor')

        # Get eval class(scoring model) and args, and create eval object
        evaluator_cfg = self.task_spec.get(EvalTaskConfig.EVALUATOR, {})
        evaluator_class, evaluator_args = self._parse_obj_cfg(evaluator_cfg)
        self.evaluator_obj = evaluator_class(**evaluator_args)
        if not isinstance(self.evaluator_obj, Evaluate):
            logger.warning('eval_obj should be an instance of llmuses.Eval')

    def _parse_obj_cfg(self, obj_cfg: dict):
        cls_ref = obj_cfg.get(EvalTaskConfig.CLASS_REF, None)
        if not cls_ref:
            logger.warning(
                f'no class reference in task config task_name={self.task_name}.'
            )
            return cls_ref, None

        cls_args = obj_cfg.get(EvalTaskConfig.CLASS_ARGS, {})
        cls = get_obj_from_cfg(cls_ref)

        return cls, cls_args

    def run(self,
            num_processes: int = 4,
            chunksize: int = 2,
            dump_mode: str = DumpMode.OVERWRITE):

        # TODO: task 流程编排  (Note: run的流程从prompts开始, 结束于scoring model的输出结果)
        #   1. 获取raw samples (examples里，user自行操作) -- 输出到work dir (默认是 ~/maas_evals/data/raw_samples)
        #   2. formatting samples (examples里，user自行操作) -- 输出到work dir (默认是 ~/maas_evals/data/formatted_samples)
        #   3. 获取prompts (examples里，user自行操作， sdk指定template)  -- 输出到work dir (默认是 ~/maas_evals/data/prompts)
        #   4. 配置task的 yaml文件 -- 写到work dir  (默认是 ~/maas_evals/tasks/config/task_name_dev_v0.yaml)
        #   5. eval_task = EvalTask(task_cfg=task_cfg)
        #   6. eval_task.run()  -- 输入是prompts jsonl文件， 输出是一一对应的eval results jsonl结果文件
        #   7. [可选] 上传到iTag
        #   8. [可选] 下载iTag数据，并解析结果
        #   9. [可选] 生成报告

        # TODO: add streaming write ? (dump one by one)

        # Run inference by specific predictor
        results_list = []
        if self.predictor_obj:
            logger.info('Start to run inference ...')
            if not self.prompts:
                raise ValueError('input prompts cannot be empty!')

            # TODO: error when using ProcessPoolExecutor
            # try:
            #     with ProcessPoolExecutor(max_workers=num_processes) as executor:
            #         results_list = list(tqdm(
            #             executor.map(self.run_inference, self.prompts, chunksize=chunksize),
            #             total=math.ceil(len(self.prompts)/chunksize)))
            # except Exception as e:
            #     raise e

            for prompt in self.prompts:
                res = self.run_inference(prompt)
                results_list.append(res)

            invalid_data_num = len(
                [item for item in results_list if item is None])
            if invalid_data_num > 0:
                logger.error(
                    f'Predictor got {invalid_data_num} null result '
                    f'in {self.predicted_samples_path} , error may occur due to multi-processing inference, '
                    f'try to decrease num_processes and chunksize to avoid the limit of predictor service. '
                    f'Alternatively, you can check input prompts which may contain invalid data.'
                )

        # Dump predicted samples
        if len(results_list) > 0:
            jsonl_dump_data(
                results_list, self.predicted_samples_path, dump_mode=dump_mode)
            logger.info(
                f'Dump predicted samples to {self.predicted_samples_path}')
            # Run auto-evaluation by using scoring model
            eval_results = self.evaluator_obj.run(self.predicted_samples_path)
        else:
            eval_results = self.evaluator_obj.run(self.prompts)

        # Dump eval result
        if len(eval_results) > 0:
            jsonl_dump_data(
                eval_results, self.eval_results_path, dump_mode=dump_mode)
            logger.info(f'Dump eval results to {self.eval_results_path}')

    def get_model_meta(self):
        ...

    def get_evaluator(self):
        # TODO: According to task config(yaml) to get scoring model object from registry.
        ...

    def get_batches(self):
        ...

    def run_inference(self, input_dict: dict) -> dict:

        if 'prompt' not in input_dict:
            logger.warning(f'prompt must be provided in input_dict '
                           f'for {self.predictor_obj.__name__}.')

        result_dict = self.predictor_obj(**input_dict)
        return result_dict

    def gen_report(self):
        ...

    def _get_prompts(self):
        # TODO: search work dir (default is ~/maas_evals/data/prompts) for prompts
        ...

    def get_model_info(self):
        """
        Get model info from predictor.
        :return:
        """
        ...
