# Copyright (c) Alibaba, Inc. and its affiliates.
import sys
from typing import Union

from evals import Eval
from evals.constants import EvalTaskConfig, DEFAULT_CACHE_DIR, TaskEnvs
from evals.predictors import Predictor
from evals.samples import GenerateSamples
from evals.utils.utils import yaml_to_dict, get_obj_from_cfg, jsonl_to_list
from evals.utils.logger import get_logger

logger = get_logger()


class EvalTask(object):

    def __init__(self, prompts: Union[list, str], task_cfg: Union[dict, str]):
        # TODO: task_name to be added in registry
        # TODO: task calling in CLI to be added
        # TODO: multi-threads to be added

        self.task_id = None
        self.task_name = None
        self.scoring_model_obj = None
        self.predictor_obj = None

        self.cache_root_dir = os.environ.get(TaskEnvs.CACHE_DIR, DEFAULT_CACHE_DIR)
        os.makedirs(self.cache_root_dir, exist_ok=True)

        if isinstance(prompts, str):
            prompts = jsonl_to_list(prompts)
        self.prompts = prompts

        if isinstance(task_cfg, str):
            task_cfg = yaml_to_dict(task_cfg)
        logger.info(f'task_cfg={task_cfg}')

        self.task_name = task_cfg.get(EvalTaskConfig.TASK_NAME, None)
        if not self.task_name:
            raise ValueError('task_name must be provided in task config.')

        self.task_spec = task_cfg.get(self.task_name)
        if not self.task_spec:
            raise ValueError(f'The task specification must be provided in task config for task_name={self.task_name}.')

        self.task_id = self.task_spec.get(EvalTaskConfig.TASK_ID, None)

        # todo: prompts jsonl

        # Get predictor class and args, and create predictor object
        predictor_cfg = self.task_spec.get(EvalTaskConfig.PREDICTOR, {})
        predictor_class, predictor_args = self._parse_obj_cfg(predictor_cfg)
        predictor_args['api_key'] = ''
        self.predictor_obj = predictor_class(**predictor_args)
        if not isinstance(self.predictor_obj, Predictor):
            raise TypeError('predictor_obj must be an instance of evals.predictors.Predictor')

        # Get eval class(scoring model) and args, and create eval object
        scoring_model_cfg = self.task_spec.get(EvalTaskConfig.SCORING_MODEL, {})
        scoring_model_class, scoring_model_args = self._parse_obj_cfg(scoring_model_cfg)
        self.scoring_model_obj = scoring_model_class(**scoring_model_args)
        if not isinstance(self.scoring_model_obj, Eval):
            raise TypeError('eval_obj must be an instance of evals.Eval')

    def _parse_obj_cfg(self, obj_cfg: dict):
        cls_ref = obj_cfg.get(EvalTaskConfig.CLASS_REF, None)
        if not cls_ref:
            raise ValueError(f'class reference must be provided in task config for task_name={self.task_name}.')

        cls_args = obj_cfg.get(EvalTaskConfig.CLASS_ARGS, {})
        cls = get_obj_from_cfg(cls_ref)

        return cls, cls_args

    def run(self):

        # Note: run的流程从prompts开始, 结束于scoring model的输出结果

        # TODO: task 流程编排
        #   1. 获取raw samples (examples里，user自行操作) -- 输出到cache dir (默认是 /tmp/maas_evals/data/raw_samples)
        #   2. formatting samples (examples里，user自行操作) -- 输出到cache dir (默认是 /tmp/maas_evals/data/formatted_samples)
        #   3. 获取prompts (examples里，user自行操作， sdk指定template)  -- 输出到cache dir (默认是 /tmp/maas_evals/data/prompts)
        #   4. 配置task的 yaml文件 -- 写到cache dir  (默认是 /tmp/maas_evals/tasks/config/task_name_dev_v0.yaml)
        #   5. eval_task = EvalTask(task_cfg=task_cfg)
        #   6. eval_task.run()  -- 输入是prompts jsonl文件， 输出是一一对应的jsonl结果文件
        #   7. [可选] 上传到iTag
        #   8. [可选] 下载iTag数据，并解析结果
        #   9. [可选] 生成报告


        # 1. get prompts
        print(self.prompts)

        sys.exit(0)


        # todo: get batches (or add_batches) --P1

        # run inference
        # todo: tqdm进度条
        results_list = []
        for prompt_dict in self.prompts:
            result_dict = self.run_inference(**prompt_dict)
            results_list.append(result_dict)
        # todo: dump predicted samples
        print(results_list)


        # TODO: run eval



        # TODO: dump result

        ...

    # def get_samples(self):
    #     # TODO: raw_samples -> formatted_samples -> prompts
    #     ...

    def get_model_meta(self):
        ...

    def get_scoring_model(self):
        # TODO: According to task config(yaml) to get scoring model object from registry.
        ...

    def get_batches(self):
        ...

    def run_inference(self, **input_args) -> dict:
        result_dict = self.predictor_obj(**input_args)
        return result_dict

    def gen_report(self):
        ...

    def dump_result(self):
        ...

    # def _get_formatted_samples(self):
    #     ...

    def _get_prompts(self):
        # TODO: search cache dir (default is /tmp/maas_evals/data/prompts) for prompts
        ...

    def get_model_info(self):
        """
        Get model info from predictor.
        :return:
        """
        ...


if __name__ == '__main__':
    import os
    task_cfg_file = '/evals/registry/tasks/task_moss_gen_poetry.yaml'
    eval_task = EvalTask(task_cfg_file)

    eval_task.run()
