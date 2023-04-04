# Copyright (c) Alibaba, Inc. and its affiliates.
import sys
from typing import Union

from evals import Eval
from evals.constants import EvalTaskConfig
from evals.predictors import Predictor
from evals.samples import GenerateSamples
from evals.tools import ItagManager
from evals.utils.maxcompute_util import MaxComputeUtil
from evals.utils.utils import yaml_reader, get_obj_from_cfg
from evals.utils.logger import get_logger

logger = get_logger()


class EvalTask(object):

    def __init__(self, task_cfg: Union[dict, str]):
        # TODO: task_name to be added in registry
        # TODO: task calling in CLI to be added
        # TODO: multi-threads to be added

        self.task_id = None
        self.task_name = None
        self.generate_samples_obj = None
        self.scoring_model_obj = None
        self.predictor_obj = None

        # predictor: Predictor, eval_cls: Eval, **kwargs

        if isinstance(task_cfg, str):
            task_cfg = yaml_reader(task_cfg)
        logger.info(f'task_cfg={task_cfg}')

        self.task_name = task_cfg.get(EvalTaskConfig.TASK_NAME, None)
        if not self.task_name:
            raise ValueError('task_name must be provided in task config.')

        self.task_spec = task_cfg.get(self.task_name)
        if not self.task_spec:
            raise ValueError(f'The task specification must be provided in task config for task_name={self.task_name}.')

        self.task_id = self.task_spec.get(EvalTaskConfig.TASK_ID, None)

        samples_cfg = self.task_spec.get(EvalTaskConfig.SAMPLES, {})
        generate_samples_class, generate_samples_args = self._parse_obj_cfg(samples_cfg)
        self.generate_samples_obj = generate_samples_class(**generate_samples_args)
        if not isinstance(self.generate_samples_obj, GenerateSamples):
            raise TypeError('generate_samples_obj must be an instance of evals.samples.GenerateSamples')

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

        # 1. get samples: task cfg (yaml) -> prompt_file_path(jsonl)
        prompts_list = self.generate_samples_obj.run()
        print(prompts_list)

        # todo: get batches (or add_batches) --P1

        # run inference
        results_list = []
        for prompt_dict in prompts_list:
            result_dict = self.run_inference(**prompt_dict)
            results_list.append(result_dict)
        print(results_list)

        # TODO: get scoring model and run eval

        sys.exit(0)

        # upload to iTag
        itag_manager = ItagManager(tenant_id='', token='', employee_id='')
        itag_manager.process(dataset_path='',
                             dataset_name='',
                             template_id='',
                             task_name='')

        # TODO: download iTag data -- 人工触发 :  实时接口（获取标注结果，api？）
        maxcomput_util = MaxComputeUtil(access_id='', access_key='', project_name='', endpoint='')
        maxcomput_util.fetch_data(table_name='', pt_condition='', output_path='')


        # TODO: gen report


        # TODO: dump result

        ...

    def get_samples(self):
        # TODO: raw_samples -> formatted_samples -> prompts
        ...

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

    def _get_formatted_samples(self):
        ...

    def _get_prompts(self):
        ...

    def get_model_info(self):
        """
        Get model info from predictor.
        :return:
        """
        ...


if __name__ == '__main__':
    import os
    task_cfg_file = '/Users/jason/workspace/work/maas/llm-eval/evals/registry/tasks/task_moss_gen_poetry.yaml'
    eval_task = EvalTask(task_cfg_file)

    eval_task.run()
