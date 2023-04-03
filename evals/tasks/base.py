# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Union

from evals import Eval
from evals.constants import EvalTaskConfig
from evals.predictors import Predictor
from evals.utils.utils import yaml_reader, get_obj_from_cfg
from evals.utils.logger import get_logger

logger = get_logger(__name__)


class EvalTask(object):

    def __init__(self, task_cfg: Union[dict, str]):
        # TODO: task_name to be added in registry
        # TODO: task calling in CLI to be added
        # TODO: multi-threads to be added

        # predictor: Predictor, eval_cls: Eval, **kwargs

        if isinstance(task_cfg, str):
            task_cfg = yaml_reader(task_cfg)
        logger.info(f'task_cfg={task_cfg}')

        self.task_name = task_cfg.get(EvalTaskConfig.TASK_NAME, None)
        if not self.task_name:
            raise ValueError('task_name must be provided in task config.')

        self.task_spec = task_cfg.get(self.task_name)

        if self.task_spec:
            self.task_id = self.task_spec.get(EvalTaskConfig.TASK_ID, None)

            eval_class_cfg = self.task_spec.get(EvalTaskConfig.EVAL_CLASS, {})
            eval_class_ref = eval_class_cfg.get(EvalTaskConfig.CLASS_REF, None)
            if not eval_class_ref:
                raise ValueError(f'class.ref must be provided in task config for task_name={self.task_name}.')
            eval_class_args = eval_class_cfg.get(EvalTaskConfig.CLASS_ARGS, {})
            eval_class = get_obj_from_cfg(eval_class_ref)
            self.eval_obj = eval_class(**eval_class_args)

            predictor_cfg = self.task_spec.get(EvalTaskConfig.PREDICTOR, {})
            predictor_ref = predictor_cfg.get(EvalTaskConfig.CLASS_REF, None)
            if not predictor_ref:
                raise ValueError(f'predictor.ref must be provided in task config for task_name={self.task_name}.')
            predictor_args = predictor_cfg.get(EvalTaskConfig.CLASS_ARGS, {})
            predictor_args['api_key'] = ''
            predictor_class = get_obj_from_cfg(predictor_ref)
            self.predictor_obj = predictor_class(**predictor_args)

            if not isinstance(self.eval_obj, Eval):
                raise TypeError('eval_obj must be an instance of evals.Eval')

            if not isinstance(self.predictor_obj, Predictor):
                raise TypeError('predictor_obj must be an instance of evals.predictors.Predictor')

        else:
            logger.warning(f'The task specification must be provided in task config for task_name={self.task_name}.')
            self.task_id = None
            self.eval_obj = None
            self.predictor_obj = None

    def run(self):

        # 1. get samples

        # 2. get model meta info

        # 3. get batches (or add_batches) --P1

        # 4. run inference
        # result_dict = self.run_inference(**input_args)

        # 5. get scoring model and run eval

        # 6. gen report

        # 7. dump result

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
    print(os.getcwd())
    task_cfg_file = '/Users/jason/workspace/work/maas/llm-eval/evals/registry/tasks/task_moss_gen_poetry.yaml'
    eval_task = EvalTask(task_cfg_file)
    # eval_task.run()
