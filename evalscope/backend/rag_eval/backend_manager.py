import os
from typing import Optional, Union

from evalscope.backend.base import BackendManager
from evalscope.utils import get_valid_list, is_module_installed
from evalscope.utils.logger import get_logger

logger = get_logger()


class Tools:
    MTEB = 'mteb'
    RAGAS = 'ragas'
    CLIP_BENCHMARK = 'clip_benchmark'


class RAGEvalBackendManager(BackendManager):

    def __init__(self, config: Union[str, dict], **kwargs):
        """BackendManager for VLM Evaluation Kit

        Args:
            config (Union[str, dict]): the configuration yaml-file or the configuration dictionary
        """
        super().__init__(config, **kwargs)

    @staticmethod
    def _check_env(module_name: str):
        if is_module_installed(module_name):
            logger.info(f'Check `{module_name}` Installed')
        else:
            logger.error(f'Please install `{module_name}` first')

    @staticmethod
    def run_mteb(model_args, eval_args):
        from evalscope.backend.rag_eval.cmteb import EvalArguments, ModelArguments, one_stage_eval, two_stage_eval

        if len(model_args) > 2:
            raise ValueError('Not support multiple models yet')

        # Convert arguments to dictionary
        model_args_list = [ModelArguments(**args).to_dict() for args in model_args]
        eval_args = EvalArguments(**eval_args).to_dict()

        if len(model_args_list) == 1:
            one_stage_eval(model_args_list[0], eval_args)
        else:  # len(model_args_list) == 2
            two_stage_eval(model_args_list[0], model_args_list[1], eval_args)

    @staticmethod
    def run_ragas(testset_args, eval_args):
        from evalscope.backend.rag_eval.ragas import EvaluationArguments, TestsetGenerationArguments, rag_eval
        from evalscope.backend.rag_eval.ragas.tasks import generate_testset

        if testset_args is not None:
            if isinstance(testset_args, dict):
                generate_testset(TestsetGenerationArguments(**testset_args))
            elif isinstance(testset_args, TestsetGenerationArguments):
                generate_testset(testset_args)
            else:
                raise ValueError('Please provide the testset generation arguments.')
        if eval_args is not None:
            if isinstance(eval_args, dict):
                rag_eval(EvaluationArguments(**eval_args))
            elif isinstance(eval_args, EvaluationArguments):
                rag_eval(eval_args)
            else:
                raise ValueError('Please provide the evaluation arguments.')

    @staticmethod
    def run_clip_benchmark(args):
        from evalscope.backend.rag_eval.clip_benchmark import Arguments, evaluate

        evaluate(Arguments(**args))

    def run(self, *args, **kwargs):
        tool = self.config_d.pop('tool')
        if tool.lower() == Tools.MTEB:
            self._check_env('mteb')
            model_args = self.config_d['model']
            eval_args = self.config_d['eval']
            self.run_mteb(model_args, eval_args)
        elif tool.lower() == Tools.RAGAS:
            self._check_env('ragas')
            testset_args = self.config_d.get('testset_generation', None)
            eval_args = self.config_d.get('eval', None)
            self.run_ragas(testset_args, eval_args)
        elif tool.lower() == Tools.CLIP_BENCHMARK:
            self._check_env('webdataset')
            self.run_clip_benchmark(self.config_d['eval'])
        else:
            raise ValueError(f'Unknown tool: {tool}')
