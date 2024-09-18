import os
from typing import Optional, Union
from evalscope.utils import is_module_installed, get_valid_list
from evalscope.backend.base import BackendManager
from evalscope.utils.logger import get_logger


logger = get_logger()


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
            logger.info(f"Check `{module_name}` Installed")
        else:
            logger.error(f"Please install `{module_name}` first")

    def run_mteb(self):
        from evalscope.backend.rag_eval.cmteb import ModelArguments, EvalArguments
        from evalscope.backend.rag_eval.cmteb import one_stage_eval, two_stage_eval

        if len(self.model_args) > 2:
            raise ValueError("Not support multiple models yet")

        # Convert arguments to dictionary
        model_args_list = [ModelArguments(**args).to_dict() for args in self.model_args]
        eval_args = EvalArguments(**self.eval_args).to_dict()

        if len(model_args_list) == 1:
            one_stage_eval(model_args_list[0], eval_args)
        else:  # len(model_args_list) == 2
            two_stage_eval(model_args_list[0], model_args_list[1], eval_args)

    def run_ragas(self):
        from evalscope.backend.rag_eval.ragas import rag_eval, testset_generation
        from evalscope.backend.rag_eval.ragas import (
            TestsetGenerationArguments,
            EvaluationArguments,
        )

        if self.testset_args is not None:
            testset_generation(TestsetGenerationArguments(**self.testset_args))
        if self.eval_args is not None:
            rag_eval(EvaluationArguments(**self.eval_args))

    def run(self, *args, **kwargs):
        tool = self.config_d.pop("tool")
        if tool.lower() == "mteb":
            self._check_env("mteb")
            self.model_args = self.config_d["model"]
            self.eval_args = self.config_d["eval"]
            self.run_mteb()
        elif tool.lower() == "ragas":
            self._check_env("ragas")
            self.testset_args = self.config_d.get("testset_generation", None)
            self.eval_args = self.config_d.get("eval", None)
            self.run_ragas()
        else:
            raise ValueError(f"Unknown tool: {tool}")
