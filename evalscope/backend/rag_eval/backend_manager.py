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
        self._check_env()
        super().__init__(config, **kwargs)

    @staticmethod
    def _check_env():
        if is_module_installed("mteb"):
            logger.info("Check `mteb` Installed")
        else:
            logger.error("Please install `mteb` and `ragas` first")

    def run_mteb(self):
        from evalscope.backend.rag_eval.cmteb import ModelArguments, EvalArguments
        from evalscope.backend.rag_eval.cmteb import one_stage_eval, two_stage_eval
        if len(self.model_args) == 1:
            model_args = ModelArguments(**self.model_args[0]).to_dict()
            eval_args = EvalArguments(**self.eval_args).to_dict()
            one_stage_eval(model_args, eval_args)
        elif len(self.model_args) == 2:
            model1_args = ModelArguments(**self.model_args[0]).to_dict()
            model2_args = ModelArguments(**self.model_args[1]).to_dict()
            eval_args = EvalArguments(**self.eval_args).to_dict()
            two_stage_eval(model1_args, model2_args, eval_args)
        else:
            raise ValueError("Not support multiple models yet")
        

    def run(self, *args, **kwargs):
        tool = self.config_d.pop("tool")
        if tool.upper() == "MTEB":
            self.model_args =self.config_d['model']
            self.eval_args = self.config_d['eval']
            self.run_mteb()
