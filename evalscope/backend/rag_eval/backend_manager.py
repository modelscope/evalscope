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
        import mteb
        from evalscope.backend.rag_eval import EmbeddingModel
        from evalscope.backend.rag_eval import cmteb
        from mteb.task_selection import results_to_dataframe

        # load task first to update instructions
        tasks = cmteb.TaskBase.get_tasks(
            task_names=self.args.tasks, instructions=self.args.instructions
        )

        evaluation = mteb.MTEB(tasks=tasks)

        model = EmbeddingModel.from_pretrained(
            model_name_or_path=self.args.model_name_or_path,
            is_cross_encoder=self.args.is_cross_encoder,
            pooling_mode=self.args.pooling_mode,
            max_seq_length=self.args.max_seq_length,
            model_kwargs=self.args.model_kwargs,
            config_kwargs=self.args.config_kwargs,
            prompts=cmteb.INSTRUCTIONS,
            hub=self.args.hub,
        )

        results = evaluation.run(
            model,
            verbosity=self.args.verbosity,
            output_folder=self.args.output_folder,
            overwrite_results=self.args.overwrite_results,
            encode_kwargs=self.args.encode_kwargs,
            limits=self.args.limits,
        )

        model_name = model.mteb_model_meta.model_name_as_path()
        revision = model.mteb_model_meta.revision

        results_df = results_to_dataframe({model_name: {revision: results}})

        save_path = os.path.join(
            self.args.output_folder,
            model_name,
            revision,
        )
        logger.info(f"Evaluation results:\n{results_df.to_markdown()}")
        logger.info(f"Evaluation results saved in {os.path.abspath(save_path)}")

    def run(self, *args, **kwargs):
        tool = self.config_d.pop("tool")
        if tool.upper() == "MTEB":
            from evalscope.backend.rag_eval.cmteb import MTEBArguments

            self.args = MTEBArguments(**self.config_d)
            self.run_mteb()
