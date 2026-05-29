# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Union

from evalscope.backend.base import BackendManager
from evalscope.utils.import_utils import is_module_installed
from evalscope.utils.logger import get_logger

logger = get_logger()


class Tools:
    MTEB = 'mteb'
    RAGAS = 'ragas'
    CLIP_BENCHMARK = 'clip_benchmark'


class RAGEvalBackendManager(BackendManager):

    def __init__(self, config: Union[str, dict], **kwargs):
        """BackendManager for RAG Evaluation.

        Args:
            config: Configuration as yaml file path, dict, or Pydantic model.
        """
        super().__init__(config, **kwargs)

    @staticmethod
    def _check_env(module_name: str) -> None:
        if is_module_installed(module_name):
            logger.info(f'Check `{module_name}` Installed')
        else:
            raise RuntimeError(f'`{module_name}` is not installed. Please install it with: pip install {module_name}')

    @staticmethod
    def run_mteb(config) -> None:
        """Run MTEB evaluation.

        Args:
            config: MTEBToolConfig instance or dict with MTEB configuration.
        """
        from evalscope.backend.rag_eval.mteb import MTEBToolConfig, run_mteb_eval

        if isinstance(config, dict):
            config = MTEBToolConfig(**config)
        run_mteb_eval(config)

    @staticmethod
    def run_ragas(config) -> None:
        """Run RAGAS evaluation and/or testset generation.

        Args:
            config: RAGASToolConfig instance or dict with RAGAS configuration.
        """
        from evalscope.backend.rag_eval.ragas import RAGASToolConfig, rag_eval
        from evalscope.backend.rag_eval.ragas.tasks import generate_testset

        if isinstance(config, dict):
            config = RAGASToolConfig(**config)

        if config.testset_generation is not None:
            generate_testset(config.testset_generation)
        if config.eval is not None:
            rag_eval(config.eval)

    @staticmethod
    def run_clip_benchmark(config) -> None:
        """Run CLIP Benchmark evaluation.

        Args:
            config: ClipBenchmarkToolConfig instance or dict.
        """
        from evalscope.backend.rag_eval.clip_benchmark import Arguments, evaluate

        if isinstance(config, dict):
            eval_args = config.get('eval', config)
        else:
            eval_args = config.eval
        evaluate(Arguments(**eval_args))

    def run(self, *args, **kwargs) -> None:
        """Run the RAG evaluation pipeline based on tool type.

        Supports both new Pydantic config objects and old dict format.
        """
        from evalscope.backend.rag_eval.mteb.arguments import MTEBToolConfig
        from evalscope.backend.rag_eval.ragas.arguments import ClipBenchmarkToolConfig, RAGASToolConfig

        config = self.config_d

        # If config is already a typed Pydantic object, route directly
        if isinstance(config, MTEBToolConfig):
            self._check_env('mteb')
            self.run_mteb(config)
        elif isinstance(config, RAGASToolConfig):
            self._check_env('ragas')
            self.run_ragas(config)
        elif isinstance(config, ClipBenchmarkToolConfig):
            self._check_env('webdataset')
            self.run_clip_benchmark(config)
        elif isinstance(config, dict):
            # Legacy dict format — route by 'tool' key
            tool = config.pop('tool', '')
            if tool.lower() == Tools.MTEB:
                self._check_env('mteb')
                # Convert legacy dict format to the new MTEBToolConfig schema
                mteb_config = self._convert_legacy_mteb_config(config)
                self.run_mteb(mteb_config)
            elif tool.lower() == Tools.RAGAS:
                self._check_env('ragas')
                self.run_ragas(config)
            elif tool.lower() == Tools.CLIP_BENCHMARK:
                self._check_env('webdataset')
                self.run_clip_benchmark(config)
            else:
                raise ValueError(f'Unknown tool: {tool}')
        else:
            raise ValueError(f'Unsupported config type: {type(config)}')

    @staticmethod
    def _convert_legacy_mteb_config(config: dict) -> dict:
        """Convert old MTEB config format to new format.

        Old format:
            {'model': [{'model_name_or_path': '...', ...}], 'eval': {'tasks': [...], ...}}
        New format:
            {'tool': 'mteb', 'models': [...], 'eval': {'task_names': [...], ...}}
        """
        result = {'tool': 'mteb'}

        # Convert 'model' key to 'models'
        if 'model' in config:
            result['models'] = config['model']
        elif 'models' in config:
            result['models'] = config['models']

        # Convert eval args
        eval_config = dict(config.get('eval', {}) or {})
        if 'tasks' in eval_config and 'task_names' not in eval_config:
            eval_config['task_names'] = eval_config.pop('tasks')
        result['eval'] = eval_config

        return result
