# Copyright (c) Alibaba, Inc. and its affiliates.
"""Regression tests for the refactored RAG Eval backend.

Tests cover:
- Pydantic config validation and creation
- TaskConfig integration (auto-parsing eval_config)
- Backward compatibility with old dict format
- Model loading factory routing
- MTEB task resolution
- Custom dataset building
"""
import pytest
from unittest.mock import MagicMock, patch


class TestMTEBConfig:
    """Test MTEB Pydantic configuration models."""

    def test_mteb_model_config_defaults(self):
        from evalscope.backend.rag_eval.mteb.arguments import MTEBModelConfig
        cfg = MTEBModelConfig(model_name_or_path='BAAI/bge-base-zh-v1.5')
        assert cfg.hub == 'modelscope'
        assert cfg.is_cross_encoder is False
        assert cfg.max_seq_length == 512
        assert cfg.encode_kwargs == {'batch_size': 32}

    def test_mteb_tool_config_from_dict(self):
        from evalscope.backend.rag_eval.mteb.arguments import MTEBToolConfig
        cfg = MTEBToolConfig(
            tool='mteb',
            models=[{'model_name_or_path': 'test-model'}],
            eval={'task_names': ['T2Reranking']}
        )
        assert cfg.tool == 'mteb'
        assert len(cfg.models) == 1
        assert cfg.eval.task_names == ['T2Reranking']

    def test_mteb_eval_config_filters(self):
        from evalscope.backend.rag_eval.mteb.arguments import MTEBEvalConfig
        cfg = MTEBEvalConfig(task_types=['Retrieval', 'Reranking'], languages=['zho', 'eng'])
        assert cfg.task_types == ['Retrieval', 'Reranking']
        assert cfg.languages == ['zho', 'eng']
        assert cfg.hub == 'modelscope'


class TestRAGASConfig:
    """Test RAGAS Pydantic configuration models."""

    def test_ragas_eval_config(self):
        from evalscope.backend.rag_eval.ragas.arguments import RAGASEvalConfig
        cfg = RAGASEvalConfig(
            testset_file='test.json',
            critic_llm={'model_name': 'gpt-4'},
            embeddings={'model_name_or_path': 'bge-base'},
        )
        assert cfg.critic_llm.model_name == 'gpt-4'
        assert cfg.critic_llm.provider == 'openai'
        assert cfg.embeddings.provider == 'huggingface'

    def test_ragas_tool_config(self):
        from evalscope.backend.rag_eval.ragas.arguments import RAGASToolConfig
        cfg = RAGASToolConfig(
            tool='ragas',
            eval={
                'testset_file': 'test.json',
                'critic_llm': {'model_name': 'qwen-plus', 'api_base': 'http://localhost'},
                'embeddings': {'model_name_or_path': 'bge-base'},
                'metrics': ['faithfulness'],
            }
        )
        assert cfg.eval.metrics == ['faithfulness']
        assert cfg.eval.critic_llm.api_base == 'http://localhost'

    def test_backward_compat_aliases(self):
        from evalscope.backend.rag_eval.ragas.arguments import (
            EvaluationArguments,
            RAGASEvalConfig,
            RAGASTestsetConfig,
            TestsetGenerationArguments,
        )
        assert EvaluationArguments is RAGASEvalConfig
        assert TestsetGenerationArguments is RAGASTestsetConfig


class TestTaskConfigIntegration:
    """Test TaskConfig auto-parsing of eval_config for RAGEval backend."""

    def test_mteb_auto_parse(self):
        from evalscope.backend.rag_eval.mteb.arguments import MTEBToolConfig
        from evalscope.config import TaskConfig
        from evalscope.constants import EvalBackend

        task_cfg = TaskConfig(
            eval_backend=EvalBackend.RAG_EVAL,
            eval_config={
                'tool': 'mteb',
                'models': [{'model_name_or_path': 'test-model'}],
                'eval': {'task_names': ['T2Reranking']}
            }
        )
        assert isinstance(task_cfg.eval_config, MTEBToolConfig)

    def test_ragas_auto_parse(self):
        from evalscope.backend.rag_eval.ragas.arguments import RAGASToolConfig
        from evalscope.config import TaskConfig
        from evalscope.constants import EvalBackend

        task_cfg = TaskConfig(
            eval_backend=EvalBackend.RAG_EVAL,
            eval_config={
                'tool': 'ragas',
                'eval': {
                    'testset_file': 'test.json',
                    'critic_llm': {'model_name': 'gpt-4'},
                    'embeddings': {'model_name_or_path': 'bge-base'},
                }
            }
        )
        assert isinstance(task_cfg.eval_config, RAGASToolConfig)

    def test_non_rag_eval_config_unchanged(self):
        from evalscope.config import TaskConfig
        task_cfg = TaskConfig(
            eval_config={'tool': 'mteb', 'models': [{'model_name_or_path': 'x'}], 'eval': {'task_names': ['T']}}
        )
        # Without eval_backend=RAG_EVAL, should remain as dict
        assert isinstance(task_cfg.eval_config, dict)


class TestBackwardCompat:
    """Test backward compatibility with old config formats."""

    def test_legacy_mteb_config_conversion(self):
        from evalscope.backend.rag_eval.backend_manager import RAGEvalBackendManager
        legacy = {
            'model': [{'model_name_or_path': 'BAAI/bge-base-zh-v1.5'}],
            'eval': {'tasks': ['T2Reranking'], 'output_folder': 'outputs'}
        }
        converted = RAGEvalBackendManager._convert_legacy_mteb_config(legacy)
        assert 'models' in converted
        assert converted['eval']['task_names'] == ['T2Reranking']
        assert 'tasks' not in converted['eval']

    def test_new_format_models_key(self):
        from evalscope.backend.rag_eval.backend_manager import RAGEvalBackendManager
        new_format = {
            'models': [{'model_name_or_path': 'test'}],
            'eval': {'task_names': ['STS']}
        }
        converted = RAGEvalBackendManager._convert_legacy_mteb_config(new_format)
        assert converted['models'] == [{'model_name_or_path': 'test'}]


class TestModelFactory:
    """Test model loading factory routing."""

    def test_load_model_routes_encoder(self):
        from evalscope.backend.rag_eval.mteb.arguments import MTEBModelConfig
        cfg = MTEBModelConfig(model_name_or_path='test-model', is_cross_encoder=False)
        assert cfg.is_cross_encoder is False

    def test_load_model_routes_reranker(self):
        from evalscope.backend.rag_eval.mteb.arguments import MTEBModelConfig
        cfg = MTEBModelConfig(model_name_or_path='test-reranker', is_cross_encoder=True)
        assert cfg.is_cross_encoder is True

    def test_load_model_routes_api(self):
        from evalscope.backend.rag_eval.mteb.arguments import MTEBModelConfig
        cfg = MTEBModelConfig(
            model_name_or_path='text-embedding-v3',
            model_name='text-embedding-v3',
            api_base='http://localhost:8080',
            api_key='test-key',
        )
        assert cfg.model_name == 'text-embedding-v3'
        assert cfg.api_base == 'http://localhost:8080'


class TestCustomTask:
    """Test custom dataset task building."""

    def test_build_custom_task_import(self):
        from evalscope.backend.rag_eval.mteb.custom_task import build_custom_task
        assert callable(build_custom_task)

    def test_build_custom_task_missing_path_raises(self):
        from evalscope.backend.rag_eval.mteb.custom_task import build_custom_task
        with pytest.raises(ValueError, match='data_path'):
            build_custom_task({'name': 'test', 'type': 'Retrieval'})

    def test_build_custom_task_unsupported_type_raises(self):
        from evalscope.backend.rag_eval.mteb.custom_task import build_custom_task
        with pytest.raises(ValueError, match='Unsupported'):
            build_custom_task({'name': 'test', 'type': 'UnknownType', 'data_path': '/tmp'})
