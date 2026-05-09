# Copyright (c) Alibaba, Inc. and its affiliates.
"""Embedding and rerank performance benchmark tests.

Covers random embedding, dataset-based embedding, batch embedding (both
random and dataset), random rerank, and dataset-based rerank.  All tests
require DASHSCOPE_API_KEY.
"""
import unittest

from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark
from tests.perf.perf_test_base import DASHSCOPE_EMBEDDINGS_URL, DASHSCOPE_RERANKS_URL, PerfTestBase


class TestPerfEmbeddingRerank(PerfTestBase):
    """Embedding and rerank API performance benchmarks."""

    # ------------------------------------------------------------------
    # Embedding tests
    # ------------------------------------------------------------------

    def test_embedding_random(self):
        """Random embedding dataset sweep.

        Generates random 256-token prompts and sends them to the DashScope
        embeddings API.  Sweeps (parallel=1, number=2) and (parallel=2,
        number=4).  Requires DASHSCOPE_API_KEY.
        """
        self.skip_without_api_key()

        task_cfg = Arguments(
            parallel=[1, 2],
            number=[2, 4],
            model='text-embedding-v4',
            url=DASHSCOPE_EMBEDDINGS_URL,
            api_key=self.api_key,
            api='openai_embedding',
            dataset='random_embedding',
            min_prompt_length=256,
            max_prompt_length=256,
            tokenizer_path='Qwen/Qwen3-Embedding-0.6B',
        )
        result = run_perf_benchmark(task_cfg)

    def test_embedding_from_dataset(self):
        """Embedding from a custom queries dataset.

        Loads queries from ``custom_eval/text/retrieval/queries.jsonl`` and
        sends them to the DashScope embeddings API.  Sweeps (parallel=1,
        number=2) and (parallel=2, number=4).  Requires DASHSCOPE_API_KEY.
        """
        self.skip_without_api_key()

        task_cfg = Arguments(
            parallel=[1, 2],
            number=[2, 4],
            model='text-embedding-v4',
            url=DASHSCOPE_EMBEDDINGS_URL,
            api_key=self.api_key,
            api='openai_embedding',
            dataset='embedding',
            tokenizer_path='Qwen/Qwen3-Embedding-0.6B',
            dataset_path='custom_eval/text/retrieval/queries.jsonl',
        )
        result = run_perf_benchmark(task_cfg)

    def test_embedding_random_batch(self):
        """Random batch embedding sweep.

        Like ``test_embedding_random`` but uses ``random_embedding_batch``
        with ``batch_size=8`` to send multiple inputs per request.
        Sweeps (parallel=1, number=2) and (parallel=2, number=4).
        Requires DASHSCOPE_API_KEY.
        """
        self.skip_without_api_key()

        task_cfg = Arguments(
            parallel=[1, 2],
            number=[2, 4],
            model='text-embedding-v4',
            url=DASHSCOPE_EMBEDDINGS_URL,
            api_key=self.api_key,
            api='openai_embedding',
            dataset='random_embedding_batch',
            min_prompt_length=256,
            max_prompt_length=256,
            tokenizer_path='Qwen/Qwen3-Embedding-0.6B',
            extra_args={'batch_size': 8},
        )
        result = run_perf_benchmark(task_cfg)

    def test_embedding_batch_from_dataset(self):
        """Batch embedding from a custom queries dataset.

        Loads queries from ``custom_eval/text/retrieval/queries.jsonl`` and
        sends them in batches to the DashScope embeddings API.  Sweeps
        (parallel=1, number=2) and (parallel=2, number=4).
        Requires DASHSCOPE_API_KEY.
        """
        self.skip_without_api_key()

        task_cfg = Arguments(
            parallel=[1, 2],
            number=[2, 4],
            model='text-embedding-v4',
            url=DASHSCOPE_EMBEDDINGS_URL,
            api_key=self.api_key,
            api='openai_embedding',
            dataset='embedding_batch',
            tokenizer_path='Qwen/Qwen3-Embedding-0.6B',
            dataset_path='custom_eval/text/retrieval/queries.jsonl',
        )
        result = run_perf_benchmark(task_cfg)

    # ------------------------------------------------------------------
    # Rerank tests
    # ------------------------------------------------------------------

    def test_rerank_random(self):
        """Random rerank dataset sweep.

        Generates random rerank queries with 5 documents each (document
        length ratio 3x the query).  Sweeps (parallel=1, number=1000) and
        (parallel=2, number=1000).  Requires DASHSCOPE_API_KEY.
        """
        self.skip_without_api_key()

        task_cfg = Arguments(
            parallel=[1, 2],
            number=[1000, 1000],
            model='qwen3-rerank',
            url=DASHSCOPE_RERANKS_URL,
            api_key=self.api_key,
            api='openai_rerank',
            dataset='random_rerank',
            min_prompt_length=256,
            max_prompt_length=256,
            tokenizer_path='Qwen/Qwen3-Embedding-0.6B',
            extra_args={
                'num_documents': 5,
                'document_length_ratio': 3,
            },
        )
        result = run_perf_benchmark(task_cfg)

    def test_rerank_from_dataset(self):
        """Rerank from a custom example dataset.

        Loads query-document pairs from ``custom_eval/text/rerank/example.jsonl``
        and sends them to the DashScope reranks API.  Sweeps (parallel=1,
        number=2) and (parallel=2, number=4).  Requires DASHSCOPE_API_KEY.
        """
        self.skip_without_api_key()

        task_cfg = Arguments(
            parallel=[1, 2],
            number=[2, 4],
            model='qwen3-rerank',
            url=DASHSCOPE_RERANKS_URL,
            api_key=self.api_key,
            api='openai_rerank',
            dataset='rerank',
            tokenizer_path='Qwen/Qwen3-Embedding-0.6B',
            dataset_path='custom_eval/text/rerank/example.jsonl',
        )
        result = run_perf_benchmark(task_cfg)


if __name__ == '__main__':
    unittest.main(buffer=False)
