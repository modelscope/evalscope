from __future__ import annotations

from types import SimpleNamespace
from typing import AsyncIterator, Dict, Type

from evalscope.perf.config.resolve import ResolvedRunSpec
from evalscope.perf.domain.workload import ConversationItem, SingleTurnItem, Turn, WorkItem, WorkloadMeta
from evalscope.perf.workloads.base import WorkloadSource
from evalscope.perf.workloads.builtins.custom import CustomDatasetPlugin, CustomMultiTurnDatasetPlugin
from evalscope.perf.workloads.builtins.embedding import (
    EmbeddingBatchDatasetPlugin,
    EmbeddingDatasetPlugin,
    RandomEmbeddingBatchDatasetPlugin,
    RandomEmbeddingDatasetPlugin,
)
from evalscope.perf.workloads.builtins.flickr8k import FlickrDatasetPlugin
from evalscope.perf.workloads.builtins.kontext_bench import KontextDatasetPlugin
from evalscope.perf.workloads.builtins.line_by_line import LineByLineDatasetPlugin
from evalscope.perf.workloads.builtins.longalpaca import LongAlpacaDatasetPlugin
from evalscope.perf.workloads.builtins.multi_turn import (
    RandomMultiTurnDatasetPlugin,
    ShareGPTEnMultiTurnPlugin,
    ShareGPTZhMultiTurnPlugin,
)
from evalscope.perf.workloads.builtins.openqa import OpenqaDatasetConverter
from evalscope.perf.workloads.builtins.random import RandomDatasetPlugin
from evalscope.perf.workloads.builtins.random_vl import RandomVLDatasetPlugin
from evalscope.perf.workloads.builtins.rerank import RandomRerankDatasetPlugin, RerankDatasetPlugin
from evalscope.perf.workloads.builtins.share_gpt import ShareGPTEnDatasetPlugin, ShareGPTZhDatasetPlugin
from evalscope.perf.workloads.builtins.speed import SpeedBenchmarkDatasetPlugin, SpeedBenchmarkLongDatasetPlugin
from evalscope.perf.workloads.builtins.swe_smith import SweSmithDatasetPlugin
from evalscope.perf.workloads.builtins.trie import TrieAgenticCodingPlugin, TrieCodeQaPlugin, TrieOfficeWorkPlugin
from evalscope.perf.workloads.registry import register_workload

_ALL_PROTOCOLS = frozenset({'openai_chat', 'openai_completions', 'openai_responses'})
_HUB_WORKLOADS = frozenset({
    'openqa',
    'longalpaca',
    'share_gpt_zh',
    'share_gpt_en',
    'flickr8k',
    'kontext_bench',
    'share_gpt_zh_multi_turn',
    'share_gpt_en_multi_turn',
    'swe_smith',
    'trie_agentic_coding',
    'trie_code_qa',
    'trie_office_work',
    'embedding',
    'embedding_batch',
    'rerank',
})
_SYNTHETIC_WORKLOADS = frozenset({
    'random',
    'random_vl',
    'speed_benchmark',
    'speed_benchmark_long',
    'random_multi_turn',
    'random_embedding',
    'random_embedding_batch',
    'random_rerank',
})


class ConverterWorkload(WorkloadSource):
    """Adapt a built-in record converter to the lazy WorkloadSource contract."""

    plugin_class: Type

    def _parameters(self, run: ResolvedRunSpec) -> SimpleNamespace:
        config = self.context.config
        workload = config.workload
        generation = config.generation
        options = workload.options
        total_count = (run.item_limit or 1) + run.warmup_count
        max_turns = getattr(run.load, 'max_turns', None) or options.get('max_turns')
        return SimpleNamespace(
            apply_chat_template=options.get('apply_chat_template', config.target.protocol == 'openai_chat'),
            data_source=workload.data_source,
            dataset_offset=options.get('dataset_offset', 0),
            dataset_path=workload.path,
            extra_args=generation.extra,
            image_format=options.get('image_format', 'RGB'),
            image_height=options.get('image_height', 224),
            image_num=options.get('image_num', 1),
            image_width=options.get('image_width', 224),
            max_prompt_length=workload.max_prompt_length,
            max_turns=max_turns,
            min_prompt_length=workload.min_prompt_length,
            min_turns=options.get('min_turns', 1),
            multi_turn=self.meta.mode == 'conversation',
            multi_turn_args=self._multi_turn_options(options.get('multi_turn_args')),
            num_workers=config.runtime.dataset_workers,
            parallel=getattr(run.load, 'concurrency', 1),
            prefix_length=options.get('prefix_length', 0),
            tokenize_prompt=options.get('tokenize_prompt', False),
            tokenizer_path=config.target.tokenizer,
            total_count=total_count,
            url=config.target.base_url,
        )

    @staticmethod
    def _convert(value) -> WorkItem:
        if isinstance(value, list) and value and hasattr(value[0], 'messages'):
            return ConversationItem(
                turns=[
                    Turn(
                        messages=turn.messages,
                        max_tokens=getattr(turn, 'max_tokens', None),
                        tool_call_latency=getattr(turn, 'tool_call_latency', None),
                    ) for turn in value
                ]
            )
        return SingleTurnItem(messages=value)

    @staticmethod
    def _multi_turn_options(value):
        if value is None or hasattr(value, 'sample_params'):
            return value
        from evalscope.perf.workloads.builtins.multi_turn_options import MultiTurnOptions

        return MultiTurnOptions.model_validate(value)

    async def iter_items(self, run: ResolvedRunSpec) -> AsyncIterator[WorkItem]:
        target = None if run.item_limit is None else run.item_limit + run.warmup_count
        yielded = 0
        while target is None or yielded < target:
            plugin = self.plugin_class(self._parameters(run))
            produced = 0
            for value in plugin.build_messages():
                yield self._convert(value)
                yielded += 1
                produced += 1
                if target is not None and yielded >= target:
                    return
            if produced == 0:
                raise ValueError(f'Workload {self.meta.name!r} produced no items')


def _define(
    name: str,
    plugin_class: Type,
    *,
    mode: str = 'single_turn',
    protocols=frozenset(_ALL_PROTOCOLS),
    local: bool = False,
    tokenizer: bool = False,
) -> None:
    if mode == 'conversation':
        protocols = frozenset(protocols) - {'openai_completions'}
    meta = WorkloadMeta(
        name=name,
        mode=mode,
        requires_dataset=name not in _SYNTHETIC_WORKLOADS,
        supports_parallel_generation=name in {'random', 'swe_smith'},
        supports_local_path=local,
        supports_modelscope=name in _HUB_WORKLOADS,
        supports_huggingface=name in _HUB_WORKLOADS,
        requires_tokenizer=tokenizer,
        protocols=protocols,
    )
    cls = type(f'{plugin_class.__name__}Workload', (ConverterWorkload, ), {'meta': meta, 'plugin_class': plugin_class})
    register_workload(cls)


_DEFINITIONS: Dict[str, tuple] = {
    'custom': (CustomDatasetPlugin, 'single_turn', True, _ALL_PROTOCOLS, False),
    'custom_multi_turn': (CustomMultiTurnDatasetPlugin, 'conversation', True, _ALL_PROTOCOLS, False),
    'openqa': (OpenqaDatasetConverter, 'single_turn', True, _ALL_PROTOCOLS, False),
    'longalpaca': (LongAlpacaDatasetPlugin, 'single_turn', True, _ALL_PROTOCOLS, False),
    'line_by_line': (LineByLineDatasetPlugin, 'single_turn', True, _ALL_PROTOCOLS, False),
    'line_by_line_oai': (LineByLineDatasetPlugin, 'single_turn', True, _ALL_PROTOCOLS, False),
    'share_gpt_zh': (ShareGPTZhDatasetPlugin, 'single_turn', True, _ALL_PROTOCOLS, False),
    'share_gpt_en': (ShareGPTEnDatasetPlugin, 'single_turn', True, _ALL_PROTOCOLS, False),
    'random': (RandomDatasetPlugin, 'single_turn', False, _ALL_PROTOCOLS, True),
    'random_vl': (RandomVLDatasetPlugin, 'single_turn', False, _ALL_PROTOCOLS, True),
    'flickr8k': (FlickrDatasetPlugin, 'single_turn', True, _ALL_PROTOCOLS, False),
    'kontext_bench': (KontextDatasetPlugin, 'single_turn', True, _ALL_PROTOCOLS, False),
    'speed_benchmark': (SpeedBenchmarkDatasetPlugin, 'single_turn', False, _ALL_PROTOCOLS, False),
    'speed_benchmark_long': (SpeedBenchmarkLongDatasetPlugin, 'single_turn', False, _ALL_PROTOCOLS, False),
    'random_multi_turn': (RandomMultiTurnDatasetPlugin, 'conversation', False, _ALL_PROTOCOLS, True),
    'share_gpt_zh_multi_turn': (ShareGPTZhMultiTurnPlugin, 'conversation', True, _ALL_PROTOCOLS, False),
    'share_gpt_en_multi_turn': (ShareGPTEnMultiTurnPlugin, 'conversation', True, _ALL_PROTOCOLS, False),
    'swe_smith': (SweSmithDatasetPlugin, 'conversation', True, _ALL_PROTOCOLS, True),
    'trie_agentic_coding': (TrieAgenticCodingPlugin, 'conversation', True, _ALL_PROTOCOLS, True),
    'trie_code_qa': (TrieCodeQaPlugin, 'conversation', True, _ALL_PROTOCOLS, True),
    'trie_office_work': (TrieOfficeWorkPlugin, 'conversation', True, _ALL_PROTOCOLS, True),
    'embedding': (EmbeddingDatasetPlugin, 'single_turn', True, frozenset({'openai_embedding'}), False),
    'embedding_batch': (EmbeddingBatchDatasetPlugin, 'single_turn', True, frozenset({'openai_embedding'}), False),
    'random_embedding': (RandomEmbeddingDatasetPlugin, 'single_turn', False, frozenset({'openai_embedding'}), True),
    'random_embedding_batch': (
        RandomEmbeddingBatchDatasetPlugin,
        'single_turn',
        False,
        frozenset({'openai_embedding'}),
        True,
    ),
    'rerank': (RerankDatasetPlugin, 'single_turn', True, frozenset({'openai_rerank'}), False),
    'random_rerank': (RandomRerankDatasetPlugin, 'single_turn', False, frozenset({'openai_rerank'}), True),
}

for _name, (_plugin, _mode, _local, _protocols, _tokenizer) in _DEFINITIONS.items():
    _define(_name, _plugin, mode=_mode, protocols=_protocols, local=_local, tokenizer=_tokenizer)
