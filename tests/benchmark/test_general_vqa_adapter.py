# Copyright (c) Alibaba, Inc. and its affiliates.

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import pytest
from PIL import Image as PILImage

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser, ContentImage, ContentText
from evalscope.api.metric.semantics import MetricSelector
from evalscope.api.registry import BENCHMARK_REGISTRY
from evalscope.benchmarks.general_qa_vqa_metrics import METRIC_SCORE_KEYS
from evalscope.benchmarks.general_vqa.general_vqa_adapter import GeneralVQAAdapter
from evalscope.config import TaskConfig

ROUGE_KEYS = METRIC_SCORE_KEYS['Rouge']
BLEU_KEYS = METRIC_SCORE_KEYS['BLEU']


@pytest.fixture
def adapter() -> GeneralVQAAdapter:
    return GeneralVQAAdapter(
        benchmark_meta=BenchmarkMeta(
            name='general_vqa',
            dataset_id='dummy',
            eval_split='test',
            metric_list=['Rouge', 'BLEU'],
            primary_metric=MetricSelector(
                name='rouge', aggregation='mean', dimensions={'variant': 'l', 'statistic': 'recall'}
            ),
        ),
        task_config=TaskConfig(datasets=['general_vqa']),
    )


@pytest.fixture
def png_bytes() -> bytes:
    image = PILImage.new(mode='RGB', size=(10, 10), color=(255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def _task_state() -> TaskState:
    return TaskState(model='mock-model', sample=Sample(input='question', target='answer'))


def _rouge_values(value: float) -> Dict[str, float]:
    return dict.fromkeys(ROUGE_KEYS, value)


def _raise_metric_error(*args: Any, **kwargs: Any) -> Dict[str, float]:
    raise LookupError('metric exploded')


class TestGeneralVQAAdapterMatchScore:

    def test_rouge_error_returns_real_zero_score_schema(
        self, adapter: GeneralVQAAdapter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            'evalscope.metrics.utils.rouge.compute_rouge_score_one_sample_zh', _raise_metric_error
        )

        score = adapter.match_score('pred', 'pred', 'answer', _task_state())

        assert score is not None
        assert score.value['Rouge-L-R'] == 0.0
        assert {key: score.value[key] for key in ROUGE_KEYS} == dict.fromkeys(ROUGE_KEYS, 0.0)
        assert score.metadata['metric_errors']['Rouge'] == 'LookupError: metric exploded'

    def test_bleu_error_keeps_successful_rouge_values(
        self, adapter: GeneralVQAAdapter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            'evalscope.metrics.utils.rouge.compute_rouge_score_one_sample_zh',
            lambda *args, **kwargs: _rouge_values(1.0),
        )
        monkeypatch.setattr('evalscope.metrics.bleu_ngram_one_sample', _raise_metric_error)

        score = adapter.match_score('answer', 'answer', 'answer', _task_state())

        assert score.value['Rouge-L-R'] == 1.0
        assert {key: score.value[key] for key in BLEU_KEYS} == dict.fromkeys(BLEU_KEYS, 0.0)
        assert score.metadata['metric_errors']['BLEU'] == 'LookupError: metric exploded'

    def test_evaluation_version(self) -> None:
        assert BENCHMARK_REGISTRY['general_vqa'].evaluation_version == 'v1.1'


class TestGeneralVQAAdapterRecordToSample:

    def test_tsv_json_string_messages(self, adapter: GeneralVQAAdapter, tmp_path: Path):
        """Messages as a JSON string (TSV format) with placeholders."""
        image_path = tmp_path / 'car.jpg'
        PILImage.new(mode='RGB', size=(10, 10), color=(128, 128, 0)).save(image_path)

        record = {
            'messages': json.dumps([
                {'role': 'user', 'content': '<image 1> What brand is this car?'}
            ]),
            'image_1': str(image_path),
            'answer': 'Tesla',
        }
        sample = adapter.record_to_sample(record)
        content_list = sample.input[0].content
        assert isinstance(content_list, list)
        assert isinstance(content_list[0], ContentImage)
        assert content_list[0].image == str(image_path)
        assert isinstance(content_list[1], ContentText)
        assert 'What brand is this car?' in content_list[1].text

    def test_no_messages_key(self, adapter: GeneralVQAAdapter):
        """Record without 'messages' key returns empty input."""
        record = {'answer': 'Nothing'}
        sample = adapter.record_to_sample(record)
        assert sample.input == []
        assert sample.target == 'Nothing'

    def test_plain_text_skips_unreferenced_media(self, adapter: GeneralVQAAdapter):
        """Malformed media cells do not affect records without placeholders."""
        sample = adapter.record_to_sample({
            'messages': [{'role': 'user', 'content': 'Describe the scene.'}],
            'image_1': {},
            'answer': 'A scene.',
        })

        assert sample.input[0].content == 'Describe the scene.'

    def test_only_referenced_media_is_parsed(self, adapter: GeneralVQAAdapter):
        """Unreferenced indexed and list media columns do not fail placeholder resolution."""
        sample = adapter.record_to_sample({
            'messages': [{'role': 'user', 'content': '<image 1> Describe the scene.'}],
            'image_1': 'https://example.com/scene.jpg',
            'audio_1': {},
            'videos': 'not-a-list',
            'answer': 'A scene.',
        })

        assert isinstance(sample.input[0].content[0], ContentImage)

    def test_indexed_media_columns_take_precedence_over_list(self, adapter: GeneralVQAAdapter):
        """Indexed media columns take precedence over the equivalent list column."""
        sample = adapter.record_to_sample({
            'messages': [{'role': 'user', 'content': '<image 1> then <image 2>'}],
            'image_1': 'https://example.com/indexed.jpg',
            'images': ['https://example.com/list-1.jpg', 'https://example.com/list-2.jpg'],
            'answer': 'A scene.',
        })

        images = [content.image for content in sample.input[0].content if isinstance(content, ContentImage)]
        assert images == ['https://example.com/indexed.jpg']

    def test_unresolved_only_placeholder_preserves_valid_content(self, adapter: GeneralVQAAdapter):
        """A missing media reference never emits an empty user message."""
        sample = adapter.record_to_sample({
            'messages': [{'role': 'user', 'content': '<image 1>'}],
            'answer': 'Unknown',
        })

        assert sample.input[0].content == '<image 1>'

    def test_empty_answer(self, adapter: GeneralVQAAdapter):
        """Record with empty answer string."""
        record = {
            'messages': [
                {'role': 'user', 'content': 'Say nothing.'}
            ],
            'answer': '',
        }
        sample = adapter.record_to_sample(record)
        assert sample.target == ''
