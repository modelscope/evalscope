# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the X2I text-rendering benchmarks (CVTG-2K, LongText-Bench) from issue #1929.

The scoring cores mirror the official evaluators (TextCrafter and X-Omni); the metric
backends are stubbed so the full pipeline runs without OCR/VLM dependencies.
"""
import base64
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

import pytest

from evalscope.api.evaluator import TaskState
from evalscope.api.metric import SampleScore, Score, SingletonMetric
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig
from evalscope.constants import EvalType, FileConstants, ModelTask, ScoreStatus
from evalscope.metrics.vision import text_render
from evalscope.run import run_task

FIXTURES = Path(__file__).parent / 'fixtures' / 'x2i'


@pytest.fixture(autouse=True)
def isolated_metric_cache(monkeypatch):
    monkeypatch.setattr(SingletonMetric, '_instances', {})


class TestTextRenderScoringCore:
    """The pure scoring functions must reproduce the official evaluator results."""

    def test_extract_quoted_words(self):
        prompt = "A sign saying 'HELLO WORLD' and 'OPEN 24H'"
        assert text_render.extract_quoted_words(prompt) == ['hello', 'world', 'open', '24h']

    def test_extract_quoted_words_empty(self):
        assert text_render.extract_quoted_words('no quotes here') == []

    def test_word_accuracy_exact_match(self):
        assert text_render.word_accuracy(['hello', 'world'], ['hello', 'world', 'extra']) == 1.0

    def test_word_accuracy_repeated_words_not_consumed(self):
        # Official rule: each GT word only checks membership in the recognized words, so a
        # repeated GT word does not consume a second occurrence in the prediction.
        assert text_render.word_accuracy(['hello', 'hello'], ['hello', 'other']) == 1.0
        score, matched, total = text_render.word_accuracy_counts(['hello', 'hello'], ['hello'])
        assert (score, matched, total) == (1.0, 2, 2)

    def test_word_accuracy_no_match(self):
        assert text_render.word_accuracy(['hello'], ['world']) == 0.0

    def test_word_accuracy_empty_gt(self):
        assert text_render.word_accuracy([], ['anything']) == 0.0

    def test_word_level_ned_best_match(self):
        # Each GT word matches its closest prediction: 'helo'~'hello' (1 - 1/5), 'cat'~'cat' (1.0)
        score = text_render.word_level_ned(['helo', 'cat'], ['hello', 'cat', 'dog'])
        assert abs(score - (0.8 + 1.0) / 2) < 1e-6

    def test_word_level_ned_counts(self):
        score, sim_sum, word_count = text_render.word_ned_counts(['helo', 'cat'], ['hello', 'cat', 'dog'])
        assert abs(sim_sum - 1.8) < 1e-6
        assert word_count == 2
        assert abs(score - 0.9) < 1e-6

    def test_word_level_ned_no_prediction(self):
        assert text_render.word_level_ned(['hello'], []) == pytest.approx(1 - 5 / (5 + 1e-5))

    def test_text_score_en(self):
        # English: word-level Counter intersection over the ground truth.
        assert text_render.text_score('Hello World', 'hello world and more', lang='en') == 1.0

    def test_text_score_en_partial(self):
        score = text_render.text_score('Hello Big World', 'hello world', lang='en')
        assert abs(score - 2 / 3) < 1e-6

    def test_text_score_zh(self):
        # Chinese: character-level; every GT char of '出口A' appears in the prediction.
        assert text_render.text_score('出口A', '出口A出口', lang='zh') == 1.0

    def test_text_score_zh_counts_each_char_once(self):
        # Official Chinese example: GT '出口A换乘2号线' (8 chars) vs '出口A换乘2号' (7) → 0.875.
        assert abs(text_render.text_score('出口A换乘2号线', '出口A换乘2号', lang='zh') - 0.875) < 1e-6

    def test_text_score_zh_removes_whitespace(self):
        # Official Chinese normalization removes all whitespace: '出口 A' equals '出口A'.
        assert text_render.text_score('出口A', '出口 A', lang='zh') == 1.0

    def test_text_score_detects_language_per_sample(self):
        # lang=None detects the language from the ground truth, so one run covers en + zh.
        assert text_render.text_score('Hello World', 'hello world') == 1.0
        assert text_render.text_score('出口A换乘2号线', '出口A换乘2号') == 0.875

    def test_text_score_ignores_illegal_chars(self):
        # Punctuation and whitespace are stripped before matching.
        assert text_render.text_score('Hello, World!', 'hello world', lang='en') == 1.0


class TestX2IAdapters:
    def test_cvtg_record_to_sample(self):
        from evalscope.api.benchmark import BenchmarkMeta
        from evalscope.benchmarks.text2image.cvtg2k_adapter import CVTG2KAdapter

        meta = BenchmarkMeta(name='cvtg_2k', dataset_id='x', subset_list=['CVTG'], metric_list=['CVTGWordAccuracy', 'CVTGNED'])
        adapter = CVTG2KAdapter(benchmark_meta=meta)
        sample = adapter.record_to_sample({'id': '0', 'prompt': "sign saying 'HI'", 'benchmark_type': 'CVTG'})
        assert sample.input[0].content == "sign saying 'HI'"
        assert sample.metadata['category'] == 'CVTG'

    def test_longtext_record_to_sample(self):
        from evalscope.api.benchmark import BenchmarkMeta
        from evalscope.benchmarks.text2image.longtext_bench_adapter import LongTextBenchAdapter

        meta = BenchmarkMeta(name='longtext_bench', dataset_id='x', subset_list=['en'], metric_list=['LongTextScore'])
        adapter = LongTextBenchAdapter(benchmark_meta=meta)
        sample = adapter.record_to_sample({'prompt_id': 0, 'prompt': 'poster', 'text': 'HELLO'})
        assert sample.target == 'HELLO'
        assert sample.input[0].content == 'poster'


_PNG_1PX = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII='
)


class _ImageHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        self.rfile.read(int(self.headers.get('Content-Length', '0')))
        body = json.dumps({'created': 1, 'data': [{'b64_json': base64.b64encode(_PNG_1PX).decode()}]}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@contextmanager
def _image_service() -> Iterator[str]:
    server = ThreadingHTTPServer(('127.0.0.1', 0), _ImageHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f'http://{host}:{port}/v1/images/generations'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _read_report_scores(work_dir: Path, benchmark: str) -> dict:
    report_file = next((work_dir).glob(f'*/reports/*/{benchmark}.json'))
    report = json.loads(report_file.read_text(encoding='utf-8'))
    return {m['identity']['name'] + json.dumps(m['identity']['dimensions'], sort_keys=True): m['score'] for m in report['metrics']}


class TestX2IFullPipeline:
    """Full flow: local jsonl dataset + local image service + stubbed scoring backends."""

    def test_run_text_rendering_benchmarks(self, tmp_path, monkeypatch):
        from evalscope.metrics.vision.metrics import LongTextScore

        monkeypatch.setattr(LongTextScore, '_transcribe', lambda self, path: 'SOUND OF FREEDOM AUGUST 16 2026 出口A换乘2号线')
        with _image_service() as api_url:
            task_cfg = TaskConfig(
                model='test-image-model',
                model_task=ModelTask.IMAGE_GENERATION,
                eval_type=EvalType.TEXT2IMAGE,
                api_url=api_url,
                api_key='test-key',
                work_dir=str(tmp_path),
                datasets=['cvtg_2k', 'longtext_bench'],
                dataset_args={
                    'cvtg_2k': {
                        'dataset_id': str(FIXTURES / 'cvtg2k.jsonl'),
                        'metric_list': [{'CVTGWordAccuracy': {'backend': 'stub'}}, {'CVTGNED': {'backend': 'stub'}}],
                    },
                    'longtext_bench': {
                        'dataset_id': str(FIXTURES / 'longtext.jsonl'),
                    },
                },
            )
            run_task(task_cfg=task_cfg)

        cvtg_scores = _read_report_scores(tmp_path, 'cvtg_2k')
        # The stub backend echoes the ground truth, so every score must be perfect.
        word_acc_key = 'cvtg_word_accuracy' + json.dumps({'scope': 'overall'}, sort_keys=True)
        ned_key = 'cvtg_ned' + json.dumps({'scope': 'overall'}, sort_keys=True)
        assert cvtg_scores[word_acc_key] == 1.0
        assert cvtg_scores[ned_key] == 1.0

        lt_scores = _read_report_scores(tmp_path, 'longtext_bench')
        assert lt_scores['text_score' + json.dumps({'scope': 'overall'}, sort_keys=True)] == 1.0


class TestCorpusAggregation:
    """Report-level regression: official corpus aggregation, not mean-of-ratios.

    Reviewer example: one fully correct one-word sample + one fully wrong three-word
    sample must score 1/4 (corpus), not 0.5 (mean of ratios).
    """

    @staticmethod
    def _sample_score(metric_name: str, score: float, matched, total, category='cvtg'):
        from evalscope.api.metric import SampleScore

        return SampleScore(
            score=Score(
                value={metric_name: score},
                metadata={'metric_metadata': {metric_name: {'corpus_counts': {'numerator': matched, 'denominator': total}}}},
            ),
            sample_id=0,
            sample_metadata={'category': category},
        )

    def test_cvtg_corpus_aggregation(self):
        from evalscope.api.benchmark import BenchmarkMeta
        from evalscope.benchmarks.text2image.cvtg2k_adapter import CVTG2KAdapter

        meta = BenchmarkMeta(name='cvtg_2k', dataset_id='x', subset_list=['CVTG'], metric_list=['CVTGWordAccuracy'])
        adapter = CVTG2KAdapter(benchmark_meta=meta)
        sample_scores = [
            self._sample_score('cvtg_word_accuracy', 1.0, 1, 1),
            self._sample_score('cvtg_word_accuracy', 0.0, 0, 3),
        ]
        rows = {a.metric_name + json.dumps(a.dimensions, sort_keys=True): a.score for a in adapter.aggregate_scores(sample_scores)}
        overall = rows['cvtg_word_accuracy' + json.dumps({'scope': 'overall'}, sort_keys=True)]
        assert abs(overall - 0.25) < 1e-9, f'corpus aggregate must be 0.25, got {overall}'

    def test_cvtg_corpus_aggregation_ned(self):
        from evalscope.api.benchmark import BenchmarkMeta
        from evalscope.benchmarks.text2image.cvtg2k_adapter import CVTG2KAdapter

        meta = BenchmarkMeta(name='cvtg_2k', dataset_id='x', subset_list=['CVTG'], metric_list=['CVTGNED'])
        adapter = CVTG2KAdapter(benchmark_meta=meta)
        sample_scores = [
            self._sample_score('cvtg_ned', 1.0, 1.0, 1),
            self._sample_score('cvtg_ned', 0.0, 0.0, 3),
        ]
        rows = {a.metric_name + json.dumps(a.dimensions, sort_keys=True): a.score for a in adapter.aggregate_scores(sample_scores)}
        overall = rows['cvtg_ned' + json.dumps({'scope': 'overall'}, sort_keys=True)]
        assert abs(overall - 0.25) < 1e-9

    def test_longtext_corpus_aggregation_by_category(self):
        from evalscope.api.benchmark import BenchmarkMeta
        from evalscope.benchmarks.text2image.longtext_bench_adapter import LongTextBenchAdapter

        meta = BenchmarkMeta(name='longtext_bench', dataset_id='x', subset_list=['en'], metric_list=['LongTextScore'])
        adapter = LongTextBenchAdapter(benchmark_meta=meta)
        sample_scores = [
            self._sample_score('text_score', 1.0, 2, 2, category='poster'),
            self._sample_score('text_score', 0.5, 1, 2, category='poster'),
            self._sample_score('text_score', 0.0, 0, 4, category='signage'),
        ]
        rows = {a.metric_name + json.dumps(a.dimensions, sort_keys=True): a.score for a in adapter.aggregate_scores(sample_scores)}
        overall = rows['text_score' + json.dumps({'scope': 'overall'}, sort_keys=True)]
        poster = rows['text_score' + json.dumps({'category': 'poster'}, sort_keys=True)]
        signage = rows['text_score' + json.dumps({'category': 'signage'}, sort_keys=True)]
        assert abs(overall - 3 / 8) < 1e-9
        assert abs(poster - 3 / 4) < 1e-9
        assert signage == 0.0

    def test_cached_scores_without_counts_require_review(self):
        from evalscope.api.benchmark import BenchmarkMeta
        from evalscope.benchmarks.text2image.longtext_bench_adapter import LongTextBenchAdapter

        meta = BenchmarkMeta(name='longtext_bench', dataset_id='x', subset_list=['en'], metric_list=['LongTextScore'])
        adapter = LongTextBenchAdapter(benchmark_meta=meta)
        # Old cached reviews cannot silently fall back to a different aggregation protocol.
        from evalscope.api.metric import SampleScore

        sample_scores = [
            SampleScore(score=Score(value={'text_score': 0.4}), sample_id=0, sample_metadata={'category': 'poster'}),
            SampleScore(score=Score(value={'text_score': 0.8}), sample_id=1, sample_metadata={'category': 'poster'}),
        ]
        with pytest.raises(ValueError, match='rerun_review=True'):
            adapter.aggregate_scores(sample_scores)


def _score_image(adapter, target: str, image_path: str, sample_id: int) -> SampleScore:
    sample = adapter.record_to_sample({'prompt': target, 'text': target, 'category': 'poster'})
    sample.metadata[FileConstants.IMAGE_PATH] = image_path
    state = TaskState(model='test', sample=sample)
    return SampleScore(
        score=adapter.match_score('', '', target, state), sample_id=sample_id, sample_metadata=sample.metadata,
    )


def test_counts_travel_through_actual_scoring_and_failed_rereview(monkeypatch):
    from evalscope.metrics.vision.metrics import LongTextScore

    config = TaskConfig(model='test', datasets=['longtext_bench'])
    adapter = get_benchmark('longtext_bench', config)
    transcripts = {'first.png': 'a', 'second.png': ''}
    monkeypatch.setattr(LongTextScore, '_transcribe', lambda self, path: transcripts[path])
    first = _score_image(adapter, 'a', 'first.png', 0)
    second = _score_image(adapter, 'b c d', 'second.png', 1)
    aggregates = adapter.aggregate_scores([first, second])
    assert next(a for a in aggregates if a.dimensions == {'scope': 'overall'}).score == 0.25

    def fail(self, path):
        raise RuntimeError('transcription failed')

    monkeypatch.setattr(LongTextScore, '_transcribe', fail)
    failed = _score_image(adapter, 'a', 'first.png', 0)
    assert failed.score.status is ScoreStatus.EXCLUDED
    assert not failed.score.value
    assert 'corpus_counts' not in str(failed.score.metadata)
    # A valid zero remains scored; a transcription failure is excluded from both score and num.
    aggregate = next(a for a in adapter.aggregate_scores([second, failed]) if a.dimensions == {'scope': 'overall'})
    assert aggregate.score == 0.0
    assert aggregate.num == 1
    assert aggregate.ids == [1]


def test_cvtg_report_retains_word_weighting_across_subsets_and_reload(monkeypatch):
    import evalscope.metrics.vision.metrics as metrics
    from evalscope.report import Report

    config = TaskConfig(model='test', datasets=['cvtg_2k'], dataset_args={
        'cvtg_2k': {'metric_list': [{'CVTGWordAccuracy': {'backend': 'stub'}}]},
    })
    adapter = get_benchmark('cvtg_2k', config)
    monkeypatch.setattr(metrics, '_recognize_cvtg_words', lambda path, *args: ['a'] if path == 'hit.png' else [])
    hit = _score_image(adapter, "sign 'a'", 'hit.png', 0)
    miss = _score_image(adapter, "sign 'b c d'", 'miss.png', 1)
    report = adapter._on_generate_report({
        'CVTG': adapter.aggregate_scores([hit]), 'CVTG-Style': adapter.aggregate_scores([miss]),
    }, 'test')
    reloaded = Report.model_validate_json(report.model_dump_json())
    assert reloaded.score == 0.25
    assert reloaded.primary_metric.categories[0].score == 0.25
    assert reloaded.primary_metric.num == 2
    assert reloaded.model_dump(mode='json') == report.model_dump(mode='json')


def test_one_failed_metric_keeps_only_the_other_metric(monkeypatch):
    from evalscope.metrics.vision.metrics import CVTGNED

    def fail(self, image_path, prompt):
        raise RuntimeError('NED recognition failed')

    monkeypatch.setattr(CVTGNED, '_score_counts', fail)
    config = TaskConfig(model='test', datasets=['cvtg_2k'], dataset_args={
        'cvtg_2k': {'metric_list': [{'CVTGWordAccuracy': {'backend': 'stub'}}, {'CVTGNED': {'backend': 'stub'}}]},
    })
    adapter = get_benchmark('cvtg_2k', config)
    sample = _score_image(adapter, "sign 'hello'", 'one.png', 0)
    assert sample.score.status is ScoreStatus.DEGRADED
    assert sample.score.value == {'cvtg_word_accuracy': 1.0}
    assert all(a.metric_name == 'cvtg_word_accuracy' for a in adapter.aggregate_scores([sample]))


def test_longtext_report_averages_languages_after_corpus_aggregation():
    from evalscope.report import Report

    adapter = get_benchmark('longtext_bench')
    samples = TestCorpusAggregation._sample_score
    report = adapter._on_generate_report({
        'en': adapter.aggregate_scores([samples('text_score', 1 / 3, 1, 3)]),
        'zh': adapter.aggregate_scores([samples('text_score', 1.0, 7, 7)]),
    }, 'test')
    reloaded = Report.model_validate_json(report.model_dump_json())
    # Official Avg. = mean(English, Chinese), not token-weighted 8/10 or image-weighted rounded inputs.
    assert reloaded.score == 0.6667
    assert reloaded.primary_metric.categories[0].score == 0.6667
    assert reloaded.model_dump(mode='json') == report.model_dump(mode='json')


def test_hf_archive_loading_preserves_standard_sampling(tmp_path, monkeypatch):
    from zipfile import ZipFile

    import evalscope.benchmarks.text2image.cvtg2k_adapter as module

    archive_path = tmp_path / 'CVTG-2K.zip'
    with ZipFile(archive_path, 'w') as archive:
        for region_count in range(2, 6):
            archive.writestr(f'CVTG-2K/CVTG/{region_count}.json', json.dumps({
                'data_list': [{'prompt': f"sign '{region_count}'", 'index': region_count}],
            }))
    downloads = []

    def download(*args, **kwargs):
        downloads.append(kwargs)
        return str(archive_path)

    monkeypatch.setattr(module, 'download_dataset_file', download)
    config = TaskConfig(model='test', eval_type='mock_llm', datasets=['cvtg_2k'], dataset_hub='huggingface',
                        limit=0.5, repeats=2, seed=7,
                        dataset_args={'cvtg_2k': {'subset_list': ['CVTG'], 'shuffle': True}})
    adapter = get_benchmark('cvtg_2k', config)
    data = adapter.load_dataset()['CVTG']
    assert len(data) == 4
    assert data[0].input == data[1].input
    assert data[2].input == data[3].input
    assert data[0].group_id != data[2].group_id
    assert downloads[0]['data_source'] == 'huggingface'
    assert downloads[0]['revision'] == adapter.dataset_revision


def test_hf_language_files_and_list_targets(tmp_path, monkeypatch):
    import evalscope.benchmarks.text2image.longtext_bench_adapter as module

    for filename, fragments in [('text_prompts.jsonl', ['hello', 'world']), ('text_prompts_zh.jsonl', ['出口', '换乘'])]:
        (tmp_path / filename).write_text(json.dumps({'prompt': 'poster', 'text': fragments}) + '\n')
    monkeypatch.setattr(module, 'download_dataset_file', lambda repo, filename, **kwargs: str(tmp_path / filename))
    config = TaskConfig(model='test', eval_type='mock_llm', datasets=['longtext_bench'], dataset_hub='huggingface')
    adapter = get_benchmark('longtext_bench', config)
    data = adapter.load_dataset()
    assert data['en'][0].target == 'hello world'
    assert data['zh'][0].target == '出口 换乘'


def test_default_transcription_key_is_independent_of_image_service(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from PIL import Image

    import evalscope.api.model as model_api
    from evalscope.api.model import ModelOutput
    from evalscope.metrics.vision.metrics import LongTextScore

    monkeypatch.setenv('DASHSCOPE_API_KEY', 'transcription-key')
    monkeypatch.setenv('EVALSCOPE_API_KEY', 'image-service-key')
    image_path = tmp_path / 'image.png'
    Image.new('RGB', (4, 4)).save(image_path)
    calls = []

    def get_model(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(generate=lambda **unused: ModelOutput.from_content('vision', 'hello'))

    monkeypatch.setattr(model_api, 'get_model', get_model)
    result = LongTextScore()(str(image_path), 'hello')
    assert result.main_value == 1.0
    assert calls[0]['api_key'] == 'transcription-key'


@pytest.mark.parametrize('status, finish_reason, transcript, expected', [
    (200, 'stop', 'hello world', 1.0),
    (200, 'stop', 'No text recognized.', 0.0),
    (200, 'length', 'hello world', None),
    (503, 'stop', '', None),
])
def test_vlm_api_backend_without_local_weights(tmp_path, monkeypatch, status, finish_reason, transcript, expected):
    import sys

    from PIL import Image

    from evalscope.metrics.vision.metrics import LongTextScore

    monkeypatch.setitem(sys.modules, 'transformers', None)
    monkeypatch.setitem(sys.modules, 'qwen_vl_utils', None)
    image_path = tmp_path / 'image.png'
    Image.new('RGB', (4, 4)).save(image_path)
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == '/v1/chat/completions'
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append(body)
            assert body['messages'][0]['content'][0]['image_url']['url'].startswith('data:image/png;base64,')
            assert body['messages'][0]['content'][1]['text'] == LongTextScore.RECOGNIZE_PROMPT
            response = {'id': 'test', 'object': 'chat.completion', 'created': 1, 'model': 'vision',
                        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': transcript},
                                     'finish_reason': finish_reason}]} if status == 200 else {'error': {'message': 'unavailable'}}
            encoded = json.dumps(response).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        metric = LongTextScore(model='vision', api_url=f'http://127.0.0.1:{server.server_port}/v1',
                               api_key='test-key', generation_config={'retries': 1, 'max_tokens': 128})
        result = metric(str(image_path), 'hello world')
        assert result.status.is_usable is (expected is not None)
        assert len(calls) == 1
        if expected is not None:
            assert result.main_value == expected
            assert result.metadata['corpus_counts'] == {'numerator': 2 * expected, 'denominator': 2}
        else:
            assert not result.value
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
