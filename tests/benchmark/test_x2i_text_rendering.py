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

from evalscope.config import TaskConfig
from evalscope.constants import EvalType, ModelTask
from evalscope.metrics.vision import text_render
from evalscope.run import run_task

FIXTURES = Path(__file__).parent / 'fixtures' / 'x2i'


class TestTextRenderScoringCore:
    """The pure scoring functions must reproduce the official evaluator results."""

    def test_extract_quoted_words(self):
        prompt = "A sign saying 'HELLO WORLD' and 'OPEN 24H'"
        assert text_render.extract_quoted_words(prompt) == ['hello', 'world', 'open', '24h']

    def test_extract_quoted_words_empty(self):
        assert text_render.extract_quoted_words('no quotes here') == []

    def test_word_accuracy_exact_match(self):
        assert text_render.word_accuracy(['hello', 'world'], ['hello', 'world', 'extra']) == 1.0

    def test_word_accuracy_counts_duplicates(self):
        # The word "hello" appears twice in the ground truth but only once in the prediction.
        assert text_render.word_accuracy(['hello', 'hello'], ['hello', 'other']) == 0.5

    def test_word_accuracy_no_match(self):
        assert text_render.word_accuracy(['hello'], ['world']) == 0.0

    def test_word_accuracy_empty_gt(self):
        assert text_render.word_accuracy([], ['anything']) == 0.0

    def test_word_level_ned_best_match(self):
        # Each GT word matches its closest prediction: 'helo'~'hello' (1 - 1/5), 'cat'~'cat' (1.0)
        score = text_render.word_level_ned(['helo', 'cat'], ['hello', 'cat', 'dog'])
        assert abs(score - (0.8 + 1.0) / 2) < 1e-6

    def test_word_level_ned_no_prediction(self):
        assert text_render.word_level_ned(['hello'], []) == 0.0

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
        # GT '出口A换乘' (5 chars); the prediction drops '乘' → 4/5.
        assert abs(text_render.text_score('出口A换乘', '出口A换', lang='zh') - 4 / 5) < 1e-6

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
    report_file = next((work_dir).glob('*/reports/*/cvtg_2k.json')) if benchmark == 'cvtg_2k' else None
    report_file = next((work_dir).glob(f'*/reports/*/{benchmark}.json'))
    report = json.loads(report_file.read_text(encoding='utf-8'))
    return {m['identity']['name'] + json.dumps(m['identity']['dimensions'], sort_keys=True): m['score'] for m in report['metrics']}


class TestX2IFullPipeline:
    """Full flow: local jsonl dataset + local image service + stubbed scoring backends."""

    def test_run_text_rendering_benchmarks(self, tmp_path):
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
                        'metric_list': [{'LongTextScore': {'backend': 'stub'}}],
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
