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


class TestCorpusAggregation:
    """Report-level regression: official corpus aggregation, not mean-of-ratios.

    Reviewer example: one fully correct one-word sample + one fully wrong three-word
    sample must score 1/4 (corpus), not 0.5 (mean of ratios).
    """

    @staticmethod
    def _sample_score(metric_name: str, score: float, matched, total, category='cvtg'):
        from evalscope.api.metric import SampleScore, Score

        return SampleScore(
            score=Score(
                value={metric_name: score},
                metadata={'corpus_counts': {metric_name: (matched, total)}},
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

    def test_zero_denominator_keeps_mean_fallback(self):
        from evalscope.api.benchmark import BenchmarkMeta
        from evalscope.benchmarks.text2image.longtext_bench_adapter import LongTextBenchAdapter

        meta = BenchmarkMeta(name='longtext_bench', dataset_id='x', subset_list=['en'], metric_list=['LongTextScore'])
        adapter = LongTextBenchAdapter(benchmark_meta=meta)
        # No corpus_counts at all (e.g. metric errored): the inherited mean must survive.
        from evalscope.api.metric import SampleScore, Score

        sample_scores = [
            SampleScore(score=Score(value={'text_score': 0.4}), sample_id=0, sample_metadata={'category': 'poster'}),
            SampleScore(score=Score(value={'text_score': 0.8}), sample_id=1, sample_metadata={'category': 'poster'}),
        ]
        rows = {a.metric_name + json.dumps(a.dimensions, sort_keys=True): a.score for a in adapter.aggregate_scores(sample_scores)}
        overall = rows['text_score' + json.dumps({'scope': 'overall'}, sort_keys=True)]
        assert abs(overall - 0.6) < 1e-9


class TestNonStubTranscriptionPath:
    """The real LongText transcription path must run end-to-end without a GPU.

    Fake ``transformers``/``qwen_vl_utils`` modules are injected so the lazy model
    loading succeeds and the prompt/image/generate/decode chain is exercised (the
    reviewer reproduced a ModuleNotFoundError here before the fix).
    """

    def test_transcribe_via_fake_transformers(self, tmp_path, monkeypatch):
        from PIL import Image

        from evalscope.metrics.vision.metrics import LongTextScore

        # Other tests may have initialized the singleton with backend='stub'; reset it so
        # this test exercises the real lazy-loading path.
        monkeypatch.delattr(LongTextScore, '_instance', raising=False)
        image_path = tmp_path / 'render.png'
        Image.new('RGB', (4, 4), 'white').save(image_path)

        class FakeProcessor:

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                assert messages[0]['content'][1]['text'] == LongTextScore.RECOGNIZE_PROMPT
                assert messages[0]['content'][0]['image'].startswith('data:image/png;base64,')
                return 'PROMPT'

            def __call__(self, text, images, return_tensors):
                import torch

                class FakeInputs(dict):

                    def to(self, device):
                        return self

                    def __getattr__(self, name):
                        try:
                            return self[name]
                        except KeyError:
                            raise AttributeError(name)

                return FakeInputs({'input_ids': torch.tensor([[1, 2, 3]]), 'pixel_values': torch.zeros(1)})

            def batch_decode(self, generated, skip_special_tokens=True):
                return ['hello world']

        class FakeModel:

            device = 'cpu'

            def generate(self, **kwargs):
                import torch

                return torch.tensor([[1, 2, 3, 4, 5]])

        import importlib.machinery
        import types

        fake_transformers = types.ModuleType('transformers')
        fake_transformers.__spec__ = importlib.machinery.ModuleSpec('transformers', None)
        fake_transformers.Qwen2_5_VLForConditionalGeneration = type(
            'Qwen2_5_VLForConditionalGeneration', (), {'from_pretrained': staticmethod(lambda *a, **k: FakeModel())}
        )
        fake_transformers.AutoProcessor = type('AutoProcessor', (), {'from_pretrained': staticmethod(lambda *a, **k: FakeProcessor())})
        fake_qwen_vl_utils = types.ModuleType('qwen_vl_utils')
        fake_qwen_vl_utils.__spec__ = importlib.machinery.ModuleSpec('qwen_vl_utils', None)
        monkeypatch.setitem(__import__('sys').modules, 'transformers', fake_transformers)
        monkeypatch.setitem(__import__('sys').modules, 'qwen_vl_utils', fake_qwen_vl_utils)

        metric = LongTextScore(backend='qwen_vl')
        transcript = metric._transcribe(str(image_path), 'HELLO WORLD')
        assert transcript == 'hello world'

        # The counts stash must reflect the real transcription through text_score_counts.
        score = metric.apply([str(image_path)], ['HELLO WORLD'])[0]
        assert score == 1.0
        assert metric.counts_for(str(image_path)) == (2, 2)
