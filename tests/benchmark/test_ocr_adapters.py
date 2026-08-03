import json
from unittest.mock import patch

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.benchmarks.ocr_bench.ocr_bench.ocr_bench_adapter import OCRBenchAdapter
from evalscope.benchmarks.omnidoc_bench.omnidoc_bench_adapter import OmniDocBenchAdapter
from evalscope.config import TaskConfig


def make_ocr_bench_adapter() -> OCRBenchAdapter:
    return OCRBenchAdapter(
        benchmark_meta=BenchmarkMeta(
            name='ocr_bench',
            dataset_id='dummy',
            eval_split='test',
            prompt_template='{question}',
        ),
        task_config=TaskConfig(datasets=['ocr_bench']),
    )


def test_ocr_bench_uses_image_first_content_order() -> None:
    adapter = make_ocr_bench_adapter()

    sample = adapter.record_to_sample({
        'question': 'Read the image.',
        'image': {
            'bytes': b'image'
        },
        'answer': ['text'],
        'dataset': 'IIIT5K',
        'question_type': 'Regular Text Recognition',
    })

    assert [content.type for content in sample.input[0].content] == ['image', 'text']


def test_ocr_bench_hme_scoring_preserves_prediction_case() -> None:
    adapter = make_ocr_bench_adapter()
    sample = Sample(input='question', target='', metadata={'dataset': 'HME100k'})
    task_state = TaskState(model='model', sample=sample)
    prediction = r'$$ 2 S_{3} = 5 S_{1} + 2 S_{2} $$'
    reference = json.dumps([r'2 S _ { 3 } = 5 S _ { 1 } + 2 S _ { 2 }'])

    score = adapter.match_score(prediction, prediction, reference, task_state)
    mismatched_score = adapter.match_score(prediction.lower(), prediction.lower(), reference, task_state)

    assert score.value == {'acc': 1}
    assert mismatched_score.value == {'acc': 0}


def test_ocr_bench_non_hme_scoring_remains_case_insensitive() -> None:
    adapter = make_ocr_bench_adapter()
    sample = Sample(input='question', target='', metadata={'dataset': 'IIIT5K'})
    task_state = TaskState(model='model', sample=sample)

    score = adapter.match_score('Centre', 'Centre', json.dumps(['CENTRE']), task_state)

    assert score.value == {'acc': 1}


def test_omnidoc_bench_uses_image_first_content_order() -> None:
    benchmark_meta = BenchmarkMeta(
        name='omni_doc_bench',
        dataset_id='dummy',
        eval_split='train',
        prompt_template='Parse this document.',
    )
    with patch('evalscope.benchmarks.omnidoc_bench.omnidoc_bench_adapter.check_import'):
        adapter = OmniDocBenchAdapter(
            benchmark_meta=benchmark_meta,
            task_config=TaskConfig(datasets=['omni_doc_bench']),
        )

    sample = adapter.record_to_sample({'image': 'aW1hZ2U=', 'answer': '{}'})

    assert [content.type for content in sample.input[0].content] == ['image', 'text']
