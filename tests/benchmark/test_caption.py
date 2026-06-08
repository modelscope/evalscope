import json
from pathlib import Path
from typing import Dict, List, Optional

from evalscope.api.benchmark import DataAdapter
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ContentImage, ContentVideo
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.caption.base import ensure_text_list, vqa_soft_accuracy
from evalscope.config import TaskConfig
from evalscope.constants import HubType
from evalscope.models.utils.openai import openai_chat_completion_part


def _adapter(name: str, dataset_args: Optional[Dict] = None) -> DataAdapter:
    config = TaskConfig(datasets=[name], dataset_args={name: dataset_args or {}})
    return get_benchmark(name, config=config)


def test_msvd_record_to_sample_uses_caption_references() -> None:
    adapter = _adapter('msvd')

    sample = adapter.record_to_sample({
        'video_id': 'video-1',
        'video': 'video-1.mp4',
        'caption': ['a man is cooking', 'a person cooks food'],
        'source': 'MSVD',
    })

    assert json.loads(sample.target) == ['a man is cooking', 'a person cooks food']
    assert sample.metadata['references'] == ['a man is cooking', 'a person cooks food']
    assert isinstance(sample.input[0].content[1], ContentVideo)
    assert sample.input[0].content[1].video == 'video-1.mp4'


def test_msr_vtt_groups_rows_and_prefers_resolved_url() -> None:
    adapter = _adapter('msr_vtt')
    records = adapter._prepare_records([
        {
            'video_id': 'video7020',
            'video': 'video7020.mp4',
            'caption': 'a woman makes a fondant flower',
            'url': 'https://example.com/video7020.mp4',
        },
        {
            'video_id': 'video7020',
            'video': 'video7020.mp4',
            'caption': 'a person creates cake decorations',
            'url': 'https://example.com/video7020.mp4',
        },
    ])

    assert len(records) == 1
    assert records[0]['references'] == ['a woman makes a fondant flower', 'a person creates cake decorations']

    sample = adapter.record_to_sample(records[0])
    assert isinstance(sample.input[0].content[1], ContentVideo)
    assert sample.input[0].content[1].video == 'https://example.com/video7020.mp4'


def test_video_content_preserves_fps_for_openai_payload() -> None:
    adapter = _adapter('msvd')
    sample = adapter.record_to_sample({
        'video_id': 'video-1',
        'video': 'https://example.com/video-1.mp4',
        'caption': ['a person talks'],
        'fps': 2,
    })

    video = sample.input[0].content[1]
    assert isinstance(video, ContentVideo)
    assert video.fps == 2

    payload = openai_chat_completion_part(video)
    assert payload['video_url']['url'] == 'https://example.com/video-1.mp4'
    assert payload['fps'] == 2


def test_caption_dataset_source_defaults_and_overrides() -> None:
    msr_vtt = _adapter('msr_vtt')
    assert msr_vtt.source_dataset_hub == HubType.MODELSCOPE
    assert msr_vtt.source_dataset_id == 'AI-ModelScope/msr-vtt'
    assert msr_vtt.source_eval_split == 'validation'
    assert msr_vtt._source_subset_name('default') is None

    msr_vtt_named = _adapter('msr_vtt', {'dataset_id': 'msr_vtt'})
    assert msr_vtt_named.source_dataset_id == 'AI-ModelScope/msr-vtt'

    msr_vtt_hf = _adapter('msr_vtt', {'extra_params': {'dataset_hub': HubType.HUGGINGFACE}})
    assert msr_vtt_hf.source_dataset_hub == HubType.HUGGINGFACE
    assert msr_vtt_hf.source_dataset_id == 'VLM2Vec/MSR-VTT'
    assert msr_vtt_hf.source_eval_split == 'test'
    assert msr_vtt_hf._source_subset_name('default') == 'test_1k'

    msr_vtt_custom = _adapter('msr_vtt', {'dataset_id': 'local-msr-vtt'})
    assert msr_vtt_custom.source_dataset_id == 'local-msr-vtt'

    vqav2 = _adapter('vqav2')
    assert vqav2.source_dataset_hub == HubType.MODELSCOPE

    msvd = _adapter('msvd')
    assert msvd.source_dataset_hub == HubType.HUGGINGFACE


def test_ensure_text_list_handles_bytes_as_text() -> None:
    assert ensure_text_list(b'hello') == ['hello']
    assert ensure_text_list([b'hello', 'world']) == ['hello', 'world']


def test_local_directory_loader_uses_split_only_file_when_subset_is_empty(tmp_path: Path) -> None:
    data_path = tmp_path / 'test.jsonl'
    data_path.write_text(
        json.dumps({
            'video_id': 'video-1',
            'video': 'video-1.mp4',
            'caption': 'a person talks',
        }) + '\n',
        encoding='utf-8',
    )
    adapter = _adapter('msvd', {
        'local_path': str(tmp_path),
        'extra_params': {
            'dataset_hub': HubType.LOCAL,
        },
    })

    records = adapter._load_local_records(None)

    assert len(records) == 1
    assert records[0]['caption'] == 'a person talks'


def test_caption_batch_scoring_updates_sample_scores(monkeypatch) -> None:
    from evalscope.benchmarks.caption import base as caption_base

    def fake_caption_scores(predictions: List[str], references: List[List[str]]) -> List[Dict[str, float]]:
        assert predictions == ['a man cooks']
        assert references == [['a man is cooking']]
        return [{
            'Bleu_1': 0.5,
            'Bleu_2': 0.4,
            'Bleu_3': 0.3,
            'Bleu_4': 0.2,
            'METEOR': 0.6,
            'ROUGE_L': 0.7,
            'CIDEr': 1.2,
        }]

    monkeypatch.setattr(caption_base, 'compute_caption_scores', fake_caption_scores)
    adapter = _adapter('msvd')
    sample = adapter.record_to_sample({
        'video_id': 'video-1',
        'video': 'video-1.mp4',
        'caption': ['a man is cooking'],
    })
    task_state = TaskState(
        model='mock',
        sample=sample,
        output=ModelOutput.from_content(model='mock', content='a man cooks'),
        completed=True,
    )

    sample_score = adapter.calculate_metrics(task_state)
    updated_scores = adapter.batch_calculate_metrics([task_state], [sample_score])

    assert updated_scores[0].score.value['CIDEr'] == 1.2
    assert updated_scores[0].score.main_score_name == 'CIDEr'


def test_vqav2_soft_accuracy_and_answer_extraction() -> None:
    adapter = _adapter('vqav2')
    sample = adapter.record_to_sample({
        'question': 'What animal is shown?',
        'image': {'bytes': b'image-bytes'},
        'answers': ['cat', 'cat', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog'],
        'multiple_choice_answer': 'dog',
        'question_id': 7,
    })
    task_state = TaskState(
        model='mock',
        sample=sample,
        output=ModelOutput.from_content(model='mock', content='ANSWER: cat'),
        completed=True,
    )

    score = adapter.calculate_metrics(task_state).score

    assert isinstance(sample.input[0].content[1], ContentImage)
    assert score.extracted_prediction == 'cat'
    assert abs(score.value['vqa_score'] - 2 / 3) < 1e-6
    assert score.value['exact_match'] == 1.0
    assert score.main_score_name == 'vqa_score'
    assert vqa_soft_accuracy('cat', ['cat', 'cat', 'cat', 'dog']) == 1.0
