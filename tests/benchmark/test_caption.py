import json
from typing import Dict, List, Optional

from evalscope.api.benchmark import DataAdapter
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ContentVideo
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig
from evalscope.constants import HubType
from evalscope.models.utils.openai import openai_chat_completion_part


def _adapter(name: str, dataset_args: Optional[Dict] = None) -> DataAdapter:
    config = TaskConfig(datasets=[name], dataset_args={name: dataset_args or {}})
    return get_benchmark(name, config=config)


def test_msvd_record_to_sample() -> None:
    adapter = _adapter('msvd')
    records = adapter._group_records([
        {'video_id': 'video-1', 'video': 'video-1.mp4', 'caption': 'a man is cooking'},
        {'video_id': 'video-1', 'video': 'video-1.mp4', 'caption': 'a person cooks food'},
    ])

    assert len(records) == 1
    sample = adapter.record_to_sample(records[0])
    assert json.loads(sample.target) == ['a man is cooking', 'a person cooks food']
    assert sample.metadata['references'] == ['a man is cooking', 'a person cooks food']
    assert isinstance(sample.input[0].content[1], ContentVideo)
    assert sample.input[0].content[1].video == 'video-1.mp4'


def test_msvd_handles_list_caption_field() -> None:
    adapter = _adapter('msvd')
    records = adapter._group_records([
        {'video_id': 'video-1', 'video': 'video-1.mp4', 'caption': ['a man is cooking', 'a person cooks food']},
    ])

    assert len(records) == 1
    assert records[0]['references'] == ['a man is cooking', 'a person cooks food']


def test_msr_vtt_groups_by_video_id_and_prefers_url() -> None:
    adapter = _adapter('msr_vtt')
    records = adapter._group_records([
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


def test_msr_vtt_source_defaults_and_overrides() -> None:
    msr_vtt = _adapter('msr_vtt')
    assert msr_vtt.source_dataset_hub == HubType.MODELSCOPE
    assert msr_vtt.source_dataset_id == 'AI-ModelScope/msr-vtt'
    assert msr_vtt.source_eval_split == 'validation'

    msr_vtt_hf = _adapter('msr_vtt', {'extra_params': {'dataset_hub': HubType.HUGGINGFACE}})
    assert msr_vtt_hf.source_dataset_hub == HubType.HUGGINGFACE
    assert msr_vtt_hf.source_dataset_id == 'VLM2Vec/MSR-VTT'
    assert msr_vtt_hf.source_eval_split == 'test'

    msr_vtt_custom = _adapter('msr_vtt', {'dataset_id': 'my-org/msr-vtt'})
    assert msr_vtt_custom.source_dataset_id == 'my-org/msr-vtt'

    msvd = _adapter('msvd')
    assert msvd.source_dataset_hub == HubType.MODELSCOPE
    assert msvd.source_dataset_id == 'evalscope/MSVD'


def test_video_content_fps_passthrough() -> None:
    adapter = _adapter('msvd')
    records = adapter._group_records([
        {'video_id': 'v1', 'video': 'https://example.com/v1.mp4', 'caption': 'a person talks', 'fps': 2},
    ])
    sample = adapter.record_to_sample(records[0])

    video = sample.input[0].content[1]
    assert isinstance(video, ContentVideo)
    assert video.fps == 2

    payload = openai_chat_completion_part(video)
    assert payload['video_url']['url'] == 'https://example.com/v1.mp4'
    assert payload['fps'] == 2


def test_caption_batch_scoring(monkeypatch) -> None:
    from evalscope.benchmarks.msvd import msvd_adapter

    def fake_scores(predictions: List[str], references: List[List[str]]) -> List[Dict[str, float]]:
        return [{
            'Bleu_1': 0.5, 'Bleu_2': 0.4, 'Bleu_3': 0.3, 'Bleu_4': 0.2,
            'METEOR': 0.6, 'ROUGE_L': 0.7, 'CIDEr': 1.2,
        }]

    monkeypatch.setattr(msvd_adapter, 'compute_caption_scores', fake_scores)
    adapter = _adapter('msvd')
    records = adapter._group_records([
        {'video_id': 'v1', 'video': 'v1.mp4', 'caption': 'a man is cooking'},
    ])
    sample = adapter.record_to_sample(records[0])
    task_state = TaskState(
        model='mock',
        sample=sample,
        output=ModelOutput.from_content(model='mock', content='a man cooks'),
        completed=True,
    )

    sample_score = adapter.calculate_metrics(task_state)
    updated = adapter.batch_calculate_metrics([task_state], [sample_score])

    assert updated[0].score.value['CIDEr'] == 1.2
    assert updated[0].score.main_score_name == 'CIDEr'
