from typing import Any

import evalscope  # noqa: F401  # imported for benchmark and metric registration side effects

from evalscope.api.messages import ChatMessageUser, ContentAudio, ContentText
from evalscope.api.registry import get_benchmark, get_metric
from evalscope.config import TaskConfig


def test_thchs30_registered_with_phone_error_rate() -> None:
    adapter = get_benchmark('thchs30', TaskConfig(datasets=['thchs30']))

    assert adapter.dataset_id == 'evalscope/THCHS-30'
    assert adapter.eval_split == 'test'
    assert adapter.metric_list == ['per']
    assert adapter.few_shot_num == 0


def test_thchs30_record_to_sample_preserves_alignment_metadata() -> None:
    adapter = get_benchmark('thchs30', TaskConfig(datasets=['thchs30']))
    record: dict[str, Any] = {
        'utt_id': 'A11_0',
        'audio': {'bytes': b'RIFF....WAVE'},
        'text': '你好',
        'phones': ['n', 'i˨˩', 'x', 'au˨˩˦'],
        'phone_starts': [0.0, 0.1, 0.3, 0.4],
        'phone_ends': [0.1, 0.3, 0.4, 0.7],
        'language': 'cmn',
        'speaker_id': 'A11',
        'duration': 0.7,
        'split': 'test',
    }

    sample = adapter.record_to_sample(record)

    assert sample.target == 'n i˨˩ x au˨˩˦'
    assert sample.metadata == {key: value for key, value in record.items() if key != 'audio'}
    assert isinstance(sample.input[0], ChatMessageUser)
    assert isinstance(sample.input[0].content[0], ContentText)
    assert isinstance(sample.input[0].content[1], ContentAudio)
    assert sample.input[0].content[1].audio.startswith('data:audio/wav;base64,')


def test_phone_error_rate_uses_phone_tokens() -> None:
    metric = get_metric('per')()

    assert metric('n i˨˩ x au˨˩˦', 'n i˨˩ x au˨˩˦') == 0.0
    assert metric('n x au˨˩˦', 'n i˨˩ x au˨˩˦') == 0.25
    assert metric('n', '') == 1.0
