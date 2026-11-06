"""Exercise WenetSpeech's registered rule-scoring path without audio or model I/O."""

import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig
from evalscope.constants import ScoreStatus


@pytest.mark.parametrize(
    ('prediction', 'expected'),
    [('hello 世界', 0.0), ('hello，世界', 0.0), ('hello世介', 1 / 3)],
)
def test_wenet_speech_scores_mixed_script_transcripts(prediction: str, expected: float) -> None:
    adapter = get_benchmark(
        'wenet_speech', TaskConfig(datasets=['wenet_speech'], judge={'strategy': 'rule'})
    )
    state = TaskState(
        model='offline',
        sample=Sample(id=0, input='Transcribe the audio.', target='hello世界'),
        output=ModelOutput.from_content('offline', prediction),
        completed=True,
    )

    sample_score = adapter.calculate_metrics(state)

    assert adapter.metric_list == ['mer']
    assert sample_score.score.status is ScoreStatus.SUCCESS
    assert sample_score.score.extracted_prediction == prediction
    assert sample_score.score.value == pytest.approx({'mer': expected})
    aggregate = adapter.aggregate_scores([sample_score])[0]
    assert aggregate.metric_name == 'mer'
    assert aggregate.num == 1
    assert aggregate.score == pytest.approx(expected)


def test_wenet_speech_versions_the_scoring_change() -> None:
    assert get_benchmark('wenet_speech').benchmark_meta.evaluation_version == 'v1.1'
