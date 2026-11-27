import sys

import pytest

from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig


def test_grounding_scores_without_caption_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, 'pycocoevalcap', None)
    monkeypatch.setitem(sys.modules, 'pycocotools', None)
    config = TaskConfig(datasets=['refcoco'], dataset_args={'refcoco': {'extra_params': {'eval_mode': 'bbox_rec'}}})
    adapter = get_benchmark('refcoco', config)
    box = [0.1, 0.2, 0.6, 0.8]
    sample = Sample(input='locate the object', target=str(box), metadata={'bbox': box, 'eval_mode': 'bbox_rec'})
    state = TaskState(model='mock', sample=sample, output=ModelOutput.from_content('mock', str(box)), completed=True)
    score = adapter.calculate_metrics(state).score
    assert score.value['IoU'] == 1.0
    assert score.value['ACC@0.5'] == 1.0


@pytest.mark.parametrize('mode', ['bbox', 'seg'])
def test_captioning_requires_the_refcoco_extra(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, 'pycocoevalcap', None)
    config = TaskConfig(datasets=['refcoco'], dataset_args={'refcoco': {'extra_params': {'eval_mode': mode}}})
    with pytest.raises(ImportError, match=r'evalscope\[refcoco\]'):
        get_benchmark('refcoco', config)
