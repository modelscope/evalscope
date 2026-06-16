import base64
import numpy as np
import pytest
from pathlib import Path
from PIL import Image
from typing import Any, List, Optional, Tuple

from evalscope.api.benchmark import BenchmarkMeta, Text2ImageAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser
from evalscope.config import TaskConfig
from evalscope.constants import EvalType, FileConstants
from evalscope.metrics.metric import PSNRScore, SSIMScore


def save_image(path: Path, color: Tuple[int, int, int]) -> None:
    Image.new('RGB', (16, 16), color=color).save(path)


def make_images(
    tmp_path: Path,
    prediction_color: Tuple[int, int, int] = (255, 0, 0),
    reference_color: Tuple[int, int, int] = (255, 0, 0),
) -> Tuple[Path, Path]:
    prediction_path = tmp_path / 'prediction.png'
    reference_path = tmp_path / 'reference.png'
    save_image(prediction_path, prediction_color)
    save_image(reference_path, reference_color)
    return prediction_path, reference_path


def make_adapter(metric_list: List[Any]) -> Text2ImageAdapter:
    return Text2ImageAdapter(
        benchmark_meta=BenchmarkMeta(
            name='image_quality_test',
            dataset_id='image_quality_test',
            metric_list=metric_list,
        ),
        task_config=TaskConfig(
            model='mock',
            datasets=['image_quality_test'],
            eval_type=EvalType.MOCK_LLM,
        ),
    )


def make_state(prediction_path: Path, reference_path: Optional[Path] = None) -> TaskState:
    metadata = {
        FileConstants.IMAGE_PATH: str(prediction_path),
    }
    if reference_path:
        metadata['reference_image_path'] = str(reference_path)
    sample = Sample(input=[ChatMessageUser(content='a red square')], metadata=metadata)
    return TaskState(model='mock', sample=sample, completed=True)


def test_image_pair_metrics_identical_images(tmp_path: Path) -> None:
    prediction_path, reference_path = make_images(tmp_path)

    assert SSIMScore()(str(prediction_path), str(reference_path)) == 1.0
    assert PSNRScore()(str(prediction_path), str(reference_path)) == 100.0


def test_image_pair_metrics_constant_opposites(tmp_path: Path) -> None:
    prediction_path, reference_path = make_images(tmp_path, (0, 0, 0), (255, 255, 255))

    c1 = 0.01**2
    assert PSNRScore()(str(prediction_path), str(reference_path)) == 0.0
    assert abs(SSIMScore()(str(prediction_path), str(reference_path)) - c1 / (1 + c1)) < 1e-12


def test_image_pair_metric_rejects_invalid_params() -> None:
    with pytest.raises(ValueError, match='data_range'):
        PSNRScore(data_range=0)

    with pytest.raises(ValueError, match='window_size'):
        SSIMScore(window_size=2)


def test_image_pair_metric_loads_long_base64_image(tmp_path: Path) -> None:
    image_path = tmp_path / 'large.png'
    Image.new('RGB', (256, 256), color=(1, 2, 3)).save(image_path)
    encoded_image = base64.b64encode(image_path.read_bytes()).decode('ascii')
    wrapped_image = '\n'.join(encoded_image[index:index + 76] for index in range(0, len(encoded_image), 76))

    assert len(encoded_image) > 255
    assert SSIMScore()(wrapped_image, wrapped_image) == 1.0


def test_text2image_adapter_scores_image_pair_metrics(tmp_path: Path) -> None:
    prediction_path, reference_path = make_images(tmp_path)
    adapter = make_adapter(['ssim', 'psnr'])
    state = make_state(prediction_path, reference_path)

    score = adapter.match_score(str(prediction_path), str(prediction_path), '', state)

    assert score.value['ssim'] == 1.0
    assert score.value['psnr'] == 100.0


def test_text2image_adapter_scores_with_missing_metadata(tmp_path: Path) -> None:
    prediction_path, reference_path = make_images(tmp_path)
    adapter = make_adapter(['ssim'])
    state = make_state(prediction_path)
    state.metadata = None

    score = adapter.match_score(str(prediction_path), str(prediction_path), str(reference_path), state)

    assert score.value['ssim'] == 1.0


def test_text2image_adapter_requires_reference_for_image_pair_metric(tmp_path: Path) -> None:
    prediction_path = tmp_path / 'prediction.png'
    save_image(prediction_path, (255, 0, 0))
    adapter = make_adapter(['ssim'])
    state = make_state(prediction_path)

    score = adapter.match_score(str(prediction_path), str(prediction_path), '', state)

    assert score.value['ssim'] == 0
    assert 'requires a reference image' in score.metadata['ssim']


def test_text2image_record_keeps_non_string_reference_in_metadata() -> None:
    adapter = make_adapter(['ssim'])
    reference_image = {'bytes': b'image-bytes'}

    sample = adapter.record_to_sample({
        'prompt': 'a red square',
        FileConstants.IMAGE_PATH: '/tmp/prediction.png',
        'reference_image': reference_image,
    })

    assert sample.target == ''
    assert sample.metadata['reference_image'] == reference_image


def test_text2image_adapter_scores_numpy_reference(tmp_path: Path) -> None:
    prediction_path, _ = make_images(tmp_path)
    adapter = make_adapter(['ssim'])
    state = make_state(prediction_path)
    state.metadata['reference_image'] = np.asarray(Image.open(prediction_path))

    score = adapter.match_score(str(prediction_path), str(prediction_path), '', state)

    assert score.value['ssim'] == 1.0
