"""Generated artifacts must survive repeated inference and prediction-cache replay."""

import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.benchmark.adapters.text2image_adapter import Text2ImageAdapter
from evalscope.api.dataset import MemoryDataset, Sample
from evalscope.api.dataset.loader import repeat_samples
from evalscope.api.evaluator.cache import ModelResult
from evalscope.api.messages import ChatMessageUser, ContentAudio, ContentImage
from evalscope.api.model import ChatCompletionChoice, ModelOutput
from evalscope.benchmarks.seed_tts_eval.seed_tts_eval_adapter import SeedTTSEvalAdapter
from evalscope.config import TaskConfig


@pytest.mark.parametrize('audio', [False, True])
@pytest.mark.parametrize('workers', [1, 8])
@pytest.mark.parametrize('source_id', ['shared-record', None])
def test_artifacts_survive_repeats_and_cache_replay(tmp_path: Path, audio: bool, workers: int,
                                                  source_id: str | None) -> None:
    name = 'seed_tts_eval' if audio else 'general_t2i'
    cls = SeedTTSEvalAdapter if audio else Text2ImageAdapter
    adapter = cls(benchmark_meta=BenchmarkMeta(name=name, dataset_id=name, metric_list=[]),
                  task_config=TaskConfig(model='mock', datasets=[name], repeats=8,
                                         eval_type='mock_llm' if audio else 'text2image'))
    sample = Sample(input=[ChatMessageUser(content='prompt')], target='target',
                    metadata={'filename': source_id, 'id': source_id})
    dataset = MemoryDataset(repeat_samples([sample], 8))
    dataset.reindex(group_size=8)
    replay_dataset = MemoryDataset([item.model_copy(deep=True) for item in dataset])
    barrier = Barrier(workers)
    key = 'generated_audio_path' if audio else 'image_path'

    def save(index: int) -> ModelResult:
        # Byte sentinels test artifact identity, without an ASR or image-model dependency.
        encoded = base64.b64encode(f'output-{index}'.encode()).decode()
        content = (ContentAudio(audio=encoded, format='wav') if audio else
                   ContentImage(image='data:image/png;base64,' + encoded))
        output = ModelOutput(model='mock', choices=[ChatCompletionChoice.from_content([content])])
        barrier.wait(timeout=10)
        state = adapter._on_inference_end(SimpleNamespace(name='mock'), dataset[index], output, str(tmp_path))
        return ModelResult.model_validate_json(ModelResult.from_task_state(state).model_dump_json())

    with ThreadPoolExecutor(max_workers=workers) as pool:
        cached = list(pool.map(save, range(8)))
    paths = [item.to_task_state(replay_dataset).metadata[key] for item in cached]
    assert len(set(paths)) == 8
    for index, path in enumerate(paths):
        assert Path(path).read_bytes() == f'output-{index}'.encode()
    # A new inference with exactly the same IDs (another subset/run) cannot mutate cached artifacts.
    barrier = Barrier(1)
    newer = save(0).to_task_state(dataset).metadata[key]
    assert newer not in paths
    assert all(Path(path).read_bytes() == f'output-{index}'.encode() for index, path in enumerate(paths))
