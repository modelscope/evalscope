import json
from unittest import mock

import pytest

pytest.importorskip('torch')
pytest.importorskip('langchain_core')

from evalscope.backend.rag_eval.clip_benchmark import task_template  # noqa: E402
from evalscope.backend.rag_eval.clip_benchmark.arguments import ClipBenchmarkEvalConfig  # noqa: E402


def test_skip_existing_skips_only_the_finished_dataset(tmp_path):
    (tmp_path / 'clip').mkdir()
    (tmp_path / 'clip' / 'muge_zeroshot_retrieval.json').write_text(json.dumps({'metrics': {}}))
    args = ClipBenchmarkEvalConfig(
        models=[{'model_name': 'org/clip'}],
        dataset_name=['muge', 'flickr8k', 'flickr30k'],
        output_dir=str(tmp_path),
        skip_existing=True,
        verbose=False,
    )

    with (
        mock.patch.object(task_template.VisionModel, 'load', return_value=mock.Mock(transform=None)),
        mock.patch.object(task_template, 'build_dataset', return_value=None),
        mock.patch.object(task_template, 'get_dataloader', side_effect=lambda name, *args: name),
        mock.patch.object(task_template.zeroshot_retrieval, 'evaluate', return_value={'recall@5': 1.0}) as evaluate,
    ):
        task_template.evaluate(args)

    assert [call.args[1] for call in evaluate.call_args_list] == ['flickr8k', 'flickr30k']
    assert sorted(path.name for path in (tmp_path / 'clip').iterdir()) == [
        'flickr30k_zeroshot_retrieval.json',
        'flickr8k_zeroshot_retrieval.json',
        'muge_zeroshot_retrieval.json',
    ]
