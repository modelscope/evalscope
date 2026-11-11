"""CLUEWSC maps FewCLUE true/false labels to A/B and scores like OpenCompass FewCLUE_cluewsc_gen."""
import json
from pathlib import Path

import pytest

from evalscope.api.benchmark import DataAdapter
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig

RECORD = {
    'target': {'span2_index': 40, 'span1_index': 6, 'span1_text': '舌头', 'span2_text': '它'},
    'idx': 0,
    'text': '刀光一闪，那舌头象一只受惊的老鼠从受刑人的嘴巴和行刑人的手之间跳出来，看那样子，它是想往天上去的。',
}


def _adapter() -> DataAdapter:
    return get_benchmark('cluewsc', TaskConfig(model='mock', datasets=['cluewsc']))


@pytest.mark.parametrize(('label', 'target'), [('true', 'A'), ('false', 'B')])
def test_record_to_sample(label: str, target: str) -> None:
    sample = _adapter().record_to_sample({**RECORD, 'label': label})
    assert sample.target == target
    assert sample.choices == ['是', '否']
    assert sample.input.endswith('此处，“它”是否指代“舌头“？')


def test_offline_run_scores_local_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evalscope import run_task
    from evalscope.models.mockllm import MockLLM

    data = tmp_path / 'cluewsc'
    data.mkdir()
    (data / 'README.md').write_text(
        '---\nconfigs:\n- config_name: default\n  data_files:\n  - split: test_public\n    path: test_public.jsonl\n---\n'
    )
    rows = [{**RECORD, 'idx': i, 'label': label} for i, label in enumerate(['true', 'false'])]
    (data / 'test_public.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows))
    monkeypatch.setattr(MockLLM, 'default_output', 'A. 是')

    work = tmp_path / 'run'
    run_task(
        TaskConfig(
            model='offline-cluewsc',
            eval_type='mock_llm',
            datasets=['cluewsc'],
            dataset_args={'cluewsc': {'local_path': str(data)}},
            no_timestamp=True,
            work_dir=str(work),
        )
    )

    report = json.loads((work / 'reports/offline-cluewsc/cluewsc.json').read_text())
    metric = report['metrics'][0]
    assert (metric['score'], metric['num']) == (0.5, 2)


@pytest.mark.parametrize(('completion', 'answer'), [('A. 是', 'A'), ('答：B', 'B'), ('答案：B', 'B'), ('否', '')])
def test_extracts_first_capital_like_opencompass(completion: str, answer: str) -> None:
    adapter = _adapter()
    sample = adapter.record_to_sample({**RECORD, 'label': 'false'})
    state = TaskState(model='mock', sample=sample, output=ModelOutput.from_content(model='mock', content=completion))
    assert adapter.extract_answer(completion, state) == answer
