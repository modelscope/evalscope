"""CrossSum-IN: subset/file mapping, the paper's prompt and corpus chrF with English reported apart from Indic."""

import json
from pathlib import Path
from typing import Any

import pytest
import sacrebleu

from evalscope.api.benchmark import DataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import SampleScore
from evalscope.api.model import ModelOutput
from evalscope.api.registry import get_benchmark
from evalscope.benchmarks.indicgenbench.crosssum_in_adapter import SUBSET_LIST
from evalscope.config import TaskConfig
from evalscope.report import ReportGenerator

PREFIX = 'I will first show a news article in English and then provide a summary of it in the'


def _adapter(**dataset_args: Any) -> DataAdapter:
    args = {'crosssum_in': dataset_args} if dataset_args else {}
    return get_benchmark('crosssum_in', TaskConfig(model='mock', datasets=['crosssum_in'], dataset_args=args))


def _record(text: str, summary: str, lang: str) -> dict:
    return {'text': text, 'summary': summary, 'lang': lang, 'source_url': 's', 'target_url': 't'}


def test_subsets() -> None:
    assert len(SUBSET_LIST) == len(set(SUBSET_LIST)) == 30
    assert SUBSET_LIST[0] == 'en'
    adapter = _adapter()
    assert adapter.subset_file('hi', 'test') == 'crosssum_english-hi_test.json'
    assert adapter.subset_file('en', 'train') == 'crosssum_english-en_train.json'
    assert adapter.few_shot_num == 1
    assert adapter.train_split == 'train' and adapter.eval_split == 'test'
    assert adapter.category_map['en'] == 'english'
    assert {adapter.category_map[s] for s in SUBSET_LIST[1:]} == {'indic'}


@pytest.mark.parametrize('subset', ['xx', 'en-hi', 'english-hi'])
def test_unknown_subset_is_rejected(subset: str) -> None:
    with pytest.raises(ValueError, match='Unknown CrossSum-IN subset'):
        _adapter().subset_file(subset, 'test')


def test_prompts() -> None:
    adapter = _adapter()
    example = adapter.record_to_sample(_record(' Rain fell. ', ' बारिश हुई। ', 'hi'))
    test = adapter.record_to_sample(_record('The match ended.', 'मैच ख़त्म हुआ।', 'hi'))
    assert example.input == 'Rain fell.' and example.target == 'बारिश हुई।'

    prompt = f'{PREFIX} Hindi language.\n\nSummarize the following article: The match ended.\nSummary:'
    assert adapter.format_prompt_template(test) == prompt
    fewshot = adapter.sample_to_fewshot(example)
    assert fewshot == f'{PREFIX} Hindi language.\n\nSummarize the following article: Rain fell.\nSummary: बारिश हुई।'
    assert adapter.format_fewshot_template(fewshot, test) == f'{fewshot}\n\n{prompt}'

    english = adapter.record_to_sample(_record('The match ended.', 'It ended.', 'en'))
    assert adapter.format_prompt_template(english).startswith(f'{PREFIX} English language.')


def _scores(adapter: DataAdapter, pairs: list[tuple[str, str]]) -> list[SampleScore]:
    scores = []
    for index, (reference, completion) in enumerate(pairs):
        state = TaskState(
            model='mock',
            sample=Sample(id=index, input='q', target=reference),
            output=ModelOutput.from_content('mock', completion),
            completed=True,
        )
        scores.append(adapter.calculate_metrics(state))
    return scores


FIXED = {
    'en': [('A storm hit the coast.', 'A storm hit the coast on Monday.\nIt was strong.')],
    'hi': [('तूफ़ान तट से टकराया।', 'तूफ़ान तट पर आया।'), ('चुनाव में भारी मतदान हुआ।', '')],
    'ta': [('புயல் கரையைத் தாக்கியது.', 'புயல் கரையைத் தாக்கியது.'), ('தேர்தல் நடந்தது.', 'தேர்தல்.')],
    'sat': [('ᱫᱟᱜ ᱫᱟᱨᱟ', 'ᱫᱟᱜ')],
}


def test_corpus_chrf_and_indic_macro_average() -> None:
    adapter = _adapter()
    subset_scores = {}
    for subset, pairs in FIXED.items():
        (agg,) = adapter.aggregate_scores(_scores(adapter, pairs))
        hypotheses = [completion.strip().split('\n')[0].strip() for _, completion in pairs]
        expected = sacrebleu.corpus_chrf(hypotheses, [[reference for reference, _ in pairs]]).score / 100
        assert agg.metric_name == 'chrf' and agg.aggregation == 'official'
        assert agg.score == expected
        subset_scores[subset] = [agg]

    report = ReportGenerator.generate_report(score_dict=subset_scores, model_name='mock', data_adapter=adapter)
    (metric,) = report.metrics
    categories = {category.name[0]: category for category in metric.categories}
    # The paper's Indic average excludes English and weighs every language equally.
    indic = [subset_scores[s][0].score for s in ('hi', 'ta', 'sat')]
    assert categories['indic'].macro_score == pytest.approx(sum(indic) / 3, abs=1e-4)
    assert categories['english'].macro_score == pytest.approx(subset_scores['en'][0].score, abs=1e-4)


def _write(directory: Path, name: str, records: list) -> None:
    (directory / name).write_text(json.dumps({'examples': records, 'canary': 'x'}), encoding='utf-8')


def test_load_local_files(tmp_path: Path) -> None:
    _write(tmp_path, 'crosssum_english-hi_test.json', [_record(f'article {i}', f'सार {i}', 'hi') for i in range(3)])
    _write(tmp_path, 'crosssum_english-hi_train.json', [_record(f'train {i}', f'प्रशिक्षण {i}', 'hi') for i in range(3)])
    test, fewshot = _adapter(dataset_id=str(tmp_path), subset_list=['hi']).load()
    assert [s.input for s in test['hi']] == ['article 0', 'article 1', 'article 2']
    # The one-shot example is the first `train` row of the same language.
    assert [s.input for s in fewshot['hi']] == ['train 0']


@pytest.mark.parametrize(
    ('record', 'message'),
    [(_record('a', 'b', 'mr'), 'expected `hi`'), (_record(' ', 'b', 'hi'), 'empty article or summary')],
)
def test_load_rejects_bad_rows(tmp_path: Path, record: dict, message: str) -> None:
    _write(tmp_path, 'crosssum_english-hi_test.json', [record])
    adapter = _adapter(dataset_id=str(tmp_path), subset_list=['hi'], few_shot_num=0)
    with pytest.raises(ValueError, match=message):
        adapter.load()
