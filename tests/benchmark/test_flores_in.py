"""FLORES-IN: subset/file mapping, the paper's prompt, first-line extraction and corpus chrF per direction."""

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
from evalscope.benchmarks.indicgenbench.flores_in_adapter import LANGUAGE_NAMES, SUBSET_LIST, subset_direction
from evalscope.benchmarks.indicgenbench.utils import first_line
from evalscope.config import TaskConfig
from evalscope.report import ReportGenerator


def _adapter(**dataset_args: Any) -> DataAdapter:
    args = {'flores_in': dataset_args} if dataset_args else {}
    return get_benchmark('flores_in', TaskConfig(model='mock', datasets=['flores_in'], dataset_args=args))


def _record(source: str, target: str, lang: str, direction: str) -> dict:
    return {'source': source, 'target': target, 'lang': lang, 'translation_direction': direction}


def test_subsets() -> None:
    assert len(LANGUAGE_NAMES) == 29
    assert len(SUBSET_LIST) == len(set(SUBSET_LIST)) == 58
    assert SUBSET_LIST[0] == 'en-as' and SUBSET_LIST[-1] == 'ur-en'
    adapter = _adapter()
    assert adapter.subset_file('en-hi', 'test') == 'flores_en_hi_test.json'
    assert adapter.subset_file('sat-en', 'dev') == 'flores_sat_en_dev.json'
    assert subset_direction('en-hi') == 'en-xx'
    assert subset_direction('hi-en') == 'xx-en'
    assert adapter.few_shot_num == 1
    assert adapter.train_split == 'dev' and adapter.eval_split == 'test'
    assert sorted(set(adapter.category_map.values())) == ['en-xx', 'xx-en']


@pytest.mark.parametrize('subset', ['en-zz', 'hi-ta', 'hi', 'en-en'])
def test_unknown_subset_is_rejected(subset: str) -> None:
    with pytest.raises(ValueError, match='Unknown FLORES-IN subset'):
        _adapter().subset_file(subset, 'test')


@pytest.mark.parametrize(
    ('record', 'language'),
    [
        (_record('Hello.', 'नमस्ते।', 'hi', 'enxx'), 'Hindi'),
        (_record('ᱡᱚᱦᱟᱨ᱾', 'Hello.', 'sat', 'xxen'), 'English'),
        (_record('Hello.', 'ཀཀ', 'bo', 'enxx'), 'Tibetan'),
    ],
)
def test_record_to_sample(record: dict, language: str) -> None:
    sample = _adapter().record_to_sample(record)
    assert sample.input == record['source']
    assert sample.target == record['target']
    assert sample.metadata['target_language_name'] == language


def test_prompts() -> None:
    adapter = _adapter()
    example = adapter.record_to_sample(_record('Good morning.', 'सुप्रभात।', 'hi', 'enxx'))
    test = adapter.record_to_sample(_record('Thank you.', 'धन्यवाद।', 'hi', 'enxx'))

    assert adapter.format_prompt_template(test) == 'Translate the following:\n\nTo Hindi: Thank you.\nOutput:'
    fewshot = adapter.sample_to_fewshot(example)
    assert fewshot == 'Translate the following:\n\nTo Hindi: Good morning.\nOutput: सुप्रभात।'
    assert adapter.format_fewshot_template(fewshot, test) == (
        'Translate the following:\n\nTo Hindi: Good morning.\nOutput: सुप्रभात।\n\n'
        'Translate the following:\n\nTo Hindi: Thank you.\nOutput:'
    )

    reverse = adapter.record_to_sample(_record('धन्यवाद।', 'Thank you.', 'hi', 'xxen'))
    assert adapter.format_prompt_template(reverse) == 'Translate the following:\n\nTo English: धन्यवाद।\nOutput:'


def test_stray_spaces_are_stripped() -> None:
    adapter = _adapter()
    padded = adapter.record_to_sample(_record(' Good morning. ', ' सुप्रभात। ', 'hi', 'enxx'))
    assert padded.input == padded.metadata['source'] == 'Good morning.'
    assert padded.target == 'सुप्रभात।'
    assert adapter.sample_to_fewshot(padded) == 'Translate the following:\n\nTo Hindi: Good morning.\nOutput: सुप्रभात।'
    # chrF ignores whitespace, so stripping the reference does not change a score.
    assert sacrebleu.sentence_chrf('सुप्रभात', [' सुप्रभात। ']).score == sacrebleu.sentence_chrf('सुप्रभात', ['सुप्रभात।']).score


@pytest.mark.parametrize(
    ('completion', 'expected'),
    [
        ('धन्यवाद।', 'धन्यवाद।'),
        ('  धन्यवाद।\n\nTranslate the following:\n\nTo Hindi: ...', 'धन्यवाद।'),
        ('\n\n धन्यवाद। \nmore', 'धन्यवाद।'),
        ('', ''),
        ('\n \n', ''),
    ],
)
def test_first_line(completion: str, expected: str) -> None:
    assert first_line(completion) == expected


# Fixed translations per subset: corpus chrF pools character n-gram counts over the subset, so it differs from
# the mean of sentence scores, and an empty translation still counts its reference.
FIXED = {
    'en-hi': [
        ('यह एक परीक्षण वाक्य है।', 'यह एक परीक्षण वाक्य है।\nextra line'),
        ('मौसम आज बहुत अच्छा है।', 'आज मौसम अच्छा है।'),
        ('वह बाज़ार गया।', ''),
    ],
    'en-ta': [('இது ஒரு சோதனை.', 'இது சோதனை.'), ('நன்றி.', 'நன்றி.')],
    'hi-en': [('The weather is very nice today.', 'Today the weather is nice.'), ('He went home.', 'He went home.')],
    'sat-en': [('The old fittings are part of their character.', 'Old fittings are their character.')],
}


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


def test_corpus_chrf_matches_sacrebleu() -> None:
    adapter = _adapter()
    subset_scores = {}
    for subset, pairs in FIXED.items():
        scores = _scores(adapter, pairs)
        (agg,) = adapter.aggregate_scores(scores)
        hypotheses = [first_line(completion) for _, completion in pairs]
        references = [reference for reference, _ in pairs]
        expected = sacrebleu.corpus_chrf(hypotheses, [references]).score / 100
        assert agg.metric_name == 'chrf' and agg.aggregation == 'official'
        assert agg.score == expected
        assert agg.num == len(pairs) and agg.ids == list(range(len(pairs)))
        for score, hypothesis, reference in zip(scores, hypotheses, references):
            assert score.score.extracted_prediction == hypothesis
            assert score.score.value['chrf'] == sacrebleu.sentence_chrf(hypothesis, [reference]).score / 100
        subset_scores[subset] = [agg]

    en_hi = subset_scores['en-hi'][0].score
    sentence_mean = sum(s.score.value['chrf'] for s in _scores(adapter, FIXED['en-hi'])) / len(FIXED['en-hi'])
    assert en_hi != pytest.approx(sentence_mean)

    # Each direction is the macro-average of its subsets' corpus scores.
    report = ReportGenerator.generate_report(score_dict=subset_scores, model_name='mock', data_adapter=adapter)
    (metric,) = report.metrics
    categories = {category.name[0]: category for category in metric.categories}
    for direction, subsets in (('en-xx', ['en-hi', 'en-ta']), ('xx-en', ['hi-en', 'sat-en'])):
        expected = sum(subset_scores[s][0].score for s in subsets) / len(subsets)
        assert categories[direction].macro_score == pytest.approx(expected, abs=1e-4)
        assert sorted(subset.name for subset in categories[direction].subsets) == subsets


def test_empty_subset_has_no_score() -> None:
    assert _adapter().aggregate_scores([]) == []


def _write(directory: Path, name: str, records: list) -> None:
    (directory / name).write_text(json.dumps({'examples': records, 'canary': 'x'}), encoding='utf-8')


def test_load_local_files(tmp_path: Path) -> None:
    _write(tmp_path, 'flores_en_hi_test.json', [_record(f'test {i}', f'परीक्षण {i}', 'hi', 'enxx') for i in range(3)])
    _write(tmp_path, 'flores_en_hi_dev.json', [_record(f'dev {i}', f'विकास {i}', 'hi', 'enxx') for i in range(3)])
    _write(tmp_path, 'flores_sat_en_test.json', [_record('ᱡᱚᱦᱟᱨ', 'hello', 'sat', 'xxen')])
    _write(tmp_path, 'flores_sat_en_dev.json', [_record('ᱥᱟᱨᱦᱟᱣ', 'thanks', 'sat', 'xxen')])

    adapter = _adapter(dataset_id=str(tmp_path), subset_list=['en-hi', 'sat-en'])
    test, fewshot = adapter.load()
    assert [s.input for s in test['en-hi']] == ['test 0', 'test 1', 'test 2']
    assert [s.target for s in test['en-hi']] == ['परीक्षण 0', 'परीक्षण 1', 'परीक्षण 2']
    # The one-shot example is the first `dev` row of the same subset.
    assert [s.input for s in fewshot['en-hi']] == ['dev 0']
    assert [s.input for s in fewshot['sat-en']] == ['ᱥᱟᱨᱦᱟᱣ']
    assert test['sat-en'][0].metadata['target_language_name'] == 'English'

    adapter = _adapter(dataset_id=str(tmp_path), subset_list=['en-hi'], few_shot_num=0)
    assert adapter.load()[1] is None


def test_load_rejects_mismatched_rows(tmp_path: Path) -> None:
    _write(tmp_path, 'flores_en_hi_test.json', [_record('x', 'y', 'hi', 'xxen')])
    adapter = _adapter(dataset_id=str(tmp_path), subset_list=['en-hi'], few_shot_num=0)
    with pytest.raises(ValueError, match='expected hi/enxx'):
        adapter.load()
