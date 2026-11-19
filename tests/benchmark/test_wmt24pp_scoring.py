import json
from pathlib import Path
from unittest.mock import patch

import pytest

from evalscope import TaskConfig, run_task
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.model import ModelOutput
from evalscope.api.model.model import ModelCache
from evalscope.api.registry import get_benchmark
from evalscope.constants import ScoreStatus


def _config(metrics: list[str]) -> TaskConfig:
    return TaskConfig(model='mock', eval_type='mock_llm', datasets=['wmt24pp'], dataset_args={
        'wmt24pp': {'metric_list': metrics, 'primary_metric': {'name': 'bleu', 'dimensions': {'ngram': 4}}, 'subset_list': ['en-de_de']}})


def bleu(prediction: str, reference: str) -> dict[str, float]:
    if reference == 'unavailable':
        raise LookupError('tokenizer unavailable')
    return {f'bleu-{n}': 0.0 for n in range(1, 5)}


@pytest.mark.parametrize(('bert', 'all_fail'), [(False, False), (True, False), (False, True)])
def test_native_wmt_scoring_cache_and_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bert: bool, all_fail: bool) -> None:
    monkeypatch.setattr(ModelCache, '_models', {})
    cfg = _config(['bleu', 'bert_score'] if bert else ['bleu'])
    data = tmp_path / 'data'
    data.mkdir()
    targets = ['unavailable', 'unavailable' if all_fail else 'valid']
    rows = [{'source': 'translate', 'target': word, 'language_pair': 'en-de_de'} for word in targets]
    (data / 'test.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    cfg.dataset_args['wmt24pp']['local_path'] = str(data)
    cfg.work_dir = str(tmp_path / 'output')
    cfg.no_timestamp = True
    cfg.eval_batch_size = 2
    with patch('evalscope.metrics.bleu_ngram_one_sample', side_effect=bleu) as fn, patch(
        'evalscope.metrics.nlp.metrics.BertScore'
    ) as scorer:
        scorer.return_value.apply.return_value = [0.75, 0.75]
        for cached in (False, True):
            if cached:
                cfg.use_cache = cfg.work_dir
            run_task(cfg)
            reviews = [json.loads(row)['sample_score']['score']
                       for file in Path(cfg.work_dir, 'reviews').rglob('*.jsonl') for row in file.read_text().splitlines()]
            assert len(reviews) == 2
            failed = next(s for s in reviews if 'bleu-4' not in s['value'])
            assert failed['status'] == ('degraded' if bert else 'excluded')
            assert 'tokenizer unavailable' in failed['metadata']['metric_errors']['bleu']
            assert failed['value'] == ({'bert_score': 0.75} if bert else {})
            if not all_fail:
                assert next(s for s in reviews if 'bleu-4' in s['value'])['status'] == 'success'
            report = json.loads(next(Path(cfg.work_dir, 'reports').rglob('wmt24pp.json')).read_text())
            usable = 2 if bert else targets.count('valid')
            assert report['execution_summary']['succeeded'] == usable
            assert report['execution_summary']['errored'] == 2 - usable
            assert all(m['num'] == (2 if m['identity']['name'] == 'bert_score' else 1) for m in report['metrics'])
            assert bool(report['metrics']) == (not all_fail)
            assert fn.call_count == 2
        # A re-review with a restored scorer should recover every row from prediction cache.
        cfg.rerun_review = True
        fn.side_effect = lambda pred, ref: bleu(pred, 'valid')
        run_task(cfg)
        report = json.loads(next(Path(cfg.work_dir, 'reports').rglob('wmt24pp.json')).read_text())
        assert report['execution_summary']['succeeded'] == 2
        assert all(m['num'] == 2 for m in report['metrics'])
        assert fn.call_count == 4


def test_batch_metric_failure_preserves_bleu_and_excludes_empty_score() -> None:
    adapter = get_benchmark('wmt24pp', _config(['bleu', 'bert_score']))
    states = [TaskState(model='mock', sample=Sample(id=i, input='translate', target=ref), completed=True,
                        output=ModelOutput.from_content('mock', 'answer')) for i, ref in enumerate(['valid', 'unavailable'])]
    with patch('evalscope.metrics.bleu_ngram_one_sample', side_effect=bleu), patch(
        'evalscope.metrics.nlp.metrics.BertScore'
    ) as scorer:
        scorer.return_value.apply.side_effect = RuntimeError('scorer unavailable')
        scores = adapter.batch_match_score(['answer'] * 2, ['answer'] * 2, [s.target for s in states], states)
    assert [s.status for s in scores] == [ScoreStatus.DEGRADED, ScoreStatus.EXCLUDED]
    assert scores[0].value['bleu-4'] == 0
    assert scores[1].value == {}
    assert all('scorer unavailable' in s.metadata['metric_errors']['bert_score'] for s in scores)


def test_short_comet_result_does_not_assign_partial_batch() -> None:
    cfg = _config(['bleu', 'comet'])
    with patch('evalscope.benchmarks.wmt.wmt24_adapter.check_import'):
        adapter = get_benchmark('wmt24pp', cfg)
    states = [TaskState(model='mock', sample=Sample(id=i, input='q', target='valid', metadata={'source_text': 'q'}),
                        completed=True, output=ModelOutput.from_content('mock', 'a')) for i in range(2)]
    with patch('evalscope.metrics.bleu_ngram_one_sample', side_effect=bleu), patch(
        'evalscope.metrics.nlp.metrics.COMETScore'
    ) as comet:
        comet.return_value.apply.return_value = [0.9]
        result = adapter.batch_match_score(['a', 'a'], ['a', 'a'], ['valid', 'valid'], states)
    assert all(s.status == ScoreStatus.DEGRADED and 'comet' not in s.value for s in result)
    assert all('Expected 2 scores' in s.metadata['metric_errors']['comet'] for s in result)


def test_usable_bleu_refines_an_unavailable_preliminary_review() -> None:
    from evalscope.api.metric import SampleScore, Score

    adapter = get_benchmark('wmt24pp', _config(['bleu']))
    state = TaskState(model='mock', sample=Sample(id=0, input='q', target='valid'), completed=True,
                      output=ModelOutput.from_content('mock', 'a'))
    review = SampleScore(sample_id=0, score=Score(status=ScoreStatus.PARSE_ERROR, metadata=None))
    with patch('evalscope.metrics.bleu_ngram_one_sample', side_effect=bleu):
        result = adapter.batch_calculate_metrics([state], [review])
    assert result[0].score.status == ScoreStatus.DEGRADED
    assert result[0].score.value['bleu-4'] == 0.0


def test_llm_recall_does_not_overwrite_raw_bleu() -> None:
    from evalscope.api.metric import Score

    cfg = TaskConfig(model='mock', datasets=['wmt24pp'], dataset_args=_config(['bleu']).dataset_args,
                     judge={'strategy': 'llm_recall', 'models': [{'model_id': 'mock-judge'}]})
    adapter = get_benchmark('wmt24pp', cfg)
    state = TaskState(model='mock', sample=Sample(id=0, input='q', target='valid'), completed=True,
                      output=ModelOutput.from_content('mock', 'a'))
    with patch('evalscope.metrics.bleu_ngram_one_sample', side_effect=bleu), patch.object(
        adapter, 'score_with_judge_contracts', return_value=Score(value={'judge': 1.0})
    ):
        preliminary = adapter.calculate_metrics(state)
        assert preliminary.score.value['bleu-1'] == 1.0
        result = adapter.batch_calculate_metrics([state], [preliminary])
    assert result[0].score.value == bleu('a', 'valid')
