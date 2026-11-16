"""End-to-end facts, cache invalidation, coverage and failure isolation; no external services."""

import json
import time
from pathlib import Path
from typing import Callable, List, Optional

import pytest

from evalscope import TaskConfig, run_task
from evalscope.analysis import diagnose_benchmark, diagnose_run
from evalscope.analysis.diagnosis import render_markdown
from evalscope.api.analysis import AnalysisReportConfig
from evalscope.api.evaluator.cache import ReviewResult
from evalscope.api.messages import ChatMessageAssistant, ChatMessageUser
from evalscope.api.metric import SampleScore, Score
from evalscope.constants import ScoreStatus
from evalscope.report import Report
from evalscope.report.diagnosis import load_diagnosis


def review(
    index: int,
    value: float = 0,
    output: str = 'ANSWER: B',
    prediction: str = 'B',
    group: Optional[str] = None,
    generation: Optional[int] = None,
) -> ReviewResult:
    return ReviewResult(
        index=index,
        target=['A'],
        messages=[ChatMessageUser(content='Select A.\u2028Keep this separator.'), ChatMessageAssistant(content=output)],
        sample_score=SampleScore(
            sample_id=index,
            group_id=group,
            generation_index=generation,
            score=Score(value={'accuracy': value}, prediction=output, extracted_prediction=prediction),
        ),
    )


@pytest.fixture
def make_run(tmp_path: Path) -> Callable:
    def write(name: str = 'run', rows: Optional[List[ReviewResult]] = None, repeats: int = 1) -> Path:
        root = tmp_path / name
        for folder in ['reviews/m', 'predictions/m', 'reports/m', 'configs']:
            (root / folder).mkdir(parents=True, exist_ok=True)
        records = rows if rows is not None else [review(0, 1, prediction='A'), review(1)]
        (root / 'configs/task_config.yaml').write_text(
            f'datasets: [demo]\nrepeats: {repeats}\ngeneration_config: {{max_tokens: 16}}\n'
        )
        (root / 'reviews/m/demo_default.jsonl').write_text(''.join(r.model_dump_json() + '\n' for r in records))
        (root / 'predictions/m/demo_default.jsonl').write_text(
            ''.join(
                json.dumps(
                    {
                        'index': r.index,
                        'model_output': {'choices': [{'stop_reason': 'stop'}], 'usage': {'output_tokens': 5}},
                    }
                )
                + '\n'
                for r in records
            )
        )
        (root / 'reports/m/demo.json').write_text(
            Report(
                dataset_name='demo',
                model_name='m',
                execution_summary={'requested': len(records), 'succeeded': len(records)},
            ).model_dump_json()
        )
        return root

    return write


def config() -> AnalysisReportConfig:
    return AnalysisReportConfig(trace={'use_providers': False})


def test_facts_are_offline_and_preserve_source_scores(make_run: Callable, monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_run()
    original = (root / 'reviews/m/demo_default.jsonl').read_bytes()
    monkeypatch.setattr(
        'evalscope.api.model.get_model', lambda *a, **kw: pytest.fail('Facts must not construct a model')
    )
    [report] = diagnose_run(str(root), config())
    assert report.cost.calls == 0 and report.coverage.samples == 2 and report.coverage.task_failed == 1
    assert (root / 'reviews/m/demo_default.jsonl').read_bytes() == original
    persisted = json.loads((root / 'reports/m/demo.json').read_text())
    assert persisted['metrics'] == [] and persisted['analysis'] == render_markdown(report)
    assert {'F-scores', 'F-mcq', 'F-truncation'} <= {f.id for f in report.facts}
    assert load_diagnosis(str(root), 'm', 'demo')[0] == 'ready'


def test_nested_runs_are_not_silently_dropped(make_run: Callable) -> None:
    root = make_run('parent/a/b/c/d/run')
    reports = diagnose_run(str(root.parents[4]), config())
    assert len(reports) == 1 and reports[0].coverage.samples == 2


@pytest.mark.parametrize('dependency', ['reviews', 'predictions', 'reports', 'configs'])
def test_every_consumed_source_invalidates_cache(make_run: Callable, dependency: str) -> None:
    root = make_run()
    before = diagnose_benchmark(str(root), 'm', 'demo', config())
    if dependency == 'reviews':
        path = root / 'reviews/m/demo_default.jsonl'
        path.write_text(path.read_text().replace('ANSWER: B', 'New answer'))
    elif dependency == 'predictions':
        path = root / 'predictions/m/demo_default.jsonl'
        path.write_text(path.read_text().replace('"stop"', '"max_tokens"'))
    elif dependency == 'reports':
        path = root / 'reports/m/demo.json'
        data = json.loads(path.read_text())
        data['dataset_description'] = 'Updated'
        path.write_text(json.dumps(data))
    else:
        path = root / 'configs/task_config.yaml'
        path.write_text(path.read_text().replace('16', '32'))
    assert load_diagnosis(str(root), 'm', 'demo')[0] == 'stale'
    after = diagnose_benchmark(str(root), 'm', 'demo', config())
    assert before.data_fingerprint != after.data_fingerprint
    assert load_diagnosis(str(root), 'm', 'demo')[0] == 'ready'


def test_markdown_writeback_does_not_invalidate_cache_and_samples_are_committed_together(make_run: Callable) -> None:
    root = make_run()
    first = diagnose_benchmark(str(root), 'm', 'demo', config())
    second = diagnose_benchmark(str(root), 'm', 'demo', config())
    assert first == second
    (root / 'analysis/m/demo/samples.jsonl').write_text('')
    assert load_diagnosis(str(root), 'm', 'demo')[0] == 'stale'
    repaired = diagnose_benchmark(str(root), 'm', 'demo', config())
    assert repaired.samples_sha256 == first.samples_sha256


def test_unknown_scores_and_missing_predictions_are_not_safe_zeroes(make_run: Callable) -> None:
    root = make_run(rows=[review(0, 0.5)])
    (root / 'predictions/m/demo_default.jsonl').unlink()
    [report] = diagnose_run(str(root), config())
    assert report.coverage.outcome_unknown == 1 and report.coverage.task_failed == 0
    assert 'F-task-failure' in report.coverage.skipped_facts
    assert 'F-truncation' in report.coverage.skipped_facts
    assert not any(f.id == 'F-truncation' for f in report.facts)


def test_stop_reason_overrides_token_budget_heuristic(make_run: Callable) -> None:
    root = make_run()
    p = root / 'predictions/m/demo_default.jsonl'
    data = [json.loads(line) for line in p.open()]
    for item in data:
        item['model_output']['usage']['output_tokens'] = 16
    data[1]['model_output']['choices'][0]['stop_reason'] = 'max_tokens'
    p.write_text(''.join(json.dumps(x) + '\n' for x in data))
    [report] = diagnose_run(str(root), config())
    fact = next(f for f in report.facts if f.id == 'F-truncation')
    assert fact.value == 0.5 and fact.data['heuristic_denominator'] == 0
    assert any(p.failure_type == 'truncated' for p in report.patterns)


def test_repeats_do_not_compact_missing_attempts(make_run: Callable) -> None:
    rows = [
        review(0, 1, group='a', generation=0),
        review(1, 0, group='a', generation=1),
        review(2, 0, group='b', generation=1),
    ]
    root = make_run(rows=rows, repeats=2)
    [report] = diagnose_run(str(root), config())
    fact = next(f for f in report.facts if f.id == 'F-repeats')
    assert fact.denominator == 1 and fact.value == 0 and fact.data['incomplete_groups'] == 1


def test_bad_rows_and_excluded_scores_remain_visible(make_run: Callable) -> None:
    row = review(0)
    row.sample_score.score.status = ScoreStatus.EXCLUDED
    row.sample_score.score.value = {}
    root = make_run(rows=[row])
    p = root / 'reviews/m/demo_default.jsonl'
    with p.open('a') as stream:
        stream.write('[]\n{broken\n')
    [report] = diagnose_run(str(root), config())
    assert report.coverage.unreadable == 2 and report.coverage.excluded == 1 and report.coverage.outcome_unknown == 1


def test_excluded_reasons_use_saved_error_and_reveal_missing_details(make_run: Callable) -> None:
    rows = [review(0), review(1), review(2, 1)]
    for row in rows[:2]:
        row.sample_score.score.status = ScoreStatus.EXCLUDED
        row.sample_score.score.value = {}
    rows[0].sample_score.score.metadata = {'error': 'Sandbox failed'}
    rows[2].sample_score.score.metadata = {'error': 'Historical error from a recovered attempt'}
    root = make_run(rows=rows)
    [report] = diagnose_run(str(root), config())
    fact = next(f for f in report.facts if f.id == 'F-score-status')
    assert fact.data['reasons'] == {'Sandbox failed': 1}
    assert report.coverage.excluded == 2
    assert '1 excluded' in report.coverage.skipped_facts['score-error-reasons']


def test_paid_levels_require_explicit_model_without_hidden_downgrades(make_run: Callable) -> None:
    root = make_run()
    with pytest.raises(ValueError, match='explicit analysis model'):
        diagnose_run(str(root), AnalysisReportConfig(level='full'))
    assert (
        TaskConfig(model='m', eval_type='mock_llm', datasets=['gsm8k'], analysis_report=True)
        .resolved_analysis_config()
        .level
        == 'facts'
    )


def test_cli_accepts_json_configuration(
    make_run: Callable, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    from evalscope.cli.cli import run_cmd

    root = make_run()
    monkeypatch.setattr(
        'sys.argv',
        [
            'evalscope',
            'analyze',
            '--run-dir',
            str(root),
            '--analysis-config',
            '{"trace":{"use_providers":false},"language":"zh"}',
        ],
    )
    run_cmd()
    assert 'calls=0' in capsys.readouterr().out
    assert '自动诊断' in (root / 'reports/m/demo.json').read_text()


@pytest.mark.parametrize('broken', [False, True])
def test_online_hook_returns_analysis_or_isolates_diagnostic_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, broken: bool
) -> None:
    if broken:

        def fail(*args: object, **kwargs: object) -> None:
            raise RuntimeError('Deliberate diagnostic failure')

        monkeypatch.setattr('evalscope.analysis.diagnosis.collect_facts', fail)
    task = TaskConfig(
        model='m',
        eval_type='mock_llm',
        datasets=['general_mcq'],
        dataset_args={'general_mcq': {'local_path': 'custom_eval/text/mcq', 'subset_list': ['example']}},
        limit=1,
        analysis_report=True,
        work_dir=str(tmp_path),
        no_timestamp=True,
    )
    returned = run_task(task)
    report = returned['general_mcq']
    if not broken:
        offline = diagnose_benchmark(str(tmp_path), 'm', 'general_mcq')
        assert report.analysis == render_markdown(offline)
    else:
        assert report.metrics and report.analysis == 'N/A'
        status, error = load_diagnosis(str(tmp_path), 'm', 'general_mcq')
        assert status == 'error' and error.coverage.analysis_errors == {'RuntimeError': 1}


def test_facts_ten_thousand_samples_under_one_minute(make_run: Callable) -> None:
    root = make_run(rows=[review(i, float(i % 2)) for i in range(10000)])
    start = time.monotonic()
    [report] = diagnose_run(str(root), config())
    assert report.coverage.samples == 10000
    assert time.monotonic() - start < 60


def test_service_filters_before_paging_and_refuses_stale_joins(make_run: Callable) -> None:
    from evalscope.service.api_models import DiagnosisResponse, SampleDiagnosesResponse
    from evalscope.service.app import create_app

    root = make_run(rows=[review(i) for i in range(4)])
    [report] = diagnose_run(str(root), config())
    client = create_app(outputs=str(root.parent)).test_client()
    path = '/api/v1/reports/runs/run/models/m/diagnosis'
    response = client.get(path, query_string={'dataset_name': 'demo'})
    parsed = DiagnosisResponse.model_validate(response.get_json())
    assert parsed.status == 'ready' and parsed.report.coverage.samples == 4
    params = {'dataset_name': 'demo', 'pattern_id': report.patterns[0].id, 'page': 2, 'page_size': 2}
    response = client.get(path + '/samples', query_string=params)
    parsed = SampleDiagnosesResponse.model_validate(response.get_json())
    assert parsed.total == 4 and [s.key.index for s in parsed.items] == [2, 3]
    assert client.get(path, query_string={'dataset_name': '../demo'}).status_code == 400
    assert client.get(path + '/samples', query_string={**params, 'page_size': 1000}).status_code == 400
    (root / 'reviews/m/demo_default.jsonl').write_text('')
    response = client.get(path + '/samples', query_string={'dataset_name': 'demo'})
    assert response.get_json()['status'] == 'stale' and response.get_json()['items'] == []


def test_generation_facts_do_not_claim_extraction_proves_a_correct_answer(make_run: Callable) -> None:
    root = make_run(rows=[review(0, output='not a valid answer', prediction='')])
    [report] = diagnose_run(str(root), config())
    assert report.coverage.task_failed == 1
    extraction = next(f for f in report.facts if f.id == 'F-extraction')
    assert extraction.numerator == 1
    assert report.patterns[0].failure_type == 'extraction_failure'
    assert report.patterns[0].attribution.value == 'unknown'


def test_html_supports_fresh_diagnosis_and_legacy_fallback(make_run: Callable) -> None:
    from evalscope.report.renderer import gen_html_report_file

    root = make_run()
    path = root / 'reports/m/demo.json'
    data = json.loads(path.read_text())
    data['analysis'] = 'Legacy analysis'
    path.write_text(json.dumps(data))
    assert 'Legacy analysis' in Path(gen_html_report_file(str(root / 'reports'))).read_text()
    diagnose_run(str(root), config())
    html = Path(gen_html_report_file(str(root / 'reports'))).read_text()
    assert 'Diagnostic facts' in html and 'No rule-based explanation' in html
    assert '暂无规则线索' in html and 'data-desc-lang="zh"' in html
    assert '<td>unexplained</td>' not in html and '<td>unknown</td>' not in html
    assert '50.0%' in html and '0.5 —' not in html


def test_schema_codegen_keeps_real_title_properties() -> None:
    from evalscope.service.api_models import WebApiContracts
    from scripts.generate_web_api_contracts import _strip_property_titles

    schema = WebApiContracts.model_json_schema(mode='serialization')
    _strip_property_titles(schema)
    fact = schema['$defs']['Fact']
    assert 'title' in fact['properties'] and fact['properties']['title']['type'] == 'string'
    assert 'title' not in fact


def test_broken_manifest_regenerates_without_force(make_run: Callable) -> None:
    root = make_run()
    diagnose_benchmark(str(root), 'm', 'demo', config())
    (root / 'analysis/m/demo/diagnosis.json').write_text('{broken')
    assert load_diagnosis(str(root), 'm', 'demo')[0] == 'error'
    assert diagnose_benchmark(str(root), 'm', 'demo', config()).coverage.samples == 2


def test_environment_secrets_are_removed_from_values_and_dynamic_keys(
    make_run: Callable, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = 'diagnosis-test-secret-123456789'
    monkeypatch.setenv('DIAGNOSIS_API_KEY', secret)
    row = review(0)
    row.sample_score.score.value = {secret: 0}
    root = make_run(rows=[row])
    diagnose_run(str(root), config())
    for path in (root / 'analysis').rglob('*'):
        if path.is_file():
            assert secret not in path.read_text()
    assert secret not in json.loads((root / 'reports/m/demo.json').read_text())['analysis']


def test_lower_is_better_and_unknown_direction_are_respected(make_run: Callable) -> None:
    root = make_run()
    options = config()
    options.trace.outcome_directions = {'demo': 'lower_is_better'}
    report = diagnose_benchmark(str(root), 'm', 'demo', options)
    assert report.coverage.task_failed == 1
    rows = [json.loads(line) for line in (root / 'analysis/m/demo/samples.jsonl').read_text().splitlines()]
    assert rows[0]['key']['index'] == 0 and rows[0]['task_success'] is False
    options.trace.outcome_directions = {'demo': 'none'}
    report = diagnose_benchmark(str(root), 'm', 'demo', options)
    assert report.coverage.task_success_known == 0 and report.coverage.outcome_unknown == 2


def test_no_reviews_retains_execution_errors_without_fabricating_zero_scores(make_run: Callable) -> None:
    root = make_run(rows=[])
    path = root / 'reports/m/demo.json'
    data = json.loads(path.read_text())
    data['execution_summary'] = {'requested': 3, 'errored': 3, 'incomplete': True}
    path.write_text(json.dumps(data))
    [report] = diagnose_run(str(root), config())
    assert report.coverage.samples == 0 and report.coverage.errored_without_record == 3
    assert report.coverage.task_success_known == 0
    assert not any(f.id == 'F-task-failure' for f in report.facts)
    assert 'Diagnostic facts' in (root / 'reports/report.html').read_text()


def test_web_evaluations_default_to_free_facts() -> None:
    from evalscope.service.blueprints.eval import _build_task_config

    task = _build_task_config({'model': 'm', 'datasets': ['gsm8k'], 'eval_type': 'mock_llm'})
    assert task.resolved_analysis_config().level == 'facts'


def test_analysis_config_snapshot_round_trip_redacts_keys(tmp_path: Path) -> None:
    from evalscope.utils.io_utils import yaml_to_dict

    task = TaskConfig(
        model='m',
        datasets=['gsm8k'],
        eval_type='mock_llm',
        api_key='secret-test-key-123456789',
        analysis_report=AnalysisReportConfig(language='zh'),
    )
    task.dump_yaml(str(tmp_path))
    text = (tmp_path / 'task_config.yaml').read_text()
    assert 'secret-test-key-123456789' not in text
    assert (
        TaskConfig.from_dict(yaml_to_dict(str(tmp_path / 'task_config.yaml'))).resolved_analysis_config().language
        == 'zh'
    )


@pytest.mark.parametrize(
    'error_type,message,issue',
    [
        ('timeout', 'Command timed out after 30s', 'environment_failure'),
        ('permission', 'Operation denied', 'guard_block'),
        ('parsing', 'Cannot parse arguments', 'bad_arguments'),
        ('unknown', "Tool 'lookup' is not registered.", 'wrong_tool'),
    ],
)
def test_correct_answer_can_keep_tool_findings_without_becoming_a_task_failure(
    make_run: Callable, error_type: str, message: str, issue: str
) -> None:
    from evalscope.api.messages import ChatMessageTool
    from evalscope.api.tool import ToolCall, ToolCallError, ToolFunction

    row = review(0, 1, prediction='A')
    row.messages = [
        ChatMessageUser(content='Look up A.'),
        ChatMessageAssistant(
            content='', tool_calls=[ToolCall(id='call-7', function=ToolFunction(name='lookup', arguments={}))]
        ),
        ChatMessageTool(
            content=message,
            tool_call_id='call-7',
            function='lookup',
            error=ToolCallError(type=error_type, message=message),
        ),
        ChatMessageAssistant(content='A'),
    ]
    root = make_run(rows=[row])
    result = diagnose_benchmark(str(root), 'm', 'demo', config())
    assert result.coverage.task_failed == 0 and result.coverage.needs_attention == 1
    assert result.patterns[0].failure_type == issue
    sample = json.loads((root / 'analysis/m/demo/samples.jsonl').read_text())
    assert sample['task_success'] is True and sample['findings'][0]['call_ids'] == ['call-7']


def test_safe_refusals_of_harmful_tasks_are_not_failure_patterns(
    make_run: Callable, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from evalscope.api.analysis import Outcome, ProviderSignals

    row = review(0, output='I cannot help with that request.')
    root = make_run(rows=[row])
    monkeypatch.setattr(
        'evalscope.analysis.trace.analyzer.resolve_provider',
        lambda name: SimpleNamespace(
            name='scripted',
            collect=lambda record, view: ProviderSignals(
                labels={'task_kind': 'harmful'}, outcome=Outcome(safety_violation=False, refused=True)
            ),
        ),
    )
    options = config()
    options.trace.use_providers = True
    report = diagnose_benchmark(str(root), 'm', 'demo', options)
    assert report.coverage.needs_attention == 0 and not report.patterns
    sample = json.loads((root / 'analysis/m/demo/samples.jsonl').read_text())
    assert any(f['category'] == 'refusal' for f in sample['findings'])
    assert sample['failure_type'] is None


def test_pattern_uses_decisive_finding_and_separates_tools(make_run: Callable) -> None:
    from evalscope.api.messages import ChatMessageTool
    from evalscope.api.tool import ToolCall, ToolCallError, ToolFunction

    rows = []
    for index, name in enumerate(['search', 'calendar']):
        row = review(index, 0)
        row.messages = [
            ChatMessageUser(content='Do the task.'),
            ChatMessageAssistant(
                content='',
                tool_calls=[
                    ToolCall(id='a', function=ToolFunction(name='first', arguments={})),
                    ToolCall(id='b', function=ToolFunction(name=name, arguments={})),
                ],
            ),
            ChatMessageTool(content='ValueError: invalid argument', tool_call_id='a', function='first'),
            ChatMessageTool(
                content='timed out',
                tool_call_id='b',
                function=name,
                error=ToolCallError(type='timeout', message='timed out'),
            ),
        ]
        rows.append(row)
    report = diagnose_benchmark(str(make_run(rows=rows)), 'm', 'demo', config())
    assert len(report.patterns) == 2
    assert {p.detail for p in report.patterns} == {'structured:timeout @ search', 'structured:timeout @ calendar'}
    assert all(
        p.failure_type == 'environment_failure' and p.attribution.value == 'environment' for p in report.patterns
    )


def test_fatal_analysis_error_is_visible_and_recovers(make_run: Callable, monkeypatch: pytest.MonkeyPatch) -> None:
    import evalscope.analysis.diagnosis as engine

    root = make_run()
    before = json.loads((root / 'reports/m/demo.json').read_text())
    with monkeypatch.context() as patch:
        patch.setattr(engine, 'collect_facts', lambda *args: (_ for _ in ()).throw(RuntimeError('Cannot analyze data')))
        with pytest.raises(RuntimeError, match='Cannot analyze'):
            diagnose_benchmark(str(root), 'm', 'demo', config())
    status, error = load_diagnosis(str(root), 'm', 'demo')
    assert status == 'error' and error.coverage.analysis_errors == {'RuntimeError': 1}
    assert json.loads((root / 'reports/m/demo.json').read_text()) == before
    result = diagnose_benchmark(str(root), 'm', 'demo', config())
    assert not result.coverage.analysis_errors
    assert load_diagnosis(str(root), 'm', 'demo')[0] == 'ready'


@pytest.mark.parametrize('prediction,invalid', [(' A ', False), ('AB', True), ('   ', True)])
def test_invalid_choice_extraction_is_observed_without_claiming_the_answer_was_correct(
    make_run: Callable, prediction: str, invalid: bool
) -> None:
    root = make_run(rows=[review(0, 1, output='Some reasoning then an answer.', prediction=prediction)])
    result = diagnose_benchmark(str(root), 'm', 'demo', config())
    fact = next(f for f in result.facts if f.id == 'F-extraction')
    assert fact.numerator == int(invalid)
    assert result.coverage.task_failed == 0
    assert any(p.failure_type == 'extraction_failure' for p in result.patterns) == invalid


@pytest.mark.parametrize('lower_is_better', [False, True])
def test_weak_subset_includes_exact_ten_point_boundary_and_respects_minimum_size(
    make_run: Callable, lower_is_better: bool
) -> None:
    from evalscope.api.metric.semantics import MetricDirection

    root = make_run()
    overall, weak, good = (0.1, 0.2, 0.05) if lower_is_better else (0.9, 0.8, 0.95)
    report = Report.from_dict(
        {
            'dataset_name': 'demo',
            'model_name': 'm',
            'metrics': [
                {
                    'name': 'mean_acc',
                    'score': overall,
                    'categories': [
                        {
                            'name': ['default'],
                            'subsets': [
                                {'name': 'weak', 'score': weak, 'num': 20},
                                {'name': 'too_small', 'score': weak, 'num': 19},
                                {'name': 'good', 'score': good, 'num': 20},
                            ],
                        }
                    ],
                }
            ],
        }
    )
    metric = report.metrics[0]
    if lower_is_better:
        metric.semantics = metric.semantics.model_copy(update={'direction': MetricDirection.LOWER_IS_BETTER})
    report.primary_metric_identity = metric.identity
    (root / 'reports/m/demo.json').write_text(report.model_dump_json())
    result = diagnose_benchmark(str(root), 'm', 'demo', config())
    fact = next(f for f in result.facts if f.id == 'F-weak-subsets')
    assert [item['subset'] for item in fact.data['items']] == ['weak']


def test_pattern_examples_prefer_confidence_and_are_capped_at_five(
    make_run: Callable, monkeypatch: pytest.MonkeyPatch
) -> None:
    import evalscope.analysis.diagnosis as engine
    from evalscope.api.analysis import Confidence
    from evalscope.api.messages import ChatMessageTool
    from evalscope.api.tool import ToolCall, ToolCallError, ToolFunction

    rows = []
    for index in range(6):
        row = review(index)
        row.messages = [
            ChatMessageUser(content='Search.'),
            ChatMessageAssistant(
                content='', tool_calls=[ToolCall(id='a', function=ToolFunction(name='search', arguments={}))]
            ),
            ChatMessageTool(
                content='bad arguments',
                tool_call_id='a',
                function='search',
                error=ToolCallError(type='parsing', message='bad arguments'),
            ),
        ]
        rows.append(row)
    original = engine.load_inputs

    def load(*args: object, **kwargs: object):
        result = original(*args, **kwargs)
        levels = [
            Confidence.LOW,
            Confidence.HIGH,
            Confidence.MEDIUM,
            Confidence.HIGH,
            Confidence.HIGH,
            Confidence.MEDIUM,
        ]
        for item, level in zip(result.samples, levels):
            item.case.findings[0].confidence = level
        return result

    monkeypatch.setattr(engine, 'load_inputs', load)
    options = config()
    options.max_examples = 10
    result = diagnose_benchmark(str(make_run(rows=rows)), 'm', 'demo', options)
    assert [key.index for key in result.patterns[0].examples] == [1, 3, 4, 2, 5]


def test_html_distinguishes_unknown_outcomes_and_escapes_tool_details(make_run: Callable) -> None:
    from evalscope.report.renderer import _render_diagnostic_facts

    root = make_run(rows=[review(0, 0.4)])
    [result] = diagnose_run(str(root), config())
    html = _render_diagnostic_facts(result)
    assert 'Task outcomes unavailable' in html and '任务成败暂无法判定' in html
    assert '0/0' not in html
    root = make_run(name='failed', rows=[review(0)])
    [result] = diagnose_run(str(root), config())
    result.patterns[0].detail = 'runtime_error @ <script>alert(1)</script>'
    html = _render_diagnostic_facts(result)
    assert '&lt;script&gt;' in html and '<script>' not in html
    assert 'Share of 1 reviewed samples' in html and '<td>P1</td>' in html
