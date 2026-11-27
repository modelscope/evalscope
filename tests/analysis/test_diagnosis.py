"""End-to-end facts, cache invalidation, coverage and failure isolation; no external services."""

import json
import time
from hashlib import sha256
from pathlib import Path
from typing import Callable

import pytest

from evalscope import TaskConfig, run_task
from evalscope.analysis import diagnose_benchmark, diagnose_run
from evalscope.analysis.diagnosis import render_markdown
from evalscope.api.analysis import AnalysisReportConfig
from evalscope.api.messages import ChatMessageAssistant, ChatMessageUser
from evalscope.constants import ScoreStatus
from evalscope.report import Report
from evalscope.report.diagnosis import load_diagnosis
from tests.analysis.helpers import review


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
    assert report.coverage.unexplained_failures == 1
    [failed] = [json.loads(line) for line in (root / 'analysis/m/demo/samples.jsonl').read_text().splitlines()]
    assert failed['task_success'] is False and failed['findings'] == failed['observed_types'] == []
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
        path.write_text(path.read_text().replace('repeats: 1', 'repeats: 2'))
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


def test_truncation_counts_only_recorded_endings(make_run: Callable) -> None:
    root = make_run(rows=[review(0, 1), review(1), review(2)])
    p = root / 'predictions/m/demo_default.jsonl'
    data = [json.loads(line) for line in p.open()]
    for item in data:
        item['model_output']['usage']['output_tokens'] = 16
    data[1]['model_output']['choices'][0]['stop_reason'] = 'max_tokens'
    data[2]['model_output']['choices'][0]['stop_reason'] = 'unknown'
    p.write_text(''.join(json.dumps(x) + '\n' for x in data))
    [report] = diagnose_run(str(root), config())
    fact = next(f for f in report.facts if f.id == 'F-truncation')
    assert fact.value == 0.5 and fact.numerator == 1 and fact.denominator == 2
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


def test_model_analysis_requires_explicit_model_without_hidden_downgrades(make_run: Callable) -> None:
    root = make_run()
    with pytest.raises(ValueError, match='explicit analysis model'):
        diagnose_run(str(root), AnalysisReportConfig(level='diagnose'))
    assert (
        TaskConfig(model='m', eval_type='mock_llm', datasets=['gsm8k'], analysis_report=True)
        .resolved_analysis_config()
        .level
        == 'facts'
    )


@pytest.mark.parametrize('level', ['facts', 'diagnose'])
def test_cli_accepts_json_configuration(
    make_run: Callable, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, level: str
) -> None:
    from evalscope.cli.cli import run_cmd

    root = make_run(rows=[review(0, 1)])
    config = {'trace': {'use_providers': False}, 'language': 'zh', 'level': level}
    if level == 'diagnose':
        config['model'] = {'model_id': 'unused'}
    monkeypatch.setattr('sys.argv', ['evalscope', 'analyze', '--run-dir', str(root), '--analysis-config', json.dumps(config)])
    run_cmd()
    output = capsys.readouterr().out
    assert 'calls=0' in output
    base = root / 'analysis/m/demo'
    artifact = (base / 'diagnose' if level == 'diagnose' else base) / 'diagnosis.json'
    assert artifact.is_file() and f'output={artifact.resolve()}' in output
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
    diagnose_run(str(root), config())
    client = create_app(outputs=str(root.parent)).test_client()
    path = '/api/v1/reports/runs/run/models/m/diagnosis'
    response = client.get(path, query_string={'dataset_name': 'demo'})
    parsed = DiagnosisResponse.model_validate(response.get_json())
    assert parsed.status == 'ready' and parsed.report.coverage.samples == 4
    params = {'dataset_name': 'demo', 'failure_type': 'unexplained', 'outcome': 'failed', 'page': 2, 'page_size': 2}
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


@pytest.mark.parametrize('media', [
    {'type': 'image', 'image': '/not-read/image.png'},
    {'type': 'audio', 'audio': '/not-read/audio.wav', 'format': 'wav'},
    {'type': 'video', 'video': '/not-read/video.mp4', 'format': 'mp4'},
    {'type': 'data', 'data': {'output': 42}},
])
def test_nontext_answers_are_not_reported_as_empty_text_generations(make_run: Callable, media: dict) -> None:
    row = review(0)
    row.messages[-1] = ChatMessageAssistant(content=[media])
    row.sample_score.score.prediction = ''
    row.sample_score.score.extracted_prediction = ''
    root = make_run(rows=[row])
    report = diagnose_benchmark(str(root), 'm', 'demo', config())
    assert not any(f.id in ('F-empty', 'F-extraction', 'F-length') for f in report.facts)
    assert report.coverage.unexplained_failures == 1
    assert all(pattern.failure_type != 'empty_output' for pattern in report.patterns)
    assert 'nontext-output' in report.coverage.skipped_facts
    persisted = (root / 'analysis/m/demo/samples.jsonl').read_text()
    assert 'empty_output' not in persisted and 'extraction_failure' not in persisted


def test_binary_primary_and_continuous_secondary_keep_the_failure_population(make_run: Callable) -> None:
    row = review(0)
    row.sample_score.score.value = {'aux': 0.6, 'accuracy': 0}
    row.sample_score.score.main_score_name = 'accuracy'
    report = diagnose_benchmark(str(make_run(rows=[row])), 'm', 'demo', config())
    assert report.coverage.task_failed == report.coverage.task_success_known == 1


@pytest.mark.parametrize('name,semantic_id,known', [
    ('custom_result', 'quality.accuracy.ratio', 2),
    ('custom_result', 'quality.exact_match.ratio', 2),
    ('custom_result', 'quality.score.ratio', 0),
    ('custom_result', 'quality.pass_at_k.ratio', 0),
    ('accuracy', 'quality.score.ratio', 0),
])
def test_task_outcome_uses_report_semantics(make_run: Callable, name: str, semantic_id: str, known: int) -> None:
    rows = [review(index) for index in range(3)]
    for row, value in zip(rows, (0, 1, 0.5)):
        row.sample_score.score.value = {name: value}
    root = make_run(rows=rows)
    path = root / 'reports/m/demo.json'
    report = json.loads(path.read_text())
    report['metrics'] = [{
        'identity': {'name': name, 'aggregation': 'mean'},
        'semantics': {
            'semantic_id': semantic_id,
            'metric_name': 'Custom result',
            'kind': 'quality',
            'direction': 'higher_is_better',
        },
    }]
    path.write_text(json.dumps(report))

    result = diagnose_benchmark(str(root), 'm', 'demo', config())
    assert result.coverage.task_success_known == known
    assert result.coverage.task_failed == known // 2
    assert result.coverage.outcome_unknown == 3 - known


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
    assert result.coverage.task_failed == 0
    assert not result.patterns
    sample = json.loads((root / 'analysis/m/demo/samples.jsonl').read_text())
    assert issue in sample['observed_types'] and sample['needs_attention']
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
    assert not report.patterns
    sample = json.loads((root / 'analysis/m/demo/samples.jsonl').read_text())
    assert any(f['category'] == 'refusal' for f in sample['findings'])
    assert not sample['needs_attention'] and sample['observed_types'] == []


def test_patterns_preserve_multiple_observations_without_assigning_a_cause(make_run: Callable) -> None:
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
    assert {p.failure_type for p in report.patterns} == {'bad_arguments', 'environment_failure'}
    assert all(p.failed == 2 for p in report.patterns)
    assert report.coverage.task_failed == 2
    samples = [json.loads(line) for line in (Path(report.run_dir) / 'analysis/m/demo/samples.jsonl').read_text().splitlines()]
    assert all(row['observed_types'] == ['bad_arguments', 'environment_failure'] for row in samples)


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
    assert not result.patterns
    rows = (root / 'analysis/m/demo/samples.jsonl').read_text()
    if invalid:
        assert 'extraction_failure' in json.loads(rows)['observed_types']
    else:
        assert not rows


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


def test_pattern_examples_follow_configured_limit_without_treating_confidence_as_causation(
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
    assert [key.index for key in result.patterns[0].examples] == [0, 1, 2, 3, 4, 5]


def test_html_distinguishes_unknown_outcomes_and_escapes_subset_names(make_run: Callable) -> None:
    from evalscope.report.renderer import _render_diagnostic_facts

    root = make_run(rows=[review(0, 0.4)])
    [result] = diagnose_run(str(root), config())
    html = _render_diagnostic_facts(result)
    assert 'Task outcomes unavailable' in html and '任务成败暂无法判定' in html
    assert '0/0' not in html
    root = make_run(name='failed', rows=[review(0)])
    [result] = diagnose_run(str(root), config())
    result.patterns[0].examples[0].subset = '<script>alert(1)</script>'
    html = _render_diagnostic_facts(result)
    assert '&lt;script&gt;' in html and '<script>' not in html
    assert 'Failed samples' in html and 'Unscored samples' in html


def test_service_priorities_and_navigation_use_the_same_distinct_samples(make_run: Callable) -> None:
    from evalscope.analysis.diagnosis import _review_issues
    from evalscope.api.analysis import SampleDiagnosis
    from evalscope.service.app import create_app

    root = make_run()
    [report] = diagnose_run(str(root), config())
    finding = {
        'detector': 'tool_failure', 'category': 'tool_failure', 'subtype': 'runtime_error',
        'attribution': 'unknown', 'confidence': 'high', 'data': {'tool': 'search'},
    }
    rows = [
        SampleDiagnosis(key={'subset': 'a', 'index': 1}, task_success=False, score_status='success',
                        observed_types=['tool_error', 'bad_arguments'], findings=[finding, finding]),
        SampleDiagnosis(key={'subset': 'b', 'index': 2}, task_success=False, score_status='success',
                        observed_types=['tool_error'], findings=[finding]),
        SampleDiagnosis(key={'subset': 'b', 'index': 3}, task_success=True, score_status='success',
                        observed_types=['tool_error'], findings=[finding]),
        SampleDiagnosis(key={'subset': 'b', 'index': 4}, score_status='parse_error',
                        observed_types=['judge_failure']),
        SampleDiagnosis(key={'subset': 'b', 'index': 5}, score_status='success',
                        observed_types=['tool_error'], findings=[finding]),
        SampleDiagnosis(key={'subset': 'b', 'index': 6}, task_success=True, score_status='success',
                        safety_violation=True),
    ]
    rows.append(SampleDiagnosis(key={'subset': 'b', 'index': 7}, task_success=False, score_status='success'))
    rows.extend(SampleDiagnosis(key={'subset': 'b', 'index': i}, task_success=True, score_status=status)
                for i, status in enumerate(('fallback', 'degraded'), 8))
    path = root / 'analysis/m/demo/samples.jsonl'
    report.patterns = _review_issues(rows, 5)
    report.coverage.risk_samples = sum(row.has_risk for row in rows)
    path.write_text('\n'.join(row.model_dump_json() for row in rows) + '\n')
    report.samples_sha256 = sha256(path.read_bytes()).hexdigest()
    path.with_name('diagnosis.json').write_text(report.model_dump_json())
    client = create_app(outputs=str(root.parent)).test_client()
    base = '/api/v1/reports/runs/run/models/m'
    response = client.get(base + '/diagnosis', query_string={'dataset_name': 'demo'})
    summary = response.get_json()
    assert response.status_code == 200
    issues = summary['report']['patterns']
    assert issues[0]['failed'] == 2
    assert issues[0]['tools'] == {'search': 2}
    assert next(issue for issue in issues if issue['failure_type'] == 'judge_failure')['unscored'] == 1
    assert next(issue for issue in issues if issue['failure_type'] == 'bad_arguments')['failed'] == 1
    assert summary['report']['coverage']['risk_samples'] == 1
    assert 'passed_observations' not in summary
    assert client.get(base + '/diagnosis/samples', query_string={'dataset_name': 'demo', 'outcome': 'passed'}).status_code == 400
    params = {'dataset_name': 'demo', 'failure_type': 'tool_error', 'outcome': 'failed', 'page_size': 1}
    first = client.get(base + '/diagnosis/samples', query_string=params).get_json()
    second = client.get(base + '/diagnosis/samples', query_string={**params, 'page': 2}).get_json()
    assert first['total'] == second['total'] == 2
    assert first['items'][0]['key'] == {'subset': 'a', 'index': 1}
    assert second['items'][0]['key'] == {'subset': 'b', 'index': 2}
    unknown = client.get(base + '/diagnosis/samples', query_string={'dataset_name': 'demo', 'failure_type': 'unexplained', 'outcome': 'failed'}).get_json()
    assert unknown['total'] == 1 and unknown['items'][0]['key']['index'] == 7
    unscored = client.get(base + '/diagnosis/samples', query_string={
        'dataset_name': 'demo', 'outcome': 'unscored',
    }).get_json()
    assert unscored['total'] == 1 and unscored['items'][0]['key']['index'] == 4
    risk = client.get(base + '/diagnosis/samples', query_string={
        'dataset_name': 'demo', 'outcome': 'risk',
    }).get_json()
    assert risk['total'] == 1 and risk['items'][0]['key']['index'] == 6
    assert client.get(base + '/diagnosis/samples', query_string={
        'dataset_name': 'demo', 'outcome': 'anything',
    }).status_code == 400
    full = client.get(base + '/predictions', query_string={
        'dataset_name': 'demo', 'subset_name': 'default',
    }).get_json()['predictions']
    single = client.get(base + '/predictions', query_string={
        'dataset_name': 'demo', 'subset_name': 'default', 'index': full[0]['Index'],
    }).get_json()['predictions']
    assert single == full[:1]


def test_prediction_etag_distinguishes_subset_and_single_sample(make_run: Callable) -> None:
    from evalscope.service.app import create_app

    root = make_run()
    client = create_app(outputs=str(root.parent)).test_client()
    url = '/api/v1/reports/runs/run/models/m/predictions'
    query = {'dataset_name': 'demo', 'subset_name': 'default'}
    whole = client.get(url, query_string=query)
    assert len(whole.json['predictions']) == 2

    selected = {**query, 'index': 1}
    one = client.get(url, query_string=selected, headers={'If-None-Match': whole.headers['ETag']})
    assert one.status_code == 200
    assert [row['Index'] for row in one.json['predictions']] == ['1']
    assert client.get(url, query_string=selected, headers={'If-None-Match': one.headers['ETag']}).status_code == 304
    other = client.get(url, query_string={**query, 'index': 0}, headers={'If-None-Match': one.headers['ETag']})
    assert other.status_code == 200
    assert [row['Index'] for row in other.json['predictions']] == ['0']

    (root / 'reviews/m/demo_default.jsonl').write_text('')
    empty = client.get(url, query_string=selected)
    assert empty.status_code == 200 and empty.json['predictions'] == []


@pytest.mark.parametrize('version', [1, 2, 3, 4, 5, 6])
def test_old_diagnostic_schema_is_stale_instead_of_silently_mixing_old_cause_counts(make_run: Callable, version: int) -> None:
    from evalscope.report.diagnosis import load_diagnosis

    root = make_run()
    diagnose_benchmark(str(root), 'm', 'demo', config())
    path = root / 'analysis/m/demo/diagnosis.json'
    data = json.loads(path.read_text())
    data['schema_version'] = version
    if version == 3:
        data['level'] = 'full'
    path.write_text(json.dumps(data))
    assert load_diagnosis(str(root), 'm', 'demo') == ('stale', None)


@pytest.mark.parametrize('payload', [None, [], False])
def test_malformed_artifact_root_is_an_error_without_breaking_report_loading(make_run: Callable, payload: object) -> None:
    from evalscope.report.diagnosis import load_diagnosis

    root = make_run()
    path = root / 'analysis/m/demo/diagnosis.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload))
    assert load_diagnosis(str(root), 'm', 'demo') == ('error', None)


def test_web_generates_analysis_for_an_existing_dataset(make_run: Callable, monkeypatch: pytest.MonkeyPatch) -> None:
    from evalscope.service.app import create_app
    from evalscope.service.blueprints import reports

    root = make_run(rows=[review(0)])
    client = create_app(outputs=str(root.parent)).test_client()
    url = '/api/v1/reports/runs/run/models/m/diagnosis'
    query = {'dataset_name': 'demo'}

    def inline(func: Callable, *args: object) -> object:
        return func(*args)

    monkeypatch.setattr(reports, 'run_in_subprocess', inline)
    assert client.get(url, query_string=query).get_json()['status'] == 'missing'
    assert client.post(url, query_string=query, json={'level': 'diagnose'}).status_code == 400
    assert client.post(url, query_string=query, json={}, headers={'Sec-Fetch-Site': 'cross-site'}).status_code == 403
    assert client.post(url, query_string=query, json={'max_diagnosed_samples': 51}).status_code == 400
    assert client.post(url, query_string={'dataset_name': '../demo'}, json={}).status_code == 400
    assert client.post(url, query_string={'dataset_name': 'missing'}, json={}).status_code == 404
    response = client.post(url, query_string=query, json={})
    assert response.status_code == 200
    assert response.get_json()['report']['level'] == 'facts'
    assert response.get_json()['report']['cost']['calls'] == 0
    assert client.get(url + '/samples', query_string=query).get_json()['items'][0]['key']['index'] == 0



def test_web_diagnosis_only_uses_the_credential_supplied_in_the_form(
    make_run: Callable, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evalscope.service.app import create_app
    from evalscope.service.blueprints import reports

    root = make_run()
    client = create_app(outputs=str(root.parent)).test_client()
    url = '/api/v1/reports/runs/run/models/m/diagnosis?dataset_name=demo'
    supplied_keys = []
    monkeypatch.setenv('EVALSCOPE_API_KEY', 'unused-server-key')

    def collect(_func: Callable, _root: str, _model: str, _dataset: str, config: AnalysisReportConfig) -> None:
        supplied_keys.append(config.model.api_key.get_secret_value())

    monkeypatch.setattr(reports, 'run_in_subprocess', collect)
    payload = {'level': 'diagnose', 'model_id': 'analyst', 'api_url': 'http://localhost/v1'}
    assert client.post(url, json={**payload, 'api_key': 'env:EVALSCOPE_API_KEY'}).status_code == 400
    assert client.post(url, json=payload).status_code == 200
    assert client.post(url, json={**payload, 'api_key': 'explicit-test-key'}).status_code == 200
    assert supplied_keys == ['EMPTY', 'explicit-test-key']


@pytest.mark.parametrize('kind', ['default', 'task_config', 'unknown'])
def test_web_upgrades_legacy_facts_only_if_original_settings_can_be_recovered(
    make_run: Callable, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    from evalscope.service.app import create_app
    from evalscope.service.blueprints import reports

    root = make_run()
    options = AnalysisReportConfig(language='zh', warning_rate=0.3 if kind != 'default' else 0.1)
    snapshot = diagnose_benchmark(str(root), 'm', 'demo', options)
    path = root / 'analysis/m/demo/diagnosis.json'
    legacy = json.loads(path.read_text())
    legacy.pop('facts_config')
    path.write_text(json.dumps(legacy))
    if kind == 'task_config':
        (root / 'configs/task_config.yaml').write_text(
            'datasets: [demo]\nrepeats: 1\nanalysis_report: ' + options.model_dump_json() + '\n'
        )
    original = path.read_bytes()

    def inline(func: Callable, *args: object) -> object:
        return func(*args)

    monkeypatch.setattr(reports, 'run_in_subprocess', inline)
    client = create_app(outputs=str(root.parent)).test_client()
    url = '/api/v1/reports/runs/run/models/m/diagnosis?dataset_name=demo'
    response = client.post(url, json={})
    if kind == 'unknown':
        assert response.status_code == 409
        assert path.read_bytes() == original
    else:
        assert response.status_code == 200
        assert response.json['report']['config_fingerprint'] == snapshot.config_fingerprint
        assert json.loads(path.read_text())['facts_config']['warning_rate'] == options.warning_rate
