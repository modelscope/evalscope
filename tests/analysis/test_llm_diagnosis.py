"""Grounding, failure isolation, stratified inference and paid-request cache boundaries."""

import json
from pathlib import Path
from typing import Callable

import pytest

from evalscope.analysis.diagnosis import diagnose_benchmark
from evalscope.analysis.llm import build_sources, select_samples, validate_evidence
from evalscope.api.analysis import (
    AnalysisReportConfig,
    DiagnosticSource,
    EvidenceReference,
    LLMDiagnosis,
    SampleDiagnosis,
    SampleKey,
)
from evalscope.api.messages import ChatMessageAssistant, ChatMessageTool
from evalscope.api.model import ModelOutput, ModelUsage
from evalscope.api.tool import ToolCall, ToolCallError, ToolFunction
from tests.analysis.helpers import review


def options(**kw: object) -> AnalysisReportConfig:
    return AnalysisReportConfig(level='diagnose', model={'model_id':'independent-reviewer'}, trace={'use_providers':False}, **kw)


def verdict(**kw: object) -> LLMDiagnosis:
    return LLMDiagnosis(failure_type='reasoning_error', attribution='model', explanation='The answer contradicts the calculation.',
                        confidence='high', evidence=[EvidenceReference(source_id='m:1',quote='ANSWER: B')], **kw)


@pytest.mark.parametrize('reference,error',[
    ({'source_id':'missing','quote':'x'},'unknown_source'),
    ({'source_id':'m:1','quote':'x','start':0,'end':99},'out_of_bounds'),
    ({'source_id':'m:1','quote':'x','start':0,'end':2},'quote_mismatch'),
    ({'source_id':'m:1','quote':'ANSWER: B','start':0},'incomplete_range'),
    ({'source_id':'m:1','quote':'hidden tail'},'quote_missing_or_ambiguous'),
])
def test_invalid_references_are_not_repaired(reference: dict,error: str) -> None:
    d=verdict().model_copy(update={'evidence':[EvidenceReference(**reference)]})
    parsed,errors=validate_evidence(d,[DiagnosticSource(id='m:1',text='ANSWER: B')])
    assert parsed.evidence==[] and errors==[error]


def test_unique_quotes_support_unicode_and_whitespace_but_not_ambiguity_or_phantom_steps() -> None:
    d=verdict().model_copy(update={'evidence':[EvidenceReference(source_id='m:1',quote='🧪 answer B')]})
    sources=[DiagnosticSource(id='m:1',text='中文🧪 answer\n B',step=4)]
    parsed,errors=validate_evidence(d,sources)
    assert not errors and parsed.evidence[0].start==2 and parsed.evidence[0].quote=='🧪 answer\n B'
    parsed,errors=validate_evidence(d.model_copy(update={'root_cause_step':3}),sources)
    assert not parsed.evidence and 'unsupported_root_step' in errors
    parsed,errors=validate_evidence(verdict(),[DiagnosticSource(id='m:1',text='ANSWER: B, ANSWER: B')])
    assert not parsed.evidence and errors==['quote_missing_or_ambiguous']


def test_whitespace_range_is_not_evidence() -> None:
    diagnosis = verdict().model_copy(update={'evidence': [
        EvidenceReference(source_id='m:1', quote=' ', start=0, end=1),
    ]})
    parsed, errors = validate_evidence(diagnosis, [DiagnosticSource(id='m:1', text=' ANSWER: B')])
    assert not parsed.evidence and errors == ['blank_quote']


def sample(index: int,subset: str='large',failed: bool=True) -> SampleDiagnosis:
    return SampleDiagnosis(key=SampleKey(subset=subset,index=index),task_success=not failed,
                           score_status='success',needs_attention=True)


def test_scoped_selection_is_repeatable_and_never_spends_failure_budget_on_passing_rows() -> None:
    rows = [sample(i) for i in range(100)] + [sample(i + 100, failed=False) for i in range(900)]
    selected, population = select_samples(rows, options())
    reverse, other = select_samples(list(reversed(rows)), options())
    assert population == other == 100 and len(selected) == 50
    assert [s.key for s in selected] == [s.key for s in reverse]
    assert all(s.task_success is False for s in selected)
    all_rows, population = select_samples(rows, options(scope='all', max_diagnosed_samples=None))
    assert len(all_rows) == population == 1000
    passing, population = select_samples(rows, options(scope='process', max_diagnosed_samples=3))
    assert population == 900 and len(passing) == 3 and all(s.task_success is True for s in passing)


def test_selection_supports_all_limits_explicit_samples_and_more_subsets_than_budget() -> None:
    rows = [sample(i, f'subset-{i}') for i in range(51)]
    selected, population = select_samples(rows, options())
    assert len(selected) == 50 and population == 51
    all_rows, _ = select_samples(rows, options(max_diagnosed_samples=None))
    assert len(all_rows) == 51
    one, _ = select_samples(rows, options(max_diagnosed_samples=1))
    assert len(one) == 1
    explicit, population = select_samples(rows, options(samples=[rows[27].key], max_diagnosed_samples=None))
    assert [s.key for s in explicit] == [rows[27].key] and population == 1
    with pytest.raises(ValueError, match='unavailable'):
        select_samples(rows, options(samples=[SampleKey(subset='missing', index=1)]))
    for limit in (0, -1, True):
        with pytest.raises(ValueError):
            options(max_diagnosed_samples=limit)


def fake_model(monkeypatch: pytest.MonkeyPatch,reply: Callable, calls: list) -> None:
    class FakeModel:
        def generate(self,messages: list) -> ModelOutput:
            prompt=messages[-1].text
            calls.append(prompt)
            content=reply(prompt,len(calls))
            result=ModelOutput.from_content(model='reviewer',content=content)
            result.usage=ModelUsage(input_tokens=20,output_tokens=10)
            return result
    monkeypatch.setattr('evalscope.analysis.llm.get_model',lambda *a,**kw:FakeModel())


@pytest.mark.parametrize('include_trace', [True, False])
def test_known_secrets_are_redacted_in_both_model_requests(
    make_run: Callable, monkeypatch: pytest.MonkeyPatch, include_trace: bool
) -> None:
    secret = 'synthetic-"token-with-quote"-1234'
    monkeypatch.setenv('DIAGNOSIS_TEST_TOKEN', secret)
    row = review(0)
    row.messages[1:1] = [
        ChatMessageAssistant(content='', tool_calls=[
            ToolCall(id='lookup-1', function=ToolFunction(name='lookup', arguments={'value': secret})),
        ]),
        ChatMessageTool(
            content=secret, tool_call_id='lookup-1', function='lookup',
            error=ToolCallError(type='timeout', message=secret),
        ),
    ]
    calls = []
    def reply(prompt: str, _: int) -> str:
        if 'Summarize evaluation' in prompt:
            return '{}'
        return verdict().model_copy(update={
            'explanation': f'The tool failed: {secret}',
            'evidence': [EvidenceReference(source_id='m:3', quote='ANSWER: B')],
        }).model_dump_json()

    fake_model(monkeypatch, reply, calls)
    report = diagnose_benchmark(str(make_run(rows=[row])), 'm', 'demo', options(include_trace=include_trace))
    assert report.coverage.diagnosed_ok == 1 and len(calls) == 2
    assert all('token-with-quote' not in prompt for prompt in calls)
    assert ('lookup-1' in calls[0]) is include_trace


def test_locatable_quote_cannot_override_known_score_status(make_run: Callable, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    diagnosis = verdict().model_copy(update={'failure_type': 'judge_failure'})
    fake_model(monkeypatch, lambda prompt, _: diagnosis.model_dump_json(), calls)
    root = make_run(rows=[review(0)])
    report = diagnose_benchmark(str(root), 'm', 'demo', options())
    row = SampleDiagnosis.model_validate_json((root / 'analysis/m/demo/diagnose/samples.jsonl').read_text())
    assert report.coverage.citations_valid == 1 and report.coverage.diagnosed_ok == 0
    assert row.llm_status == 'unexplained' and row.evidence_errors == ['contradicts_recorded_score_status']
    assert report.narrative_status == 'no_findings' and len(calls) == 1


def test_default_selection_skips_only_recorded_stops_with_no_other_issue() -> None:
    stopped, estimated, other, extraction, risk = [sample(i) for i in range(5)]
    stopped.stop_reason = other.stop_reason = risk.stop_reason = 'max_tokens'
    stopped.observed_types = estimated.observed_types = risk.observed_types = ['truncated']
    risk.safety_violation = True
    other.observed_types = ['tool_error', 'truncated']
    extraction.observed_types = ['extraction_failure']
    rows = [stopped, estimated, other, extraction, risk]

    no_answer = {(stopped.key.subset, stopped.key.index)}
    selected, population = select_samples(rows, options(), no_answer)
    assert population == 4 and {row.key.index for row in selected} == {1, 2, 3, 4}
    assert select_samples(rows, options())[1] == 5
    assert select_samples(rows, options(samples=[stopped.key]), no_answer) == ([stopped], 1)
    assert select_samples(rows, options(scope='all'), no_answer)[1] == 5


def test_output_limit_does_not_hide_a_failed_sample_with_an_extracted_answer(
    make_run: Callable, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_run(rows=[review(0), review(1, prediction='')])
    path = root / 'predictions/m/demo_default.jsonl'
    path.write_text(path.read_text().replace('"stop"', '"max_tokens"'))
    calls = []
    fake_model(monkeypatch, lambda prompt, _: '{}' if 'Summarize evaluation' in prompt else verdict().model_dump_json(), calls)
    report = diagnose_benchmark(str(root), 'm', 'demo', options())
    with (root / 'analysis/m/demo/diagnose/samples.jsonl').open(encoding='utf-8') as stream:
        samples = [json.loads(line) for line in stream]
    assert report.coverage.diagnosis_population == report.coverage.diagnosed_ok == 1
    assert [sample['llm_status'] for sample in samples] == ['ok', 'not_selected']


def test_model_diagnosis_preserves_scores_and_resumes_per_request_cache(make_run: Callable,monkeypatch: pytest.MonkeyPatch) -> None:
    root=make_run()
    original=(root/'reviews/m/demo_default.jsonl').read_bytes()
    calls=[]
    def reply(prompt: str,n: int) -> str:
        if 'Summarize evaluation' in prompt:
            return json.dumps({'findings':[{'text':'One failure needs review.','refs':['F-task-failure']},
                                           {'text':'Invented.','refs':['does-not-exist']}], 'recommendations':[], 'limitations':[]})
        return verdict().model_dump_json()
    fake_model(monkeypatch,reply,calls)
    report=diagnose_benchmark(str(root),'m','demo',options())
    assert report.coverage.selected==report.coverage.diagnosed_ok==1
    assert report.cost.calls==2 and report.cost.input_tokens==40
    assert len(report.narrative.findings)==1 and len(calls)==2
    assert report.coverage.diagnosis_population == 1
    assert report.coverage.selection_scope == 'failed'
    assert (root/'reviews/m/demo_default.jsonl').read_bytes()==original
    rows=[json.loads(line) for line in (root/'analysis/m/demo/diagnose/samples.jsonl').read_text().split('\n') if line.strip()]
    assert rows[0]['llm']['evidence'][0]['quote']=='ANSWER: B'
    assert rows[0]['observed_types'] == []
    assert rows[0]['llm']['failure_type'] == 'reasoning_error'
    assert report.patterns[0].failure_type == 'unexplained'
    (root/'analysis/m/demo/diagnose/diagnosis.json').unlink()
    resumed=diagnose_benchmark(str(root),'m','demo',options())
    assert resumed.cost.calls==0 and resumed.cost.cached_calls==2 and len(calls)==2
    changed=diagnose_benchmark(str(root),'m','demo',options(seed=43))
    assert changed.config_fingerprint!=report.config_fingerprint


@pytest.mark.parametrize('mode,status', [('parse','parse_error'),('transport','transport_error'),('evidence','invalid_evidence'),('low','unexplained')])
def test_failed_diagnoses_never_count_as_explained(make_run: Callable,monkeypatch: pytest.MonkeyPatch,mode: str,status: str) -> None:
    root=make_run()
    calls=[]
    def reply(prompt: str,n: int) -> str:
        if 'Summarize evaluation' in prompt:
            return json.dumps({'findings': [], 'recommendations': [], 'limitations': []})
        if mode=='transport':
            raise ConnectionError('private credential must not leak')
        if mode=='parse':
            return 'I think it failed.'
        d=verdict()
        if mode=='evidence':
            d.evidence=[EvidenceReference(source_id='other_sample',quote='ANSWER: B')]
        if mode=='low':
            d.confidence='low'
        return d.model_dump_json()
    fake_model(monkeypatch,reply,calls)
    report=diagnose_benchmark(str(root),'m','demo',options())
    assert report.coverage.diagnosis_statuses=={status:1}
    assert report.coverage.diagnosed_ok == 0 and report.coverage.unexplained_failures == 1
    assert report.patterns[0].failure_type=='unexplained'
    if mode in ('parse', 'transport', 'evidence'):
        diagnose_benchmark(str(root),'m','demo',options())
        assert sum('Summarize evaluation' not in prompt for prompt in calls) == 2


def test_partial_sample_references_do_not_promote_an_explanation(
    make_run: Callable, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def reply(prompt: str, count: int) -> str:
        assert 'Summarize evaluation' not in prompt
        result = verdict()
        result.evidence.append(EvidenceReference(source_id='score', quote='A reason absent from the score record.'))
        return result.model_dump_json()

    fake_model(monkeypatch, reply, calls)
    root = make_run(rows=[review(0)])
    report = diagnose_benchmark(str(root), 'm', 'demo', options())
    row = SampleDiagnosis.model_validate_json((root / 'analysis/m/demo/diagnose/samples.jsonl').read_text())
    assert row.llm_status == 'invalid_evidence'
    assert row.evidence_errors == ['quote_missing_or_ambiguous']
    assert report.coverage.diagnosed_ok == 0
    assert (report.coverage.citations_valid, report.coverage.citations_total) == (1, 2)
    assert len(calls) == 1 and report.narrative_status == 'no_findings'


@pytest.mark.parametrize('unscored', [True, False])
@pytest.mark.parametrize('has_other_sample', [True, False])
def test_recorded_error_is_shown_without_spending_model_budget(
    make_run: Callable, monkeypatch: pytest.MonkeyPatch, unscored: bool, has_other_sample: bool,
) -> None:
    from evalscope.constants import ScoreStatus

    row = review(0)
    score = row.sample_score.score
    secret = 'diagnostic-test-secret-982613'
    monkeypatch.setenv('DIAGNOSIS_API_KEY', secret)
    reason = 'Distribution exceeds the probability sum tolerance.' if unscored else 'Sandbox pool is empty.'
    raw_reason = f'{reason} {secret}'
    kind = 'judge_failure' if unscored else 'environment_failure'
    if unscored:
        score.status = ScoreStatus.EXCLUDED
        score.value = {}
        score.explanation = raw_reason
    else:
        score.metadata = {'execution_result': {'status': 'error', 'error': raw_reason}}
    calls = []

    def reply(prompt: str, count: int) -> str:
        assert has_other_sample and reason not in prompt
        if 'Summarize evaluation' not in prompt:
            return verdict().model_dump_json()
        payload, _ = json.JSONDecoder().raw_decode(prompt[prompt.index('{"facts":'):])
        assert [item['sample']['index'] for item in payload['sample_hypotheses']] == [1]
        return json.dumps({'findings': [], 'recommendations': [], 'limitations': []})

    fake_model(monkeypatch, reply, calls)
    root = make_run(rows=[row, review(1)] if has_other_sample else [row])
    keys = [SampleKey(subset='default', index=i) for i in range(2 if has_other_sample else 1)]
    report = diagnose_benchmark(str(root), 'm', 'demo', options(samples=keys, max_diagnosed_samples=1))
    artifacts = (root / 'analysis/m/demo/diagnose/samples.jsonl').read_text()
    persisted = next(
        sample for line in artifacts.split('\n') if line
        if (sample := SampleDiagnosis.model_validate_json(line)).key.index == 0
    )
    assert reason in persisted.recorded_error and secret not in artifacts
    assert persisted.llm_status == 'not_selected' and persisted.llm is None and not persisted.sources
    assert kind in persisted.observed_types
    assert report.coverage.diagnosis_population == report.coverage.selected == int(has_other_sample)
    assert report.cost.calls == len(calls) == 2 * has_other_sample
    if not has_other_sample:
        assert report.cost.estimated_calls == 0 and report.narrative_status == 'no_findings'


def test_score_reason_is_citable_once_without_changing_saved_metadata(make_run: Callable) -> None:
    from evalscope.analysis.inputs import load_inputs

    row = review(0)
    reason = "Distribution exceeds this benchmark's probability sum tolerance."
    row.sample_score.score.explanation = reason
    row.sample_score.score.metadata = {'error': reason, 'metric_unavailable': True}
    root = make_run(rows=[row])
    config = options()
    inputs = load_inputs(str(root), 'm', 'demo', config.trace)
    item = inputs.samples[0]
    sources = build_sources(item, inputs, config)
    source = next(source for source in sources if source.id == 'score')
    parsed, errors = validate_evidence(
        verdict().model_copy(update={'evidence': [EvidenceReference(source_id='score', quote=reason)]}), sources
    )
    assert not errors and parsed.evidence[0].quote == reason
    assert json.loads(source.text)['metadata'] == {'metric_unavailable': True}
    assert item.record.sample_score.score.metadata['error'] == reason


def test_no_model_configuration_never_silently_calls_evaluation_model(make_run: Callable,monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('evalscope.analysis.llm.get_model',lambda *a,**kw:pytest.fail('No explicit model'))
    with pytest.raises(ValueError,match='explicit analysis model'):
        diagnose_benchmark(str(make_run()),'m','demo',AnalysisReportConfig(level='diagnose'))


def test_validation_does_not_turn_locatable_citations_into_semantic_accuracy() -> None:
    from evalscope.analysis.validation import DiagnosticLabel, evaluate_labels
    s=sample(0)
    s.llm=verdict()
    s.llm_status='ok'
    label=DiagnosticLabel(key=s.key,failure_type='knowledge_error',attribution='benchmark',
                          evidence_supported=False,rule_decisions={'absent.rule':True},annotator='blind-reviewer')
    result=evaluate_labels([s],[label])
    assert result['type_accuracy']==0 and result['citation_locatability']==1
    assert result['evidence_support_rate']==0 and result['experimental']
    assert result['rules']['absent.rule']['recall']==0
    assert result['attribution_confusion']=={'benchmark->model':1}
    with pytest.raises(ValueError,match='Duplicate'):
        evaluate_labels([s],[label,label])


def test_excerpting_preserves_the_answer_tail_and_records_real_generation_end(make_run: Callable) -> None:
    from evalscope.analysis.inputs import load_inputs
    root=make_run(rows=[review(0,0,output='head '+('long thought '*500)+' FINAL ANSWER: B')])
    data=load_inputs(str(root),'m','demo',options().trace)
    sources=build_sources(data.samples[0],data,options(max_input_chars=2000))
    by_id={s.id:s for s in sources}
    assert by_id['m:1'].excerpted and by_id['m:1:tail'].excerpted
    assert by_id['m:1:tail'].text.endswith('FINAL ANSWER: B')
    assert '"generation_stop_reason": "stop"' in by_id['score'].text
    assert by_id['m:1:tail'].original_start>0


def test_normal_generation_ending_cannot_be_explained_as_budget_truncation(make_run: Callable,monkeypatch: pytest.MonkeyPatch) -> None:
    calls=[]
    d=verdict().model_copy(update={'failure_type':'truncated'})
    fake_model(monkeypatch,lambda *args:d.model_dump_json(),calls)
    r=diagnose_benchmark(str(make_run()),'m','demo',options())
    assert r.coverage.diagnosed_ok==0
    assert r.coverage.diagnosis_statuses=={'unexplained':1}


def test_unknown_taxonomy_is_a_contract_error() -> None:
    from evalscope.api.judge import OutputContract
    d=verdict().model_dump()
    d['failure_type']='Output truncated'
    assert not OutputContract(schema_model=LLMDiagnosis).parse(json.dumps(d)).ok
    d = verdict().model_dump()
    d['gold_suspect'] = True
    assert OutputContract(schema_model=LLMDiagnosis).parse(json.dumps(d)).ok
    d['secondary_types'] = ['gold_suspect']
    assert not OutputContract(schema_model=LLMDiagnosis).parse(json.dumps(d)).ok


def test_narrative_references_are_rendered_once_and_labels_require_taxonomy_ids() -> None:
    from evalscope.analysis.validation import DiagnosticLabel
    from evalscope.api.analysis import NarrativeItem
    item=NarrativeItem(text='Check the output [F-truncation, P-abcd].',refs=['F-truncation'])
    assert item.text == 'Check the output.'
    item=NarrativeItem(text='Check the output (F-truncation, P-abcd).',refs=['F-truncation'])
    assert item.text == 'Check the output.'
    item=NarrativeItem(text='请核对（P-abcd）及【F-truncation】。',refs=['P-abcd'])
    assert item.text == '请核对及。'
    item=NarrativeItem(text='Errors (5/44, P-abcd).',refs=['P-abcd'])
    assert item.text == 'Errors (5/44).'
    with pytest.raises(ValueError,match='taxonomy IDs'):
        DiagnosticLabel(key=SampleKey(subset='s',index=1),failure_type='Output truncated',attribution='model',annotator='reviewer')


def test_trace_can_be_omitted_without_inventing_step_references(make_run: Callable) -> None:
    from evalscope.analysis.inputs import load_inputs
    from evalscope.api.agent.trace import AgentTrace, AgentTraceEvent
    root=make_run()
    data=load_inputs(str(root),'m','demo',options().trace)
    item=data.samples[0]
    item.record.agent_trace=AgentTrace(events=[AgentTraceEvent(type='model_generate',step=4,message_id=item.record.messages[1].id)])
    item.record.sample_score.score.metadata = {'tool_output': 'private middle step'}
    sources=build_sources(item,data,options(include_trace=False))
    assert not any('private middle step' in source.text for source in sources)
    assert all(source.step is None for source in sources)
    assert not any(source.id.startswith(('event:','calls:')) for source in sources)


def test_all_passing_without_findings_does_not_make_a_summary_request(make_run: Callable,monkeypatch: pytest.MonkeyPatch) -> None:
    calls=[]
    fake_model(monkeypatch,lambda *args:pytest.fail('No useful cited summary is possible'),calls)
    root=make_run(rows=[review(0,1,output='ANSWER: A',prediction='A')])
    r=diagnose_benchmark(str(root),'m','demo',options())
    assert r.coverage.selected==0 and r.cost.calls==0 and r.narrative_status=='no_findings'


def test_new_reports_record_the_current_taxonomy_version(make_run: Callable) -> None:
    from evalscope.api.analysis.taxonomy import TAXONOMY_VERSION
    r=diagnose_benchmark(str(make_run()),'m','demo',AnalysisReportConfig(trace={'use_providers':False}))
    assert r.taxonomy_version==TAXONOMY_VERSION


def test_a_single_remaining_source_character_cannot_include_the_whole_message(make_run: Callable) -> None:
    from evalscope.analysis.inputs import load_inputs

    data=load_inputs(str(make_run()),'m','demo',options().trace)
    item=data.samples[0]
    item.record.sample_score.score.metadata={'padding': ''}
    config=options(max_input_chars=2000)
    score=next(s for s in build_sources(item,data,config) if s.id=='score')
    item.record.sample_score.score.metadata['padding']='x' * (1994-len(score.text))
    sources=build_sources(item,data,config)
    assert sum(len(s.text) for s in sources) <= config.max_input_chars


def test_validation_retains_missing_roots_and_high_confidence_false_positives() -> None:
    from evalscope.analysis.validation import DiagnosticLabel, evaluate_labels
    from evalscope.api.analysis import Finding

    s=sample(0)
    s.llm=verdict().model_copy(update={'root_cause_step': 2})
    s.llm_status='ok'
    s.findings=[Finding(category='tool_failure',subtype='runtime_error',confidence='high',
                        detector='tool_failure',rule_id='model.bad_arguments')]
    label=DiagnosticLabel(key=s.key,failure_type='reasoning_error',attribution='model',
                          root_cause_step=2,rule_decisions={'model.bad_arguments':False},
                          annotator='human',user_reviewed=True)
    result=evaluate_labels([s],[label])
    assert result['type_accuracy']==1 and result['experimental']
    assert result['rules']['model.bad_arguments']['high_precision']==0
    missing=label.model_copy(update={'key':SampleKey(subset='large',index=1)})
    result=evaluate_labels([s],[label,missing])
    assert result['root_step_labeled']==2 and result['root_step_accuracy']==0.5


def test_complete_short_messages_include_the_middle_without_artificial_thirty_message_cutoff(make_run: Callable) -> None:
    from evalscope.analysis.inputs import load_inputs
    from evalscope.api.messages import ChatMessageAssistant

    data = load_inputs(str(make_run()), 'm', 'demo', options().trace)
    item = data.samples[0]
    item.record.messages = [ChatMessageAssistant(content=f'message {i}') for i in range(60)]
    sources = build_sources(item, data, options())
    assert {s.message_index for s in sources if s.id.startswith('m:')} == set(range(60))
    assert not any(s.excerpted for s in sources)


def test_long_input_keeps_middle_finding_recovery_and_final_answer(make_run: Callable) -> None:
    from evalscope.analysis.inputs import load_inputs
    from evalscope.api.analysis import Evidence, Finding
    from evalscope.api.messages import ChatMessageAssistant

    data = load_inputs(str(make_run()), 'm', 'demo', options().trace)
    item = data.samples[0]
    item.record.messages = [ChatMessageAssistant(content=f'message {i} ' + 'irrelevant ' * 300) for i in range(60)]
    item.record.messages[20].content = 'The tool failed.'
    item.record.messages[21].content = 'The retry succeeded; recovered.'
    item.record.messages[59].content = 'FINAL ANSWER: B'
    item.case.findings = [Finding(category='tool_failure', subtype='runtime_error', confidence='high',
                                  detector='test', evidence=[Evidence(message_index=20, excerpt='The tool failed.')])]
    sources = build_sources(item, data, options(max_input_chars=4000))
    by_id = {s.id: s for s in sources}
    assert by_id['m:20'].text == 'The tool failed.'
    assert by_id['m:21'].text == 'The retry succeeded; recovered.'
    assert by_id['m:59'].text == 'FINAL ANSWER: B'
    assert sum(len(source.text) for source in sources) <= 4000


def test_model_analysis_never_claims_population_cause_estimates(make_run: Callable, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    fake_model(monkeypatch, lambda prompt, n: json.dumps({'findings': [], 'recommendations': [], 'limitations': []})
               if 'Summarize evaluation' in prompt else verdict().model_dump_json(), calls)
    report = diagnose_benchmark(str(make_run()), 'm', 'demo', options(max_diagnosed_samples=None))
    assert report.coverage.selected == report.coverage.diagnosis_population == 1
    assert 'failed_unexplained_estimate' not in report.coverage.model_dump()
    assert all('estimated_share' not in pattern.model_dump() for pattern in report.patterns)


@pytest.mark.parametrize('value,expected', [('all', None), ('1', 1), ('137', 137)])
def test_cli_model_sample_budget_accepts_all_and_specific_counts(monkeypatch: pytest.MonkeyPatch, value: str, expected: object) -> None:
    from argparse import Namespace

    from evalscope.cli.analyze import AnalyzeCMD

    captured = []
    def diagnose(*args: object) -> list:
        captured.append(args[1])
        return []
    monkeypatch.setattr('evalscope.analysis.diagnose_run', diagnose)
    args = Namespace(analysis_config=None, level='facts', max_samples=value, run_dir='unused', model=None, benchmark=None, force=False)
    with pytest.raises(ValueError, match='No matching'):
        AnalyzeCMD(args).execute()
    assert captured[0].max_diagnosed_samples == expected


def test_explicit_passing_sample_is_retained_without_becoming_a_failure(make_run: Callable, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    fake_model(monkeypatch, lambda *args: verdict().model_dump_json(), calls)
    root = make_run(rows=[review(0, 1)])
    result = diagnose_benchmark(str(root), 'm', 'demo', options(samples=[SampleKey(subset='default', index=0)]))
    rows = [json.loads(line) for line in (root / 'analysis/m/demo/diagnose/samples.jsonl').read_text().split('\n') if line.strip()]
    assert result.coverage.task_failed == 0 and len(rows) == 1
    assert result.coverage.selection_scope == 'specified'
    assert rows[0]['task_success'] is True and rows[0]['observed_types'] == []
    assert rows[0]['llm']['failure_type'] == 'reasoning_error'


def test_trace_only_tool_evidence_is_not_lost_when_messages_do_not_contain_it(make_run: Callable) -> None:
    from evalscope.analysis.inputs import load_inputs
    from evalscope.api.agent.trace import AgentTrace, AgentTraceEvent

    data = load_inputs(str(make_run()), 'm', 'demo', options().trace)
    item = data.samples[0]
    item.record.agent_trace = AgentTrace(events=[
        AgentTraceEvent(type='tool_call', step=7, payload={'tool': 'search', 'args': {'query': 'specific'}}),
        AgentTraceEvent(type='tool_result', step=7, payload={'result': 'found evidence'}),
    ])
    sources = {source.id: source for source in build_sources(item, data, options())}
    assert sources['event:0'].step == 7 and 'specific' in sources['event:0'].text
    assert sources['event:1'].step == 7 and 'found evidence' in sources['event:1'].text


def test_agreeing_on_unknown_never_certifies_model_explanation_accuracy() -> None:
    from evalscope.analysis.validation import DiagnosticLabel, evaluate_labels

    record = sample(0)
    label = DiagnosticLabel(key=record.key, failure_type='unexplained', attribution='unknown',
                            evidence_supported=True, annotator='human', user_reviewed=True)
    metrics = evaluate_labels([record], [label])
    assert metrics['type_accuracy'] == 1
    assert metrics['proposal_type_accuracy'] is None and metrics['proposal_coverage'] == 0
    assert metrics['experimental']


@pytest.mark.parametrize('citation_kind', ['sample', 'group', 'foreign_sample'])
def test_summary_excludes_passing_observations_and_keeps_outcomes_separate(
    make_run: Callable, monkeypatch: pytest.MonkeyPatch, citation_kind: str,
) -> None:
    from evalscope.analysis.llm import summarize
    from evalscope.api.analysis import Fact

    root = make_run()
    report = diagnose_benchmark(str(root), 'm', 'demo', AnalysisReportConfig(trace={'use_providers': False}))
    report.facts.append(Fact(id='F-safety', kind='agent', title='Safety signals', definition='Benchmark signal',
                             numerator=1, denominator=32))
    rows = [SampleDiagnosis(key={'subset': 'main', 'index': i}, task_success=True, score_status='success',
                            observed_types=['tool_error' if i < 29 else 'loop_or_budget']) for i in range(30)]
    rows += [SampleDiagnosis(key={'subset': 'main', 'index': 30}, task_success=False, score_status='success', observed_types=['tool_error']),
             SampleDiagnosis(key={'subset': 'main', 'index': 31}, score_status='parse_error', observed_types=['tool_error'])]
    rows[-2].llm = verdict()
    rows[-2].llm_status = 'ok'
    rows[-2].sources = [DiagnosticSource(id='m:1', text='ANSWER: B', step=7)]
    rows[0].llm = verdict()
    rows[0].llm_status = 'invalid_evidence'
    captured = []
    class Reviewer:
        def generate(self, messages: list) -> ModelOutput:
            captured.extend(messages)
            payload, _ = json.JSONDecoder().raw_decode(messages[-1].text[messages[-1].text.index('{"facts":'):])
            refs = payload['sample_hypotheses'][0]['refs'] if citation_kind == 'sample' else ['P-tools' if citation_kind == 'group' else 'S-not-supplied']
            return ModelOutput.from_content(model='reviewer', content=json.dumps({
                'findings': [{'text': 'One failed task had a tool error.', 'refs': refs}],
                'recommendations': [], 'limitations': [],
            }))
    monkeypatch.setattr('evalscope.analysis.llm.get_model', lambda *args, **kwargs: Reviewer())
    def write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    summarize(report, rows, options(language='zh'), root / 'summary-test', False, write)
    assert 'Write explanations, narrative text and limitations in Chinese' in captured[0].text
    prompt = captured[-1].text
    payload, _ = json.JSONDecoder().raw_decode(prompt[prompt.index('{"facts":'):])
    assert 'F-safety' in {fact['id'] for fact in payload['facts']}
    assert 'skipped_facts' not in payload['coverage'] and 'diagnosed_ok' not in payload['coverage']
    schema = LLMDiagnosis.model_json_schema()
    assert schema['properties']['secondary_types']['items']['enum'] == schema['properties']['failure_type']['enum']
    assert 'patterns' not in payload
    example = payload['sample_hypotheses'][0]
    assert len(payload['sample_hypotheses']) == 1
    assert example['sample'] == {'subset': 'main', 'index': 30}
    assert example['hypothesis'] == verdict().model_dump(mode='json')
    ref = example['refs'][0]
    assert ref.startswith('S-')
    if citation_kind != 'sample':
        assert report.narrative_status == 'invalid_references'
        assert report.narrative_samples == {}
        return
    assert report.narrative_samples[ref].sample.index == 30
    assert report.narrative_samples[ref].step == 7
    assert report.narrative.findings[0].refs == [ref]
    from evalscope.analysis.diagnosis import render_markdown
    markdown = render_markdown(report, 'zh')
    assert '[main · 样本 30 · 步骤 7]' in markdown
    assert ref not in markdown
    assert report.narrative_status == 'ok'


def test_analysis_config_and_cli_expose_only_basic_and_model_analysis() -> None:
    from argparse import ArgumentParser

    from evalscope.api.analysis import DiagnosisReport
    from evalscope.cli.analyze import AnalyzeCMD

    assert AnalysisReportConfig().level == 'facts'
    for schema in (AnalysisReportConfig.model_json_schema(), DiagnosisReport.model_json_schema()):
        assert schema['properties']['level']['enum'] == ['facts', 'diagnose']
    with pytest.raises(ValueError):
        AnalysisReportConfig(level='full')
    parser = ArgumentParser()
    AnalyzeCMD.define_args(parser.add_subparsers())
    for level in ('facts', 'diagnose'):
        assert parser.parse_args(['analyze', '--run-dir', 'run', '--level', level]).level == level
    with pytest.raises(SystemExit):
        parser.parse_args(['analyze', '--run-dir', 'run', '--level', 'full'])


@pytest.mark.parametrize('failure', ['parse', 'transport'])
def test_summary_failure_preserves_sample_diagnosis_and_resumes_without_reanalyzing_samples(
    make_run: Callable, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    root = make_run()
    original = (root / 'reviews/m/demo_default.jsonl').read_bytes()
    calls = []
    def reply(prompt: str, n: int) -> str:
        if 'Summarize evaluation' in prompt:
            if failure == 'transport':
                raise ConnectionError('Scripted summary failure')
            return 'Not JSON'
        return verdict().model_dump_json()
    fake_model(monkeypatch, reply, calls)
    report = diagnose_benchmark(str(root), 'm', 'demo', options())
    assert report.level == 'diagnose' and report.coverage.diagnosed_ok == 1
    assert report.narrative_status == ('parse_error' if failure == 'parse' else 'transport_error')
    assert report.cost.calls == 2
    resumed = diagnose_benchmark(str(root), 'm', 'demo', options())
    assert resumed.coverage.diagnosed_ok == 1
    assert resumed.cost.calls == resumed.cost.cached_calls == 1
    assert sum('Summarize evaluation' not in prompt for prompt in calls) == 1
    assert (root / 'reviews/m/demo_default.jsonl').read_bytes() == original


@pytest.mark.parametrize('include_trace,redacted,expected', [(True, False, True), (False, False, False), (True, True, False)])
@pytest.mark.parametrize('agent', [False, True])
def test_saved_reasoning_is_citable_and_omissions_are_explicit(
    make_run: Callable, monkeypatch: pytest.MonkeyPatch, include_trace: bool, redacted: bool, expected: bool, agent: bool,
) -> None:
    from evalscope.api.agent.trace import AgentTrace, AgentTraceEvent
    from evalscope.api.messages import ChatMessageAssistant

    row = review(0)
    row.messages[-1] = ChatMessageAssistant(content=[
        {'type': 'reasoning', 'reasoning': 'I calculate 2 + 2 = 5.', 'redacted': redacted},
        {'type': 'text', 'text': 'ANSWER: B'},
    ])
    if agent:
        row.agent_trace = AgentTrace(events=[AgentTraceEvent(type='model_generate', step=4, message_id=row.messages[-1].id)])
    calls = []
    d = verdict().model_copy(update={'evidence': [EvidenceReference(source_id='reasoning:1:0', quote='2 + 2 = 5')]})
    fake_model(monkeypatch, lambda prompt, n: json.dumps({'findings': [], 'recommendations': [], 'limitations': []})
               if 'Summarize evaluation' in prompt else d.model_dump_json(), calls)
    root = make_run(rows=[row])
    report = diagnose_benchmark(str(root), 'm', 'demo', options(include_trace=include_trace))
    persisted = SampleDiagnosis.model_validate_json((root / 'analysis/m/demo/diagnose/samples.jsonl').read_text())
    assert ('I calculate 2 + 2 = 5.' in calls[0]) is expected
    assert persisted.context_omitted is (not expected)
    if expected:
        assert report.coverage.diagnosed_ok == 1
        assert persisted.llm.evidence[0].quote == '2 + 2 = 5'
        source = next(s for s in persisted.sources if s.id == 'reasoning:1:0')
        assert source.message_index == 1 and source.step == (4 if agent else None)
    else:
        assert persisted.llm_status == 'invalid_evidence'


@pytest.mark.parametrize('media', [
    {'type': 'image', 'image': '/not-read/image.png'},
    {'type': 'audio', 'audio': '/not-read/audio.wav', 'format': 'wav'},
    {'type': 'video', 'video': '/not-read/video.mp4', 'format': 'mp4'},
    {'type': 'data', 'data': {'secret_payload': 'MUST_NOT_SEND'}},
])
def test_media_is_not_silently_treated_as_complete_text_context(
    make_run: Callable, monkeypatch: pytest.MonkeyPatch, media: dict,
) -> None:
    from evalscope.api.messages import ChatMessageUser

    row = review(0)
    row.messages[0] = ChatMessageUser(content=[{'type': 'text', 'text': 'Inspect this input.'}, media])
    calls = []
    fake_model(monkeypatch, lambda prompt, n: json.dumps({'findings': [], 'recommendations': [], 'limitations': []})
               if 'Summarize evaluation' in prompt else verdict().model_dump_json(), calls)
    root = make_run(rows=[row])
    diagnose_benchmark(str(root), 'm', 'demo', options())
    persisted = SampleDiagnosis.model_validate_json((root / 'analysis/m/demo/diagnose/samples.jsonl').read_text())
    assert persisted.context_omitted
    note = next(s for s in persisted.sources if s.id == 'omitted-content')
    assert media['type'] in note.text
    assert '/not-read/' not in calls[0] and 'MUST_NOT_SEND' not in calls[0]


def test_summary_considers_all_valid_hypotheses_instead_of_the_display_example_limit(
    make_run: Callable, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evalscope.analysis.llm import summarize

    root = make_run()
    report = diagnose_benchmark(str(root), 'm', 'demo', AnalysisReportConfig(trace={'use_providers': False}))
    rows = []
    for i in range(60):
        row = sample(i)
        row.llm = verdict().model_copy(update={'failure_type': 'environment_failure' if i == 59 else 'reasoning_error'})
        row.llm_status = 'ok'
        row.sources = [DiagnosticSource(id='m:1', text='ANSWER: B')]
        rows.append(row)
    calls = []
    fake_model(monkeypatch, lambda prompt, n: json.dumps({'findings': [], 'recommendations': [], 'limitations': []}), calls)
    summarize(report, rows, options(max_examples=1, max_input_chars=2000), root, False, lambda *a: None)
    payload, _ = json.JSONDecoder().raw_decode(calls[0][calls[0].index('{"facts":'):])
    assert len(json.dumps(payload, ensure_ascii=False)) <= 2000
    counts = payload['inspected_hypotheses']
    assert counts == {'reasoning_error': 59, 'environment_failure': 1}
    assert {h['hypothesis']['failure_type'] for h in payload['sample_hypotheses']} == set(counts)
    assert any(h['sample']['index'] == 59 for h in payload['sample_hypotheses'])


def test_unknown_continuous_outcomes_are_not_silently_counted_as_failure_candidates(
    make_run: Callable, monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = review(0, value=0.6)
    calls = []
    fake_model(monkeypatch, lambda *a: pytest.fail('No failure candidates'), calls)
    root = make_run(rows=[row])
    report = diagnose_benchmark(str(root), 'm', 'demo', options())
    assert report.coverage.selected == 0 and report.coverage.outcome_unknown == 1
    assert 'selection-outcome-unknown' in report.coverage.skipped_facts


def test_low_rate_failures_remain_available_to_contextualize_a_hypothesis(
    make_run: Callable, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evalscope.analysis.llm import summarize

    root = make_run(rows=[review(0)] + [review(i, 1, output='ANSWER: A', prediction='A') for i in range(1, 50)])
    report = diagnose_benchmark(str(root), 'm', 'demo', AnalysisReportConfig(trace={'use_providers': False}))
    calls = []
    fake_model(monkeypatch, lambda *args: json.dumps({
        'findings': [{'text': 'One of fifty tasks did not pass; its cause remains unverified.', 'refs': ['F-task-failure']}],
        'recommendations': [], 'limitations': [],
    }), calls)
    row = sample(0)
    row.llm = verdict()
    row.llm_status = 'ok'
    summarize(report, [row], options(), root, False, lambda *a: None)
    payload, _ = json.JSONDecoder().raw_decode(calls[0][calls[0].index('{"facts":'):])
    fact = next(f for f in payload['facts'] if f['id'] == 'F-task-failure')
    assert fact['numerator'] == 1 and fact['denominator'] == 50
    assert len(payload['sample_hypotheses']) == 1 and report.narrative_status == 'ok'


@pytest.mark.parametrize('fixture', ['gsm8k_invalid_type.txt', 'gsm8k_invalid_json.txt'])
def test_real_gsm8k_bad_replies_remain_rejected_and_are_not_cached(
    make_run: Callable, monkeypatch: pytest.MonkeyPatch, fixture: str,
) -> None:
    raw = (Path(__file__).with_name('fixtures') / fixture).read_text()
    calls = []
    def reply(prompt: str, n: int) -> str:
        if 'Summarize evaluation' in prompt:
            return json.dumps({'findings': [], 'recommendations': [], 'limitations': []})
        return raw
    fake_model(monkeypatch, reply, calls)
    root = make_run()
    report = diagnose_benchmark(str(root), 'm', 'demo', options())
    assert report.coverage.diagnosis_statuses == {'parse_error': 1}
    assert report.coverage.diagnosed_ok == 0
    diagnose_benchmark(str(root), 'm', 'demo', options())
    assert sum('Summarize evaluation' not in prompt for prompt in calls) == 2
    prompt = calls[0]
    schema = json.loads(prompt[prompt.index('\nJSON schema:\n') + len('\nJSON schema:\n'):])
    assert schema['additionalProperties'] is False
    assert 'gold_suspect' not in schema['properties']['secondary_types']['items']['enum']
    assert '(additional keys are allowed)' not in prompt


def test_summary_uses_short_whitelisted_refs_without_repairing_real_reference_typos(
    make_run: Callable, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evalscope.analysis.llm import summarize

    root = make_run()
    report = diagnose_benchmark(str(root), 'm', 'demo', AnalysisReportConfig(trace={'use_providers': False}))
    row = sample(37, 'Level 5')
    row.llm = verdict()
    row.llm_status = 'ok'
    row.sources = [DiagnosticSource(id='m:1', text='ANSWER: B')]
    actual_bad = json.loads((Path(__file__).with_name('fixtures') / 'math_invalid_reference.json').read_text())
    calls = []
    def reply(prompt: str, n: int) -> str:
        schema = json.loads(prompt[prompt.index('\nJSON schema:\n') + len('\nJSON schema:\n'):])
        allowed = schema['$defs']['NarrativeItem']['properties']['refs']['items']['enum']
        assert 'S-1' in allowed and actual_bad['refs'][0] not in allowed
        return json.dumps({
            'findings': [{'text': 'Inspect this sample.', 'refs': ['S-1']}],
            'recommendations': [actual_bad, {'text': 'Do not keep a partly invented citation.', 'refs': ['S-1', 'S-11']}],
            'limitations': [],
        })
    fake_model(monkeypatch, reply, calls)
    summarize(report, [row], options(), root, False, lambda *a: None)
    assert report.narrative_status == 'partial_references'
    assert report.narrative.recommendations == []
    assert report.narrative_samples['S-1'].sample == row.key
    assert set(report.narrative_samples) == {'S-1'}


@pytest.mark.parametrize('legacy_layout', [False, True])
def test_new_facts_keep_model_analysis_and_hide_it_when_sources_change(
    make_run: Callable, monkeypatch: pytest.MonkeyPatch, legacy_layout: bool
) -> None:
    from evalscope.report.diagnosis import load_diagnosis_snapshot

    calls = []
    fake_model(
        monkeypatch, lambda prompt, _: '{}' if 'Summarize evaluation' in prompt else verdict().model_dump_json(), calls
    )
    root = make_run(rows=[review(0)])
    model = diagnose_benchmark(str(root), 'm', 'demo', options())
    assert model.coverage.diagnosed_ok == 1
    base = root / 'analysis/m/demo'
    target = base / 'diagnose'
    original = (target / 'samples.jsonl').read_bytes()
    if legacy_layout:
        for file in ('diagnosis.json', 'samples.jsonl'):
            (target / file).rename(base / file)
    report_path = root / 'reports/m/demo.json'
    before_report = json.loads(report_path.read_text())

    facts = diagnose_benchmark(str(root), 'm', 'demo', AnalysisReportConfig(trace=options().trace), force=True)
    assert facts.level == 'facts' and facts.cost.calls == 0
    assert (target / 'samples.jsonl').read_bytes() == original and len(calls) == 2
    assert load_diagnosis_snapshot(str(root), 'm', 'demo') == ('ready', model, target)
    assert json.loads(report_path.read_text()) == before_report

    (root / 'reviews/m/demo_default.jsonl').write_text('')
    assert load_diagnosis_snapshot(str(root), 'm', 'demo')[0] == 'stale'
    new_facts = diagnose_benchmark(str(root), 'm', 'demo')
    assert load_diagnosis_snapshot(str(root), 'm', 'demo') == ('ready', new_facts, base)
    assert (target / 'samples.jsonl').read_bytes() == original
    after_report = json.loads(report_path.read_text())
    assert {k: v for k, v in after_report.items() if k != 'analysis'} == {
        k: v for k, v in before_report.items() if k != 'analysis'
    }


def test_web_reuses_cli_facts_settings_for_both_analysis_levels(
    make_run: Callable, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evalscope.service.app import create_app
    from evalscope.service.blueprints import reports

    records = [review(0), review(1, 1)]
    for row, score in zip(records, (0.6, 1.0)):
        row.sample_score.score.value = {'quality': score}
        row.sample_score.score.main_score_name = 'quality'
    root = make_run(rows=records)
    original_reviews = (root / 'reviews/m/demo_default.jsonl').read_bytes()
    facts_config = AnalysisReportConfig(
        trace={'outcome_thresholds': {'demo': {'quality': 0.8}}}, max_examples=1
    )
    baseline = diagnose_benchmark(str(root), 'm', 'demo', facts_config)

    def inline(func: Callable, *args: object) -> object:
        return func(*args)

    monkeypatch.setattr(reports, 'run_in_subprocess', inline)
    calls = []
    fake_model(
        monkeypatch, lambda prompt, _: '{}' if 'Summarize evaluation' in prompt else verdict().model_dump_json(), calls
    )
    client = create_app(outputs=str(root.parent)).test_client()
    path = '/api/v1/reports/runs/run/models/m/diagnosis?dataset_name=demo'
    result = client.post(path, json={
        'level': 'diagnose', 'model_id': 'reviewer', 'api_url': 'http://localhost/v1',
        'api_key': 'diagnostic-credential', 'max_diagnosed_samples': 1,
    })
    assert result.status_code == 200
    analysis = result.json['report']
    assert analysis['level'] == 'diagnose'
    assert analysis['facts'] == [fact.model_dump(mode='json') for fact in baseline.facts]
    assert analysis['coverage']['task_success_known'] == 2
    assert analysis['coverage']['task_failed'] == 1
    assert analysis['coverage']['selected'] == analysis['coverage']['diagnosed_ok'] == 1
    assert 'facts_config' not in client.get(path).json['report']
    paid_artifact = root / 'analysis/m/demo/diagnose/diagnosis.json'
    paid_snapshot = paid_artifact.read_bytes()
    assert b'diagnostic-credential' not in paid_snapshot

    refreshed = client.post(path, json={})
    assert refreshed.status_code == 200 and refreshed.json['report']['level'] == 'diagnose'
    assert len(calls) == 2 and paid_artifact.read_bytes() == paid_snapshot

    updated = AnalysisReportConfig(trace={'outcome_thresholds': {'demo': {'quality': 0.5}}})
    diagnose_benchmark(str(root), 'm', 'demo', updated)
    current = client.get(path).json['report']
    assert current['level'] == 'facts' and current['coverage']['task_failed'] == 0
    assert paid_artifact.read_bytes() == paid_snapshot
    assert (root / 'reviews/m/demo_default.jsonl').read_bytes() == original_reviews
