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
from evalscope.api.model import ModelOutput, ModelUsage
from tests.analysis.helpers import review


def options(**kw: object) -> AnalysisReportConfig:
    return AnalysisReportConfig(level='full', model={'model_id':'independent-reviewer'}, trace={'use_providers':False}, **kw)


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


def sample(index: int,subset: str='large',failed: bool=True) -> SampleDiagnosis:
    return SampleDiagnosis(key=SampleKey(subset=subset,index=index),task_success=not failed,
                           score_status='success',needs_attention=True,failure_type='unexplained')


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


def test_full_diagnosis_preserves_scores_and_resumes_per_request_cache(make_run: Callable,monkeypatch: pytest.MonkeyPatch) -> None:
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
    rows=[json.loads(line) for line in (root/'analysis/m/demo/samples.jsonl').read_text().split('\n') if line.strip()]
    assert rows[0]['llm']['evidence'][0]['quote']=='ANSWER: B'
    assert rows[0]['failure_type'] == 'unexplained' and rows[0]['observed_types'] == []
    assert rows[0]['llm']['failure_type'] == 'reasoning_error'
    assert report.patterns[0].failure_type == 'unexplained' and report.patterns[0].source == 'none'
    (root/'analysis/m/demo/diagnosis.json').unlink()
    resumed=diagnose_benchmark(str(root),'m','demo',options())
    assert resumed.cost.calls==0 and resumed.cost.cached_calls==2 and len(calls)==2
    changed=diagnose_benchmark(str(root),'m','demo',options(seed=43))
    assert changed.config_fingerprint!=report.config_fingerprint


@pytest.mark.parametrize('mode,status', [('parse','parse_error'),('transport','transport_error'),('evidence','invalid_evidence'),('low','unexplained')])
def test_failed_diagnoses_never_count_as_explained(make_run: Callable,monkeypatch: pytest.MonkeyPatch,mode: str,status: str) -> None:
    root=make_run()
    calls=[]
    def reply(prompt: str,n: int) -> str:
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
    report=diagnose_benchmark(str(root),'m','demo',options().model_copy(update={'level':'diagnose'}))
    assert report.coverage.diagnosis_statuses=={status:1}
    assert report.coverage.diagnosed_ok == 0 and report.coverage.unexplained_failures == 1
    assert report.patterns[0].failure_type=='unexplained'
    if mode in ('parse','transport'):
        diagnose_benchmark(str(root),'m','demo',options().model_copy(update={'level':'diagnose'}))
        assert len(calls)==2


def test_no_model_configuration_never_silently_calls_evaluation_model(make_run: Callable,monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('evalscope.analysis.llm.get_model',lambda *a,**kw:pytest.fail('No explicit model'))
    with pytest.raises(ValueError,match='explicit analysis model'):
        diagnose_benchmark(str(make_run()),'m','demo',AnalysisReportConfig(level='full'))


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
    sources=build_sources(item,data,options(include_trace=False))
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


def test_full_model_analysis_never_claims_population_cause_estimates(make_run: Callable, monkeypatch: pytest.MonkeyPatch) -> None:
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
    rows = [json.loads(line) for line in (root / 'analysis/m/demo/samples.jsonl').read_text().split('\n') if line.strip()]
    assert result.coverage.task_failed == 0 and len(rows) == 1
    assert result.coverage.selection_scope == 'specified'
    assert rows[0]['task_success'] is True and rows[0]['failure_type'] is None
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


def test_summary_excludes_passing_observations_and_keeps_outcomes_separate(
    make_run: Callable, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evalscope.analysis.llm import summarize
    from evalscope.api.analysis import Fact, FailurePattern

    root = make_run()
    report = diagnose_benchmark(str(root), 'm', 'demo', AnalysisReportConfig(trace={'use_providers': False}))
    report.facts.append(Fact(id='F-safety', kind='agent', title='Safety signals', definition='Benchmark signal',
                             numerator=1, denominator=32))
    report.patterns = [FailurePattern(id='P-tools', failure_type='tool_error', attribution='unknown',
                                    count=30, denominator=31, share=30 / 31),
                       FailurePattern(id='P-passing', failure_type='loop_or_budget', attribution='unknown',
                                    count=1, denominator=31, share=1 / 31)]
    rows = [SampleDiagnosis(key={'subset': 'main', 'index': i}, task_success=True, score_status='success',
                            observed_types=['tool_error' if i < 29 else 'loop_or_budget']) for i in range(30)]
    rows += [SampleDiagnosis(key={'subset': 'main', 'index': 30}, task_success=False, score_status='success', observed_types=['tool_error']),
             SampleDiagnosis(key={'subset': 'main', 'index': 31}, score_status='parse_error', observed_types=['tool_error'])]
    rows[-2].llm = verdict()
    rows[-2].llm_status = 'ok'
    rows[0].llm = verdict()
    rows[0].llm_status = 'invalid_evidence'
    captured = []
    class Reviewer:
        def generate(self, messages: list) -> ModelOutput:
            captured.extend(messages)
            return ModelOutput.from_content(model='reviewer', content=json.dumps({
                'findings': [{'text': 'One failed task had a tool error.', 'refs': ['P-tools']}],
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
    assert payload['patterns'] == [{'id': 'P-tools', 'observation_type': 'tool_error'}]
    assert payload['sample_hypotheses'] == [{'sample': {'subset': 'main', 'index': 30},
                                            'refs': ['P-tools'], 'hypothesis': verdict().model_dump(mode='json')}]
    assert report.narrative_status == 'ok'
