"""Offline tests for the ACEBench adapter.

Everything here runs without a network call or a live model: the checkers are exercised directly,
and the agent rollouts are driven by scripted models against the vendored scenario classes.
"""
import copy
import json
import os
import re
import sys
from typing import Any, Dict, List

import pytest

from evalscope.api.messages import ChatMessage
from evalscope.api.metric import SampleScore, Score
from evalscope.api.model import ChatCompletionChoice, GenerateConfig, Model, ModelAPI, ModelOutput
from evalscope.benchmarks.acebench.checker import (
    check_agent_end_state,
    check_normal_answer,
    check_special_answer,
    milestone_accuracy,
    multi_turn_accuracy,
)
from evalscope.benchmarks.acebench.parser import CallFormatError, decode_calls
from evalscope.benchmarks.acebench.prompts import build_single_turn_prompts
from evalscope.benchmarks.acebench.rollout import run_rollout
from evalscope.benchmarks.acebench.utils import category_of_record, resolve_categories, split_of_category

# ---------------------------------------------------------------------------
# Milestone scoring, the regression guard for issue #1538
# ---------------------------------------------------------------------------


def test_empty_milestones_are_vacuously_satisfied():
    assert milestone_accuracy([], []) == 1.0
    assert milestone_accuracy([], [[]]) == 1.0


@pytest.mark.parametrize(
    'milestones',
    [
        ['not a call'],
        [{'unexpected_schema': 'value'}],
        [['not a call'], ['[a()]']],
    ],
)
def test_unmatched_milestones_score_zero_not_one(milestones):
    """Milestones that exist but are never matched must not read as a perfect score."""
    assert milestone_accuracy([], milestones) == 0.0


def test_milestone_denominator_is_the_declared_count():
    """A milestone that cannot be matched lowers the score instead of leaving the denominator."""
    milestones = [
        "[login_food_platform(username='Eve', password='password123')]",
        # Not valid Python: the ground truth contains unescaped apostrophes.
        "[add_food_delivery_order(username='Eve', merchant_name='Domino's', items=[])]",
        "[add_reminder(title='Today's spending', description='88.0 yuan', time='2024-07-15 09:30')]",
    ]
    assert milestone_accuracy(milestones[:1], milestones) == pytest.approx(0.333)
    assert milestone_accuracy(milestones, milestones) == 1.0


def test_milestones_are_matched_in_order():
    milestones = ['[a()]', '[b()]', '[c()]']
    assert milestone_accuracy(['[a()]', '[b()]', '[c()]'], milestones) == 1.0
    assert milestone_accuracy(['[c()]', '[b()]', '[a()]'], milestones) == pytest.approx(0.333)
    # Extra calls in between are tolerated as long as the milestones still appear in order.
    assert milestone_accuracy(['[x()]', '[a()]', '[y()]', '[b()]', '[c()]'], milestones) == 1.0


def test_milestone_cursor_never_rewinds():
    """The scan is single-pass, so a milestone skipped once cannot be picked up later."""
    milestones = ['[a()]', '[b()]', '[c()]']
    assert milestone_accuracy(['[x()]', '[a()]', '[y()]', '[c()]'], milestones) == pytest.approx(0.333)


def test_best_milestone_candidate_wins():
    candidates = [['[a()]', '[b()]'], ['[c()]']]
    assert milestone_accuracy(['[c()]'], candidates) == 1.0


# ---------------------------------------------------------------------------
# Output decoding follows the official contract
# ---------------------------------------------------------------------------


def test_decode_bracketed_call_list():
    assert decode_calls("[Api(a='b', c=2)]", 'normal_atom_bool') == [{'Api': {'a': 'b', 'c': 2}}]


def test_decode_ignores_text_around_the_call_list():
    assert decode_calls("Sure! [Api(a='b')] hope that helps", 'normal_atom_bool') == [{'Api': {'a': 'b'}}]


def test_decode_keeps_nested_structures():
    decoded = decode_calls("[Api(items=[{'product': 'x', 'quantity': 1}])]", 'normal_atom_list')
    assert decoded == [{'Api': {'items': [{'product': 'x', 'quantity': 1}]}}]


def test_decode_accepts_a_call_without_arguments():
    assert decode_calls('[Api()]', 'normal_atom_bool') == [{'Api': {}}]


@pytest.mark.parametrize('text', ["Api(a='b')", 'not a call at all', '', 'I cannot help with that'])
def test_undecodable_output_is_rejected(text):
    """The official evaluator scores such answers zero rather than rescuing them."""
    with pytest.raises(CallFormatError):
        decode_calls(text, 'normal_atom_bool')


def test_multi_turn_decoding_strips_whitespace_like_upstream():
    assert decode_calls("[Api(city='New York')]", 'normal_multi_turn_user_adjust') == [{'Api': {'city': 'NewYork'}}]


# ---------------------------------------------------------------------------
# Normal answer checking, mirroring model_eval/checker.py
# ---------------------------------------------------------------------------

FUNCTIONS = [{
    'name': 'book',
    'description': 'Book a room.',
    'parameters': {
        'type': 'object',
        'properties': {
            'city': {'type': 'string', 'description': 'City name.'},
            'nights': {'type': 'number', 'description': 'Number of nights.'},
            'smoking': {'type': 'boolean', 'description': 'Smoking room.'},
            'guests': {'type': 'array', 'items': {'type': 'string'}, 'description': 'Guest names.'},
            'contact': {
                'type': 'object',
                'description': 'Contact details.',
                'properties': {'email': {'type': 'string', 'description': 'Email.'}},
            },
        },
        'required': ['city'],
    },
}]

GROUND_TRUTH = {
    'book': {
        'city': 'Athens',
        'nights': 3,
        'smoking': False,
        'guests': ['Alice', 'Bob'],
        'contact': {'email': 'a@b.c'},
    }
}


def check(arguments: Dict[str, Any], category: str = 'normal_atom_object_short') -> bool:
    return check_normal_answer(FUNCTIONS, [{'book': arguments}], GROUND_TRUTH, category)['valid']


def test_exact_answer_passes():
    assert check(dict(GROUND_TRUTH['book']))


def test_optional_parameter_may_be_omitted():
    """Upstream only insists on the parameters the schema marks required."""
    arguments = dict(GROUND_TRUTH['book'])
    arguments.pop('nights')
    assert check(arguments)


def test_required_parameter_may_not_be_omitted():
    arguments = dict(GROUND_TRUTH['book'])
    arguments.pop('city')
    assert not check(arguments)


def test_parameter_outside_the_ground_truth_is_rejected():
    arguments = dict(GROUND_TRUTH['book'], unexpected='x')
    assert not check(arguments)


def test_string_is_compared_as_a_substring_for_normal_data():
    assert check(dict(GROUND_TRUTH['book'], city='Athens, Greece'))
    assert not check(dict(GROUND_TRUTH['book'], city='Ath'))


def test_list_order_matters():
    assert not check(dict(GROUND_TRUTH['book'], guests=['Bob', 'Alice']))


def test_boolean_written_as_text_is_accepted():
    assert check(dict(GROUND_TRUTH['book'], smoking='false'))


def test_boolean_values_are_only_type_checked():
    """Upstream has no value check for booleans, so a flipped flag still passes.

    Reproduced deliberately: changing it would move the reported numbers away from the official
    leaderboard.
    """
    assert check(dict(GROUND_TRUTH['book'], smoking=True))
    # A non-boolean in a boolean field is still rejected.
    assert not check(dict(GROUND_TRUTH['book'], smoking=3))


def test_wrong_function_name_is_rejected():
    assert not check_normal_answer(FUNCTIONS, [{'reserve': GROUND_TRUTH['book']}], GROUND_TRUTH, 'x')['valid']


def test_call_count_must_match():
    calls = [{'book': dict(GROUND_TRUTH['book'])}] * 2
    assert not check_normal_answer(FUNCTIONS, calls, GROUND_TRUTH, 'x')['valid']


def test_any_candidate_answer_may_match():
    candidates = [{'book': {'city': 'Rome'}}, GROUND_TRUTH]
    result = check_normal_answer(FUNCTIONS, [{'book': dict(GROUND_TRUTH['book'])}], candidates, 'x')
    assert result['valid']


def test_parallel_calls_ignore_the_index_suffix():
    """Ground truth stores repeated calls as ``name`` and ``name_1``."""
    ground_truth = {'book': {'city': 'Athens'}, 'book_1': {'city': 'Rome'}}
    calls = [{'book': {'city': 'Athens'}}, {'book': {'city': 'Rome'}}]
    assert check_normal_answer(FUNCTIONS, calls, ground_truth, 'x')['valid']


# ---------------------------------------------------------------------------
# Special categories are graded on their diagnostic wording
# ---------------------------------------------------------------------------


def test_special_incomplete_requires_the_contract_sentence():
    ground_truth = {'RealEstateManager_manageProperty': ['propertyDetails', 'rentalManagement']}
    good = ('["Missing necessary parameters (propertyDetails, rentalManagement) '
            'for the api (RealEstateManager_manageProperty)"]')
    assert check_special_answer(good, ground_truth, 'special_incomplete')['valid']
    assert not check_special_answer('[RealEstateManager_manageProperty()]', ground_truth,
                                    'special_incomplete')['valid']


def test_special_incomplete_needs_every_missing_parameter():
    ground_truth = {'Api': ['alpha', 'beta']}
    partial = '["Missing necessary parameters (alpha) for the api (Api)"]'
    assert not check_special_answer(partial, ground_truth, 'special_incomplete')['valid']


def test_special_error_param_requires_the_offending_value():
    ground_truth = {'plan_id': ['1234WrongID']}
    good = '["There is incorrect value (1234WrongID) for the parameters (plan_id)."]'
    assert check_special_answer(good, ground_truth, 'special_error_param')['valid']
    assert not check_special_answer('["There is incorrect value somewhere."]', ground_truth,
                                    'special_error_param')['valid']


def test_special_irrelevant_requires_the_refusal():
    text = '["Due to the limitations of the function, I cannot solve this problem."]'
    assert check_special_answer(text, {}, 'special_irrelevant')['valid']
    assert not check_special_answer('[SomeApi(x=1)]', {}, 'special_irrelevant')['valid']


# ---------------------------------------------------------------------------
# Agent end state and dialogue aggregation
# ---------------------------------------------------------------------------


def test_agent_end_state_compares_every_class():
    expected = [{'BaseApi': {'wifi': True, 'logged_in': True}}]
    assert check_agent_end_state([{'BaseApi': {'wifi': True, 'logged_in': True}}], expected)['valid']
    assert not check_agent_end_state([{'BaseApi': {'wifi': False, 'logged_in': True}}], expected)['valid']


def test_agent_end_state_requires_the_same_number_of_classes():
    expected = [{'BaseApi': {'wifi': True}}, {'MessageApi': {'inbox': {}}}]
    result = check_agent_end_state([{'BaseApi': {'wifi': True}}], expected)
    assert not result['valid']
    assert result['error_type'] == 'wrong number of class'


def test_multi_turn_accuracy_needs_every_step():
    assert multi_turn_accuracy([True, True]) == (1.0, 1.0)
    assert multi_turn_accuracy([True, False]) == (0.0, 0.5)
    assert multi_turn_accuracy([False, False]) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Category bookkeeping
# ---------------------------------------------------------------------------


def test_family_names_expand_to_categories():
    assert resolve_categories(['special']) == ['special_incomplete', 'special_error_param', 'special_irrelevant']
    assert resolve_categories(['agent_multi_step']) == ['agent_multi_step']
    assert len(resolve_categories(['test_all'])) == 17


def test_unknown_category_is_rejected():
    with pytest.raises(ValueError):
        resolve_categories(['not_a_category'])


def test_category_maps_to_its_split():
    assert split_of_category('normal_atom_bool') == 'normal'
    assert split_of_category('special_incomplete') == 'special'
    assert split_of_category('agent_multi_turn') == 'agent'


def test_category_is_read_from_the_record():
    assert category_of_record({'sub_category': 'data_normal_atom_bool'}) == 'normal_atom_bool'
    assert category_of_record({'id': 'normal_multi_turn_user_adjust_3_1'}) == 'normal_multi_turn_user_adjust'


def test_single_turn_prompts_use_the_official_templates():
    record = {'question': 'user: hi\n', 'function': [{'name': 'a'}], 'time': 'now', 'profile': ''}
    system, user = build_single_turn_prompts(record, 'normal_atom_bool', 'en')
    assert "[ApiName(key1='value1', key2='value2', ...)]" in system
    assert str(record['function']) in system
    assert user == 'Conversation history 1..t:\nuser: hi\n'

    special_system, _ = build_single_turn_prompts(record, 'special_incomplete', 'en')
    assert 'Missing necessary parameters' in special_system

    zh_system, zh_user = build_single_turn_prompts(record, 'normal_atom_bool', 'zh')
    assert 'API说明' in zh_system
    assert zh_user.startswith('对话历史1..t:')


# ---------------------------------------------------------------------------
# Agent rollout against the vendored scenario classes
# ---------------------------------------------------------------------------


class _ScriptedAPI(ModelAPI):
    """Replays a fixed list of messages, then ends the conversation."""

    def __init__(self, script: List[str]) -> None:
        super().__init__(model_name='scripted')
        self.script = list(script)
        self.prompts: List[List[ChatMessage]] = []

    def generate(self, input, tools=None, tool_choice=None, config=None, **kwargs):  # noqa: A002
        self.prompts.append(input)
        index = len(self.prompts) - 1
        content = self.script[index] if index < len(self.script) else 'finish conversation'
        return ModelOutput(model='scripted', choices=[ChatCompletionChoice.from_content(content)])


def scripted_model(script: List[str]) -> Model:
    return Model(api=_ScriptedAPI(script), config=GenerateConfig())


def phone_sample(test_category: str = 'agent_multi_step') -> Dict[str, Any]:
    return {
        'id': 'agent_multi_step_test',
        'test_category': test_category,
        'language': 'en',
        'question': 'Send Frank a message saying hello. You are Grace.',
        'functions': [{'name': 'send_message'}, {'name': 'turn_on_wifi'}],
        'initial_config': {'BaseApi': {'wifi': False, 'logged_in': True}},
        'involved_classes': ['BaseApi', 'MessageApi'],
    }


def test_multi_step_rollout_changes_the_environment():
    metadata = phone_sample()
    # The inbox starts full, so a message has to be deleted before a new one fits.
    script = [
        '[turn_on_wifi()]',
        '[get_earliest_message_id()]',
        '[delete_message(message_id=1)]',
        "[send_message(sender_name='Grace', receiver_name='Frank', message='hello')]",
        'finish conversation',
    ]
    result = run_rollout(scripted_model(script), metadata, max_steps=16)

    assert result.process == script[:4]
    state = {name: attributes for entry in result.end_state for name, attributes in entry.items()}
    assert state['BaseApi']['wifi'] is True
    assert any(message['message'] == 'hello' for message in state['MessageApi']['inbox'].values())


def test_wifi_is_shared_across_the_phone_apis():
    """Every involved instance inherits BaseApi, so the flag has to be set on all of them."""
    metadata = phone_sample()
    result = run_rollout(scripted_model(['[turn_on_wifi()]', 'finish conversation']), metadata, max_steps=8)
    state = {name: attributes for entry in result.end_state for name, attributes in entry.items()}
    assert state['BaseApi']['wifi'] is True


def test_multi_step_transcript_hides_the_agents_own_turns():
    """Upstream folds only user and execution lines into the multi-step transcript.

    ``multi_step_inference`` calls ``get_inference_message`` solely in its agent branch, so the
    agent never sees its own previous messages. This costs real accuracy -- a model that needs that
    history repeats one call until the step budget runs out -- but it is what the official numbers
    measure, so widening the transcript here would inflate scores above the leaderboard.
    """
    api = _ScriptedAPI(['[turn_on_wifi()]', '[get_earliest_message_id()]', 'finish conversation'])
    run_rollout(Model(api=api, config=GenerateConfig()), phone_sample(), max_steps=10)

    second_turn = api.prompts[1][-1].text
    assert 'agent:' not in second_turn
    assert 'execution result:' in second_turn
    assert second_turn.count('user:') == 1


def test_multi_turn_transcript_includes_every_turn():
    """``multi_turn_inference`` calls ``get_inference_message`` in all three branches."""
    agent = _ScriptedAPI(['Who should I message?', "[send_message(sender_name='Grace', receiver_name='F', message='x')]"])
    user = scripted_model(['Please message Frank.', 'Message Frank saying hi.'])
    run_rollout(Model(api=agent, config=GenerateConfig()), phone_sample('agent_multi_turn'), 20, user)

    # The agent's own question is quoted back to it on a later turn, unlike multi_step.
    assert any('agent:Who should I message?' in prompt[-1].text for prompt in agent.prompts)


def test_multi_step_rollout_survives_a_non_call_answer():
    metadata = phone_sample()
    result = run_rollout(scripted_model(['I need more information.', 'finish conversation']), metadata, max_steps=8)
    assert result.process == ['I need more information.']
    transcript = '\n'.join(message.text for message in result.messages)
    assert 'Please do not ask me any questions' in transcript


def test_multi_turn_rollout_uses_the_user_simulator():
    metadata = phone_sample('agent_multi_turn')
    agent = scripted_model([
        'Who should I message?',
        '[turn_on_wifi()]',
        '[delete_message(message_id=1)]',
        "[send_message(sender_name='Grace', receiver_name='Frank', message='hello')]",
        'Done. finish conversation',
    ])
    user = scripted_model(['Please message Frank.', 'Say hello to Frank.', 'finish conversation'])

    result = run_rollout(agent, metadata, max_steps=20, user_model=user)

    assert result.process == [
        '[turn_on_wifi()]',
        '[delete_message(message_id=1)]',
        "[send_message(sender_name='Grace', receiver_name='Frank', message='hello')]",
    ]
    state = {name: attributes for entry in result.end_state for name, attributes in entry.items()}
    assert any(message['message'] == 'hello' for message in state['MessageApi']['inbox'].values())
    # The user simulator was actually consulted.
    assert user.api.prompts


def test_multi_turn_rollout_requires_a_user_model():
    with pytest.raises(ValueError, match='user simulator'):
        run_rollout(scripted_model([]), phone_sample('agent_multi_turn'), max_steps=4)


def test_model_output_is_never_executed(tmp_path):
    """Argument expressions must not be evaluated; upstream's eval() path allows injection."""
    marker = tmp_path / 'pwned'
    metadata = phone_sample()
    script = [f"[send_message(sender_name=__import__('pathlib').Path('{marker}').write_text('x'))]"]
    result = run_rollout(scripted_model(script), metadata, max_steps=6)

    assert not marker.exists()
    state = {name: attributes for entry in result.end_state for name, attributes in entry.items()}
    assert len(state['MessageApi']['inbox']) == 6  # untouched initial inbox


def test_unknown_api_reports_an_error_without_raising():
    metadata = phone_sample()
    result = run_rollout(scripted_model(['[no_such_api()]', 'finish conversation']), metadata, max_steps=8)
    transcript = '\n'.join(message.text for message in result.messages)
    assert 'Error during execution' in transcript


# ---------------------------------------------------------------------------
# Trace and perf plumbing consumed by the web dashboard
# ---------------------------------------------------------------------------


def _events_by_step(result) -> Dict[int, List[Any]]:
    grouped: Dict[int, List[Any]] = {}
    for event in result.trace.events:
        grouped.setdefault(event.step, []).append(event)
    return grouped


def test_rollout_trace_groups_each_turn_with_the_calls_it_produced():
    """The dashboard groups by ``step`` and resolves messages via ``message_id``.

    A turn and the execution of its calls must therefore share a step, which the raw loop index
    cannot express because upstream spends a separate iteration on the executor.
    """
    metadata = phone_sample()
    script = ['[turn_on_wifi()]', "[send_message(sender_name='Grace', receiver_name='Frank', message='hi')]"]
    result = run_rollout(scripted_model(script), metadata, max_steps=10)

    by_step = _events_by_step(result)
    message_ids = {message.id for message in result.messages}
    by_id = {message.id: message for message in result.messages}

    # Every referenced message must actually be in the transcript, or the view drops the group.
    for events in by_step.values():
        for event in events:
            if event.message_id is not None:
                assert event.message_id in message_ids

    # Step 0 holds the first agent turn and the result of the call it made.
    step0 = {event.type.value for event in by_step[0]}
    assert step0 == {'model_generate', 'tool_call', 'tool_result'}
    generate = next(e for e in by_step[0] if e.type.value == 'model_generate')
    assert by_id[generate.message_id].role == 'assistant'
    assert generate.latency_ms is not None

    # tool_call and tool_result are paired through payload['id'], and the observation message
    # carries the same id so the two render as one entry.
    call = next(e for e in by_step[0] if e.type.value == 'tool_call')
    outcome = next(e for e in by_step[0] if e.type.value == 'tool_result')
    assert call.payload['name'] == 'turn_on_wifi'
    assert call.payload['id'] == outcome.payload['id']
    observation = by_id[outcome.message_id]
    assert observation.role == 'tool'
    assert observation.tool_call_id == call.payload['id']
    assert observation.function == 'turn_on_wifi'


def test_rollout_trace_records_the_framework_and_termination():
    metadata = phone_sample()
    result = run_rollout(scripted_model(['[turn_on_wifi()]', 'finish conversation']), metadata, max_steps=8)
    assert result.trace.framework == 'acebench'
    assert result.trace.strategy == 'agent_multi_step'
    assert [e.type.value for e in result.trace.events][-1] == 'submit'


def test_rollout_trace_records_running_out_of_steps():
    metadata = phone_sample()
    # Never emits the finish marker, so the budget is what stops the loop.
    result = run_rollout(scripted_model(['[turn_on_wifi()]'] * 6), metadata, max_steps=4)
    last = result.trace.events[-1]
    assert last.type.value == 'error'
    assert last.payload['message'] == 'max_steps_exceeded'


def test_agent_messages_keep_the_perf_metrics_the_model_api_attached():
    """PerfCollector reads ``perf_metrics`` off assistant messages, so they must not be rebuilt."""
    from evalscope.api.messages import PerformanceMetrics

    class _PerfAPI(_ScriptedAPI):

        def generate(self, input, tools=None, tool_choice=None, config=None, **kwargs):  # noqa: A002
            output = super().generate(input, tools, tool_choice, config, **kwargs)
            output.message.perf_metrics = PerformanceMetrics(latency=1.5, input_tokens=11, output_tokens=7)
            return output

    model = Model(api=_PerfAPI(['[turn_on_wifi()]', 'finish conversation']), config=GenerateConfig())
    result = run_rollout(model, phone_sample(), max_steps=8)

    assistant = [m for m in result.messages if m.role == 'assistant']
    assert assistant, 'the rollout must keep the model\'s own assistant messages'
    assert all(m.perf_metrics is not None for m in assistant)
    assert assistant[0].perf_metrics.input_tokens == 11


def test_user_simulator_turns_are_traced_but_excluded_from_model_perf():
    metadata = phone_sample('agent_multi_turn')
    # The agent asks a question first, so the simulated user has to answer mid-conversation.
    agent = scripted_model([
        'Who should I message?',
        "[send_message(sender_name='Grace', receiver_name='Frank', message='hi')]",
    ])
    user = scripted_model(['Please message Frank.', 'Message Frank saying hi.'])
    result = run_rollout(agent, metadata, max_steps=20, user_model=user)

    # The simulated user must not look like the evaluated model, or it would pollute the perf table.
    assert all(m.role != 'assistant' for m in result.messages if m.text.startswith('user: '))
    assert all(m.perf_metrics is None for m in result.messages if m.role != 'assistant')

    # Mid-conversation user turns must be referenced by an event, otherwise the trace view drops them.
    simulator_events = [e for e in result.trace.events if e.payload.get('source') == 'user_simulator']
    referenced = {e.message_id for e in simulator_events}
    assert referenced, 'user simulator turns need a trace event to stay visible'
    assert referenced <= {m.id for m in result.messages}

    # The opening turn precedes any agent turn and is intentionally event-free, so the dashboard
    # renders it as the conversation preamble rather than inside a step group.
    assert result.messages[0].text == 'user: Please message Frank.'
    assert result.messages[0].id not in referenced

    # A reply shares its step with the agent turn it answers, so that step holds two
    # ``model_generate`` events. The dashboard reads the *first* one as the step's assistant perf
    # header, so the agent's event has to come first.
    by_id = {m.id: m for m in result.messages}
    for events in _events_by_step(result).values():
        generates = [e for e in events if e.type.value == 'model_generate']
        if len(generates) < 2:
            continue
        assert by_id[generates[0].message_id].role == 'assistant'
        assert generates[0].payload.get('source') != 'user_simulator'


# ---------------------------------------------------------------------------
# Adapter aggregation
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def adapter():
    from evalscope.api.registry import get_benchmark
    from evalscope.config import TaskConfig

    return get_benchmark('acebench', config=TaskConfig(model='mock-llm', datasets=['acebench']))


def test_all_categories_stay_loadable_without_a_user_simulator(adapter):
    """Loading must not depend on extra_params.

    Filtering categories out of the load path also hides them from the dataset statistics that
    ``make docs-pipeline`` records, which silently desynchronised the generated docs from the
    declared sample counts.
    """
    assert not adapter.user_model_id
    assert 'agent_multi_turn' in adapter.subset_list
    assert len(adapter.subset_list) == 17


def test_multi_turn_subsets_aggregate_per_dialogue(adapter):
    """A dialogue counts as correct only when all of its steps are."""
    steps = [
        ('normal_multi_turn_user_adjust_0_0', 1.0),
        ('normal_multi_turn_user_adjust_0_1', 1.0),
        ('normal_multi_turn_user_adjust_1_0', 1.0),
        ('normal_multi_turn_user_adjust_1_1', 0.0),
    ]
    sample_scores = [
        SampleScore(
            score=Score(value={'acc': acc}, main_score_name='acc'),
            sample_id=sample_id,
            sample_metadata={
                'test_category': 'normal_multi_turn_user_adjust',
                'dialogue_id': sample_id.rsplit('_', 1)[0],
            },
        )
        for sample_id, acc in steps
    ]

    scores = {agg.metric_name: agg for agg in adapter.aggregate_scores(sample_scores)}
    assert scores['accuracy'].score == pytest.approx(0.5)
    assert scores['accuracy'].num == 2
    assert scores['process_acc'].score == pytest.approx(0.75)


def test_report_adds_the_official_groups(adapter):
    from evalscope.report import Category, Metric, Report, Subset

    subsets = [Subset(name=name, score=0.5, num=10) for name in adapter.subset_list]
    report = Report(name='acebench', metrics=[Metric(name='acc', categories=[Category(name='d', subsets=subsets)])])

    adapter._on_generate_report_end(report, output_dir='/tmp')

    groups = {
        subset.name: subset
        for category in report.metrics[0].categories for subset in category.subsets if subset.name.isupper()
    }
    assert set(groups) >= {'ATOM', 'SINGLE_TURN', 'MULTI_TURN', 'NORMAL', 'SPECIAL', 'AGENT', 'OVERALL'}
    assert groups['OVERALL'].score == pytest.approx(0.5)


def test_overall_uses_the_official_weights(adapter):
    from evalscope.report import Category, Metric, Report, Subset

    scores = {'normal': 1.0, 'special': 0.0, 'agent': 0.0}
    subsets = [
        Subset(name=name, score=scores[split_of_category(name)], num=10)
        for name in adapter.subset_list
    ]
    report = Report(name='acebench', metrics=[Metric(name='acc', categories=[Category(name='d', subsets=subsets)])])

    adapter._on_generate_report_end(report, output_dir='/tmp')

    overall = next(
        subset for category in report.metrics[0].categories
        for subset in category.subsets if subset.name == 'OVERALL'
    )
    # All three families are evaluated, so the weights are used as-is without renormalization.
    assert overall.score == pytest.approx(0.578, abs=1e-3)


# ---------------------------------------------------------------------------
# Differential checks against a local ACEBench checkout.
#
# The scoring logic and the simulated APIs are ports of upstream code, so the guarantee that they
# behave identically is a differential run against the real thing. Point ACEBENCH_REPO at a clone
# of https://github.com/ACEBench/ACEBench to execute these; they are skipped otherwise.
# ---------------------------------------------------------------------------

ACEBENCH_REPO = os.environ.get('ACEBENCH_REPO', '')
requires_official = pytest.mark.skipif(not ACEBENCH_REPO, reason='set ACEBENCH_REPO to a local ACEBench checkout')

NORMAL_CATEGORIES = [
    'normal_single_turn_single_function', 'normal_single_turn_parallel_function',
    'normal_multi_turn_user_adjust', 'normal_multi_turn_user_switch', 'normal_similar_api',
    'normal_preference', 'normal_atom_bool', 'normal_atom_enum', 'normal_atom_number',
    'normal_atom_list', 'normal_atom_object_deep', 'normal_atom_object_short',
]


@pytest.fixture
def official():
    """Import the upstream modules from the checkout named by ACEBENCH_REPO."""
    if ACEBENCH_REPO and ACEBENCH_REPO not in sys.path:
        sys.path.insert(0, ACEBENCH_REPO)
    from model_eval import checker
    from model_inference.multi_step import multi_step_utils
    return checker, multi_step_utils


def _official_records(language, category):
    base = f'{ACEBENCH_REPO}/data_all/data_{language}'
    records = {}
    with open(f'{base}/data_{category}.json', encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            records[row['id']] = {'prompt': row}
    with open(f'{base}/possible_answer/data_{category}.json', encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            records[row['id']]['answer'] = row
    return records


def _perturbations(calls, functions):
    """Mutate an oracle answer in ways that probe each checker rule."""
    def spec_of(name):
        return next((f for f in functions if f['name'] == name), None)

    yield 'oracle', calls

    name, arguments = next(iter(calls[0].items()))
    spec = spec_of(name)
    required = spec['parameters'].get('required', []) if spec else []

    for label, keys in [('drop_optional', [p for p in arguments if p not in required]),
                        ('drop_required', [p for p in arguments if p in required])]:
        if keys:
            mutated = copy.deepcopy(calls)
            next(iter(mutated[0].values())).pop(keys[0])
            yield label, mutated

    if spec:
        extra = next((p for p in spec['parameters'].get('properties', {}) if p not in arguments), None)
        if extra:
            mutated = copy.deepcopy(calls)
            next(iter(mutated[0].values()))[extra] = 'extra_value'
            yield 'extra_param', mutated

    mutated = copy.deepcopy(calls)
    next(iter(mutated[0].values()))['__not_in_schema__'] = 1
    yield 'unknown_param', mutated

    yield 'wrong_function', [{f'{name}_x': copy.deepcopy(arguments)}] + copy.deepcopy(calls[1:])
    yield 'duplicate_call', copy.deepcopy(calls) + copy.deepcopy(calls[:1])

    mutators = [
        ('string_superset', lambda v: isinstance(v, str) and v, lambda v: v + ' extra'),
        ('string_prefix', lambda v: isinstance(v, str) and len(v) > 4, lambda v: v[:2]),
        ('list_reorder', lambda v: isinstance(v, list) and len(v) > 1, lambda v: list(reversed(v))),
        ('list_truncate', lambda v: isinstance(v, list) and v, lambda v: v[:-1]),
        ('int_as_float', lambda v: isinstance(v, int) and not isinstance(v, bool), float),
        ('number_bump', lambda v: isinstance(v, (int, float)) and not isinstance(v, bool), lambda v: v + 1),
        ('bool_flip', lambda v: isinstance(v, bool), lambda v: not v),
        ('none_value', lambda v: True, lambda v: None),
        ('dict_bad_value', lambda v: isinstance(v, dict) and v, lambda v: {**v, list(v)[0]: '__wrong__'}),
    ]
    for label, predicate, mutate in mutators:
        target = next((key for key, value in arguments.items() if predicate(value)), None)
        if target is not None:
            mutated = copy.deepcopy(calls)
            mutated_args = next(iter(mutated[0].values()))
            mutated_args[target] = mutate(mutated_args[target])
            yield label, mutated


@requires_official
@pytest.mark.parametrize('language', ['en', 'zh'])
def test_checker_matches_upstream(official, language):
    """Every verdict must agree with the official checker, quirks included."""
    official_checker, _ = official
    disagreements = []
    total = 0

    for category in NORMAL_CATEGORIES:
        for record in _official_records(language, category).values():
            ground_truth = record['answer']['ground_truth']
            first = ground_truth[0] if isinstance(ground_truth, list) else ground_truth
            oracle = [{re.sub(r'_\d+$', '', name): copy.deepcopy(args)} for name, args in first.items()]

            for label, calls in _perturbations(oracle, record['prompt']['function']):
                total += 1
                try:
                    expected = bool(
                        official_checker.normal_checker(
                            record['prompt']['function'], calls, first, record['prompt']['question'], category
                        )['valid']
                    )
                except Exception as error:  # noqa: BLE001
                    expected = f'raise:{type(error).__name__}'
                try:
                    actual = bool(check_normal_answer(record['prompt']['function'], calls, first, category)['valid'])
                except Exception as error:  # noqa: BLE001
                    actual = f'raise:{type(error).__name__}'

                if expected != actual:
                    disagreements.append((record['prompt']['id'], label, expected, actual))

    assert total > 1000, f'expected the full {language} normal set, only checked {total} cases'
    assert not disagreements, f'{len(disagreements)} verdicts differ, first few: {disagreements[:5]}'


@requires_official
def test_simulated_apis_match_upstream(official):
    """The vendored API classes plus the safe dispatch must reach the same state as upstream."""
    from evalscope.benchmarks.acebench.parser import decode_execution_calls
    from evalscope.benchmarks.acebench.rollout import _dispatch, _reparse, _serialize
    from evalscope.benchmarks.acebench.scenarios import SAVED_ATTRIBUTES, load_scenario_instances, snapshot_states

    _, multi_step_utils = official

    sequences = [
        ['[turn_on_wifi()]', '[login_device()]'],
        ["[send_message(sender_name='Grace', receiver_name='Frank', message='Lunch?')]"],
        ['[turn_on_wifi()]', '[get_earliest_message_id()]', '[delete_message(message_id=1)]'],
        ["[add_reminder(title='Spend', description='88 yuan', time='2024-07-15 09:30')]"],
        ['[turn_on_wifi()]', "[login_food_platform(username='Eve', password='password123')]",
         '[add_food_delivery_order(username="Eve", merchant_name="Domino\'s", '
         "items=[{'product': 'Super Supreme Pizza', 'quantity': 1}])]"],
        ["[login_food_platform(username='Eve', password='wrong')]"],
        ["[delete_message(message_id='not-an-int')]"],
        ["[get_flight_details(origin='Beijing', destination='Shanghai')]"],
    ]
    configs = [
        ({'BaseApi': {'wifi': False, 'logged_in': True}}, ['BaseApi', 'MessageApi', 'ReminderApi', 'FoodPlatform']),
        ({'BaseApi': {'wifi': True, 'logged_in': True}}, ['BaseApi', 'MessageApi', 'FoodPlatform']),
        ({}, ['Travel']),
    ]

    for config_index, (initial_config, involved_classes) in enumerate(configs):
        for sequence_index, sequence in enumerate(sequences):
            execution_list = []
            for message in sequence:
                for call in decode_execution_calls(message):
                    name, arguments = next(iter(call.items()))
                    execution_list.append(f'{name}({", ".join(f"{k}={v!r}" for k, v in arguments.items())})')

            _, upstream_instances = multi_step_utils.execute_agent_func_call(
                func_call_list=execution_list,
                initial_config=copy.deepcopy(initial_config),
                involved_classes=involved_classes,
                model_name='parity',
                test_entry_id=f'{config_index}_{sequence_index}',
                language='en',
            )
            upstream_state = [{
                name: {key: value for key, value in vars(instance).items() if key in SAVED_ATTRIBUTES.get(name, [])}
            } for name, instance in upstream_instances.items()]
            # Upstream writes this state to a result file and reads it back before grading, which
            # turns integer dict keys into strings. snapshot_states reproduces that conversion.
            upstream_state = json.loads(json.dumps(upstream_state, default=str))

            instances = load_scenario_instances(copy.deepcopy(initial_config), involved_classes, 'en')
            for message in sequence:
                [_reparse(_serialize(outcome)) for outcome in _dispatch(decode_execution_calls(message), instances)]

            assert snapshot_states(instances) == upstream_state, f'state differs for {sequence}'
