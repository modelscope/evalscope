"""Tests for workload_trace dataset plugin."""

import json
import numpy as np
import os
import pytest

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.openai_api import OpenaiPlugin
from evalscope.perf.plugin.datasets.workload_trace import WorkloadTraceDatasetPlugin
from evalscope.perf.utils.body_meta import BODY_META_ARRIVAL_OFFSET, BODY_META_HEADERS, BODY_META_REQUEST_ID


def _make_trace_file(records, tmp_path):
    path = os.path.join(tmp_path, 'trace.jsonl')
    with open(path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    return path


def _make_args(trace_path, **kwargs):
    return Arguments(
        url='http://localhost:8080/v1/chat/completions',
        dataset='workload_trace',
        dataset_path=trace_path,
        open_loop=True,
        rate=1,
        **kwargs,
    )


def _record(body=None, timestamp=0.0, headers=None, request_id=None, completion_tokens=None):
    r = {
        'body': body or {'model': 'qwen-72b', 'messages': [{'role': 'user', 'content': 'hi'}], 'temperature': 0.5},
        'timestamp': timestamp,
    }
    if headers is not None:
        r['headers'] = headers
    if request_id is not None:
        r['request_id'] = request_id
    if completion_tokens is not None:
        r['completion_tokens'] = completion_tokens
    return r


# ---------------------------------------------------------------------------
# Loading and basic passthrough
# ---------------------------------------------------------------------------


class TestBasicLoading:

    def test_loads_records(self, tmp_path):
        path = _make_trace_file([_record(timestamp=1.0), _record(timestamp=2.0)], tmp_path)
        plugin = WorkloadTraceDatasetPlugin(_make_args(path))
        messages = list(plugin.build_messages())
        assert len(messages) == 2

    def test_body_fields_preserved(self, tmp_path):
        body = {'model': 'qwen-72b', 'messages': [{'role': 'user', 'content': 'hello'}], 'temperature': 0.9, 'max_tokens': 100}
        path = _make_trace_file([_record(body=body, timestamp=0.0)], tmp_path)
        plugin = WorkloadTraceDatasetPlugin(_make_args(path))
        result = list(plugin.build_messages())[0]
        assert result['temperature'] == 0.9
        assert result['max_tokens'] == 100

    def test_body_survives_build_request(self, tmp_path):
        """Body fields should survive through OpenaiPlugin.build_request (setdefault)."""
        body = {'model': 'qwen-72b', 'messages': [{'role': 'user', 'content': 'hi'}], 'temperature': 0.9, 'max_tokens': 100}
        path = _make_trace_file([_record(body=body, timestamp=0.0)], tmp_path)
        args = _make_args(path)
        plugin = WorkloadTraceDatasetPlugin(args)
        api = OpenaiPlugin(args)
        messages = list(plugin.build_messages())[0]
        request = api.build_request(messages)
        assert request['temperature'] == 0.9
        assert request['max_tokens'] == 100
        assert request['model'] == 'qwen-72b'

    def test_skips_blank_lines(self, tmp_path):
        path = os.path.join(tmp_path, 'trace.jsonl')
        with open(path, 'w') as f:
            f.write(json.dumps(_record(timestamp=0.0)) + '\n')
            f.write('\n')
            f.write('   \n')
            f.write(json.dumps(_record(timestamp=1.0)) + '\n')
        plugin = WorkloadTraceDatasetPlugin(_make_args(path))
        assert len(list(plugin.build_messages())) == 2

    def test_sets_number_to_record_count(self, tmp_path):
        records = [_record(timestamp=float(i)) for i in range(5)]
        path = _make_trace_file(records, tmp_path)
        args = _make_args(path)
        WorkloadTraceDatasetPlugin(args)
        assert args.number == 5

    def test_missing_optional_fields(self, tmp_path):
        path = _make_trace_file([{'body': {'model': 'm', 'messages': []}, 'timestamp': 0.0}], tmp_path)
        plugin = WorkloadTraceDatasetPlugin(_make_args(path))
        msg = list(plugin.build_messages())[0]
        assert BODY_META_HEADERS not in msg
        assert BODY_META_REQUEST_ID not in msg


# ---------------------------------------------------------------------------
# Model rewriting
# ---------------------------------------------------------------------------


class TestModelRewriting:

    def test_model_override(self, tmp_path):
        path = _make_trace_file([_record(timestamp=0.0)], tmp_path)
        args = _make_args(path, dataset_args={'model_override': 'new-model'})
        plugin = WorkloadTraceDatasetPlugin(args)
        msg = list(plugin.build_messages())[0]
        assert msg['model'] == 'new-model'

    def test_model_mapping_match(self, tmp_path):
        path = _make_trace_file([_record(timestamp=0.0)], tmp_path)
        args = _make_args(path, dataset_args={'model_mapping': {'qwen-72b': 'qwen-72b-v2'}})
        plugin = WorkloadTraceDatasetPlugin(args)
        msg = list(plugin.build_messages())[0]
        assert msg['model'] == 'qwen-72b-v2'

    def test_model_mapping_no_match_falls_back_to_override(self, tmp_path):
        path = _make_trace_file([_record(timestamp=0.0)], tmp_path)
        args = _make_args(path, dataset_args={'model_mapping': {'other': 'x'}, 'model_override': 'fallback'})
        plugin = WorkloadTraceDatasetPlugin(args)
        msg = list(plugin.build_messages())[0]
        assert msg['model'] == 'fallback'

    def test_model_mapping_no_match_no_override_keeps_original(self, tmp_path):
        path = _make_trace_file([_record(timestamp=0.0)], tmp_path)
        args = _make_args(path, dataset_args={'model_mapping': {'other': 'x'}})
        plugin = WorkloadTraceDatasetPlugin(args)
        msg = list(plugin.build_messages())[0]
        assert msg['model'] == 'qwen-72b'

    def test_model_mapping_takes_priority_over_override(self, tmp_path):
        path = _make_trace_file([_record(timestamp=0.0)], tmp_path)
        args = _make_args(path, dataset_args={'model_mapping': {'qwen-72b': 'mapped'}, 'model_override': 'overridden'})
        plugin = WorkloadTraceDatasetPlugin(args)
        msg = list(plugin.build_messages())[0]
        assert msg['model'] == 'mapped'

    def test_cli_model_does_not_rewrite_body(self, tmp_path):
        """--model must not rewrite trace bodies (issue #1489); the trace's own
        model is preserved so multi-model routing stays intact."""
        path = _make_trace_file([_record(timestamp=0.0)], tmp_path)
        args = Arguments(
            model='cli-model', url='http://localhost:8080/v1/chat/completions',
            dataset='workload_trace', dataset_path=path, open_loop=True, rate=1,
        )
        plugin = WorkloadTraceDatasetPlugin(args)
        msg = list(plugin.build_messages())[0]
        assert msg['model'] == 'qwen-72b'


# ---------------------------------------------------------------------------
# Arrival schedule
# ---------------------------------------------------------------------------


class TestArrivalSchedule:

    def test_offsets_at_speed_1(self, tmp_path):
        records = [_record(timestamp=10.0), _record(timestamp=12.0), _record(timestamp=15.0)]
        path = _make_trace_file(records, tmp_path)
        plugin = WorkloadTraceDatasetPlugin(_make_args(path))
        messages = list(plugin.build_messages())
        offsets = [m[BODY_META_ARRIVAL_OFFSET] for m in messages]
        np.testing.assert_allclose(offsets, [0.0, 2.0, 5.0])

    def test_offsets_at_speed_2(self, tmp_path):
        records = [_record(timestamp=10.0), _record(timestamp=14.0)]
        path = _make_trace_file(records, tmp_path)
        args = _make_args(path, dataset_args={'speed': 2.0})
        plugin = WorkloadTraceDatasetPlugin(args)
        messages = list(plugin.build_messages())
        offsets = [m[BODY_META_ARRIVAL_OFFSET] for m in messages]
        np.testing.assert_allclose(offsets, [0.0, 2.0])

    def test_offsets_at_speed_half(self, tmp_path):
        records = [_record(timestamp=0.0), _record(timestamp=1.0)]
        path = _make_trace_file(records, tmp_path)
        args = _make_args(path, dataset_args={'speed': 0.5})
        plugin = WorkloadTraceDatasetPlugin(args)
        messages = list(plugin.build_messages())
        offsets = [m[BODY_META_ARRIVAL_OFFSET] for m in messages]
        np.testing.assert_allclose(offsets, [0.0, 2.0])


# ---------------------------------------------------------------------------
# Body meta keys
# ---------------------------------------------------------------------------


class TestBodyMetaKeys:

    def test_headers_injected(self, tmp_path):
        path = _make_trace_file([_record(timestamp=0.0, headers={'X-Route': 'canary'})], tmp_path)
        plugin = WorkloadTraceDatasetPlugin(_make_args(path))
        msg = list(plugin.build_messages())[0]
        assert msg[BODY_META_HEADERS] == {'X-Route': 'canary'}

    def test_request_id_injected(self, tmp_path):
        path = _make_trace_file([_record(timestamp=0.0, request_id='req-123')], tmp_path)
        plugin = WorkloadTraceDatasetPlugin(_make_args(path))
        msg = list(plugin.build_messages())[0]
        assert msg[BODY_META_REQUEST_ID] == 'req-123'

    def test_arrival_offset_injected(self, tmp_path):
        path = _make_trace_file([_record(timestamp=5.0)], tmp_path)
        plugin = WorkloadTraceDatasetPlugin(_make_args(path))
        msg = list(plugin.build_messages())[0]
        assert BODY_META_ARRIVAL_OFFSET in msg

    def test_body_meta_keys_survive_build_request(self, tmp_path):
        """Body meta keys should survive build_request (they're not OpenAI params)."""
        path = _make_trace_file([_record(timestamp=0.0, headers={'X-A': '1'}, request_id='r1')], tmp_path)
        args = _make_args(path)
        plugin = WorkloadTraceDatasetPlugin(args)
        api = OpenaiPlugin(args)
        msg = list(plugin.build_messages())[0]
        request = api.build_request(msg)
        # Body meta keys pass through build_request (stripped later in process_request).
        assert BODY_META_HEADERS in request
        assert BODY_META_REQUEST_ID in request


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:

    def test_requires_open_loop(self, tmp_path):
        path = _make_trace_file([_record(timestamp=0.0)], tmp_path)
        args = Arguments(
            model='m', url='http://localhost:8080/v1/chat/completions',
            dataset='workload_trace', dataset_path=path, open_loop=False, rate=1,
        )
        with pytest.raises(ValueError, match='open-loop'):
            WorkloadTraceDatasetPlugin(args)

    def test_requires_dataset_path(self, tmp_path):
        args = Arguments(
            model='m', url='http://localhost:8080/v1/chat/completions',
            dataset='workload_trace', open_loop=True, rate=1,
        )
        with pytest.raises(ValueError, match='dataset-path'):
            WorkloadTraceDatasetPlugin(args)

    def test_empty_file(self, tmp_path):
        path = os.path.join(tmp_path, 'empty.jsonl')
        with open(path, 'w') as f:
            f.write('')
        with pytest.raises(ValueError, match='empty'):
            WorkloadTraceDatasetPlugin(_make_args(path))

    def test_invalid_json(self, tmp_path):
        path = os.path.join(tmp_path, 'bad.jsonl')
        with open(path, 'w') as f:
            f.write('not json\n')
        with pytest.raises(ValueError, match='line 1.*invalid JSON'):
            WorkloadTraceDatasetPlugin(_make_args(path))

    def test_missing_body(self, tmp_path):
        path = _make_trace_file([{'timestamp': 0.0}], tmp_path)
        with pytest.raises(ValueError, match=r'(?s)line 1.*body.*(?:missing|required)'):
            WorkloadTraceDatasetPlugin(_make_args(path))

    def test_missing_timestamp(self, tmp_path):
        path = _make_trace_file([{'body': {'model': 'm', 'messages': []}}], tmp_path)
        with pytest.raises(ValueError, match=r'(?s)line 1.*timestamp.*(?:missing|required)'):
            WorkloadTraceDatasetPlugin(_make_args(path))

    def test_non_monotonic_timestamps_sorted(self, tmp_path, caplog):
        records = [_record(timestamp=5.0), _record(timestamp=3.0)]
        path = _make_trace_file(records, tmp_path)
        plugin = WorkloadTraceDatasetPlugin(_make_args(path))
        # Should sort rather than reject
        assert plugin._records[0].timestamp == 3.0
        assert plugin._records[1].timestamp == 5.0
        assert 'not monotonic' in caplog.text

    def test_body_meta_key_collision(self, tmp_path):
        body = {'model': 'm', 'messages': [], BODY_META_HEADERS: {}}
        path = _make_trace_file([{'body': body, 'timestamp': 0.0}], tmp_path)
        with pytest.raises(ValueError, match='reserved'):
            WorkloadTraceDatasetPlugin(_make_args(path))

    def test_invalid_speed(self, tmp_path):
        path = _make_trace_file([_record(timestamp=0.0)], tmp_path)
        with pytest.raises(Exception, match='speed'):
            WorkloadTraceDatasetPlugin(_make_args(path, dataset_args={'speed': -1.0}))

    def test_unknown_dataset_arg_rejected(self, tmp_path):
        path = _make_trace_file([_record(timestamp=0.0)], tmp_path)
        with pytest.raises(Exception):
            WorkloadTraceDatasetPlugin(_make_args(path, dataset_args={'bogus': True}))

    def test_error_reports_line_number(self, tmp_path):
        path = os.path.join(tmp_path, 'trace.jsonl')
        with open(path, 'w') as f:
            f.write(json.dumps(_record(timestamp=0.0)) + '\n')
            f.write('bad line\n')
        with pytest.raises(ValueError, match='line 2'):
            WorkloadTraceDatasetPlugin(_make_args(path))


# ---------------------------------------------------------------------------
# ISO-8601 timestamps
# ---------------------------------------------------------------------------


class TestTimestampFormats:

    def test_iso8601_utc_z(self, tmp_path):
        path = _make_trace_file([
            {'body': {'model': 'm', 'messages': []}, 'timestamp': '2025-01-15T10:00:00Z'},
            {'body': {'model': 'm', 'messages': []}, 'timestamp': '2025-01-15T10:00:05Z'},
        ], tmp_path)
        plugin = WorkloadTraceDatasetPlugin(_make_args(path))
        messages = list(plugin.build_messages())
        offsets = [m[BODY_META_ARRIVAL_OFFSET] for m in messages]
        np.testing.assert_allclose(offsets, [0.0, 5.0])

    def test_iso8601_with_offset(self, tmp_path):
        path = _make_trace_file([
            {'body': {'model': 'm', 'messages': []}, 'timestamp': '2025-01-15T18:00:00+08:00'},
            {'body': {'model': 'm', 'messages': []}, 'timestamp': '2025-01-15T18:00:03+08:00'},
        ], tmp_path)
        plugin = WorkloadTraceDatasetPlugin(_make_args(path))
        messages = list(plugin.build_messages())
        offsets = [m[BODY_META_ARRIVAL_OFFSET] for m in messages]
        np.testing.assert_allclose(offsets, [0.0, 3.0])

    def test_integer_timestamp(self, tmp_path):
        path = _make_trace_file([
            {'body': {'model': 'm', 'messages': []}, 'timestamp': 1000},
            {'body': {'model': 'm', 'messages': []}, 'timestamp': 1002},
        ], tmp_path)
        plugin = WorkloadTraceDatasetPlugin(_make_args(path))
        messages = list(plugin.build_messages())
        offsets = [m[BODY_META_ARRIVAL_OFFSET] for m in messages]
        np.testing.assert_allclose(offsets, [0.0, 2.0])

    def test_mixed_numeric_and_iso_accepted(self, tmp_path):
        """Pydantic converts both to float, so mixed formats work if monotonic."""
        path = _make_trace_file([
            {'body': {'model': 'm', 'messages': []}, 'timestamp': 1000.0},
            {'body': {'model': 'm', 'messages': []}, 'timestamp': '2025-01-15T10:00:00Z'},
        ], tmp_path)
        plugin = WorkloadTraceDatasetPlugin(_make_args(path))
        assert len(list(plugin.build_messages())) == 2

    def test_unparseable_timestamp_rejected(self, tmp_path):
        path = _make_trace_file([
            {'body': {'model': 'm', 'messages': []}, 'timestamp': 'not-a-date'},
        ], tmp_path)
        with pytest.raises(ValueError, match='cannot parse timestamp'):
            WorkloadTraceDatasetPlugin(_make_args(path))


# ---------------------------------------------------------------------------
# Body as JSON string
# ---------------------------------------------------------------------------


class TestBodyAsString:

    def test_json_string_body_parsed(self, tmp_path):
        body_dict = {'model': 'qwen', 'messages': [{'role': 'user', 'content': 'hi'}], 'temperature': 0.7}
        path = _make_trace_file([
            {'body': json.dumps(body_dict), 'timestamp': 0.0},
        ], tmp_path)
        plugin = WorkloadTraceDatasetPlugin(_make_args(path))
        msg = list(plugin.build_messages())[0]
        assert msg['model'] == 'qwen'
        assert msg['temperature'] == 0.7

    def test_invalid_json_string_body_rejected(self, tmp_path):
        path = _make_trace_file([
            {'body': 'not valid json', 'timestamp': 0.0},
        ], tmp_path)
        with pytest.raises(ValueError, match='not valid JSON'):
            WorkloadTraceDatasetPlugin(_make_args(path))

    def test_json_string_body_non_dict_rejected(self, tmp_path):
        path = _make_trace_file([
            {'body': json.dumps([1, 2, 3]), 'timestamp': 0.0},
        ], tmp_path)
        with pytest.raises(ValueError, match='must.*dict'):
            WorkloadTraceDatasetPlugin(_make_args(path))


# ---------------------------------------------------------------------------
# Warmup sideband key
# ---------------------------------------------------------------------------


class TestWarmup:

    def test_warmup_absolute(self, tmp_path):
        records = [_record(timestamp=float(i)) for i in range(5)]
        path = _make_trace_file(records, tmp_path)
        args = _make_args(path, warmup_num=2)
        plugin = WorkloadTraceDatasetPlugin(args)
        messages = list(plugin.build_messages())
        warmup_flags = ['__evalscope_is_warmup' in m for m in messages]
        assert warmup_flags == [True, True, False, False, False]

    def test_warmup_ratio(self, tmp_path):
        records = [_record(timestamp=float(i)) for i in range(10)]
        path = _make_trace_file(records, tmp_path)
        args = _make_args(path, warmup_num=0.2)
        plugin = WorkloadTraceDatasetPlugin(args)
        messages = list(plugin.build_messages())
        warmup_flags = ['__evalscope_is_warmup' in m for m in messages]
        assert sum(warmup_flags) == 2
        assert warmup_flags[:2] == [True, True]
        assert all(not f for f in warmup_flags[2:])

    def test_warmup_capped_at_record_count(self, tmp_path):
        records = [_record(timestamp=float(i)) for i in range(5)]
        path = _make_trace_file(records, tmp_path)
        args = _make_args(path, warmup_num=100)
        plugin = WorkloadTraceDatasetPlugin(args)
        messages = list(plugin.build_messages())
        assert all('__evalscope_is_warmup' in m for m in messages)
        assert len(messages) == 5

    def test_warmup_disables_framework_split(self, tmp_path):
        records = [_record(timestamp=float(i)) for i in range(5)]
        path = _make_trace_file(records, tmp_path)
        args = _make_args(path, warmup_num=2)
        WorkloadTraceDatasetPlugin(args)
        assert args.warmup_num == 0

    def test_no_warmup_by_default(self, tmp_path):
        records = [_record(timestamp=float(i)) for i in range(3)]
        path = _make_trace_file(records, tmp_path)
        plugin = WorkloadTraceDatasetPlugin(_make_args(path))
        messages = list(plugin.build_messages())
        assert all('__evalscope_is_warmup' not in m for m in messages)


# ---------------------------------------------------------------------------
# Rate validation bypass
# ---------------------------------------------------------------------------


class TestRateValidation:

    def test_no_rate_required(self, tmp_path):
        """workload_trace with --open-loop should not require --rate."""
        path = _make_trace_file([_record(timestamp=0.0)], tmp_path)
        args = Arguments(
            model='m',
            url='http://localhost:8080/v1/chat/completions',
            dataset='workload_trace',
            dataset_path=path,
            open_loop=True,
        )
        plugin = WorkloadTraceDatasetPlugin(args)
        assert len(list(plugin.build_messages())) == 1

    def test_explicit_rate_still_works(self, tmp_path):
        path = _make_trace_file([_record(timestamp=0.0)], tmp_path)
        args = Arguments(
            model='m',
            url='http://localhost:8080/v1/chat/completions',
            dataset='workload_trace',
            dataset_path=path,
            open_loop=True,
            rate=5,
        )
        plugin = WorkloadTraceDatasetPlugin(args)
        assert len(list(plugin.build_messages())) == 1


# ---------------------------------------------------------------------------
# Match output length
# ---------------------------------------------------------------------------


class TestMatchOutputLength:

    def test_sets_max_tokens_and_ignore_eos(self, tmp_path):
        path = _make_trace_file([_record(completion_tokens=42, timestamp=0.0)], tmp_path)
        args = _make_args(path, dataset_args={'match_output_length': True})
        plugin = WorkloadTraceDatasetPlugin(args)
        msg = list(plugin.build_messages())[0]
        assert msg['max_tokens'] == 42
        assert msg['ignore_eos'] is True

    def test_no_effect_when_disabled(self, tmp_path):
        path = _make_trace_file([_record(completion_tokens=42, timestamp=0.0)], tmp_path)
        args = _make_args(path)
        plugin = WorkloadTraceDatasetPlugin(args)
        msg = list(plugin.build_messages())[0]
        assert 'ignore_eos' not in msg
        assert msg.get('max_tokens') is None  # original body has no max_tokens

    def test_skips_when_completion_tokens_missing(self, tmp_path):
        """Records without completion_tokens are left untouched."""
        path = _make_trace_file([_record(timestamp=0.0)], tmp_path)
        args = _make_args(path, dataset_args={'match_output_length': True})
        plugin = WorkloadTraceDatasetPlugin(args)
        msg = list(plugin.build_messages())[0]
        assert 'ignore_eos' not in msg

    def test_overwrites_existing_max_tokens(self, tmp_path):
        body = {'model': 'qwen', 'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 999}
        path = _make_trace_file([_record(body=body, completion_tokens=50, timestamp=0.0)], tmp_path)
        args = _make_args(path, dataset_args={'match_output_length': True})
        plugin = WorkloadTraceDatasetPlugin(args)
        msg = list(plugin.build_messages())[0]
        assert msg['max_tokens'] == 50

    @pytest.mark.parametrize('bad_value', [0, -1, -100])
    def test_rejects_non_positive_completion_tokens(self, tmp_path, bad_value):
        path = _make_trace_file([_record(completion_tokens=bad_value, timestamp=0.0)], tmp_path)
        with pytest.raises(ValueError, match='completion_tokens must be > 0'):
            WorkloadTraceDatasetPlugin(_make_args(path))

    # -- constrained-decoding guard: ignore_eos must be skipped ----------------

    @pytest.mark.parametrize(
        'rf_type',
        ['json_object', 'json_schema'],
    )
    def test_skips_ignore_eos_for_response_format(self, tmp_path, rf_type, caplog):
        """response_format with json_object/json_schema triggers constrained
        decoding — ignore_eos must NOT be injected."""
        body = {
            'model': 'qwen',
            'messages': [{'role': 'user', 'content': 'hi'}],
            'response_format': {'type': rf_type},
        }
        path = _make_trace_file([_record(body=body, completion_tokens=50, timestamp=0.0)], tmp_path)
        args = _make_args(path, dataset_args={'match_output_length': True})
        plugin = WorkloadTraceDatasetPlugin(args)
        msg = list(plugin.build_messages())[0]
        assert msg['max_tokens'] == 50
        assert 'ignore_eos' not in msg
        assert 'constrained decoding' in caplog.text

    def test_allows_ignore_eos_for_response_format_text(self, tmp_path):
        """response_format with type=text does NOT trigger constrained decoding."""
        body = {
            'model': 'qwen',
            'messages': [{'role': 'user', 'content': 'hi'}],
            'response_format': {'type': 'text'},
        }
        path = _make_trace_file([_record(body=body, completion_tokens=50, timestamp=0.0)], tmp_path)
        args = _make_args(path, dataset_args={'match_output_length': True})
        plugin = WorkloadTraceDatasetPlugin(args)
        msg = list(plugin.build_messages())[0]
        assert msg['max_tokens'] == 50
        assert msg['ignore_eos'] is True

    @pytest.mark.parametrize(
        'tool_choice',
        ['required', {'type': 'function', 'function': {'name': 'get_weather'}}],
    )
    def test_skips_ignore_eos_for_forced_tool_choice(self, tmp_path, tool_choice, caplog):
        """tools + tool_choice=required or named function → constrained decoding."""
        body = {
            'model': 'qwen',
            'messages': [{'role': 'user', 'content': 'hi'}],
            'tools': [{'type': 'function', 'function': {'name': 'get_weather'}}],
            'tool_choice': tool_choice,
        }
        path = _make_trace_file([_record(body=body, completion_tokens=50, timestamp=0.0)], tmp_path)
        args = _make_args(path, dataset_args={'match_output_length': True})
        plugin = WorkloadTraceDatasetPlugin(args)
        msg = list(plugin.build_messages())[0]
        assert msg['max_tokens'] == 50
        assert 'ignore_eos' not in msg
        assert 'constrained decoding' in caplog.text

    @pytest.mark.parametrize('tool_choice', ['auto', 'none'])
    def test_allows_ignore_eos_for_auto_none_tool_choice(self, tmp_path, tool_choice):
        """tools + tool_choice=auto/none → no grammar constraint, ignore_eos safe."""
        body = {
            'model': 'qwen',
            'messages': [{'role': 'user', 'content': 'hi'}],
            'tools': [{'type': 'function', 'function': {'name': 'get_weather'}}],
            'tool_choice': tool_choice,
        }
        path = _make_trace_file([_record(body=body, completion_tokens=50, timestamp=0.0)], tmp_path)
        args = _make_args(path, dataset_args={'match_output_length': True})
        plugin = WorkloadTraceDatasetPlugin(args)
        msg = list(plugin.build_messages())[0]
        assert msg['max_tokens'] == 50
        assert msg['ignore_eos'] is True

    def test_allows_ignore_eos_for_tools_without_tool_choice(self, tmp_path):
        """tools present but no tool_choice key → defaults to auto, safe."""
        body = {
            'model': 'qwen',
            'messages': [{'role': 'user', 'content': 'hi'}],
            'tools': [{'type': 'function', 'function': {'name': 'get_weather'}}],
        }
        path = _make_trace_file([_record(body=body, completion_tokens=50, timestamp=0.0)], tmp_path)
        args = _make_args(path, dataset_args={'match_output_length': True})
        plugin = WorkloadTraceDatasetPlugin(args)
        msg = list(plugin.build_messages())[0]
        assert msg['max_tokens'] == 50
        assert msg['ignore_eos'] is True

    def test_mixed_constrained_and_plain(self, tmp_path, caplog):
        """Only constrained requests skip ignore_eos; plain ones still get it."""
        constrained_body = {
            'model': 'qwen',
            'messages': [{'role': 'user', 'content': 'json'}],
            'response_format': {'type': 'json_object'},
        }
        plain_body = {
            'model': 'qwen',
            'messages': [{'role': 'user', 'content': 'text'}],
        }
        records = [
            _record(body=constrained_body, completion_tokens=30, timestamp=0.0),
            _record(body=plain_body, completion_tokens=60, timestamp=1.0),
        ]
        path = _make_trace_file(records, tmp_path)
        args = _make_args(path, dataset_args={'match_output_length': True})
        plugin = WorkloadTraceDatasetPlugin(args)
        msgs = list(plugin.build_messages())

        # constrained → max_tokens set, no ignore_eos
        assert msgs[0]['max_tokens'] == 30
        assert 'ignore_eos' not in msgs[0]

        # plain → both set
        assert msgs[1]['max_tokens'] == 60
        assert msgs[1]['ignore_eos'] is True

        assert '1 request(s) use constrained decoding' in caplog.text
