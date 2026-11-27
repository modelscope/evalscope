"""Score a recorded OpenTelemetry GenAI trace without re-running the agent.

This example shows the v1 contract: one export per sample, the sample's own
final answer becomes the prediction, and the recorded tool result reaches the
scorer through the reconstructed message history.

Run it:

    python examples/agent/otel_trace_import.py

The trace fixtures are generated in the OpenTelemetry OTLP JSON shape (the
format the OpenTelemetry SDKs emit). In a real setup you would point
``trace_path`` at your exporter's output.
"""
import json
import os
import tempfile
from typing import Any, Dict, List, Tuple

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import DatasetDict, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import Score
from evalscope.api.benchmark.adapters.agent_adapter import AgentAdapter
from evalscope.api.registry import register_benchmark
from evalscope.config import TaskConfig
from evalscope.run import run_task


# ---------------------------------------------------------------------------
# 1. A minimal agent benchmark that scores from the saved run
# ---------------------------------------------------------------------------
@register_benchmark(
    BenchmarkMeta(
        name='otel_trace_demo',
        pretty_name='OTel-Trace-Import-Demo',
        dataset_id='local',
        description='Scores a recorded OTel trace against the sample it answered.',
        subset_list=['default'],
        metric_list=['acc'],
        eval_split='test',
        prompt_template='{question}',
    )
)
class OTelTraceDemoAdapter(AgentAdapter):
    """Two samples; each expects a specific answer the recorded run produced."""

    def load(self) -> Tuple[DatasetDict, None]:
        # Sample.id is an integer; the export file is named ``<id>.otlp.json``.
        samples = [
            Sample(id=0, input='What is the weather?', target='rainy'),
            Sample(id=1, input='What is the capital of France?', target='paris'),
        ]
        return DatasetDict({'default': samples}), None

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """Exact containment: the final answer must contain the expected word."""
        return Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value={'acc': 1.0 if reference.lower() in (filtered_prediction or '').lower() else 0.0},
        )


# ---------------------------------------------------------------------------
# 3. OTLP fixture helpers (mirror the OpenTelemetry SDK JSON export)
# ---------------------------------------------------------------------------
def _str(value: str) -> Dict[str, Any]:
    return {'stringValue': value}


def _int(value: int) -> Dict[str, Any]:
    return {'intValue': value}


def _text_part(content: str) -> Dict[str, Any]:
    return {'type': 'text', 'content': content}


def _messages(*entries: Dict[str, Any]) -> Dict[str, Any]:
    return {'arrayValue': {'values': [{'kvlistValue': {'values': _kv_list(e)}} for e in entries]}}


def _any(value: Any) -> Dict[str, Any]:
    if isinstance(value, int):
        return _int(value)
    if isinstance(value, dict):
        return {'kvlistValue': {'values': _kv_list(value)}}
    if isinstance(value, list):
        return {'arrayValue': {'values': [_any(v) for v in value]}}
    return _str(str(value))


def _kv_list(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{'key': k, 'value': _any(v)} for k, v in obj.items()]


def _chat_span(span_id: str, trace_id: str, *, question: str, tool_call: bool, tool_result: str, answer: str):
    """One chat turn; the tool result is encoded as a tool_call_response part."""
    history = [{'role': 'user', 'parts': [_text_part(question)]}]
    if tool_call:
        history.append(
            {
                'role': 'assistant',
                'parts': [{'type': 'tool_call', 'id': f'{span_id}-call', 'name': 'lookup', 'arguments': {'q': question}}],
            }
        )
        history.append(
            {
                'role': 'tool',
                'parts': [{'type': 'tool_call_response', 'id': f'{span_id}-call', 'response': tool_result}],
            }
        )
    return {
        'traceId': trace_id,
        'spanId': span_id,
        'parentSpanId': '',
        'name': 'chat',
        'startTimeUnixNano': '1000000',
        'endTimeUnixNano': '2000000',
        'attributes': [
            {'key': 'gen_ai.operation.name', 'value': _str('chat')},
            {'key': 'gen_ai.request.model', 'value': _str('demo-model')},
            {'key': 'gen_ai.usage.input_tokens', 'value': _int(10)},
            {'key': 'gen_ai.usage.output_tokens', 'value': _int(5)},
            {'key': 'gen_ai.input.messages', 'value': _messages(*history)},
            {'key': 'gen_ai.output.messages', 'value': _messages({'role': 'assistant', 'parts': [_text_part(answer)]})},
        ],
    }


def write_fixtures(directory: str) -> None:
    """One OTLP export per sample: <sample_id>.otlp.json."""
    fixtures = {
        # Answer produced by the recorded run. The tool result ("rainy") is what
        # makes the answer correct — a scorer that never saw it would fail.
        0: _chat_span(
            'c1', 'trace-1', question='What is the weather?', tool_call=True, tool_result='rainy', answer='The weather is rainy.'
        ),
        1: _chat_span(
            'c2', 'trace-2', question='What is the capital?', tool_call=True, tool_result='paris', answer='The capital is Paris.'
        ),
    }
    for sample_id, span in fixtures.items():
        payload = {'resourceSpans': [{'scopeSpans': [{'spans': [span]}]}]}
        with open(os.path.join(directory, f'{sample_id}.otlp.json'), 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)


# ---------------------------------------------------------------------------
# 4. Run through the same TaskConfig the CLI parses
# ---------------------------------------------------------------------------
def main() -> None:
    with tempfile.TemporaryDirectory() as trace_dir:
        write_fixtures(trace_dir)

        task_config = TaskConfig(
            model='otel_demo_model',
            datasets=['otel_trace_demo'],
            agent_config={'mode': 'otel_trace', 'trace_path': trace_dir},
            eval_type='mock_llm',
            work_dir=os.path.join(trace_dir, 'outputs'),
        )
        scores = run_task(task_config)

        # Acceptance case 2: the recorded tool result must reach the scorer.
        # Re-import sample 0 and assert the tool observation is in the history;
        # the final answer ("...rainy.") is only correct because of it.
        from types import SimpleNamespace

        from evalscope.api.agent.otel import load_otlp_trace_for_sample
        from evalscope.api.agent.otel_config import OTelTraceConfig
        from evalscope.api.messages import ChatMessageTool

        imported = load_otlp_trace_for_sample(
            OTelTraceConfig(trace_path=trace_dir),
            SimpleNamespace(name='demo'),
            SimpleNamespace(id=0),
        )
        tool_results = [m.text for m in imported.messages if isinstance(m, ChatMessageTool)]
        assert tool_results == ['rainy'], f'tool evidence lost: {tool_results}'
        assert [m.role for m in imported.messages] == ['user', 'assistant', 'tool', 'assistant']
        print('\n[tool evidence] tool results reaching the scorer:', tool_results)
        print('[turn order]', [m.role for m in imported.messages])

    print('\n=== Imported-trace scores ===')
    print(json.dumps(scores, indent=2, default=str))


if __name__ == '__main__':
    main()
