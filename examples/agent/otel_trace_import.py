"""Score a recorded OpenTelemetry GenAI trace without re-running the agent.

This is the end-to-end entry point for the v1 contract: one export per sample,
the sample's own final answer becomes the prediction, and the recorded tool
result reaches the scorer through the reconstructed message history.

Run it from the repository root:

    python examples/agent/otel_trace_import.py

The two exports under ``tests/agent/fixtures/otel/`` were produced by the
OpenTelemetry Python SDK (see ``_generate.py`` next to them), so this scores a
real OTLP/JSON export, not a hand-written dict. In a real setup you would
point ``trace_path`` at your exporter's output directory.
"""
import tempfile
from pathlib import Path

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.benchmark.adapters.agent_adapter import AgentAdapter
from evalscope.api.dataset import DatasetDict, Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageTool
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.config import TaskConfig
from evalscope.run import run_task

FIXTURES = Path(__file__).resolve().parents[2] / 'tests' / 'agent' / 'fixtures' / 'otel'


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
    """Two samples; each expects the answer the recorded run produced."""

    def load(self):
        # Sample.id is an integer; the export is ``<id>.otlp.json``.
        return DatasetDict(
            {
                'default': [
                    Sample(id=0, input='What is the weather in Paris?', target='rainy'),
                    Sample(id=1, input='What is the capital of France?', target='Paris'),
                ]
            }
        ), None

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: str, task_state: TaskState
    ) -> Score:
        """The answer must contain the expected word *and* the tool result must
        reach the scorer — without the reconstructed tool message the score is 0."""
        tool_results = [m.text for m in task_state.messages if isinstance(m, ChatMessageTool)]
        correct = reference in (filtered_prediction or '') and reference in tool_results
        return Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value={'acc': 1.0 if correct else 0.0},
        )


def main() -> None:
    with tempfile.TemporaryDirectory() as work_dir:
        scores = run_task(
            TaskConfig(
                model='otel_demo_model',
                datasets=['otel_trace_demo'],
                agent_config={'mode': 'otel_trace', 'trace_path': str(FIXTURES)},
                eval_type='mock_llm',
                work_dir=work_dir,
            )
        )

    # Show the tool evidence and turn order the importer reconstructed.
    from types import SimpleNamespace

    from evalscope.api.agent.otel import load_otlp_trace_for_sample
    from evalscope.api.agent.otel_config import OTelTraceConfig

    imported = load_otlp_trace_for_sample(
        OTelTraceConfig(trace_path=str(FIXTURES)),
        SimpleNamespace(name='demo'),
        SimpleNamespace(id=0),
        n_subsets_with_id=1,
    )
    tool_results = [m.text for m in imported.messages if isinstance(m, ChatMessageTool)]
    print('\n[tool evidence] tool results reaching the scorer:', tool_results)
    print('[turn order]', [m.role for m in imported.messages])
    print('[score]', scores['otel_trace_demo'].metrics[0].num_correct, '/ 2 correct')


if __name__ == '__main__':
    main()
