from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from evalscope.agent.environments.local import TemporaryLocalAgentEnvironment
from evalscope.agent.tools.bash import BASH_TOOL_INFO, run_bash
from evalscope.api.agent import AgentEnvironment
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.benchmark.adapters import AgentLoopAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.judge import (
    CaseVerdict,
    JudgeCase,
    JudgeContext,
    JudgeDefinition,
    JudgeRequest,
    OutputContract,
    ReducedVerdict,
)
from evalscope.api.messages import ChatMessageSystem, ChatMessageUser
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import ScoringPolicy, Tags
from evalscope.utils.logger import get_logger
from .utils import (
    BINARY_SYSTEM_PROMPT,
    BINARY_USER_PROMPT,
    CHUNK_SYSTEM_PROMPT,
    CHUNK_USER_PROMPT,
    SYNTHESIS_USER_PROMPT,
    BinaryGrade,
    ChunkGrade,
    chunk_document,
)

logger = get_logger()

_EXTRA_PARAMS: Dict[str, Any] = {
    'judge_context_limit': {
        'type': 'int',
        'description': 'Estimated token limit before rubric judging switches to chunking.',
        'value': 150000,
    },
    'judge_chunk_size': {
        'type': 'int',
        'description': 'Maximum estimated tokens in each document chunk sent to the judge.',
        'value': 100000,
    },
}

_DESCRIPTION = """
## Overview

ResearchRubrics evaluates Deep Research agents on realistic, open-ended research tasks. Each task pairs a user prompt
with expert-written, fine-grained rubrics covering explicit and implicit requirements, information synthesis,
references, communication quality, and instruction following.

## Task Description

- **Task Type**: Multi-turn research agent / long-form report generation
- **Input**: One open-ended research prompt
- **Output**: A Markdown research report produced after iterative tool use
- **Dataset**: 101 tasks and 2,593 weighted rubric criteria
- **Metric**: Binary rubric compliance score

## Key Features

- 101 open-ended Deep Research tasks paired with 2,593 expert-written, weighted rubric criteria.
- Rubrics cover explicit and implicit requirements, information synthesis, references, communication
  quality, and instruction following, and each is graded independently.
- Negative-weight criteria capture undesirable behaviours and subtract from the score when present.
- Long reports are graded with the official chunk-evidence-synthesis procedure once they exceed the
  configured judge context threshold.

## Agent Environment

- Uses EvalScope's built-in agent environment by default and does not require ``agent_config``. The agent can use ``bash``
  to access the network, gather information, and produce a final report.
- The default environment uses the host network and a temporary working directory, but does not provide complete filesystem
  isolation. Do not run untrusted models on shared or sensitive machines.
- The default strategy is ``function_calling`` with a 50-step limit. Use ``NativeAgentConfig`` to override the strategy
  or step limit; ``react`` is also available. Both strategies require native function calling support.
- Add dedicated search or web-fetching tools through ``NativeAgentConfig``, or use ``ExternalAgentConfig`` to run the
  task with another agent framework.
- When the step limit is reached, the model is asked to produce a final report from the information already collected so
  the result can still be reviewed and scored.

## Evaluation Notes

- ResearchRubrics requires ``judge.models`` and ``judge.strategy='auto'`` or ``'llm'``. Gemini 2.5 Pro is the
  recommended judge for comparison with the paper, but no provider or model is hard-coded.
- Every rubric is graded independently as Satisfied (1) or Not Satisfied (0), matching the public binary grader. The
  paper's ternary scores are not directly comparable.
- Negative-weight criteria subtract from the numerator when the undesirable behavior is present. Scores are not
  clipped.
- Long reports are evaluated with the official chunk-evidence-synthesis approach when they exceed the configured judge
  context threshold.
- A full run performs 2,593 rubric evaluations and can be expensive. Current-events tasks are also sensitive to the
  date and web sources available at evaluation time.

## Configuration

- ``judge_context_limit``: 150,000 estimated tokens
- ``judge_chunk_size``: 100,000 estimated tokens

The judge must be configured explicitly. For example:

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='YOUR_AGENT_MODEL',
    datasets=['researchrubrics'],
    judge={
        'strategy': 'llm',
        'models': {
            'model_id': 'YOUR_JUDGE_MODEL',
            'api_url': 'OPENAI_COMPATIBLE_JUDGE_URL',
            'api_key': 'YOUR_JUDGE_API_KEY',
            'generation_config': {'temperature': 0.0},
        },
    },
    limit=1,
))
```

Resources: [Paper](https://arxiv.org/abs/2511.07685) |
[GitHub](https://github.com/scaleapi/researchrubrics) |
[Dataset](https://modelscope.cn/datasets/evalscope/researchrubrics)
"""


@register_benchmark(
    BenchmarkMeta(
        name='researchrubrics',
        pretty_name='ResearchRubrics',
        tags=[Tags.AGENT, Tags.MULTI_TURN, Tags.RETRIEVAL, Tags.REASONING],
        description=_DESCRIPTION,
        dataset_id='evalscope/researchrubrics',
        paper_url='https://arxiv.org/abs/2511.07685',
        subset_list=['default'],
        default_subset='default',
        eval_split='train',
        metric_list=['compliance_score'],
        prompt_template='{question}',
        extra_params=_EXTRA_PARAMS,
    )
)
class ResearchRubricsAdapter(AgentLoopAdapter):
    """Deep Research agent benchmark with binary rubric-based LLM judging."""
    scoring_policy = ScoringPolicy.JUDGE_ONLY

    strategy_name = 'function_calling'
    max_steps_default = 50

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.judge_context_limit = int(self.extra_params.get('judge_context_limit', 150000))
        self.judge_chunk_size = int(self.extra_params.get('judge_chunk_size', 100000))
        if self.judge_context_limit <= 0 or self.judge_chunk_size <= 0:
            raise ValueError('ResearchRubrics judge context and chunk limits must be greater than 0.')
        self.use_batch_scoring = True

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        rubrics = record.get('rubrics')
        if not isinstance(rubrics, list):
            raise ValueError('ResearchRubrics record must contain a rubrics list.')
        return Sample(
            input=str(record['prompt']),
            target=json.dumps(rubrics, ensure_ascii=False),
            tools=[BASH_TOOL_INFO],
            metadata={
                'sample_id': record.get('sample_id'),
                'domain': record.get('domain'),
                'conceptual_breadth': record.get('conceptual_breadth'),
                'logical_nesting': record.get('logical_nesting'),
                'exploration': record.get('exploration'),
            },
        )

    def build_tools(self, sample: Sample) -> Dict[str, Any]:
        return {'bash': run_bash}

    def build_environment(self, sample: Sample) -> Optional[AgentEnvironment]:
        sample_id = sample.metadata.get('sample_id') or sample.id or 'unknown'
        return TemporaryLocalAgentEnvironment(sample_id=sample_id, prefix='evalscope-researchrubrics-')

    def build_max_steps_finalization_message(self, sample: Sample) -> str:
        return (
            'The tool-use budget is exhausted. Using the research already gathered, write and return the complete '
            'final Markdown report now. Do not call any tools.'
        )

    def calculate_metrics(self, task_state: TaskState) -> SampleScore:
        """Return a placeholder; expensive rubric judging runs after predictions are persisted."""
        return SampleScore(
            score=Score(
                extracted_prediction=task_state.output.completion,
                prediction=task_state.output.completion,
                value={},
            ),
            sample_id=task_state.sample_id,
            group_id=task_state.group_id,
            sample_metadata=task_state.metadata,
        )

    def batch_calculate_metrics(
        self,
        task_states: List[TaskState],
        sample_scores: List[SampleScore],
    ) -> List[SampleScore]:
        if not task_states:
            return sample_scores
        self.llm_judge  # Initialize once before worker threads access it.
        workers = min(max(int(self._task_config.eval_batch_size), 1), len(task_states))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            reviewed_scores = list(executor.map(self._score_task_state, task_states))
        for sample_score, reviewed_score in zip(sample_scores, reviewed_scores):
            sample_score.score = reviewed_score
        return sample_scores

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        if not sample_scores:
            return []

        aggregate_groups: Dict[str, List[SampleScore]] = {'compliance_score': sample_scores}
        for field in ['domain', 'conceptual_breadth', 'logical_nesting', 'exploration']:
            grouped: Dict[str, List[SampleScore]] = defaultdict(list)
            for sample_score in sample_scores:
                value = (sample_score.sample_metadata or {}).get(field)
                if value:
                    grouped[str(value)].append(sample_score)
            for value, scores in grouped.items():
                aggregate_groups[f'{field}/{value}'] = scores

        axis_names = []
        for sample_score in sample_scores:
            for key in sample_score.score.value:
                if key.startswith('axis/') and key not in axis_names:
                    axis_names.append(key)

        results = [self._mean_agg_score('compliance_score', sample_scores, 'compliance_score')]
        for axis_name in axis_names:
            scores = [sample_score for sample_score in sample_scores if axis_name in sample_score.score.value]
            results.append(self._mean_agg_score(axis_name, scores, axis_name))
        for metric_name, scores in aggregate_groups.items():
            if metric_name == 'compliance_score':
                continue
            results.append(self._mean_agg_score(metric_name, scores, 'compliance_score'))
        return [result for result in results if result is not None]

    def _score_task_state(self, task_state: TaskState) -> Score:
        report = task_state.output.completion or ''
        if not report.strip():
            raise ValueError(f'ResearchRubrics sample {task_state.metadata.get("sample_id")} produced an empty report.')
        try:
            rubrics = json.loads(task_state.target)
        except json.JSONDecodeError as exc:
            raise ValueError('ResearchRubrics target must be a JSON-encoded rubric list.') from exc
        if not isinstance(rubrics, list) or not rubrics:
            raise ValueError('ResearchRubrics target must contain at least one rubric.')

        score = self.score_with_judge_contracts(report, report, task_state.target, task_state)

        trace = task_state.agent_trace
        tool_names = {'bash'}
        if trace:
            for event in trace.events:
                tool_name = event.payload.get('name') or event.payload.get('tool_name') or event.payload.get('function')
                if tool_name:
                    tool_names.add(str(tool_name))
        score.metadata = {
            **(score.metadata or {}),
            'agent': {
                'framework': trace.framework if trace else None,
                'strategy': trace.strategy if trace else self.strategy_name,
                'environment': trace.environment if trace else 'local',
                'max_steps': trace.max_steps if trace else self.max_steps,
                'tools': sorted(tool_names),
            },
        }
        return score

    # -- Judge contract hooks --

    def _uses_chunking(self, report: str) -> bool:
        """Whether the report is judged chunk-by-chunk. Read by both the case and expansion hooks:
        if the two disagreed, chunk cases would be emitted with no synthesis to fold them."""
        return len(report) // 4 > self.judge_context_limit

    def _report_chunks(self, report: str) -> List[str]:
        """Chunks of one report, cached so each rubric and request does not re-split it.

        The cached tuple is read once: samples are scored on worker threads, so a two-step
        read could hand one sample's chunks to another.
        """
        cached = getattr(self, '_chunk_cache', None)
        if cached is not None and cached[0] == report:
            return cached[1]
        chunks = chunk_document(report, max_tokens=self.judge_chunk_size)
        self._chunk_cache = (report, chunks)
        return chunks

    def judge_definition(self, context: JudgeContext) -> JudgeDefinition:
        return JudgeDefinition.workflow(
            cases=self._build_cases(context),
            request=self._build_request,
            reduce=self._reduce_verdicts,
            main_score_name='compliance_score',
            expand=self._expand_cases,
            finalize=self._finalize_score
        )

    def _build_cases(self, context: JudgeContext) -> List[JudgeCase]:
        rubrics = json.loads(context.reference)
        report = context.filtered_prediction
        used_chunking = self._uses_chunking(report)
        binary_contract = OutputContract(schema_model=BinaryGrade)
        chunk_contract = OutputContract(schema_model=ChunkGrade)
        cases: List[JudgeCase] = []
        for idx, rubric in enumerate(rubrics):
            criterion = str(rubric.get('criterion', '')).strip()
            axis = str(rubric.get('axis', '')).strip()
            weight = float(rubric.get('weight', 0))
            if used_chunking:
                chunks = self._report_chunks(report)
                for chunk_idx in range(len(chunks)):
                    cases.append(
                        JudgeCase(
                            case_id=f'rubric_{idx}_chunk_{chunk_idx}',
                            output_contract=chunk_contract,
                            metadata={
                                'kind': 'chunk',
                                'rubric_idx': idx,
                                'chunk_idx': chunk_idx,
                                'total_chunks': len(chunks),
                                'criterion': criterion,
                                'axis': axis,
                                'weight': weight,
                            },
                        )
                    )
            else:
                cases.append(
                    JudgeCase(
                        case_id=f'rubric_{idx}',
                        output_contract=binary_contract,
                        metadata={
                            'kind': 'binary',
                            'rubric_idx': idx,
                            'criterion': criterion,
                            'axis': axis,
                            'weight': weight,
                        },
                    )
                )
        return cases

    def _expand_cases(self, stage: int, completed_cases: List[CaseVerdict], context: JudgeContext) -> List[JudgeCase]:
        if stage != 1:
            return []
        # Emit synthesis cases for rubrics whose chunks are all complete.
        rubrics = json.loads(context.reference)
        report = context.filtered_prediction
        if not self._uses_chunking(report):
            return []
        # Collect evidence per rubric from completed chunk verdicts.
        evidence_by_rubric: Dict[int, List[str]] = defaultdict(list)
        for cv in completed_cases:
            if cv.metadata.get('kind') != 'chunk':
                continue
            evidence_by_rubric[cv.metadata['rubric_idx']].extend(cv.value.relevant_evidence)
        synthesis_cases: List[JudgeCase] = []
        for rubric_idx, evidence in evidence_by_rubric.items():
            rubric = rubrics[rubric_idx]
            synthesis_cases.append(
                JudgeCase(
                    case_id=f'rubric_{rubric_idx}_synthesis',
                    output_contract=OutputContract(schema_model=BinaryGrade),
                    metadata={
                        'kind': 'synthesis',
                        'rubric_idx': rubric_idx,
                        'criterion': str(rubric.get('criterion', '')).strip(),
                        'axis': str(rubric.get('axis', '')).strip(),
                        'weight': float(rubric.get('weight', 0)),
                        'evidence': evidence,
                    },
                )
            )
        return synthesis_cases

    def _build_request(self, case, placement, completed_cases, context) -> JudgeRequest:
        meta = case.metadata
        kind = meta['kind']
        if kind == 'binary':
            prompt = BINARY_USER_PROMPT.format(
                document_content=context.filtered_prediction,
                rubric_title=meta['criterion'],
                rubric_category=meta['axis'],
                rubric_weight=meta['weight'],
            )
            return JudgeRequest(
                messages=[ChatMessageSystem(content=BINARY_SYSTEM_PROMPT),
                          ChatMessageUser(content=prompt)]
            )
        elif kind == 'chunk':
            chunks = self._report_chunks(context.filtered_prediction)
            chunk_content = chunks[meta['chunk_idx']]
            chunk_num = meta['chunk_idx'] + 1
            prompt = CHUNK_USER_PROMPT.format(
                chunk_num=chunk_num,
                total_chunks=meta['total_chunks'],
                context_summary='Previous chunks evaluated' if chunk_num > 1 else 'First chunk',
                chunk_content=chunk_content,
                rubric_title=meta['criterion'],
                rubric_category=meta['axis'],
            )
            return JudgeRequest(
                messages=[ChatMessageSystem(content=CHUNK_SYSTEM_PROMPT),
                          ChatMessageUser(content=prompt)]
            )
        else:  # synthesis
            synthesis_prompt = SYNTHESIS_USER_PROMPT.format(
                all_evidence=json.dumps(meta['evidence'], ensure_ascii=False, indent=2),
                rubric_title=meta['criterion'],
                rubric_category=meta['axis'],
            )
            return JudgeRequest(
                messages=[ChatMessageSystem(content=BINARY_SYSTEM_PROMPT),
                          ChatMessageUser(content=synthesis_prompt)]
            )

    def _reduce_verdicts(self, case_verdicts: List[CaseVerdict], context: JudgeContext) -> ReducedVerdict:
        rubrics = json.loads(context.reference)
        # Collect final scores: binary verdicts + synthesis verdicts (skip raw chunk verdicts).
        rubric_scores: Dict[int, Dict[str, Any]] = {}
        used_chunking = False
        for cv in case_verdicts:
            if '_chunk_' in cv.case_id and '_synthesis' not in cv.case_id:
                used_chunking = True
                continue
            parts = cv.case_id.replace('_synthesis', '').split('_')
            rubric_idx = int(parts[1])
            if '_synthesis' in cv.case_id:
                used_chunking = True
            rubric = rubrics[rubric_idx]
            rubric_scores[rubric_idx] = {
                'score': cv.value.score,
                'weight': float(rubric.get('weight', 0)),
                'axis': str(rubric.get('axis', '')).strip(),
            }

        entries = list(rubric_scores.values())
        compliance = self._weighted_compliance(entries)
        values: Dict[str, float] = {'compliance_score': compliance}
        axis_entries: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for entry in entries:
            axis_entries[entry['axis']].append(entry)
        for axis, axis_list in axis_entries.items():
            if any(e['weight'] > 0 for e in axis_list):
                values[f'axis/{axis}'] = self._weighted_compliance(axis_list)

        return ReducedVerdict(
            value=values,
            metadata={
                'source': 'researchrubrics_binary_judge',
                'grading_mode': 'binary',
                'rubric_count': len(rubric_scores),
                'used_chunking': used_chunking,
            },
        )

    def _finalize_score(self, score: Score, review, context) -> Score:
        score.explanation = f'Binary rubric compliance across {len(json.loads(context.reference))} criteria.'
        return score

    @staticmethod
    def _weighted_compliance(entries: List[Dict[str, Any]]) -> float:
        denominator = sum(float(entry['weight']) for entry in entries if float(entry['weight']) > 0)
        if denominator <= 0:
            raise ValueError('ResearchRubrics requires at least one positive-weight rubric.')
        numerator = sum(float(entry['score']) * float(entry['weight']) for entry in entries)
        return numerator / denominator

    @staticmethod
    def _mean_agg_score(metric_name: str, sample_scores: List[SampleScore], value_key: str) -> Optional[AggScore]:
        # A sample whose judge review was unusable carries an empty value dict, so it holds no
        # value for this key: it is excluded from the mean rather than counted as 0. A group with
        # no usable sample yields no metric row at all.
        scored = [sample_score for sample_score in sample_scores if value_key in sample_score.score.value]
        if not scored:
            return None
        values = [float(sample_score.score.value[value_key]) for sample_score in scored]
        return AggScore(
            metric_name=metric_name,
            score=sum(values) / len(values),
            aggregation='mean',
            num=len(values),
            ids=[sample_score.sample_id for sample_score in scored],
        )
