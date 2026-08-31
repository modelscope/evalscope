from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict

from evalscope.api.agent import AgentEnvironment
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.benchmark.adapters import AgentLoopAdapter
from evalscope.api.dataset import Sample
from evalscope.api.judge import JudgeCase, JudgeContext, JudgeDefinition, JudgeRequest, OutputContract, ReducedVerdict
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.api.tool import ToolCall, ToolInfo
from evalscope.constants import ScoringPolicy, Tags
from evalscope.utils.logger import get_logger

from .metadata import DATASET_ID, DEFAULT_MCP_SERVER_URL, DEFAULT_SYSTEM_PROMPT, DESCRIPTION, EXTRA_PARAMS
from .utils import (
    MCPAtlasClient,
    MCPAtlasServerUnavailable,
    claim_judge_prompt,
    extract_claims,
    extract_required_servers,
    field,
    mcp_tool_to_tool_info,
    parse_enabled_tools,
    server_unavailable_message,
    tool_name_to_server,
)

logger = get_logger()


class ClaimVerdict(BaseModel):
    model_config = ConfigDict(extra='ignore')

    coverage_outcome: Literal['fulfilled', 'partially_fulfilled', 'not_fulfilled']
    justification: str = ''


# The claim_judge_prompt already asks for this JSON, so no instruction() is appended.
CLAIM_CONTRACT = OutputContract(schema_model=ClaimVerdict)
_OUTCOME_SCORE = {'fulfilled': 1.0, 'partially_fulfilled': 0.5, 'not_fulfilled': 0.0}


@register_benchmark(
    BenchmarkMeta(
        name='mcp_atlas',
        pretty_name='MCP-Atlas',
        tags=[Tags.AGENT, Tags.MULTI_TURN],
        description=DESCRIPTION,
        dataset_id=DATASET_ID,
        subset_list=['default'],
        default_subset='default',
        eval_split='train',
        prompt_template='{question}',
        metric_list=['coverage_score', 'pass_rate'],
        primary_metric='pass_rate',
        extra_params=EXTRA_PARAMS,
        paper_url='https://static.scale.com/uploads/674f4cc7a74e35bcaae1c29a/MCP_Atlas.pdf',
    )
)
class MCPAtlasAdapter(AgentLoopAdapter):
    """EvalScope-native MCP-Atlas adapter using MCP-Atlas agent-environment."""

    scoring_policy = ScoringPolicy.JUDGE_ONLY

    strategy_name = 'function_calling'
    max_steps_default = 100

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.mcp_server_url = str(self.extra_params.get('mcp_server_url') or DEFAULT_MCP_SERVER_URL)
        self.filter_enabled_servers = bool(self.extra_params.get('filter_enabled_servers', True))
        self.max_tool_calls = int(self.extra_params.get('max_tool_calls', 100))
        self.request_timeout = float(self.extra_params.get('request_timeout', 60.0))
        self.list_tools_timeout = float(self.extra_params.get('list_tools_timeout', 180.0))
        self.use_system_prompt = bool(self.extra_params.get('use_system_prompt', False))
        self.pass_threshold = float(self.extra_params.get('pass_threshold', 0.75))
        self._client: Optional[MCPAtlasClient] = None
        self._enabled_servers: Optional[List[str]] = None
        self._tool_infos_by_name: Optional[Dict[str, ToolInfo]] = None
        self._tool_calls_by_sample: Dict[int, int] = {}
        self._server_failures_by_sample: Dict[int, Dict[str, str]] = {}
        self._excluded_tasks: List[Dict[str, Any]] = []

    @property
    def client(self) -> MCPAtlasClient:
        if self._client is None:
            self._client = MCPAtlasClient(
                base_url=self.mcp_server_url,
                request_timeout=self.request_timeout,
                list_tools_timeout=self.list_tools_timeout,
            )
        return self._client

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        task_id = str(field(record, 'TASK', 'task', 'task_id') or '')
        prompt = str(field(record, 'PROMPT', 'prompt') or '')
        enabled_tools = parse_enabled_tools(field(record, 'ENABLED_TOOLS', 'enabled_tools') or '[]')
        trajectory = field(record, 'TRAJECTORY', 'trajectory') or '[]'
        claims = extract_claims(field(record, 'GTFA_CLAIMS', 'gtfa_claims', 'rubrics') or '[]')
        required_servers = extract_required_servers(trajectory)

        input_text = prompt
        if self.use_system_prompt:
            input_text = f'{DEFAULT_SYSTEM_PROMPT}\n\n{prompt}'

        tool_infos = self._ensure_tool_infos()
        return Sample(
            input=input_text,
            target=json.dumps(claims, ensure_ascii=False),
            tools=[tool_infos[name] for name in enabled_tools if name in tool_infos],
            metadata={
                'task_id': task_id,
                'prompt': prompt,
                'enabled_tools': enabled_tools,
                'trajectory': trajectory,
                'gtfa_claims': claims,
                'required_servers': required_servers,
                'mcp_server_url': self.mcp_server_url,
            },
        )

    def build_tools(self, sample: Sample) -> Dict[str, Any]:
        sample_key = int(sample.id or 0)
        self._tool_calls_by_sample[sample_key] = 0
        self._server_failures_by_sample[sample_key] = {}
        handlers = {}
        for tool_name in sample.metadata.get('enabled_tools', []):
            handlers[tool_name] = self._make_tool_handler(tool_name, sample_key)
        return handlers

    def build_environment(self, sample: Sample) -> Optional[AgentEnvironment]:
        return None

    def judge_definition(self, context: JudgeContext) -> JudgeDefinition:
        cases = [
            JudgeCase(case_id=f'claim_{index}', output_contract=CLAIM_CONTRACT, metadata={'claim': claim})
            for index, claim in enumerate(extract_claims(context.reference))
        ]

        def request(case, placement, completed_cases, judge_context) -> JudgeRequest:
            return JudgeRequest(
                messages=[
                    ChatMessageUser(
                        content=claim_judge_prompt(case.metadata['claim'], judge_context.filtered_prediction)
                    )
                ]
            )

        def reduce(case_verdicts, judge_context) -> ReducedVerdict:
            scores = [_OUTCOME_SCORE[verdict.value.coverage_outcome] for verdict in case_verdicts]
            total = len(scores)
            coverage_score = sum(scores) / total if total else 0.0
            return ReducedVerdict(
                value={'coverage_score': coverage_score, 'pass': 1.0 if coverage_score >= self.pass_threshold else 0.0},
                metadata={
                    'pass_threshold': self.pass_threshold,
                    'total_claims': total,
                    'fully_covered_claims': scores.count(1.0),
                    'partially_covered_claims': scores.count(0.5),
                },
            )

        return JudgeDefinition.workflow(cases=cases, request=request, reduce=reduce, main_score_name='coverage_score')

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        if not sample_scores:
            return []
        # A sample whose judge review was unusable carries an empty value dict; it is excluded from
        # the means rather than counted as 0.
        scored = [sample_score for sample_score in sample_scores if sample_score.score.value]
        if not scored:
            return []
        coverage_values = [float(sample_score.score.value.get('coverage_score', 0.0)) for sample_score in scored]
        pass_values = [float(sample_score.score.value.get('pass', 0.0)) for sample_score in scored]
        sample_ids = [sample_score.sample_id for sample_score in scored]
        return [
            AggScore(
                metric_name='coverage_score',
                score=sum(coverage_values) / len(coverage_values),
                aggregation='mean',
                num=len(coverage_values),
                ids=sample_ids,
            ),
            AggScore(
                metric_name='pass_rate',
                score=sum(pass_values) / len(pass_values),
                aggregation='mean',
                num=len(pass_values),
                ids=sample_ids,
                metadata={'pass_threshold': self.pass_threshold},
            ),
        ]

    def _preflight(self) -> None:
        try:
            self._enabled_servers = self.client.enabled_servers()
            self._tool_infos_by_name = {
                tool.name: tool for tool in (mcp_tool_to_tool_info(raw_tool) for raw_tool in self.client.list_tools())
            }
        except Exception as exc:
            raise RuntimeError(
                'MCP-Atlas agent-environment is not available. Start the MCP-Atlas Docker service so '
                f'{self.mcp_server_url}/enabled-servers and /list-tools are reachable. Original error: {exc}'
            ) from exc

    def _ensure_tool_infos(self) -> Dict[str, ToolInfo]:
        if self._tool_infos_by_name is None:
            self._preflight()
        return self._tool_infos_by_name or {}

    def sample_filter(self, sample: Sample) -> bool:
        if not self.filter_enabled_servers:
            return True
        if self._enabled_servers is None:
            self._preflight()
        enabled = set(self._enabled_servers or [])
        required_servers = sample.metadata.get('required_servers') or []
        missing = [server for server in required_servers if server not in enabled]
        if not missing:
            return True
        self._excluded_tasks.append(
            {
                'task_id': sample.metadata.get('task_id'),
                'missing_servers': missing,
            }
        )
        logger.warning(
            'Skipping MCP-Atlas task %s because required servers are not enabled: %s',
            sample.metadata.get('task_id'),
            missing,
        )
        return False

    def _make_tool_handler(self, tool_name: str, sample_key: int):

        async def _handler(call: ToolCall, env: Optional[AgentEnvironment]) -> str:
            del env
            server_name = tool_name_to_server(tool_name)
            failures = self._server_failures_by_sample.setdefault(sample_key, {})
            if server_name in failures:
                return server_unavailable_message(server_name, failures[server_name])
            count = self._tool_calls_by_sample.get(sample_key, 0)
            if count >= self.max_tool_calls:
                return f'MCP-Atlas tool call limit exceeded ({self.max_tool_calls}).'
            self._tool_calls_by_sample[sample_key] = count + 1
            try:
                return await asyncio.to_thread(self.client.call_tool, tool_name, call.function.arguments)
            except MCPAtlasServerUnavailable as exc:
                failures[exc.server_name] = exc.message
                return server_unavailable_message(exc.server_name, exc.message)

        return _handler
