from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from evalscope.api.agent import AgentEnvironment
from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.benchmark.adapters import AgentLoopAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import AggScore, SampleScore, Score
from evalscope.api.registry import register_benchmark
from evalscope.api.tool import ToolCall, ToolInfo
from evalscope.constants import Tags
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
    parse_claim_judge_response,
    parse_enabled_tools,
    server_unavailable_message,
    tool_name_to_server,
)

logger = get_logger()


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
        metric_list=['coverage_score', 'pass'],
        extra_params=EXTRA_PARAMS,
        paper_url='https://static.scale.com/uploads/674f4cc7a74e35bcaae1c29a/MCP_Atlas.pdf',
    )
)
class MCPAtlasAdapter(AgentLoopAdapter):
    """EvalScope-native MCP-Atlas adapter using MCP-Atlas agent-environment."""

    strategy_name = 'function_calling'
    max_steps_default = 100

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._use_llm_judge = True
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

    def match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        raise ValueError('MCP-Atlas requires LLM judge scoring; set judge_strategy to auto/llm with judge_model_args.')

    def llm_match_score(
        self,
        original_prediction: str,
        filtered_prediction: str,
        reference: str,
        task_state: TaskState,
    ) -> Score:
        claims = extract_claims(reference)
        claim_results = [self._judge_claim(claim, filtered_prediction) for claim in claims]
        total = len(claim_results)
        coverage_score = sum(result['score'] for result in claim_results) / total if total else 0.0
        passed = coverage_score >= self.pass_threshold

        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
            value={
                'coverage_score': coverage_score,
                'pass': 1.0 if passed else 0.0,
            },
            metadata={
                'source': 'mcp_atlas_claim_judge',
                'pass_threshold': self.pass_threshold,
                'claims': claim_results,
                'total_claims': total,
                'fully_covered_claims': sum(1 for result in claim_results if result['score'] == 1.0),
                'partially_covered_claims': sum(1 for result in claim_results if result['score'] == 0.5),
                'model': self.llm_judge.model_id,
            },
            main_score_name='coverage_score',
        )
        return score

    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        if not sample_scores:
            return []
        coverage_values = [float(sample_score.score.value.get('coverage_score', 0.0)) for sample_score in sample_scores]
        pass_values = [float(sample_score.score.value.get('pass', 0.0)) for sample_score in sample_scores]
        sample_ids = [sample_score.sample_id for sample_score in sample_scores]
        return [
            AggScore(
                metric_name='coverage_score',
                score=sum(coverage_values) / len(coverage_values),
                aggregation_name='mean',
                num=len(coverage_values),
                ids=sample_ids,
            ),
            AggScore(
                metric_name='pass_rate',
                score=sum(pass_values) / len(pass_values),
                aggregation_name='mean',
                num=len(pass_values),
                ids=sample_ids,
                metadata={'pass_threshold': self.pass_threshold},
            ),
        ]

    def _preflight(self) -> None:
        try:
            self._enabled_servers = self.client.enabled_servers()
            self._tool_infos_by_name = {
                tool.name: tool
                for tool in (mcp_tool_to_tool_info(raw_tool) for raw_tool in self.client.list_tools())
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
        self._excluded_tasks.append({
            'task_id': sample.metadata.get('task_id'),
            'missing_servers': missing,
        })
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

    def _judge_claim(self, claim: str, response: str) -> Dict[str, Any]:
        prompt = claim_judge_prompt(claim, response)
        judge_response = self.llm_judge.judge(prompt)
        outcome, justification, confidence = parse_claim_judge_response(judge_response)
        score = {
            'fulfilled': 1.0,
            'partially_fulfilled': 0.5,
            'not_fulfilled': 0.0,
        }.get(outcome, 0.0)
        return {
            'claim': claim,
            'coverage_outcome': outcome,
            'score': score,
            'justification': justification,
            'confidence': confidence,
            'raw_judge_response': judge_response,
        }
