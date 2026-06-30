import asyncio
import csv
import json
import pytest
from pathlib import Path
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.metric import SampleScore, Score
from evalscope.api.registry import get_benchmark
from evalscope.api.tool import ToolCall, ToolFunction
from evalscope.benchmarks.mcp_atlas.mcp_atlas_adapter import MCPAtlasAdapter
from evalscope.benchmarks.mcp_atlas.utils import (
    MCPAtlasClient,
    MCPAtlasServerUnavailable,
    extract_claims,
    extract_required_servers,
    is_transport_error,
    mcp_tool_to_tool_info,
    parse_claim_judge_response,
    parse_enabled_tools,
)
from evalscope.config import TaskConfig
from evalscope.constants import HubType


class FakeClient:

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def enabled_servers(self) -> List[str]:
        return ['wikipedia']

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                'name': 'wikipedia_get_article',
                'description': 'Fetch a Wikipedia article.',
                'inputSchema': {
                    'type': 'object',
                    'properties': {
                        'title': {
                            'type': 'string',
                            'description': 'Article title.',
                        }
                    },
                    'required': ['title'],
                },
            },
            {
                'name': 'github_search_repositories',
                'description': 'Search repositories.',
                'inputSchema': {
                    'type': 'object',
                    'properties': {},
                },
            },
        ]

    def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        self.calls.append({'tool_name': tool_name, 'tool_args': tool_args})
        return 'tool result'


class FakeJudge:
    model_id = 'judge-model'

    def __init__(self, responses: List[str]) -> None:
        self.responses = responses
        self.prompts: List[str] = []

    def judge(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


def make_adapter(limit: Any = None, local_path: str = '', **extra_params: Any) -> MCPAtlasAdapter:
    base_extra_params = {
        'mcp_server_url': 'http://localhost:1984',
        'filter_enabled_servers': True,
        'max_steps': 100,
        'max_tool_calls': 100,
        'request_timeout': 60.0,
        'list_tools_timeout': 180.0,
        'use_system_prompt': False,
        'pass_threshold': 0.75,
    }
    base_extra_params.update(extra_params)
    meta = BenchmarkMeta(
        name='mcp_atlas',
        dataset_id='ScaleAI/MCP-Atlas',
        subset_list=['default'],
        default_subset='default',
        eval_split='train',
        prompt_template='{question}',
        metric_list=['coverage_score', 'pass'],
        extra_params=base_extra_params,
    )
    dataset_args = {'extra_params': extra_params}
    if local_path:
        dataset_args['local_path'] = local_path
        meta._update({'local_path': local_path})
    cfg = TaskConfig(
        datasets=['mcp_atlas'],
        dataset_args={'mcp_atlas': dataset_args},
        dataset_hub=HubType.MODELSCOPE,
        limit=limit,
    )
    return MCPAtlasAdapter(benchmark_meta=meta, task_config=cfg)


def write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['TASK', 'PROMPT', 'ENABLED_TOOLS', 'TRAJECTORY', 'GTFA_CLAIMS'])
        writer.writeheader()
        writer.writerows(rows)


def test_mcp_atlas_registered_under_short_name() -> None:
    cfg = TaskConfig(datasets=['mcp_atlas'])

    adapter = get_benchmark('mcp_atlas', cfg)

    assert isinstance(adapter, MCPAtlasAdapter)
    assert adapter.name == 'mcp_atlas'


def test_parse_enabled_tools_accepts_strings_and_objects() -> None:
    tools = parse_enabled_tools(
        json.dumps([
            'wikipedia_get_article',
            {
                'name': 'github_search_repositories'
            },
            'wikipedia_get_article',
            123,
        ])
    )

    assert tools == ['wikipedia_get_article', 'github_search_repositories']


def test_extract_required_servers_uses_trajectory_tool_calls() -> None:
    trajectory = json.dumps([{
        'role': 'assistant',
        'tool_calls': [
            {
                'function': {
                    'name': 'wikipedia_get_article'
                }
            },
            {
                'function': {
                    'name': 'MongoDB_find'
                }
            },
            {
                'function': {
                    'name': 'brave_brave_web_search'
                }
            },
        ],
    }])

    assert extract_required_servers(trajectory) == ['brave-search', 'mongodb', 'wikipedia']


def test_record_to_sample_sets_metadata_and_claims() -> None:
    adapter = make_adapter()
    adapter._client = FakeClient()
    sample = adapter.record_to_sample({
        'TASK': 'task-1',
        'PROMPT': 'Find the answer.',
        'ENABLED_TOOLS': json.dumps(['wikipedia_get_article']),
        'TRAJECTORY': json.dumps([]),
        'GTFA_CLAIMS': json.dumps([{
            'claim': 'The answer cites the page.'
        }]),
    })

    assert sample.input == 'Find the answer.'
    assert extract_claims(sample.target) == ['The answer cites the page.']
    assert sample.metadata['task_id'] == 'task-1'
    assert sample.metadata['enabled_tools'] == ['wikipedia_get_article']
    assert sample.tools[0].name == 'wikipedia_get_article'


def test_extract_claims_accepts_python_list_literal() -> None:
    claims = extract_claims("['claim one', 'claim two']")

    assert claims == ['claim one', 'claim two']


def test_extract_claims_flattens_cached_single_literal_list() -> None:
    claims = extract_claims(json.dumps(["['claim one', 'claim two']"]))

    assert claims == ['claim one', 'claim two']


def test_tool_info_conversion_accepts_input_schema() -> None:
    tool_info = mcp_tool_to_tool_info({
        'name': 'wikipedia_get_article',
        'description': 'Fetch a Wikipedia article.',
        'inputSchema': {
            'type': 'object',
            'properties': {
                'title': {
                    'type': 'string',
                    'description': 'Article title.',
                }
            },
            'required': ['title'],
        },
    })

    assert tool_info.name == 'wikipedia_get_article'
    assert tool_info.parameters.required == ['title']
    assert tool_info.parameters.properties['title'].type == 'string'


def test_client_uses_mcp_atlas_http_endpoints(monkeypatch: Any) -> None:
    requests_seen: List[Dict[str, Any]] = []

    class Response:

        def __init__(self, status_code: int, payload: Any) -> None:
            self.status_code = status_code
            self.payload = payload
            self.text = json.dumps(payload)

        def raise_for_status(self) -> None:
            pass

        def json(self) -> Any:
            return self.payload

    def fake_get(url: str, timeout: float) -> Response:
        requests_seen.append({'method': 'GET', 'url': url, 'timeout': timeout})
        return Response(200, {'enabled_servers': ['wikipedia']})

    def fake_post(url: str, **kwargs: Any) -> Response:
        requests_seen.append({'method': 'POST', 'url': url, **kwargs})
        if url.endswith('/list-tools'):
            return Response(200, [{'name': 'wikipedia_get_article'}])
        if url.endswith('/call-tool'):
            return Response(200, [{'type': 'text', 'text': 'article text'}])
        raise AssertionError(f'Unexpected URL: {url}')

    monkeypatch.setattr('requests.get', fake_get)
    monkeypatch.setattr('requests.post', fake_post)

    client = MCPAtlasClient('http://localhost:1984/', request_timeout=7.0, list_tools_timeout=11.0)

    assert client.enabled_servers() == ['wikipedia']
    assert client.list_tools() == [{'name': 'wikipedia_get_article'}]
    assert client.call_tool('wikipedia_get_article', {'title': 'MCP'}) == 'article text'
    assert requests_seen[0] == {
        'method': 'GET',
        'url': 'http://localhost:1984/enabled-servers',
        'timeout': 11.0,
    }
    assert requests_seen[1]['url'] == 'http://localhost:1984/list-tools'
    assert requests_seen[2]['json'] == {
        'tool_name': 'wikipedia_get_article',
        'tool_args': {
            'title': 'MCP'
        },
    }


def test_client_accepts_servers_dict_response(monkeypatch: Any) -> None:

    class Response:

        def raise_for_status(self) -> None:
            pass

        def json(self) -> Dict[str, Any]:
            return {
                'servers': {
                    'wikipedia': 'OK',
                    'github': 'ERROR_NOT_ONLINE',
                }
            }

    def fake_get(url: str, timeout: float) -> Response:
        return Response()

    monkeypatch.setattr('requests.get', fake_get)
    client = MCPAtlasClient('http://localhost:1984', request_timeout=7.0, list_tools_timeout=11.0)

    assert client.enabled_servers() == ['wikipedia']


def test_client_marks_transport_500_as_unavailable(monkeypatch: Any) -> None:

    class Response:
        status_code = 500
        text = '{"detail":"connect ECONNREFUSED 199.193.116.105:443"}'

        def json(self) -> Any:
            return {}

    def fake_post(url: str, **kwargs: Any) -> Response:
        return Response()

    monkeypatch.setattr('requests.post', fake_post)
    client = MCPAtlasClient('http://localhost:1984', request_timeout=7.0, list_tools_timeout=11.0)

    with pytest.raises(MCPAtlasServerUnavailable) as exc_info:
        client.call_tool('open-library_get_book_by_title', {'title': 'The Sins of the Wolf'})

    assert exc_info.value.server_name == 'open-library'
    assert is_transport_error(str(exc_info.value))


def test_load_dataset_filters_missing_servers_and_attaches_tools(tmp_path: Path) -> None:
    dataset_path = tmp_path / 'mcp_atlas.csv'
    write_rows(
        dataset_path, [
            {
                'TASK': 'keep',
                'PROMPT': 'Use Wikipedia.',
                'ENABLED_TOOLS': json.dumps(['wikipedia_get_article']),
                'TRAJECTORY': json.dumps([{
                    'tool_calls': [{
                        'function': {
                            'name': 'wikipedia_get_article'
                        }
                    }]
                }]),
                'GTFA_CLAIMS': json.dumps(['Uses the Wikipedia result.']),
            },
            {
                'TASK': 'drop',
                'PROMPT': 'Use GitHub.',
                'ENABLED_TOOLS': json.dumps(['github_search_repositories']),
                'TRAJECTORY': json.dumps([{
                    'tool_calls': [{
                        'function': {
                            'name': 'github_search_repositories'
                        }
                    }]
                }]),
                'GTFA_CLAIMS': json.dumps(['Uses the GitHub result.']),
            },
        ]
    )
    adapter = make_adapter(local_path=str(tmp_path))
    adapter._client = FakeClient()

    dataset = adapter.load_dataset()['default']

    assert len(dataset) == 1
    assert dataset[0].metadata['task_id'] == 'keep'
    assert dataset[0].tools[0].name == 'wikipedia_get_article'
    assert adapter._excluded_tasks == [{'task_id': 'drop', 'missing_servers': ['github']}]


def test_build_tools_enforces_tool_call_limit() -> None:
    adapter = make_adapter(max_tool_calls=1)
    fake_client = FakeClient()
    adapter._client = fake_client
    sample = Sample(
        input='prompt',
        id=3,
        metadata={'enabled_tools': ['wikipedia_get_article']},
    )
    handler = adapter.build_tools(sample)['wikipedia_get_article']
    call = ToolCall(
        id='call-1',
        function=ToolFunction(name='wikipedia_get_article', arguments={'title': 'MCP'}),
    )

    first = asyncio.run(handler(call, None))
    second = asyncio.run(handler(call, None))

    assert first == 'tool result'
    assert second == 'MCP-Atlas tool call limit exceeded (1).'
    assert fake_client.calls == [{'tool_name': 'wikipedia_get_article', 'tool_args': {'title': 'MCP'}}]


def test_build_tools_short_circuits_failed_server() -> None:

    class FailingClient(FakeClient):

        def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
            self.calls.append({'tool_name': tool_name, 'tool_args': tool_args})
            raise MCPAtlasServerUnavailable(tool_name, 'connect ECONNREFUSED 199.193.116.105:443')

    adapter = make_adapter()
    fake_client = FailingClient()
    adapter._client = fake_client
    sample = Sample(
        input='prompt',
        id=4,
        metadata={'enabled_tools': ['open-library_get_book_by_title', 'open-library_get_authors_by_name']},
    )
    handlers = adapter.build_tools(sample)
    first_call = ToolCall(
        id='call-1',
        function=ToolFunction(name='open-library_get_book_by_title', arguments={'title': 'The Sins of the Wolf'}),
    )
    second_call = ToolCall(
        id='call-2',
        function=ToolFunction(name='open-library_get_authors_by_name', arguments={'name': 'Anne Perry'}),
    )

    first = asyncio.run(handlers['open-library_get_book_by_title'](first_call, None))
    second = asyncio.run(handlers['open-library_get_authors_by_name'](second_call, None))

    assert "MCP server 'open-library' is unavailable" in first
    assert "MCP server 'open-library' is unavailable" in second
    assert fake_client.calls == [{
        'tool_name': 'open-library_get_book_by_title',
        'tool_args': {
            'title': 'The Sins of the Wolf'
        },
    }]


def test_llm_match_score_computes_claim_coverage() -> None:
    adapter = make_adapter(pass_threshold=0.75)
    adapter.llm_judge = FakeJudge([
        json.dumps({
            'coverage_outcome': 'fulfilled',
            'justification': 'covered',
            'confidence_level': 0.9,
        }),
        json.dumps({
            'coverage_outcome': 'partially_fulfilled',
            'justification': 'partly covered',
            'confidence_level': 0.6,
        }),
        json.dumps({
            'coverage_outcome': 'not_fulfilled',
            'justification': 'missing',
            'confidence_level': 0.8,
        }),
    ])
    sample = Sample(
        input='question',
        target=json.dumps(['claim one', 'claim two', 'claim three']),
        id=0,
    )
    state = TaskState(model='mock', sample=sample, completed=True)

    score = adapter.llm_match_score(
        original_prediction='answer',
        filtered_prediction='answer',
        reference=sample.target,
        task_state=state,
    )

    assert score.value['coverage_score'] == pytest.approx(0.5)
    assert score.value['pass'] == 0.0
    assert score.metadata['fully_covered_claims'] == 1
    assert score.metadata['partially_covered_claims'] == 1


def test_parse_claim_judge_response_accepts_string_confidence() -> None:
    outcome, justification, confidence = parse_claim_judge_response(
        json.dumps({
            'coverage_outcome': 'fulfilled',
            'justification': 'covered',
            'confidence_level': 'high',
        })
    )

    assert outcome == 'fulfilled'
    assert justification == 'covered'
    assert confidence == 1.0


def test_aggregate_scores_returns_mean_coverage_and_pass_rate() -> None:
    adapter = make_adapter()
    scores = adapter.aggregate_scores([
        SampleScore(sample_id=0, score=Score(value={
            'coverage_score': 1.0,
            'pass': 1.0
        })),
        SampleScore(sample_id=1, score=Score(value={
            'coverage_score': 0.5,
            'pass': 0.0
        })),
    ])

    assert [(score.metric_name, score.score) for score in scores] == [
        ('coverage_score', 0.75),
        ('pass_rate', 0.5),
    ]
