from __future__ import annotations

import ast
import csv
import json
import re
import requests
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from evalscope.api.tool import ToolInfo, ToolParams
from evalscope.utils.json_schema import JSONSchema


class MCPAtlasServerUnavailable(Exception):
    """Transport-level failure from a backing MCP server."""

    def __init__(self, tool_name: str, message: str) -> None:
        self.tool_name = tool_name
        self.server_name = tool_name_to_server(tool_name)
        self.message = message
        super().__init__(message)


class MCPAtlasClient:
    """Small HTTP client for the MCP-Atlas agent-environment service."""

    def __init__(self, base_url: str, request_timeout: float, list_tools_timeout: float) -> None:
        self.base_url = base_url.rstrip('/')
        self.request_timeout = request_timeout
        self.list_tools_timeout = list_tools_timeout

    def enabled_servers(self) -> List[str]:
        response = requests.get(f'{self.base_url}/enabled-servers', timeout=self.list_tools_timeout)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and 'servers' in data:
            return [str(name) for name, status in data['servers'] if status == 'OK']
        if isinstance(data, dict):
            return [str(name) for name in data.get('enabled_servers', [])]
        raise ValueError(f'Unexpected /enabled-servers response: {type(data).__name__}')

    def list_tools(self) -> List[Dict[str, Any]]:
        response = requests.post(
            f'{self.base_url}/list-tools',
            headers={'Content-Type': 'application/json'},
            timeout=self.list_tools_timeout
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            raise ValueError(f'Unexpected /list-tools response: {type(data).__name__}')
        return [tool for tool in data if isinstance(tool, dict)]

    def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        try:
            response = requests.post(
                f'{self.base_url}/call-tool',
                json={
                    'tool_name': tool_name,
                    'tool_args': tool_args,
                },
                headers={'Content-Type': 'application/json'},
                timeout=self.request_timeout,
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            raise MCPAtlasServerUnavailable(tool_name, str(exc)) from exc
        if response.status_code != 200:
            if is_transport_error(response.text):
                raise MCPAtlasServerUnavailable(tool_name, response.text)
            return response.text
        return format_tool_response(response.json())


def load_local_records(path: str) -> List[Dict[str, Any]]:
    local_path = Path(path)
    if local_path.is_dir():
        csv_files = sorted(local_path.glob('*.csv'))
        if not csv_files:
            raise FileNotFoundError(f'No CSV file found in MCP-Atlas local path: {path}')
        local_path = csv_files[0]
    if local_path.suffix == '.jsonl':
        records = []
        with local_path.open(encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records
    csv.field_size_limit(sys.maxsize)
    with local_path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def parse_enabled_tools(value: Any) -> List[str]:
    parsed = maybe_parse_json(value, default=[])
    if not isinstance(parsed, list):
        return []
    seen = set()
    tools = []
    for item in parsed:
        if isinstance(item, dict):
            name = item.get('name')
        elif isinstance(item, str):
            name = item
        else:
            name = None
        if name and name not in seen:
            seen.add(name)
            tools.append(str(name))
    return tools


def extract_claims(value: Any) -> List[str]:
    parsed = maybe_parse_json(value, default=value)
    if isinstance(parsed, list):
        claims = []
        for item in parsed:
            if isinstance(item, dict) and item.get('claim'):
                claims.append(str(item['claim']).strip())
            elif isinstance(item, str):
                nested = maybe_parse_json(item, default=None)
                if isinstance(nested, list):
                    claims.extend(extract_claims(nested))
                else:
                    claims.append(item.strip())
            elif item is not None:
                claims.append(str(item).strip())
        return [claim for claim in claims if claim]
    if not isinstance(parsed, str):
        return []
    text = parsed.strip()
    if not text:
        return []
    if '\n' in text:
        return [clean_claim_text(line) for line in text.splitlines() if clean_claim_text(line)]
    return [text]


def clean_claim_text(text: str) -> str:
    cleaned = re.sub(r'^\s*(?:[-*]|\d+[.)])\s*', '', text).strip()
    return cleaned.strip('"\'')


def extract_required_servers(trajectory: Any) -> List[str]:
    parsed = maybe_parse_json(trajectory, default=[])
    if not isinstance(parsed, list):
        return []
    servers = set()
    for message in parsed:
        if not isinstance(message, dict):
            continue
        for tool_call in message.get('tool_calls') or []:
            if not isinstance(tool_call, dict):
                continue
            function_info = tool_call.get('function') or {}
            if isinstance(function_info, dict) and function_info.get('name'):
                servers.add(tool_name_to_server(str(function_info['name'])))
    return sorted(servers)


def mcp_tool_to_tool_info(raw_tool: Dict[str, Any]) -> ToolInfo:
    schema = raw_tool.get('inputSchema') or raw_tool.get('input_schema') or {}
    if not isinstance(schema, dict):
        schema = {}
    properties = {}
    for key, value in (schema.get('properties') or {}).items():
        if isinstance(value, dict):
            try:
                properties[key] = JSONSchema.model_validate(value)
            except Exception:
                properties[key] = JSONSchema(type='string')
    return ToolInfo(
        name=str(raw_tool['name']),
        description=str(raw_tool.get('description') or raw_tool['name']),
        parameters=ToolParams(
            properties=properties,
            required=list(schema.get('required') or []),
            additionalProperties=bool(schema.get('additionalProperties', False)),
        ),
    )


def parse_claim_judge_response(response: Any) -> Tuple[str, str, float]:
    if not isinstance(response, str):
        return 'not_fulfilled', 'Judge response was not text.', 0.0
    text = strip_json_fence(response)
    try:
        parsed = json.loads(text)
        outcome = str(parsed.get('coverage_outcome') or 'not_fulfilled')
        justification = str(parsed.get('justification') or '')
        confidence = parse_confidence(parsed.get('confidence_level', 0.0))
        return outcome, justification, confidence
    except Exception:
        lowered = text.lower()
        for outcome in ['partially_fulfilled', 'not_fulfilled', 'fulfilled']:
            if outcome in lowered:
                return outcome, text, 0.0
    return 'not_fulfilled', text or 'Unable to parse judge response.', 0.0


def strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith('```'):
        return stripped
    lines = stripped.splitlines()[1:]
    if lines and lines[-1].strip() == '```':
        lines = lines[:-1]
    return '\n'.join(lines).strip()


def parse_confidence(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return 0.0
    lowered = value.strip().lower()
    if lowered in {'high', 'high confidence'}:
        return 1.0
    if lowered in {'medium', 'moderate', 'medium confidence', 'moderate confidence'}:
        return 0.5
    if lowered in {'low', 'low confidence'}:
        return 0.0
    try:
        return float(lowered)
    except ValueError:
        return 0.0


def is_transport_error(text: str) -> bool:
    lowered = text.lower()
    markers = [
        'econnrefused',
        'etimedout',
        'enotfound',
        'eai_again',
        'connection refused',
        'connection timed out',
        'connect timeout',
        'read timeout',
        'timed out',
        'failed to establish a new connection',
    ]
    return any(marker in lowered for marker in markers)


def server_unavailable_message(server_name: str, message: str) -> str:
    return (
        f"MCP server '{server_name}' is unavailable for this sample due to a transport error. "
        f'Skipping further calls to this server in the same sample. Error: {message}'
    )


def claim_judge_prompt(claim: str, response: str) -> str:
    return f"""You are evaluating how well a model's response addresses a specific expert-defined claim.
Treat the claim as the authoritative reference. Do not use external knowledge to dispute or correct the claim.
Judge only whether the model response explicitly states, entails, or omits the claim.
Return JSON with keys: claim_text, coverage_outcome, justification, confidence_level.
coverage_outcome must be one of: fulfilled, partially_fulfilled, not_fulfilled.

CLAIM TO EVALUATE:
{claim}

MODEL RESPONSE TO ANALYZE:
{response}"""


def format_tool_response(value: Any) -> str:
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and item.get('type') == 'text':
                parts.append(str(item.get('text', '')))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return '\n\n'.join(part for part in parts if part)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def maybe_parse_json(value: Any, default: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return default


def field(record: Dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def tool_name_to_server(tool_name: str) -> str:
    mappings = {
        'brave_brave_web_search': 'brave-search',
        'MongoDB_aggregate': 'mongodb',
        'MongoDB_collection-schema': 'mongodb',
        'MongoDB_count': 'mongodb',
        'MongoDB_find': 'mongodb',
        'MongoDB_list-collections': 'mongodb',
        'MongoDB_list-databases': 'mongodb',
    }
    return mappings.get(tool_name, tool_name).split('_')[0]
