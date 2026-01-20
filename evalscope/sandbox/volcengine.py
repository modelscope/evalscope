# -*- coding: utf-8 -*-
"""
Volcengine Sandbox backend for EvalScope.

This backend targets the SandboxFusion HTTP service:
  - POST {base_url}/run_code  {"code": "...", "language": "python|bash|..."}
Docs:
  - https://bytedance.github.io/SandboxFusion/docs/docs/get-started/
"""

from __future__ import annotations

import json
import requests
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Union


class ExecutionStatus(str, Enum):
    SUCCESS = 'success'
    ERROR = 'error'
    TIMEOUT = 'timeout'


@dataclass
class ExecutionResult:
    """
    A lightweight result object to mimic typical sandbox tool execution results.

    Many EvalScope code-bench evaluators only need:
      - status
      - output (stderr/stdout)
      - tool_name
      - metadata
    """
    status: ExecutionStatus
    tool_name: str
    output: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def ok(self) -> bool:
        return self.status == ExecutionStatus.SUCCESS


class SandboxFusionClient:

    def __init__(
        self,
        base_url: str,
        *,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        verify_ssl: bool = True,
        run_code_path: str = '/run_code',
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        if not base_url:
            raise ValueError('SandboxFusionClient: base_url is required.')
        self.base_url = base_url.rstrip('/')
        self.run_code_path = run_code_path if run_code_path.startswith('/') else f'/{run_code_path}'
        self.timeout = float(timeout)
        self.verify_ssl = bool(verify_ssl)

        self.session = requests.Session()
        headers: Dict[str, str] = {'Content-Type': 'application/json'}
        if api_key:
            # allow either raw token or "Bearer xxx"
            headers['Authorization'] = api_key
        if extra_headers:
            headers.update(dict(extra_headers))
        self.session.headers.update(headers)

    def _post_json(self, path: str, payload: Dict[str, Any], *, timeout: Optional[float] = None) -> Dict[str, Any]:
        url = f'{self.base_url}{path}'
        request_timeout = self.timeout if timeout is None else float(timeout)
        resp = self.session.post(url, json=payload, timeout=request_timeout, verify=self.verify_ssl)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise ValueError(f'SandboxFusion response is not a JSON object: {data!r}')
        return data

    def run_code(self, *, code: str, language: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        # Keep payload minimal to avoid incompatibility with different SandboxFusion versions.
        payload = {'code': code, 'language': language}
        return self._post_json(self.run_code_path, payload, timeout=timeout)

    def execute_tool(
        self,
        *,
        tool_name: str,
        tool_input: Union[str, Dict[str, Any]],
        tool_language_override: Optional[str] = None,
        dataset_language_map: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        """
        Convert EvalScope-like tool invocation into SandboxFusion /run_code call.

        Supported mappings:
          - python_executor -> python
          - shell_executor  -> bash
          - multi_code_executor -> expects {"code": "...", "language": "..."} or JSON string

        dataset_language_override is the language override from the dataset configuration.
        e.g. {"r": "R", "d": "D_ut"} for volcengine sandbox
        """
        # 1) Normalize input to dict if possible
        input_dict: Optional[Dict[str, Any]] = None
        input_str: Optional[str] = None

        if isinstance(tool_input, dict):
            input_dict = tool_input
        else:
            input_str = str(tool_input)
            trimmed = input_str.strip()
            look_like_json = ((trimmed.startswith('{') and trimmed.endswith('}'))
                              or (trimmed.startswith('[') and trimmed.endswith(']')))
            if look_like_json:
                try:
                    parsed = json.loads(trimmed)
                    if isinstance(parsed, dict):
                        input_dict = parsed
                except Exception:
                    # keep as raw string
                    input_dict = None

        # 2) Decide language + code
        tool_name_norm = (tool_name or '').strip()

        default_tool_lang_map = {
            'python_executor': 'python',
            'shell_executor': 'bash',
        }

        language: Optional[str] = tool_language_override or default_tool_lang_map.get(tool_name_norm)

        code: Optional[str] = None

        if tool_name_norm in ('multi_code_executor', 'run_code'):
            # Expect dict: {"code": "...", "language": "..."}
            if input_dict and 'code' in input_dict and 'language' in input_dict:
                code = str(input_dict['code'])
                language = str(input_dict['language'])
            else:
                # fallback: treat as python
                code = input_str or ''
                language = language or 'python'
        else:
            # python_executor/shell_executor or unknown tool
            if input_dict and 'code' in input_dict:
                code = str(input_dict['code'])
            elif input_dict and 'command' in input_dict:
                code = str(input_dict['command'])
            elif input_dict and 'script' in input_dict:
                code = str(input_dict['script'])
            else:
                code = input_str or ''

            if not language:
                # safest default for EvalScope code benchmarks is python
                language = 'python'

        # 3) Call SandboxFusion
        try:
            if dataset_language_map:
                language = dataset_language_map.get(language, language)
            resp = self.run_code(code=code, language=language, timeout=timeout)
        except requests.Timeout as e:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                tool_name=tool_name_norm,
                output=f'SandboxFusion request timeout: {e}',
                metadata={'exception': repr(e)},
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                tool_name=tool_name_norm,
                output=f'SandboxFusion request failed: {e}',
                metadata={'exception': repr(e)},
            )

        # 4) Convert response to ExecutionResult
        status_field = resp.get('status')
        message = resp.get('message') or ''

        compile_result = resp.get('compile_result')
        run_result = resp.get('run_result') or {}

        stdout = (run_result.get('stdout') or '') if isinstance(run_result, dict) else ''
        stderr = (run_result.get('stderr') or '') if isinstance(run_result, dict) else ''
        return_code = run_result.get('return_code') if isinstance(run_result, dict) else None

        compile_stderr = ''
        if isinstance(compile_result, dict):
            compile_stderr = compile_result.get('stderr') or ''

        merged_output = ''
        if stdout:
            merged_output += stdout
        if stderr:
            merged_output += ('' if merged_output.endswith('\n') or not merged_output else '\n') + stderr
        if compile_stderr:
            merged_output += ('' if merged_output.endswith('\n') or not merged_output else '\n') + compile_stderr
        if message and message not in merged_output:
            merged_output += ('' if merged_output.endswith('\n') or not merged_output else '\n') + message

        ok = (status_field == 'Success') and (return_code in (0, '0', None))
        return ExecutionResult(
            status=ExecutionStatus.SUCCESS if ok else ExecutionStatus.ERROR,
            tool_name=tool_name_norm,
            output=merged_output,
            metadata={
                'sandbox_fusion': resp,
                'language': language
            },
        )


class SandboxFusionSandbox:
    """
    A lightweight sandbox object. Implements common tool invocation methods used by evaluators.
    """

    def __init__(
        self,
        client: SandboxFusionClient,
        *,
        tool_language_map: Optional[Dict[str, str]] = None,
        dataset_language_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.client = client
        self.tool_language_map = tool_language_map or {}
        self.dataset_language_map = dataset_language_map or {}

    def execute(
        self,
        tool_name: str,
        tool_input: Union[str, Dict[str, Any]],
        *,
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        lang_override = self.tool_language_map.get(tool_name)
        if lang_override is None:
            lang_override = self.tool_language_map.get(tool_name)
        return self.client.execute_tool(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_language_override=lang_override,
            dataset_language_map=self.dataset_language_map,
            timeout=timeout,
        )

    # aliases (to match different caller styles)
    run_tool = execute
    invoke_tool = execute

    def __call__(
        self,
        tool_name: str,
        tool_input: Union[str, Dict[str, Any]],
        *,
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        return self.execute(tool_name, tool_input, timeout=timeout)


class SandboxFusionSandboxManager:
    """
    Manager wrapper to look like a "sandbox manager" (even though SandboxFusion is a stateless HTTP service).
    """

    def __init__(
        self,
        sandbox_manager_config: Optional[Dict[str, Any]] = None,
        sandbox_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        sandbox_manager_config = sandbox_manager_config or {}
        sandbox_config = sandbox_config or {}

        base_url = (
            sandbox_manager_config.get('base_url') or sandbox_manager_config.get('endpoint')
            or sandbox_manager_config.get('url')
        )
        if not base_url:
            raise ValueError('volcengine sandbox requires sandbox_manager_config.base_url (SandboxFusion service URL).')

        api_key = sandbox_manager_config.get('api_key')
        timeout = sandbox_manager_config.get('timeout', 30.0)
        verify_ssl = sandbox_manager_config.get('verify_ssl', True)
        run_code_path = sandbox_manager_config.get('run_code_path', '/run_code')
        extra_headers = sandbox_manager_config.get('headers')

        self.client = SandboxFusionClient(
            base_url=str(base_url),
            api_key=api_key,
            timeout=float(timeout),
            verify_ssl=bool(verify_ssl),
            run_code_path=str(run_code_path),
            extra_headers=extra_headers,
        )

        # Tool language overrides from config (optional)
        tool_language_map: Dict[str, str] = {}
        # 1) direct override field
        if isinstance(sandbox_config.get('tool_language_map'), dict):
            tool_language_map.update({str(k): str(v) for k, v in sandbox_config['tool_language_map'].items()})
        # 2) optional nested tools_config mapping (non-breaking)
        #    e.g. {"tools_config": {"shell_executor": {"language": "bash"}}}
        if isinstance(sandbox_config.get('tools_config'), dict):
            for tool_name, cfg in sandbox_config['tools_config'].items():
                if isinstance(cfg, dict) and 'language' in cfg:
                    tool_language_map[str(tool_name)] = str(cfg['language'])
        # 3) The highest level is manual configuration
        if isinstance(sandbox_manager_config.get('tool_language_map'), dict):
            tool_language_map.update({str(k): str(v) for k, v in sandbox_manager_config['tool_language_map'].items()})

        # Language map from config (optional)
        dataset_language_map: Dict[str, str] = {}
        if isinstance(sandbox_manager_config.get('dataset_language_map'), dict):
            dataset_language_map.update({
                str(k): str(v)
                for k, v in sandbox_manager_config['dataset_language_map'].items()
            })

        self._sandbox = SandboxFusionSandbox(
            self.client, tool_language_map=tool_language_map, dataset_language_map=dataset_language_map
        )

    def create_sandbox(self, sandbox_config: Optional[Dict[str, Any]] = None) -> SandboxFusionSandbox:
        # SandboxFusion is stateless; return a shared sandbox instance.
        return self._sandbox

    def get_sandbox(self) -> SandboxFusionSandbox:
        return self._sandbox

    def close(self) -> None:
        # keep-alive session
        try:
            self.client.session.close()
        except Exception:
            pass

    # context manager support
    def __enter__(self) -> 'SandboxFusionSandboxManager':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
