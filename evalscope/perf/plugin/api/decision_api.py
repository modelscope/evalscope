"""Non-stream decision protocols using the standard performance scheduler and HTTP timing."""

import json
from typing import Any

from evalscope.api.model.decision import DecisionInput, parse_decision_response, parse_decision_usage
from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.default_api import DefaultApiPlugin
from evalscope.perf.plugin.registry import register_api


@register_api('systemone')
class SystemOnePerfPlugin(DefaultApiPlugin):
    """Replay complete single-question System One JSON bodies."""

    protocol = 'systemone'

    def build_request(self, messages: Any, param: Arguments | None = None) -> dict:
        """Validate a complete request without adding text-generation defaults."""
        if not isinstance(messages, dict):
            raise ValueError('Decision perf requires complete JSON objects via --dataset line_by_line.')
        body = dict(messages)
        forbidden = {'stream', 'temperature', 'top_p', 'top_k', 'max_tokens', 'n'} & body.keys()
        if forbidden:
            raise ValueError(f'Decisions do not accept generation controls: {sorted(forbidden)}')
        if self.protocol == 'systemone':
            body.setdefault('model', (param or self.param).model)
        self._request_contract(body)
        return body

    def _request_contract(self, body: dict) -> DecisionInput:
        if self.protocol == 'systemone':
            if set(body.get('questions', {})) != {'answer'}:
                raise ValueError('Decision perf currently requires one question named answer.')
            question = body['questions']['answer']
            return DecisionInput(
                kind=question['type'],
                state=body['state'],
                instructions=question['instructions'],
                criteria=question.get('criteria'),
            )
        kind = body['kind']
        criteria = (
            {str(i): option for i, option in enumerate(body['options'])}
            if kind == 'choice'
            else [str(i) for i in range(6)]
            if kind == 'score'
            else None
        )
        if body.get('thinking', 'off') != 'off':
            raise ValueError('This benchmark measures System One only; set thinking=off.')
        return DecisionInput(kind=kind, state=body['state'], instructions=body['question'], criteria=criteria)

    def parse_responses(self, responses: list, request: str = None, **kwargs: Any) -> tuple[int, int]:
        """Read usage using either provider's token naming convention."""
        raw = responses[-1]
        if isinstance(raw, str):
            raw = json.loads(raw)
        return parse_decision_usage(raw)

    async def process_request(self, client_session: Any, url: str, headers: dict, body: dict) -> Any:
        """Count a request as successful only after validating its decision distribution."""
        output = await super().process_request(client_session, url, headers, body)
        if output.success:
            try:
                if output.is_stream or len(output.response_messages) != 1:
                    raise ValueError('Expected one non-stream JSON decision response.')
                parse_decision_response(output.response_messages[0], self._request_contract(body), self.protocol)
                output.prompt_tokens, output.completion_tokens = self.parse_responses(output.response_messages)
            except (ValueError, TypeError, KeyError) as exc:
                output.success = False
                output.error = f'Invalid decision response: {exc}'
        if not output.success:
            output.response_messages.append({'evalscope_error': output.error, 'http_status': output.status_code})
        output.first_chunk_latency = 0.0
        output.time_per_output_token = 0.0
        return output

    def set_request_max_tokens(self, request: dict, max_tokens: int) -> None:
        """Reject token-cap workloads instead of injecting unsupported fields."""
        raise ValueError('Decision requests have no output-token cap.')


@register_api('autotrust')
class AutoTrustPerfPlugin(SystemOnePerfPlugin):
    """Replay native AutoTrust /decide requests without hidden generation defaults."""

    protocol = 'autotrust'
