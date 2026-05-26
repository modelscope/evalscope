"""VendorVerifierAdapter: shared base for vendor verifier benchmarks.

Vendor verifier benchmarks (k2_verifier, minimax_verifier, kimi_verifier) test
whether a third-party API deployment faithfully reproduces an official model's
behavior. They share:

- a thin OpenAI-compatible inference loop (``_on_inference``),
- JSON-schema validation of tool calls (``validate_tool_call``),
- an always-on detector for "reasoning-only" deployment regressions
  (``detect_error_only_reasoning``).

The base intentionally provides no metric aggregation logic; each vendor
adapter computes its own ``aggregate_scores`` because the metric names and
formulas are vendor-specific.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from evalscope.api.dataset import Sample
from evalscope.api.model.model import Model, ModelOutput
from evalscope.api.tool import ToolCall
from evalscope.utils.logger import get_logger
from .agent_adapter import AgentAdapter

logger = get_logger()


class VendorVerifierAdapter(AgentAdapter):
    """Shared base for vendor verifier benchmarks.

    Subclasses must still declare their own ``BenchmarkMeta`` via
    ``@register_benchmark`` and override ``record_to_sample`` /
    ``match_score`` / ``aggregate_scores``. The base only supplies inference
    and validation helpers.
    """

    def _on_inference(self, model: Model, sample: Sample) -> ModelOutput:
        """Default OpenAI-compatible chat completion with tool support.

        Errors are captured into the ``ModelOutput.error`` field so the
        adapter's ``match_score`` can decide how to score failures.
        """
        try:
            return model.generate(input=sample.input, tools=sample.tools)
        except Exception as e:
            logger.error(f'Error during model inference: {e}')
            return ModelOutput.from_content(
                model=model.name,
                content='',
                stop_reason='unknown',
                error=str(e),
            )

    @staticmethod
    def validate_tool_call(tool_calls: List[ToolCall], tools: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Validate each tool call's arguments against its declared JSON schema.

        Returns ``(passed, reason)`` where ``passed`` is True only if all
        provided tool calls have a matching schema and their arguments
        validate. ``reason`` is empty on success or a short human-readable
        description of the first failure.
        """
        from jsonschema import ValidationError, validate

        tool_name = ''
        try:
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                schema = next(
                    (t['function']['parameters'] for t in tools if t['function']['name'] == tool_name),
                    None,
                )
                if not schema:
                    return False, f"No schema found for tool '{tool_name}'"

                args = tool_call.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError as e:
                        return False, f"JSON parse failed for tool '{tool_name}' arguments: {e}"

                validate(instance=args, schema=schema)

        except ValidationError as e:
            return False, f"Schema validation failed for tool '{tool_name}': {e.message}"
        except KeyError as e:
            return False, f'Tool call format error, missing field: {e}'
        except Exception as e:
            return False, f'Unexpected error during validation: {e}'
        return True, ''

    @staticmethod
    def detect_error_only_reasoning(model_output: ModelOutput) -> bool:
        """Detect the "reasoning-only" deployment regression.

        A model is in this failure mode when it emits a chain-of-thought
        ``reasoning`` block but no final ``content`` and no ``tool_calls``.
        This typically indicates a vendor parsing or stop-token bug rather
        than a model capability issue. Patterned after MiniMax-Provider-
        Verifier's same-named check.
        """
        try:
            message = model_output.message
            content = message.text or ''
            tool_calls = message.tool_calls or []
            has_reasoning = False
            if isinstance(message.content, list):
                has_reasoning = any(getattr(c, 'type', None) == 'reasoning' for c in message.content)
            return has_reasoning and not content and not tool_calls
        except Exception:
            return False
