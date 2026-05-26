"""Internal string constants for the AgentLoop implementation.

Centralizes magic strings that flow through ``trace`` event payloads,
``AgentContext.metadata`` and the nudge prompt.  These values are part
of the loop's externally observable contract — strategies (e.g.
``swe_bench_*``) write ``metadata['submission_source'] = 'sentinel'``
and tests assert ``payload['message'] == 'max_steps_exceeded'`` — so
each constant's ``value`` must remain byte-for-byte identical to the
literal it replaces.

Kept module-private (not re-exported from ``__init__.py``) to avoid
expanding the public API surface.
"""


class ToolSchemaModes:
    """Mirrors :data:`evalscope.api.agent.types.ToolSchemaMode` literals."""

    FUNCTION_CALLING = 'function_calling'
    TEXTUAL_BLOCK = 'textual_block'
    NONE = 'none'


class MetadataKeys:
    """Well-known keys written into :attr:`AgentContext.metadata`."""

    SUBMISSION_SOURCE = 'submission_source'


class SubmissionSources:
    """Possible values of ``metadata['submission_source']`` / SUBMIT payload.

    Strategies (e.g. ``swe_bench_backticks``) may write ``SENTINEL`` to
    indicate the completion was detected inside a tool observation.  The
    loop falls back to ``POST_TOOL`` when the strategy did not annotate
    the source.  ``IMPLICIT_NO_NUDGE`` is emitted by the loop itself when
    a strategy returns no tool calls and declines a nudge.
    """

    SENTINEL = 'sentinel'
    POST_TOOL = 'post_tool'
    IMPLICIT_NO_NUDGE = 'implicit_no_nudge'


class LoopMessages:
    """Stable ``payload['message']`` strings asserted by tests."""

    MAX_STEPS_EXCEEDED = 'max_steps_exceeded'
    MODEL_CONTEXT_OVERFLOW = 'model_context_overflow'
    NO_TOOL_CALL_REMINDER = 'no_tool_call_reminder'


class TraceSources:
    """Stable ``payload['source']`` tags emitted by the loop."""

    PARSE = 'parse'
    NUDGE = 'nudge'
    LOOP = 'loop'


NUDGE_PROMPT = ('No tool was called. Please use an available tool '
                'or call the submit tool with your final answer.')
"""User-facing reminder injected when the model produces no tool call."""

__all__ = [
    'LoopMessages',
    'MetadataKeys',
    'NUDGE_PROMPT',
    'SubmissionSources',
    'ToolSchemaModes',
    'TraceSources',
]
