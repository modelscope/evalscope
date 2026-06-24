"""Gemini generateContent SSE synthesizer.

Gemini's streaming format (``streamGenerateContent``) uses standard SSE
with ``data:`` lines containing JSON chunks. Each chunk is a complete
``generateContent`` response with partial content.

Unlike the Anthropic / OpenAI protocols, Gemini streaming sends the
full response object in a single ``data:`` frame (the server sends
multiple frames as content is generated). Since our bridge resolves
the full ModelOutput before streaming, we emit a single-chunk stream
that looks like a completed generation — sufficient for Gemini CLI
which accepts this pattern.

**Important:** Gemini CLI's bundled ``@google/genai`` SDK uses a custom
streaming parser (``extractNextJSONFromText``) that is **NOT** a
standards-compliant SSE parser.  It scans the raw response body for
JSON objects delimited by balanced braces.  Any non-JSON text (such as
SSE comments ``: keep-alive``) left in the parser's buffer when the
stream ends triggers ``"Incomplete JSON segment at the end"``.  For
this reason, we do **NOT** emit SSE keep-alive comments and instead
just wait silently for the model.

Frame sequence:
    data: {candidates: [{content: {parts: [...], role: 'model'}, finishReason: 'STOP'}], ...}
"""

import asyncio
import json
from typing import Any, AsyncIterator, Dict, Optional

from evalscope.api.model import ModelOutput
from .translate_gemini import model_output_to_gemini_response


def _sse_data(payload: Dict[str, Any]) -> bytes:
    """Encode one Gemini SSE data frame."""
    return f'data: {json.dumps(payload, ensure_ascii=False)}\n\n'.encode('utf-8')


async def stream_gemini_response(
    generate_task: 'asyncio.Future[ModelOutput]',
    *,
    request_model: Optional[str] = None,
) -> AsyncIterator[bytes]:
    """Yield SSE bytes for the Gemini streaming format.

    Gemini CLI expects ``text/event-stream`` with ``data:`` lines.
    We wait silently for the model to resolve, then emit the full
    response as a single data frame.

    We do NOT send SSE keep-alive comments because Gemini CLI's custom
    stream parser treats them as residual non-JSON text.
    """
    output: ModelOutput = await generate_task

    if output.error:
        error_resp: Dict[str, Any] = {
            'error': {
                'code': 500,
                'message': output.error,
                'status': 'INTERNAL',
            }
        }
        yield _sse_data(error_resp)
        return

    # Emit the full response as a single data frame
    response = model_output_to_gemini_response(output, request_model=request_model)
    yield _sse_data(response)
