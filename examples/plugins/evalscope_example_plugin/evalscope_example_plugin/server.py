"""A toy HTTP endpoint speaking a non-OpenAI "arith" protocol, used by the example plugin.

Request:  ``POST /v1/arith`` with ``{"model": str, "prompt": str, "max_tokens": int}``
Response: ``{"id": str, "text": str, "usage": {"input_tokens": int, "output_tokens": int}}``
and an ``X-Request-Id`` header.

Run it with ``python -m evalscope_example_plugin.server --port 8901``.
"""

import argparse
import json
import re
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_EXPRESSION = re.compile(r'(-?\d+)\s*([-+*])\s*(-?\d+)')


def answer(prompt: str) -> str:
    """Answer the last ``a <op> b`` expression in the prompt."""
    matches = _EXPRESSION.findall(prompt)
    if not matches:
        return 'I can only answer arithmetic questions.'
    left, operator, right = matches[-1]
    a, b = int(left), int(right)
    result = {'+': a + b, '-': a - b, '*': a * b}[operator]
    return f'The answer is {result}.'


class ArithHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip('/') != '/v1/arith':
            self._send(404, {'error': f'unknown path {self.path}'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            body = json.loads(self.rfile.read(length) or b'{}')
            prompt = body['prompt']
        except (ValueError, KeyError) as e:
            self._send(400, {'error': f'invalid request: {e}'})
            return
        text = answer(prompt)
        self._send(
            200,
            {
                'id': uuid.uuid4().hex,
                'text': text,
                'usage': {'input_tokens': len(prompt.split()), 'output_tokens': len(text.split())},
            },
        )

    def _send(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('X-Request-Id', payload.get('id', ''))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description='Toy "arith" endpoint for the EvalScope example plugin.')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8901)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), ArithHandler)
    print(f'Serving the arith protocol on http://{args.host}:{args.port}/v1/arith')
    server.serve_forever()


if __name__ == '__main__':
    main()
