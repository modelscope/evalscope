"""Launch the native CLI with literal credentials supplied outside process arguments."""

import json
import os
import sys
from typing import List


def main(argv: List[str]) -> None:
    """Inject credentials after CLI parsing without changing native scenario resolution."""
    from agentx_cli import app

    command, bound, _ = app.parse_args(argv)
    config = bound.arguments['cli_config']
    api_key = os.environ.get('EVALSCOPE_AGENTX_API_KEY')
    if api_key:
        config.api_key = api_key
    headers = json.loads(os.environ.get('EVALSCOPE_AGENTX_HEADERS', '{}'))
    if headers:
        config.headers = list(headers.items())
    command(*bound.args, **bound.kwargs)


if __name__ == '__main__':
    main(sys.argv[1:])
