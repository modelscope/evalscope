import argparse
import asyncio

from evalscope.benchmarks.toolathlon.client import run_ws_proxy


def main() -> None:
    parser = argparse.ArgumentParser(description='Toolathlon private-mode WebSocket relay.')
    parser.add_argument('--server-url', required=True)
    parser.add_argument('--llm-base-url', required=True)
    parser.add_argument('--llm-api-key', default='')
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    asyncio.run(run_ws_proxy(args.server_url, args.llm_base_url, args.llm_api_key, args.job_id))


if __name__ == '__main__':
    main()
