import aiohttp
import asyncio
import json
from dotenv import dotenv_values
from typing import Any, Dict, Optional

env = dotenv_values('.env')


class AsyncEvalClient:

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def submit_eval_task(self, payload: Dict[str, Any]) -> Dict[str, str]:
        """
        Submit asynchronous evaluation task.
        """
        return await self.submit_task('eval', payload)

    async def submit_perf_task(self, payload: Dict[str, Any]) -> Dict[str, str]:
        """
        Submit asynchronous performance task.
        """
        return await self.submit_task('perf', payload)

    async def submit_task(self, task_type: str, payload: Dict[str, Any]) -> Dict[str, str]:
        """
        Submit asynchronous task (eval or perf).

        Returns:
            Dictionary containing request_id and other headers.
        """
        url = f'{self.base_url}/api/v1/{task_type}'
        headers = {'Content-Type': 'application/json', 'X-Fc-Invocation-Type': 'Async'}

        print(f'[Submit Task] Sending request to: {url}')
        print(f'[Submit Task] Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}')

        async with self.session.post(url, json=payload, headers=headers) as response:
            print(f'[Submit Task] Status Code: {response.status}')

            # Get all response headers
            response_headers = dict(response.headers)
            print('[Submit Task] Response Headers:')
            for key, value in response_headers.items():
                print(f'  {key}: {value}')

            if response.status == 202:
                request_id = response_headers.get('X-Fc-Request-Id')
                task_id = response_headers.get('X-Fc-Async-Task-Id')

                print('[Submit Task] Task submitted successfully!')
                print(f'[Submit Task] Request ID: {request_id}')
                print(f'[Submit Task] Task ID: {task_id}')

                return {'request_id': request_id, 'task_id': task_id, 'headers': response_headers}
            elif response.status == 200:
                # Handle synchronous response (e.g. local server)
                resp_json = await response.json()
                request_id = resp_json.get('request_id')
                print('[Submit Task] Task completed synchronously.')
                print(f'[Submit Task] Request ID: {request_id}')
                return {'request_id': request_id, 'headers': response_headers, 'status': 'finished'}
            else:
                error_text = await response.text()
                raise Exception(f'Task submission failed: {response.status}, {error_text}')

    async def get_task_log(self, request_id: str, start_line: int = 0, task_type: str = 'eval') -> str:
        """
        Get task logs.
        """
        url = f'{self.base_url}/api/v1/{task_type}/log'
        params = {'request_id': request_id, 'start_line': start_line}

        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.text()
        except Exception:
            pass
        return ''

    async def poll_task_result(
        self, request_id: str, max_attempts: int = 60, interval: float = 5.0, task_type: str = 'eval'
    ) -> None:
        """
        Poll task logs until completion.

        Args:
            request_id: Task ID
            max_attempts: Maximum polling attempts
            interval: Polling interval (seconds)
            task_type: Task type (eval/perf)
        """
        print(f'\n[Poll Logs] Start polling task: {request_id}')
        print(f'[Poll Logs] Max attempts: {max_attempts}, Interval: {interval}s')

        current_log_line = 0
        finish_marker = '*** [EvalScope Service] Task finished at'

        for attempt in range(1, max_attempts + 1):
            # 1. Get and print new logs
            new_logs = await self.get_task_log(request_id, current_log_line, task_type)
            if new_logs:
                print(new_logs, end='')
                current_log_line += new_logs.count('\n')

                # 2. Check for finish marker
                if finish_marker in new_logs:
                    print('\n[Poll Logs] ✓ Task finished!')
                    return

            if attempt < max_attempts:
                await asyncio.sleep(interval)

        raise TimeoutError(f'Polling timeout: Attempted {max_attempts} times')


async def main():
    # Configuration
    base_url = 'https://evalsco-service-cwbmixobdj.cn-hangzhou.fcapp.run'

    # Request payload
    payload = {
        'model': 'qwen-plus',
        'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'api_key': env.get('DASHSCOPE_API_KEY', ''),
        'datasets': ['gsm8k'],
        'limit': 5,
        'generation_config': {
            'temperature': 0.0,
            'max_tokens': 512
        },
    }

    # Use client
    async with AsyncEvalClient(base_url) as client:
        # 1. Submit task
        task_info = await client.submit_eval_task(payload)
        request_id = task_info['request_id']

        # 2. Poll logs
        try:
            await client.poll_task_result(
                request_id=request_id,
                max_attempts=60,  # Max 60 attempts
                interval=5  # Poll every 10 seconds
            )

        except TimeoutError as e:
            print(f'\nError: {e}')
        except Exception as e:
            print(f'\nError occurred: {str(e)}')


if __name__ == '__main__':
    # Run async main function
    asyncio.run(main())
