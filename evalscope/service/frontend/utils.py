import asyncio
import json
from async_client import AsyncEvalClient
from dataclasses import fields
from typing import AsyncGenerator, Optional, Tuple

from evalscope.config import TaskConfig
from evalscope.perf.arguments import Arguments as PerfArguments

MIN_POLL_INTERVAL_SECONDS = 5
MAX_POLL_INTERVAL_SECONDS = 3600  # 1 hour upper bound to prevent unreasonably long sleeps


def convert_eval_args_to_config(**kwargs) -> dict:
    """Helper function to convert UI arguments to evaluation task configuration dictionary."""
    gen_config = {}
    gen_keys = ['temperature', 'max_tokens', 'top_p', 'top_k']
    for key in gen_keys:
        if key in kwargs and kwargs[key] is not None:
            val = kwargs.pop(key)
            if key in ['max_tokens', 'top_k']:
                val = int(val)
            gen_config[key] = val

    if 'datasets' in kwargs:
        if isinstance(kwargs['datasets'], str):
            kwargs['datasets'] = [d.strip() for d in kwargs['datasets'].split(',') if d.strip()]

    if 'dataset_args' in kwargs and isinstance(kwargs['dataset_args'], str):
        try:
            kwargs['dataset_args'] = json.loads(kwargs['dataset_args'])
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in 'dataset_args': {e}")

    if 'limit' in kwargs and kwargs['limit'] is not None:
        kwargs['limit'] = int(kwargs['limit'])
        if kwargs['limit'] <= 0:
            kwargs['limit'] = None  # Treat non-positive limit as no limit

    for key in ['eval_batch_size', 'repeats']:
        if key in kwargs and kwargs[key] is not None:
            kwargs[key] = int(kwargs[key])

    valid_keys = {f.name for f in fields(TaskConfig)}
    payload = {k: v for k, v in kwargs.items() if k in valid_keys}
    payload['generation_config'] = gen_config
    return payload


def convert_perf_args_to_config(**kwargs) -> dict:
    """Helper function to convert UI arguments to performance testing task configuration dictionary."""
    int_fields = [
        'rate', 'max_tokens', 'min_tokens', 'max_prompt_length', 'min_prompt_length', 'top_k', 'connect_timeout',
        'read_timeout'
    ]
    for key in int_fields:
        if key in kwargs and kwargs[key] is not None:
            kwargs[key] = int(kwargs[key])

    for key in ['parallel', 'number']:
        if key in kwargs and kwargs[key] is not None:
            val = kwargs[key]
            if isinstance(val, str):
                kwargs[key] = [int(x.strip()) for x in val.split(',') if x.strip()]
            elif isinstance(val, (int, float)):
                kwargs[key] = [int(val)]

    valid_keys = {f.name for f in fields(PerfArguments)}
    payload = {k: v for k, v in kwargs.items() if k in valid_keys}
    return payload


async def submit_and_poll(
    service_url: str,
    task_type: str,
    payload: dict,
    poll_interval: int,
) -> AsyncGenerator[Tuple[str, str], None]:  # Returns (log content, progress status text)
    """
    Generic function to submit a task and poll for logs, returning log content and progress status text independently.
    """

    # Clamp poll_interval to a reasonable range to avoid excessively frequent or infrequent polling.

    if poll_interval < MIN_POLL_INTERVAL_SECONDS:
        poll_interval = MIN_POLL_INTERVAL_SECONDS
    elif poll_interval > MAX_POLL_INTERVAL_SECONDS:
        poll_interval = MAX_POLL_INTERVAL_SECONDS
    logs = []
    current_progress_status = '当前状态: 准备就绪'

    # Initial return, set initial status
    yield ''.join(logs), current_progress_status

    try:
        async with AsyncEvalClient(service_url) as client:
            # 1. Submit task
            msg = f'Submitting {task_type} task to {service_url}...\n'
            logs.append(msg)
            current_progress_status = '🚀 提交任务中...'
            yield ''.join(logs), current_progress_status

            try:
                if task_type == 'eval':
                    resp = await client.submit_eval_task(payload)
                else:
                    resp = await client.submit_perf_task(payload)
            except Exception as e:
                logs.append(f'❌ 提交任务失败: {str(e)}\n')
                current_progress_status = f'❌ 任务提交失败: {str(e)}'
                yield ''.join(logs), current_progress_status
                return

            request_id = resp.get('request_id')
            logs.append(f'✅ 任务提交成功。请求ID: {request_id}\n')
            logs.append('等待日志输出...\n')
            current_progress_status = f'✅ 任务提交成功，请求ID: {request_id}。正在等待日志...'
            yield ''.join(logs), current_progress_status

            # 2. Poll logs
            current_line = 0
            finish_marker = '*** [EvalScope Service] Task finished at'
            while True:

                for i in range(poll_interval):
                    remaining = poll_interval - i
                    current_progress_status = f'⏳ 正在等待日志更新 ({remaining:.1f}秒后再次查询)'
                    yield ''.join(logs), current_progress_status
                    await asyncio.sleep(1)

                current_progress_status = '🔄 正在获取新日志...'
                yield ''.join(logs), current_progress_status  # Update status before fetching

                # Fetch logs
                try:
                    new_content = await client.get_task_log(request_id, current_line, task_type)
                except Exception as fetch_err:
                    logs.append(f'\n ⚠️ 获取日志失败: {fetch_err}')
                    current_progress_status = f'⚠️ 获取日志失败: {fetch_err}'
                    yield ''.join(logs), current_progress_status
                    # Continue polling even if log fetch fails
                    continue

                if new_content:
                    logs.append(new_content)
                    current_line += new_content.count('\n')
                    # Update once with new log content
                    yield ''.join(logs), current_progress_status

                    if finish_marker in new_content:
                        logs.append('\n✅ 任务完成。')
                        current_progress_status = '✅ 任务已完成'
                        yield ''.join(logs), current_progress_status
                        break
                else:
                    # Ensure logs and status update even if no new content (e.g., after progress animation)
                    yield ''.join(logs), current_progress_status

    except Exception as e:
        logs.append(f'\n❌ 发生错误: {str(e)}')
        current_progress_status = f'❌ 发生错误: {str(e)}'
        yield ''.join(logs), current_progress_status
