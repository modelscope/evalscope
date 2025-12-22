import asyncio
import gradio as gr
import json
import os
from async_client import AsyncEvalClient
from dataclasses import fields
from typing import AsyncGenerator, Tuple

from evalscope.config import TaskConfig
from evalscope.perf.arguments import Arguments as PerfArguments

# Default configuration
DEFAULT_SERVICE_URL = os.getenv('EVALSCOPE_SERVICE_URL', 'http://127.0.0.1:9000')
VALID_EVAL_BENCHMARKS = ['gsm8k', 'mmlu', 'cmmlu', 'ceval', 'arc', 'math_500', 'aime24', 'aime25']


def convert_eval_args_to_config(**kwargs) -> dict:
    """Helper to convert UI arguments to eval task configuration dicts."""
    # Handle generation_config nested fields
    gen_config = {}
    gen_keys = ['temperature', 'max_tokens', 'top_p', 'top_k']
    for key in gen_keys:
        if key in kwargs and kwargs[key] is not None:
            val = kwargs.pop(key)
            if key in ['max_tokens', 'top_k']:
                val = int(val)
            gen_config[key] = val

    # Handle datasets list
    if 'datasets' in kwargs:
        if isinstance(kwargs['datasets'], str):
            kwargs['datasets'] = [d.strip() for d in kwargs['datasets'].split(',') if d.strip()]
        # If list (from Dropdown), keep as is

    # Handle dataset_args
    if 'dataset_args' in kwargs and isinstance(kwargs['dataset_args'], str):
        try:
            kwargs['dataset_args'] = json.loads(kwargs['dataset_args'])
        except Exception:
            kwargs['dataset_args'] = {}

    # Handle limit
    if 'limit' in kwargs and kwargs['limit'] is not None:
        kwargs['limit'] = int(kwargs['limit'])

    # Handle other int fields
    for key in ['eval_batch_size', 'repeats']:
        if key in kwargs and kwargs[key] is not None:
            kwargs[key] = int(kwargs[key])

    # Filter keys based on TaskConfig
    valid_keys = {f.name for f in fields(TaskConfig)}
    payload = {k: v for k, v in kwargs.items() if k in valid_keys}

    # Add generation_config back
    payload['generation_config'] = gen_config

    return payload


def convert_perf_args_to_config(**kwargs) -> dict:
    """Helper to convert UI arguments to perf task configuration dicts."""
    # Handle numeric conversions
    int_fields = [
        'rate', 'max_tokens', 'min_tokens', 'max_prompt_length', 'min_prompt_length', 'top_k', 'connect_timeout',
        'read_timeout'
    ]
    for key in int_fields:
        if key in kwargs and kwargs[key] is not None:
            kwargs[key] = int(kwargs[key])

    # Handle list[int] fields
    for key in ['parallel', 'number']:
        if key in kwargs and kwargs[key] is not None:
            val = kwargs[key]
            if isinstance(val, str):
                kwargs[key] = [int(x.strip()) for x in val.split(',') if x.strip()]
            elif isinstance(val, (int, float)):
                kwargs[key] = [int(val)]

    # Filter keys based on PerfArguments
    valid_keys = {f.name for f in fields(PerfArguments)}
    payload = {k: v for k, v in kwargs.items() if k in valid_keys}

    return payload


async def submit_and_poll(service_url: str, task_type: str, payload: dict,
                          poll_interval: int) -> AsyncGenerator[Tuple[str, str], None]:
    """Generic function to submit task and poll logs."""

    if poll_interval < 5:
        poll_interval = 5

    logs = []

    try:
        async with AsyncEvalClient(service_url) as client:
            # 1. Submit Task
            msg = f'Submitting {task_type} task to {service_url}...\n'
            logs.append(msg)
            yield ''.join(logs), '🚀 Submitting task...'

            try:
                if task_type == 'eval':
                    resp = await client.submit_eval_task(payload)
                else:
                    resp = await client.submit_perf_task(payload)
            except Exception as e:
                logs.append(f'Error submitting task: {str(e)}\n')
                yield ''.join(logs), f'❌ Error: {str(e)}'
                return

            request_id = resp.get('request_id')
            logs.append(f'Task submitted successfully. Request ID: {request_id}\nWaiting for logs...\n')
            yield ''.join(logs), f'✅ Task submitted: {request_id}'

            # 2. Poll Logs
            current_line = 0
            finish_marker = '*** [EvalScope Service] Task finished at'

            while True:
                # Countdown
                for i in range(poll_interval, 0, -1):
                    yield ''.join(logs), f'⏳ Refresh in {i}s...'
                    await asyncio.sleep(1)

                yield ''.join(logs), '🔄 Fetching logs...'

                new_content = await client.get_task_log(request_id, current_line, task_type)
                if new_content:
                    logs.append(new_content)
                    current_line += new_content.count('\n')

                    if finish_marker in new_content:
                        logs.append('\nTask Completed.')
                        yield ''.join(logs), '🏁 Task Completed.'
                        break

                    yield ''.join(logs), f'✅ Updated (Lines: {current_line})'
                else:
                    # Yield existing logs to keep connection alive/UI updated
                    yield ''.join(logs), '💤 No new logs'

    except Exception as e:
        logs.append(f'\nAn error occurred: {str(e)}')
        yield ''.join(logs), f'❌ Exception: {str(e)}'


def create_eval_tab(service_url_input, poll_interval_input, global_model, global_api_url, global_api_key):
    """
    Eval Tab: Uses global model/url/key inputs.
    """
    with gr.Tab('模型评估 (Evaluation)'):
        # Specific configurations for Eval
        with gr.Row():
            eval_datasets = gr.Dropdown(
                label='数据集',
                choices=VALID_EVAL_BENCHMARKS,
                value=['gsm8k'],
                multiselect=True,
                allow_custom_value=True,
                info='选择评估数据集'
            )
            eval_limit = gr.Number(label='限制数量', value=5, info='评估样本数量限制')

        with gr.Accordion('高级配置', open=False):
            with gr.Row():
                eval_batch_size = gr.Number(label='批处理大小', value=1, precision=0, info='评估时的批处理大小')
                eval_repeats = gr.Number(label='重复次数', value=1, precision=0, info='数据集重复次数')
                eval_timeout = gr.Number(label='超时时间(秒)', value=3600, info='请求超时时间')
                eval_stream = gr.Checkbox(label='流式输出', value=True, info='是否使用流式输出')
            dataset_args = gr.Code(label='数据集参数 (JSON)', language='json', value='{}', lines=3)

        with gr.Accordion('生成配置', open=False):
            with gr.Row():
                eval_temp = gr.Slider(label='温度', minimum=0.0, maximum=1.0, value=0.0, info='采样温度')
                eval_max_tokens = gr.Number(label='最大生成长度', value=1024, precision=0, info='最大生成token数')
                eval_top_p = gr.Slider(label='Top P', minimum=0.0, maximum=1.0, value=1.0, info='核采样概率')
                eval_top_k = gr.Number(label='Top K', value=50, precision=0, info='Top K采样')

        btn_eval = gr.Button('开始评估', variant='primary')
        eval_status = gr.Markdown('Ready')
        eval_logs = gr.Code(label='日志', language='shell', interactive=False, lines=40, max_lines=40)

        async def run_eval_wrapper(
            url, model, datasets, limit, ds_args, api_url, api_key, batch_size, repeats, timeout, stream, temp,
            max_tokens, top_p, top_k, interval
        ):
            payload = convert_eval_args_to_config(
                model=model,
                datasets=datasets,
                dataset_args=ds_args,
                limit=limit,
                api_url=api_url,
                api_key=api_key,
                eval_batch_size=batch_size,
                repeats=repeats,
                timeout=timeout,
                stream=stream,
                temperature=temp,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k
            )
            async for log, status in submit_and_poll(url, 'eval', payload, interval):
                yield log, status

        btn_eval.click(
            run_eval_wrapper,
            inputs=[
                service_url_input,
                global_model,  # Shared Input
                eval_datasets,
                eval_limit,
                dataset_args,
                global_api_url,  # Shared Input
                global_api_key,  # Shared Input
                eval_batch_size,
                eval_repeats,
                eval_timeout,
                eval_stream,
                eval_temp,
                eval_max_tokens,
                eval_top_p,
                eval_top_k,
                poll_interval_input
            ],
            outputs=[eval_logs, eval_status]
        )


def create_perf_tab(service_url_input, poll_interval_input, global_model, global_api_url, global_api_key):
    """
    Perf Tab: Uses global model/url/key inputs.
    """
    with gr.Tab('性能测试 (Performance Test)'):
        # Specific configurations for Perf
        with gr.Row():
            with gr.Column():
                perf_api = gr.Dropdown(label='API类型', choices=['openai'], value='openai', info='API接口类型')
            with gr.Column():
                perf_parallel = gr.Textbox(label='并发数', value='1', info='并发请求数量 (支持列表，逗号分隔)')
                perf_number = gr.Textbox(label='总请求数', value='10', info='总共发送的请求数量 (支持列表，逗号分隔)')

        with gr.Accordion('请求配置', open=False):
            with gr.Row():
                perf_rate = gr.Number(label='速率限制(req/s)', value=-1, precision=0, info='每秒请求数限制')
                perf_max_tokens = gr.Number(label='最大生成长度', value=2048, precision=0, info='最大生成token数')
                perf_min_tokens = gr.Number(label='最小生成长度', value=0, precision=0, info='最小生成token数')
            with gr.Row():
                perf_temp = gr.Slider(label='温度', minimum=0.0, maximum=1.0, value=0.0, info='采样温度')
                perf_top_p = gr.Slider(label='Top P', minimum=0.0, maximum=1.0, value=1.0, info='核采样概率')
                perf_top_k = gr.Number(label='Top K', value=None, precision=0, info='Top K采样')
            with gr.Row():
                perf_freq_penalty = gr.Number(label='频率惩罚', value=0.0, info='频率惩罚系数')
                perf_rep_penalty = gr.Number(label='重复惩罚', value=0.0, info='重复惩罚系数')

        with gr.Accordion('数据集与提示词', open=False):
            with gr.Row():
                perf_dataset = gr.Dropdown(
                    label='数据集', choices=['openqa', 'line_by_line', 'random'], value='openqa', info='测试使用的数据集'
                )
                perf_max_prompt_len = gr.Number(label='最大提示词长度', value=1024, precision=0, info='最大输入提示词长度')
                perf_min_prompt_len = gr.Number(label='最小提示词长度', value=0, precision=0, info='最小输入提示词长度')

        btn_perf = gr.Button('开始性能测试', variant='primary')
        perf_status = gr.Markdown('Ready')
        perf_logs = gr.Code(label='日志', language='shell', interactive=False, lines=40, max_lines=40)

        async def run_perf_wrapper(
            service_url, model, url, api, api_key, parallel, number, rate, max_tokens, min_tokens, temp, top_p, top_k,
            freq_penalty, rep_penalty, dataset, max_prompt_len, min_prompt_len, interval
        ):
            payload = convert_perf_args_to_config(
                model=model,
                url=url,  # Mapped from global_api_url
                api=api,
                api_key=api_key,
                parallel=parallel,
                number=number,
                rate=rate,
                max_tokens=max_tokens,
                min_tokens=min_tokens,
                temperature=temp,
                top_p=top_p,
                top_k=top_k,
                frequency_penalty=freq_penalty,
                repetition_penalty=rep_penalty,
                dataset=dataset,
                max_prompt_length=max_prompt_len,
                min_prompt_length=min_prompt_len,
            )
            async for log, status in submit_and_poll(service_url, 'perf', payload, interval):
                yield log, status

        btn_perf.click(
            run_perf_wrapper,
            inputs=[
                service_url_input,
                global_model,  # Shared Input
                global_api_url,  # Shared Input (Passed as 'url' to payload)
                perf_api,
                global_api_key,  # Shared Input
                perf_parallel,
                perf_number,
                perf_rate,
                perf_max_tokens,
                perf_min_tokens,
                perf_temp,
                perf_top_p,
                perf_top_k,
                perf_freq_penalty,
                perf_rep_penalty,
                perf_dataset,
                perf_max_prompt_len,
                perf_min_prompt_len,
                poll_interval_input
            ],
            outputs=[perf_logs, perf_status]
        )


def create_interface():
    with gr.Blocks(title='EvalScope Service Dashboard', theme=gr.themes.Soft()) as demo:
        gr.Markdown('# 🚀 EvalScope Service Dashboard')

        # === 1. EvalScope Service Configuration ===
        with gr.Row():
            service_url_input = gr.Textbox(
                label='服务地址 (EvalScope Service)', value=DEFAULT_SERVICE_URL, info='EvalScope服务的地址'
            )
            poll_interval_input = gr.Number(label='日志轮询间隔(秒)', value=5, minimum=5, info='日志轮询间隔时间')

        # === 2. Global Model Configuration (Shared by Eval & Perf) ===
        gr.Markdown('### 🌍 通用模型配置 (Common Model Config)')
        with gr.Row(variant='panel'):
            global_model = gr.Textbox(label='测试模型名称', value='qwen-plus', info='被评估/测试的模型名称 (model_id)')
            global_api_url = gr.Textbox(
                label='API URL',
                value='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
                info='OpenAI兼容接口地址'
            )
            global_api_key = gr.Textbox(
                label='API Key', value=os.getenv('DASHSCOPE_API_KEY', ''), type='password', info='模型服务的API密钥'
            )

        # === 3. Task Tabs ===
        with gr.Tabs():
            # Pass the global inputs down to the tabs
            create_eval_tab(service_url_input, poll_interval_input, global_model, global_api_url, global_api_key)
            create_perf_tab(service_url_input, poll_interval_input, global_model, global_api_url, global_api_key)

    return demo


if __name__ == '__main__':
    demo = create_interface()
    demo.queue().launch()
