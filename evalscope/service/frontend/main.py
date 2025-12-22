import asyncio
import gradio as gr
import json
import os
from async_client import AsyncEvalClient
from dataclasses import fields
from typing import AsyncGenerator, Optional, Tuple

from evalscope.config import TaskConfig
from evalscope.perf.arguments import Arguments as PerfArguments

# Default configuration
DEFAULT_SERVICE_URL = os.getenv('EVALSCOPE_SERVICE_URL', 'http://127.0.0.1:9000')
VALID_EVAL_BENCHMARKS = ['gsm8k', 'mmlu', 'cmmlu', 'ceval', 'arc', 'math_500', 'aime24', 'aime25']


def convert_eval_args_to_config(**kwargs) -> dict:
    """Helper to convert UI arguments to eval task configuration dicts."""
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
        except Exception:
            kwargs['dataset_args'] = {}

    if 'limit' in kwargs and kwargs['limit'] is not None:
        kwargs['limit'] = int(kwargs['limit'])

    for key in ['eval_batch_size', 'repeats']:
        if key in kwargs and kwargs[key] is not None:
            kwargs[key] = int(kwargs[key])

    valid_keys = {f.name for f in fields(TaskConfig)}
    payload = {k: v for k, v in kwargs.items() if k in valid_keys}
    payload['generation_config'] = gen_config
    return payload


def convert_perf_args_to_config(**kwargs) -> dict:
    """Helper to convert UI arguments to perf task configuration dicts."""
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
    progress: Optional[gr.Progress] = None
) -> AsyncGenerator[str, None]:
    """
    Generic function to submit task and poll logs.
    """

    if poll_interval < 5:
        poll_interval = 5  # Minimum 5 seconds interval

    logs = []

    if progress is not None:
        progress(0, desc='🚀 Submitting Task...')

    try:
        async with AsyncEvalClient(service_url) as client:
            # 1. Submit Task
            msg = f'Submitting {task_type} task to {service_url}...\n'
            logs.append(msg)
            yield ''.join(logs)

            try:
                if task_type == 'eval':
                    resp = await client.submit_eval_task(payload)
                else:
                    resp = await client.submit_perf_task(payload)
            except Exception as e:
                logs.append(f'❌ Error submitting task: {str(e)}\n')
                yield ''.join(logs)
                return

            request_id = resp.get('request_id')
            logs.append(f'✅ Task submitted successfully. Request ID: {request_id}\n')
            logs.append('Waiting for logs...\n')
            yield ''.join(logs)

            # 2. Poll Logs
            current_line = 0
            finish_marker = '*** [EvalScope Service] Task finished at'

            loop_count = 0
            while True:
                loop_count += 1

                # --- Progress Bar Animation ---
                if progress is not None:
                    steps = 20
                    step_time = poll_interval / steps
                    for i in range(steps):
                        pct = (i + 1) / steps
                        remaining = poll_interval - (i * step_time)
                        progress(pct, desc=f'⏳ Polling in {remaining:.1f}s (Cycle {loop_count})')
                        await asyncio.sleep(step_time)

                    progress(None, desc='🔄 Fetching new logs...')
                else:
                    await asyncio.sleep(poll_interval)

                # Fetch logs
                try:
                    new_content = await client.get_task_log(request_id, current_line, task_type)
                except Exception as fetch_err:
                    logs.append(f'\n[Warning] Fetch log failed: {fetch_err}')
                    yield ''.join(logs)
                    continue

                if new_content:
                    logs.append(new_content)
                    current_line += new_content.count('\n')
                    yield ''.join(logs)

                    if finish_marker in new_content:
                        logs.append('\n🏁 Task Completed.')
                        if progress is not None:
                            progress(1.0, desc='✅ Task Completed')
                        yield ''.join(logs)
                        break

                yield ''.join(logs)

    except Exception as e:
        logs.append(f'\n❌ An error occurred: {str(e)}')
        yield ''.join(logs)


def create_eval_interface(service_url_input, poll_interval_input):
    """Creates the content for the Evaluation Tab"""
    with gr.Row():
        # --- Left Column: Configuration (Scale 2) ---
        with gr.Column(scale=2, variant='panel'):
            gr.Markdown('### 🛠️ 评估配置 (Eval Config)')

            with gr.Group():
                gr.Markdown('#### 模型设置')
                eval_model = gr.Textbox(label='测试模型名称', value='qwen-plus', placeholder='e.g., qwen-max')
                eval_api_url = gr.Textbox(
                    label='API URL', value='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
                )
                eval_api_key = gr.Textbox(label='API Key', value=os.getenv('DASHSCOPE_API_KEY', ''), type='password')

            with gr.Group():
                gr.Markdown('#### 任务设置')
                eval_datasets = gr.Dropdown(
                    label='数据集',
                    choices=VALID_EVAL_BENCHMARKS,
                    value=['gsm8k'],
                    multiselect=True,
                    allow_custom_value=True
                )
                # 垂直堆叠
                eval_limit = gr.Number(label='限制数量', value=5, precision=0)
                eval_batch_size = gr.Number(label='批大小', value=1, precision=0)

            with gr.Accordion('更多参数', open=False):
                # 垂直堆叠
                eval_repeats = gr.Number(label='重复次数', value=1, precision=0)
                eval_timeout = gr.Number(label='超时时间(秒)', value=3600)
                eval_stream = gr.Checkbox(label='流式输出', value=True)

                # 垂直堆叠生成参数
                eval_temp = gr.Slider(label='Temperature', minimum=0.0, maximum=1.0, value=0.0)
                eval_top_p = gr.Slider(label='Top P', minimum=0.0, maximum=1.0, value=1.0)
                eval_max_tokens = gr.Number(label='Max Tokens', value=1024, precision=0)
                eval_top_k = gr.Number(label='Top K', value=50, precision=0)

                dataset_args = gr.Code(label='Dataset Args (JSON)', language='json', value='{}', lines=2, max_lines=10)

            btn_eval = gr.Button('🚀 开始评估 (Start Eval)', variant='primary', size='lg')

        # --- Right Column: Logs (Scale 3) ---
        with gr.Column(scale=3):
            gr.Markdown('### 📝 运行日志 (Logs)')
            eval_logs = gr.Code(
                label='Console Output', language='shell', interactive=False, lines=30, elem_classes=['log-panel']
            )

    # Logic
    async def run_eval_wrapper(
        url,
        interval,
        model,
        api_url,
        api_key,
        datasets,
        limit,
        batch_size,
        repeats,
        timeout,
        stream,
        temp,
        top_p,
        max_tokens,
        top_k,
        ds_args,
        progress=gr.Progress()
    ):
        payload = convert_eval_args_to_config(
            model=model,
            api_url=api_url,
            api_key=api_key,
            datasets=datasets,
            limit=limit,
            eval_batch_size=batch_size,
            repeats=repeats,
            timeout=timeout,
            stream=stream,
            temperature=temp,
            top_p=top_p,
            max_tokens=max_tokens,
            top_k=top_k,
            dataset_args=ds_args
        )
        async for log in submit_and_poll(url, 'eval', payload, interval, progress):
            yield log

    btn_eval.click(
        run_eval_wrapper,
        inputs=[
            service_url_input, poll_interval_input, eval_model, eval_api_url, eval_api_key, eval_datasets, eval_limit,
            eval_batch_size, eval_repeats, eval_timeout, eval_stream, eval_temp, eval_top_p, eval_max_tokens,
            eval_top_k, dataset_args
        ],
        outputs=[eval_logs]
    )


def create_perf_interface(service_url_input, poll_interval_input):
    """Creates the content for the Performance Tab"""
    with gr.Row():
        # --- Left Column: Configuration (Scale 2) ---
        with gr.Column(scale=2, variant='panel'):
            gr.Markdown('### ⚡ 性能测试配置 (Perf Config)')

            with gr.Group():
                gr.Markdown('#### 模型设置')
                perf_model = gr.Textbox(label='测试模型名称', value='qwen-plus')
                perf_api_url = gr.Textbox(
                    label='API URL',
                    value='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
                    lines=2
                )
                perf_api_key = gr.Textbox(label='API Key', value=os.getenv('DASHSCOPE_API_KEY', ''), type='password')
                perf_api_type = gr.Dropdown(label='API类型', choices=['openai'], value='openai')

            with gr.Group():
                gr.Markdown('#### 压测设置')
                perf_parallel = gr.Textbox(label='并发数', value='1', placeholder='e.g. 1,2,4')
                perf_number = gr.Textbox(label='总请求数', value='10', placeholder='e.g. 10,20')
                perf_rate = gr.Number(label='速率限制 (req/s)', value=-1, precision=0, info='-1 表示不限制')

                perf_max_tokens = gr.Number(label='Max Tokens', value=2048, precision=0)
                perf_min_tokens = gr.Number(label='Min Tokens', value=0, precision=0)

                perf_temp = gr.Slider(label='Temp', minimum=0.0, maximum=1.0, value=0.0)
                perf_top_p = gr.Slider(label='Top P', minimum=0.0, maximum=1.0, value=1.0)

                perf_freq_penalty = gr.Number(label='Freq Penalty', value=0.0)
                perf_rep_penalty = gr.Number(label='Rep Penalty', value=0.0)

                gr.Markdown('#### 数据集设置')
                perf_dataset = gr.Dropdown(label='测试数据集', choices=['openqa', 'line_by_line', 'random'], value='openqa')

                # 垂直堆叠
                perf_max_prompt = gr.Number(label='Max Prompt Len', value=1024, precision=0)
                perf_min_prompt = gr.Number(label='Min Prompt Len', value=0, precision=0)

            btn_perf = gr.Button('⚡ 开始性能测试 (Start Perf)', variant='primary', size='lg')

        # --- Right Column: Logs (Scale 3) ---
        with gr.Column(scale=3):
            gr.Markdown('### 📝 运行日志 (Logs)')
            perf_logs = gr.Code(
                label='Console Output', language='shell', interactive=False, lines=30, elem_classes=['log-panel']
            )

    # Logic
    async def run_perf_wrapper(
        service_url,
        interval,
        model,
        url,
        api_key,
        api_type,
        parallel,
        number,
        rate,
        max_tokens,
        min_tokens,
        temp,
        top_p,
        freq_p,
        rep_p,
        dataset,
        max_pl,
        min_pl,
        progress=gr.Progress()
    ):
        payload = convert_perf_args_to_config(
            model=model,
            url=url,
            api=api_type,
            api_key=api_key,
            parallel=parallel,
            number=number,
            rate=rate,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            temperature=temp,
            top_p=top_p,
            frequency_penalty=freq_p,
            repetition_penalty=rep_p,
            dataset=dataset,
            max_prompt_length=max_pl,
            min_prompt_length=min_pl
        )
        async for log in submit_and_poll(service_url, 'perf', payload, interval, progress):
            yield log

    btn_perf.click(
        run_perf_wrapper,
        inputs=[
            service_url_input, poll_interval_input, perf_model, perf_api_url, perf_api_key, perf_api_type,
            perf_parallel, perf_number, perf_rate, perf_max_tokens, perf_min_tokens, perf_temp, perf_top_p,
            perf_freq_penalty, perf_rep_penalty, perf_dataset, perf_max_prompt, perf_min_prompt
        ],
        outputs=[perf_logs]
    )


def create_interface():
    with gr.Blocks(title='EvalScope Dashboard', theme=gr.themes.Soft()) as demo:
        gr.Markdown('# 🚀 EvalScope Service Dashboard')

        # Global Service Settings (Top Bar)
        with gr.Row(variant='panel'):
            service_url_input = gr.Textbox(label='EvalScope Service URL', value=DEFAULT_SERVICE_URL, scale=3)
            poll_interval_input = gr.Number(label='Poll Interval (s)', value=5, minimum=2, scale=1)

        with gr.Tabs():
            with gr.TabItem('模型评估 (Evaluation)'):
                create_eval_interface(service_url_input, poll_interval_input)

            with gr.TabItem('性能测试 (Performance)'):
                create_perf_interface(service_url_input, poll_interval_input)

    return demo


if __name__ == '__main__':
    demo = create_interface()
    demo.queue().launch()
