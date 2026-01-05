import gradio as gr
import os
from typing import AsyncGenerator, Tuple
from utils import convert_eval_args_to_config, convert_perf_args_to_config, submit_and_poll

DEFAULT_SERVICE_URL = os.getenv('EVALSCOPE_SERVICE_URL', 'http://127.0.0.1:9000')
VALID_EVAL_BENCHMARKS = ['gsm8k', 'mmlu', 'cmmlu', 'ceval', 'arc', 'math_500', 'aime24', 'aime25']


def create_eval_interface(service_url_input, poll_interval_input, common_model_name, common_api_url, common_api_key):
    """Create the content for the evaluation task interface"""
    with gr.Row():
        # --- Left Column: Configuration (Scale 2) ---
        with gr.Column(scale=2, variant='panel'):
            gr.Markdown('### 🛠️ 评估配置')

            with gr.Accordion('评估设置', open=True):
                eval_datasets = gr.Dropdown(
                    label='数据集',
                    choices=VALID_EVAL_BENCHMARKS,
                    value=['gsm8k'],
                    multiselect=True,
                    allow_custom_value=True,
                    info='选择或输入用于评估的数据集名称，支持多选或自定义输入（逗号分隔）'
                )
                eval_limit = gr.Number(label='限制数量', value=5, precision=0, info='每个数据集的任务数量限制，-1表示不限制')
                eval_batch_size = gr.Number(label='批大小', value=1, precision=0, info='每次模型请求的批处理大小')

            with gr.Accordion('更多参数', open=False):
                eval_repeats = gr.Number(label='重复次数', value=1, precision=0, info='重复运行评估的次数')
                eval_timeout = gr.Number(label='超时时间 (秒)', value=3600, info='任务运行的最大超时时间')
                eval_stream = gr.Checkbox(label='流式输出', value=True, info='是否以流式方式获取模型响应')

                gr.Markdown('#### 模型生成参数')
                eval_temp = gr.Slider(
                    label='Temperature (随机性)', minimum=0.0, maximum=1.0, value=0.0, info='生成文本的随机性，值越大越随机'
                )
                eval_top_p = gr.Slider(
                    label='Top P (核采样)', minimum=0.0, maximum=1.0, value=1.0, info='核采样参数，只考虑累积概率达到P的词'
                )
                eval_max_tokens = gr.Number(label='Max Tokens (最大生成)', value=1024, precision=0, info='模型生成文本的最大长度')
                eval_top_k = gr.Number(label='Top K (Top K 采样)', value=50, precision=0, info='Top K 采样参数，只考虑概率最高的K个词')

                dataset_args = gr.Code(label='数据集参数 (JSON)', language='json', value='{}', lines=2, max_lines=10)

            btn_eval = gr.Button('🚀 开始评估', variant='primary', size='lg')

        # --- Right Column: Logs and Progress (Scale 3) ---
        with gr.Column(scale=3):
            gr.Markdown('### 运行状态与日志')
            eval_progress_status = gr.Markdown('当前状态: 准备就绪', label='评估任务状态')
            eval_logs = gr.Code(
                label='控制台输出', language='shell', interactive=False, lines=30, elem_classes=['log-panel'], max_lines=30
            )

    # Logic handling function
    async def run_eval_wrapper(
        service_url,
        interval,
        model,  # Get from common model name
        api_url,  # Get from common API URL
        api_key,  # Get from common API Key
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
    ) -> AsyncGenerator[Tuple[str, str], None]:
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
        async for log_content, progress_status_text in submit_and_poll(service_url, 'eval', payload, interval):
            yield log_content, progress_status_text

    btn_eval.click(
        run_eval_wrapper,
        inputs=[
            service_url_input, poll_interval_input, common_model_name, common_api_url, common_api_key, eval_datasets,
            eval_limit, eval_batch_size, eval_repeats, eval_timeout, eval_stream, eval_temp, eval_top_p,
            eval_max_tokens, eval_top_k, dataset_args
        ],
        outputs=[eval_logs, eval_progress_status]
    )


def create_perf_interface(service_url_input, poll_interval_input, common_model_name, common_api_url, common_api_key):
    """Create the content for the performance testing interface"""
    with gr.Row():
        # --- Left Column: Configuration (Scale 2) ---
        with gr.Column(scale=2, variant='panel'):
            gr.Markdown('### ⚡ 性能测试配置')

            with gr.Accordion('压测设置', open=True):
                perf_api_type = gr.Dropdown(
                    label='API类型', choices=['openai'], value='openai', info='指定API接口类型，目前支持OpenAI兼容接口'
                )
                perf_parallel = gr.Textbox(
                    label='并发数 (逗号分隔)', value='1', placeholder='例如: 1,2,4', info='逗号分隔的并发用户数列表，可定义多个并发等级'
                )
                perf_number = gr.Textbox(
                    label='总请求数 (逗号分隔)', value='10', placeholder='例如: 10,20', info='逗号分隔的总请求数列表，对应每个并发等级的总请求数'
                )
                perf_rate = gr.Number(label='速率限制 (请求/秒)', value=-1, precision=0, info='-1 表示不限制每秒请求数')

            with gr.Accordion('模型生成参数', open=False):
                perf_max_tokens = gr.Number(label='Max Tokens (最大生成)', value=2048, precision=0, info='模型生成文本的最大长度')
                perf_min_tokens = gr.Number(label='Min Tokens (最小生成)', value=0, precision=0, info='模型生成文本的最小长度')

                perf_temp = gr.Slider(
                    label='Temperature (随机性)', minimum=0.0, maximum=1.0, value=0.0, info='生成文本的随机性，值越大越随机'
                )
                perf_top_p = gr.Slider(
                    label='Top P (核采样)', minimum=0.0, maximum=1.0, value=1.0, info='核采样参数，只考虑累积概率达到P的词'
                )

            with gr.Accordion('数据集设置', open=False):
                perf_dataset = gr.Dropdown(
                    label='测试数据集', choices=['openqa', 'line_by_line', 'random'], value='openqa', info='用于性能测试的数据集类型'
                )

                perf_max_prompt = gr.Number(
                    label='Max Prompt Len (最大Prompt长度)', value=1024, precision=0, info='生成Prompt的最大长度'
                )
                perf_min_prompt = gr.Number(
                    label='Min Prompt Len (最小Prompt长度)', value=0, precision=0, info='生成Prompt的最小长度'
                )

            btn_perf = gr.Button('⚡ 开始性能测试', variant='primary', size='lg')

        # --- Right Column: Logs and Progress (Scale 3) ---
        with gr.Column(scale=3):
            gr.Markdown('### 运行状态与日志')
            perf_progress_status = gr.Markdown('当前状态: 准备就绪', label='性能测试任务状态')
            perf_logs = gr.Code(
                label='控制台输出', language='shell', interactive=False, lines=30, elem_classes=['log-panel'], max_lines=30
            )

    # Logic handling function
    async def run_perf_wrapper(
        service_url,
        interval,
        model,  # Get from common model name
        url,  # Get from common API URL
        api_key,  # Get from common API Key
        api_type,
        parallel,
        number,
        rate,
        max_tokens,
        min_tokens,
        temp,
        top_p,
        dataset,
        max_pl,
        min_pl,
    ) -> AsyncGenerator[Tuple[str, str], None]:
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
            dataset=dataset,
            max_prompt_length=max_pl,
            min_prompt_length=min_pl
        )
        async for log_content, progress_status_text in submit_and_poll(service_url, 'perf', payload, interval):
            yield log_content, progress_status_text

    btn_perf.click(
        run_perf_wrapper,
        inputs=[
            service_url_input, poll_interval_input, common_model_name, common_api_url, common_api_key, perf_api_type,
            perf_parallel, perf_number, perf_rate, perf_max_tokens, perf_min_tokens, perf_temp, perf_top_p,
            perf_dataset, perf_max_prompt, perf_min_prompt
        ],
        outputs=[perf_logs, perf_progress_status]
    )


def create_interface():
    with gr.Blocks(title='EvalScope Dashboard', theme=gr.themes.Soft()) as demo:
        gr.Markdown('# 🚀 EvalScope 服务面板')

        # Global Service Settings (Top Bar)
        with gr.Accordion('全局设置', open=True):
            with gr.Row():
                service_url_input = gr.Textbox(
                    label='EvalScope 服务URL', value=DEFAULT_SERVICE_URL, scale=3, info='EvalScope后端服务的访问地址'
                )
                poll_interval_input = gr.Number(label='日志轮询间隔 (秒)', value=5, minimum=5, scale=1, info='获取任务日志的间隔时间')

            with gr.Row():
                common_model_name = gr.Textbox(
                    label='模型名称', value='qwen-plus', placeholder='例如: qwen-max, gpt-4', scale=1, info='用于评估或性能测试的模型名称'
                )
                common_api_url = gr.Textbox(
                    label='模型API URL',
                    value='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
                    scale=2,
                    info='模型API的请求地址'
                )
                common_api_key = gr.Textbox(
                    label='模型API Key',
                    value=os.getenv('DASHSCOPE_API_KEY', ''),
                    type='password',
                    scale=1,
                    info='访问模型API所需的密钥'
                )

        with gr.Tabs():
            with gr.TabItem('模型评估'):
                create_eval_interface(
                    service_url_input, poll_interval_input, common_model_name, common_api_url, common_api_key
                )

            with gr.TabItem('性能测试'):
                create_perf_interface(
                    service_url_input, poll_interval_input, common_model_name, common_api_url, common_api_key
                )

    return demo


if __name__ == '__main__':
    demo = create_interface()
    demo.queue().launch()
