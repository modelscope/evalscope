# 自定义使用

## 自定义结果分析
该工具在测试期间会将所有数据保存到 sqlite3 数据库中，包括请求和响应。您可以在测试后分析测试数据。

```python
import base64
import json
import pickle
import sqlite3

db_path = 'your db path'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 获取列名
cursor.execute('PRAGMA table_info(result)')
columns = [info[1] for info in cursor.fetchall()]
print('列名：', columns)

cursor.execute('SELECT * FROM result WHERE success=1 AND first_chunk_latency > 1')
rows = cursor.fetchall()
print(f'len(rows): {len(rows)}')

for row in rows:
    row_dict = dict(zip(columns, row))
    # 解码request
    row_dict['request'] = pickle.loads(base64.b64decode(row_dict['request']))
    # 解码response_messages
    row_dict['response_messages'] = pickle.loads(base64.b64decode(row_dict['response_messages']))
    print(
        f"request_id: {json.loads(row_dict['response_messages'][0])['id']}, first_chunk_latency: {row_dict['first_chunk_latency']}"  # noqa: E501
    )
    # 如果只想看一个可以break
    # break
```

## 自定义请求 API
目前内置支持 `openai` 和 `dashscope`。要扩展 API，请继承 `ApiPluginBase` 或 `DefaultApiPlugin`，并使用 `@register_api("api名称")` 注册插件。必须实现以下方法：

- build_request(messages, param) -> Dict  
  根据输入构造请求体，使用 `param.model`、`param.max_tokens`、`param.temperature` 等参数。

- parse_responses(responses: List[Dict], request: str | None = None) -> Tuple[int, int]  
  解析响应，返回 `(prompt_tokens, completion_tokens)`。若 API 不提供 usage，可用分词器估算。

- process_request(...) -> BenchmarkData  
  发送请求并收集响应与时延数据。若自定义 API 与 OpenAI 兼容（JSON + SSE），推荐继承 `DefaultApiPlugin` 直接复用其 HTTP 与流式处理逻辑，仅需实现 `build_request`、`parse_responses`。

示例：继承 `DefaultApiPlugin` 最小实现（推荐）

```python
# 仅示例，用于文档；真正文件见 evalscope/perf/plugin/api/custom_api.py
import json
from typing import Any, Dict, List, Tuple, Union

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.default_api import DefaultApiPlugin
from evalscope.perf.plugin.registry import register_api
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_api('custom')
class CustomPlugin(DefaultApiPlugin):
    """自定义 API 插件（OpenAI 兼容推荐继承 DefaultApiPlugin）。"""

    def __init__(self, param: Arguments):
        super().__init__(param)
        # 可选：用于在 API 未返回 usage 时做 token 估算
        if param.tokenizer_path is not None:
            from modelscope import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(param.tokenizer_path)
        else:
            self.tokenizer = None

    def build_request(self, messages: Union[List[Dict], str], param: Arguments = None) -> Dict:
        """将输入消息/字符串构造成自定义 API 的请求体。"""
        param = param or self.param
        try:
            if isinstance(messages, str):
                payload = {'input_text': messages}
            else:
                payload = {'messages': messages}

            # 添加常见的推理参数
            payload['model'] = param.model
            if param.max_tokens is not None:
                payload['max_tokens'] = param.max_tokens
            if param.temperature is not None:
                payload['temperature'] = param.temperature
            if param.top_p is not None:
                payload['top_p'] = param.top_p
            if param.top_k is not None:
                payload['top_k'] = param.top_k
            if param.stream is not None:
                payload['stream'] = param.stream
                payload['stream_options'] = {'include_usage': True}
            if param.extra_args:
                payload.update(param.extra_args)

            return payload
        except Exception as e:
            logger.exception(e)
            return {}

    def parse_responses(self, responses: List[Dict], request: str = None, **kwargs: Any) -> Tuple[int, int]:
        """从响应列表中提取 token 计数；若无 usage，则用分词器估算。"""
        try:
            last = responses[-1] if responses else {}
            if isinstance(last, dict) and last.get('usage'):
                usage = last['usage'] or {}
                return usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0)

            # 回退：使用分词器估算
            if self.tokenizer is not None:
                prompt_text = ''
                if request:
                    try:
                        req_js = json.loads(request)
                        if isinstance(req_js, dict):
                            if 'messages' in req_js:
                                prompt_text = ' '.join(m.get('content', '') for m in req_js.get('messages', []))
                            elif 'input_text' in req_js:
                                prompt_text = req_js.get('input_text') or ''
                    except Exception:
                        pass

                completion_text = ''
                for resp in responses:
                    if not isinstance(resp, dict):
                        continue
                    for choice in resp.get('choices', []) or []:
                        msg = choice.get('message') or {}
                        if isinstance(msg, dict) and msg.get('content'):
                            completion_text += msg.get('content') or ''
                        else:
                            completion_text += choice.get('text') or ''

                return len(self.tokenizer.encode(prompt_text)), len(self.tokenizer.encode(completion_text))

            return 0, 0
        except Exception as e:
            logger.error(f'解析响应出错: {e}')
            return 0, 0
```

使用方式示例：

```python
from dotenv import dotenv_values
from evalscope.perf.arguments import Arguments
from evalscope.perf.main import run_perf_benchmark

env = dotenv_values('.env')

args = Arguments(
    model='your-model',
    url='https://your-endpoint',
    api_key=env.get('YOUR_API_KEY'),
    api='custom',     # 使用上面注册的插件
    dataset='openqa',
    number=1,
    max_tokens=16,
    stream=True,      # 若支持流式
    debug=True,
)

run_perf_benchmark(args)
```

若你的 API 与 OpenAI 流式协议不兼容，请在自定义插件中自行实现 `process_request(...) -> BenchmarkData`（可参考 `evalscope/perf/plugin/api/default_api.py` 的实现方式）。

## 自定义数据集

要自定义数据集，您可以继承 `DatasetPluginBase` 类，并使用 `@register_dataset('数据集名称')` 进行注解，然后实现 `build_messages` 方法以返回一个 message，格式参考 [OpenAI API](https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages)。在参数中指定 `dataset` 为自定义数据集名称，即可使用自定义的数据集。

以下是一个完整的示例代码：

```python
from typing import Dict, Iterator, List

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.base import DatasetPluginBase
from evalscope.perf.plugin.registry import register_dataset


@register_dataset('custom')
class CustomDatasetPlugin(DatasetPluginBase):
    """读取数据集并返回 prompt。"""

    def __init__(self, query_parameters: Arguments):
        super().__init__(query_parameters)

    def build_messages(self) -> Iterator[List[Dict]]:
        """构建消息列表。"""
        for item in self.dataset_line_by_line(self.query_parameters.dataset_path):
            prompt = item.strip()
            if len(prompt) > self.query_parameters.min_prompt_length and len(
                    prompt) < self.query_parameters.max_prompt_length:
                if self.query_parameters.apply_chat_template:
                    yield [{'role': 'user', 'content': prompt}]
                else:
                    yield prompt


if __name__ == '__main__':
    from evalscope.perf.arguments import Arguments
    from evalscope.perf.main import run_perf_benchmark

    args = Arguments(
        model='your-model-name',
        url='https://your-api-endpoint',
        dataset_path='path/to/your/dataset.txt',  # 自定义数据集路径
        api_key='your-api-key',
        dataset='custom',  # 自定义数据集名称
    )

    run_perf_benchmark(args)
```

## 注意事项

1. API 插件开发  
   - 必须实现 `build_request`、`parse_responses`，并提供 `process_request(...) -> BenchmarkData`（或继承 `DefaultApiPlugin` 复用默认实现）。
   - 使用 `@register_api("api名称")` 注册插件。
   - 优先使用 `DefaultApiPlugin` 以复用 HTTP、SSE、usage 收集等通用逻辑。

2. 数据集插件开发  
   - 实现 `build_messages` 并使用 `@register_dataset("数据集名称")` 注册。

3. 调试建议  
   - 使用 `logger` 输出关键信息。
   - 确保响应结构与解析逻辑一致，必要时打印原始响应便于定位问题。
