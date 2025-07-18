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
目前支持的 API 请求格式有 `openai` 和 `dashscope`。要扩展 API，您可以继承 `ApiPluginBase` 类，并使用 `@register_api("api名称")` 进行注解，需实现如下方法：

- **`build_request()`**  
  通过 `messages` 和 `param` 中的 `model` 和 `query_template` 来构建请求，该请求将发送到目标 API。

- **`process_request()`**  
  将请求发送到目标 API，并处理返回的响应（是否成功、响应码、响应内容）。

- **`parse_responses()`**  
  解析响应，返回 `prompt_tokens` 和 `completion_tokens` 的数量，用于计算推理速度。

以下是一个完整的 `custom` 示例代码：

```python
import json
import aiohttp
from typing import Any, AsyncGenerator, Dict, List, Tuple, Union

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.base import ApiPluginBase
from evalscope.perf.plugin.registry import register_api
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_api('custom')
class CustomPlugin(ApiPluginBase):
    """支持自定义 API 实现的插件模板。"""

    def __init__(self, param: Arguments):
        """初始化插件，加载必要的参数和 tokenizer。

        Args:
            param (Arguments): 配置参数，包括：
                - tokenizer_path: 用于计数的分词器路径
                - model: 要使用的模型名称
                - 其他请求参数，如 max_tokens、temperature 等
        """
        super().__init__(param=param)
        if param.tokenizer_path is not None:
            from modelscope import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(param.tokenizer_path)
        else:
            self.tokenizer = None

    def build_request(self, messages: Union[List[Dict], str], param: Arguments = None) -> Dict:
        """构建自定义 API 请求体。

        Args:
            messages (Union[List[Dict], str]): 输入消息，可以是消息字典的列表（用于聊天模型）或字符串（用于完成模型）。
            param (Arguments, optional): 请求参数。默认为 self.param。

        Returns:
            Dict: 格式正确的自定义 API 请求体。
        """
        param = param or self.param
        try:
            # 如果没有提供模板，则创建默认查询格式
            if isinstance(messages, str):
                query = {'input_text': messages}
            else:
                query = {'messages': messages}
            
            # 将模型参数添加到请求中
            return self._add_parameters_to_request(query, param)
        except Exception as e:
            logger.exception(e)
            return None

    def _add_parameters_to_request(self, payload: Dict, param: Arguments) -> Dict:
        """向请求体中添加模型参数。

        此辅助方法根据自定义 API 支持的内容，将温度、最大令牌等各种参数添加到请求中。

        Args:
            payload (Dict): 基础请求负载。
            param (Arguments): 要添加的参数。

        Returns:
            Dict: 添加了参数的请求负载。
        """
        # 添加模型名称
        payload['model'] = param.model
            
        # 如果提供了，则添加各种参数
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

        # 添加通过命令行传递的任何额外参数
        if param.extra_args is not None:
            payload.update(param.extra_args)
            
        return payload

    def parse_responses(self, responses: List[str], request: Any = None, **kwargs) -> Tuple[int, int]:
        """解析响应并返回 token 数量。

        此方法从 API 响应中提取输入和输出 token 的数量。
        不同的 API 可能以不同的格式返回此信息，或者您可能需要使用分词器计算它。

        Args:
            responses (List[str]): API 响应字符串的列表。
            request (Any, optional): 原始请求，可能在 token 计算中需要。
            **kwargs: 其他参数。

        Returns:
            Tuple[int, int]: (input_tokens, output_tokens) - 提示和完成中的 token 数量。
        """
        try:
            # 示例 1：尝试从 API 响应中获取 token 计数
            last_response = json.loads(responses[-1])
            
            # 如果 API 提供了 token 使用信息
            if 'usage' in last_response and last_response['usage']:
                input_tokens = last_response['usage'].get('prompt_tokens', 0)
                output_tokens = last_response['usage'].get('completion_tokens', 0)
                return input_tokens, output_tokens
                
            # 示例 2：如果没有使用信息，则使用分词器计算 token
            if self.tokenizer is not None:
                input_text = ""
                output_text = ""
                
                # 从请求中提取输入文本
                if request and 'messages' in request:
                    # 对于聊天 API
                    input_text = " ".join([msg['content'] for msg in request['messages']])
                elif request and 'input_text' in request:
                    # 对于完成 API
                    input_text = request['input_text']
                
                # 从响应中提取输出文本
                for response in responses:
                    js = json.loads(response)
                    if 'choices' in js:
                        for choice in js['choices']:
                            if 'message' in choice and 'content' in choice['message']:
                                output_text += choice['message']['content']
                            elif 'text' in choice:
                                output_text += choice['text']
                
                # 计数 token
                input_tokens = len(self.tokenizer.encode(input_text))
                output_tokens = len(self.tokenizer.encode(output_text))
                return input_tokens, output_tokens
                
            # 如果没有使用信息且没有分词器，则引发错误
            raise ValueError("无法确定 token 计数：响应中没有使用信息且未提供分词器。")
            
        except Exception as e:
            logger.error(f"解析响应时出错：{e}")
            return 0, 0

    async def process_request(self, client_session: aiohttp.ClientSession, url: str, headers: Dict,
                              body: Dict) -> AsyncGenerator[Tuple[bool, int, str], None]:
        """处理 HTTP 请求并处理响应。

        此方法处理将请求发送到您的 API 和处理响应，包括处理流响应（如果支持）。

        Args:
            client_session (aiohttp.ClientSession): aiohttp 客户端会话。
            url (str): API 端点 URL。
            headers (Dict): 请求头。
            body (Dict): 请求体。

        Yields:
            Tuple[bool, int, str]: (is_error, status_code, response_data)
                - is_error: 响应是否表示错误
                - status_code: HTTP 状态码
                - response_data: 响应内容
        """
        try:
            # 设置内容类型头
            headers = {'Content-Type': 'application/json', **headers}
            
            # 将主体转换为 JSON
            data = json.dumps(body, ensure_ascii=False)
            
            # 发送请求
            async with client_session.request('POST', url=url, data=data, headers=headers) as response:
                status_code = response.status
                
                # 检查是否为流响应
                if 'text/event-stream' in response.content_type:
                    # 处理流响应
                    async for line in response.content:
                        line_str = line.decode('utf-8').strip()
                        if not line_str:
                            continue
                        
                        # 检查服务器发送事件中的数据前缀
                        if line_str.startswith('data: '):
                            data = line_str[6:]  # 移除 'data: ' 前缀
                            
                            # 检查是否为流的结束
                            if data == '[DONE]':
                                break
                                
                            try:
                                # 解析 JSON 数据
                                parsed_data = json.loads(data)
                                yield (False, status_code, json.dumps(parsed_data))
                            except json.JSONDecodeError:
                                yield (True, status_code, f"解析 JSON 失败：{data}")
                else:
                    # 处理常规响应
                    if 'application/json' in response.content_type:
                        # JSON 响应
                        content = await response.json()
                        yield (status_code >= 400, status_code, json.dumps(content))
                    else:
                        # 文本响应
                        content = await response.text()
                        yield (status_code >= 400, status_code, content)
                        
        except Exception as e:
            logger.error(f"process_request 中出错：{e}")
            yield (True, 500, str(e))

if __name__ == "__main__":
    # 自定义 API 插件的示例用法
    from dotenv import dotenv_values
    env = dotenv_values('.env')
    
    from evalscope.perf.arguments import Arguments
    from evalscope.perf.main import run_perf_benchmark

    args = Arguments(
        model='your-model-name',
        url='https://your-api-endpoint',
        api_key='your-api-key',
        api='custom',  # 使用自定义 API 插件
        dataset='openqa',
        number=1,
        max_tokens=10,
        debug=True,
    )

    run_perf_benchmark(args)
```

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

1. **API 插件开发**  
   - 确保实现 `build_request`、`process_request` 和 `parse_responses` 方法。
   - 使用 `@register_api("api名称")` 注册插件。

2. **数据集插件开发**  
   - 确保实现 `build_messages` 方法。
   - 使用 `@register_dataset("数据集名称")` 注册插件。

3. **调试建议**  
   - 使用日志记录 (`logger`) 来调试插件行为。
   - 确保 API 响应格式与解析逻辑一致。

通过以上示例，您可以轻松扩展支持新的 API 和数据集格式。
