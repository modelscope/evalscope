from typing import Any, Dict, List, Optional

from evalscope.api.messages import ChatMessage
from evalscope.api.model import GenerateConfig, ModelAPI, ModelOutput
from evalscope.api.registry import register_model_api
from evalscope.api.tool import ToolChoice, ToolInfo


# 1. 使用register_model_api注册模型
@register_model_api(name='my_custom_model')
class MyCustomModel(ModelAPI):
    """自定义模型实现"""

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Dict[str, Any],
    ) -> None:
        super().__init__(model_name, base_url, api_key, config)
        self.model_args = model_args
        print(self.model_args)

        # 2. 在这里初始化您的模型
        # 例如：加载模型文件、建立连接等

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        # 3. 实现模型推理逻辑

        # 3.1 处理输入消息
        input_text = self._process_messages(input)

        # 3.2 调用您的模型
        response = self._call_model(input_text, config)

        # 3.3 返回标准化输出
        return ModelOutput.from_content(
            model=self.model_name,
            content=response
        )

    def _process_messages(self, messages: List[ChatMessage]) -> str:
        """将聊天消息转换为文本"""
        text_parts = []
        for message in messages:
            role = getattr(message, 'role', 'user')
            content = getattr(message, 'content', str(message))
            text_parts.append(f'{role}: {content}')
        return '\n'.join(text_parts)

    def _call_model(self, input_text: str, config: GenerateConfig) -> str:
        """调用您的模型进行推理"""
        # 在这里实现您的模型调用逻辑
        # 例如：调用 API、本地模型推理等
        return f'Response to: {input_text}'

def test_model_api():
    from evalscope import TaskConfig, run_task

    # 创建模型实例
    custom_model = MyCustomModel(
        model_name='my-model',
        model_args={'test': 'test'}
    )

    # 配置评测任务
    task_config = TaskConfig(
        model=custom_model,
        datasets=['gsm8k'],
        limit=5
    )

    # 运行评测
    results = run_task(task_cfg=task_config)

def test_registry():
    from evalscope import TaskConfig, run_task

    # 使用注册的模型
    task_config = TaskConfig(
        model='my-model',
        eval_type='my_custom_model', # registered model name
        datasets=['gsm8k'],
        model_args={'test': 'test'},
        limit=5
    )

    results = run_task(task_cfg=task_config)

if __name__ == '__main__':
    test_model_api()
    test_registry()
