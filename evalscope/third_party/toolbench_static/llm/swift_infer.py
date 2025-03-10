import os
from dataclasses import dataclass
from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, get_template

# 设置GPU环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

@dataclass
class SwiftInferArgs:
    model_id_or_path: str
    model_type: str
    infer_backend: str = 'vllm'  # 可选 'pt', 'vllm', 'lmdeploy'
    max_new_tokens: int = 2048
    temperature: float = 0.1
    max_batch_size: int = 16

class SwiftInfer:

    def __init__(self, args: SwiftInferArgs):
        # infer backend模型初始化
        if args.infer_backend == 'pt':
            self.engine: InferEngine = PtEngine(args.model_id_or_path, max_batch_size=args.max_batch_size)
        elif args.infer_backend == 'vllm':
            from swift.llm import VllmEngine
            self.engine: InferEngine = VllmEngine(args.model_id_or_path, max_model_len=8192)
        elif args.infer_backend == 'lmdeploy':
            from swift.llm import LmdeployEngine
            self.engine: InferEngine = LmdeployEngine(args.model_id_or_path)
        else:
            raise ValueError(f'Unsupported infer_backend: {args.infer_backend}')

        # 基本配置获取 （可选）
        self.request_config = RequestConfig(
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            stream=False  # 可以透传参数改为True进行流式推理
        )

    def predict(self, system: str, query: str, history: list):
        # Swift 3.0标准接口中，消息传入的格式是：
        # messages: [{"role": "system", "content": "<SYSTEM_PROMPT>"},
        #            {"role": "user", "content": "用户问题内容"},
        #            {"role": "assistant", "content": "助手回答内容"}, ...]

        messages = []
        if system.strip():
            messages.append({'role': 'system', 'content': system})

        # 将历史对话拼接进message中
        for qa_pair in history:
            # 假定 history 中每个元素形如 ("user input", "model response")，请根据你的数据格式进行调整。
            user_answer, model_response = qa_pair
            messages.append({'role': 'user', 'content': user_answer})
            messages.append({'role': 'assistant', 'content': model_response})

        # 添加本次用户问题
        messages.append({'role': 'user', 'content': query})

        infer_request = InferRequest(messages=messages)

        # 进行推理
        response = self.engine.infer([infer_request], self.request_config)

        # 提取模型返回的文本结果（假设非stream模式）
        result_text = response[0].choices[0].message.content.strip()

        return result_text
