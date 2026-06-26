import os
from openai import OpenAI
from typing import Dict, Optional


def request_url(llm_config: Dict[str, str], content: str) -> Optional[str]:
    """Send a chat completion request using the provided LLM config.

    Args:
        llm_config: Dict with keys 'api_key', 'base_url', 'model_name'.
        content: The user message content to send.

    Returns:
        The assistant response text, or None on failure.
    """
    if not llm_config or not all(k in llm_config for k in ('api_key', 'base_url', 'model_name')):
        return None
    try:
        client = OpenAI(
            api_key=llm_config['api_key'],
            base_url=llm_config['base_url'],
        )
        completion = client.chat.completions.create(
            model=llm_config['model_name'],
            messages=[{'role': 'user', 'content': content}],
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(e)
        return None


def request_qwen(content: str) -> Optional[str]:
    """Send a chat completion request to DashScope qwen-max.

    Requires DASHSCOPE_API_KEY environment variable to be set.
    """
    try:
        client = OpenAI(
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        )
        completion = client.chat.completions.create(
            model='qwen-max',
            messages=[{'role': 'user', 'content': content}],
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(e)
        return None
