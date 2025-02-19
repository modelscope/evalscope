import os
from openai import OpenAI


def request_qwen(content):
    try:
        client = OpenAI(
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        )

        completion = client.chat.completions.create(
            model='qwen-plus',
            messages=[{'role': 'user', 'content': content}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(e)
