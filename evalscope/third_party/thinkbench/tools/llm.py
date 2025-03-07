import os
from openai import OpenAI


def request_url(llm_config, content):
    try:
        client = OpenAI(
            api_key=llm_config['api_key'],
            base_url=llm_config['base_url'],
        )
        completion = client.chat.completions.create(
            model=llm_config['model_name'],
            messages=[{'role': 'user', 'content': content}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(e)
        return None

def request_qwen(content):
    try:
        client = OpenAI(
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        )

        completion = client.chat.completions.create(
            model='qwen-max',
            messages=[{'role': 'user', 'content': content}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(e)


def request_local(content):
    try:
        client = OpenAI(
            api_key='EMPTY',
            base_url='http://0.0.0.0:8801/v1',
        )
        completion = client.chat.completions.create(
            model='Qwen2.5-72B-Instruct',
            messages=[{'role': 'user', 'content': content}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(e)
