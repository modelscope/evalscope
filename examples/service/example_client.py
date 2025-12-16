#!/usr/bin/env python3
# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Example script demonstrating the EvalScope Flask service usage.

Environment setup:
    1. Install dependencies:
       pip install evalscope[service]

    2. Set API key (choose one method):
       - Environment variable: export DASHSCOPE_API_KEY=your-api-key
       - Or create .env file with: DASHSCOPE_API_KEY=your-api-key

    3. Start the service (in another terminal):
       evalscope service --host 0.0.0.0 --port 9000

    4. Run this example:
       python examples/service/example_client.py
"""
import requests
import time
from dotenv import dotenv_values, load_dotenv

load_dotenv('.env')

env = dotenv_values('.env')


def test_health_check(base_url):
    """Test the health check endpoint."""
    print('\n=== Testing Health Check ===')
    response = requests.get(f'{base_url}/health')
    print(f'Status Code: {response.status_code}')
    print(f'Response: {response.json()}')
    return response.status_code == 200


def test_evaluation(base_url, model_api_url, api_key='EMPTY'):
    """Test the evaluation endpoint."""
    print('\n=== Testing Evaluation ===')

    eval_request = {
        'model': 'qwen-plus',
        'api_url': model_api_url,
        'api_key': api_key,
        'datasets': ['gsm8k'],
        'limit': 5,
        'generation_config': {
            'temperature': 0.0,
            'max_tokens': 512
        },
        'debug': True,
        'work_dir': 'outputs/eval_test',
        'no_timestamp': True
    }

    print(f'Request: {eval_request}')

    try:
        response = requests.post(
            f'{base_url}/api/v1/eval',
            json=eval_request,
            timeout=300  # 5 minutes timeout
        )
        print(f'Status Code: {response.status_code}')
        result = response.json()

        if response.status_code == 200:
            print(f"Status: {result['status']}")
            print(f"Message: {result['message']}")
            print(f"Output Dir: {result['output_dir']}")
            return True
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            return False
    except requests.Timeout:
        print('Request timed out')
        return False
    except Exception as e:
        print(f'Exception: {str(e)}')
        return False


def test_performance_multi(base_url, model_api_url, api_key='EMPTY'):
    """Test the performance test endpoint."""
    print('\n=== Testing Performance Benchmark ===')

    perf_request = {
        'model': 'qwen-plus',
        'url': f'{model_api_url}/chat/completions',
        'api': 'openai',
        'api_key': api_key,
        'number': [1, 2],
        'parallel': [1, 2],
        'dataset': 'openqa',
        'max_tokens': 128,
        'temperature': 0.0,
        'stream': True,
        'debug': True
    }

    print(f'Request: {perf_request}')

    try:
        response = requests.post(
            f'{base_url}/api/v1/perf',
            json=perf_request,
            timeout=300  # 5 minutes timeout
        )
        print(f'Status Code: {response.status_code}')
        result = response.json()
        print(f'Result: {result}')

        if response.status_code == 200:
            print(f"Status: {result['status']}")
            print(f"Message: {result['message']}")
            print(f"Output Dir: {result['output_dir']}")
            if 'metrics' in result:
                print(f"Metrics: {result['metrics']}")
            return True
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            return False
    except requests.Timeout:
        print('Request timed out')
        return False
    except Exception as e:
        print(f'Exception: {str(e)}')
        return False


def test_performance_single(base_url, model_api_url, api_key='EMPTY'):
    """Test the performance test endpoint with single run."""
    print('\n=== Testing Performance Benchmark (Single Run) ===')

    perf_request = {
        'model': 'qwen-plus',
        'url': f'{model_api_url}/chat/completions',
        'api': 'openai',
        'api_key': api_key,
        'number': 1,
        'parallel': 1,
        'dataset': 'openqa',
        'max_tokens': 128,
        'temperature': 0.0,
        'stream': True,
        'debug': True,
        'outputs_dir': 'outputs/perf_single_run',
        'no_timestamp': True
    }

    print(f'Request: {perf_request}')

    try:
        response = requests.post(
            f'{base_url}/api/v1/perf',
            json=perf_request,
            timeout=300  # 5 minutes timeout
        )
        print(f'Status Code: {response.status_code}')
        result = response.json()
        print(f'Result: {result}')

        if response.status_code == 200:
            print(f"Status: {result['status']}")
            print(f"Message: {result['message']}")
            print(f"Output Dir: {result['output_dir']}")
            if 'metrics' in result:
                print(f"Metrics: {result['metrics']}")
            return True
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
            return False
    except requests.Timeout:
        print('Request timed out')
        return False
    except Exception as e:
        print(f'Exception: {str(e)}')
        return False


def print_curl_examples(base_url, model_api_url, api_key):
    """Print curl command examples for all endpoints."""
    print('\n' + '=' * 60)
    print('=== cURL Command Examples ===')
    print('=' * 60)

    # Health check
    print('\n# 1. Health Check')
    print(f"curl -X GET '{base_url}/health'")

    # Evaluation
    print('\n# 2. Run Evaluation')
    print(f"""curl -X POST '{base_url}/api/v1/eval' \\
  -H 'Content-Type: application/json' \\
  -d '{{
    "model": "qwen-plus",
    "api_url": "{model_api_url}",
    "api_key": "{api_key}",
    "datasets": ["gsm8k"],
    "limit": 5,
    "generation_config": {{
      "temperature": 0.0,
      "max_tokens": 512
    }},
    "debug": true
  }}'""")

    # Performance test (multi)
    print('\n# 3. Run Performance Benchmark (Multiple Runs)')
    print(f"""curl -X POST '{base_url}/api/v1/perf' \\
  -H 'Content-Type: application/json' \\
  -d '{{
    "model": "qwen-plus",
    "url": "{model_api_url}/chat/completions",
    "api": "openai",
    "api_key": "{api_key}",
    "number": [1, 2],
    "parallel": [1, 2],
    "dataset": "openqa",
    "max_tokens": 128,
    "temperature": 0.0,
    "stream": true,
    "debug": true
  }}'""")

    # Performance test (single)
    print('\n# 4. Run Performance Benchmark (Single Run)')
    print(f"""curl -X POST '{base_url}/api/v1/perf' \\
  -H 'Content-Type: application/json' \\
  -d '{{
    "model": "qwen-plus",
    "url": "{model_api_url}/chat/completions",
    "api": "openai",
    "api_key": "{api_key}",
    "number": 1,
    "parallel": 1,
    "dataset": "openqa",
    "max_tokens": 128,
    "temperature": 0.0,
    "stream": true,
    "debug": true
  }}'""")

    print('\n' + '=' * 60 + '\n')


def main():
    """Main test function."""
    # Configuration
    SERVICE_URL = 'http://localhost:9000'
    MODEL_API_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    API_KEY = env.get('DASHSCOPE_API_KEY', 'EMPTY')

    print('=== EvalScope Service Test ===')
    print(f'Service URL: {SERVICE_URL}')
    print(f'Model API URL: {MODEL_API_URL}')

    # Print curl examples first
    print_curl_examples(SERVICE_URL, MODEL_API_URL, API_KEY)

    # Test health check
    if not test_health_check(SERVICE_URL):
        print('Health check failed!')
        return

    # Wait a bit
    time.sleep(1)

    print('\n' + '=' * 60)
    print('Make sure the EvalScope service is running at:', SERVICE_URL)
    print('Using DashScope API at:', MODEL_API_URL)
    print('=' * 60)
    test_evaluation(SERVICE_URL, MODEL_API_URL, API_KEY)
    # test_performance_multi(SERVICE_URL, MODEL_API_URL, API_KEY)
    test_performance_single(SERVICE_URL, MODEL_API_URL, API_KEY)

    print('\n=== Test Complete ===')


if __name__ == '__main__':
    main()
