# EvalScope Service Deployment

## Introduction

EvalScope service mode provides HTTP API-based evaluation and stress testing capabilities, designed to address the following scenarios:

1. **Remote Invocation**: Support remote evaluation functionality through network without configuring complex evaluation environments locally
2. **Service Integration**: Easily integrate evaluation capabilities into existing workflows, CI/CD pipelines, or automated testing systems
3. **Multi-user Collaboration**: Support multiple users or systems calling the evaluation service simultaneously, improving resource utilization
4. **Unified Management**: Centrally manage evaluation resources and configurations for easier maintenance and monitoring
5. **Flexible Deployment**: Can be deployed on dedicated servers or container environments, decoupled from business systems

The Flask service encapsulates EvalScope's core evaluation (eval) and stress testing (perf) functionalities, providing services through standard RESTful APIs, making evaluation capabilities callable and integrable like other microservices.

## Features

- **Model Evaluation** (`/api/v1/eval`): Support evaluation of OpenAI API-compatible models, request parameters refer to [documentation](../get_started/parameters.md)
- **Performance Testing** (`/api/v1/perf/invoke`): Run typed performance suites against supported HTTP protocols; request parameters refer to [documentation](./stress_test/parameters.md)

## Environment Setup


### Full Installation (Recommended)

```bash
pip install evalscope[service]
```

### Development Environment Installation

```bash
# Clone repository
git clone https://github.com/modelscope/evalscope.git
cd evalscope

# Install development version with service
pip install -e '.[service]'
```

## Starting the Service

### Command Line Launch

```bash
# Use default configuration (host: 0.0.0.0, port: 9000)
evalscope service

# Custom host and port
evalscope service --host 127.0.0.1 --port 9000

# Enable debug mode
evalscope service --debug
```

### Python Code Launch

```python
from evalscope.service import run_service

# Start service
run_service(host='0.0.0.0', port=9000, debug=False)
```

## API Endpoints

### 1. Health Check

```bash
GET /health
```

**Response Example:**
```json
{
  "status": "ok",
  "service": "evalscope",
  "timestamp": "2025-12-04T10:00:00"
}
```

### 2. Model Evaluation

```bash
POST /api/v1/eval
```

**Request Body Example:**
```json
{
  "model": "qwen-plus",
  "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "api_key": "your-api-key",
  "datasets": ["gsm8k", "iquiz"],
  "limit": 10,
  "generation_config": {
    "temperature": 0.0,
    "max_tokens": 2048
  }
}
```

**Required Parameters:**
- `model`: Model name
- `datasets`: List of datasets
- `api_url`: API endpoint URL (OpenAI-compatible)

**Optional Parameters:**
- `api_key`: API key (default: "EMPTY")
- `limit`: Evaluation sample quantity limit
- `eval_batch_size`: Batch size (default: 1)
- `generation_config`: Generation configuration
  - `temperature`: Temperature parameter (default: 0.0)
  - `max_tokens`: Maximum generation tokens (default: 2048)
  - `top_p`: Nucleus sampling parameter
  - `top_k`: Top-k sampling parameter
- `work_dir`: Output directory
- `debug`: Debug mode
- `seed`: Random seed (default: 42)

```{seealso}
For detailed parameter descriptions, refer to: [Evaluation Parameter Documentation](../get_started/parameters.md)
```

**Response Example:**
```json
{
  "status": "success",
  "message": "Evaluation completed",
  "result": {"...": "..."},
  "output_dir": "/path/to/outputs/20251204_100000"
}
```

### 3. Performance Testing

```bash
POST /api/v1/perf/invoke
```

**Request Body Example:**
```json
{
  "target": {
    "model": "qwen-plus",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "protocol": "openai_chat",
    "api_key": "your-api-key"
  },
  "workload": {"name": "openqa"},
  "generation": {"max_tokens": 2048, "temperature": 0.0},
  "suite": {
    "loads": [
      {"mode": "closed_loop", "concurrency": 10, "request_count": 100}
    ]
  }
}
```

**Required Parameters:**
- `target`: Model, base URL, protocol, authentication, and timeout configuration
- `suite.loads`: Explicit closed-loop, open-loop, or conversation load specifications

**Optional Parameters:**
- `workload`: Dataset/workload source and plugin options
- `generation`: Request generation parameters
- `runtime`: Seed, bounded queue, progress, and visualizer settings
- `metrics`: Workload aggregation windows
- `sla`: Optional typed SLA search configuration

```{seealso}
For detailed parameter descriptions, refer to: [Performance Parameter Documentation](./stress_test/parameters.md)
```

**Response Example:**
```json
{
  "status": "completed",
  "task_id": "perf-demo",
  "result": {
    "run_id": "perf",
    "runs": [
      {
        "run_spec": {"load_id": "load-000", "...": "..."},
        "summary": {"...": "..."},
        "percentiles": {"...": "..."}
      }
    ],
    "artifacts": {"...": "..."}
  }
}
```

## Usage Examples

### Testing Evaluation Endpoint with curl

```bash
curl -X POST http://localhost:9000/api/v1/eval \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-plus",
    "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key": "your-api-key",
    "datasets": ["gsm8k"],
    "limit": 5
  }'
```

### Testing Performance Endpoint with curl

```bash
curl -X POST http://localhost:9000/api/v1/perf/invoke \
  -H "Content-Type: application/json" \
  -H "EvalScope-Task-Id: perf-demo" \
  -d '{
    "target": {"model": "qwen-plus", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "protocol": "openai_chat"},
    "workload": {"name": "openqa"},
    "suite": {"loads": [{"mode": "closed_loop", "concurrency": 5, "request_count": 50}]}
  }'
```

### Using Python requests

```python
import requests

# Evaluation request
eval_response = requests.post(
    'http://localhost:9000/api/v1/eval',
    json={
        'model': 'qwen-plus',
        'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'api_key': 'your-api-key',
        'datasets': ['gsm8k', 'iquiz'],
        'limit': 10,
        'generation_config': {
            'temperature': 0.0,
            'max_tokens': 2048
        }
    }
)
print(eval_response.json())

# Performance test request
perf_response = requests.post(
    'http://localhost:9000/api/v1/perf/invoke',
    headers={'EvalScope-Task-Id': 'perf-demo'},
    json={
        'target': {
            'model': 'qwen-plus',
            'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'protocol': 'openai_chat'
        },
        'workload': {'name': 'openqa'},
        'suite': {'loads': [{'mode': 'closed_loop', 'concurrency': 10, 'request_count': 100}]}
    }
)
print(perf_response.json())
```

## Important Notes

1. **OpenAI API-Compatible Models Only**: This service is designed specifically for OpenAI API-compatible models
2. **Long-Running Tasks**: Evaluation and performance testing tasks may take considerable time. We recommend setting appropriate HTTP timeout values on the client side, as the API calls are synchronous and will block until completion.
3. **Output Directory**: Evaluation results are saved in the configured `work_dir`, default is `outputs/`
4. **Error Handling**: The service returns detailed error messages and stack traces (in debug mode)
5. **Resource Management**: Pay attention to concurrency settings during stress testing to avoid server overload

## Error Codes

- `400`: Invalid request parameters
- `404`: Endpoint not found
- `500`: Internal server error

## Example Scenarios

### Scenario 1: Quick Evaluation of Qwen Model

```bash
curl -X POST http://localhost:9000/api/v1/eval \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-plus",
    "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key": "sk-...",
    "datasets": ["gsm8k"],
    "limit": 100
  }'
```

### Scenario 2: Stress Testing Locally Deployed Model

```bash
curl -X POST http://localhost:9000/api/v1/perf/invoke \
  -H "Content-Type: application/json" \
  -H "EvalScope-Task-Id: local-perf" \
  -d '{
    "target": {"model": "qwen2.5", "base_url": "http://localhost:8000/v1", "protocol": "openai_chat"},
    "generation": {"max_tokens": 2048},
    "suite": {"loads": [{"mode": "closed_loop", "concurrency": 20, "request_count": 1000}]}
  }'
```

### Scenario 3: Multi-Dataset Evaluation

```bash
curl -X POST http://localhost:9000/api/v1/eval \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-plus",
    "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "datasets": ["gsm8k", "iquiz", "ceval"],
    "limit": 50,
    "eval_batch_size": 4
  }'
```
