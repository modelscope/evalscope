# EvalScope 服务部署

## 简介

EvalScope 服务模式提供了基于 HTTP API 的评测和压测能力,旨在解决以下场景的需求:

1. **远程调用**: 支持通过网络远程调用评测功能,无需在本地配置复杂的评测环境
2. **服务集成**: 方便将评测能力集成到现有的工作流、CI/CD 流程或自动化测试系统中
3. **多用户协作**: 支持多个用户或系统同时调用评测服务,提高资源利用率
4. **统一管理**: 集中管理评测资源和配置,便于维护和监控
5. **灵活部署**: 可以部署在专用服务器或容器环境中,与业务系统解耦

Flask 服务封装了 EvalScope 的核心评测(eval)和压测(perf)功能,通过标准的 RESTful API 对外提供服务,使得评测能力可以像其他微服务一样被调用和集成。

## 功能特性

- **模型评测** (`/api/v1/eval`): 支持OpenAI API兼容模型的评测，请求参数请参考[文档](../get_started/parameters.md)
- **性能压测** (`/api/v1/perf/invoke`): 使用 typed 配置对支持的 HTTP 协议执行压测，请求参数参考[文档](./stress_test/parameters.md)

## 安装环境


### 完整安装(推荐)

```bash
pip install evalscope[service]
```

### 开发环境安装

```bash
# Clone仓库
git clone https://github.com/modelscope/evalscope.git
cd evalscope

# 安装包含service的开发版本
pip install -e '.[service]'
```

## 启动服务

### 命令行启动

```bash
# 使用默认配置 (host: 0.0.0.0, port: 9000)
evalscope service

# 自定义主机和端口
evalscope service --host 127.0.0.1 --port 9000

# 启用调试模式
evalscope service --debug
```

### Python代码启动

```python
from evalscope.service import run_service

# 启动服务
run_service(host='0.0.0.0', port=9000, debug=False)
```

## API端点

### 1. 健康检查

```bash
GET /health
```

**响应示例:**
```json
{
  "status": "ok",
  "service": "evalscope",
  "timestamp": "2025-12-04T10:00:00"
}
```

### 2. 模型评测

```bash
POST /api/v1/eval
```

**请求体示例:**
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


**必需参数:**
- `model`: 模型名称
- `datasets`: 数据集列表
- `api_url`: API端点URL (OpenAI兼容)

**可选参数:**
- `api_key`: API密钥 (默认: "EMPTY")
- `limit`: 评测样本数量限制
- `eval_batch_size`: 批处理大小 (默认: 1)
- `generation_config`: 生成配置
  - `temperature`: 温度参数 (默认: 0.0)
  - `max_tokens`: 最大生成token数 (默认: 2048)
  - `top_p`: nucleus采样参数
  - `top_k`: top-k采样参数
- `work_dir`: 输出目录
- `debug`: 调试模式
- `seed`: 随机种子 (默认: 42)

```{seealso}
具体参数说明请参考：[评测参数文档](../get_started/parameters.md)
```

**响应示例:**
```json
{
  "status": "success",
  "message": "Evaluation completed",
  "result": {"...": "..."},
  "output_dir": "/path/to/outputs/20251204_100000"
}
```

### 3. 性能压测

```bash
POST /api/v1/perf/invoke
```

**请求体示例:**
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

**必需参数:**
- `target`: 模型、基础 URL、协议、认证和超时配置
- `suite.loads`: 显式的 closed-loop、open-loop 或 conversation 负载配置

**可选参数:**
- `workload`: 数据集/工作负载来源及插件参数
- `generation`: 请求生成参数
- `runtime`: 随机种子、有界队列、进度和可视化配置
- `metrics`: workload 聚合时间窗口
- `sla`: 可选的 typed SLA 搜索配置

```{seealso}
具体参数说明请参考：[性能压测参数文档](./stress_test/parameters.md)
```

**响应示例:**
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

## 使用示例

### 使用curl测试评测端点

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

### 使用curl测试压测端点

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

### 使用Python requests

```python
import requests

# 评测请求
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

# 压测请求
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

## 注意事项

1. **仅支持OpenAI API兼容模型**: 此服务专为OpenAI API兼容的模型设计
2. **长时间运行的任务**: 评测和压测任务可能需要较长时间，建议客户端设置合适的HTTP超时时间，因为API调用是同步的，会阻塞直到任务完成。
3. **输出目录**: 评测结果会保存在配置的`work_dir`中，默认为`outputs/`
4. **错误处理**: 服务会返回详细的错误信息和堆栈跟踪(在debug模式下)
5. **资源管理**: 压测时注意并发数设置，避免过载服务器

## 错误码

- `400`: 请求参数错误
- `404`: 端点不存在
- `500`: 服务器内部错误

## 示例场景

### 场景1: 快速评测Qwen模型

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

### 场景2: 压测本地部署的模型

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

### 场景3: 多数据集评测

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
