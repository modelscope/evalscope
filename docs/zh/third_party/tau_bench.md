# τ-bench

## 简介

τ-bench (TAU Bench) 是一个专门用于评估大语言模型在**动态对话代理场景**中工具使用能力的基准测试框架。该基准模拟了用户（由语言模型模拟）与语言代理之间的动态对话，代理被提供特定领域的API工具和策略指导。

- **项目地址**：[https://github.com/sierra-research/tau-bench](https://github.com/sierra-research/tau-bench)

该评测基准的核心特点：

- **动态交互**：模拟真实用户与AI代理的多轮对话
- **工具集成**：代理需要合理使用提供的API工具
- **策略遵循**：代理需要遵循特定的业务策略和指导原则
- **领域专业性**：涵盖航空公司客服、零售客服等实际业务场景

支持的评测领域：
- `airline`：航空公司客服场景
- `retail`：零售客服场景

## 安装依赖

在运行评测之前需要安装以下依赖：

```bash
pip install evalscope  # 安装 evalscope
pip install git+https://github.com/sierra-research/tau-bench  # 安装 tau-bench
```

## 使用方法

运行下面的代码即可启动评测。下面以qwen-plus模型为例进行评测。

**⚠️ 注意：仅支持API模型服务评测，本地模型评测建议使用vLLM等框架预先拉起服务。官方默认使用user model为gpt-4o，要获得榜单效果，请使用此模型进行评测。**

```python
import os
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='qwen-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type='openai_api',  # 使用API模型服务
    datasets=['tau_bench'],
    dataset_args={
        'tau_bench': {
            'subset_list': ['airline', 'retail'],  # 选择评测领域
            'extra_params': {
                'user_model': 'qwen-plus',  # 用户模拟模型
                'api_key': os.getenv('DASHSCOPE_API_KEY'),
                'api_base': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'generation_config': {
                    'temperature': 0.7,
                    'max_tokens': 1024
                }
            }
        }
    },
    eval_batch_size=5,
    limit=5,  # 限制评测数量，便于快速测试，正式评测时建议去掉此项
    generation_config={
        'temperature': 0.6,
        'n': 1,
        'max_tokens': 4096,
    },
)

run_task(task_cfg=task_cfg)
```


## 评测方法说明

### 对话流程

τ-bench 的评测流程包含以下关键步骤：

1. **任务初始化**：系统为代理提供特定领域的API工具集和策略指导
2. **用户模拟**：用户模拟器根据预定义场景生成自然的用户请求
3. **代理响应**：被测试的代理模型根据工具和策略生成响应
4. **多轮交互**：用户和代理进行多轮对话直到任务完成或失败
5. **结果评估**：基于任务完成度和策略遵循度进行评分

### 评估维度

- 代理是否成功完成用户请求的核心任务
- 是否正确使用了必要的API工具

### 领域特性

#### 航空公司场景 (Airline)
- **API工具**：航班查询、预订修改、座位选择、退改签等
- **策略要求**：费用透明、客户优先级、安全规定遵循
- **典型任务**：航班改签、座位升级、行李问题处理

#### 零售场景 (Retail)
- **API工具**：商品查询、订单管理、库存检查、支付处理等
- **策略要求**：价格准确、库存实时性、客户满意度
- **典型任务**：商品推荐、订单跟踪、退换货处理

## 结果分析

### 输出示例

```text
+-----------+-----------+--------+---------+-------+---------+
| Model     | Dataset   | Metric | Subset  |   Num |   Score |
+===========+===========+========+=========+=======+=========+
| qwen-plus | tau_bench | Pass^1 | airline |    10 |   0.5   |
+-----------+-----------+--------+---------+-------+---------+
| qwen-plus | tau_bench | Pass^1 | retail  |    10 |   0.6   |
+-----------+-----------+--------+---------+-------+---------+
| qwen-plus | tau_bench | Pass^1 | OVERALL |    20 |   0.55  |
+-----------+-----------+--------+---------+-------+---------+
```

### 性能分析指标

**Pass^1**: 首次尝试成功完成任务的比例
- 衡量模型在单次对话中解决用户问题的能力
- 综合考虑任务完成度、策略遵循度和对话质量

