# SWE-bench

## 简介

[SWE-bench](https://www.swebench.com/) 是一个用于评估大型语言模型（LLMs）在软件工程任务中性能的基准测试套件。它包含各种编程挑战，涵盖从代码生成到调试和优化的多个方面。SWE-bench 旨在为研究人员和开发者提供一个标准化的平台，以比较不同 LLMs 在实际软件开发任务中的表现。使用EvalScope 集成 SWE-bench，可以方便地运行和评估各种 LLMs 在这些任务上的性能。

### 支持的评测数据集

以下数据集构成了SWE-bench的核心评估体系，用于测试AI系统自动解决真实GitHub问题的能力。它们通过Issue-Pull Request配对和单元测试验证来评估模型性能，从全面评估（Verified）到轻量测试（Mini）再到精选子集（Lite），满足不同场景的评估需求。

- **SWE-bench Verified (`swe_bench_verified`)**：从SWE-bench测试集中精选的500个人工验证样本，用于测试系统自动解决GitHub问题的能力，质量经过严格把关

- **SWE-bench Verified Mini (`swe_bench_verified_mini`)**：包含50个数据点的轻量级子集，存储需求仅5GB（相比原版130GB），保持与原始数据集相同的性能分布和难度特征

- **SWE-bench Lite (`swe_bench_lite`)**：包含300个来自11个流行Python项目的Issue-PR配对，通过单元测试验证评估，使用PR合并后行为作为参考解决方案

### 支持的推理数据集

以下数据集是SWE-bench的不同检索增强版本，专门为语言模型生成代码补丁而优化格式化。所有数据集都指导模型生成标准的patch格式输出（diff格式），可直接与SWE-bench推理脚本配合使用。主要区别在于代码上下文的检索方式和大小限制。

- **princeton-nlp/SWE-bench_bm25_13K**：使用Pyserini的BM25检索格式化数据集，代码上下文限制为13,000个tokens（cl100k_base），可直接用于LM生成patch格式文件

- **princeton-nlp/SWE-bench_bm25_27K**：使用Pyserini的BM25检索格式化数据集，代码上下文限制为27,000个tokens（cl100k_base），提供更大的上下文窗口用于生成patch文件

- **princeton-nlp/SWE-bench_bm25_40K**：使用Pyserini的BM25检索格式化数据集，代码上下文限制为40,000个tokens（cl100k_base），提供最大的上下文窗口支持复杂问题修复

- **princeton-nlp/SWE-bench_oracle**：使用"Oracle"检索设置格式化的数据集，提供理想化的检索结果作为上限基准，可直接用于生成标准patch格式文件

## 使用方法

### 安装依赖

SWE-bench 使用 Docker 来确保评估的可复现性。

1. **安装Docker**：请参考 [Docker安装指南](https://docs.docker.com/engine/install/) 在您的机器上安装Docker

2. **Linux用户额外配置**：如果您在Linux系统上配置，建议查看 [安装后配置步骤](https://docs.docker.com/engine/install/linux-postinstall/)，以获得更好的使用体验

3. 在运行评测之前需要安装以下依赖：

```bash
pip install evalscope
pip install swebench==4.1.0
```

**提示**：正确配置Docker是运行SWE-bench评估的前提条件，请确保安装完成后Docker服务正常运行。


### 运行评测

```{note}
首次运行swe_bench任务时，系统需要构建/下载必要的Docker镜像。这个过程对资源要求较高：

- **时间消耗**：对于完整的SWE-bench Verified数据集，构建时间可能长达数小时
- **存储需求**：需要预留约130GB的存储空间，对于轻量级的SWE-bench Verified Mini数据集，存储需求约为5GB

**建议**：请确保有足够的磁盘空间和时间预算，建议在首次运行时选择网络状况良好的环境。
```
运行下面的代码即可启动评测。下面以qwen-plus模型为例进行评测。


```python
import os
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='qwen-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type='openai_api',  # 使用API模型服务
    datasets=['swe_bench_verified'], # 选择评测数据集, 也可以选择'swe_bench_verified_mini'或'swe_bench_lite'
    dataset_args={
        'swe_bench_verified': {
            'extra_params': {
                'build_docker_images': True, # 是否构建评测所需的Docker镜像, 首次运行建议设置为True
                'pull_remote_images_if_available': True, # 如果远程有可用镜像则拉取, 建议设置为True
                'inference_dataset_id': 'princeton-nlp/SWE-bench_oracle' # 选择推理数据集, 可选 'princeton-nlp/SWE-bench_bm25_13K', 'princeton-nlp/SWE-bench_bm25_27K', 'princeton-nlp/SWE-bench_bm25_40K' 或 'princeton-nlp/SWE-bench_oracle'
            }
        }
    },
    eval_batch_size=5,  # 推理时的批处理大小
    judge_worker_num=5, # 并行评测任务的工作线程数, docker container数量
    limit=5,  # 限制评测数量，便于快速测试，正式评测时建议去掉此项
    generation_config={
        'temperature': 0.1,
    }
)
run_task(task_cfg=task_cfg)
```

输出结果示例：

评测中间结果将保存在`outputs/xxxxx/swebench_log`目录下，包含`patch.diff`等文件。最终的评测结果示例如下：

```text
+-----------+--------------------+----------+----------+-------+---------+---------+
| Model     | Dataset            | Metric   | Subset   |   Num |   Score | Cat.0   |
+===========+====================+==========+==========+=======+=========+=========+
| qwen-plus | swe_bench_verified | mean_acc | default  |     5 |     0.2 | default |
+-----------+--------------------+----------+----------+-------+---------+---------+ 
```