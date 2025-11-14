# SWE-bench

## Introduction

[SWE-bench](https://www.swebench.com/) is a benchmark suite for evaluating the performance of Large Language Models (LLMs) on software engineering tasks. It contains various programming challenges covering multiple aspects from code generation to debugging and optimization. SWE-bench aims to provide researchers and developers with a standardized platform to compare the performance of different LLMs on practical software development tasks. Using EvalScope's integration with SWE-bench, you can conveniently run and evaluate the performance of various LLMs on these tasks.

### Supported Evaluation Datasets

The following datasets form the core evaluation system of SWE-bench, designed to test AI systems' ability to automatically resolve real GitHub issues. They evaluate model performance through Issue-Pull Request pairing and unit test verification, ranging from comprehensive evaluation (Verified) to lightweight testing (Mini) to curated subsets (Lite), meeting evaluation needs for different scenarios.

- **SWE-bench Verified (`swe_bench_verified`)**: 500 manually verified samples selected from the SWE-bench test set, used to test systems' ability to automatically resolve GitHub issues, with strictly controlled quality

- **SWE-bench Verified Mini (`swe_bench_verified_mini`)**: A lightweight subset containing 50 data points, requiring only 5GB storage (compared to the original 130GB), maintaining the same performance distribution and difficulty characteristics as the original dataset

- **SWE-bench Lite (`swe_bench_lite`)**: Contains 300 Issue-PR pairs from 11 popular Python projects, evaluated through unit test verification, using post-PR merge behavior as the reference solution

### Supported Inference Datasets

The following datasets are different retrieval-augmented versions of SWE-bench, specifically formatted and optimized for language models to generate code patches. All datasets guide models to generate standard patch format output (diff format), which can be directly used with SWE-bench inference scripts. The main differences lie in the retrieval method and size limitations of code context.

- **princeton-nlp/SWE-bench_bm25_13K**: Dataset formatted using Pyserini's BM25 retrieval, with code context limited to 13,000 tokens (cl100k_base), can be directly used for LM to generate patch format files

- **princeton-nlp/SWE-bench_bm25_27K**: Dataset formatted using Pyserini's BM25 retrieval, with code context limited to 27,000 tokens (cl100k_base), providing a larger context window for generating patch files

- **princeton-nlp/SWE-bench_bm25_40K**: Dataset formatted using Pyserini's BM25 retrieval, with code context limited to 40,000 tokens (cl100k_base), providing the largest context window to support complex issue fixes

- **princeton-nlp/SWE-bench_oracle**: Dataset formatted using the "Oracle" retrieval setting, providing idealized retrieval results as an upper bound baseline, can be directly used to generate standard patch format files

## Usage

### Install Dependencies

SWE-bench uses Docker to ensure evaluation reproducibility.

1. **Install Docker**: Please refer to the [Docker Installation Guide](https://docs.docker.com/engine/install/) to install Docker on your machine

2. **Additional Configuration for Linux Users**: If you are configuring on a Linux system, it is recommended to check the [Post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/) for a better user experience

3. Install the following dependencies before running evaluation:

```bash
pip install evalscope
pip install swebench==4.1.0
```

**Tip**: Properly configuring Docker is a prerequisite for running SWE-bench evaluations. Please ensure the Docker service is running normally after installation.


### Run Evaluation

```{note}
When running swe_bench tasks for the first time, the system needs to build/download necessary Docker images. This process has high resource requirements:

- **Time Consumption**: For the complete SWE-bench Verified dataset, build time may take several hours
- **Storage Requirements**: Approximately 130GB of storage space is needed. For the lightweight SWE-bench Verified Mini dataset, storage requirements are about 5GB

**Recommendation**: Please ensure sufficient disk space and time budget. It is recommended to choose an environment with good network conditions for the first run.
```
Run the following code to start evaluation. Below is an example using the qwen-plus model for evaluation.


```python
import os
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='qwen-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type='openai_api',  # Use API model service
    datasets=['swe_bench_verified'], # Select evaluation dataset, can also choose 'swe_bench_verified_mini' or 'swe_bench_lite'
    dataset_args={
        'swe_bench_verified': {
            'extra_params': {
                'build_docker_images': True, # Whether to build Docker images required for evaluation, recommended to set to True for first run
                'pull_remote_images_if_available': True, # Pull remote images if available, recommended to set to True
                'inference_dataset_id': 'princeton-nlp/SWE-bench_oracle' # Select inference dataset, options: 'princeton-nlp/SWE-bench_bm25_13K', 'princeton-nlp/SWE-bench_bm25_27K', 'princeton-nlp/SWE-bench_bm25_40K' or 'princeton-nlp/SWE-bench_oracle'
            }
        }
    },
    eval_batch_size=5,  # Batch size for inference
    judge_worker_num=5, # Number of worker threads for parallel evaluation tasks, number of docker containers
    limit=5,  # Limit evaluation quantity for quick testing, recommended to remove this for formal evaluation
    generation_config={
        'temperature': 0.1,
    }
)
run_task(task_cfg=task_cfg)
```

Example output:

Intermediate evaluation results will be saved in the `outputs/xxxxx/swebench_log` directory, including files such as `patch.diff`. The final evaluation result example is as follows:

```text
+-----------+--------------------+----------+----------+-------+---------+---------+
| Model     | Dataset            | Metric   | Subset   |   Num |   Score | Cat.0   |
+===========+====================+==========+==========+=======+=========+=========+
| qwen-plus | swe_bench_verified | mean_acc | default  |     5 |     0.2 | default |
+-----------+--------------------+----------+----------+-------+---------+---------+ 
```