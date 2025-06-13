# AIGC评测集

本框架也支持文生图等AIGC相关的评测集，具体数据集如下：

| 名称   | 数据集ID        | 任务类别         | 备注      |
|-------|----------------|-----------------|----------|
| `general_t2i`   |        | 通用文生图         |  参考教程         |
| `evalmuse`   |   [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/summary)     | 图文一致性         |  EvalMuse 子数据集，默认指标为`FGA_BLIP2Score`          |
| `genai_bench`   |   [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/)     | 图文一致性         |  GenAI-Bench-1600 子数据集，默认指标为`VQAScore`          |
| `hpdv2`   |   [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/)     | 图文一致性         |  HPDv2 子数据集，默认指标为`HPSv2.1Score`          |
| `tifa160`   |   [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/)     | 图文一致性         |  TIFA160 子数据集，默认指标为`PickScore`         |

