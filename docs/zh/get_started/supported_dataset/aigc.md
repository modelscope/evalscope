# AIGC评测集

以下是支持的AIGC评测集列表，点击数据集标准名称可跳转详细信息。

| 数据集名称 | 标准名称 | 任务类别 |
|------------|----------|----------|
| `evalmuse` | [EvalMuse](#evalmuse) | `TextToImage` |
| `gedit` | [GEdit-Bench](#gedit-bench) | `ImageEditing` |
| `genai_bench` | [GenAI-Bench](#genai-bench) | `TextToImage` |
| `general_t2i` | [general_t2i](#general_t2i) | `Custom`, `TextToImage` |
| `hpdv2` | [HPD-v2](#hpd-v2) | `TextToImage` |
| `tifa160` | [TIFA-160](#tifa-160) | `TextToImage` |

---

## 数据集详情

### EvalMuse

[返回目录](#aigc评测集)
- **数据集名称**: `evalmuse`
- **数据集ID**: [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/summary)
- **数据集介绍**:
  > EvalMuse文本到图像基准，用于评估精细生成图像的质量和语义一致性
- **任务类别**: `TextToImage`
- **评估指标**: `FGA_BLIP2Score`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `test`
- **数据集子集**: `EvalMuse`


---

### GEdit-Bench

[返回目录](#aigc评测集)
- **数据集名称**: `gedit`
- **数据集ID**: [stepfun-ai/GEdit-Bench](https://modelscope.cn/datasets/stepfun-ai/GEdit-Bench/summary)
- **数据集介绍**:
  > GEdit-Bench 是基于真实使用场景构建的图像编辑基准，旨在支持对图像编辑模型进行更真实、全面的评估。
- **任务类别**: `ImageEditing`
- **评估指标**: `Perceptual Similarity`, `Semantic Consistency`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 是
- **默认提示方式**: 0-shot
- **评测数据集划分**: `train`
- **数据集子集**: `background_change`, `color_alter`, `material_alter`, `motion_change`, `ps_human`, `style_change`, `subject-add`, `subject-remove`, `subject-replace`, `text_change`, `tone_transfer`

- **额外参数**: 
```json
{
    "language": {
        "type": "str",
        "description": "Language of the instruction. Choices: ['en', 'cn'].",
        "value": "en",
        "choices": [
            "en",
            "cn"
        ]
    }
}
```

---

### GenAI-Bench

[返回目录](#aigc评测集)
- **数据集名称**: `genai_bench`
- **数据集ID**: [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/summary)
- **数据集介绍**:
  > GenAI-Bench 文本到图像基准，包含 1600 个文本到图像任务的提示。
- **任务类别**: `TextToImage`
- **评估指标**: `VQAScore`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `test`
- **数据集子集**: `GenAI-Bench-1600`


---

### general_t2i

[返回目录](#aigc评测集)
- **数据集名称**: `general_t2i`
- **数据集ID**: general_t2i
- **数据集介绍**:
  > 通用文生图基准测试
- **任务类别**: `Custom`, `TextToImage`
- **评估指标**: `PickScore`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `test`
- **数据集子集**: `default`


---

### HPD-v2

[返回目录](#aigc评测集)
- **数据集名称**: `hpdv2`
- **数据集ID**: [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/summary)
- **数据集介绍**:
  > HPDv2 文本到图像基准。基于人类偏好的评估指标，训练于人类偏好数据集（HPD v2）
- **任务类别**: `TextToImage`
- **评估指标**: `HPSv2.1Score`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `test`
- **数据集子集**: `HPDv2`


---

### TIFA-160

[返回目录](#aigc评测集)
- **数据集名称**: `tifa160`
- **数据集ID**: [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/summary)
- **数据集介绍**:
  > TIFA-160 文本到图像基准测试
- **任务类别**: `TextToImage`
- **评估指标**: `PickScore`
- **聚合方法**: `mean`
- **是否需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **评测数据集划分**: `test`
- **数据集子集**: `TIFA-160`

