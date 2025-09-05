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
- **数据集描述**:  
  > EvalMuse Text-to-Image Benchmark. Used for evaluating the quality and semantic alignment of finely generated images
- **任务类别**: `TextToImage`
- **评估指标**: `FGA_BLIP2Score`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `EvalMuse`


---

### GEdit-Bench

[返回目录](#aigc评测集)
- **数据集名称**: `gedit`
- **数据集ID**: [stepfun-ai/GEdit-Bench](https://modelscope.cn/datasets/stepfun-ai/GEdit-Bench/summary)
- **数据集描述**:  
  > GEdit-Bench Image Editing Benchmark, grounded in real-world usages is developed to support more authentic and comprehensive evaluation of image editing models.
- **任务类别**: `ImageEditing`
- **评估指标**: `Perceptual Similarity`, `Semantic Consistency`
- **需要LLM Judge**: 是
- **默认提示方式**: 0-shot
- **数据集子集**: `background_change`, `color_alter`, `material_alter`, `motion_change`, `ps_human`, `style_change`, `subject-add`, `subject-remove`, `subject-replace`, `text_change`, `tone_transfer`

- **额外参数**: 
```json
{
    "language": "# language of the instruction, choose `en` or `cn`, default to `en`"
}
```

---

### GenAI-Bench

[返回目录](#aigc评测集)
- **数据集名称**: `genai_bench`
- **数据集ID**: [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/summary)
- **数据集描述**:  
  > GenAI-Bench Text-to-Image Benchmark. Includes 1600 prompts for text-to-image task.
- **任务类别**: `TextToImage`
- **评估指标**: `VQAScore`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `GenAI-Bench-1600`


---

### general_t2i

[返回目录](#aigc评测集)
- **数据集名称**: `general_t2i`
- **数据集ID**: general_t2i
- **数据集描述**:  
  > General Text-to-Image Benchmark
- **任务类别**: `Custom`, `TextToImage`
- **评估指标**: `PickScore`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `default`


---

### HPD-v2

[返回目录](#aigc评测集)
- **数据集名称**: `hpdv2`
- **数据集ID**: [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/summary)
- **数据集描述**:  
  > HPDv2 Text-to-Image Benchmark. Evaluation metrics based on human preferences, trained on the Human Preference Dataset (HPD v2)
- **任务类别**: `TextToImage`
- **评估指标**: `HPSv2.1Score`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `HPDv2`


---

### TIFA-160

[返回目录](#aigc评测集)
- **数据集名称**: `tifa160`
- **数据集ID**: [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/summary)
- **数据集描述**:  
  > TIFA-160 Text-to-Image Benchmark
- **任务类别**: `TextToImage`
- **评估指标**: `PickScore`
- **需要LLM Judge**: 否
- **默认提示方式**: 0-shot
- **数据集子集**: `TIFA-160`

