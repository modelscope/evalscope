# AIGC Benchmarks

Below is the list of supported AIGC benchmarks. Click on a benchmark name to jump to details.

| Benchmark Name | Pretty Name | Task Categories |
|------------|----------|----------|
| `evalmuse` | [EvalMuse](#evalmuse) | `TextToImage` |
| `gedit` | [GEdit-Bench](#gedit-bench) | `ImageEditing` |
| `genai_bench` | [GenAI-Bench](#genai-bench) | `TextToImage` |
| `general_t2i` | [general_t2i](#general_t2i) | `Custom`, `TextToImage` |
| `hpdv2` | [HPD-v2](#hpd-v2) | `TextToImage` |
| `tifa160` | [TIFA-160](#tifa-160) | `TextToImage` |

---

## Benchmark Details

### EvalMuse

[Back to Top](#aigc-benchmarks)
- **Dataset Name**: `evalmuse`
- **Dataset ID**: [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/summary)
- **Description**:
  > EvalMuse Text-to-Image Benchmark. Used for evaluating the quality and semantic alignment of finely generated images
- **Task Categories**: `TextToImage`
- **Evaluation Metrics**: `FGA_BLIP2Score`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `EvalMuse`


---

### GEdit-Bench

[Back to Top](#aigc-benchmarks)
- **Dataset Name**: `gedit`
- **Dataset ID**: [stepfun-ai/GEdit-Bench](https://modelscope.cn/datasets/stepfun-ai/GEdit-Bench/summary)
- **Description**:
  > GEdit-Bench Image Editing Benchmark, grounded in real-world usages is developed to support more authentic and comprehensive evaluation of image editing models.
- **Task Categories**: `ImageEditing`
- **Evaluation Metrics**: `Perceptual Similarity`, `Semantic Consistency`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: Yes
- **Default Shots**: 0-shot
- **Evaluation Split**: `train`
- **Subsets**: `background_change`, `color_alter`, `material_alter`, `motion_change`, `ps_human`, `style_change`, `subject-add`, `subject-remove`, `subject-replace`, `text_change`, `tone_transfer`

- **Extra Parameters**: 
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

[Back to Top](#aigc-benchmarks)
- **Dataset Name**: `genai_bench`
- **Dataset ID**: [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/summary)
- **Description**:
  > GenAI-Bench Text-to-Image Benchmark. Includes 1600 prompts for text-to-image task.
- **Task Categories**: `TextToImage`
- **Evaluation Metrics**: `VQAScore`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `GenAI-Bench-1600`


---

### general_t2i

[Back to Top](#aigc-benchmarks)
- **Dataset Name**: `general_t2i`
- **Dataset ID**: general_t2i
- **Description**:
  > General Text-to-Image Benchmark
- **Task Categories**: `Custom`, `TextToImage`
- **Evaluation Metrics**: `PickScore`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `default`


---

### HPD-v2

[Back to Top](#aigc-benchmarks)
- **Dataset Name**: `hpdv2`
- **Dataset ID**: [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/summary)
- **Description**:
  > HPDv2 Text-to-Image Benchmark. Evaluation metrics based on human preferences, trained on the Human Preference Dataset (HPD v2)
- **Task Categories**: `TextToImage`
- **Evaluation Metrics**: `HPSv2.1Score`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `HPDv2`


---

### TIFA-160

[Back to Top](#aigc-benchmarks)
- **Dataset Name**: `tifa160`
- **Dataset ID**: [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/summary)
- **Description**:
  > TIFA-160 Text-to-Image Benchmark
- **Task Categories**: `TextToImage`
- **Evaluation Metrics**: `PickScore`
- **Aggregation Methods**: `mean`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Evaluation Split**: `test`
- **Subsets**: `TIFA-160`

