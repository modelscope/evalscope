# LLM Benchmarks

Below is the list of supported LLM benchmarks. Click on a benchmark name to jump to details.

| Benchmark Name | Pretty Name | Task Categories |
|------------|----------|----------|
| `evalmuse` | [evalmuse](#evalmuse) | `TextToImage` |
| `gedit` | [GEdit-Bench](#gedit-bench) | `ImageEditing` |
| `genai_bench` | [genai_bench](#genai_bench) | `TextToImage` |
| `general_t2i` | [general_t2i](#general_t2i) | `TextToImage` |
| `hpdv2` | [hpdv2](#hpdv2) | `TextToImage` |
| `tifa160` | [tifa160](#tifa160) | `TextToImage` |

---

## Benchmark Details

### evalmuse

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `evalmuse`
- **Dataset ID**: [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/summary)
- **Description**:  
  > EvalMuse Text-to-Image Benchmark
- **Task Categories**: `TextToImage`
- **Evaluation Metrics**: `FGA_BLIP2Score`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `EvalMuse`


---

### GEdit-Bench

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `gedit`
- **Dataset ID**: [stepfun-ai/GEdit-Bench](https://modelscope.cn/datasets/stepfun-ai/GEdit-Bench/summary)
- **Description**:  
  > GEdit-Bench Image Editing Benchmark, grounded in real-world usages is developed to support more authentic and comprehensive evaluation of image editing models.
- **Task Categories**: `ImageEditing`
- **Evaluation Metrics**: `Perceptual Similarity`, `Semantic Consistency`
- **Requires LLM Judge**: Yes
- **Default Shots**: 0-shot
- **Subsets**: `background_change`, `color_alter`, `material_alter`, `motion_change`, `ps_human`, `style_change`, `subject-add`, `subject-remove`, `subject-replace`, `text_change`, `tone_transfer`

- **Extra Parameters**: 
```json
{
    "language": "# language of the instruction, choose `en` or `cn`, default to `en`"
}
```

---

### genai_bench

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `genai_bench`
- **Dataset ID**: [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/summary)
- **Description**:  
  > GenAI-Bench Text-to-Image Benchmark
- **Task Categories**: `TextToImage`
- **Evaluation Metrics**: `VQAScore`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `GenAI-Bench-1600`


---

### general_t2i

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `general_t2i`
- **Dataset ID**: general_t2i
- **Description**:  
  > General Text-to-Image Benchmark
- **Task Categories**: `TextToImage`
- **Evaluation Metrics**: `PickScore`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `default`


---

### hpdv2

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `hpdv2`
- **Dataset ID**: [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/summary)
- **Description**:  
  > HPDv2 Text-to-Image Benchmark
- **Task Categories**: `TextToImage`
- **Evaluation Metrics**: `HPSv2.1Score`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `HPDv2`


---

### tifa160

[Back to Top](#llm-benchmarks)
- **Dataset Name**: `tifa160`
- **Dataset ID**: [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/summary)
- **Description**:  
  > TIFA-160 Text-to-Image Benchmark
- **Task Categories**: `TextToImage`
- **Evaluation Metrics**: `PickScore`
- **Requires LLM Judge**: No
- **Default Shots**: 0-shot
- **Subsets**: `TIFA-160`

