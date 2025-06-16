# AIGC Benchmarks

This framework also supports evaluation datasets related to text-to-image and other AIGC tasks. The specific datasets are as follows:

| Name          | Dataset ID       | Task Type       | Remarks                        |
|---------------|------------------|-----------------|--------------------------------|
| `general_t2i` |                  | General Text-to-Image | Refer to the tutorial          |
| `evalmuse`    | [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/summary) | Text-Image Consistency | EvalMuse subset, default metric is `FGA_BLIP2Score` |
| `genai_bench` | [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/) | Text-Image Consistency | GenAI-Bench-1600 subset, default metric is `VQAScore` |
| `hpdv2`       | [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/) | Text-Image Consistency | HPDv2 subset, default metric is `HPSv2.1Score` |
| `tifa160`     | [AI-ModelScope/T2V-Eval-Prompts](https://modelscope.cn/datasets/AI-ModelScope/T2V-Eval-Prompts/) | Text-Image Consistency | TIFA160 subset, default metric is `PickScore` |