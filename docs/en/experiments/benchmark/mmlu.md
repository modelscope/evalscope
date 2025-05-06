# MMLU

> This is a large-scale multi-task assessment comprised of multiple-choice questions from various knowledge domains. The test covers humanities, social sciences, hard sciences, and other significant areas of study, encompassing 57 tasks, including basic mathematics, American history, computer science, law, among others. To achieve a high accuracy rate on this test, models must possess a broad knowledge of the world and problem-solving abilities. [Dataset Link](https://modelscope.cn/datasets/modelscope/mmlu/summary)

## Experimental Setup

- Split: test
- Total number: 13985
- 0-shot

## Experimental Results

| Model                                                                                            | Revision | Precision | Humanities  | STEM       | Social Science | Other   | Weighted Avg | Target      | Delta  |
|--------------------------------------------------------------------------------------------------|----------|-----------|-------------|------------|----------------|---------|--------------|-------------|--------|
| [Baichuan2-7B-Base](https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Base/summary)         | v1.0.2   | fp16      | 0.4111      | 0.3807     | 0.5233         | 0.504   | 0.4506       | -           |        |
| [Baichuan2-7B-Chat](https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-chat/summary)         | v1.0.4   | fp16      | 0.4439      | 0.374      | 0.5524         | 0.5458  | 0.4762       | -           |        |
| [chatglm2-6b](https://modelscope.cn/models/ZhipuAI/chatglm2-6b/summary)                          | v1.0.12  | fp16      | 0.3834      | 0.3413     | 0.4708         | 0.4445  | 0.4077       | 0.4546 (CoT) | -4.69% |
| [chatglm3-6b-base](https://modelscope.cn/models/ZhipuAI/chatglm3-6b-base/summary)                | v1.0.1   | fp16      | 0.5435      | 0.5087     | 0.7227         | 0.6471  | 0.5992       | 0.614       | -1.48% |
| [internlm-chat-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-7b/summary) | v1.0.1   | fp16      | 0.4005      | 0.3547     | 0.4953         | 0.4796  | 0.4297       | -           |        |
| [Llama-2-13b-ms](https://modelscope.cn/models/modelscope/Llama-2-13b-ms/summary)                 | v1.0.2   | fp16      | 0.4371      | 0.3887     | 0.5579         | 0.5437  | 0.4778       | -           |        |
| [Llama-2-7b-ms](https://modelscope.cn/models/modelscope/Llama-2-7b-ms/summary)                   | v1.0.2   | fp16      | 0.3146      | 0.3037     | 0.4134         | 0.3885  | 0.3509       | -           |        |
| [Qwen-14B-Chat](https://modelscope.cn/models/qwen/Qwen-14B-Chat/summary)                         | v1.0.6   | bf16      | 0.5326      | 0.5397     | 0.7184         | 0.6859  | 0.6102       | -           |        |
| [Qwen-7B](https://modelscope.cn/models/qwen/Qwen-7B/summary)                                     | v1.1.6   | bf16      | 0.387       | 0.4        | 0.5403         | 0.5139  | 0.4527       | -           |        |
| [Qwen-7B-Chat-Int8](https://modelscope.cn/models/qwen/Qwen-7B-Chat-Int8/summary)                 | v1.1.6   | int8      | 0.4322      | 0.4277     | 0.6088         | 0.5778  | 0.5035       | -           |        |

- Target -- The official declared score of the model on the dataset
- Delta -- The difference between the weighted average score and the target score

### Settings: (Split: test, Total number: 13985, 5-shot)

| Model               | Revision | Precision | Humanities | STEM   | Social Science | Other  | Weighted Avg | Avg    | Target             | Delta   |
|---------------------|----------|-----------|------------|--------|----------------|--------|--------------|--------|--------------------|---------|
| Baichuan2-7B-Base   | v1.0.2   | fp16      | 0.4295     | 0.398  | 0.5736         | 0.5325 | 0.4781       | 0.4918 | 0.5416 (official)  | -4.98%  |
| Baichuan2-7B-Chat   | v1.0.4   | fp16      | 0.4344     | 0.3937 | 0.5814         | 0.5462 | 0.4837       | 0.5029 | 0.5293 (official)  | -2.64%  |
| chatglm2-6b         | v1.0.12  | fp16      | 0.3941     | 0.376  | 0.4897         | 0.4706 | 0.4288       | 0.4442 | -                  | -       |
| chatglm3-6b-base    | v1.0.1   | fp16      | 0.5356     | 0.4847 | 0.7175         | 0.6273 | 0.5857       | 0.5995 | -                  | -       |
| internlm-chat-7b    | v1.0.1   | fp16      | 0.4171     | 0.3903 | 0.5772         | 0.5493 | 0.4769       | 0.4876 | -                  | -       |
| Llama-2-13b-ms      | v1.0.2   | fp16      | 0.484      | 0.4133 | 0.6157         | 0.5809 | 0.5201       | 0.5327 | 0.548 (official)   | -1.53%  |
| Llama-2-7b-ms       | v1.0.2   | fp16      | 0.3747     | 0.3363 | 0.4372         | 0.4514 | 0.3979       | 0.4089 | 0.453 (official)   | -4.41%  |
| Qwen-14B-Chat       | v1.0.6   | bf16      | 0.574      | 0.553  | 0.7403         | 0.684  | 0.6313       | 0.6414 | 0.646 (official)   | -0.46%  |
| Qwen-7B             | v1.1.6   | bf16      | 0.4587     | 0.426  | 0.6078         | 0.5629 | 0.5084       | 0.5151 | 0.567 (official)   | -5.2%   |
| Qwen-7B-Chat-Int8   | v1.1.6   | int8      | 0.4697     | 0.4383 | 0.6284         | 0.5967 | 0.5271       | 0.5347 | 0.554 (official)   | -1.93%  |
