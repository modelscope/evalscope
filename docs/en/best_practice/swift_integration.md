# ms-swift Integration

## Introduction

[ms-swift](https://github.com/modelscope/ms-swift)'s eval capability utilizes the [EvalScope evaluation framework](https://github.com/modelscope/eval-scope) from the ModelScope community and [Open-Compass](https://hub.opencompass.org.cn/home) and provides advanced encapsulation to support evaluation needs for various models. Currently, we support the evaluation process for **standard evaluation sets** and **user-defined evaluation sets**. The **standard evaluation sets** include:

NLP eval datasets：
```text
'obqa', 'AX_b', 'siqa', 'nq', 'mbpp', 'winogrande', 'mmlu', 'BoolQ', 'cluewsc', 'ocnli', 'lambada',
'CMRC', 'ceval', 'csl', 'cmnli', 'bbh', 'ReCoRD', 'math', 'humaneval', 'eprstmt', 'WSC', 'storycloze',
'MultiRC', 'RTE', 'chid', 'gsm8k', 'AX_g', 'bustm', 'afqmc', 'piqa', 'lcsts', 'strategyqa', 'Xsum', 'agieval',
'ocnli_fc', 'C3', 'tnews', 'race', 'triviaqa', 'CB', 'WiC', 'hellaswag', 'summedits', 'GaokaoBench',
'ARC_e', 'COPA', 'ARC_c', 'DRCD'
```
Check out the detail descriptions of these datasets: https://hub.opencompass.org.cn/home

Multi Modal eval datasets：
```text
'COCO_VAL', 'MME', 'HallusionBench', 'POPE', 'MMBench_DEV_EN', 'MMBench_TEST_EN', 'MMBench_DEV_CN', 'MMBench_TEST_CN',
'MMBench', 'MMBench_CN', 'MMBench_DEV_EN_V11', 'MMBench_TEST_EN_V11', 'MMBench_DEV_CN_V11',
'MMBench_TEST_CN_V11', 'MMBench_V11', 'MMBench_CN_V11', 'SEEDBench_IMG', 'SEEDBench2',
'SEEDBench2_Plus', 'ScienceQA_VAL', 'ScienceQA_TEST', 'MMT-Bench_ALL_MI', 'MMT-Bench_ALL',
'MMT-Bench_VAL_MI', 'MMT-Bench_VAL', 'AesBench_VAL', 'AesBench_TEST', 'CCBench', 'AI2D_TEST', 'MMStar',
'RealWorldQA', 'MLLMGuard_DS', 'BLINK', 'OCRVQA_TEST', 'OCRVQA_TESTCORE', 'TextVQA_VAL', 'DocVQA_VAL',
'DocVQA_TEST', 'InfoVQA_VAL', 'InfoVQA_TEST', 'ChartQA_TEST', 'MathVision', 'MathVision_MINI',
'MMMU_DEV_VAL', 'MMMU_TEST', 'OCRBench', 'MathVista_MINI', 'LLaVABench', 'MMVet', 'MTVQA_TEST',
'MMLongBench_DOC', 'VCR_EN_EASY_500', 'VCR_EN_EASY_100', 'VCR_EN_EASY_ALL', 'VCR_EN_HARD_500',
'VCR_EN_HARD_100', 'VCR_EN_HARD_ALL', 'VCR_ZH_EASY_500', 'VCR_ZH_EASY_100', 'VCR_ZH_EASY_ALL',
'VCR_ZH_HARD_500', 'VCR_ZH_HARD_100', 'VCR_ZH_HARD_ALL', 'MMDU', 'MMBench-Video', 'Video-MME',
'MMBench_DEV_EN', 'MMBench_TEST_EN', 'MMBench_DEV_CN', 'MMBench_TEST_CN', 'MMBench', 'MMBench_CN',
'MMBench_DEV_EN_V11', 'MMBench_TEST_EN_V11', 'MMBench_DEV_CN_V11', 'MMBench_TEST_CN_V11', 'MMBench_V11',
'MMBench_CN_V11', 'SEEDBench_IMG', 'SEEDBench2', 'SEEDBench2_Plus', 'ScienceQA_VAL', 'ScienceQA_TEST',
'MMT-Bench_ALL_MI', 'MMT-Bench_ALL', 'MMT-Bench_VAL_MI', 'MMT-Bench_VAL', 'AesBench_VAL',
'AesBench_TEST', 'CCBench', 'AI2D_TEST', 'MMStar', 'RealWorldQA', 'MLLMGuard_DS', 'BLINK'
```
Check out the detail descriptions of these datasets: https://github.com/open-compass/VLMEvalKit

```{tip}
At the first time of running eval, a resource dataset will be downloaded: [link](https://www.modelscope.cn/datasets/swift/evalscope_resource/resolve/master/eval.zip)

If downloading fails, you can manually download the dataset to your local disk, please pay attention to the log of the `eval` command.
```

## Environment Setup

```shell
pip install ms-swift[eval] -U
```

or install from source code:

```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[eval]'
```

## Evaluation
This section presents the evaluation of the original model and the LoRA fine-tuned qwen2-7b-instruct model. The evaluation supports accelerated inference using vLLM.

**Original Model Evaluation**
```shell
CUDA_VISIBLE_DEVICES=0 swift eval \
  --model_type qwen2-7b-instruct \
  --model_id_or_path /path/to/qwen2-7b-instruct \
  --eval_dataset ARC_e \
  --infer_backend vllm \ # Options: [pt, vllm, lmdeploy]
  --eval_backend Native  # Options: [Native, OpenCompass]
```

**LoRA Fine-Tuned Model Evaluation**
```shell
CUDA_VISIBLE_DEVICES=0 swift eval \
  --ckpt_dir qwen2-7b-instruct/vx-xxx/checkpoint-xxx \ # Path to LoRA weights
  --eval_dataset ARC_e \
  --infer_backend vllm \
  --eval_backend Native \
  --merge_lora true 
```
````{note}
For a list of evaluation parameters, refer to [eval parameters](https://swift.readthedocs.io/en/latest/LLM/Command-line-parameters.html#eval-parameters).

The evaluation results will be stored in `{--eval_output_dir}/{--name}/{timestamp}`, and if the user hasn't changed the storage configuration, the default path is:
```text
{current directory (`pwd` path)}/eval_outputs/default/20240628_190000/xxx
```
````

### Evaluation Using Deployment
```{note}
Specifying `eval_backend` as `OpenCompass` will automatically use the deployment method for evaluation. The following evaluation method applies to both the original and the LoRA fine-tuned models.
```

Launch the deployment using ms-swift:
```shell
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen2-7b-instruct
```

Launch the evaluation using ms-swift:
```shell
swift eval --eval_url http://127.0.0.1:8000/v1 --eval_dataset ARC_e
```

````{tip}
When using API for evaluation, if it's not deployed using swift, you need to additionally pass `--eval_is_chat_model` and `--model_type` parameters.

For example:
```shell
swift eval \
  --eval_url http://127.0.0.1:8000/v1 \
  --eval_dataset ARC_e \
  --eval_is_chat_model true \
  --model_type qwen2-7b-instruct
```
````


## Custom Evaluation Sets

In addition, we support users in customizing their own evaluation sets. Custom evaluation sets must be consistent with the data format (pattern) of an official evaluation set. Below, we explain step-by-step how to use your own evaluation set for evaluation.

### Preparing Your Own Evaluation Set

Currently, we support two patterns of evaluation sets: multiple-choice format (CEval) and question-answer format (General-QA).

#### Multiple-choice: CEval Format

The CEval format is suitable for multiple-choice scenarios, where you select the correct answer from four options, and the evaluation metric is `accuracy`. It is recommended to **directly modify** the [CEval scaffold directory](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/eval_example/custom_ceval). This directory contains two files:

```text
default_dev.csv # Used for few-shot evaluation, must contain at least the number of entries specified by eval_few_shot. If it is 0-shot evaluation, this CSV can be empty.
default_val.csv # Data used for actual evaluation.
```

The CEval CSV file should be in the following format:

```text
id,question,A,B,C,D,answer,explanation
1,Typically, how many amino acids make up animal proteins? ,4,22,20,19,C,1. Currently, it is known that 20 amino acids make up animal proteins.
2,Among the following substances present in blood, which one is not a metabolic end product? ,urea,uric acid,pyruvic acid,carbon dioxide,C,"A metabolic end product is a substance that cannot be further utilized in the body's metabolism and needs to be excreted. Pyruvic acid is a product of carbohydrate metabolism and can be further metabolized for energy or synthesis of other substances, so it is not a metabolic end product."
```

In this format, `id` is the evaluation sequence number, `question` is the question, `A`, `B`, `C`, `D` are options (if there are fewer than four options, leave the corresponding fields empty), `answer` is the correct option, and `explanation` is the explanation.

The `default` file name is the sub-dataset name for the CEval evaluation, which can be changed and will be used in the configuration below.

#### Question-Answer: General-QA

The General-QA format is suitable for question-answer scenarios, and the evaluation metrics are `rouge` and `bleu`. It is recommended to **directly modify** the [General-QA scaffold directory](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/eval_example/custom_general_qa). This directory contains one file:

```text
default.jsonl
```

This JSONL file should be in the following format:

```json
{"history": [], "query": "What is the capital of China?", "response": "The capital of China is Beijing."}
{"history": [], "query": "What is the highest mountain in the world?", "response": "It is Mount Everest."}
{"history": [], "query": "Why can't you see penguins in the Arctic?", "response": "Because most penguins live in the Antarctic."}
```

Note that `history` is currently a reserved field and is not yet supported.

### Defining a Configuration File for the Evaluation Command

After preparing the files above, you need to write a JSON file to pass into the evaluation command. It is recommended to directly modify the [official configuration scaffold file](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/eval_example/custom_config.json). The content of this file is as follows:

```json
[
    {
        "name": "custom_general_qa", // Name of the evaluation item, can be freely specified
        "pattern": "general_qa", // Pattern of this evaluation set
        "dataset": "eval_example/custom_general_qa", // Directory of this evaluation set, it is strongly recommended to use an absolute path to prevent read failures
        "subset_list": ["default"] // Sub-datasets to be evaluated, i.e., the `default_x` file name above
    },
    {
        "name": "custom_ceval",
        "pattern": "ceval",
        "dataset": "eval_example/custom_ceval", // Directory of this evaluation set, it is strongly recommended to use an absolute path to prevent read failures
        "subset_list": ["default"]
    }
]
```

You can then pass this configuration file for evaluation:

```shell
swift eval \
    --model_type "qwen-7b-chat" \
    --eval_dataset no \ # eval_dataset can also be set, running both official and custom datasets together
    --```shell
    --infer_backend pt \
    --custom_eval_config eval_example/custom_config.json
```

The results will be output as follows:

```text
2024-04-10 17:21:33,275 - llmuses - INFO - *** Report table ***
+------------------------------+----------------+---------------------------------+
| Model                        | custom_ceval   | custom_general_qa               |
+==============================+================+=================================+
| qa-custom_ceval_qwen-7b-chat | 1.0 (acc)      | 0.8888888888888888 (rouge-1-r)  |
|                              |                | 0.33607503607503614 (rouge-1-p) |
|                              |                | 0.40616618868713145 (rouge-1-f) |
|                              |                | 0.39999999999999997 (rouge-2-r) |
|                              |                | 0.27261904761904765 (rouge-2-p) |
|                              |                | 0.30722525589718247 (rouge-2-f) |
|                              |                | 0.8333333333333334 (rouge-l-r)  |
|                              |                | 0.30742204655248134 (rouge-l-p) |
|                              |                | 0.3586824745225346 (rouge-l-f)  |
|                              |                | 0.3122529644268775 (bleu-1)     |
|                              |                | 0.27156862745098037 (bleu-2)    |
|                              |                | 0.25 (bleu-3)                   |
|                              |                | 0.2222222222222222 (bleu-4)     |
+------------------------------+----------------+---------------------------------+
```
