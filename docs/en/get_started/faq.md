# ❓ FAQ

Below are some common issues encountered during the use of EvalScope.

```{important}
Most issues with using EvalScope may have been fixed in the latest version. It is recommended to first pull the code from the main branch and try installing it again to see if the issue can be resolved. Please ensure you are using the latest version of the code.
```

## Model Benchmark Testing

### Q0: Why are the evaluation results 0 or significantly incorrect?

A: Use the following methods to troubleshoot the problem:
  1. Confirm whether the model interface can perform inference normally.
  2. Check the model’s output in the `outputs/2025xxxxx/predictions/` path and confirm whether the model has output and whether the output is normal.
  3. Start the visualization interface with `evalscope app` to check if the evaluation results are normal.

### Q1: Why is the accuracy measured by the inference model very low, such as the QwQ model on ifeval?

A: Add `"filters": {"remove_until": "</think>"}` to the ifeval in `--datast-args` to remove the model's thinking process.

### Q2: When using the API model service to evaluate embeddings, an error occurs: openai.BadRequestError: Error code: 400 - {'object': 'error', 'message': 'dimensions is currently not supported', 'type': 'BadRequestError', 'param': None, 'code': 400}

A: Set `'dimensions': None` or do not set this parameter.

### Q3: In the outputs/2025xxxxx/predictions/ path, the content of the last few cases of the model output is null.

A: The output length is insufficient and was truncated prematurely.

### Q4: Does the current built-in evaluation set of evalscope (such as LiveCodebench, AIME, MATH-500) only support pass1 evaluation? Does it support passk evaluation?

A: 
1. This framework supports the `n_sample` parameter in QwQ evaluation. You can set `n` in the generation config to calculate the average metrics of multiple samples. Refer to: https://evalscope.readthedocs.io/zh-cn/latest/best_practice/eval_qwq.html#id5
2. This framework supports the `pass@k` metric. Refer to https://evalscope.readthedocs.io/zh-cn/latest/get_started/parameters.html#id3 in the `metrics_list`.

### Q5: An error occurred when loading the dataset from the local path, missing `dtype`.

A: There is an issue with loading local datasets, which will be fixed in the next version of modelscope. The temporary solution is to manually delete the `dataset_infos.json` file in the dataset directory.

### Q6: When evaluating Qwen2-audio, after running several text metrics, the reply content is all exclamation marks.

A: Refer to the reproduction code:
```python
from evalscope.run import run_task

task_cfg = {
    'model': '/opt/nas/n/AudioLLM/allkinds_ckpt/Qwen/Qwen2-Audio-7B-Instruct',
    'datasets': ['gsm8k', 'math_500', "gpqa", "mmlu_pro", "mmlu_redux"],
    'limit': 100
}

run_task(task_cfg=task_cfg)
```
Currently, support for locally loaded multimodal models is not very comprehensive. It is recommended to use an inference service such as vllm to pull up the api for evaluation.

### Q7: Error when evaluating large multimodal models: Unknown benchmark.

A: Refer to [here](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/vlmevalkit_backend.html#vlmevalkit) for multimodal evaluation. You need to use the VLMEval tool.

### Q8: When evaluating Gemma3 series models, a RuntimeError: CUDA error: device-side assert triggered occurs.

A: Gemma3 is a multimodal model. The current chat_adapter of the framework does not support multimodal models well. It is recommended to use a model inference framework (such as vllm) to pull up the model service for evaluation.

### Q9: How to perform multi-card evaluation?

A: Currently, data parallel acceleration is not supported.

### Q10: The visualization tool for the model inference service's stress test cannot find the report.

A: This visualization tool is specifically for displaying model evaluation results and is not suitable for visualizing stress test results of model inference services. For visualizing stress test results of model inference services, refer to the [stress test result visualization guide](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/examples.html#wandb).

### Q11: Is there an available docker?

A: You can view the [latest image](https://modelscope.cn/docs/intro/environment-setup#%E6%9C%80%E6%96%B0%E9%95%9C%E5%83%8F) using modelscope's official image, which includes the evalscope library.

### Q12: When evaluating the ifeval dataset, an error occurs: Unable to detect language for text कामाची घाई.

A: The error message contains:
due to Need to load profiles.
NotADirectoryError: [Errno 20] Not a directory: '/nltk_data/tokenizers/punkt_tab.zip/punkt_tab/english/collocations.tab'

Solution:
1. `unzip /path/to/nltk_data/tokenizers/punkt_tab.zip`
2. Command as follows
```shell
!evalscope eval
--model xxxx
--api-url xxxx
--api-key xxxxx
--generation-config temperature=1.0
--eval-type service
--eval-batch-size 50
--datasets ifeval
--judge-worker-num 1
```

### Q13: Incorrect bad case set when evaluating the Math-500 dataset.

A: The mathematical parsing rules have issues, and writing these matching rules is quite complex, making it difficult to cover all cases. You can set a judge model and use LLM for recall, which can reduce misjudgments, as follows:
```python
judge_strategy=JudgeStrategy.LLM_RECALL,
judge_model_args={
    'model_id': 'qwen2.5-72b-instruct',
    'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    'api_key': os.getenv('DASHSCOPE_API_KEY'),
}
```
Refer to: [Parameter Explanation](https://evalscope.readthedocs.io/zh-cn/latest/get_started/parameters.html#judge), [Usage Example](https://evalscope.readthedocs.io/zh-cn/latest/get_started/basic_usage.html#id9)

### Q14: Using qwen2.5-72b-instruct to segment solution, the === in the figure indicates different solutions separated out. This prompt cannot constrain the model to segment correctly.

A: This prompt:
https://github.com/modelscope/evalscope/blob/595ac60f22b1248d5333a27ffd4b9eeae7f57727/evalscope/third_party/thinkbench/resources/reformat_template.txt

This prompt is used to segment steps, not to divide sub-solutions. You can adjust the prompt to divide sub-solutions.

### Q15: What is the default temperature when evaluating service?

A: The default is 0.

### Q16: What should I do if the results are inaccurate or unstable when evaluating on AIME24?

A: The default metric for AIME is pass@1, and it is estimated to be more accurate with sufficient samples. You can set n to a larger value, or set temperature and seed to make the model output as consistent as possible.

### Q17: The gradio program for visualizing evaluation results does not work offline (without public network).

A: You can refer to the solution here [gradio-app/gradio#7934](https://github.com/gradio-app/gradio/issues/7934).

### Q18: Does the multimodal custom Q&A format not support judges?

A: Custom Q&A requires implementing the judge logic yourself.

### Q19: Running the aime 2024 evaluation reports an SSLError.

A: Example of the error:
```text
requests.exceptions.SSLError: HTTPSConnectionPool(host='www.modelscope.cn', port=443): Max retries exceeded with url: /api/v1/datasets/HuggingFaceH4/aime_2024 (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1007)')))
```

The error reason is that the data-args is written incorrectly, and it should be like this:
```python
dataset_args={
    'aime24': {
        'local_path': "/var/AIME_2024/",
        'few_shot_num': 3
    }
},
```

### Q20: How to set the number of times a sample is inferred to generate several answers during dataset evaluation?

A: Specify in the generation config.

Refer to: https://evalscope.readthedocs.io/zh-cn/latest/get_started/parameters.html#id2

### Q21: What is the warning about modelscope - WARNING - Use trust_remote_code=True. Will invoke codes from ceval-exam. Please make sure that you can trust the external codes. How can the trust_remote_code=True parameter be passed?

A: This is a warning and does not affect the evaluation process. The framework already defaults to `trust_remote_code=True`.

### Q22: What should I do if a base model exceeds the maximum token during api evaluation and reports an error?

A: The api evaluation uses the `chat` interface. Evaluating the base model may have some problems (the model output will not stop), and it is recommended to use the Instruct model for testing.

### Q23: When starting a service with vllm, it repeatedly reports retrying request issues and then starts reporting Error when calling OpenAI API: Request timed out.

A: The model output is relatively long. Try adding the `stream` parameter and increasing `timeout`.

### Q24: How to evaluate the performance of multimodal models (such as Qwen-2.5-vl) on language model evaluation datasets (such as MMLU)?

A: It is recommended to use vllm and other frameworks to pull up services for evaluation of multimodal models. Local loading of multimodal models is not yet supported.

Refer to: https://evalscope.readthedocs.io/zh-cn/latest/get_started/basic_usage.html#api

### Q25: The stream parameter reports an error: EvalScope Command Line tool: error: unrecognized arguments: --stream True.

A: Use `--stream` directly without adding `True`.

### Q26: An error occurs when executing an example: RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking argument for argument index in method wrapper_CUDA__index_select).

A: First confirm whether the video memory is sufficient. Since the default device map is auto, some weights may be allocated to the cpu. You can try adding `--model-args device_map=cuda`.

### Q27: For r1-type models, does the evaluation process ignore the thinking part and directly evaluate the generated final result, or does it evaluate the answer with the thinking process and result together?

A: Currently, no additional processing is done on `<think>` content. By default, `<think>` and `<answer>` are placed together, and the answer is parsed from there for evaluation. Post-processing filters are supported, and it is recommended to filter out the thinking part for inference models.
Usage example:
```shell
--datasets ifeval
--dataset-args '{"ifeval": {"filters": {"remove_until": "</think>"}}'
```

### Q28: Abnormal chart display in the visualization interface.

A: Try downgrading plotly to version 5.23.0.

### Q29: Is there currently an entry for evaluating directly based on prediction results?

A: Refer to this: https://evalscope.readthedocs.io/zh-cn/latest/get_started/parameters.html#id5, set the use_cache parameter.

### Q30: The evaluation was interrupted, how can I resume it (checkpoint evaluation)?

A: It is supported. Please use the `use_cache` parameter to pass in the path of the previous evaluation output to reuse the model's prediction results and review outcomes.


### Q31: Unable to access evalscope app via `http://localhost:port` or `http://ip:port`

A: Upgrade gradio to version 5.4.0 to resolve this issue.

### Q32: Why are the IFEval benchmark metrics in Evalscope far worse than those reported in the technical report? For example, Qwen3's IFEval strict prompt is 39.9 in the report, but even after removing chain-of-thought, I only get 23.5.

A: See the best practices here: https://evalscope.readthedocs.io/zh-cn/latest/best_practice/qwen3.html

You need to set the generation config accordingly.

### Q33: I set `"remove_until": ""` to remove chain-of-thought, but the saved JSON file still contains "和" in the response.

A: This does not affect the results; post-processing will be performed when calculating metrics.

### Q34: If data parallelism is not supported, is model parallelism supported?

A: For model parallelism, you need to launch the model service yourself, e.g., use vLLM with the `--tp` parameter.

### Q35: I used evalscope in my work. How should I cite it?

A: Please use the following format:
```
@misc{evalscope_2024,
    title={{EvalScope}: Evaluation Framework for Large Models},
    author={ModelScope Team},
    year={2024},
    url={https://github.com/modelscope/evalscope}
}
```

### Q36: During bfcl_v3 testing, I encounter an error due to excessive length, even after setting `os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'`.

A: Set this environment variable when starting the vLLM model service.
You can also set `ignore_errors=True` in the evaluation command to skip problematic samples.

### Q37: Environment conflicts when installing both evalscope[app]==0.16.3 and bfcl-eval.

A: Please install bfcl-eval version `2025.6.16`.

Try installing the two packages separately, not together.

### Q38: Error compiling dependencies when creating a py310 environment with conda and running `pip install evalscope[all]`.

A: First run `pip install dotenv`, then proceed with installation. This usually resolves the issue.

### Q39: Does VLMEvalKit backend support streaming outputs for VLM models? The stream parameter doesn't work.

A: Streaming is not supported yet.

### Q40: The `--max-prompt-length` parameter is inaccurate; prompts sometimes exceed the set value. Is input-token length not being truncated?

A: See this explanation: https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/examples.html#random

The number of tokens may change after encoding and decoding random token IDs.

### Q41: Are all benchmarks and models supported by VLMEvalkit also supported by evalscope?

A: Most are supported. See the supported dataset list: https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/other/vlmevalkit.html

### Q42: How can I customize the prompt content in JSON format?

A: For customizing multimodal model tests, refer to the custom dataset documentation: https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/custom.html#id3

### Q43: When evaluating MATH500 using llm judge, the pred field contains the full answer instead of the extracted answer, unlike rule-based evaluation.

A: When using llm judge, answers are judged directly by the model without rule-based extraction.

### Q44: In inference speed tests, the TTFT (Time To First Token) metric under high concurrency seems unreasonable. Single concurrency TTFT is ~0.2s, but with 10 concurrency it's up to 15s, and with 100 concurrency, some reach 200+s. Is this accurate? How is TTFT calculated in evalscope?

A: When the model service can't handle many requests at once, requests queue up, causing TTFT to increase.

### Q45: How can I use a local model as the judge model?

A: You can use vLLM to launch a local model service as the judge model.

### Q46: Can the database include time info for each request? I'd like to output the request_id for slow requests for easier review.

A: The database has the following fields you can use:
```
'''CREATE TABLE IF NOT EXISTS result(
          request TEXT,
          start_time REAL,
          chunk_times TEXT,
          success INTEGER,
          response_messages TEXT,
          completed_time REAL,
          latency REAL,
          first_chunk_latency REAL,
          n_chunks INTEGER,
          chunk_time REAL,
          prompt_tokens INTEGER,
          completion_tokens INTEGER,
          max_gpu_memory_cost REAL)'''
```

### Q47: Can I set a timeout for the judge model during review?

A: Set the timeout parameter in generation_config of `--judge-model-args`.

### Q48: How to support custom model API formats for non-OpenAI style models?

A: See this [tutorial](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_model.html) for implementing a custom model evaluation interface.

### Q49: When adding a custom bench using llm as judge and implementing llm_match, how should the match function be implemented?

A: If you don't need rule-based matching, just put `pass` in that function and specify `--judge-strategy=llm` during evaluation.

See the custom benchmark doc: https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/add_benchmark.html

### Q50: Is modifying the evalscope library files directly in the Python environment the only way?

A: Please install from source as described here: https://evalscope.readthedocs.io/zh-cn/latest/get_started/installation.html#id2

The key command is:
```
pip install -e xxx
```
which lets you debug the code.

### Q51: Why do I keep seeing {'num_prompt_tokens': 0, 'num_generated_tokens': 0, 'num_samples': 0, 'runtime': 20.00112988, 'samples/s': 0.0, 'tokens/s': 0.0}?

A: It's just a log message and can be ignored.

### Q52: Evaluating Qwen2.5-Coder-1.5B on humaneval yields a score of 0. The model repeats itself a lot. The Instruct version scores 0.4695, still much lower than the official report's 0.7.

A: Per the model [description](https://modelscope.cn/models/Qwen/Qwen2.5-Coder-1.5B):
> We do not recommend using base language models for conversations. Instead, you can apply post-training, e.g., SFT, RLHF, continued pretraining, etc., or fill in the middle tasks on this model.

Evalscope evaluates in chat mode, so use the instruct model for testing. Also, try other generation configs. The default max output length is 512, which may be insufficient.

### Q53: What is the principle behind the evaluation metric, and how is score calculated?

A: See this document for the overall process: https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/add_benchmark.html#id1

For detailed calculation, see the source code: https://github.com/modelscope/evalscope/blob/main/evalscope/metrics/named_metrics.py

### Q54: How do I set a system prompt for performance evaluation?

A: See this guide: https://evalscope.readthedocs.io/zh-cn/latest/get_started/parameters.html#id3

### Q55: evalscope - ERROR - Error when calling remote API: Connection error

A: alpaca_eval requires you to specify a judge model. The default uses OpenAI's model. If not set up, you'll get this error. See: https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset.html#llm

### Q56: simpleQA: Downloaded dataset locally, reviewing is slow after inference, using evalscope master version from source.

A: Set the judge model.

See: https://evalscope.readthedocs.io/zh-cn/latest/get_started/parameters.html#judge

### Q57: What do -f, -p, -r mean in ROUGE evaluation? Is there documentation?

A: Not in the docs, but these are common metrics:

- **-r (Recall):** Measures how much of the reference text is covered by the generated text (coverage). E.g., $R_{lcs} = \frac{LCS(ref, pred)}{m}$, $m$=reference length.
- **-p (Precision):** Measures accuracy—how much of the generated text matches the reference. E.g., $P_{lcs} = \frac{LCS(ref, pred)}{n}$, $n$=generated length.
- **-f (F-measure):** The harmonic mean of recall and precision; balances both, with $\beta$ controlling the weight.

Summary:
- -r: coverage (recall)
- -p: accuracy (precision)
- -f: balance. Adjust weight as needed.

Use depends on task. For MT or summarization, recall (-r) is often more important.

### Q58: RuntimeError: Cannot run the event loop while another loop is running

A: Don't run in a notebook environment. Use a Python script and run it from the terminal.

### Q59: Using Qwen2.5-0.5B-Instruct, evalscope speed benchmark (local vLLM inference) gives "Cannot connect to host 127.0.0.1:8877 ssl:default" error

A: Just wait a bit; the local vLLM service will start automatically.

### Q60: How to evaluate multimodal large models?

A: See: https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/parameters.html#id5

Currently supports the flickr8k multimodal dataset.

### Q61: Does evaluating via custom model API support SGLang?

A: Evalscope is compatible with OpenAI API-style interfaces, so services like SGLang, VLLM, OLLAMA, or cloud APIs are supported.

### Q62: gsm8k base model accuracy is low, and there is repetition

A: Try increasing the temperature.

### Q63: Evaluation of code generation datasets gives 0 result

A: See best practices: https://evalscope.readthedocs.io/zh-cn/latest/best_practice/eval_qwq.html#id5

Try setting `"filters": {"remove_until": "</think>"}`.

Also, use the use_cache parameter to reuse generated results for evaluation.

### Q64: With identical settings (fewshot, token length, sampling, etc.), can evalscope and opencompass yield aligned metrics? Is the evaluation code for the same datasets identical? Noticing discrepancies in results.

A: Evaluation code and metrics differ across frameworks; it's hard to align. Please compare metrics within the same framework.

### Q65: How can I evaluate qwen2.5-omni on DOCVQA test?

A: Use the VLMEvalKit backend, launch a model inference service, and follow this guide: https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/vlmevalkit_backend.html#id4

VLMEvalKit doesn't support direct qwen2.5-omni inference, but you can use vLLM to serve the model and specify the API URL for evaluation.

### Q66: Why is Total requests not equal to Expected number of requests?

A: The speed_benchmark only issues 8 requests, but the default Expected number of requests is 100.

### Q67: Does it support direct evaluation on NPU multi-card setups?

A: 
- You can use vLLM to serve the model API and specify tp.
- For local model loading, specify `device_map=auto` in --model-args to automatically distribute the model.

### Q68: Why do I get different results for inference on the same machine?

A: Set a seed in the 'models' parameter, e.g.:
```
'models': [
{'path': '/data00/models/deepseek-r1-w4a8',
'openai_api_base': 'http://221.194.152.47:8000/v1/chat/completions',
'is_chat': True,
'batch_size': 100,
'seed': 42
},
```

### Q69: In Native mode, extracting answers for MATH_500 is inaccurate when there are multiple answers or matrices.

A: See the solution: https://evalscope.readthedocs.io/zh-cn/latest/get_started/faq.html#q13-math-500badcase

### Q70: Suppose I have an R1-like thinking model that uses `<|beginning_of_thinking|> <|end_of_thinking|>` tags instead of `<think></think>`. Can I use evalscope as is, or do I need to adapt code? What should I modify?

A: See: https://evalscope.readthedocs.io/zh-cn/latest/get_started/faq.html#q1-qwqifeval
Replace `</think>` with `<|end_of_thinking|>`.

### Q71: Does /generate endpoint testing supported?

A: Currently, only OpenAI API-compatible endpoints are supported; /generate is not.

### Q72: How to include System Prompt in VLMEvalKit backend?

A: Add it in the model parameter:
```
'model': [{'api_base': 'http://localhost:12345/v1/chat/completions',
'key': 'token-abc123',
...
'system_prompt': 'xxx'
}
```

### Q73: Datasets downloaded from modelscope lack dtype in dataset_infos.json. Loading many datasets locally requires specifying dataset_id down to the jsonl file. For mmlu, need to specify data/test directory. Can't just use the dataset name.

A: Known issue. As a temporary fix, delete the dataset_infos.json file from the dataset.

### Q74: Can you provide the source code for ms-opencompass and ms-vlmeval?

A: Both are forks with modifications:
- ms-vlmeval: https://github.com/Yunnglin/VLMEvalKit/tree/eval_scope
- ms-opencompass: https://github.com/wangxingjun778/opencompass

### Q75: Configuring `--dataset-hub "local" --dataset-dir ~/.cache/modelscope/hub/datasets` still triggers online download

A: See the tutorial: https://evalscope.readthedocs.io/zh-cn/latest/get_started/basic_usage.html#id13
Command parameters are incorrect.

### Q76: For realwordqa evaluation, is the output the accuracy of selecting abcd options?

A: Yes, it is the option accuracy. See: https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset.html#id3

MCQ tasks use option accuracy.

### Q77: QwQ IFEval accuracy in evalscope does not align with official results. On A800, QwQ-32B gets 51 on IFEval in evalscope, but 82 with QwQ's official scripts. Any special settings needed?

A: In datast-args, add `"filters": {"remove_until": "</think>"}` for ifeval.

### Q78: API model evaluation of embeddings gives openai.BadRequestError: Error code: 400 - {'object': 'error', 'message': 'dimensions is currently not supported', ...}

A: Set `'dimensions': None` and try again.

### Q79: In the dataset overview in evalscope app, only one score is shown per model; but in dataset details, each metric is shown. In multi-model comparison, only BLEU-1 is shown. Is BLEU-1 prioritized or is the first metric used?

A: Multi-model comparison shows the first metric of the dataset by default.

### Q80: Why is concurrent performance low with ollama?

A: Try setting `export OLLAMA_NUM_PARALLEL=10`.

### Q81: For a custom multiple-choice test set, how should the answer field be written?

A: See: https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/llm.html#mcq

### Q82: When using Huawei MindIE as backend, --min-tokens and --max-tokens do not control output length

A: `--min-tokens` is not supported by all model services. Check your API service documentation.

### Q83: Built-in evalscope benchmarks (e.g. LiveCodebench, AIME, MATH-500) only support pass@1, different from mainstream community practice (e.g. QwQ).

A: 
1. The framework supports the n_sample parameter from QwQ evaluation; set n in generation config for average over multiple samples, see: https://evalscope.readthedocs.io/zh-cn/latest/best_practice/eval_qwq.html#id5
2. The framework supports pass@k metrics; see https://evalscope.readthedocs.io/zh-cn/latest/get_started/parameters.html#id3 metrics_list

### Q84: evalscope Live_code_bench hangs during review phase

A: Set `judge_worker_num=1`.

### Q85: Evaluating Qwen2-audio, replies contain only exclamation marks for text metrics

A: Local multimodal model support is limited; use vllm or other inference services with APIs for evaluation.

### Q86: Speed benchmark script returns error when using /v1/chat/completions

A: For speed testing, --url should use the /v1/completions endpoint, not /v1/chat/completions, to avoid chat template processing affecting input length.

### Q87: Should the stream statistics include reasoning_content?

A: reasoning_content is part of the model output but does not affect inference speed. The final completion token length includes reasoning.

### Q88: Error "Unknown benchmark" when evaluating multimodal large models

A: See [here](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/vlmevalkit_backend.html#vlmevalkit) for multimodal evaluation; you need to use the VLMEval tool.

### Q89: Evaluating Gemma3 series gives RuntimeError: CUDA error: device-side assert triggered

A: Gemma3 is a multimodal model; current chat_adapter support for multimodal models is limited. Use an inference framework (e.g., vllm) to launch an API for evaluation.


## Model Stress Testing

### Q1: When testing ollama, I found that when the concurrency is greater than 5, the Throughput (average tokens/s) value always does not go up. My graphics card, cpu, memory, and io have no bottlenecks. What is the problem?

A: Refer to the reproduction code:
```shell
ollama run deepseek-r1:7b

evalscope perf --url http://127.0.0.1:11434/v1/chat/completions --parallel 20 --model deepseek-r1:7b --number 50 --api openai --dataset longalpaca --stream --tokenizer-path /home/data/DeepSeek-R1-Distill-Qwen-7B/
```

Add an export OLLAMA_NUM_PARALLEL=10.

### Q2: Unable to use --min-tokens 2048 --max-tokens 2048 \ to control the output length.

A: `--min-tokens` is not supported by all model services. Please check the documentation of the corresponding API service.

- Explanation: The corresponding API service documentation refers to the documentation of the model service being tested, whether it is provided by an inference engine service or a cloud service provider.

### Q3: An error occurs when running the speed benchmark script.

A: Refer to the error message
```text
2025-03-31 08:56:52,172 - evalscope - http_client.py - on_request_chunk_sent - 125 - DEBUG - Request sent: <method='POST', url=URL('https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'), truncated_chunk='{"prompt": "熵", "model": "qwen2.5-72b-instruct", "max_tokens": 2048, "min_tokens": 2048, "seed": 42, "stop": [], "stop_token_ids": []}'>
2025-03-31 08:56:52,226 - evalscope - http_client.py - on_response_chunk_received - 137 - DEBUG - Request received: <method='POST', url=URL('https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'), truncated_chunk='{"error":{"code":"missing_required_parameter","param":"message","message":"you must provide a messages parameter","type":"invalid_request_error"},"request_id":"chatcmpl-816a021e-5d7e-9eff-91a2-36aed4641546"}'>
```
Refer to the reproduction code
```shell
evalscope perf
--parallel 1
--url 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
--model 'qwen2.5-72b-instruct'
--log-every-n-query 5
--connect-timeout 6000
--read-timeout 6000
--max-tokens 2048
--min-tokens 2048
--api openai
--api-key 'sk-xxxxxx'
--dataset speed_benchmark
--debug
```
For speed testing, the `--url` needs to use the `/v1/completions` endpoint instead of the `/v1/chat/completions`, to avoid the extra handling of the chat template affecting the input length.

### Q4: Does perf stress testing support custom parsing of the return body?

A: Refer to the documentation: https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/custom.html#api

### Q5: Which parameter can be adjusted to increase concurrent processing?

A: You can refer to this: [vllm-project/vllm#3561](https://github.com/vllm-project/vllm/issues/3561).

### Q6: When executing with the stream, but under 128 concurrency, it waits for the entire batch of concurrency to finish before proceeding with the next 128 concurrent requests, while without the stream it completes one and enters a new request. This results in much lower throughput with the stream.

A: Refer to the example code:
```shell
evalscope perf --url 'http://127.0.0.1:8000/v1/chat/completions'
--parallel 128
--model 'test'
--log-every-n-query 10
--read-timeout=1200
--dataset-path '/model/open_qa.jsonl'
-n 1000
--max-prompt-length 32000
--api openai
--stop '<|im_end|>'
--dataset openqa
```
Reduce concurrency and try again.

### Q7: TTFT test results seem incorrect, as the total time for completing 50 requests is only 30 seconds, and TTFT is also 30 seconds. What is going on?

A: To accurately measure the Time to First Token (TTFT) metric, the request must include the --stream parameter; otherwise, TTFT will be the same as Latency.

### Q8: How to test a custom API model (not openai or vllm service), and which aspects should be modified, what parameters are required?

A: 
1. For model performance testing, any service compatible with OpenAI API format is supported.
2. For model inference service stress testing, refer to [custom request API](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/custom.html#api).

The --no-test-connection parameter is now supported to skip connection testing.

### Q9: Why does the ttft time output differ significantly from the ttft time collected by vllm?

A: The TTFT obtained by evalscope is the end-to-end time, starting from when the request is sent and ending when the first token is received. It includes network transmission and processing time, which may differ from the service-side statistics.

### Q10: If the request times out, can a longer timeout parameter be set?

A: Yes, just add the following parameters:
```shell
--connect-timeout 60000 \
--read-timeout 60000 \
```

### Q11: In the example of testing the inference speed of model services, how is the model understood?

A: The `model` is the name of the model deployed by the model service framework, such as `gpt-4o`, `o1-mini`, etc.

### Q12: KTransformers stream output cannot be recognized and reports ZeroDivisionError: float division by zero.

A: The deployed model service seems not to return usage information, which is different from the standard OpenAI API format. It requires the `--tokenizer-path` parameter to calculate the number of `tokens`.

### Q13: How can I perform stress testing on a multimodal large model, and how do I input images?

A: Currently, setting the dataset to flickr8k is supported for stress testing of multimodal models. Please [refer to](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/parameters.html#id5) for more information.

Here are the English translations for Q14–Q25:



### Q14: Running `evalscope perf --dataset aime25` results in KeyError: 'aime25'

A: According to the documentation, the datasets supported by the perf command are listed here:  
https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/parameters.html#dataset-configuration

"aime25" is a dataset supported by the eval command, not by perf.



### Q15: Unable to select reports in Chrome

A: The visualization method used is incorrect.

See this documentation: https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/quick_start.html#id6



### Q16: During model inference performance testing, why is TTFT (Time To First Token) smaller with more requests? Under the same input token count, output token count, and concurrency, larger number of requests shows smaller TTFT.

A: It’s recommended to test again using the "random" dataset, which allows you to fix the input prompt length, e.g., to 100.

See usage documentation: https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/examples.html#random



### Q17: During stress testing, errors occur. Using evalscope 0.16 or 0.14 to stress test the deepseek-r1-32b model deployed on Ascend mindie with 10 concurrency & 100 requests works, but with 20 concurrency & 200 requests errors occur midway. The issue happens whether evalscope is installed on the Ascend server or elsewhere. Testing the same model on RTX-4090 with vLLM has no issues.

A: Add the `--tokenizer-path` parameter.



### Q18: evalscope perf inference stress test results in nan

A: First, check if the provided data format is correct. For "openqa", a JSONL file with the "question" field is used as the prompt. If `dataset_path` is not specified, the dataset will be automatically downloaded from ModelScope.



### Q19: Is there no app visualization for stress testing?

A: Stress testing does not support app visualization, but you can use wandb and swanlab for visualization. See https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/quick_start.html#id6



### Q20: I want to use the eval command to call my local HTTPS service and evaluate my own dataset. The request needs to include a token in the header. With the perf command I can use `--header` like `--headers param_name=mytoken`. Can I do similar with eval?

A: Set `{"extra_headers": {"key": "value"}}` in the generation_config.



### Q21: How to use a local dataset with evalscope perf?

A: See here: https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/parameters.html#id5

You need to set the dataset to `line_by_line` and provide the `dataset_path`. Each line of the txt file will be used as a prompt.



### Q22: evalscope - WARNING - Retrying... <404> {"detail": "Not Found"} perf port cannot connect

A: You need to deploy the model service before starting the stress test.



### Q23: The speed_benchmark code checks if the interface is "v1/chat/completion", causing errors when stress testing v1/chat/completion

A: speed_benchmark only supports the v1/completions endpoint.

https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/speed_benchmark.html#api



### Q24: When running evalscope perf on a model served by ollama, setting --max-tokens results in outputs with fewer than 4096 tokens

A: Ollama’s service likely does not support the max-tokens parameter.



### Q25: How to stress test the qwen2.5vl-3b model deployed locally with vLLM using perf?

A: Refer to: https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/parameters.html#id5

Set the dataset to `flickr8k`.