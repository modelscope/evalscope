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