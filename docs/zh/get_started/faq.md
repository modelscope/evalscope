# ❓ 常见问题

下面是EvalScope使用过程中遇到的一些常见问题

```{important}
EvalScope使用中出现的大部分问题可能已经在最新版本中修复，建议先拉main分支代码安装再试试，看问题是否可以解决，请确保使用的是最新版本的代码。
```

## 模型benchmark测试

### Q0: 为什么评测的结果为0/评测结果明显不对？

A: 使用如下方法排查问题：
  1. 确认模型接口是否可以正常推理
  2. 在`outputs/2025xxxxx/predictions/`路径下查看模型的输出，确认模型是否有输出，输出是否正常
  3. 使用`evalscope app`启动可视化界面，查看评测结果是否正常

### Q1: 为什么推理模型测出来的精度很低，例如QwQ模型在ifeval上?

A: `--datast-args` 的 ifeval加上 `"filters": {"remove_until": "</think>"}'`，去掉模型的思考过程。

### Q2: 使用API模型服务评测embeddings报错 openai.BadRequestError: Error code: 400 - {'object': 'error', 'message': 'dimensions is currently not supported', 'type': 'BadRequestError', 'param': None, 'code': 400}

A: 设置`'dimensions': None`，或者不设置该参数

### Q3: 查看outputs/2025xxxxx/predictions/路径下面模型的输出最后几个case的内容为null

A: 输出长度不够，被提前截断了

### Q4: evalscope 当前内置的评测集（例如 LiveCodebench、AIME、MATH-500）等只支持 pass1 评测吗？支持 passk 评测吗？

A: 
1. 本框架支持QwQ评测中的`n_sample`参数，在generation config中设置`n`可计算多个sample的平均指标，参考：https://evalscope.readthedocs.io/zh-cn/latest/best_practice/eval_qwq.html#id5
2. 本框架支持 `pass@k` 指标，参考 https://evalscope.readthedocs.io/zh-cn/latest/get_started/parameters.html#id3 中的`metrics_list`

### Q5: 数据集从本地加载报错，缺少`dtype` ?

A: 数据集本地加载是有点问题，需要modelscope下个版本修复，临时的解决方案是先手动删掉数据集目录下的`dataset_infos.json` 这个文件

### Q6: 评估Qwen2-audio的时候，跑了几个文本指标，回复的内容全是感叹号

A: 参考复现代码：
```python
from evalscope.run import run_task

task_cfg = {
    'model': '/opt/nas/n/AudioLLM/allkinds_ckpt/Qwen/Qwen2-Audio-7B-Instruct',
    'datasets': ['gsm8k', 'math_500', "gpqa", "mmlu_pro", "mmlu_redux"],
    'limit': 100
}

run_task(task_cfg=task_cfg)
```
目前对于本地加载的多模态模型支持并不完善，建议使用vllm等推理服务拉起api来评测

### Q7: 评测多模态大模型时报错：Unknown benchmark

A: 多模态评测参考[这里](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/vlmevalkit_backend.html#vlmevalkit) ，需要使用VLMEval 工具

### Q8: 评估Gemma3系列模型时出现RuntimeError: CUDA error: device-side assert triggered错误

A: gemma3是多模态模型，目前框架的chat_adapter对于多模态模型的支持不是很完善，建议使用模型推理框架（vllm等）拉起模型服务来进行评测

### Q9: 如何进行多卡评估？

A: 目前暂不支持data parallel的加速方式

### Q10: 模型推理服务的压测使用可视化工具找不到报告

A: 该可视化工具专门用于展示模型评测结果，不适用于模型推理服务的压测结果可视化。如需查看模型推理服务的压测结果可视化，请参考[压测结果可视化指南](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/examples.html#wandb)。

### Q11: 是否有可用的docker？

A: 使用镜像可以查看[这里](https://modelscope.cn/docs/intro/environment-setup#%E6%9C%80%E6%96%B0%E9%95%9C%E5%83%8F)，使用modelscope的官方镜像，里面包含了evalscope库

### Q12: ifeval数据集做评测的时候报-Unable to detect language for text कामाची घाई

A: 报错信息包含：
due to Need to load profiles.
NotADirectoryError: [Errno 20] Not a directory: '/nltk_data/tokenizers/punkt_tab.zip/punkt_tab/english/collocations.tab'

解决方案：
1. `unzip /path/to/nltk_data/tokenizers/punkt_tab.zip`
2. 命令如下
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

### Q13: Math-500数据集评估结果误判badcase集合

A: 这是数学解析规则出问题了，写这些匹配规则比较复杂，case也很难覆盖完全。
可以设置judge model，用LLM做召回，能减少误判，如下：
```python
judge_strategy=JudgeStrategy.LLM_RECALL,
judge_model_args={
    'model_id': 'qwen2.5-72b-instruct',
    'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    'api_key': os.getenv('DASHSCOPE_API_KEY'),
}
```
参考：[参数说明](https://evalscope.readthedocs.io/zh-cn/latest/get_started/parameters.html#judge), [使用示例](https://evalscope.readthedocs.io/zh-cn/latest/get_started/basic_usage.html#id9)

### Q14：使用qwen2.5-72b-instruct分割solution，图中===表示分隔出的不同solution，该prompt无法约束模型正确分隔

A: 这个prompt：
https://github.com/modelscope/evalscope/blob/595ac60f22b1248d5333a27ffd4b9eeae7f57727/evalscope/third_party/thinkbench/resources/reformat_template.txt

这个prompt是用来分割step的，不是划分sub-solution的，你可以调整prompt来划分sub-solution

### Q15: 在评测service时，默认temperature是多少？

A: 默认是0

### Q16: 在AIME24上进行评测的时候结果不准或者不稳定怎么办？

A: aime 默认的指标是 pass@1，采样得够多才估计得更准，可以设置 n 为较大的值，也可以设置temperature 和 seed，让模型的输出尽量一致

### Q17: 评测结果可视化的gradio程序，离线部署后无法工作(无公网)

A: 可以参考这里的解决方法 [gradio-app/gradio#7934](https://github.com/gradio-app/gradio/issues/7934)

### Q18: 多模态自定义问答题格式不支持裁判么？

A: 自定义问答题需要自己实现judge的逻辑

### Q19: 运行aime 2024 评估, 报SSLError错误

A: 报错示例：
```text
requests.exceptions.SSLError: HTTPSConnectionPool(host='www.modelscope.cn', port=443): Max retries exceeded with url: /api/v1/datasets/HuggingFaceH4/aime_2024 (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1007)')))
```

报错原因是data-args写的不对，应该是这样：
```python
dataset_args={
    'aime24': {
        'local_path': "/var/AIME_2024/",
        'few_shot_num': 3
    }
},
```

### Q20: 数据集评测时如何设置一个样本推理几次生成几个答案？

A: 在generation config里面指定

参考：https://evalscope.readthedocs.io/zh-cn/latest/get_started/parameters.html#id2

### Q21: modelscope - WARNING - Use trust_remote_code=True. Will invoke codes from ceval-exam. Please make sure that you can trust the external codes. 这个警告是怎么回事？trust_remote_code=True这个参数要怎么传递？

A: 这个是warning不影响评测流程，框架已经默认`trust_remote_code=True`

### Q22: api评测的时候使用base模型超过最大token报错怎么办？

A: api评测走的是 `chat` 接口，base模型评测可能会有点问题（模型输出不会停止），建议用Instruct模型来测试

### Q23: 用vllm起服务总是报几次retrying request问题后，就开始报 Error when calling OpenAI API: Request timed out.

A: 模型输出比较长，尝试加上`stream`参数, `timeout`加大

### Q24: 请问如何评测多模态模型（如Qwen-2.5-vl）在语言模型评测数据集（如MMLU）上的性能？

A: 多模态模型建议用vllm这种框架拉起服务再评测，目前还没支持多模态模型本地加载

参考：https://evalscope.readthedocs.io/zh-cn/latest/get_started/basic_usage.html#api

### Q25: stream参数报错 EvalScope Command Line tool: error: unrecognized arguments: --stream True

A: 直接用 `--stream` 不要填加 `True` 

### Q26： 执行示例报错 RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking argument for argument index in method wrapper_CUDA__index_select)

A: 首先确认显存是否足够，由于默认的device map是auto，可能有些权重被分到cpu上了，可以尝试增加`--model-args device_map=cuda `

### Q27： 对于r1类的模型，评估过程会忽略思考部分，直接对生成的最终结果进行评测吗？还是将思考过程和结果一起作为答案进行评测？

A: 目前并没有对`<think>`内容做额外的处理，默认是`<think>`和`<answer>`放在一起的，然后从里面解析答案来评测。已经支持后处理过滤器, 建议对推理模型过滤掉思考部分。
使用示例：
```shell
--datasets ifeval
--dataset-args '{"ifeval": {"filters": {"remove_until": "</think>"}}'
```

### Q28： 可视化界面图表展示异常

A: plotly 尝试降到 5.23.0 版本

### Q29: 现阶段是否有直接基于预测结果进行评估的入口

A: 参考这个 https://evalscope.readthedocs.io/zh-cn/latest/get_started/parameters.html#id5 ，设置use_cache参数

## 模型压测

### Q1: 测试ollama发现，当并发数大于5后，Throughput(average tokens/s)的值始终上不去，我的显卡 cpu 内存 io都不存在瓶颈，是怎么回事？

A: 参考复现代码：
```shell
ollama run deepseek-r1:7b

evalscope perf --url http://127.0.0.1:11434/v1/chat/completions --parallel 20 --model deepseek-r1:7b --number 50 --api openai --dataset longalpaca --stream --tokenizer-path /home/data/DeepSeek-R1-Distill-Qwen-7B/
```

加一个 export OLLAMA_NUM_PARALLEL=10

### Q2：无法使用--min-tokens 2048 --max-tokens 2048 \控制输出的长度

A: `--min-tokens` 不是所有模型服务都支持该参数，请查看对应API服务的文档。

- 解释：对应API服务的文档指的是测试的模型服务的文档，就是谁提供的API服务，可能是推理引擎拉起的服务，也可能是云服务商提供的服务。

### Q3: 速度基准测试脚本运行报错 

A: 参考报错信息
```text
2025-03-31 08:56:52,172 - evalscope - http_client.py - on_request_chunk_sent - 125 - DEBUG - Request sent: <method='POST', url=URL('https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'), truncated_chunk='{"prompt": "熵", "model": "qwen2.5-72b-instruct", "max_tokens": 2048, "min_tokens": 2048, "seed": 42, "stop": [], "stop_token_ids": []}'>
2025-03-31 08:56:52,226 - evalscope - http_client.py - on_response_chunk_received - 137 - DEBUG - Request received: <method='POST', url=URL('https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'), truncated_chunk='{"error":{"code":"missing_required_parameter","param":"message","message":"you must provide a messages parameter","type":"invalid_request_error"},"request_id":"chatcmpl-816a021e-5d7e-9eff-91a2-36aed4641546"}'>
```
参考复现代码
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
速度测试`--url`需要使用`/v1/completions`端点，而不是`/v1/chat/completions`，避免chat template的额外处理对输入长度有影响。

### Q4: perf压测支持自定义解析返回体吗？

A: 请参考文档：https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/custom.html#api

### Q5: 调整那个参数可以加大并发处理吗

A: 你可以参考一下这个：[vllm-project/vllm#3561](https://github.com/vllm-project/vllm/issues/3561)

### Q6: 带stream的执行命令，但是在128并发的情况下，他会等同一个批次的并发全部执行完后再进行第二个128并发的请求， 而不带stream的时候会完成一个进去一个新的请求，导致在stream情况下最后的到的吞吐量会低很多

A: 参考示例代码：
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
降低并发再尝试一下

### Q7: TTFT测试结果不对劲，我完成50个请求的总时间才30多秒，TTFT也是30多秒，什么情况

A: 要准确统计Time to First Token (TTFT)指标，需要在请求中包含--stream参数，否则TTFT将与Latency相同。

### Q8: 如何测试自定义API模型（非openai、vllm服务），应该修改哪些地方，有哪些参数是必需的？

A: 
1. 模型性能测试的话，只要是兼容OpenAI API格式的服务都支持
2. 模型推理服务压测的话，参考[自定义请求API](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/custom.html#api)

现在已支持--no-test-connection参数，可以跳过链接测试

### Q9: 为什么输出的ttft时间与vllm收集的ttft时间相差较大

A: evalscope得到的TTFT是end-to-end的时间，从请求发出开始计时，到接受到第一个token结束，中间有网络传输和处理时间，跟服务端统计结果可能有些偏差

### Q10: 如果请求超时了，可以设置更长的timeout参数嘛？

A: 可以，添加下面的参数即可
```shell
 --connect-timeout 60000 \
 --read-timeout 60000 \
```

### Q11: 测试模型服务的推理速度的示例中，model怎么理解？

A: `model`填的是模型服务框架部署的模型名称，比如OpenAI的服务有`gpt-4o`, `o1-mini`等模型

### Q12: KTransformers 流输出无法识别报错ZeroDivisionError: float division by zero

A: 部署的模型服务似乎没有返回使用信息，这与标准的 OpenAI API 格式不同，需要传递 `--tokenizer-path` 参数来计算 `token` 数量。
