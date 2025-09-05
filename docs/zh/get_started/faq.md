# ❓ 常见问题

下面是EvalScope使用过程中遇到的一些常见问题

```{important}
EvalScope使用中出现的大部分问题可能已经在最新版本中修复，建议先拉main分支代码安装再试试，看问题是否可以解决，请确保使用的是最新版本的代码。
```

## 评测结果明显异常的排查方法


1. 确认模型接口是否可以正常推理。
2. 在`outputs/2025xxxxx/predictions/`路径下查看模型的输出，确认模型是否有输出，输出是否正常。
3. 使用`evalscope app`启动可视化界面，查看评测结果是否正常。


## 模型benchmark测试

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

### Q30: 评测中断了，如何继续评测（断点续评）？

A: 支持的，请使用`use_cache`参数定传入上次评测输出的路径即可重用模型预测结果以及review结果。

### Q31: evalscope app 使用 `http://localhost:port` 和 `http://ip:port` 都无法访问

A: 升级gradio版本到5.4.0后即可解决。

### Q32: 使用Evalscope进行评测时，使用IFEval基准评测，指标为啥与技术报告上的指标相差很远？例如在qwen3的技术报告上IFEval strict prompt是39.9，但是我已经去掉思维链，指标才是23.5

A: 这里有最佳实践：https://evalscope.readthedocs.io/zh-cn/latest/best_practice/qwen3.html

需要对照着设置generation config

### Q33: "remove_until": "" 已经移除了思维链，但是为什么保存的json文件中回复仍然带着“和”

A: 这个不影响，计算指标时会进行后处理

### Q34: 如果不支持数据并行，那是否可以支持模型并行的方式呢？

A: 模型并行需要你自行拉起模型服务，例如使用vLLM并设置 `--tp` 参数

### Q35: 我在我的工作中使用了evalscope，我该如何引用它？

A: 可以使用下面的格式来引用：
```
@misc{evalscope_2024,
    title={{EvalScope}: Evaluation Framework for Large Models},
    author={ModelScope Team},
    year={2024},
    url={https://github.com/modelscope/evalscope}
}
```

### Q36: bfcl_v3测试过程中，长度太长，报错, 设置了os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

A: 这个环境变量需要在启动vLLM模型服务的时候配置，
你也可以在评测命令中设置ignore_errors=True，来跳过报错的样例

### Q37: evalscope[app]==0.16.3和bfcl-eval安装环境冲突

A: 请安装bfcl-eval `2025.6.16` 版本

并且尝试分别安装这两个库，不一起安装

### Q38: conda 创建py310环境，pip install evalscope[all] 安装依赖包时编译出现错误，未安装成功

A: 先安装`pip install dotenv`，再执行全部安装，这通常可以解决问题。

### Q39: VLMEvalKit 后端处理VLM模型输出时，不支持流式么，stream参数不起效

A: 暂不支持

### Q40: --max-prompt-length 参数实际不精确，明明指定了--max-prompt-length，但是实际触发会超出设定值，没有对input-token做长度截断么？

A: 这里对这个问题有说明：https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/examples.html#random

random的token ids经过encode再decode，token数量会不一致

### Q41: 是不是VLMEvalkit支持的benchmark和模型，evalscope都是支持的？

A: 基本都支持，这里有支持的数据集列表：https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/other/vlmevalkit.html

### Q42: 如何自定义prompt内容，json格式

A: 如果想自定义测试多模态模型，建议参考自定义数据集的写法：https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/custom.html#id3

### Q43: 评测MATH500时采用llm judge遇到问题, 如果用规则的话，pred字段里是可以解析出答案, 但如果用llm judge，那pred字段中是评测模型的完整回答，就像没有解析答案这个步骤一样

A: 用llm judge就不会做规则解析，让模型去直接判断答案是否正确

### Q44: 模型推理性能测试时，多并发下的TTFT（首token时长）指标不合理, 单并发下，首token时间是0.2s左右，但是在多并发情况下，首token时间会变得很长，比如10并发下，有的请求首token时间就达到了15s左右，100并发下首token时间有的达到了200+s。这个值是不是不太对呀？evalscope在多并发下是怎么计算TTFT的呢？

A: 模型服务无法同时处理这么多请求，请求就会排队，导致TTFT变长

### Q45: 如何使用本地模型作为裁判模型

A: 可以用vllm拉起本地模型服务作为裁判模型

### Q46: db里面是否能支持加入每个请求的时间信息？希望做到大于某个时间就输出该请求的request_id方便查看

A: db中有如下字段，可以参考使用：

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

### Q47: 裁判模型做review的时候可以限制超时时间吗？

A: --judge-model-args的generation_config中设置timeout参数

### Q48: 怎么支持自定义模型接口格式以兼容非OpenAI风格模型

A: 参考这个[教程](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_model.html)，实现自定义的模型评测接口

### Q49: 增加自定义的bench，使用llm作为judge，已经实现llm_match的逻辑，请问match函数该如何实现

A: 如果不需要基于规则的match，直接在这个函数下面写pass即可，评测时指定`--judge-strategy`为`llm`

自定义benchmark的文档：https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/add_benchmark.html

### Q50: 是否只能直接修改 Python 环境中安装的 evalscope 库文件？

A: 请使用源码方式安装：https://evalscope.readthedocs.io/zh-cn/latest/get_started/installation.html#id2

关键是下面的命令，可以调试代码：

```
pip install -e xxx
```

### Q51: 总是遇到{'num_prompt_tokens': 0, 'num_generated_tokens': 0, 'num_samples': 0, 'runtime': 20.00112988, 'samples/s': 0.0, 'tokens/s': 0.0}是什么原因？

A: 这是打印的log，可以忽略

### Q52: 评估Qwen2.5-Coder-1.5B在humaneval上score为0，发现存在大量复读机的情况，用Qwen2.5-Coder-1.5B-Instruct测试，score为0.4695，仍远低于官方报告中的0.7

A: 参考模型 https://modelscope.cn/models/Qwen/Qwen2.5-Coder-1.5B 描述：
> We do not recommend using base language models for conversations. Instead, you can apply post-training, e.g., SFT, RLHF, continued pretraining, etc., or fill in the middle tasks on this model.

evalscope框架使用chat方式进行评测，因此请使用instruct模型来进行测试。此外，请尝试其他的generation config，默认输出最大长度为512，可能不够

### Q53: 评估指标原理，score的计算原理是什么？

A: 可以通过这个文档来了解一下整个流程：https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/add_benchmark.html#id1

具体的score分数计算建议参考源代码 https://github.com/modelscope/evalscope/blob/main/evalscope/metrics/named_metrics.py

### Q54: 性能评测如何设置system prompt呢

A:  请参考这个文档设置：https://evalscope.readthedocs.io/zh-cn/latest/get_started/parameters.html#id3

### Q55: evalscope - ERROR - Error when calling remote API: Connection error

A: alpaca_eval 需要指定Judge Model进行评测，默认模型调用的是openai的模型，没有配置的话就会报错。这个文档里有说明：https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/index.html#llm

### Q56: simpleQA, 将数据集下载到本地，推理结束后的reviewing很慢，使用的evalscope版本是目前的master源码安装

A: 设置judge model

参考这个文档：https://evalscope.readthedocs.io/zh-cn/latest/get_started/parameters.html#judge

### Q57: rouge后面的-f、-p、-r有什么侧重吗，是否有文档解释一下

A: 这个在文档中没有，但这些是常见指标，相关介绍也比较多：

在ROUGE评估指标中，-f、-p、-r分别代表不同的计算方式或侧重点：

1. -r（Recall）：
-r表示召回率（Recall），用于衡量生成文本覆盖参考文本的程度。它关注的是生成文本中有多少内容与参考文本相关且被正确覆盖。例如，在ROUGE-L中，召回率的计算基于最长公共子序列（LCS）的比例，即$R_{lcs} = \frac{LCS(ref, pred)}{m}$，其中$m$是参考文本的长度。

2. -p（Precision）：
-p表示精确率（Precision），用于衡量生成文本的准确性。它关注的是生成文本中有多少内容是与参考文本匹配的，但不考虑参考文本是否完全覆盖。例如，在ROUGE-L中，精确率的计算为$P_{lcs} = \frac{LCS(ref, pred)}{n}$，其中$n$是生成文本的长度。

3. -f（F-measure）：
-f表示F值（F-measure），是召回率和精确率的调和平均数。它综合了召回率和精确率的表现，通常通过参数$\beta$来调整两者的权重。当$\beta$较大时，更重视召回率；当$\beta$较小时，更重视精确率。在ROUGE中，如果特别关注召回率，$\beta$会被设置为一个较大的数值。

总结来说：

- -r侧重于生成文本对参考文本的覆盖程度。
- -p侧重于生成文本的准确性和相关性。
- -f则是两者的平衡，根据具体需求调整权重。

这些指标的选择取决于任务的具体目标。例如，在机器翻译或文本摘要任务中，通常更关注召回率（-r），因为漏掉重要信息的影响更大。

### Q58: RuntimeError: Cannot run the event loop while another loop is running

A: 不要在notebook环境中运行，请写一个python脚本，在终端中运行

### Q59: 使用Qwen2.5-0.5B-Instruct模型，evalscope速度基准测试（本地vLLM推理）报Cannot connect to host 127.0.0.1:8877 ssl:default错误

A: 请等待一会儿，会自动从本地vllm拉起服务

### Q60: 如何针对多模态大模型进行测评

A: 参考：https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/parameters.html#id5

目前支持flickr8k这个多模态数据集

### Q61: 指定模型API评估支持SGLang嘛

A: evalscope对接的是OpenAI API兼容的API接口，服务引擎是SGLang，VLLM，OLLAMA等，或者是云上的API都是可以的。

### Q62: gsm8k base model正确率低，有复读现象

A: 调大temperature试试

### Q63: 代码生成类数据集进行评测的时候结果为0

A: 参考这里最佳实践：https://evalscope.readthedocs.io/zh-cn/latest/best_practice/eval_qwq.html#id5

设置"filters": {"remove_until": "</think>"}试试

此外，用use_cache参数可以复用模型生成结果直接跑评测

### Q64: 在相同设置的情况下（是否是fewshot，token长度，采样参数等），evalscope和opencompass跑出来的指标能否对齐，对应评测集的评测代码是否一致呢，因为发现一些评测集用这两个评测框架跑出来的指标不一致

A: 不同评测框架评测代码并不一致，评测指标也很难对齐，建议在同一框架下对比指标

### Q65: 请问要评测qwen2.5-omni在DOCVQA test上的性能应该怎么做

A: 需要使用VLMEvalKit backend，并拉起一个模型推理服务，具体可以参考使用文档：https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/vlmevalkit_backend.html#id4

vlmevalkit不支持qwen2.5omni直接推理，但可以用vLLM拉起一个模型推理服务，指定api url就可以评测了。

### Q66: 为什么Total requests 不等于 Expected number of requests ？

A: speed_benchmark 只有8个请求，而默认 Expected number of requests 是 100

### Q67: 是否支持NPU多卡直接评测

A: 
- 可以使用vLLM拉起模型API服务，这样可以指定tp
- 本地加载模型可以在--model-args中指定 device_map 参数为 auto, 可以自动划分模型

### Q68: 相同机器上推理两次结果不一样这是为啥呢

A: 可以尝试在 'models' 参数中设置 seed，例如：

```
'models': [
{'path': '/data00/models/deepseek-r1-w4a8',
'openai_api_base': 'http://221.194.152.47:8000/v1/chat/completions',
'is_chat': True,
'batch_size': 100,
'seed': 42
},
```

### Q69: Native模式MATH_500存在多个答案时/答案存在矩阵时提取不准确

A: 参考这个解决方法：https://evalscope.readthedocs.io/zh-cn/latest/get_started/faq.html#q13-math-500badcase

### Q70: 假设我现在有一个类R1的带思考的模型，但它不使用<think></think>标签，而是<|beginning_of_thinking|> <|end_of_thinking|>还可以直接使用evalscope进行评估吗？是否需要自己实现部分适配代码？需要的话建议修改哪一部分呢

A: 参考：https://evalscope.readthedocs.io/zh-cn/latest/get_started/faq.html#q1-qwqifeval
把</think>替换成<|end_of_thinking|>

### Q71: 是否支持/generate接口进行测试

A: 目前只考虑兼容openai api的接口，generate接口暂不考虑

### Q72: 如何在VLMEvalKit backend中引入System Prompt

A: 在model参数里面设置：

```
'model': [{'api_base': 'http://localhost:12345/v1/chat/completions',
'key': 'token-abc123',
...
'system_prompt': 'xxx'
}
```

### Q73: 从modelscope下载的数据集，dataset_infos.json中缺少dtype是什么问题？很多数据集从本地加载的话，貌似 dataset_id要指定精确到jsonl文件。mmlu数据集需要指定到data/test目录。都不能只指定数据集名那个目录

A: 已知问题，临时解决方法是删掉数据集中的dataset_infos.json文件


### Q74: ms-opencompass和ms-vlmeval的源码可以提供一下吗

A: 这两个是fork了opencompass和vlmeval 原始仓库上进行了一些修改并打包：

ms-vlmeval 是 https://github.com/Yunnglin/VLMEvalKit/tree/eval_scope
ms-opencompass 是 https://github.com/wangxingjun778/opencompass

### Q75: 配置--dataset-hub "local" --dataset-dir ~/.cache/modelscope/hub/datasets 还是走的在线下载

A: 请查看教程：https://evalscope.readthedocs.io/zh-cn/latest/get_started/basic_usage.html#id13
执行的命令参数不对

### Q76: realwordqa 评测模型最后输出来的是选abcd的准确率吗？

A: 是选项的准确率，参考：https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/index.html#id3

MCQ基本都是选项的准确率

### Q77: QwQ在ifeval上精度和官方不对齐, 我在A800上测QwQ-32B在IFEval数据集上的精度，用evalscope测出来只有51，但是用QwQ官方仓库的脚本测却有82，请问evalscope测QwQ是有啥特殊的设置嘛？

A: datast-args 的 ifeval加上 `"filters": {"remove_until": "</think>"}'`

### Q78: API模型服务评测embeddings报错,openai.BadRequestError: Error code: 400 - {'object': 'error', 'message': 'dimensions is currently not supported', 'type': 'BadRequestError', 'param': None, 'code': 400}

A: 设置`'dimensions': None`再试试

### Q79: 请教一下app里面的数据集概览显示情况，evalscope app里面界面中我选的单模型——数据集概览，分数表中显示的只有一个分数；但单模型——数据集详情显示很详细，每个指标都给出分数了。同时多模型比较里面显示的也是只有BLEU-1。所以有个疑问，概览里面只展示BLEU-1是比较看重这个指标吗？还是默认选择详情的第一条作为显示？

A: 多模型比较里面默认使用了数据集的第一个metric进行展示

### Q80: 为啥测ollama并发性能上不去？

A: 可以加一个 export OLLAMA_NUM_PARALLEL=10 试试

### Q81: 请问自己构建的多选题格式测试集，answer应该怎么写

A: LLM的自定义多选题格式参考这里：https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_dataset/llm.html#mcq

### Q82: 后端推理框架使用华为的mindie时，无法使用--min-tokens 2048 \--max-tokens 2048 \控制输出的长度

A: `--min-tokens` 不是所有模型服务都支持该参数，请查看对应API服务的文档。

### Q83: evalscope 当前内置的评测集（例如 LiveCodebench、AIME、MATH-500）等只支持 pass1 评测，与社区的主流做法存在差异（例如 QwQ 提供的评测方案

A: 
1. 本框架支持QwQ评测中的n_sample参数，在generation config中设置n可计算多个sample的平均指标，参考：https://evalscope.readthedocs.io/zh-cn/latest/best_practice/eval_qwq.html#id5
2. 本框架支持 pass@k 指标，参考 https://evalscope.readthedocs.io/zh-cn/latest/get_started/parameters.html#id3 中的metrics_list

### Q84: evalscope Live_code_bench review阶段卡住

A: 设置`judge_worker_num=1`

### Q85: 评估Qwen2-audio的时候，跑了几个文本指标，回复的内容全是感叹号

A: 目前对于本地加载的多模态模型支持并不完善，建议使用vllm等推理服务拉起api来评测

### Q86: 速度基准测试脚本使用/v1/chat/completions运行报错

A: 速度测试--url需要使用/v1/completions端点，而不是/v1/chat/completions，避免chat template的额外处理对输入长度有影响。

### Q87: --stream 的统计是否需要加上reasoning_content的内容

A: reasoning_content 也是模型输出的一部分，对于模型推理的速度没有影响，最终completion token长度里面是加上了reasoning的长度的

### Q88: 评测多模态大模型时报错：Unknown benchmark

A: 多模态评测参考[这里](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/vlmevalkit_backend.html#vlmevalkit) ，需要使用VLMEval 工具

### Q89: 评估Gemma3系列模型时出现RuntimeError: CUDA error: device-side assert triggered错误

A: gemma3是多模态模型，目前框架的chat_adapter对于多模态模型的支持不是很完善，建议使用模型推理框架（vllm等）拉起模型服务来进行评测




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

### Q13: 如何进行多模态大模型的压测，如何输入图片呢？

A: 目前支持dataset设置为flickr8k进行多模态模型压测，请[参考](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/parameters.html#id5)

### Q14: 执行evalscope perf --dataset aime25命令出现 KeyError：'aime25'

A: 根据文档，perf 支持的数据集列在此处：
https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/parameters.html#dataset-configuration。

“aime25”是 eval 支持的数据集，而不是 perf 支持的数据集。

### Q15: 在Chrome下无法选择报告

A: 使用的可视化方法不对

参考这个文档：https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/quick_start.html#id6

### Q16: 模型推理性能压测时，请求数更多的情况下TTFT更小, 在相同输入Token、输出Token、并发数的情况下，请求数较大的TTFT反而比较小

A: 建议使用random数据集再测试一下，可以固定输入prompt的长度，例如100

参考使用文档：https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/examples.html#random

### Q17: 压测过程中报错, 使用evalscope 0.16和0.14版本对昇腾mindie部署的deepseek-r1-32b模型进行压力测试，10并发100请求正常生成报告，20并发200请求每次都是进度一半左右出现报错，把evalscope装到昇腾服务器还是换到别的服务器上都遇到这个问题，而测试RTX-4090平台vLLM部署的相同模型就没任何问题。

A: 添加`--tokenizer-path`参数

### Q18: evalscope perf推理压测结果为 nan

A: 优先检查提供的数据格式是否正确。openqa 使用jsonl文件的 question 字段作为prompt。不指定dataset_path将从modelscope自动下载数据集

### Q19: 压测是没有那个 app 可视化的吗

A: 压测不支持app可视化，但可以用wandb和swanlab来可视化，参考 https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/quick_start.html#id6

### Q20: 我想使用eval命令调用本地https服务评测自己的数据集，请求服务时需要往请求头里塞一个token, 调用https模型服务的时候需要在headers里放入一个token参数，我使用perf命令是支持的，通过--header参数，类似于--headers 参数名=mytoken这种形式

A: 在generation_config中设置 `{"extra_headers": {"key": "value"}}`即可。

### Q21: 使用 evalscope perf 如何 调用 本地的数据集

A: 参考这里：https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/parameters.html#id5

需指定dataset为 line_by_line 并提供dataset_path，会逐行将txt文件的每一行作为一个提示

### Q22: evalscope - WARNING - Retrying... <404> {"detail": "Not Found"} perf 端口连接不上

A: 你需要先部署模型服务才能开始压测

### Q23: speed_benchmark代码判断接口是否为"v1/chat/completion",会导致v1/chat/completion压测报错

A: speed_benchmark 只支持 v1/completions 接口

https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/speed_benchmark.html#api

### Q24: 模型是ollma运行的，evalscope perf --max-tokens这个参数设置后看结果的输出token不到4096

A: 应该是ollama的服务不支持max-tokens这个参数

### Q25: 本地vllm部署qwen2.5vl-3b模型，如何通过perf压测？

A: 参考：https://evalscope.readthedocs.io/zh-cn/latest/user_guides/stress_test/parameters.html#id5

数据集设置`flickr8k`