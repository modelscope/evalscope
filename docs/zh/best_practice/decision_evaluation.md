# 使用 EvalScope 评测决策模型

决策模型把一条输入转成选项概率或等级分布。评测时需要保留完整分布：选对答案与概率可信是两个独立问题。本实践覆盖 **Choice、Noul、Score**，接入 System One 与 AutoTrust 两种接口，并给出 ModelScope 下载、部署、效果评测和并发压测的完整路径。

## 1. 先选任务与协议

| 题型 | 输入与返回 | 主要评测方式 |
|---|---|---|
| Choice | 原始选项 ID、描述及每个选项的概率 | Accuracy；固定类别任务的 macro-F1；NLL、Brier、ECE |
| Noul | 判断问题，返回 `P(yes)` | 默认阈值 0.5 的分类指标、AUROC、AUPRC、NLL、二分类 Brier、ECE |
| Score | 有顺序的等级描述及每档概率 | 期望等级的 MAE、Spearman、Pearson；离散 gold 另报 nRPS 和 argmax Accuracy |

| EvalScope `eval_type` | 地址形式 | 适用服务 |
|---|---|---|
| `systemone_api` | 基地址 `/v1`，自动请求 `/systemone` | System One；本实践使用 Clef 官方 PyTorch 实现 |
| `autotrust_api` | 基地址 `/v1`，自动请求 `/decide` | AutoTrust 原生格式；本实践使用 JEV 的 LoRA、head、bias 和 temperature |

聊天模型的 `/chat/completions` 不会自动变成决策模型。部署 JEV 时必须加载决策 adapter/head，并读取全部候选 token；本实践不启用 thinking、选项裁剪或等级重映射。

| benchmark | 数据与评测划分 | 注意事项 |
|---|---|---|
| `arc` | `allenai/ai2_arc`，`ARC-Challenge` / test | 显式选择 Challenge |
| `banking77` | `mteb/banking77` / test | 每题保留全部 77 类 |
| `boolq` | `google/boolq` / validation | 默认 Choice；可显式选择 Noul |
| `reward_bench` | `allenai/reward-bench` / filtered | v1 成对偏好；现有 sample-weighted 聚合，不是官方 category-weighted 榜单成绩 |
| `stsb` | `sentence-transformers/stsb` / test | 原标签 0–1 乘以 5；保留小数 gold，使用六档 Score |
| `jevbench` | `evalscope/jevbench` / test | Benchmark Heaven 公开 231 题，含四档、五档 Score |
| `open_decisions` | `evalscope/OpenDecisions` / train | 全部 2,352 条仅用于评测；Score 有 3–6 档 |
| `general_decision` | 本地 `default_test.jsonl` | 不需要上传数据 |

这里的 JevBench 来自 [fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench)，与 HF 上的同名一致性评测项目不同。[ModelScope 镜像](https://modelscope.cn/datasets/evalscope/jevbench)固定上游 commit `609a664e81f46e03731600bd7bd9f1b7a61257c5`，包含 72 original、48 easy、111 hard；README、原始 JSONL、许可证、转换脚本及文件校验和均可追溯。完整榜单的 Composite/Capability 不在这个公开子集内。

## 2. 安装与 ModelScope 下载

在 EvalScope 仓库根目录使用独立环境。普通 CUDA 开发环境可安装：

```bash
pip install -e '.[perf]'
pip install modelscope transformers accelerate safetensors fastapi uvicorn httpx
python examples/decision/download_models.py \
  --model Cloudflare/clef-flash --output ./models/clef-flash
python examples/decision/download_models.py \
  --model autotrust/JEV-27B --output ./models/JEV-27B
```

下载脚本固定已验证的 ModelScope Git commit，并逐文件校验 SHA-256，生成 `evalscope_release_manifest.json`。由于部分 ModelScope SDK 的 snapshot 接口只接受分支/标签，脚本先核对命名 ref 与固定 commit 的文件树完全一致，再下载和逐字节验证；分支发生变化时会明确失败。不要只下载底座权重，也不要遗漏 adapter、专用 head、校准文件和模型推理代码。

数据通过 EvalScope 默认的 ModelScope 加载器下载，不需要访问 Hugging Face。复现时同时归档数据 revision、模型 manifest、EvalScope commit、Python/torch/transformers/vLLM 版本、设备及配置。凭据通过环境变量或 SDK 登录缓存提供，不写入配置和日志。

## 3. 部署与最小请求

### Clef：完整三类与可变等级

```bash
CUDA_VISIBLE_DEVICES=0 python examples/decision/serve_clef.py \
  --model-path ./models/clef-flash --port 18131
```

该示例调用发布包中的 `load_release_model()` 和 `systemone()`，加载 backbone 与 joint schema head，使用 PyTorch eager 推理。服务串行处理请求；并发主要用于验证排队、超时和服务容量。它没有实现动态批处理，压测结果不能解释为 vLLM 性能。

```bash
curl http://127.0.0.1:18131/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"Cloudflare/clef-flash","state":"The sky is blue.","questions":{"answer":{"type":"choice","instructions":"What color is the sky?","criteria":{"blue":"Blue","red":"Red"}}}}'
```

响应的最小结构：

```json
{"model":"Cloudflare/clef-flash","answers":{"answer":{"type":"choice","choice":"blue","probabilities":{"blue":0.97,"red":0.03},"confidence":0.97}},"usage":{"input_tokens":150,"output_tokens":0}}
```

Noul 的问题为 `{"type":"noul","instructions":"Is the sky blue?"}`，回答为 `{"type":"noul","noul":0.97}`。Score 的 `criteria` 是按低到高排列的字符串数组；回答包含 `score`、`legend` 和以 `"0"` 开始的每档 `probabilities`。实际概率以服务返回为准，以上数字仅说明格式。

### AutoTrust：原生 `/decide`

先启动支持该架构、混合 attention/GDN 和 `lm_head` LoRA 的 vLLM。常规安装示例：

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve ./models/JEV-27B \
  --served-model-name autotrust/JEV-27B \
  --enable-lora --max-lora-rank 32 \
  --lora-modules jev-decision=./models/JEV-27B/adapter_vllm \
  --max-model-len 16384 --max-logprobs 256 \
  --logprobs-mode processed_logprobs --port 18132
python examples/decision/serve_autotrust.py \
  --model-path ./models/JEV-27B --upstream http://127.0.0.1:18132/v1 --port 18133
```

薄服务复用标准 `/completions`，按官方 bare prompt、verbalizer bias 和分题型 temperature 计算概率。每题只执行一次模型请求；候选 logprobs 缺失或非有限值会报错。Choice 超过原生训练的 16 个选项时使用官方 single 扩展读出，保留全部类别；其效果必须独立验证。

```bash
curl http://127.0.0.1:18133/v1/decide \
  -H 'Content-Type: application/json' \
  -d '{"kind":"choice","state":"The sky is blue.","question":"What color is the sky?","options":["Blue","Red"],"thinking":"off"}'
```

```json
{"model":"autotrust/JEV-27B","kind":"choice","options":["Blue","Red"],"probabilities":[0.97,0.03],"choice_index":0,"choice":"Blue","usage":{"prompt_tokens":150,"completion_tokens":1}}
```

数组概率必须按请求中的 `options` 原始顺序返回。Noul 使用 `kind="noul"`，返回 `options=["false","true"]`；Score 使用 `kind="score"`，返回 `options=["0","1","2","3","4","5"]`。AutoTrust 这一路固定六档，不能完整评测 JevBench/OpenDecisions；EvalScope 对其他档位明确报不支持。

### PPU 环境

保留厂商匹配的 torch/vLLM 环境。每个 SSH 会话先初始化 SDK，并选择空闲设备：

```bash
export PPU_SDK=/usr/local/PPU_SDK
source "$PPU_SDK/envsetup.sh"
export UMD_PLATFORM_TYPE=1 HGGC_DRIVER_CANDIDATE=UMD
```

本实践的厂商 vLLM 0.20.1 已包含 Qwen3.5 文本实现，但缺少文本架构注册、混合缓存声明、文本 M-RoPE 位置接口及输出 head 的 LoRA 声明。`examples/decision/ppu_vllm` 提供独立注册插件，复用厂商现有实现，不修改共享环境的 vLLM 源码：

```bash
python -m pip install --no-deps --target ./decision-runtime ./examples/decision/ppu_vllm
export PYTHONPATH="$PWD/decision-runtime:${PYTHONPATH:-}"
export VLLM_PLUGINS=evalscope_decision_ppu
```

随后启动上述 vLLM 命令；本次验证使用 `--enforce-eager`。如果当前版本已有对应架构注册，插件不会替换它。PPU 版包必须与 SDK 配套，不能直接用通用 CUDA wheel 覆盖。Clef 使用官方 PyTorch 路径，其专用 head 的 vLLM 部署不在本次已验证范围内。

## 4. 使用 EvalScope 评测

```python
from evalscope import TaskConfig, run_task

run_task(TaskConfig(
    model='Cloudflare/clef-flash',
    eval_type='systemone_api',
    api_url='http://127.0.0.1:18131/v1',
    datasets=['jevbench', 'open_decisions', 'stsb'],
    eval_batch_size=4,
    generation_config={'timeout': 300, 'retries': 1},
    limit=5,  # 正式评测删除；limit 对每个 subset 生效
))
```

P0 配置：

```python
run_task(TaskConfig(
    model='Cloudflare/clef-flash', eval_type='systemone_api',
    api_url='http://127.0.0.1:18131/v1',
    datasets=['arc', 'banking77', 'boolq', 'reward_bench', 'stsb'],
    dataset_args={'arc': {'subset_list': ['ARC-Challenge']}},
    generation_config={'timeout': 300, 'retries': 1},
    limit=5,
))
```

BoolQ 显式 Noul：

```python
run_task(TaskConfig(
    model='autotrust/JEV-27B', eval_type='autotrust_api',
    api_url='http://127.0.0.1:18133/v1', datasets=['boolq'],
    dataset_args={'boolq': {'extra_params': {'decision_kind': 'noul', 'decision_threshold': 0.5}}},
    generation_config={'timeout': 300, 'retries': 1}, limit=5,
))
```

仅提交 state、问题、选项/等级和显式配置的任务说明。gold、rationale、教师预测和 provenance 只用于评分。不要给决策 API 配置 `temperature`、`max_tokens`、流式输出或工具。`retries` 表示总尝试次数，效果验证可显式设为 1 便于核对请求数。

### 自定义 JSONL

建立目录 `./decision-data`，放入 `default_test.jsonl`，一行一题：

```json
{"id":"c1","state":{"text":"I am happy."},"question":{"type":"choice","instructions":"Classify sentiment.","criteria":{"positive":"Positive","negative":"Negative"}},"expected":"positive"}
{"id":"n1","state":"The sky is blue.","question":{"type":"noul","instructions":"Is the sky blue?"},"expected":"yes"}
{"id":"s1","state":"The service was excellent.","question":{"type":"score","instructions":"Rate satisfaction.","criteria":["Poor","Fair","Good","Excellent"]},"expected":"3"}
```

```bash
evalscope eval --model Cloudflare/clef-flash --eval-type systemone_api \
  --api-url http://127.0.0.1:18131/v1 --datasets general_decision \
  --dataset-args '{"general_decision":{"local_path":"./decision-data"}}'
```

自定义固定类别分类可在每条记录设置 `fixed_classes=true`，计算保留全部类别的 macro-F1。普通多选题的 A/B/C/D 只是题内选项位置，不视为跨题分类类别。

## 5. 解释结果与复核

输出目录保留原始请求/响应、规范化决策结果、评分输入与 JSON/HTML 报告。原概率不改写；允许的舍入偏差会另外生成评分分布，并在 review 的 `probabilities_normalized_for_scoring` 标记中说明。旧 Choice 缓存字段仍可读取；使用现有 `use_cache` 和 `rerun_review` 重新评分，无需重复推理。

- **唯一主指标**：JevBench 为公开子集 Accuracy（缺失计错），自定义/P0 分类为 Accuracy，STS-B 为 MAE，OpenDecisions 为 Capability。其他数值用于解释原因。
- **NLL**：越低越好，数值下限为 `1e-12`，每条 review 都记录该值。
- **Brier**：Noul 为 `(p_yes-y)^2`；Choice/离散 Score 为各类别平方误差之和。不同定义不混合平均。
- **AUPRC**：使用平均精确率 AP 的阶梯积分，保留相同预测概率的并列组。
- **ECE**：默认 10 个等宽分箱。Noul 校准 `P(yes)`；Choice/离散 Score 校准最大选项概率与正确率。
- **Score**：不同档数的通用指标分组报告，避免把不同等级尺度的 MAE/相关系数混在一起。期望值是 `sum(level * probability)`。例如 `[0.1,0.2,0.3,0.4]` 的期望为 2.0，而 argmax 是 3；MAE 与 Accuracy 因此衡量不同问题。STS-B 不为计算 Accuracy/nRPS 而离散化 gold。
- **可靠性**：按题型分别输出风险—覆盖率曲线及 AURC；Choice/Noul 风险为分类错误率，Score 为按等级跨度归一化的期望值绝对误差（含连续 gold）；最大选项概率与厂商 confidence 分开统计。相同置信度按样本顺序稳定排序，曲线不是重新训练得到的校准器。
- **不可计算**：单类 AUROC、常量相关系数等会省略数值，在聚合 metadata 记录原因和有效数量；不会补零。
- **失败与覆盖率**：普通指标只使用有效样本。检查 execution summary 中 requested/succeeded/errored，不能只看分数。

JevBench 公开 scorer 在 Noul 概率并列时选 no；概率总和偏差不超过 0.001 时保留原值，超过 0.001 且不超过 0.02 时才归一化。公开子集 Accuracy 和 family-macro 使用全部所选任务为分母。通用指标使用有效样本，报告中与公开协议指标分开。

可以把 JSON 报告中的曲线导出为可交互的独立 HTML（`plotly` 随 perf 依赖安装）：

```bash
python examples/decision/plot_reliability.py \
  --report outputs/<run>/reports/<model>/jevbench.json \
  --output reliability.html
```

OpenDecisions 对齐其固定版本官方 `score.py`，Intelligence、Calibration、Capability 在报告中存为 0–1 比例并按百分比显示。官方 Intelligence 使用全部所选任务作为分母，缺失预测计错；普通指标使用有效分布。默认阈值 0.5 与部分 JevBench 版本的 Noul 弃答规则不同。`limit` 后的成绩只代表所选样本，不能与完整榜单直接比较。

## 6. perf：完整决策请求的性能

先停止该模型的效果评测流量，再导出固定请求集。每个文件对应一种题型，Choice 使用完整 BANKING77：

```bash
python examples/decision/prepare_perf_requests.py \
  --api systemone --model Cloudflare/clef-flash --output ./perf-input/systemone
python examples/decision/prepare_perf_requests.py \
  --api autotrust --model autotrust/JEV-27B --output ./perf-input/autotrust
```

manifest 记录请求数、UTF-8 字节数、字符数及 SHA-256；服务实际 tokenizer 的输入长度记录在 usage/perf 输出中。不要把字符数当 token 数。

每个模型、每种题型独立运行：

```bash
for kind in choice noul score; do
  evalscope perf --api systemone --model Cloudflare/clef-flash \
    --url http://127.0.0.1:18131/v1 --dataset line_by_line \
    --dataset-path "./perf-input/systemone/$kind.jsonl" \
    --parallel 1 4 8 16 --number 1000 1000 1000 1000 \
    --warmup-num 20 --no-stream --read-timeout 300 \
    --outputs-dir "./perf-results/clef-$kind"
done
```

AutoTrust 改为 `--api autotrust`、模型 `autotrust/JEV-27B`、地址 `http://127.0.0.1:18133/v1` 及对应输入文件。插件不会注入生成参数，不做隐藏重试；HTTP 200 的非法答案同样记失败。

读取每档的 `benchmark_summary.json`、`benchmark_percentile.json` 和 HTML：

- 每档 20 次预热不进入正式 1,000 次统计。关注 p50/p90/p95/p99 请求延迟、成功 req/s、成功率、失败和 Timeout Requests。
- 本实践一请求一题，因此成功 req/s 就是成功 decisions/s。输入吞吐为成功输入 token 总数除以完整测量窗口。
- TTFT/TPOT/ITL 不适用于非流式决策；摘要用 `Not Applicable Metrics` 显式标记，兼容字段中的零值不代表零延迟。
- 固定推理配置、缓存开关、数据顺序、设备占用与输入集合。相同请求集会重复使用，vLLM prefix cache 会影响结果，必须说明是否开启。
- 客户端 HTTP 延迟包含网络、服务排队、模型推理、完整响应接收和 JSON 解码；请求准备与协议校验在 HTTP 计时之外，吞吐测量窗口包含这些客户端开销；若服务另报内部耗时，单独列出。PPU 压测应在服务器本机请求 localhost，避免把 SSH 隧道延迟当推理延迟。
- Clef 参考服务串行执行，增加并发可能主要增加排队延迟。验收不预设吞吐门槛；优化需另行对比相同输入和正确性。

## 7. 验证记录与常见问题

2026-10-15 已在 PPU-ZW810 完成真实验证，模型和数据均来自 ModelScope。Clef 使用官方 PyTorch，JEV 使用厂商 vLLM 0.20.1 与本文的注册插件；普通 CUDA 命令是部署示例，未作 CUDA 实测。完整表和机器可读记录位于仓库 `examples/decision/VALIDATION.md`、`validation_results.json`。

| 验证项 | 实测结果 |
|---|---|
| Clef 完整评测 | JevBench 231、OpenDecisions 2,352、ARC-Challenge 1,172、BANKING77 3,076、BoolQ Choice/Noul 各 3,270、RewardBench 2,985、STS-B 1,379；全部预测与评分成功 |
| 官方口径 | JevBench 公开准确率 82.25%、family-macro 82.70%；OpenDecisions Intelligence 55.3260%、Calibration 79.0701%、Capability 67.1981%，均经独立官方 scorer 核对 |
| AutoTrust | BANKING77 完整 3,076 条、77 类；另跑 Noul、六档 Score、ARC、RewardBench 冒烟；其他 Score 档位显式拒绝 |
| perf | 两模型 × 三题型 × 四并发，共 24,000 次正式请求，失败和超时均为 0；每组另有 20 次预热 |
| 本地 | 合并主分支后 1,164 项相关测试、CI-lite、lint 与架构检查通过；GitHub CI 状态单独看 PR |

以并发 1 → 16 为例，Clef Choice 为 3.8880 → 4.0865 req/s、p99 为 0.2909 → 3.9250 s，反映串行排队；JEV Score 为 4.7369 → 19.3471 req/s、p99 为 0.2485 → 0.8732 s。全部并发和输入长度见验收记录。这些是不同模型与后端的运行结果，不是受控速度排名，也不是未经训练污染核查的榜单声明。

实测请求和响应在开发期间保存，最终以归档实现 `6e9970881fdf8a3cf28de6ba0e893db78513e00c` 重放全部 20,962 条效果预测、24,000 条 perf 响应并重新评分；最终部署示例另通过六个真实请求与模型 revision 校验。Clef 发布实现的 state 窗口为 16,384 token，本轮最大观测输入为 4,032 token，未验证更长输入。结束后本轮服务、端口和设备均已清理，原始证据保留。

| 现象 | 检查方向 |
|---|---|
| PPU 首次推理退出，提示 SDK 不存在 | 在启动服务的同一个 shell 中 source PPU SDK |
| Qwen3.5 config 类型或 hybrid cache 报错 | 检查厂商版本与文本架构注册插件，保留日志，不绕过 head |
| adapter 报 `lm_head` 不支持 | 确认输出 head 的 LoRA 声明及匹配的模型实现 |
| 候选 logprobs 缺失/非有限 | 检查 `--max-logprobs`、processed logprobs、候选 token 限制及 generation defaults |
| AutoTrust 无法跑四档/五档 Score | 模型协议固定六档；使用 Clef 完整评测，不能改写 benchmark 等级 |
| 下载数据报 Offline mode | 效果评测进程不要设置全局 HF_HUB_OFFLINE；ModelScope SDK 可能复用此开关。服务可在本地模型下载后离线启动 |
| 分数正常但覆盖率不足 | 检查逐条错误、超时与标签/等级一致性，修复后补跑 |
| ECE 很低但 Accuracy 低 | 校准和任务能力是两个问题，应同时读主指标、NLL/Brier 和覆盖率 |
