# Benchmark 使用次数反向索引与 EvalScope 覆盖核查

> 核查日期：2026-11-24。数据来自[配套文档](2026_11_model_benchmark_official_scores.md)的结构化表格；EvalScope 状态基于 2026-11-24 的 `main` 分支注册索引与 adapter 元数据。

## 统计口径

- 统计单位是 **8 个模型组**，不是模型变体数或分数行数。同一组对同一 Benchmark 的多个 effort、工具设置或来源只计 1 次。
- 主表共 **98 个 Benchmark 家族、160 个“模型组—Benchmark”关系、164 行分数**；71 个家族只被 1 个模型组使用。GLM-5.4 没有本体公开标准跑分，30B 消融不计入主统计。
- 为观察技术栈共性，HLE Full/Diamond/text-only、Terminal-Bench 4.0/4.1、AutomationBench public/private/AA 等按家族合并计数；版本与设置仍保留在“覆盖说明”和正向文档中，不能据此直接横比分数。
- `[3P]` 表示该模型组对此 Benchmark 只有官方材料明确归因于第三方的分数。总组数包含这些结果，阅读“厂商自行运行次数”时应剔除相应 `[3P]` 组。
- **是否支持**只有两档：**支持**（至少一个公开版本能用 EvalScope 运行）、**不支持**（没有等价 adapter）。其他未接入版本只在后一列说明，不再因此降级；支持不等于所有版本或厂商设置均可复现。
- “未找到”指当前 `_index.json` 与 adapter 源码未找到等价可运行 adapter，不表示 EvalScope 永远无法支持，也不把近似名称当作集成证据。

模型组简称：GPT = GPT-6.2 Astra / Sol；Claude = Fable 5.2 / Mythos 5.2；Gemini = Gemini 4 Neon；Qwen = Qwen4-Max；Kimi = Kimi K3.1 Pro / Highspeed；MiniMax = M3.1 Flash；DeepSeek = V4.1-Flash-1017。

## 一、被至少两个模型组使用的 Benchmark

| Benchmark | 模型组数 | 模型组 | 是否支持 | 支持到什么程度（注册名 / 版本 / 限制） |
| --- | ---: | --- | --- | --- |
| AutomationBench | 7 | GPT、Claude[3P]、Gemini、Qwen[3P]、Kimi、MiniMax、DeepSeek | 支持 | `automation_bench`：官方公开 600 题并 pin commit；不覆盖 private/AA，adapter 未声明公开集与 v1.0.6 的精确映射 |
| Agents’ Last Exam | 6 | GPT、Gemini、Qwen、Kimi、MiniMax、DeepSeek | 不支持 | 未找到等价注册 adapter；v1.1 与未标版本不能混为精确复现 |
| HLE | 6 | Claude、Gemini、Qwen、Kimi、MiniMax、DeepSeek | 支持 | `hle`：公开 Full 2,500 题，可过滤 text-only；`hle_diamond`：另一套公开 1,000 题数据；工具有无是运行设置 |
| Terminal-Bench | 6 | Claude、Gemini、Qwen、Kimi、MiniMax、DeepSeek | 支持 | `terminal_bench_v4`：dataset_revision `4.0.0`；Gemini 4.1 未找到 |
| Terminal-Bench-Science | 6 | GPT、Claude、Gemini、Kimi、MiniMax[3P]、DeepSeek[3P] | 不支持 | `terminal_bench_v4` 不是 Terminal-Bench-Science 0.1 |
| Chartography | 5 | Claude、Gemini、Qwen、Kimi、DeepSeek | 不支持 | 未找到等价 adapter；with/without tools 分数不可直接比较 |
| DeepSWE | 5 | Claude、Qwen、Kimi、MiniMax、DeepSeek | 支持 | `deep_swe`：`evalscope/deep-swe`；metadata 未声明 v1.1，不能仅凭同名声称精确覆盖 |
| ProgramBench | 4 | Claude、Qwen、Kimi、DeepSeek | 不支持 | 未找到等价注册 adapter |
| AA-Briefcase | 3 | Claude[3P]、Qwen[3P]、Kimi | 不支持 | 未找到等价 adapter；Elo 与 rubric 是不同指标 |
| APEX-Agents | 3 | Qwen、Kimi、DeepSeek | 不支持 | 未找到等价 adapter；DeepSeek 原名为 Apex Bench v1.1 |
| BabyVision | 3 | Kimi、MiniMax、DeepSeek | 支持 | `baby_vision`：388 题；厂商 with Python/tools 设置不属于该 VLM adapter 的精确复现 |
| BenchCAD | 3 | GPT、Claude、Kimi | 不支持 | 未找到等价 adapter；2.1 Core 与未标版本需区分 |
| FrontierCode | 3 | GPT、Claude、Gemini | 不支持 | 未找到等价 adapter；当前均为 2.0，但 effort 不同 |
| OSWorld | 3 | Claude、Kimi、MiniMax | 不支持 | 未找到等价 adapter；2.0、2.1、offline partial/strict 不可混用 |
| Artificial Analysis Intelligence Index | 2 | GPT[3P]、Gemini[3P] | 不支持 | Composite/Index，不是单一原子 Benchmark；当前均为 v5.0.1 |
| BioMysteryBench | 2 | Qwen、Kimi | 不支持 | 未找到等价 adapter |
| CharXiv | 2 | Kimi、MiniMax | 支持 | `charxiv`：含 reasoning 子集；with Python 不是该 adapter 的精确设置 |
| CursorBench | 2 | Claude[3P]、Gemini | 不支持 | 未找到等价 adapter；Claude 4.0 与 Gemini 4.1 不同版 |
| ExploitBench | 2 | GPT、Kimi | 不支持 | 未找到等价 adapter；GPT Internal Port 与 Kimi 版本不能视为同任务 |
| ExploitGym | 2 | Gemini、Kimi | 不支持 | 未找到等价 adapter；时间预算设置不同 |
| GDP.pdf | 2 | Kimi、MiniMax | 不支持 | 未找到等价 adapter；AA、with/without Python 口径不同 |
| GDPval-AA | 2 | Claude[3P]、Kimi | 不支持 | `gdpval` 是公开 220 题 gold submission 流程，不是 GDPval-AA v2.1 Elo |
| Harvey LAB-AA | 2 | GPT[3P]、Claude[3P] | 不支持 | 未找到等价 adapter；hallucination-gated all-pass |
| NL2Repo-Bench | 2 | MiniMax、DeepSeek | 不支持 | 未找到等价注册 adapter |
| OfficeQA Pro | 2 | Claude、Kimi | 支持 | `officeqa` 默认 `officeqa_pro` 133 题；Kimi Pro v2 未在 metadata 声明 |
| PostTrainBench | 2 | Kimi、MiniMax | 不支持 | 未找到等价 adapter；v1.2 与 v1.1 不同版 |
| ZeroBench | 2 | Kimi、DeepSeek | 支持 | `zerobench`：main 100 题；pass@5、avg@5 和工具设置需另行复现 |

## 二、只被一个模型组使用的 Benchmark

| Benchmark | 模型组数 | 模型组 | 是否支持 | 支持到什么程度（注册名 / 版本 / 限制） |
| --- | ---: | --- | --- | --- |
| AA Coding Agent Index | 1 | Gemini[3P] | 不支持 | Composite/Index，未找到等价 adapter |
| AA-3D | 1 | GPT[3P] | 不支持 | 未找到等价 adapter |
| AA-AnalystAgent | 1 | GPT[3P] | 不支持 | 未找到等价 adapter |
| ARC-AGI-3 | 1 | Claude[3P] | 不支持 | `arc_agi_2` 是不同 Benchmark，不能据此宣称覆盖 ARC-AGI-3 |
| Agent Arena Index | 1 | Gemini[3P] | 不支持 | Composite/Index；与 Qwen AgentArena Net Improvement 不合并 |
| AgentArena | 1 | Qwen | 不支持 | 未找到等价 adapter；不是 Agent Arena Index |
| AutoBaxBench | 1 | Gemini | 不支持 | hard 子集未找到等价 adapter |
| BEAM | 1 | Kimi | 不支持 | 1M 长上下文版本未找到等价 adapter |
| BankerToolBench | 1 | MiniMax | 不支持 | `tool_bench` 不是同一 Benchmark |
| BrowseComp | 1 | Kimi | 支持 | `browsecomp`：1,266 题；浏览工具与 agent harness 需另行配置 |
| Business Operation | 1 | Kimi | 不支持 | 内部评测，未找到公开等价 adapter |
| CVE Rediscovery | 1 | Claude | 不支持 | 内部网络安全任务，未找到公开等价 adapter |
| CWE-Bench | 1 | Gemini | 不支持 | v2 未找到；`cybench` 不等价 |
| CWE-bench | 1 | DeepSeek[3P] | 不支持 | 版本未披露；不与 Gemini CWE-Bench v2 合并，`cybench` 不等价 |
| CodeArena | 1 | Qwen | 不支持 | Code Arena WebDev Elo 未找到等价 adapter |
| CyberGym | 1 | MiniMax | 不支持 | `cybench` 不是 CyberGym |
| DECK-Arena | 1 | Kimi | 不支持 | 内部 Elo 评测 |
| DRACO | 1 | Claude | 不支持 | 未找到等价 adapter |
| DeepBinary | 1 | GPT | 不支持 | 美元价值指标，未找到等价 adapter |
| Design Bench | 1 | Kimi | 不支持 | 内部 Elo 评测 |
| Financial Modeling | 1 | Kimi | 不支持 | 内部金融任务 |
| Financial Research | 1 | Kimi | 不支持 | 内部金融任务 |
| Frontier Finance | 1 | Kimi | 不支持 | 未找到等价 adapter |
| FrontierSWE | 1 | Claude | 不支持 | v2 未找到等价 adapter |
| GDPval | 1 | GPT | 支持 | `gdpval`：公开 220 题 gold、导出 submission；不覆盖 GPT 内部 v2，且无内置官方质量 judge |
| GMMLU | 1 | Claude | 不支持 | `mmlu`、`mmmlu`、`cmmlu` 均不等于 GMMLU |
| GPQA-Diamond | 1 | Kimi | 支持 | `gpqa_diamond`；厂商无工具分数可对应基础任务，但生成设置仍需一致 |
| GraphWalks | 1 | Gemini | 不支持 | BFS F1 未找到等价 adapter |
| Gray Swan IPI | 1 | Gemini | 不支持 | ASR@15 安全攻击指标，未找到等价 adapter |
| HANDBOOK.md | 1 | Kimi | 不支持 | 官方主表中的 Benchmark 名称；未找到等价 adapter |
| HLAB-Vals | 1 | Qwen[3P] | 不支持 | v1.1；不是 Vals 总指数 |
| Harvey Legal Agent | 1 | Kimi | 不支持 | 不擅自并入 Harvey LAB-AA 或 HLAB-Vals |
| HealthBench Professional | 1 | Qwen | 支持 | `health_bench` 支持 All/Consensus/Hard，不包含 Professional 精确版本 |
| JobBench | 1 | Qwen | 支持 | `job_bench`：`evalscope/job-bench`；未声明 dataset_revision，精确数据版本未证实 |
| Kimi Code Bench | 1 | Kimi | 不支持 | 内部 2.0 与 3.0 Long-horizon |
| Kimi Hallu Bench | 1 | Kimi | 不支持 | 内部 Hallu Bench；越低越好 |
| LOCA-Bench | 1 | MiniMax | 不支持 | 256K 版本未找到等价 adapter |
| LVBench | 1 | Gemini | 不支持 | 未找到等价 adapter |
| Last Secure Code Bench | 1 | Qwen | 不支持 | 两个安全生成指标均未找到等价 adapter |
| LifeVal | 1 | GPT | 不支持 | 未找到等价 adapter |
| MILU | 1 | Claude | 支持 | `milu` 同名注册 adapter |
| MKA | 1 | Gemini | 不支持 | 未找到等价 adapter |
| MMMU-Pro | 1 | Kimi | 支持 | `mmmu_pro` 明确存在；支持 standard/vision，with Python 不是精确默认设置 |
| OCRBench | 1 | Kimi | 支持 | `ocr_bench` 同名注册 adapter；KVV 分数为 0.872 原始口径 |
| OfficeQA | 1 | Claude | 支持 | `officeqa` 注册存在，但当前 `subset_list` 仅配置 `officeqa_pro`，未配置完整 246 题版本 |
| OmniCompass-VL | 1 | DeepSeek[3P] | 不支持 | 未找到等价 adapter；checkpoint 身份亦未独立认证 |
| PatchBench | 1 | GPT | 不支持 | 未找到等价 adapter |
| PerceptionBench | 1 | Kimi | 支持 | `perception_bench` 注册存在；Kimi 标为内部版本，不能仅凭同名认定一致 |
| PhysicianBench | 1 | Claude | 不支持 | 未找到等价 adapter |
| Real-World Finance | 1 | Claude | 不支持 | v2 内部 Elo |
| Remote Labor Index | 1 | Claude[3P] | 不支持 | 外部职业任务 Index，未找到等价 adapter |
| RiemannBench | 1 | Gemini | 不支持 | 未找到等价 adapter |
| RoboDojo Sim | 1 | Claude | 不支持 | 仿真机器人任务未找到等价 adapter |
| SEC-Bench Pro | 1 | Gemini | 不支持 | `swe_bench_pro`、`cybench` 均不等价 |
| STEM Professor Bench | 1 | Claude | 不支持 | 内部教授级 STEM 任务 |
| SWE Refactor Bench | 1 | MiniMax | 不支持 | 未找到等价 adapter |
| SWE-Marathon | 1 | Kimi | 不支持 | v1.1 未找到等价 adapter |
| SWE-bench Multilingual | 1 | Claude | 支持 | `swe_bench_multilingual_agentic`：使用 `SWE-bench/SWE-bench_Multilingual` test |
| SWE-bench Multimodal | 1 | Claude | 不支持 | 多语言 adapter 不等于 Multimodal |
| SWE-bench Pro | 1 | Claude | 支持 | `swe_bench_pro`：使用 `ScaleAI/SWE-bench_Pro` test，agentic Docker/E2B 流程 |
| SWE-sweep | 1 | Gemini | 不支持 | provider 与 Mini-SWE 两个 harness 均未找到等价 adapter |
| SecureSWE | 1 | GPT | 不支持 | 内部安全编码任务 |
| ToolCall bench | 1 | Kimi | 不支持 | `tool_bench` 不能据此认定等价 |
| Vals Index | 1 | Gemini[3P] | 不支持 | v2.2 Composite/Index |
| Vals Science Index | 1 | GPT[3P] | 不支持 | v1 Composite/Index |
| Vending-Bench | 1 | Gemini | 不支持 | v2 美元指标，未找到等价 adapter |
| Video-Index | 1 | Qwen | 不支持 | 未找到等价 adapter |
| WANDR | 1 | Claude | 不支持 | modified 版本未找到等价 adapter |
| WebExploitBench | 1 | Gemini | 不支持 | v1.2 未找到等价 adapter |
| Work360 | 1 | Qwen | 不支持 | 内部工作任务 |
| xdev-bench-hard | 1 | Kimi | 不支持 | 内部开发任务 |

## 三、不计入主统计但已核查的项目

| 项目 | 为什么不计入 98 个主统计家族 | EvalScope 状态 |
| --- | --- | --- |
| AIME25 | 仅为 GLM 30B Loop 架构消融，不是 GLM-5.4 本体 | `aime25` 已注册 |
| GPQA-Diamond（GLM 消融） | 同上；Kimi 本体结果已在主表计 1 组 | `gpqa_diamond` 已注册 |
| AA-LCR | 仅为 GLM 30B 消融 | `aa_lcr` 已注册 |
| MMLU-Pro（GLM 消融） | 仅为 GLM 30B 消融 | `mmlu_pro` 已注册 |
| IFBench | 仅为 GLM 30B 消融 | `ifbench` 已注册 |
| MCP Atlas | Kimi 官方材料只有协议脚注，没有分数 | `mcp_atlas` 已注册，但无分项目不进入使用次数 |
| GLM-5.4 内部 pairwise | 不是公开 Benchmark，任务集、样本数、judge、effort 未披露 | 不适用 |
| Qwen4-Max-Base BPB | Base 模型不同训练阶段，且 BPB 不与后训练 Benchmark 百分数合并 | 不按本表口径核查 |
| DeepSeek 基础版 V4.1-Flash 技术报告 | 报告不包含 `1017`，不能拿来补 1017 分数 | 不按本表口径核查 |

## 四、EvalScope 核查依据与使用建议

1. 注册名首先核对生成的 `evalscope/benchmarks/_index.json`，再读对应 adapter 的 `BenchmarkMeta`，避免把目录名、搜索近似匹配或文档提及误当成可运行集成。
2. 精确版本的关键证据包括：
   - `terminal_bench_v4` 明确使用 Terminal-Bench 4.0.0；Gemini 的 4.1 不在覆盖内。
   - `automation_bench` 固定官方仓库 commit 并运行公开 600 题；官方 leaderboard private set 与本地公开分数方向性可比但不相同。
   - `hle` 是公开的 2,500 题 Full 数据，支持排除多模态题；`hle_diamond` 是另一套公开的 1,000 题数据；外部浏览工具属于运行设置。
   - `health_bench` 只有 All/Consensus/Hard；Qwen 的 HealthBench Professional 不能标为精确覆盖。
   - `officeqa` 当前默认且仅暴露 `officeqa_pro`；Kimi Pro v2 和 Claude 完整 OfficeQA 需要另核数据版本。
   - `gdpval` 是公开 220 题 gold 的 submission 导出流程，不内置官方质量评分；GDPval 内部 v2、GDPval-AA v2.1 均不能由它替代。
   - `mmmu_pro` 明确存在于索引；此前仅靠模糊搜索漏报会得出错误结论。
3. 若目标是复现厂商分数，优先选择已有精确/同名 adapter 的 Terminal-Bench 4.0、SWE-bench Pro、SWE-bench Multilingual、GPQA-Diamond、MMMU-Pro、CharXiv、BabyVision、ZeroBench、BrowseComp，再逐项对齐 harness、工具、数据 revision、judge、采样和 Pass@k。
4. 若目标是补齐 EvalScope 的公共能力覆盖，高频且尚未找到 adapter 的候选依次是 Agents’ Last Exam、Terminal-Bench-Science、Chartography、ProgramBench、FrontierCode、BenchCAD、OSWorld；AutomationBench/HLE/DeepSWE 则更需要做版本与 harness 对齐，而不是重复新增同名 adapter。

## 五、Benchmark 镜像搬运范围（2026-11-24）

| Benchmark | 搬运范围 | 源头 |
| --- | --- | --- |
| Terminal-Bench 4.1 | 全量任务镜像 | Harbor Hub revision 5 / tag 4.1.0 |
| SWE-bench Multimodal | 全部 480 个实例的任务镜像 | 官方数据集 + 官方 swebench harness |
| SEC-bench Pro | 只补 2 个 V8 镜像：`430572435`、`446124893` | `hwiwonlee/v8.x86_64:430572435`、`hwiwonlee/v8.x86_64:446124893` |
| CWE-Bench v2 | 全部 100 个任务的 Harbor/Docker 环境 | 官方数据集 |
| Last Secure Code Bench | 全部 450 个任务的环境，预计镜像约 58 GB | 官方仓库 |
| AutoBaxBench hard | hard 子集 140 个任务的多语言框架镜像 | 官方数据集 + 官方构建工具 |
| LOCA-Bench 256K | 官方容器化运行环境及 mock MCP 依赖 | 官方仓库 |
