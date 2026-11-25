# 2026 年 11 月八组模型官方 Benchmark 评分汇总

> 核查日期：2026-11-24。范围：GPT-6.2 Astra / Sol、Claude Fable 5.2 / Mythos 5.2、Gemini 4 Neon、Qwen4-Max、GLM-5.4、Kimi K3.1 Pro / Highspeed、MiniMax M3.1 Flash、DeepSeek-V4.1-Flash-1017。

## 口径与限制

- 只采用厂商官方发布页、System Card、模型卡、官方 Hugging Face / GitHub 和官方技术报告。优先保留发布页或模型卡的能力主表；System Card 用于交叉核对，不把其独立的生物安全、拒答、策略、alignment、monitorability 和内部行为监测附表混入通用能力清单。
- `官方主表`表示厂商直接发布的结果；`内部`表示厂商自有或未公开数据集的评测；`第三方`表示官方材料明确归因于外部机构的结果；`Composite/Index`不是原子 Benchmark，单独标识。
- 分数默认是百分数或官方原始分，不擅自把 Elo、美元、BPB、胜率、Index 转成百分比。`L/M/H/X/Max`依次表示 Low、Medium、High、Xhigh、Max effort；未披露的设置不补猜。
- 同一 Benchmark 的不同数据版本、工具、harness、Pass@k、均次和是否 text-only 均保留。不同设置不能横向直接比较。
- 本文把 Astra / Sol 计为一个模型组，Fable / Mythos 计为一个模型组，Kimi Pro / Highspeed 计为一个模型组。Highspeed 没有独立分数，不复制 Pro 结果。
- GLM-5.4 的 30B Loop Transformer 消融、Qwen4-Max-Base 的 BPB、DeepSeek 基础版 V4.1-Flash 技术报告均不计入对应后训练模型的主表。

## 1. GPT-6.2 Astra / Sol

官方来源：[发布页](https://openai.com/index/introducing-gpt-6-2-astra-sol/)、[System Card](https://deploymentsafety.openai.com/gpt-6-2-astra-sol)、[System Card PDF](https://deploymentsafety.openai.com/gpt-6-2-astra-sol/gpt-6-2-astra-sol.pdf)。发布页有完整图表数据，System Card 用于核对评测含义与安全边界。

| Benchmark | 版本 / 设置 | 分数 | 口径 / harness | 性质 |
| --- | --- | --- | --- | --- |
| Agents’ Last Exam | v1.1 | Astra L/M/H/X/Max：65.77 / 72.06 / 75.96 / 78.33 / 77.61；Sol：66.65 / 70.26 / 72.04 / 72.48 / 73.04 | 五档 effort | 官方主表 |
| Terminal-Bench-Science | 0.1 | Astra：56.3 / 72.3 / 74.9 / 74.0 / 77.1；Sol：43.4 / 63.1 / 67.7 / 64.3 / 68.0 | 五档 effort | 官方主表 |
| FrontierCode | 2.0 | Astra：38.1 / 42.2 / 43.6 / 44.1 / 44.2；Sol：33.1 / 38.3 / 39.7 / 39.8 / 43.8 | 五档 effort | 官方主表 |
| LifeVal | 发布页版本 | Astra：46.3 / 78.9 / 83.1 / 83.7 / 86.1；Sol：25.3 / 70.3 / 80.8 / 82.6 / 83.8 | 五档 effort | 官方主表 |
| GDPval | v2，内部下一版 | Astra：28.9 / 39.3 / 45.0 / 47.8 / 50.4；Sol：23.1 / 34.8 / 42.7 / 45.0 / 46.9 | 五档 effort；不是公开 GDPval gold 或 GDPval-AA | 内部 |
| BenchCAD | 2.1 Core | Astra：42.6 / 68.5 / 74.6 / 77.6 / 82.5；Sol：23.9 / 41.2 / 64.0 / 70.4 / 73.1 | 五档 effort | 官方主表 |
| AutomationBench | 1.0.6 | Astra：37.1 / 43.8 / 47.9 / 47.2 / 52.4；Sol：31.2 / 40.2 / 41.2 / 40.6 / 41.1 | 五档 effort；发布页 Vega 图表原始值乘 100 | 官方主表 |
| SecureSWE | 内部 | Astra：54.5 / 57.7 / 61.7 / 63.8 / 65.7；Sol：43.1 / 47.1 / 51.9 / 54.0 / 55.6 | 五档 effort | 内部 |
| PatchBench | 发布页版本 | Astra：71.1 / 75.6 / 78.8 / 79.7 / 79.7；Sol：65.6 / 70.5 / 74.1 / 74.2 / 75.6 | 五档 effort | 官方主表 |
| ExploitBench | Internal Port | Astra：19.5 / 46.5 / 48.0 / 53.0 / 53.5；Sol：0 / 24.5 / 47.0 / 45.5 / 50.5 | 五档 effort；不等于公开 ExploitBench V8 | 内部 |
| DeepBinary | 发布页版本 | Astra：$1,038,250；Sol：$687,250 | 美元价值指标，非百分数 | 官方主表 |
| Artificial Analysis Intelligence Index | v5.0.1 | Astra：45.7 / 57.0 / 60.1 / 61.6 / 62.1；Sol：40.6 / 52.7 / 54.9 / 55.7 / 57.3 | 五档 effort | 第三方 Composite/Index |
| Vals Science Index | v1 | Astra：59.5 / 67.6 / 68.1 / 64.8 / 68.4；Sol：46.3 / 56.5 / 57.6 / 58.5 / 57.8 | 五档 effort | 第三方 Composite/Index |
| AA-AnalystAgent | 发布页版本 | Astra：62.5 | Sol 未列分 | 第三方 |
| AA-3D | 发布页版本 | Astra：1318；Sol：1259 | 原始分 | 第三方 |
| Harvey LAB-AA | v1.1 | Astra：2.2 / 8.1 / 13.3 / 12.8 / 15.0；Sol：3.1 / 5.0 / 5.8 / 6.1 / 11.1 | hallucination-gated all-pass；五档 effort | 第三方 |

## 2. Claude Fable 5.2 / Mythos 5.2

官方来源：[发布页](https://www.anthropic.com/claude-fable-and-mythos-5-2)、[System Card](https://www.anthropic.com/document/claude-fable-5-2-mythos-5-2-system-card)、[System Card PDF](https://assets.claude.com/27d60918d218125e7504bc7354c4b4626ec56b11.pdf)。Fable 与 Mythos 是同一底层模型的不同 safeguard 配置，因此按一个模型组统计。

| Benchmark | 版本 / 设置 | 分数 | 口径 / harness | 性质 |
| --- | --- | --- | --- | --- |
| SWE-bench Pro | 发布页版本 | Fable：93.0 | agentic coding | 官方主表 |
| SWE-bench Multilingual | 发布页版本 | Fable：96.6 | agentic coding | 官方主表 |
| SWE-bench Multimodal | 发布页版本 | Fable：74.1 | agentic coding | 官方主表 |
| DeepSWE | v1.1 | Fable：74.3 | agentic coding | 官方主表 |
| FrontierCode | 2.0 | Fable L/M/H/X/Max：34.2 / 37.6 / 45.3 / 49.1 / 45.7 | 五档 effort | 官方主表 |
| Terminal-Bench | 4.0 | Fable：67.6；Mythos：70.5 | Claude Code | 官方主表 |
| Terminal-Bench-Science | 0.1 | Fable：74.6；Mythos：75.3 | Fable 为 Claude Code Max；发布页另引 AA mini-swe-agent 结果，不能混用 | 官方主表 |
| FrontierSWE | v2 | Fable：69.9 | agentic coding | 官方主表 |
| STEM Professor Bench | 内部 | Fable：61.8 | 内部教授级 STEM 任务 | 内部 |
| ProgramBench | 发布页版本 | Fable：85.6；显示剩余 context：91.6 | 两种上下文设置 | 官方主表 |
| HLE | with tools | Fable Low：47.6；Max：74.2 | 工具版 | 官方主表 |
| DRACO | 发布页版本 | Low：64.6；Xhigh：86.7；Max：87.7 | effort 档位 | 官方主表 |
| WANDR | modified | L/M/H/X/Max：10.0 / 73.6 / 79.3 / 80.6 / 82.1 | 修改版，不能当官方原始 WANDR | 官方主表 |
| Chartography | no tools / with tools | Fable：73.4 / 91.6 | 两种工具设置 | 官方主表 |
| BenchCAD | 2.1 Core | no tools：75.4；with tools：83.5 | 两种工具设置 | 官方主表 |
| OSWorld | 2.1 offline | partial：89.3；strict：54.6 | 两种判分口径 | 官方主表 |
| OfficeQA | 发布页版本 | Fable：83.4 | agentic document QA | 官方主表 |
| OfficeQA Pro | 发布页版本 | Fable：72.9 | agentic document QA | 官方主表 |
| Real-World Finance | v2，内部 | Fable：Elo 1637 | Elo，非百分数 | 内部 |
| PhysicianBench | 发布页版本 | Low：55.0；High：61.8；Xhigh：64.6；Max：64.4 | effort 档位 | 官方主表 |
| GMMLU | 发布页版本 | Fable：96.0 | 多语言知识与推理 | 官方主表 |
| MILU | 发布页版本 | Fable：94.1 | 多语言知识与推理 | 官方主表 |
| RoboDojo Sim | 发布页版本 | Fable：64.9 | 仿真机器人 | 官方主表 |
| CVE Rediscovery | 内部 | Mythos：89.6 | 内部网络安全任务 | 内部 |
| CursorBench | 4.0 | L/M/H/X/Max：53.2 / 54.8 / 56.9 / 56.3 / 60.0 | 五档 effort | 第三方 |
| Harvey LAB-AA | v1.1 | Fable：7.2 | hallucination-gated all-pass | 第三方 |
| GDPval-AA | v2.1 | Max：Elo 1831；Xhigh：Elo 1751 | 不是公开 GDPval gold，也不是 OpenAI 内部 GDPval v2 | 第三方 |
| AA-Briefcase | v1.1 | Max：Elo 1785、rubric 66.1；Xhigh：Elo 1738 | Elo 与 rubric 是不同指标 | 第三方 |
| AutomationBench | 1.0.6 private | Low：39.7；Max：51.4 | 私有集，不等于公开 600 题 | 第三方 |
| Remote Labor Index | 发布页版本 | Fable：41.3 | 外部职业任务指数 | 第三方 |
| ARC-AGI-3 | High / Max | High Standard/Provider：79.26 / 97.72；Max：65.66 / 94.90 | 两种官方口径均保留 | 第三方 |

## 3. Gemini 4 Neon

官方来源：[发布页](https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-gemini-4-neon/)、[Model Card](https://deepmind.google/models/model-cards/gemini-4-neon/)、[Model Card PDF](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-4-Neon-Model-Card.pdf)、[评测方法 PDF](https://storage.googleapis.com/deepmind-media/gemini/gemini_4_neon_model_evaluation.pdf)。

| Benchmark | 版本 / 设置 | 分数 | 口径 / harness | 性质 |
| --- | --- | --- | --- | --- |
| CursorBench | 4.1 | 65.4 | 与 Claude 的 4.0 不是同版 | 官方主表 |
| Terminal-Bench | 4.1 | 65.2 | cheating-adjusted；不是 4.0 | 官方主表 |
| FrontierCode | 2.0 | 46.2 | High effort | 官方主表 |
| SWE-sweep | 发布页版本 | provider harness：39.0；Mini-SWE harness：12.8 | 两个 harness 不可合并 | 官方主表 |
| AutomationBench | 发布页版本 | 56.3 | 版本未在该单元格披露 | 官方主表 |
| Agents’ Last Exam | v1.1 | 75.8 | xhigh | 官方主表 |
| RiemannBench | 发布页版本 | 84.0 | 数学研究任务 | 官方主表 |
| Terminal-Bench-Science | 0.1 | 70.5 | 5 次平均，不是 pass@5 | 官方主表 |
| HLE | Diamond | 71.3 | HLE-Diamond，不是 HLE-Full | 官方主表 |
| MKA | 发布页版本 | 56.5 | 官方原始分 | 官方主表 |
| Chartography | no tools | 73.5 | 无工具 | 官方主表 |
| LVBench | 发布页版本 | 89.0 | 长视频理解 | 官方主表 |
| GraphWalks | 发布页版本 | 95.2 | BFS F1 | 官方主表 |
| Vending-Bench | 2 | $16,843 | 美元，非百分数 | 官方主表 |
| CWE-Bench | v2 | 70.7 | 精确版本为 v2 | 官方主表 |
| SEC-Bench Pro | 发布页版本 | 90.5 | 网络安全能力 | 官方主表 |
| WebExploitBench | v1.2 | 51.26；Model Card 简写 51.3 | 保留原始精度与四舍五入差异 | 官方主表 |
| AutoBaxBench | hard | 76.0 | hard 子集 | 官方主表 |
| ExploitGym | 发布页版本 | 10h：48.3；no time cap：47.9 | 两种时间预算 | 官方主表 |
| Gray Swan IPI | 发布页版本 | ASR@15：0.0% | 攻击成功率，越低越好 | 官方主表 |
| Artificial Analysis Intelligence Index | v5.0.1 | 61 | xhigh | 第三方 Composite/Index |
| Vals Index | v2.2 | 70.9 | 总指数，不是 HLAB-Vals | 第三方 Composite/Index |
| Agent Arena Index | 发布页版本 | 1336 | Index 原始分 | 第三方 Composite/Index |
| AA Coding Agent Index | 发布页版本 | 58.6 | Index 原始分 | 第三方 Composite/Index |

Legal Research Bench 与 Tax Agent Bench 在官方材料中只有名次，没有可核验分数，因此不填数。

## 4. Qwen4-Max

官方来源：[Qwen 官方博客](https://qwen.ai/blog?id=qwen4-max)、[阿里集团发布稿](https://www.alibabagroup.com/en-US/document-2060139246777270272)、[官方主图](https://img.alicdn.com/imgextra/i3/O1CN01d8WOyjWDWEKAxlwO_!!6000000007934-2-tps-5400-3600.png)。未找到独立的 Qwen4-Max 公开技术报告、模型卡或开放权重仓库。

| Benchmark | 版本 / 设置 | 分数 | 口径 / harness | 性质 |
| --- | --- | --- | --- | --- |
| CodeArena | Code Arena WebDev | Elo 1750，第 6 | xHigh；Elo 非百分数 | 官方引用榜单 |
| AgentArena | 发布页版本 | Net Improvement 7.5 | 不是 Agent Arena Index 的 1336 口径 | 官方主表 |
| DeepSWE | v1.1 | 75.7 | agentic coding | 官方主表 |
| Terminal-Bench | 4.0 | 48.5 | XHigh 来自官方效率图交叉对应，主图单元格未显式写 effort | 官方主表 |
| ProgramBench | 发布页版本 | 31.5 | 官方原始分 | 官方主表 |
| AA-Briefcase | v1.1 | Elo 1686 | Elo 非百分数 | 官方引用第三方 |
| Agents’ Last Exam | v1.1 | 46.7 | 官方主图 | 官方主表 |
| JobBench | 发布页版本 | 68.5 | agentic professional work | 官方主表 |
| APEX-Agents | v1.1 | 72.1 | 官方主图 | 官方主表 |
| HLAB-Vals | 1.1 | 21.3 | 不是 Vals 总指数 | 官方引用第三方 |
| AutomationBench | AA | 72.5 | AutomationBench-AA，不等于公开集或 private 1.0.6 | 官方引用第三方 |
| BioMysteryBench | 发布页版本 | 85.0 | 生物推理能力，不是安全拒答 | 官方主表 |
| HealthBench Professional | 发布页版本 | 60.0 | Professional 版，不等于 HealthBench All/Consensus/Hard | 官方主表 |
| HLE | 版本与工具未披露 | 51.7 | 不补猜工具或 effort | 官方主表 |
| Chartography | 发布页版本 | 54.2 | 工具设置未披露 | 官方主表 |
| Video-Index | 发布页版本 | 68.1 | 视频 Index | 官方主表 |
| Last Secure Code Bench | 发布页版本 | 函数级双通过 31.8%；三粒度同时安全生成 14.7% | 两个不同安全指标 | 官方主表 |
| Work360 | 内部 | 52.69 | 内部工作任务 | 内部 |

官方云资源调度结果存在 Qwen 博客 10.42% 与阿里集团稿约 9% 的表述差异，二者不是同一精度的标准 Benchmark 分数，未并入主表。Base 模型 BPB 图仅属于 Qwen4-Max-Base。

## 5. GLM-5.4

官方来源：[官方博客](https://z.ai/blog/glm-5.4)、[官方文档](https://docs.z.ai/guides/llm/glm-5.4)、[GLM-5 GitHub](https://github.com/zai-org/GLM-5)。截至核查日，未找到 GLM-5.4 本体公开标准 Benchmark 数值表、公开技术报告或公开模型卡。

官方只披露内部 pairwise Win/Tie/Lose：对 GLM-5.3 为 51.1/25.7/23.1，对 Kimi K3.1 Pro 为 41.8/26.9/31.3，对 DeepSeek-V4.1-Flash-1017 为 36.6/30.2/33.2。任务集、样本数、judge 和 effort 未披露，因此这些不是可复现的公开 Benchmark，不进入反向索引。

以下仅为 **30B Loop Transformer 架构消融**，不是 GLM-5.4 旗舰本体成绩：

| 30B 消融 Benchmark | 原始模型 | Loop Transformer | 变化 |
| --- | ---: | ---: | ---: |
| AIME25 | 19.17 | 30.42 | +11.25 |
| GPQA-Diamond | 47.58 | 56.26 | +8.68 |
| AA-LCR | 40.33 | 47.67 | +7.34 |
| MMLU-Pro | 71.65 | 75.05 | +3.40 |
| IFBench | 50.20 | 53.47 | +3.27 |

## 6. Kimi K3.1 Pro / Highspeed

官方来源：[官方博客](https://www.kimi.ai/blog/kimi-k3-1-pro)、[Hugging Face 模型卡](https://huggingface.co/moonshotai/Kimi-K3.1-Pro)、[固定 commit README](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/raw/d9e99b0792a5e93b78ea9bea62a68167e36ddd9e/README.md)、[Kimi Vendor Verifier](https://github.com/MoonshotAI/Kimi-Vendor-Verifier)。默认 Max effort、temperature 1.0、top-p 0.95；BrowseComp top-p 1.0。Highspeed 只声明同能力与 2–3× 速度，没有独立 Benchmark 表。

| Benchmark | 版本 / 设置 | 分数 | 口径 / harness | 性质 |
| --- | --- | --- | --- | --- |
| Terminal-Bench | 4.0 | 65.7 | Max effort | 官方主表 |
| Terminal-Bench-Science | 0.1 | 38.1 | Max effort | 官方主表 |
| SWE-Marathon | v1.1 | 51.7 | agentic coding | 官方主表 |
| BenchCAD | 发布页版本 | no tools：72.8；with tools：92.7 | 两种工具设置 | 官方主表 |
| ProgramBench | 发布页版本 | 31.5 | binary score | 官方主表 |
| DeepSWE | v1.1 | 75.5 | Max effort | 官方主表 |
| PostTrainBench | v1.2 | 36.5 | 与 MiniMax v1.1 不同版 | 官方主表 |
| HLE | Full，with tools / without tools | 70.0 / 51.1 | 两种工具设置 | 官方主表 |
| HLE | Diamond，without tools | 39.4 | Diamond 子集 | 官方主表 |
| BrowseComp | top-p 1.0 | 92.1 | browsing agent | 官方主表 |
| BioMysteryBench | 发布页版本 | 80.6 | 生物推理能力 | 官方主表 |
| CharXiv | RQ，with Python / without Python | 93.4 / 89.0 | reasoning questions；两种工具设置 | 官方主表 |
| MMMU-Pro | with Python / without Python | 90.0 / 84.1 | 两种工具设置 | 官方主表 |
| GDP.pdf | with Python / without Python | 85.7 / 34.6 | 同一名称但工具差异极大 | 官方主表 |
| Chartography | with Python | 90.2 | 工具版 | 官方主表 |
| BabyVision | with Python | 95.4 | 工具版 | 官方主表 |
| ZeroBench | main，with Python | pass@5：63.0 | 不是 avg@5 | 官方主表 |
| ZeroBench | main，without tools | avg@5：29.8；pass@5：39.0 | 两种指标 | 官方主表 |
| Agents’ Last Exam | v1.1 | ALL：43.2；CLI：44.2 | 两种运行入口 | 官方主表 |
| AutomationBench | AA | 73.3 | AutomationBench-AA | 官方主表 |
| GDPval-AA | v2.1 | Elo 1846 | Elo 非百分数 | 官方主表 |
| AA-Briefcase | v1.1 | Elo 1678 | Elo 非百分数 | 官方主表 |
| APEX-Agents | 发布页版本 | 69.4 | 官方主表 | 官方主表 |
| OfficeQA Pro | v2 / 未标 v2 | 64.4 / 67.4 | 两个版本口径均保留 | 官方主表 |
| Frontier Finance | 发布页版本 | 58.6 | 金融任务 | 官方主表 |
| Harvey Legal Agent | 发布页版本 | 23.7 | 不擅自并入 Harvey LAB-AA 或 HLAB-Vals | 官方主表 |
| HANDBOOK.md | 发布页版本 | 44.6 | 官方主表中的 Benchmark 名称 | 官方主表 |
| OSWorld | 2.1 | 79.6 | 官方主表 | 官方主表 |
| GPQA-Diamond | without tools | 94.2 | 无工具 | 官方主表 |
| ExploitBench | 发布页版本 | 74.7 | 不等于 OpenAI Internal Port | 官方主表 |
| ExploitGym | 发布页版本 | 20.7 | 时间预算未在该单元格披露 | 官方主表 |
| Kimi Code Bench | 2.0 / 3.0 Long-horizon | 77.1 / 61.2 | 内部两个版本 | 内部 |
| Business Operation | 内部 | 37.8 | 博客图片 alt 曾写 37，采用主表 37.8 | 内部 |
| DECK-Arena | 内部 | Elo 1456 | Elo 非百分数 | 内部 |
| Design Bench | 内部 | Elo 957 | Elo 非百分数 | 内部 |
| Kimi Hallu Bench | Hard with tools / without tools | 8.3 / 5.4 | 越低越好；两种设置 | 内部 |
| PerceptionBench | 内部版本 | 68.3 | 厂商标为内部，不能仅凭同名认定公开 adapter 精确复现 | 内部 |
| xdev-bench-hard | 内部 | 81.0 | 内部开发任务 | 内部 |
| Financial Modeling | 内部 | 57.6 | 内部金融任务 | 内部 |
| Financial Research | 内部 | 70.4 | 内部金融任务 | 内部 |

官方 KVV 仓库另给出 Moonshot 行的复测结果；同一模型组内与主表重复的 Benchmark 不增加使用次数。

| Benchmark | 版本 / 设置 | 分数 | 口径 / harness | 性质 |
| --- | --- | --- | --- | --- |
| OCRBench | KVV Moonshot 行 | 0.872 | 官方仓库复测 | 官方仓库 |
| MMMU-Pro | Vision，KVV Moonshot 行 | 0.841 | 与主表 84.1 小数口径一致 | 官方仓库 |
| BEAM | 1M，KVV Moonshot 行 | 0.3917 | 长上下文 | 官方仓库 |
| DeepSWE | KVV Moonshot 行 | 0.755 | 与主表 75.5 小数口径一致 | 官方仓库 |
| ToolCall bench | KVV Moonshot 行 | 0.972 | 工具调用 | 官方仓库 |

MCP Atlas 只在协议脚注中被提及，没有主表分数，因此不填数、不进入反向索引。

## 7. MiniMax M3.1 Flash

官方来源：[官方博客](https://www.minimax.io/blog/minimax-m31f)、[Hugging Face 模型卡](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)。MiniMax Sparse Attention 论文不包含 M3.1 / M3.1-Flash 实验，未并入。

| Benchmark | 版本 / 设置 | 分数 | 口径 / harness | 性质 |
| --- | --- | --- | --- | --- |
| AutomationBench | v1.0.6 | L/M/H/X/Max：52.8 / 57.7 / 59.2 / 61.2 / 62.8 | 五档 effort | 官方主表 |
| BankerToolBench | 发布页版本 | 76.2 | 工具调用 | 官方主表 |
| GDP.pdf | AA | 20.8 | 与 Kimi with/without Python 口径不同 | 官方主表 |
| Terminal-Bench | 4.0 | 38.1 | 官方主表 | 官方主表 |
| LOCA-Bench | 256K | 60.3 | 长上下文 agent | 官方主表 |
| Agents’ Last Exam | 发布页版本 | 29.6 | Max effort | 官方主表 |
| OSWorld | 2.0 | 61.7 | 不是 2.1 | 官方主表 |
| PostTrainBench | v1.1 | 39.3 | 与 Kimi v1.2 不同版 | 官方主表 |
| DeepSWE | v1.1 | High：68.1；Xhigh：72.1；Max：73.7 | 三档 effort | 官方主表 |
| NL2Repo-Bench | 发布页版本 | 61.2 | 仓库生成 | 官方主表 |
| SWE Refactor Bench | 发布页版本 | 23.0 | 重构任务 | 官方主表 |
| CyberGym | 发布页版本 | 82.2 | 博客另写 82%，采用模型卡精度 82.2 | 官方主表 |
| BabyVision | with tools | 91.0 | 工具版 | 官方主表 |
| HLE | without tools，text-only 2,158 题 | 43.4 | 不是 Full 2,500 题 | 官方主表 |
| CharXiv | RQ | 82.5 | reasoning questions | 官方主表 |
| Terminal-Bench-Science | 0.1 | 3.3 | Artificial Analysis；mini-swe-agent；Pass@1 三次平均 | 第三方 |

## 8. DeepSeek-V4.1-Flash-1017

官方来源：[官方新闻](https://api-docs.deepseek.com/news/news261017/)、[Hugging Face 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)。官方链接的 51 页技术报告是基础版 V4.1-Flash，不含 `1017`，因此没有用基础版分数补齐本表。

| Benchmark | 版本 / 设置 | 分数 | 口径 / harness | 性质 |
| --- | --- | --- | --- | --- |
| HLE | without tools | Full：42.6；text-only：44.0 | 两种数据范围 | 官方主表 |
| Terminal-Bench | 4.0 | 42.3 | Claude Code | 官方主表 |
| DeepSWE | v1.1 | High：72.56637168141593；Xhigh：73.00884955752213；Max：76.99115044247786 | Mini-SWE-V2；Pass@1；成本约 0.11215 / 0.14620 / 0.19538 USD/task | 官方主表 |
| ProgramBench | 发布页版本 | 29.1 | 官方原始分 | 官方主表 |
| NL2Repo-Bench | 发布页版本 | 68.2 | 仓库生成 | 官方主表 |
| AutomationBench | v1.0.6 | 60.5 | 官方主表 | 官方主表 |
| Agents’ Last Exam | 发布页版本 | 35.4 | 官方主表 | 官方主表 |
| APEX-Agents | Apex Bench v1.1 | 72.1 | 名称归一为 APEX-Agents，保留原别名 | 官方主表 |
| Chartography | with tools | 83.0 | 工具版 | 官方主表 |
| BabyVision | with tools | 92.5 | 工具版 | 官方主表 |
| ZeroBench | main，with tools | 51.0 | 官方模型卡未在该单元格给 Pass@k | 官方主表 |
| Terminal-Bench-Science | 0.1 | 12.9 | 官方模型卡转载的外部结果 | 第三方 |
| CWE-bench | 版本未披露 | 57.0 | 不与 Gemini CWE-Bench v2 合并 | 第三方 |
| OmniCompass-VL | 发布页版本 | 28.0 | 后端 checkpoint 身份未独立认证 | 第三方 |

## 交叉核对结论与未决缺口

1. **有完整官方能力表的模型组**：GPT-6.2、Claude 5.2、Gemini 4 Neon、Qwen4-Max、Kimi K3.1 Pro、MiniMax M3.1 Flash、DeepSeek-V4.1-Flash-1017。
2. **没有本体公开标准跑分表**：GLM-5.4。其 pairwise 与 30B 消融不能伪装成旗舰本体 Benchmark。
3. **没有独立分数**：Kimi K3.1 Highspeed；不能复制 Pro 分数。
4. **同名不等于同评测**：Terminal-Bench 4.0/4.1、AutomationBench public/private/AA、HLE Full/Diamond/text-only、GDPval 内部 v2/GDPval-AA/公开 gold、PostTrainBench v1.1/v1.2、OSWorld 2.0/2.1 均须分开理解。
5. **官方材料内部差异已保留**：MiniMax CyberGym 82% 与 82.2、Kimi Business Operation 37 与 37.8、Gemini WebExploitBench 51.26 与 51.3、Qwen 云资源调度 10.42% 与约 9%。主表采用更精确或主表值，不静默覆盖原始差异。
6. 反向使用次数与 EvalScope 覆盖状态见[配套文档](2026_11_benchmark_evalscope_coverage.md)。
