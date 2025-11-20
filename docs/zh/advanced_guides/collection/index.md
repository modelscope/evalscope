# 构建评测指数（Index）

## 为什么需要自定义 **Index**？
- 单一数据集无法代表真实场景。你的场景也许是 RAG 问答、金融分析或代码补全。
- 公共榜单不等于你的最佳选择。你需要一个能服务业务决策的“专属指数”。

就像 [Artificial Analysis Intelligence Index](https://artificialanalysis.ai/methodology/intelligence-benchmarking#intelligence-index-evaluation-suite-summary) 衡量模型通用智力能力，[Vals Index](https://www.vals.ai/benchmarks/vals_index) 衡量模型商业价值一样，EvalScope 的 **Collection** 允许你“定义属于你的 **Index**”。不再盲目相信通用的 **MMLU** 等单独的榜单分数，而是根据你的业务场景（如：**RAG** 问答、金融分析、代码补全）配置**权重**，生成最适合你的模型选型指标。

## 操作方法
1) **定义 Schema**：选择数据集，并按业务价值设定权重（你的价值观）。
2) **采样数据**：按权重/分层/均匀策略抽取代表性数据，得到混合数据集 JSONL。
3) **统一评测**：对混合集一次跑测，汇总多维得分，得到你的指数视图。



:::{toctree}
:maxdepth: 2

schema.md
sample.md
evaluate.md
:::
