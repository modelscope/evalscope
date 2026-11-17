# 评测诊断

诊断帮助你从“成绩是多少”继续查到“哪些样本值得看、有什么现成线索”。默认使用 `facts`：读取已有评分、预测和 Agent 轨迹，做确定性统计，不调用 LLM，也不重新评测。

## 使用

对已有结果补跑：

```bash
evalscope analyze --run-dir outputs/20261116_120000 --level facts
```

也可以传包含多次运行的父目录。用 `--model <目录中的模型ID>`、`--benchmark <基准名>` 缩小范围，两项均可重复传入；`--force` 忽略缓存。命令更新每次运行的 HTML 报告。

评测时开启：

```python
from evalscope import TaskConfig, run_task
from evalscope.api.analysis import AnalysisReportConfig

run_task(TaskConfig(
    model='your-model',
    datasets=['gsm8k'],
    analysis_report=AnalysisReportConfig(level='facts', language='zh'),
))
```

Python 的 `analysis_report=True` 和 CLI 的 `--analysis-report` 均默认启用免费 `facts`（后续也保持此默认值）；默认不开启。Web 发起的原生评测默认开启。`diagnose`、`full` 必须显式配置分析模型，详见下文。

离线配置可传 JSON 或 JSON 文件，例如：

```bash
evalscope analyze --run-dir outputs/20261116_120000 \
  --analysis-config '{"language":"zh","weak_subset_min_samples":20,"trace":{"use_providers":false}}'
```

Python 入口是 `evalscope.analysis.diagnose_run()` 和 `diagnose_benchmark()`，它们使用同一条处理链路。统一命令入口为 `evalscope analyze`。

## 怎么看结果

Web 报告按“评测结果 → 结果诊断 → 样本详情”组织。多数据集报告左侧展示各数据集的指标、成绩和样本数；选择数据集后，切换标签会保留当前数据集。换数据集会清除旧的子集、诊断筛选和样本位置，窄屏使用下拉选择。耗时与用量中的 Token 数使用 K、M 等缩写，悬停可查看精确值；延迟与 Token 明细、任务配置默认展开，也可手动收起。

打开“结果诊断”标签，顶部依次显示未通过比例、规则线索覆盖、相关环节和分析用量。模式表下方是子集热力矩阵与详细统计；悬停查看计算口径，展开“查看全部统计”查看分布等明细。先看任务未通过数和成败已知的分母。评分被排除、成败未知、未产生 review 的执行错误单独列明，不补成 0 分，也不算通过。

- **问题模式**：按观察到的问题类型、相关组件和细节分组。占比的分母是“需检查样本”，不是官方失败率；代表样本与子集交叉表可以点击。
- **事实**：评分分布、较弱子集、评分状态、judge 覆盖率、已有性能和 token 数据；普通生成的空输出、答案抽取缺失、截断、拒答文本和长度分组；单字母单选题的选项分布与混淆对；完整重复采样题组的成败稳定性；Agent 的工具与过程发现。
- **过程发现**：“样本详情”可按问题类型、相关组件和过程发现筛选。从问题类型进入匹配样本集合，点击代表样本直接定位该条；可返回结果诊断，样本链接也支持刷新和直接打开。徽标只定位有明确工具调用 ID 的步骤；无法定位的不猜位置。

截断优先读取 predictions 的 `stop_reason`；缺失时，才在有输出 token 数与上限的情况下近似判断。缺少必要数据的检查会显示“跳过”，不会伪造零发生率。连续分数没有明确阈值时，任务成败可能未知。较弱子集只对有声明区间与方向的指标比较。

默认比例达到 10% 且有事件时标为“注意”，由 `warning_rate` 配置；较弱子集须比总体低至少 10 个百分点且不少于 20 条，由 `weak_subset_gap`、`weak_subset_min_samples` 配置。其余事实标为统计；P1 不自动生成严重级别结论。选择题要求至少 90% 的标准答案为单个大写字母；多选或选项全文格式暂不统计，缺失选项映射时不判断某个字母是否越界。

问题类型可扩展，不能穷举所有失败原因。“暂无规则线索”保留未被当前规则识别的失败。即便答对，也可能出现工具错误或风险线索；工具参数来自先前返回值、状态改变调用本身都不证明越权。抽取值为空也不证明原答案正确或评分器出错。搜索无结果会保留为工具报错，不直接归为参数错误。**发现线索不等于证明失败根因**；下文介绍的模型诊断也只是待复核的解释。

## 产物与缓存

```text
<run>/analysis/<model>/<benchmark>/diagnosis.json
<run>/analysis/<model>/<benchmark>/samples.jsonl
<run>/reports/<model>/<benchmark>.json  # 仅更新 analysis 字段
<run>/reports/report.html
```

输出包含统计定义、覆盖量、代表样本、规则发现及分析用量（`facts` 为零）。导出证据按预算截断并脱敏；原始结果不复制到诊断文件。

缓存检查 reviews、相关 predictions、报告评分元数据和使用到的配置；排除自身的 `analysis` 回写。来源或样本产物发生变化时，Web 标记过期并阻止用旧记录跳转。诊断不会改成绩，在线诊断异常也不会使已完成的评测失败。旧报告只有 `Report.analysis` 时，“结果诊断”与 HTML 仍能显示原摘要。

## 模型分析、证据与统计口径

`analysis_report=True` 继续默认免费 `facts`。`diagnose` 对样本分析，`full` 再生成带引用的总结；两者必须显式配置分析模型，不自动沿用评测或评分模型。模型分析目前为实验性能力。设置 `include_trace=False` 可仅发送任务和最终回答，省略中间消息和工具返回。

```python
from evalscope.api.analysis import AnalysisReportConfig

analysis = AnalysisReportConfig(
    level='full',
    model={
        'model_id': 'qwen3.8-max',
        'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'api_key': 'env:DASHSCOPE_API_KEY',
    },
    max_diagnosed_samples=50,
    seed=42,
)
```

离线执行时，把同样配置保存为 JSON 文件：

```bash
evalscope analyze --run-dir outputs/<run> --analysis-config analysis.json
```

证据校验分两层：

- **可定位性**：来源 ID 必须存在于实际发送的、已经脱敏和截断的内容中。引用给出唯一原文时，由程序推导范围；显式范围按 Unicode 字符计数，从 0 开始、右端不含，必须与原文匹配。只归一化空白；不存在、重复而无法唯一定位、越界或不匹配均判无效。步骤必须来自实际发送的来源，且至少有一条有效引用位于该步骤。
- **语义支持**：引用存在不代表它支持解释。模型结论仍需独立标注与复核；全部引用无效、调用/格式失败或低置信度的诊断不计为获得解释。标准答案和分数保留原样，怀疑标准答案只标记为待复核。

统计保留不同分母：官方任务失败率只用成败已知的样本；评分不可用和安全信号单列。模型选样使用互斥的“任务未通过 → 评分不可用 → 安全信号 → 过程线索”优先级，再按子集分层。每层至少抽 1 条，预算不足覆盖所有层时明确报错；不能把规则高置信度视为根因已解释而跳过。

问题表的数量和占比是当前样本分类，未抽中的样本仍保留规则分类；“全量估计”另行根据各层抽中概率加权。任务失败中的待解释比例只在任务失败层计算，未抽中、模型失败及无有效解释的部分不被当作已解决。区间采用分层比例与经 Bonferroni 校正的层内 Wilson 区间组合，不对配额样本套用一个总体 Wilson 区间。全量普查层使用精确比例；这些区间表示抽样不确定性，不表示解释正确的概率。

样本请求按模型、生成设置、协议、提示词及实际来源内容缓存，可中断续跑；不合规回复和传输错误不缓存为成功。`--force` 会重新发起模型请求。报告区分本次调用量和复用量；传输重试由已有模型层处理，分析器不额外重试。调用数是分析器请求数，不包含供应商内部重试；token 为接口实际返回用量，缺失时单独计数，费用需按供应商账单核对。

独立标注的比较工具：

```bash
python -m evalscope.analysis.validation --samples <samples.jsonl> --labels <labels.jsonl> --output validation.json
```

标注使用 `DiagnosticLabel`，每份文件只对应一个 run/model/benchmark。`rule_decisions` 显式标注检测项的正反例；未标的检测项不假定为负例。工具分别报告分类一致性、归因混淆、根因步骤、引用可定位性和已复核证据支持率。模型辅助标注须保留标注者身份，不能当作人工准确率验收。
