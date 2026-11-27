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

Python 的 `analysis_report=True` 和 CLI 的 `--analysis-report` 均默认启用免费 `facts`（后续也保持此默认值）；默认不开启。Web 发起的原生评测默认开启。`diagnose` 必须显式配置分析模型，详见下文。

离线配置可传 JSON 或 JSON 文件，例如：

```bash
evalscope analyze --run-dir outputs/20261116_120000 \
  --analysis-config '{"language":"zh","weak_subset_min_samples":20,"trace":{"use_providers":false}}'
```

Python 入口是 `evalscope.analysis.diagnose_run()` 和 `diagnose_benchmark()`，它们使用同一条处理链路。统一命令入口为 `evalscope analyze`。

## 怎么看结果

在 Web 报告打开「结果诊断」，可生成或更新基础分析，无需重跑评测。选择「配置模型分析」后，为当前数据集输入兼容 OpenAI 的接口和模型，选择分析范围及样本上限，再主动运行。页面会显示模型请求次数上限，也可取消发送中间步骤和工具输出。Web 单次最多分析 50 条样本，其他配置继续使用 CLI 或 Python。

Web 的两档分析沿用已保存的基础分析设置，包括评分阈值。统计设置有变化时，报告先展示更新后的基础分析，待模型分析使用同一套设置后再展示模型提议。如果旧产物的设置无法恢复，请先用原来的配置运行一次 `evalscope analyze`。

对于有评分但未定义通过线的数据集，可打开「查看样本评分」，按主指标方向查看优先排查的样本。用问题列表或统计引用定位样本后，链接会保留当前问题、样本位置和可定位的引用步骤。

Web 报告按“评测结果 → 结果诊断 → 样本详情”组织。多数据集报告左侧展示各数据集的指标、成绩和样本数；选择数据集后，切换标签会保留当前数据集。换数据集会清除旧的子集、诊断筛选和样本位置，窄屏使用下拉选择。耗时与用量中的 Token 数使用 K、M 等缩写，悬停可查看精确值；延迟与 Token 明细、任务配置默认展开，也可手动收起。

诊断读取已保存的评测记录，无需为每个 benchmark 单独适配。平均值指标如果声明了准确率或精确匹配的语义，即使评分字段名称不同，也能使用主评分中的 0/1 结果。少数 benchmark 可以补充安全判定等专属信号；基础分析不依赖这些扩展。

打开“结果诊断”标签，先看未通过数、成败已知的分母，以及优先排查的问题。点击问题后，右侧显示对应样本。无法评分、成败未知、没有留下记录的执行错误分开显示。分析记录列出分析范围；模型分析还会显示引用校验与用量。

- **问题模式**按未通过、无法评分样本中可见的现象分组，同一样本可属于多个组。点开问题可核对样本和已有的模型提议。
- **事实**：评分分布、较弱子集、评分状态、judge 覆盖率、已有性能和 token 数据；普通生成的空输出、答案抽取缺失、截断、拒答文本和长度分组；单字母单选题的选项分布与混淆对；完整重复采样题组的成败稳定性；Agent 的工具与过程发现。
- **过程发现**：“样本详情”可按问题类型、相关组件和过程发现筛选。从问题类型进入匹配样本集合，点击代表样本直接定位该条；可返回结果诊断，样本链接也支持刷新和直接打开。徽标只定位有明确工具调用 ID 的步骤；无法定位的不猜位置。

达到输出上限的统计只采用 predictions 里记录的结束原因，缺失原因的样本不计入分母。达到上限仍可能已给出答案。缺少必要数据的检查会显示“跳过”，不会伪造零发生率。连续分数没有明确阈值时，任务成败可能未知。较弱子集只对有声明区间与方向的指标比较。

任务成败优先使用 benchmark 的明确结果，再使用配置的评分阈值，主评分为准确率或精确匹配时，才使用记录的 0/1 结果判定；辅助指标是否连续不影响主评分的判断。连续评分没有通过线时保持未知，不自动进入默认的未通过样本分析。可用 `trace={'outcome_thresholds': {'your_benchmark': {'quality': 0.8}}}` 指定阈值，比较方向沿用指标声明或 `outcome_directions`；也可用 `scope='all'` 或 `samples` 指定分析范围。非文本输出不参与空文本、答案抽取和文本长度等统计，避免把图片或音频回答误报为空。

默认比例达到 10% 且有事件时标为“注意”，由 `warning_rate` 配置；较弱子集须比总体低至少 10 个百分点且不少于 20 条，由 `weak_subset_gap`、`weak_subset_min_samples` 配置。其余事实标为统计；P1 不自动生成严重级别结论。选择题要求至少 90% 的标准答案为单个大写字母；多选或选项全文格式暂不统计，缺失选项映射时不判断某个字母是否越界。

问题类型可扩展，不能穷举所有失败原因。“暂无规则线索”保留未被当前规则识别的失败。即便答对，也可能出现工具错误或风险线索；工具参数来自先前返回值、状态改变调用本身都不证明越权。抽取值为空也不证明原答案正确或评分器出错。搜索无结果会保留为工具报错，不直接归为参数错误。**发现线索不等于证明失败根因**；下文介绍的模型诊断也只是待复核的解释。

## 产物与缓存

```text
<run>/analysis/<model>/<benchmark>/diagnosis.json
<run>/analysis/<model>/<benchmark>/samples.jsonl
<run>/analysis/<model>/<benchmark>/diagnose/diagnosis.json
<run>/analysis/<model>/<benchmark>/diagnose/samples.jsonl
<run>/reports/<model>/<benchmark>.json  # 仅更新 analysis 字段
<run>/reports/report.html
```

输出包含统计定义、覆盖量、代表样本、规则发现及分析用量（`facts` 为零）。导出证据按预算截断并脱敏；原始结果不复制到诊断文件。

两档分别保存。有效的模型分析会继续显示，更新基础分析保留已有模型提议。评测数据变化后会提示过期，更新基础分析后可先查看最新事实，再按需重跑模型分析。

缓存检查 reviews、相关 predictions、报告评分元数据和使用到的配置；排除自身的 `analysis` 回写。来源或样本产物发生变化时，Web 标记过期并阻止用旧记录跳转。诊断不会改成绩，在线诊断异常也不会使已完成的评测失败。旧报告只有 `Report.analysis` 时，“结果诊断”与 HTML 仍能显示原摘要。

## 模型分析、证据与统计口径

`analysis_report=True` 继续默认免费 `facts`。`diagnose` 在基础分析上对所选样本进行模型分析、校验引用，并生成带引用的总结与检查建议；必须显式配置分析模型，不自动沿用评测或评分模型。模型分析目前为实验性能力。设置 `include_trace=False` 可仅发送任务和最终回答，省略中间消息和工具返回。

```python
from evalscope.api.analysis import AnalysisReportConfig

analysis = AnalysisReportConfig(
    level='diagnose',
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
- **语义支持**：引用存在不代表它支持解释。模型结论仍需独立复核。只要有一条引用校验失败，就不展示整条提议或用于总结。评分记录里明确保存了排除原因或执行错误时，两档分析都直接展示脱敏后的原文。该样本不调用模型，也不占用选样名额；没有候选样本时，不请求模型总结。调用/格式失败或低置信度的诊断也不计为有效模型提议。标准答案和分数保留原样。

事实统计始终处理全部可读取样本，不受模型样本上限影响。默认按未通过选样时，只有达到输出上限是唯一异常，且评分明确记录抽取答案为空，才从模型选样中跳过该样本。需要继续追查时可指定样本或使用 `scope='all'`。只有能纳入至少一条有效样本提议时，才请求模型总结；规则统计已在报告里。规则观察和模型提议分别保存；模型分析不覆盖观察分组。同一样本可出现多个观察，各项计数不可相加。引用通过校验仅表示可定位，不代表原因已确认；不再输出总体原因比例或“未解释率”估计。

模型分析默认只选任务未通过样本。`scope` 可设为 `failed`（默认）、`unscored`（评分不可用）、`risk`（安全风险信号）、`process`（已通过且存在过程问题）或 `all`（全部样本）。也可用 `samples=[{'subset': 'main', 'index': 12}]` 明确指定样本，此时覆盖 scope；不存在的样本会在请求前报错。

`max_diagnosed_samples` 是每个 benchmark 的费用保护上限：正整数表示最多分析多少条，Python 的 `None`／JSON 的 `null` 表示全量分析当前范围。默认 50，不是质量保证，也不是整次评测的总预算。名额在子集间轮流分配；子集多于名额时仍可执行，报告记录实际检查数和符合条件数，不外推未检查样本的原因。

```bash
# Requires the explicit model configuration in analysis.json
evalscope analyze --run-dir outputs/<run> --analysis-config analysis.json --max-samples 100
evalscope analyze --run-dir outputs/<run> --analysis-config analysis.json --max-samples all
```

完整内容能放入 `max_input_chars` 时全部保留。超出时优先保留题目、最终回答和相关发现的前后文；报告标明上下文是否省略，不能据此认定某个步骤不存在或错误未被恢复。

设置 `include_trace=True` 时，已保存且未被标记为受保护的推理文本也作为独立来源提供，可引用原文并定位到消息/步骤；推理内容仍是模型陈述，不是独立事实。图片、音频、视频、内部数据和受保护推理不发送给诊断模型；样本明确标记内容未完整参与分析。不恢复隐藏推理，也不根据缺失媒体推断内容。

样本请求按模型、生成设置、协议、提示词及实际来源内容缓存，可中断续跑；不合规回复和传输错误不缓存为成功。`--force` 会重新发起模型请求。报告区分本次调用量和复用量；传输重试由已有模型层处理，分析器不额外重试。调用数是分析器请求数，不包含供应商内部重试；token 为接口实际返回用量，缺失时单独计数，费用需按供应商账单核对。

独立标注的比较工具：

```bash
python -m evalscope.analysis.validation --samples <samples.jsonl> --labels <labels.jsonl> --output validation.json
```

标注使用 `DiagnosticLabel`，每份文件只对应一个 run/model/benchmark。`rule_decisions` 显式标注检测项的正反例；未标的检测项不假定为负例。工具分别报告分类一致性、归因混淆、根因步骤、引用可定位性和已复核证据支持率。模型辅助标注须保留标注者身份，不能当作人工准确率验收。

Web 中的“分析信息”按模式展示：`facts`（基础分析）显示全量规则统计范围、未调用模型和生成时间；`diagnose`（模型分析）增加实际选样范围、选取数、引用校验、模型、本次调用与缓存用量，以及总结生成状态。总结参考全部有效模型提议的问题类型与数量，在 `max_input_chars` 内轮流选取各类型的代表证据，不受展示示例数 `max_examples` 限制；未传入全部解释时记录范围。数量只描述已检查样本中的模型提议，不外推总体原因。主要发现与排查建议各最多 3 条，分开展示，点击统计引用可查看对应依据。引用可定位与原因已确认是两回事。页面不再单独提供“已通过样本的过程线索”入口。

诊断页左侧只显示问题名称与数量；“其他未通过样本”表示尚未匹配到已知记录异常，不代表没有模型提议。嵌入样本使用顶部导航和步骤定位菜单，完整问答与轨迹仍与“样本详情”共用查看器。总结中的解释引用具体子集、样本及可定位的步骤；点击后打开该样本，进入完整详情时保留步骤位置。统计引用展开对应统计及分母。“分析信息”继续保留。旧版分组引用不会自动猜成样本引用，需重新运行 `evalscope analyze` 生成新诊断；`facts` 仍不调用模型。

旧实验配置中的 `level='full'` 请改为 `level='diagnose'`，旧诊断产物需重新生成。样本分析与总结分别缓存；总结失败会保留已获得的样本分析，续跑时复用有效缓存。
