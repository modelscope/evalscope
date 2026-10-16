# 决策模型评测示例

配合[最佳实践文章](../../../docs/zh/best_practice/decision_evaluation.md)，准备公开内容审核题、中文评论示例，以及不含标准答案的性能测试请求。

在已安装当前源码的 EvalScope 环境中，从仓库根目录执行：

```bash
python examples/decision/comment_filter/prepare_data.py --model "$DECISION_MODEL"
```

`--model` 应与评测命令使用的模型名一致，默认是 `decision-model-preview`。完整的质量评测、Web UI 和性能测试命令见文章。

## 数据内容

- 从 [ModelScope OpenDecisions](https://modelscope.cn/datasets/evalscope/OpenDecisions) 选取 `use_case == 'moderation_trust_safety'` 的公开题，保留原文、选项、答案和逐题来源。本次快照共 59 题。
- `chinese_comments.json` 包含按评论区规则编写的 36 条中文示例：24 条 Noul、6 条 Choice、6 条 Score。规则定义在 `prepare_data.py` 中，可按自己的用途修改。
- 公开数据来自 [jgeuter/OpenDecisions](https://huggingface.co/datasets/jgeuter/OpenDecisions)，许可为 CC BY-SA 4.0。

默认输出到 `outputs/comment-filter/`：

| 路径 | 内容 |
| --- | --- |
| `data/open_decisions_moderation/train.parquet` | 公开题，供 `open_decisions` 读取 |
| `data/open_decisions_records.jsonl` | 同一批公开题的可读原始记录 |
| `data/zh_{noul,choice,score}_test.jsonl` | 中文题，供 `general_decision` 读取 |
| `data/manifest.json` | 来源、样本数及文件校验和 |
| `perf_noul.jsonl` | 24 条中文 Noul 性能请求 |

替换中文题时，可以参考[自定义决策数据格式](../../../docs/zh/advanced_guides/custom_dataset/decision.md)。公开题的评测结果只对应所选题目，完整 OpenDecisions 评测可直接使用 `--datasets open_decisions`。
