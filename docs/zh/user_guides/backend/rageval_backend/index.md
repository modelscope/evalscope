(rageval)=

# RAGEval
:::{toctree}
:hidden:
:maxdepth: 1

mteb.md
ragas.md
:::

经过RAG评测相关的[调研](../../../blog/RAG/RAG_Evaluation.md)，本项目现支持两种评估RAG的方法：独立评估(Independent Evaluation)和端到端评估(End-to-End Evaluation)：

- 独立评估方法：单独评估检索模块(Retrieval Module)和生成模块(Generation Module)，其中检索模块评估指标包括指标包括 **命中率(Hit Rate)、平均排名倒数(Mean Reciprocal Rank, MRR)、归一化折扣累积增益(Normalized Discounted Cumulative Gain, NDCG)、准确率(Precision)** 等，这些指标用于测量系统在根据查询或任务排名项目方面的有效性。而生成模块评估通常在端到端评估中进行。评估指标主要关注上下文相关性，测量检索到的文档与查询问题的相关性​​。

- 端到端评估方法：评估RAG模型对给定输入生成的最终响应，包括模型生成答案与输入查询的相关性和对齐程度。从内容生成目标视角来评估可以将评估划分为未标记内容和已标记内容。未标记内容评估指标包括 **答案忠实度(Answer Fidelity)、答案相关性(Answer Relevance)、无害性(Harmlessness)** 等，而已标记内容评估指标包括准确率(Accuracy)和精确匹配(Exact Match, EM)。

本框架支持使用[MTEB/CMTEB](mteb.md)进行检索模块的独立评测，以及使用[RAGAS](ragas.md)进行端到端评测。

::::{grid} 2
:::{grid-item-card}  MTEB
:link: mteb
:link-type: ref

进行检索模块的独立评测，支持embedding模型和reranker模型。
:::
:::{grid-item-card}  RAGAS
:link: ragas
:link-type: ref

进行端到端评测，支持自动生成评测集。
:::
::::