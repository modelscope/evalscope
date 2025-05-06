(rageval)=

# RAGEval
:::{toctree}
:hidden:
:maxdepth: 1

mteb.md
clip_benchmark.md
ragas.md
:::

本项目支持RAG和多模态RAG的独立评测(Independent Evaluation)和端到端评测(End-to-End Evaluation)：

- 独立评测方法：单独评测检索模块，其中检索模块评测指标包括指标包括 **命中率(Hit Rate)、平均排名倒数(Mean Reciprocal Rank, MRR)、归一化折扣累积增益(Normalized Discounted Cumulative Gain, NDCG)、准确率(Precision)** 等，这些指标用于测量系统在根据查询或任务排名项目方面的有效性。

- 端到端评测方法：评测RAG模型对给定输入生成的最终响应，包括模型生成答案与输入查询的相关性和对齐程度。从内容生成目标视角来评测可以将评测划分为**无参考答案**和**有参考答案**：无参考答案评测指标包括**上下文相关性(Context Relevance)、忠实度(Faithfulness)** 等；而有参考答案评测指标包括**准确率(Accuracy)、BLEU、ROUGE等**。

```{seealso}
RAG评测相关[调研](../../../blog/RAG/RAG_Evaluation.md)
```

本框架支持使用[MTEB/CMTEB](mteb.md)进行文本检索模块的独立评测，使用[CLIP Benchmark](clip_benchmark.md)进行多模态图文检索模块的独立评测，以及使用[RAGAS](ragas.md)进行RAG和多模态RAG端到端生成评测。

::::{grid} 3
:::{grid-item-card}  MTEB/CMTEB
:link: mteb
:link-type: ref

进行检索模块的独立评测，支持embedding模型和reranker模型。
:::

:::{grid-item-card}  CLIP Benchmark
:link: clip_benchmark
:link-type: ref

进行多模态图文检索模块的独立评测，支持CLIP模型。
:::

:::{grid-item-card}  RAGAS
:link: ragas
:link-type: ref

进行RAG和多模态RAG端到端生成评测，且支持自动生成评测集。
:::
::::
