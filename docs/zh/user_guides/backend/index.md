# 评测后端

:::{toctree}
:hidden:

opencompass_backend.md
vlmevalkit_backend.md
rageval_backend/index.md
:::

EvalScope支持多个评测后端，以集成多个工具实现不同的评测任务，具体介绍如下：

::::{grid} 2

:::{grid-item-card} OpenCompass
:link: opencompass
:link-type: ref

+++
通过EvalScope发起轻量化的OpenCompass评估任务，支持多种LLM评测任务。
:::

:::{grid-item-card} VLMEvalKit
:link: vlmeval
:link-type: ref

+++
通过EvalScope发起轻量化的VLMEvalKit评估任务，支持多种VLM评测任务。
:::

:::{grid-item-card} RAGEval
:link: rageval
:link-type: ref

+++
支持RAG独立评估和端到端评估：支持使用MTEB/CMTEB进行检索模块的独立评测，以及使用RAGAS进行端到端评测。
:::

::::
