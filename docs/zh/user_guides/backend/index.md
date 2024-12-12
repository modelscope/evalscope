# 其他评测后端

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
通过EvalScope发起轻量化的OpenCompass评测任务，支持多种LLM评测任务。
:::

:::{grid-item-card} VLMEvalKit
:link: vlmeval
:link-type: ref

+++
通过EvalScope发起轻量化的VLMEvalKit评测任务，支持多种VLM评测任务。
:::

:::{grid-item-card} RAGEval
:link: rageval
:link-type: ref

+++
支持RAG和图文多模态RAG评测：支持检索模块的独立评测，以及端到端生成评测。
:::

::::
