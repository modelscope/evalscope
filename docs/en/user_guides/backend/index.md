# Evaluation Backends
:::{toctree}
:hidden:
opencompass_backend.md
vlmevalkit_backend.md
rageval_backend/index.md
:::

EvalScope supports multiple evaluation backends to integrate various tools for different evaluation tasks, detailed as follows:

::::{grid} 2
:::{grid-item-card} OpenCompass
:link: opencompass
:link-type: ref

+++
Initiate lightweight OpenCompass evaluation tasks through EvalScope, supporting various LLM evaluation tasks.
:::

:::{grid-item-card} VLMEvalKit
:link: vlmeval
:link-type: ref

+++
Initiate lightweight VLMEvalKit evaluation tasks through EvalScope, supporting various VLM evaluation tasks.
:::

:::{grid-item-card} RAGEval
:link: rageval
:link-type: ref

+++
Supports independent and end-to-end evaluation for RAG: allows independent evaluation of the retrieval module using MTEB/CMTEB, as well as end-to-end evaluation using RAGAS.
:::
::::