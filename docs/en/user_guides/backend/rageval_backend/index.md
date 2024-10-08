(rageval)=
# RAGEval
:::{toctree}
:hidden:
mteb.md
ragas.md
:::

After the research on RAG evaluation-related topics [investigated](../../../blog/RAG/RAG_Evaluation.md), this project now supports two methods for evaluating RAG: Independent Evaluation and End-to-End Evaluation:

- **Independent Evaluation Method**: Evaluates the Retrieval Module and Generation Module separately. The evaluation metrics for the Retrieval Module include **Hit Rate, Mean Reciprocal Rank (MRR), Normalized Discounted Cumulative Gain (NDCG), Precision**, etc. These metrics are used to measure the effectiveness of the system in ranking items based on queries or tasks. The evaluation for the Generation Module is usually conducted during the End-to-End Evaluation. The metrics focus primarily on contextual relevance, measuring the relevance of retrieved documents to the query.

- **End-to-End Evaluation Method**: Evaluates the final response generated by the RAG model for a given input, including the relevance and alignment of the model-generated answer with the input query. From the perspective of content generation objectives, evaluation can be divided into unannotated and annotated content. Metrics for unannotated content include **Answer Fidelity, Answer Relevance, Harmlessness**, while metrics for annotated content include Accuracy and Exact Match (EM).

This framework supports independent evaluation of the retrieval module using [MTEB/CMTEB](mteb.md) and end-to-end evaluation using [RAGAS](ragas.md).


::::{grid} 2
:::{grid-item-card}  MTEB
:link: mteb
:link-type: ref

Independent evaluation of the retrieval module, supporting embedding models and reranker models.
:::

:::{grid-item-card}  RAGAS
:link: ragas
:link-type: ref

End-to-end evaluation, supporting the automatic generation of evaluation sets.
:::
::::