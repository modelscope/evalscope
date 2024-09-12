import os
import torch
import mteb
from evalscope.backend.rag_eval import EmbeddingModel
from evalscope.backend.rag_eval import cmteb
from mteb.task_selection import results_to_dataframe

model_name = "Jerry0/m3e-base"
# model_name = "../models/embedding/MiniCPM-Embedding"
# model_name = "Xorbits/bge-reranker-base"
# model_name = "OpenBMB/MiniCPM-Reranker"


model = EmbeddingModel.from_pretrained(
    model_name_or_path=model_name,
    is_cross_encoder=False,
    pooling_mode="cls",
    max_seq_length=512,
    model_kwargs=dict(torch_dtype=torch.bfloat16),
    prompts=cmteb.INSTRUCTIONS,
    hub="modelscope",
)


def evaluate_models(dual_encoder_model_name, cross_encoder_model_name):

    tasks = cmteb.TaskBase.get_tasks(task_names=["T2Retrieval"], instructions={})

    # 加载模型
    # dual_encoder = EmbeddingModel.from_pretrained(dual_encoder_model_name, is_cross_encoder=False)
    cross_encoder = EmbeddingModel.from_pretrained(
        cross_encoder_model_name,
        is_cross_encoder=True,
        encode_kwargs=dict(
            show_progress_bar=True, batch_size=32
        ),
    )
    cross_encoder.prompt = cmteb.INSTRUCTIONS["T2Retrieval"]
    
    for task in tasks:
        # 初始化评估工具
        evaluation = mteb.MTEB(tasks=task)

        # 第一次评估：双编码器
        evaluation.run(
            dual_encoder,
            save_predictions=True,
            output_folder="outputs/stage1",
            overwrite_results=True,
            encode_kwargs=dict(show_progress_bar=True, batch_size=32, normalize_embeddings=False),
            hub="modelscope",
            limits=100,
        )

        evaluation.run(
            cross_encoder,
            top_k=5,
            save_predictions=True,
            output_folder="outputs/stage2",
            previous_results=f"outputs/stage1/{task.metadata.name}_default_predictions.json",
            overwrite_results=True,
            hub="modelscope",
            limits=100,
        )


# 使用示例
evaluate_models(
    dual_encoder_model_name="Jerry0/m3e-base",
    cross_encoder_model_name="OpenBMB/MiniCPM-Reranker",
)
