import os
from datasets import Dataset
from evalscope.backend.rag_eval import EmbeddingModel, LLM
from evalscope.utils.logger import get_logger
from .arguments import EvaluationArguments

logger = get_logger()


def rag_eval(
    args: EvaluationArguments,
) -> None:

    from ragas import evaluate, RunConfig
    import importlib

    def dynamic_import(module_name, *function_names):
        # 动态导入指定模块
        module = importlib.import_module(module_name)

        functions = [getattr(module, name) for name in function_names]
        return functions

    llm = LLM.load(**args.critic_llm)
    embedding = EmbeddingModel.load(**args.embeddings)

    # load dataset
    dataset = Dataset.from_json(args.testset_file)

    runconfig = RunConfig(timeout=30, max_retries=1, max_wait=30, max_workers=1)
    score = evaluate(
        dataset,
        metrics=dynamic_import("ragas.metrics", *args.metrics),
        llm=llm,
        embeddings=embedding,
        run_config=runconfig,
    )
    score_df = score.to_pandas()
    # logger.info(score_df.to_string())

    output_path = args.testset_file.split(".")[0] + "_score.json"
    score_df.to_json(output_path, indent=4, index=False, orient="records")

    logger.info(f"Eval score saved to {output_path}")
