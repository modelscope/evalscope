import asyncio
import os
from datasets import Dataset

from evalscope.backend.rag_eval import LLM, EmbeddingModel
from evalscope.backend.rag_eval.ragas.tasks.translate_prompt import translate_prompts
from evalscope.utils.logger import get_logger
from .arguments import EvaluationArguments

logger = get_logger()


def rag_eval(args: EvaluationArguments, ) -> None:

    import importlib
    from ragas import RunConfig, evaluate
    from ragas.llms import LangchainLLMWrapper

    def dynamic_import(*function_names):
        functions = []
        for name in function_names:
            module = importlib.import_module('ragas.metrics')
            functions.append(getattr(module, name)())
        return functions

    llm = LLM.load(**args.critic_llm)
    embedding = EmbeddingModel.load(**args.embeddings)

    # load metrics
    metrics = dynamic_import(*args.metrics)
    asyncio.run(
        translate_prompts(
            prompts=metrics,
            target_lang=args.language,
            llm=LangchainLLMWrapper(llm),
            adapt_instruction=True,
        ))
    # load dataset
    dataset = Dataset.from_json(args.testset_file)

    # evaluate
    runconfig = RunConfig(timeout=600, max_retries=2, max_wait=60, max_workers=1)
    score = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embedding,
        run_config=runconfig,
    )
    score_df = score.to_pandas()
    logger.info(score_df)

    output_path = args.testset_file.replace('.json', '_score.json')
    score_df.to_json(output_path, indent=4, index=False, orient='records', force_ascii=False)

    logger.info(f'Eval score saved to {output_path}')
