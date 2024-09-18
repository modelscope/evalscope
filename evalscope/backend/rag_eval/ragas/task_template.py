import os
from evalscope.backend.rag_eval import EmbeddingModel, LLM
from evalscope.utils.logger import get_logger
from .arguments import TestsetGenerationArguments, EvaluationArguments

logger = get_logger()


def testset_generation(args: TestsetGenerationArguments) -> None:
    from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
    from ragas.testset.generator import TestsetGenerator
    from ragas.testset.evolutions import (
        simple,
        reasoning,
        multi_context
    )

    # load data
    file_path = args.docs
    loader = UnstructuredFileLoader(file_path)
    data = loader.load()

    # generator with models
    generator_llm = LLM(**args.generator_llm)
    critic_llm = LLM(**args.critic_llm)
    embeddings = EmbeddingModel.from_pretrained(**args.embeddings)

    # Change resulting question type distribution
    distributions = {
        simple: args.distribution["simple"],
        multi_context: args.distribution["multi_context"],
        reasoning: args.distribution["reasoning"],
    }

    generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings)

    # runconfig = RunConfig(timeout=30, max_retries=1, max_wait=30, max_workers=1)
    testset = generator.generate_with_langchain_docs(
        data, args.test_size, distributions, with_debugging_logs=True, is_async=False
    )

    # save file
    testset_df = testset.to_pandas()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    testset_df.to_json(args.output_file, indent=4, index=False)


def rag_eval(
    args: EvaluationArguments,
) -> None:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_recall,
        context_precision,
        answer_correctness,
    )
    from evalscope.backend.rag_eval import EmbeddingModel, LLM

    llm = LLM(**args.critic_llm)
    embedding = EmbeddingModel.from_pretrained(**args.embeddings)

    dataset = Dataset.from_json(args.testset_file)

    score = evaluate(
        dataset,
        metrics=[answer_correctness, faithfulness],
        llm=llm,
        embeddings=embedding
    )
    print(score.to_pandas().to_markdown())
