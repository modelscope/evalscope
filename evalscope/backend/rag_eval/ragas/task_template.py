import os
from evalscope.backend.rag_eval import EmbeddingModel, LLM
from evalscope.utils.logger import get_logger

logger = get_logger()

def testset_generation(
    model_args,
    eval_args,
) -> None:
    from langchain_community.document_loaders import UnstructuredFileLoader
    markdown_path = ["README.md", "docs/en/user_guides/opencompass_backend.md"]
    loader = UnstructuredFileLoader(markdown_path)
    data = loader.load()

    from evalscope.backend.rag_eval import EmbeddingModel, LLM
    from ragas.testset.generator import TestsetGenerator
    from ragas.testset.evolutions import simple, reasoning, multi_context, ComplexEvolution
    from ragas.testset.filters import QuestionFilter, EvolutionFilter, NodeFilter
    from ragas.testset.prompts import (
        context_scoring_prompt,
        evolution_elimination_prompt,
        filter_question_prompt,
    )

    # remove demonstrations from examples
    for prompt in [
        context_scoring_prompt,
        evolution_elimination_prompt,
        filter_question_prompt,
    ]:
        prompt.examples = []


    # generator with openai models
    generator_llm = LLM(model_name_or_path="qwen/Qwen2-7B-Instruct", template_type="qwen")
    critic_llm = LLM(
        model_name_or_path="QwenCollection/Ragas-critic-llm-Qwen1.5-GPTQ",
        template_type="qwen",
    )
    embeddings = EmbeddingModel.from_pretrained(
        model_name_or_path="Jerry0/m3e-base", is_cross_encoder=False
    )

    qa_filter = QuestionFilter(critic_llm, filter_question_prompt)
    node_filter = NodeFilter(critic_llm, context_scoring_prompt=context_scoring_prompt)
    evolution_filter = EvolutionFilter(critic_llm, evolution_elimination_prompt)

    # customise the filters

    # Change resulting question type distribution
    distributions = {simple: 0.5, multi_context: 0.4, reasoning: 0.1}

    for evolution in distributions:
        if evolution.question_filter is None:
            evolution.question_filter = qa_filter
        if evolution.node_filter is None:
            evolution.node_filter = node_filter

        if isinstance(evolution, ComplexEvolution):
            if evolution.evolution_filter is None:
                evolution.evolution_filter = evolution_filter

    generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings)

    # runconfig = RunConfig(timeout=30, max_retries=1, max_wait=30, max_workers=1)
    # use generator.generate_with_llamaindex_docs if you use llama-index as document loader
    testset = generator.generate_with_langchain_docs(
        data, 10, distributions, with_debugging_logs=True
    )
    testset_df = testset.to_pandas()
    testset_df.to_json("testset.json", indent=4, index=False)


def rag_eval(
    args,
) -> None:
    from datasets import Dataset
    from ragas import evaluate, RunConfig
    from ragas.metrics import faithfulness, answer_correctness
    from evalscope.backend.rag_eval import EmbeddingModel, LLM

    llm = LLM(model_name_or_path="qwen/Qwen2-7B-Instruct", template_type="qwen")
    embedding = EmbeddingModel.from_pretrained(model_name_or_path="Jerry0/m3e-base", is_cross_encoder=False)

    data_samples = {
        'question': ['When was the first super bowl?', 'Who won the most super bowls?'],
        'answer': ['The first superbowl was held on Jan 15, 1967', 'The most super bowls have been won by The New England Patriots'],
        'contexts' : [['The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,'],
        ['The Green Bay Packers...Green Bay, Wisconsin.','The Packers compete...Football Conference']],
        'ground_truth': ['The first superbowl was held on January 15, 1967', 'The New England Patriots have won the Super Bowl a record six times']
    }

    runconfig = RunConfig(timeout=600, max_retries=1, max_wait=600, max_workers=1)

    dataset = Dataset.from_dict(data_samples)

    # score = evaluate(dataset, metrics=[faithfulness], llm=llm, embeddings=embedding, run_config=runconfig)
    score = evaluate(dataset, metrics=[answer_correctness], llm=llm, embeddings=embedding, run_config=runconfig)
    print(score.to_pandas().to_markdown())