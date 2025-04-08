import os
import pandas as pd
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from tqdm import tqdm

from evalscope.backend.rag_eval import LLM, ChatOpenAI, EmbeddingModel
from evalscope.backend.rag_eval.ragas.arguments import TestsetGenerationArguments
from evalscope.utils.logger import get_logger
from .build_distribution import default_query_distribution
from .build_transform import default_transforms

logger = get_logger()


def get_knowledge_graph(documents, transforms, local_file, run_config):
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType
    from ragas.testset.transforms import apply_transforms

    if os.path.exists(local_file):
        logger.info(f'Loading knowledge graph from {local_file}')
        return KnowledgeGraph.load(local_file)
    # convert the documents to Ragas nodes
    nodes = []
    for doc in documents:
        node = Node(
            type=NodeType.DOCUMENT,
            properties={
                'page_content': doc.page_content,
                'document_metadata': doc.metadata,
            },
        )
        nodes.append(node)

    kg = KnowledgeGraph(nodes=nodes)

    # apply transforms and update the knowledge graph
    apply_transforms(kg, transforms, run_config=run_config)

    # save the knowledge graph
    output_path = os.path.dirname(local_file)
    os.makedirs(output_path, exist_ok=True)
    kg.save(local_file)
    logger.info(f'Knowledge graph saved to {local_file}')
    return kg


def get_persona(llm, kg, language):
    from ragas.testset.persona import PersonaGenerationPrompt, generate_personas_from_kg

    from evalscope.backend.rag_eval.ragas.prompts.persona_prompt import PersonaGenerationPromptZH

    if language == 'chinese':
        persona_prompt = PersonaGenerationPromptZH()
    else:
        persona_prompt = PersonaGenerationPrompt()
    # NOTE: can't translate this yet
    # asyncio.run(
    #     translate_prompts(
    #         prompts=[persona_prompt],
    #         target_lang=language,
    #         llm=llm,
    #         adapt_instruction=True,
    #     ))

    return generate_personas_from_kg(llm=llm, kg=kg, num_personas=3, persona_generation_prompt=persona_prompt)


def load_data(file_path):
    import nltk
    from langchain_unstructured import UnstructuredLoader

    if nltk.data.find('taggers/averaged_perceptron_tagger_eng') is False:
        # need to download nltk data for the first time
        nltk.download('averaged_perceptron_tagger_eng')

    loader = UnstructuredLoader(file_path)
    data = loader.load()
    return data


def generate_testset(args: TestsetGenerationArguments) -> None:

    from ragas import RunConfig
    from ragas.testset import TestsetGenerator

    # load data
    documents = load_data(args.docs)

    # generator with models
    generator_llm = LLM.load(**args.generator_llm)
    embeddings = EmbeddingModel.load(**args.embeddings)

    wrapped_llm = LangchainLLMWrapper(generator_llm)
    wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # get transforms
    transforms = default_transforms(
        documents,
        wrapped_llm,
        wrapped_embeddings,
        args.language,
    )

    run_config = RunConfig(timeout=600, max_retries=10, max_wait=120, max_workers=1, log_tenacity=True)
    # get knowledge graph
    knowledge_graph = get_knowledge_graph(documents, transforms, args.knowledge_graph, run_config)
    # get persona
    persona_list = get_persona(llm=wrapped_llm, kg=knowledge_graph, language=args.language)

    # Change resulting question type distribution
    distributions = default_query_distribution(wrapped_llm, knowledge_graph, args.language)

    # generate testset
    generator = TestsetGenerator(
        llm=wrapped_llm, embedding_model=wrapped_embeddings, knowledge_graph=knowledge_graph, persona_list=persona_list)

    testset = generator.generate(
        testset_size=args.test_size,
        query_distribution=distributions,
        run_config=run_config,
        with_debugging_logs=True,
        raise_exceptions=True,
    )

    # save file
    testset_df = testset.to_pandas()
    output_path = os.path.dirname(args.output_file)
    os.makedirs(output_path, exist_ok=True)
    testset_df.to_json(args.output_file, indent=4, index=False, orient='records', force_ascii=False)

    # get answer
    testset_with_answer = get_answer(testset_df, generator_llm, args.language)
    testset_with_answer.to_json(
        args.output_file.replace('.json', '_with_answer.json'),
        indent=4,
        index=False,
        orient='records',
        force_ascii=False,
    )


def get_answer(testset_df, generator_llm, language: None):
    template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Answer in {language}.
Question: {question}
Context: {contexts}
Answer:
"""

    items = []
    for i in tqdm(range(len(testset_df)), desc='Generating Answers'):
        row = testset_df.iloc[i]
        question = row['user_input']
        contexts = '\n'.join(row['reference_contexts'])

        # Combine question and contexts as input for the LLM
        input_text = template.format(language=language, question=question, contexts=contexts)

        # Generate the answer using the generator LLM
        answer = generator_llm.invoke(input_text)
        if isinstance(generator_llm, ChatOpenAI):
            answer = answer.content
        items.append({
            'user_input': question,
            'retrieved_contexts': row['reference_contexts'],
            'response': answer,
            'reference': row['reference'],
        })

    return pd.DataFrame.from_dict(items)
