import asyncio
import os

import pandas as pd
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from tqdm import tqdm

from evalscope.backend.rag_eval import LLM, ChatOpenAI, EmbeddingModel
from evalscope.backend.rag_eval.ragas.arguments import TestsetGenerationArguments
from evalscope.utils.logger import get_logger
from .translate_prompt import translate_prompts

logger = get_logger()


def get_transform(llm, embedding, language):
    """
    Creates and returns a default set of transforms for processing a knowledge graph.
    """
    from ragas.testset.transforms.engine import Parallel
    from ragas.testset.transforms.extractors import (
        EmbeddingExtractor,
        HeadlinesExtractor,
        SummaryExtractor,
    )
    from ragas.testset.transforms.extractors.llm_based import NERExtractor, ThemesExtractor
    from ragas.testset.transforms.relationship_builders import (
        CosineSimilarityBuilder,
        OverlapScoreBuilder,
    )
    from ragas.testset.transforms.splitters import HeadlineSplitter
    from ragas.testset.transforms.filters import CustomNodeFilter
    from ragas.testset.graph import NodeType
    from ragas.utils import num_tokens_from_string

    def summary_filter(node):
        return (node.type == NodeType.DOCUMENT and num_tokens_from_string(node.properties['page_content']) > 500)

    summary_extractor = SummaryExtractor(llm=llm, filter_nodes=lambda node: summary_filter(node))
    ner_extractor = NERExtractor(llm=llm, filter_nodes=lambda node: node.type == NodeType.CHUNK)
    theme_extractor = ThemesExtractor(llm=llm)
    headline_extractor = HeadlinesExtractor(llm=llm)

    asyncio.run(
        translate_prompts(
            prompts=[
                summary_extractor,
                theme_extractor,
                ner_extractor,
                headline_extractor,
            ],
            target_lang=language,
            llm=llm,
            adapt_instruction=True,
        ))

    splitter = HeadlineSplitter(min_tokens=500)

    summary_emb_extractor = EmbeddingExtractor(
        embedding_model=embedding,
        property_name='summary_embedding',
        embed_property_name='summary',
        filter_nodes=lambda node: summary_filter(node),
    )

    cosine_sim_builder = CosineSimilarityBuilder(
        property_name='summary_embedding',
        new_property_name='summary_similarity',
        threshold=0.7,
        filter_nodes=lambda node: summary_filter(node),
    )

    ner_overlap_sim = OverlapScoreBuilder(threshold=0.01, filter_nodes=lambda node: node.type == NodeType.CHUNK)

    node_filter = CustomNodeFilter(llm=llm, filter_nodes=lambda node: node.type == NodeType.CHUNK)

    transforms = [
        headline_extractor,
        splitter,
        summary_extractor,
        node_filter,
        Parallel(summary_emb_extractor, theme_extractor, ner_extractor),
        Parallel(cosine_sim_builder, ner_overlap_sim),
    ]

    return transforms


def get_distribution(llm, distribution, language):
    from ragas.testset.synthesizers.multi_hop import (
        MultiHopAbstractQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer,
    )
    from ragas.testset.synthesizers.single_hop.specific import (
        SingleHopSpecificQuerySynthesizer, )

    single_hop = SingleHopSpecificQuerySynthesizer(llm=llm)
    multi_hop_abs = MultiHopAbstractQuerySynthesizer(llm=llm)
    multi_hop_spec = MultiHopSpecificQuerySynthesizer(llm=llm)

    asyncio.run(
        translate_prompts(
            prompts=[
                single_hop,
                multi_hop_abs,
                multi_hop_spec,
            ],
            target_lang=language,
            llm=llm,
            adapt_instruction=True,
        ))

    mapping = {
        'simple': single_hop,
        'multi_context': multi_hop_abs,
        'reasoning': multi_hop_spec,
    }

    return [(mapping[key], distribution[key]) for key in mapping if key in distribution]


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
    from evalscope.backend.rag_eval.ragas.prompts.persona_prompt import PersonaGenerationPromptZH
    from ragas.testset.persona import generate_personas_from_kg, PersonaGenerationPrompt
    from ragas.testset.graph import Node

    def filter(node: Node) -> bool:
        if (node.type.name == 'DOCUMENT' and node.properties.get('summary_embedding') is not None):
            return True
        else:
            return False

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

    return generate_personas_from_kg(
        llm=llm,
        kg=kg,
        num_personas=3,
        persona_generation_prompt=persona_prompt,
        filter_fn=filter,
    )


def load_data(file_path):
    from langchain_community.document_loaders import UnstructuredFileLoader

    loader = UnstructuredFileLoader(file_path, mode='elements')
    data = loader.load()
    return data


def generate_testset(args: TestsetGenerationArguments) -> None:

    from ragas.testset import TestsetGenerator
    from ragas import RunConfig

    # load data
    documents = load_data(args.docs)

    # generator with models
    generator_llm = LLM.load(**args.generator_llm)
    embeddings = EmbeddingModel.load(**args.embeddings)

    wrapped_llm = LangchainLLMWrapper(generator_llm)
    wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # Change resulting question type distribution
    distributions = get_distribution(wrapped_llm, args.distribution, args.language)

    run_config = RunConfig(timeout=600, max_retries=3, max_wait=120, max_workers=1, log_tenacity=True)
    # get transforms
    transforms = get_transform(
        wrapped_llm,
        wrapped_embeddings,
        args.language,
    )

    # get knowledge graph
    knowledge_graph = get_knowledge_graph(documents, transforms, args.knowledge_graph, run_config)

    persona_list = get_persona(llm=wrapped_llm, kg=knowledge_graph, language=args.language)

    generator = TestsetGenerator(llm=wrapped_llm, knowledge_graph=knowledge_graph, persona_list=persona_list)

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
