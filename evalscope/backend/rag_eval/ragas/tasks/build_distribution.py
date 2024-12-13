import asyncio
from ragas.llms import BaseRagasLLM
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.synthesizers.multi_hop import MultiHopAbstractQuerySynthesizer, MultiHopSpecificQuerySynthesizer
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer

from .translate_prompt import translate_prompts


def default_query_distribution(llm: BaseRagasLLM, kg: KnowledgeGraph, language: str):
    """
    Generates a distribution of query synthesizers, optionally tailored to a
        specific knowledge graph (KG) and translated into a given language.
    """

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

    default_queries = [
        single_hop,
        multi_hop_abs,
        multi_hop_spec,
    ]
    if kg is not None:
        available_queries = []
        for query in default_queries:
            if query.get_node_clusters(kg):
                available_queries.append(query)
    else:
        available_queries = default_queries

    return [(query, 1 / len(available_queries)) for query in available_queries]
