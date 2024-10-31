from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any


@dataclass
class TestsetGenerationArguments:
    docs: List[str] = field(default_factory=list)
    test_size: int = 10
    output_file: str = 'outputs/testset.json'
    knowledge_graph: str = 'outputs/knowledge_graph.json'
    """
    For local LLM support, you can use the following fields:
        model_name_or_path: str
        model_revision: str = "master"
        template_type: str = "default"
        generation_config: Optional[Dict]

    For API LLM support, you can use the following fields:
        model_name="gpt-4o-mini"
        api_base: str = "",
        api_key: Optional[str] = None
    """
    generator_llm: Dict = field(default_factory=dict)
    embeddings: Dict = field(default_factory=dict)
    distribution: str = field(
        default_factory=lambda: {'simple': 0.5, 'multi_context': 0.4, 'reasoning': 0.1}
    )
    # For LLM based evaluation
    # available: ['english', 'hindi', 'marathi', 'chinese', 'spanish', 'amharic', 'arabic',
    # 'armenian', 'bulgarian', 'urdu', 'russian', 'polish', 'persian', 'dutch', 'danish',
    # 'french', 'burmese', 'greek', 'italian', 'japanese', 'deutsch', 'kazakh', 'slovak']
    language: str = 'english'


@dataclass
class EvaluationArguments:
    testset_file: str
    critic_llm: Dict = field(default_factory=dict)
    embeddings: Dict = field(default_factory=dict)
    metrics: List[str] = field(
        default_factory=lambda: ['answer_relevancy', 'faithfulness']
    )
    # For LLM based evaluation
    # available: ['english', 'hindi', 'marathi', 'chinese', 'spanish', 'amharic', 'arabic',
    # 'armenian', 'bulgarian', 'urdu', 'russian', 'polish', 'persian', 'dutch', 'danish',
    # 'french', 'burmese', 'greek', 'italian', 'japanese', 'deutsch', 'kazakh', 'slovak']
    language: str = 'english'
