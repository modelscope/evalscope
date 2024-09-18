from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any


@dataclass
class TestsetGenerationArguments:
    docs: List[str] = field(default_factory=list)
    test_size: int = 10
    output_file: str = "outputs/testset.json"
    generator_llm: Dict = field(default_factory=dict)
    critic_llm: Dict = field(default_factory=dict)
    embeddings: Dict = field(default_factory=dict)
    distribution: str = field(
        default_factory=lambda: {"simple": 0.5, "multi_context": 0.4, "reasoning": 0.1}
    )


@dataclass
class EvaluationArguments:
    testset_file: str
    critic_llm: Dict = field(default_factory=dict)
    embeddings: Dict = field(default_factory=dict)
    metrics: List[str] = field(
        default_factory=lambda: ["answer_relevancy", "faithfulness"]
    )
