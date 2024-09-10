from mteb import AbsTaskPairClassification
from mteb.abstasks.TaskMetadata import TaskMetadata

class Ocnli(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="Ocnli",
        description="Original Chinese Natural Language Inference dataset",
        reference="https://arxiv.org/abs/2010.05444",
        category="s2s",
        type="PairClassification",
        eval_splits=["validation"],
        eval_langs=["zh"],
        main_score="ap",
        dataset={
            "path": "C-MTEB/OCNLI",
            "revision": None,
        }
    )

class Cmnli(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="Cmnli",
        description="Chinese Multi-Genre NLI",
        reference="https://huggingface.co/datasets/clue/viewer/cmnli",
        category="s2s",
        type="PairClassification",
        eval_splits=["validation"],
        eval_langs=["zh"],
        main_score="ap",
        dataset={
            "path": "C-MTEB/CMNLI",
            "revision": None,
        }
    )
