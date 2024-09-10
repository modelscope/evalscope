from mteb.abstasks.AbsTaskSTS import AbsTaskSTS
from mteb.abstasks.TaskMetadata import TaskMetadata

class ATEC(AbsTaskSTS):
    metadata = TaskMetadata(
        name="ATEC",
        description="ATEC NLP sentence pair similarity competition",
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["zh"],
        main_score="cosine_spearman",
        min_score=0,
        max_score=1,
        dataset={
            "path": "C-MTEB/ATEC",
            "revision": None
        }
    )

class BQ(AbsTaskSTS):
    metadata = TaskMetadata(
        name="BQ",
        description="Bank Question Semantic Similarity",
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["zh"],
        main_score="cosine_spearman",
        min_score=0,
        max_score=1,
        dataset={
            "path": "C-MTEB/BQ",
            "revision": None
        }
    )

class LCQMC(AbsTaskSTS):
    metadata = TaskMetadata(
        name="LCQMC",
        description="A large-scale Chinese question matching corpus.",
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["zh"],
        main_score="cosine_spearman",
        min_score=0,
        max_score=1,
        dataset={
            "path": "C-MTEB/LCQMC",
            "revision": None
        }
    )

class PAWSX(AbsTaskSTS):
    metadata = TaskMetadata(
        name="PAWSX",
        description="Translated PAWS evaluation pairs",
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["zh"],
        main_score="cosine_spearman",
        min_score=0,
        max_score=1,
        dataset={
            "path": "C-MTEB/PAWSX",
            "revision": None
        }
    )

class STSB(AbsTaskSTS):
    metadata = TaskMetadata(
        name="STSB",
        description="Translate STS-B into Chinese",
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["zh"],
        main_score="cosine_spearman",
        min_score=0,
        max_score=5,
        dataset={
            "path": "C-MTEB/STSB",
            "revision": None
        }
    )

class AFQMC(AbsTaskSTS):
    metadata = TaskMetadata(
        name="AFQMC",
        description="Ant Financial Question Matching Corpus",
        type="STS",
        category="s2s",
        eval_splits=["validation"],
        eval_langs=["zh"],
        main_score="cosine_spearman",
        min_score=0,
        max_score=1,
        dataset={
            "path": "C-MTEB/AFQMC",
            "revision": None
        }
    )

class QBQTC(AbsTaskSTS):
    metadata = TaskMetadata(
        name="QBQTC",
        description="QQ Browser Query Title Corpus",
        reference="https://github.com/CLUEbenchmark/QBQTC/tree/main/dataset",
        type="STS",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["zh"],
        main_score="cosine_spearman",
        min_score=0,
        max_score=1,
        dataset={
            "path": "C-MTEB/QBQTC",
            "revision": None
        }
    )
