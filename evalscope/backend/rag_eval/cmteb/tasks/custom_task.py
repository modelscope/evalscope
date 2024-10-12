from mteb import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class CustomRetrieval(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="CustomRetrieval",
        description="T2Ranking: A large-scale Chinese Benchmark for Passage Ranking",
        reference="",
        dataset={
            "path": "C-MTEB/T2Retrieval",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["text"],
        eval_langs=["cmn-Hans"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation="",
        descriptive_stats={},
    )
