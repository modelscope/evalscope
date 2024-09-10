from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata

class CLSClusteringS2S(AbsTaskClustering):
    metadata = TaskMetadata(
        name="CLSClusteringS2S",
        description="Clustering of titles from CLS dataset. Clustering of 13 sets, based on the main category.",
        reference="https://arxiv.org/abs/2209.05034",
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["zh"],
        main_score="v_measure",
        dataset={
            "path": "C-MTEB/CLSClusteringS2S",
            "revision": None,
        }
    )

class CLSClusteringP2P(AbsTaskClustering):
    metadata = TaskMetadata(
        name="CLSClusteringP2P",
        description="Clustering of titles + abstract from CLS dataset. Clustering of 13 sets, based on the main category.",
        reference="https://arxiv.org/abs/2209.05034",
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["zh"],
        main_score="v_measure",
        dataset={
            "path": "C-MTEB/CLSClusteringP2P",
            "revision": None,
        }
    )

class ThuNewsClusteringS2S(AbsTaskClustering):
    metadata = TaskMetadata(
        name="ThuNewsClusteringS2S",
        description="Clustering of titles from the THUCNews dataset",
        reference="http://thuctc.thunlp.org/",
        type="Clustering",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["zh"],
        main_score="v_measure",
        dataset={
            "path": "C-MTEB/ThuNewsClusteringS2S",
            "revision": None,
        }
    )

class ThuNewsClusteringP2P(AbsTaskClustering):
    metadata = TaskMetadata(
        name="ThuNewsClusteringP2P",
        description="Clustering of titles + abstracts from the THUCNews dataset",
        reference="http://thuctc.thunlp.org/",
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["zh"],
        main_score="v_measure",
        dataset={
            "path": "C-MTEB/ThuNewsClusteringP2P",
            "revision": None,
        }
    )
