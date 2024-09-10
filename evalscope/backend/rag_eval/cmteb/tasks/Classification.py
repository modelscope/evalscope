from mteb import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class TNews(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TNews",
        description="Short Text Classification for News",
        reference="https://www.cluebenchmarks.com/introduce.html",
        type="Classification",
        category="s2s",
        eval_splits=["validation"],
        eval_langs=["zho"],
        main_score="accuracy",
        samples_per_label=32,
        dataset={
            "path": "C-MTEB/TNews-classification",
            "revision": None,
        },
    )


class IFlyTek(AbsTaskClassification):
    metadata = TaskMetadata(
        name="IFlyTek",
        description="Long Text classification for the description of Apps",
        reference="https://www.cluebenchmarks.com/introduce.html",
        type="Classification",
        category="s2s",
        eval_splits=["validation"],
        eval_langs=["zho"],
        main_score="accuracy",
        samples_per_label=32,
        n_experiments=5,
        dataset={
            "path": "C-MTEB/IFlyTek-classification",
            "revision": None,
        },
    )


class MultilingualSentiment(AbsTaskClassification):
    metadata = TaskMetadata(
        name="MultilingualSentiment",
        description="A collection of multilingual sentiments datasets grouped into 3 classes -- positive, neutral, negative",
        reference="https://github.com/tyqiangz/multilingual-sentiment-datasets",
        type="Classification",
        category="s2s",
        eval_splits=["validation"],
        eval_langs=["zho"],
        main_score="accuracy",
        samples_per_label=32,
        dataset={
            "path": "C-MTEB/MultilingualSentiment-classification",
            "revision": None,
        },
    )


class JDReview(AbsTaskClassification):
    metadata = TaskMetadata(
        name="JDReview",
        description="review for iphone",
        reference=None,  # 因为原始代码中没有引用，如果有可以进行修改
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["zho"],
        main_score="accuracy",
        samples_per_label=32,
        dataset={
            "path": "C-MTEB/JDReview-classification",
            "revision": None,
        },
    )


class OnlineShopping(AbsTaskClassification):
    metadata = TaskMetadata(
        name="OnlineShopping",
        description="Sentiment Analysis of User Reviews on Online Shopping Websites",
        reference=None,
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["zho"],
        main_score="accuracy",
        samples_per_label=32,
        dataset={
            "path": "C-MTEB/OnlineShopping-classification",
            "revision": None,
        },
    )


class Waimai(AbsTaskClassification):
    metadata = TaskMetadata(
        name="Waimai",
        description="Sentiment Analysis of user reviews on takeaway platforms",
        reference=None, 
        type="Classification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["zho"],
        main_score="accuracy",
        samples_per_label=32,
        dataset={
            "path": "C-MTEB/waimai-classification",
            "revision": None,
        },
    )
