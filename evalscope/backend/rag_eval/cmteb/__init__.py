from typing import TYPE_CHECKING

from evalscope.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from evalscope.backend.rag_eval.cmteb.arguments import EvalArguments, ModelArguments
    from evalscope.backend.rag_eval.cmteb.base import TaskBase
    from evalscope.backend.rag_eval.cmteb.task_template import one_stage_eval, two_stage_eval
    from evalscope.backend.rag_eval.cmteb.tasks import (
        AFQMC,
        ATEC,
        BQ,
        CLS_CLASSIFICATION,
        CLS_CLUSTERING,
        CLS_CUSTOM,
        CLS_DICT,
        CLS_PAIR_CLASSIFICATION,
        CLS_RERANKING,
        CLS_RETRIEVAL,
        CLS_STS,
        LCQMC,
        PAWSX,
        QBQTC,
        STSB,
        CLSClusteringFastP2P,
        CLSClusteringFastS2S,
        CmedqaRetrieval,
        CMedQAv1,
        CMedQAv2,
        Cmnli,
        CovidRetrieval,
        CustomRetrieval,
        DuRetrieval,
        EcomRetrieval,
        IFlyTek,
        JDReview,
        MedicalRetrieval,
        MMarcoReranking,
        MMarcoRetrieval,
        MultilingualSentiment,
        Ocnli,
        OnlineShopping,
        T2Reranking,
        T2Retrieval,
        ThuNewsClusteringFastP2P,
        ThuNewsClusteringFastS2S,
        TNews,
        VideoRetrieval,
        Waimai,
    )

else:
    _task_exports = [
        'AFQMC',
        'ATEC',
        'BQ',
        'CLS_CLASSIFICATION',
        'CLS_CUSTOM',
        'CLS_DICT',
        'CLS_CLUSTERING',
        'CLS_PAIR_CLASSIFICATION',
        'CLS_RERANKING',
        'CLS_RETRIEVAL',
        'CLS_STS',
        'CMedQAv1',
        'CMedQAv2',
        'CLSClusteringFastP2P',
        'CLSClusteringFastS2S',
        'CmedqaRetrieval',
        'Cmnli',
        'CovidRetrieval',
        'CustomRetrieval',
        'DuRetrieval',
        'EcomRetrieval',
        'IFlyTek',
        'JDReview',
        'LCQMC',
        'MMarcoReranking',
        'MMarcoRetrieval',
        'MedicalRetrieval',
        'MultilingualSentiment',
        'Ocnli',
        'OnlineShopping',
        'PAWSX',
        'QBQTC',
        'STSB',
        'T2Reranking',
        'T2Retrieval',
        'TNews',
        'ThuNewsClusteringFastP2P',
        'ThuNewsClusteringFastS2S',
        'VideoRetrieval',
        'Waimai',
    ]
    _import_structure = {
        'arguments': [
            'EvalArguments',
            'ModelArguments',
        ],
        'base': [
            'TaskBase',
        ],
        'task_template': [
            'one_stage_eval',
            'two_stage_eval',
        ],
        'tasks': _task_exports,
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
