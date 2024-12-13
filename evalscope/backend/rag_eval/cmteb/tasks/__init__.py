from .Classification import *
from .Clustering import *
from .CustomTask import *
from .PairClassification import *
from .Reranking import *
from .Retrieval import *
from .STS import *

CLS_CLASSIFICATION = {
    'TNews': TNews,
    'IFlyTek': IFlyTek,
    'MultilingualSentiment': MultilingualSentiment,
    'JDReview': JDReview,
    'OnlineShopping': OnlineShopping,
    'Waimai': Waimai,
}

CLS_CLUSTERING = {
    'CLSClusteringS2S': CLSClusteringFastS2S,
    'CLSClusteringP2P': CLSClusteringFastP2P,
    'ThuNewsClusteringS2S': ThuNewsClusteringFastS2S,
    'ThuNewsClusteringP2P': ThuNewsClusteringFastP2P,
}

CLS_PAIR_CLASSIFICATION = {
    'Ocnli': Ocnli,
    'Cmnli': Cmnli,
}

CLS_RERANKING = {
    'T2Reranking': T2Reranking,
    'MMarcoReranking': MMarcoReranking,
    'CMedQAv1': CMedQAv1,
    'CMedQAv2': CMedQAv2,
}

CLS_RETRIEVAL = {
    'T2Retrieval': T2Retrieval,
    'MMarcoRetrieval': MMarcoRetrieval,
    'DuRetrieval': DuRetrieval,
    'CovidRetrieval': CovidRetrieval,
    'CmedqaRetrieval': CmedqaRetrieval,
    'EcomRetrieval': EcomRetrieval,
    'MedicalRetrieval': MedicalRetrieval,
    'VideoRetrieval': VideoRetrieval,
}

CLS_STS = {
    'ATEC': ATEC,
    'BQ': BQ,
    'LCQMC': LCQMC,
    'PAWSX': PAWSX,
    'STSB': STSB,
    'AFQMC': AFQMC,
    'QBQTC': QBQTC,
}

CLS_CUSTOM = {
    'CustomRetrieval': CustomRetrieval,
}

CLS_DICT = {
    **CLS_CLASSIFICATION,
    **CLS_CLUSTERING,
    **CLS_PAIR_CLASSIFICATION,
    **CLS_RERANKING,
    **CLS_RETRIEVAL,
    **CLS_STS,
}
