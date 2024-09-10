from mteb import AbsTask
from modelscope import MsDataset
import datasets

prompt_dict = {
    "TNews": None,
    "IFlyTek": None,
    "MultilingualSentiment": None,
    "JDReview": None,
    "OnlineShopping": None,
    "Waimai": None,
    "CLSClusteringS2S": None,
    "CLSClusteringP2P": None,
    "ThuNewsClusteringS2S": None,
    "ThuNewsClusteringP2P": None,
    "Ocnli": None,
    "Cmnli": None,
    "T2Reranking": None,
    "MMarcoReranking": None,
    "CMedQAv1": None,
    "CMedQAv2": None,
    "T2Retrieval": None,
    "MMarcoRetrieval": None,
    "DuRetrieval": None,
    "CovidRetrieval": None,
    "CmedqaRetrieval": None,
    "EcomRetrieval": None,
    "MedicalRetrieval": None,
    "VideoRetrieval": None,
    "ATEC": None,
    "BQ": None,
    "LCQMC": None,
    "PAWSX": None,
    "STSB": None,
    "AFQMC": None,
    "QBQTC": None,
}


def load_data(self, **kwargs):
    """Load dataset from hub
    Patch for loading from modelscope
    """
    if self.data_loaded:
        return
    hub = self.metadata.dataset.get("hub", "modelscope")
    path = self.metadata.dataset.get("path", None)
    assert path is not None, "path must be specified in dataset"

    if hub == "modelscope":
        self.dataset = MsDataset.load(path)
    else:
        self.dataset = datasets.load_dataset(**self.metadata.dataset)  # type: ignore
    self.dataset_transform()
    self.data_loaded = True


AbsTask.load_data = load_data
