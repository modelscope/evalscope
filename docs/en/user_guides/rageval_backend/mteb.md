(mteb)=

# MTEB
This framework supports [MTEB](https://github.com/embeddings-benchmark/mteb) and [CMTEB](https://github.com/embeddings-benchmark/mteb), with the following details:

- MTEB (Massive Text Embedding Benchmark) is a large-scale benchmark designed to measure the performance of text embedding models across diverse embedding tasks. MTEB includes 56 datasets covering 8 tasks and supports over 112 different languages. The goal of this benchmark is to assist developers in finding the best text embedding models suitable for various tasks.

- C-MTEB (Chinese Massive Text Embedding Benchmark) is a dedicated evaluation benchmark for Chinese text vectors, built on MTEB, aimed at assessing the performance of Chinese text vector models. C-MTEB collects 35 public datasets and is divided into 6 categories of evaluation tasks, including retrieval, re-ranking, semantic text similarity (STS), classification, pair classification, and clustering.

## Supported Datasets
Here is an overview of the available tasks and datasets in C-MTEB:

| Name | Hub Link | Description | Type | Category | Number of Test Samples |
|-----|-----|---------------------------|-----|-----|-----|
| [T2Retrieval](https://arxiv.org/abs/2304.03679) | [C-MTEB/T2Retrieval](https://modelscope.cn/datasets/C-MTEB/T2Retrieval) | T2Ranking: A large-scale Chinese paragraph ranking benchmark | Retrieval | s2p | 24,832 |
| [MMarcoRetrieval](https://github.com/unicamp-dl/mMARCO) | [C-MTEB/MMarcoRetrieval](https://modelscope.cn/datasets/C-MTEB/MMarcoRetrieval) | mMARCO is the multilingual version of the MS MARCO paragraph ranking dataset | Retrieval | s2p | 7,437 |
| [DuRetrieval](https://aclanthology.org/2022.emnlp-main.357.pdf) | [C-MTEB/DuRetrieval](https://modelscope.cn/datasets/C-MTEB/DuRetrieval) | A large-scale Chinese web search engine paragraph retrieval benchmark | Retrieval | s2p | 4,000 |
| [CovidRetrieval](https://aclanthology.org/2022.emnlp-main.357.pdf) | [C-MTEB/CovidRetrieval](https://modelscope.cn/datasets/C-MTEB/CovidRetrieval) | COVID-19 news articles | Retrieval | s2p | 949 |
| [CmedqaRetrieval](https://aclanthology.org/2022.emnlp-main.357.pdf) | [C-MTEB/CmedqaRetrieval](https://modelscope.cn/datasets/C-MTEB/CmedqaRetrieval) | Online medical consultation texts | Retrieval | s2p | 3,999 |
| [EcomRetrieval](https://arxiv.org/abs/2203.03367) | [C-MTEB/EcomRetrieval](https://modelscope.cn/datasets/C-MTEB/EcomRetrieval) | Paragraph retrieval dataset collected from Alibaba e-commerce search engine systems | Retrieval | s2p | 1,000 |
| [MedicalRetrieval](https://arxiv.org/abs/2203.03367) | [C-MTEB/MedicalRetrieval](https://modelscope.cn/datasets/C-MTEB/MedicalRetrieval) | Paragraph retrieval dataset collected from Alibaba medical search engine systems | Retrieval | s2p | 1,000 |
| [VideoRetrieval](https://arxiv.org/abs/2203.03367) | [C-MTEB/VideoRetrieval](https://modelscope.cn/datasets/C-MTEB/VideoRetrieval) | Paragraph retrieval dataset collected from Alibaba video search engine systems | Retrieval | s2p | 1,000 |
| [T2Reranking](https://arxiv.org/abs/2304.03679) | [C-MTEB/T2Reranking](https://modelscope.cn/datasets/C-MTEB/T2Reranking) | T2Ranking: A large-scale Chinese paragraph ranking benchmark | Re-ranking | s2p | 24,382 |
| [MMarcoReranking](https://github.com/unicamp-dl/mMARCO) | [C-MTEB/MMarco-reranking](https://modelscope.cn/datasets/C-MTEB/Mmarco-reranking) | mMARCO is the multilingual version of the MS MARCO paragraph ranking dataset | Re-ranking | s2p | 7,437 |
| [CMedQAv1](https://github.com/zhangsheng93/cMedQA) | [C-MTEB/CMedQAv1-reranking](https://modelscope.cn/datasets/C-MTEB/CMedQAv1-reranking) | Chinese community medical Q&A | Re-ranking | s2p | 2,000 |
| [CMedQAv2](https://github.com/zhangsheng93/cMedQA2) | [C-MTEB/CMedQAv2-reranking](https://modelscope.cn/datasets/C-MTEB/C-MTEB/CMedQAv2-reranking) | Chinese community medical Q&A | Re-ranking | s2p | 4,000 |
| [Ocnli](https://arxiv.org/abs/2010.05444) | [C-MTEB/OCNLI](https://modelscope.cn/datasets/C-MTEB/OCNLI) | Original Chinese natural language inference dataset | Pair Classification | s2s | 3,000 |
| [Cmnli](https://modelscope.cn/datasets/clue/viewer/cmnli) | [C-MTEB/CMNLI](https://modelscope.cn/datasets/C-MTEB/CMNLI) | Chinese multi-class natural language inference | Pair Classification | s2s | 139,000 |
| [CLSClusteringS2S](https://arxiv.org/abs/2209.05034) | [C-MTEB/CLSClusteringS2S](https://modelscope.cn/datasets/C-MTEB/C-MTEB/CLSClusteringS2S) | Clustering titles from the CLS dataset. Clustering based on 13 sets of main categories. | Clustering | s2s | 10,000 |
| [CLSClusteringP2P](https://arxiv.org/abs/2209.05034) | [C-MTEB/CLSClusteringP2P](https://modelscope.cn/datasets/C-MTEB/CLSClusteringP2P) | Clustering titles + abstracts from the CLS dataset. Clustering based on 13 sets of main categories. | Clustering | p2p | 10,000 |
| [ThuNewsClusteringS2S](http://thuctc.thunlp.org/) | [C-MTEB/ThuNewsClusteringS2S](https://modelscope.cn/datasets/C-MTEB/ThuNewsClusteringS2S) | Clustering titles from the THUCNews dataset | Clustering | s2s | 10,000 |
| [ThuNewsClusteringP2P](http://thuctc.thunlp.org/) | [C-MTEB/ThuNewsClusteringP2P](https://modelscope.cn/datasets/C-MTEB/ThuNewsClusteringP2P) | Clustering titles + abstracts from the THUCNews dataset | Clustering | p2p | 10,000 |
| [ATEC](https://github.com/IceFlameWorm/NLP_Datasets/tree/master/ATEC) | [C-MTEB/ATEC](https://modelscope.cn/datasets/C-MTEB/ATEC) | ATEC NLP Sentence Pair Similarity Competition | STS | s2s | 20,000 |
| [BQ](https://huggingface.co/datasets/shibing624/nli_zh) | [C-MTEB/BQ](https://modelscope.cn/datasets/C-MTEB/BQ) | Banking Question Semantic Similarity | STS | s2s | 10,000 |
| [LCQMC](https://huggingface.co/datasets/shibing624/nli_zh) | [C-MTEB/LCQMC](https://modelscope.cn/datasets/C-MTEB/LCQMC) | Large-scale Chinese Question Matching Corpus | STS | s2s | 12,500 |
| [PAWSX](https://arxiv.org/pdf/1908.11828.pdf) | [C-MTEB/PAWSX](https://modelscope.cn/datasets/C-MTEB/PAWSX) | Translated PAWS evaluation pairs | STS | s2s | 2,000 |
| [STSB](https://github.com/pluto-junzeng/CNSD) | [C-MTEB/STSB](https://modelscope.cn/datasets/C-MTEB/STSB) | Translated STS-B into Chinese | STS | s2s | 1,360 |
| [AFQMC](https://github.com/CLUEbenchmark/CLUE) | [C-MTEB/AFQMC](https://modelscope.cn/datasets/C-MTEB/AFQMC) | Ant Financial Question Matching Corpus | STS | s2s | 3,861 |
| [QBQTC](https://github.com/CLUEbenchmark/QBQTC) | [C-MTEB/QBQTC](https://modelscope.cn/datasets/C-MTEB/QBQTC) | QQ Browser Query Title Corpus | STS | s2s | 5,000 |
| [TNews](https://github.com/CLUEbenchmark/CLUE) | [C-MTEB/TNews-classification](https://modelscope.cn/datasets/C-MTEB/TNews-classification) | News Short Text Classification | Classification | s2s | 10,000 |
| [IFlyTek](https://github.com/CLUEbenchmark/CLUE) | [C-MTEB/IFlyTek-classification](https://modelscope.cn/datasets/C-MTEB/IFlyTek-classification) | Long Text Classification of Application Descriptions | Classification | s2s | 2,600 |
| [Waimai](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/waimai_10k/intro.ipynb) | [C-MTEB/waimai-classification](https://modelscope.cn/datasets/C-MTEB/waimai-classification) | Sentiment Analysis of User Reviews on Food Delivery Platforms | Classification | s2s | 1,000 |
| [OnlineShopping](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/online_shopping_10_cats/intro.ipynb) | [C-MTEB/OnlineShopping-classification](https://modelscope.cn/datasets/C-MTEB/OnlineShopping-classification) | Sentiment Analysis of User Reviews on Online Shopping Websites | Classification | s2s | 1,000 |
| [MultilingualSentiment](https://github.com/tyqiangz/multilingual-sentiment-datasets) | [C-MTEB/MultilingualSentiment-classification](https://modelscope.cn/datasets/C-MTEB/MultilingualSentiment-classification) | A set of multilingual sentiment datasets grouped into three categories: positive, neutral, negative | Classification | s2s | 3,000 |
| [JDReview](https://huggingface.co/datasets/kuroneko5943/jd21) | [C-MTEB/JDReview-classification](https://modelscope.cn/datasets/C-MTEB/JDReview-classification) | Reviews of iPhone | Classification | s2s | 533 |

For retrieval tasks, a sample of 100,000 candidates (including the ground truth) is drawn from the entire corpus to reduce inference costs.

## Environment Setup
Install dependencies
```bash
pip install mteb
```

## Configure Evaluation Parameters
The framework supports two evaluation modes: single-stage evaluation and two-stage evaluation:
- **Single-stage evaluation**: Directly use the model for prediction and calculate metrics.
- **Two-stage evaluation**: Use the model for prediction, then apply a retrieval model for reranking, and calculate metrics.

### Single-stage Evaluation
Example configuration file:
```python
one_stage_task_cfg = {
    "eval_backend": "RAGEval",
    "eval_config": {
        "tool": "MTEB",
        "model": [
            {
                "model_name_or_path": "AI-ModelScope/m3e-base",
                "pooling_mode": None,
                "max_seq_length": 512,
                "prompt": "",
                "model_kwargs": {"torch_dtype": "auto"},
                "encode_kwargs": {
                    "batch_size": 128,
                },
            }
        ],
        "eval": {
            "tasks": [
                "TNews",
                "CLSClusteringS2S",
                "T2Reranking",
                "T2Retrieval",
                "ATEC",
            ],
            "verbosity": 2,
            "output_folder": "outputs",
            "overwrite_results": True,
            "limits": 500,
        },
    },
}
```
#### Parameter Explanation
- `eval_backend`: Default value is `RAGEval`, indicating the use of the RAGEval evaluation backend.
- `eval_config`: A dictionary containing the following fields:
    - `tool`: Evaluation tool, using `MTEB`.
    - `model`: List of model configurations, **only one model can be placed here for single-stage evaluation**, containing the following fields:
        - `model_name_or_path`: `str` Model name or path
        - `is_cross_encoder`: `bool` Indicates whether the model is a cross-encoder, default is False.  
        - `pooling_mode`: `Optional[str]` Pooling mode, possible values are: “cls”, “lasttoken”, “max”, “mean”, “mean_sqrt_len_tokens” or “weightedmean”.  
        - `max_seq_length`: `int` Maximum sequence length, default is 512.  
        - `prompt`: `str` Prompt for large language model-based assistance.  
        - `model_kwargs`: `dict` Model keyword arguments, default value is `{"torch_dtype": "auto"}`.  
        - `config_kwargs`: `Dict[str, Any]` Configuration keyword arguments, default is an empty dictionary.  
        - `encode_kwargs`: `dict` Encoding keyword arguments, default is:  
            ```python  
            {  
                "show_progress_bar": True,  
                "batch_size": 32
            }  
            ```  
        - `hub`: `str` Model source, can be "modelscope" or "huggingface".  
    - `eval`: Dictionary containing the following fields:
        - `tasks`: `List[str]` Task names  
        - `verbosity`: `int` Level of detail, ranging from 0-3  
        - `output_folder`: `str` Output folder, default is "outputs"  
        - `overwrite_results`: `bool` Indicates whether to overwrite results, default is True  
        - `limits`: `Optional[int]` Limiting the number of samples, default is None  
        - `hub`: `str` Model source, can be "modelscope" or "huggingface"  
        - `top_k`: `int` Select the top K results, used in retrieval tasks  


### Two-stage Evaluation
Example configuration file: first perform retrieval, then reranking:
```python
two_stage_task_cfg = {
    "eval_backend": "RAGEval",
    "eval_config": {
        "tool": "MTEB",
        "model": [
            {
                "model_name_or_path": "AI-ModelScope/m3e-base",
                "is_cross_encoder": False,
                "max_seq_length": 512,
                "model_kwargs": {"torch_dtype": "auto"},
                "encode_kwargs": {
                    "batch_size": 64,
                },
            },
            {
                "model_name_or_path": "OpenBMB/MiniCPM-Reranker",
                "is_cross_encoder": True,
                "max_seq_length": 512,
                "prompt": "Generate a retrieval representation for this question",
                "model_kwargs": {"torch_dtype": "auto"},
                "encode_kwargs": {
                    "batch_size": 32,
                },
            },
        ],
        "eval": {
            "tasks": ["T2Retrieval"],
            "verbosity": 2,
            "output_folder": "outputs",
            "overwrite_results": True,
            "limits": 100,
        },
    },
}
```
#### Parameter Explanation
Basic parameters are the same as in single-stage evaluation, differing in the `model` field, where two models are required: the first model for retrieval and the second model for reranking, with the reranking model needing to be a cross encoder.

## Model Evaluation
```python
from evalscope.run import run_task
from evalscope.utils.logger import get_logger
logger = get_logger()

one_stage_task_cfg = one_stage_task_cfg
# or
# two_stage_task_cfg = two_stage_task_cfg

# Run task
run_task(task_cfg=one_stage_task_cfg) # or run_task(task_cfg=two_stage_task_cfg)
```