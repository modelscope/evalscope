(mteb)=

# MTEB
This framework supports [MTEB](https://github.com/embeddings-benchmark/mteb) and [CMTEB](https://github.com/FlagOpen/FlagEmbedding/blob/master/C_MTEB/README.md), with the following details:

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

```{seealso}
- [Datasets Supported by CMTEB](https://github.com/FlagOpen/FlagEmbedding/blob/master/research/C_MTEB/README.md)
- [Datasets Supported by MTEB](https://embeddings-benchmark.github.io/mteb/overview/)
```

## Environment Setup
Install dependencies
```bash
pip install "mteb<2"
```

## Configure Evaluation Parameters
The framework supports two evaluation modes: single-stage evaluation and two-stage evaluation:
- Single-Stage Evaluation: Directly use the model for prediction and compute metrics. Supports tasks such as retrieval, re-ranking, and classification for embedding models.

- Two-Stage Evaluation: Use the model for retrieval first, then use the model for re-ranking, and compute metrics. Supports re-ranking models.

### Single-stage Evaluation
Example configuration file:
```python
one_stage_task_cfg = {
    "work_dir": "outputs",
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
            "overwrite_results": True,
            "topk": 10,
            "limits": 500,
        },
    },
}
```

### Evaluation of API Model Services

When using a remote API model service, the configuration file example is as follows:

```python
from evalscope import TaskConfig

task_cfg = TaskConfig(
    eval_backend='RAGEval',  # Specifies the evaluation backend to use
    eval_config={
        'tool': 'MTEB',  # The evaluation tool to be used
        'model': [
            {
                'model_name': 'text-embedding-v3',  # Name of the model
                'api_base': 'https://dashscope.aliyuncs.com/compatible-mode/v1',  # Base URL for the API service
                'api_key': env.get('DASHSCOPE_API_KEY', 'EMPTY'),  # API key for authentication
                'dimensions': 1024,  # Dimensionality of the model's output
                'encode_kwargs': {  # Encoding arguments
                    'batch_size': 10,  # Size of the batch to process at once
                },
            }
        ],
        'eval': {
            'tasks': [
                'T2Retrieval',  # Task or tasks to evaluate
            ],
            'verbosity': 2,  # Level of detail in evaluation output
            'overwrite_results': True,  # Whether to overwrite existing results
            'limits': 30,  # Limit on the number of items to evaluate
        },
    },
)
```

### Two-stage Evaluation
Example configuration file: first perform retrieval, then reranking:
```python
two_stage_task_cfg = {
    "work_dir": "outputs",
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
            "overwrite_results": True,
            "topk": 5,
            "limits": 100,
        },
    },
}
```
## Parameter Explanation

- `eval_backend`: Default value is `RAGEval`, indicating the use of the RAGEval evaluation backend.
- `eval_config`: A dictionary containing the following fields:
    - `tool`: Evaluation tool, using `MTEB`.
    - `model`: List of model configurations. **For single-stage evaluation, only one model can be placed; for two-stage evaluation, two models are passed in, with the first model used for retrieval and the second model used for reranking**, including the following fields:
      - **For locally loaded model support**:
        - `model_name_or_path`: `str`: Model name or path, supports automatic model download from the ModelScope repository.
        - `is_cross_encoder`: `bool`: Whether the model is a cross encoder, default is False; for reranking models, set to `True`.
        - `pooling_mode`: `Optional[str]`: Pooling mode, default is `mean`. Options are: "cls", "lasttoken", "max", "mean", "mean_sqrt_len_tokens", or "weightedmean". For `bge` series models, set to "cls".
        - `max_seq_length`: `int`: Maximum sequence length, default is 512.
        - `prompt`: `str` A prompt used before the model for retrieval tasks, with a default value of None.
        - `prompts`: `Dict[str, str]` A dictionary for setting prompts before the model for retrieval tasks, allowing different prompts for various tasks. The default is None, where the key is the task name and the value is the corresponding prompt. This only takes effect when a prompt has not been set.
        - `model_kwargs`: `dict`: Model keyword arguments, default value is `{"torch_dtype": "auto"}`.
        - `config_kwargs`: `Dict[str, Any]`: Configuration keyword arguments, default is an empty dictionary.
        - `encode_kwargs`: `dict`: Encoding keyword arguments, default is:
          ```python
          {
              "show_progress_bar": True,
              "batch_size": 32
          }
          ```
        - `hub`: `str`: Model source, can be "modelscope" or "huggingface".

      - **For remote API model service support**:
        - `model_name`: `str`: Model name.
        - `api_base`: `str`: Model API service address.
        - `api_key`: `str`: Model API key.
        - `dimension`: `int`: Model output dimension.
        - `encode_kwargs`: `dict`: Encoding keyword arguments, default is:
          ```python
          {
              "batch_size": 10
          }
          ```
        - `hub`: `str` Source of the model, can be "modelscope" or "huggingface".  
    - `eval`: A dictionary containing the following fields:
        - `tasks`: `List[str]` Task names, refer to the [task list](#supported-datasets)
        - `top_k`: `int` Select the top K results, for retrieval tasks  
        - `verbosity`: `int` Level of detail, ranging from 0-3  
        - `overwrite_results`: `bool` Whether to overwrite results, default is True  
        - `limits`: `Optional[int]` Limit on the number of samples, default is None; not recommended to set for retrieval tasks
        - `hub`: `str` Source of the dataset, can be "modelscope" or "huggingface"  



## Model Evaluation
```python
from evalscope.run import run_task
from evalscope.utils.logger import get_logger
logger = get_logger()

one_stage_task_cfg = one_stage_task_cfg
# or
# two_stage_task_cfg = two_stage_task_cfg

# Run task
run_task(task_cfg=one_stage_task_cfg) 
# or 
# run_task(task_cfg=two_stage_task_cfg)
```

The following is an example of the output:

**One-Stage Evaluation**

<details><summary>Outputs</summary>

```{code-block} json
:caption: outputs/m3e-base/master/TNews.json

{
  "dataset_revision": "317f262bf1e6126357bbe89e875451e4b0938fe4",
  "evaluation_time": 16.50650382041931,
  "kg_co2_emissions": null,
  "mteb_version": "1.14.15",
  "scores": {
    "validation": [
      {
        "accuracy": 0.4744,
        "f1": 0.44562489526640825,
        "f1_weighted": 0.47540307398330806,
        "hf_subset": "default",
        "languages": [
          "cmn-Hans"
        ],
        "main_score": 0.4744,
        "scores_per_experiment": [
          {
            "accuracy": 0.48,
            "f1": 0.4536376605217497,
            "f1_weighted": 0.47800277926811163
          },
          {
            "accuracy": 0.48,
            "f1": 0.44713633954639176,
            "f1_weighted": 0.4826984434763292
          },
          {
            "accuracy": 0.462,
            "f1": 0.433365706955334,
            "f1_weighted": 0.4640970055245127
          },
          {
            "accuracy": 0.484,
            "f1": 0.4586732839614161,
            "f1_weighted": 0.4857359110392786
          },
          {
            "accuracy": 0.462,
            "f1": 0.4293797541165097,
            "f1_weighted": 0.4632657330831137
          },
          {
            "accuracy": 0.474,
            "f1": 0.44775120246296396,
            "f1_weighted": 0.4737182842092953
          },
          {
            "accuracy": 0.47,
            "f1": 0.4431197566080463,
            "f1_weighted": 0.4714830140231783
          },
          {
            "accuracy": 0.472,
            "f1": 0.44322381694059326,
            "f1_weighted": 0.47100005556357255
          },
          {
            "accuracy": 0.484,
            "f1": 0.45454749692062835,
            "f1_weighted": 0.4856239367465818
          },
          {
            "accuracy": 0.476,
            "f1": 0.44541393463044954,
            "f1_weighted": 0.47840557689910646
          }
        ]
      }
    ]
  },
  "task_name": "TNews"
}
```
</details>

<br>

**Two-stage Evaluation**

<details><summary>first stage</summary>

```{code-block} json
:caption: outputs/stage1/m3e-base/v1/T2Retrieval.json

{
  "dataset_revision": "8731a845f1bf500a4f111cf1070785c793d10e64",
  "evaluation_time": 599.5170171260834,
  "kg_co2_emissions": null,
  "mteb_version": "1.14.15",
  "scores": {
    "dev": [
      {
        "hf_subset": "default",
        "languages": [
          "cmn-Hans"
        ],
        "main_score": 0.73143,
        "map_at_1": 0.22347,
        "map_at_10": 0.63237,
        "map_at_100": 0.67533,
        "map_at_1000": 0.67651,
        "map_at_20": 0.66282,
        "map_at_3": 0.43874,
        "map_at_5": 0.54049,
        "mrr_at_1": 0.7898912852884447,
        "mrr_at_10": 0.8402654617870331,
        "mrr_at_100": 0.8421827758769684,
        "mrr_at_1000": 0.8422583001072272,
        "mrr_at_20": 0.8415411456315557,
        "mrr_at_3": 0.8307469752761716,
        "mrr_at_5": 0.8368029984218875,
        "nauc_map_at_1000_diff1": 0.17749400860890877,
        "nauc_map_at_1000_max": 0.42844516520725967,
        "nauc_map_at_1000_std": 0.18789871694419072,
        "nauc_map_at_100_diff1": 0.17747467084779375,
        "nauc_map_at_100_max": 0.42732291785494575,
        "nauc_map_at_100_std": 0.18694287087286737,
        "nauc_map_at_10_diff1": 0.19976199493034202,
        "nauc_map_at_10_max": 0.3374436217668296,
        "nauc_map_at_10_std": 0.07951451707732717,
        "nauc_map_at_1_diff1": 0.41727578149080663,
        "nauc_map_at_1_max": -0.1402656422184478,
        "nauc_map_at_1_std": -0.26168722519030313,
        "nauc_map_at_20_diff1": 0.1811898211371171,
        "nauc_map_at_20_max": 0.40563441466210043,
        "nauc_map_at_20_std": 0.15927727170010608,
        "nauc_map_at_3_diff1": 0.31255422845809033,
        "nauc_map_at_3_max": 0.007523677231905161,
        "nauc_map_at_3_std": -0.19578481884353466,
        "nauc_map_at_5_diff1": 0.26073699217160473,
        "nauc_map_at_5_max": 0.14665611579604088,
        "nauc_map_at_5_std": -0.09600383298672226,
        "nauc_mrr_at_1000_diff1": 0.3819666309367981,
        "nauc_mrr_at_1000_max": 0.6285393024619401,
        "nauc_mrr_at_1000_std": 0.3294970299417527,
        "nauc_mrr_at_100_diff1": 0.3819436006743644,
        "nauc_mrr_at_100_max": 0.6286346262471935,
        "nauc_mrr_at_100_std": 0.32963045935037844,
        "nauc_mrr_at_10_diff1": 0.3819124721154632,
        "nauc_mrr_at_10_max": 0.6292778905762176,
        "nauc_mrr_at_10_std": 0.3298187966196067,
        "nauc_mrr_at_1_diff1": 0.3862589251033909,
        "nauc_mrr_at_1_max": 0.589976680174432,
        "nauc_mrr_at_1_std": 0.2780515387897469,
        "nauc_mrr_at_20_diff1": 0.38198959771391816,
        "nauc_mrr_at_20_max": 0.6290569436652999,
        "nauc_mrr_at_20_std": 0.3301570340189363,
        "nauc_mrr_at_3_diff1": 0.3825046940733129,
        "nauc_mrr_at_3_max": 0.6282507269128365,
        "nauc_mrr_at_3_std": 0.3260807934869131,
        "nauc_mrr_at_5_diff1": 0.3816317396711923,
        "nauc_mrr_at_5_max": 0.6288655177904692,
        "nauc_mrr_at_5_std": 0.3298854062538469,
        "nauc_ndcg_at_1000_diff1": 0.21319598381916555,
        "nauc_ndcg_at_1000_max": 0.5328295949130256,
        "nauc_ndcg_at_1000_std": 0.2946773445135694,
        "nauc_ndcg_at_100_diff1": 0.2089807772703975,
        "nauc_ndcg_at_100_max": 0.5239397690321543,
        "nauc_ndcg_at_100_std": 0.29123456982125717,
        "nauc_ndcg_at_10_diff1": 0.20555333230027603,
        "nauc_ndcg_at_10_max": 0.44316027023003046,
        "nauc_ndcg_at_10_std": 0.1921835220940756,
        "nauc_ndcg_at_1_diff1": 0.3862589251033909,
        "nauc_ndcg_at_1_max": 0.589976680174432,
        "nauc_ndcg_at_1_std": 0.2780515387897469,
        "nauc_ndcg_at_20_diff1": 0.20754208582741446,
        "nauc_ndcg_at_20_max": 0.4786092392092643,
        "nauc_ndcg_at_20_std": 0.23536973680564616,
        "nauc_ndcg_at_3_diff1": 0.1902823773882388,
        "nauc_ndcg_at_3_max": 0.5400466380622567,
        "nauc_ndcg_at_3_std": 0.2713874990424778,
        "nauc_ndcg_at_5_diff1": 0.18279298790691637,
        "nauc_ndcg_at_5_max": 0.4916119327522918,
        "nauc_ndcg_at_5_std": 0.2375397192963552,
        "nauc_precision_at_1000_diff1": -0.20510380600112582,
        "nauc_precision_at_1000_max": 0.4958820760698651,
        "nauc_precision_at_1000_std": 0.5402465580496146,
        "nauc_precision_at_100_diff1": -0.1994322347949809,
        "nauc_precision_at_100_max": 0.5206762748551254,
        "nauc_precision_at_100_std": 0.5568154081333078,
        "nauc_precision_at_10_diff1": -0.16707155441197413,
        "nauc_precision_at_10_max": 0.5600612846655972,
        "nauc_precision_at_10_std": 0.49419688804691536,
        "nauc_precision_at_1_diff1": 0.3862589251033909,
        "nauc_precision_at_1_max": 0.589976680174432,
        "nauc_precision_at_1_std": 0.2780515387897469,
        "nauc_precision_at_20_diff1": -0.18471041949530417,
        "nauc_precision_at_20_max": 0.5458950955439645,
        "nauc_precision_at_20_std": 0.5355982267058214,
        "nauc_precision_at_3_diff1": -0.03826790088047189,
        "nauc_precision_at_3_max": 0.5833083970750171,
        "nauc_precision_at_3_std": 0.380196662597275,
        "nauc_precision_at_5_diff1": -0.11789367842600275,
        "nauc_precision_at_5_max": 0.5708494593335263,
        "nauc_precision_at_5_std": 0.42860609671688105,
        "nauc_recall_at_1000_diff1": 0.1341309660059583,
        "nauc_recall_at_1000_max": 0.5923755841077135,
        "nauc_recall_at_1000_std": 0.5980459502693942,
        "nauc_recall_at_100_diff1": 0.12181394285840096,
        "nauc_recall_at_100_max": 0.47090136790318127,
        "nauc_recall_at_100_std": 0.3959369184297595,
        "nauc_recall_at_10_diff1": 0.17356300971546512,
        "nauc_recall_at_10_max": 0.25475707245853674,
        "nauc_recall_at_10_std": 0.041819982320384745,
        "nauc_recall_at_1_diff1": 0.41727578149080663,
        "nauc_recall_at_1_max": -0.1402656422184478,
        "nauc_recall_at_1_std": -0.26168722519030313,
        "nauc_recall_at_20_diff1": 0.14273713155999543,
        "nauc_recall_at_20_max": 0.36251116771924663,
        "nauc_recall_at_20_std": 0.1912123941692314,
        "nauc_recall_at_3_diff1": 0.2873719855400218,
        "nauc_recall_at_3_max": -0.041198403561830285,
        "nauc_recall_at_3_std": -0.21921947922872737,
        "nauc_recall_at_5_diff1": 0.23680082643694844,
        "nauc_recall_at_5_max": 0.06580524171324151,
        "nauc_recall_at_5_std": -0.14104561361502632,
        "ndcg_at_1": 0.78989,
        "ndcg_at_10": 0.73143,
        "ndcg_at_100": 0.78829,
        "ndcg_at_1000": 0.80026,
        "ndcg_at_20": 0.75787,
        "ndcg_at_3": 0.7417,
        "ndcg_at_5": 0.72641,
        "precision_at_1": 0.78989,
        "precision_at_10": 0.37304,
        "precision_at_100": 0.04828,
        "precision_at_1000": 0.00511,
        "precision_at_20": 0.21403,
        "precision_at_3": 0.65461,
        "precision_at_5": 0.54942,
        "recall_at_1": 0.22347,
        "recall_at_10": 0.73318,
        "recall_at_100": 0.91093,
        "recall_at_1000": 0.97197,
        "recall_at_20": 0.81286,
        "recall_at_3": 0.46573,
        "recall_at_5": 0.59383
      }
    ]
  },
  "task_name": "T2Retrieval"
}

```
</details>

<details><summary>second stage</summary>

```{code-block} json
:caption: outputs/stage2/jina-reranker-v2-base-multilingual/master/T2Retrieval.json

{
  "dataset_revision": "8731a845f1bf500a4f111cf1070785c793d10e64",
  "evaluation_time": 332.15709686279297,
  "kg_co2_emissions": null,
  "mteb_version": "1.14.15",
  "scores": {
    "dev": [
      {
        "hf_subset": "default",
        "languages": [
          "cmn-Hans"
        ],
        "main_score": 0.661,
        "map_at_1": 0.24264,
        "map_at_10": 0.56291,
        "map_at_100": 0.56291,
        "map_at_1000": 0.56291,
        "map_at_20": 0.56291,
        "map_at_3": 0.4714,
        "map_at_5": 0.56291,
        "mrr_at_1": 0.841969139049623,
        "mrr_at_10": 0.8689147524694633,
        "mrr_at_100": 0.8689147524694633,
        "mrr_at_1000": 0.8689147524694633,
        "mrr_at_20": 0.8689147524694633,
        "mrr_at_3": 0.8664883979192248,
        "mrr_at_5": 0.8689147524694633,
        "nauc_map_at_1000_diff1": 0.12071580301051653,
        "nauc_map_at_1000_max": 0.2536691069727338,
        "nauc_map_at_1000_std": 0.343624832364704,
        "nauc_map_at_100_diff1": 0.12071580301051653,
        "nauc_map_at_100_max": 0.2536691069727338,
        "nauc_map_at_100_std": 0.343624832364704,
        "nauc_map_at_10_diff1": 0.12071580301051653,
        "nauc_map_at_10_max": 0.2536691069727338,
        "nauc_map_at_10_std": 0.343624832364704,
        "nauc_map_at_1_diff1": 0.47964980727810325,
        "nauc_map_at_1_max": -0.08015044571696166,
        "nauc_map_at_1_std": 0.3507257834956417,
        "nauc_map_at_20_diff1": 0.12071580301051653,
        "nauc_map_at_20_max": 0.2536691069727338,
        "nauc_map_at_20_std": 0.343624832364704,
        "nauc_map_at_3_diff1": 0.23481937699306626,
        "nauc_map_at_3_max": 0.10372745264123306,
        "nauc_map_at_3_std": 0.45345158923063256,
        "nauc_map_at_5_diff1": 0.12071580301051653,
        "nauc_map_at_5_max": 0.2536691069727338,
        "nauc_map_at_5_std": 0.343624832364704,
        "nauc_mrr_at_1000_diff1": 0.23393918304502795,
        "nauc_mrr_at_1000_max": 0.8703379129725659,
        "nauc_mrr_at_1000_std": 0.5785333616122065,
        "nauc_mrr_at_100_diff1": 0.23393918304502795,
        "nauc_mrr_at_100_max": 0.8703379129725659,
        "nauc_mrr_at_100_std": 0.5785333616122065,
        "nauc_mrr_at_10_diff1": 0.23393918304502795,
        "nauc_mrr_at_10_max": 0.8703379129725659,
        "nauc_mrr_at_10_std": 0.5785333616122065,
        "nauc_mrr_at_1_diff1": 0.2520016067648708,
        "nauc_mrr_at_1_max": 0.8560897633767299,
        "nauc_mrr_at_1_std": 0.5642467684745208,
        "nauc_mrr_at_20_diff1": 0.23393918304502795,
        "nauc_mrr_at_20_max": 0.8703379129725659,
        "nauc_mrr_at_20_std": 0.5785333616122065,
        "nauc_mrr_at_3_diff1": 0.2343988881957151,
        "nauc_mrr_at_3_max": 0.8695482778251757,
        "nauc_mrr_at_3_std": 0.5799167198804328,
        "nauc_mrr_at_5_diff1": 0.23393918304502795,
        "nauc_mrr_at_5_max": 0.8703379129725659,
        "nauc_mrr_at_5_std": 0.5785333616122065,
        "nauc_ndcg_at_1000_diff1": 0.11252208055013257,
        "nauc_ndcg_at_1000_max": 0.3417865079349515,
        "nauc_ndcg_at_1000_std": 0.3623961771041499,
        "nauc_ndcg_at_100_diff1": 0.11252208055013257,
        "nauc_ndcg_at_100_max": 0.3417865079349515,
        "nauc_ndcg_at_100_std": 0.3623961771041499,
        "nauc_ndcg_at_10_diff1": 0.10015448775533999,
        "nauc_ndcg_at_10_max": 0.3761759074862075,
        "nauc_ndcg_at_10_std": 0.35152523471339914,
        "nauc_ndcg_at_1_diff1": 0.2524564785684737,
        "nauc_ndcg_at_1_max": 0.8566368743831702,
        "nauc_ndcg_at_1_std": 0.5635391925059349,
        "nauc_ndcg_at_20_diff1": 0.11228113618796766,
        "nauc_ndcg_at_20_max": 0.34274993051851965,
        "nauc_ndcg_at_20_std": 0.36216437469674284,
        "nauc_ndcg_at_3_diff1": -0.062134030685870506,
        "nauc_ndcg_at_3_max": 0.7183183844837573,
        "nauc_ndcg_at_3_std": 0.3352626268658533,
        "nauc_ndcg_at_5_diff1": -0.04476981761624879,
        "nauc_ndcg_at_5_max": 0.6272060974309411,
        "nauc_ndcg_at_5_std": 0.21341258393783158,
        "nauc_precision_at_1000_diff1": -0.3554940965683014,
        "nauc_precision_at_1000_max": 0.605443274008298,
        "nauc_precision_at_1000_std": -0.13073611213585504,
        "nauc_precision_at_100_diff1": -0.35549409656830133,
        "nauc_precision_at_100_max": 0.6054432740082977,
        "nauc_precision_at_100_std": -0.13073611213585531,
        "nauc_precision_at_10_diff1": -0.3554940965683011,
        "nauc_precision_at_10_max": 0.6054432740082981,
        "nauc_precision_at_10_std": -0.1307361121358551,
        "nauc_precision_at_1_diff1": 0.2524564785684737,
        "nauc_precision_at_1_max": 0.8566368743831702,
        "nauc_precision_at_1_std": 0.5635391925059349,
        "nauc_precision_at_20_diff1": -0.3554940965683011,
        "nauc_precision_at_20_max": 0.6054432740082981,
        "nauc_precision_at_20_std": -0.1307361121358551,
        "nauc_precision_at_3_diff1": -0.3377658698816377,
        "nauc_precision_at_3_max": 0.6780151277792397,
        "nauc_precision_at_3_std": 0.12291559606586676,
        "nauc_precision_at_5_diff1": -0.3554940965683011,
        "nauc_precision_at_5_max": 0.6054432740082981,
        "nauc_precision_at_5_std": -0.1307361121358551,
        "nauc_recall_at_1000_diff1": 0.1091970342988605,
        "nauc_recall_at_1000_max": 0.18339955163544436,
        "nauc_recall_at_1000_std": 0.30756376767627086,
        "nauc_recall_at_100_diff1": 0.1091970342988605,
        "nauc_recall_at_100_max": 0.18339955163544436,
        "nauc_recall_at_100_std": 0.30756376767627086,
        "nauc_recall_at_10_diff1": 0.1091970342988605,
        "nauc_recall_at_10_max": 0.18339955163544436,
        "nauc_recall_at_10_std": 0.30756376767627086,
        "nauc_recall_at_1_diff1": 0.47964980727810325,
        "nauc_recall_at_1_max": -0.08015044571696166,
        "nauc_recall_at_1_std": 0.3507257834956417,
        "nauc_recall_at_20_diff1": 0.1091970342988605,
        "nauc_recall_at_20_max": 0.18339955163544436,
        "nauc_recall_at_20_std": 0.30756376767627086,
        "nauc_recall_at_3_diff1": 0.22013063499116758,
        "nauc_recall_at_3_max": 0.054749114965246065,
        "nauc_recall_at_3_std": 0.4258163949018153,
        "nauc_recall_at_5_diff1": 0.1091970342988605,
        "nauc_recall_at_5_max": 0.18339955163544436,
        "nauc_recall_at_5_std": 0.30756376767627086,
        "ndcg_at_1": 0.84188,
        "ndcg_at_10": 0.661,
        "ndcg_at_100": 0.6534,
        "ndcg_at_1000": 0.6534,
        "ndcg_at_20": 0.65358,
        "ndcg_at_3": 0.7826,
        "ndcg_at_5": 0.74517,
        "precision_at_1": 0.84188,
        "precision_at_10": 0.27471,
        "precision_at_100": 0.02747,
        "precision_at_1000": 0.00275,
        "precision_at_20": 0.13736,
        "precision_at_3": 0.68633,
        "precision_at_5": 0.54942,
        "recall_at_1": 0.24264,
        "recall_at_10": 0.59383,
        "recall_at_100": 0.59383,
        "recall_at_1000": 0.59383,
        "recall_at_20": 0.59383,
        "recall_at_3": 0.48934,
        "recall_at_5": 0.59383
      }
    ]
  },
  "task_name": "T2Retrieval"
}
```

</details>

## Custom Dataset Evaluation

```{seealso}
[Custom Retrieval Dataset](../../../advanced_guides/custom_dataset/embedding.md)
```
