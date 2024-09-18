(mteb)=

# MTEB

本框架支持 [MTEB](https://github.com/embeddings-benchmark/mteb) 和 [CMTEB](https://github.com/embeddings-benchmark/mteb)，具体介绍如下：

- MTEB（Massive Text Embedding Benchmark）是一个大规模的基准测试，旨在衡量文本嵌入模型在多样化嵌入任务上的性能。MTEB 包括56个数据集，涵盖8个任务，并且支持超过112种不同的语言。这个基准测试的目标是帮助开发者找到适用于多种任务的最佳文本嵌入模型。

- C-MTEB（Chinese Massive Text Embedding Benchmark）是一个专门针对中文文本向量的评测基准，它基于MTEB构建，旨在评估中文文本向量模型的性能。C-MTEB收集了35个公共数据集，并分为6类评估任务，包括检索（retrieval）、重排序（re-ranking）、语义文本相似度（STS）、分类（classification）、对分类（pair classification）和聚类（clustering）。

## 支持的数据集

以下是C-MTEB中可用任务和数据集的概述：

| 名称 | Hub链接 | 描述 | 类型 | 类别 | 测试样本数量 |
|-----|-----|---------------------------|-----|-----|-----|
| [T2Retrieval](https://arxiv.org/abs/2304.03679) | [C-MTEB/T2Retrieval](https://modelscope.cn/datasets/C-MTEB/T2Retrieval) | T2Ranking：一个大规模的中文段落排序基准 | 检索 | s2p | 24,832 |
| [MMarcoRetrieval](https://github.com/unicamp-dl/mMARCO) | [C-MTEB/MMarcoRetrieval](https://modelscope.cn/datasets/C-MTEB/MMarcoRetrieval) | mMARCO是MS MARCO段落排序数据集的多语言版本 | 检索 | s2p | 7,437 |
| [DuRetrieval](https://aclanthology.org/2022.emnlp-main.357.pdf) | [C-MTEB/DuRetrieval](https://modelscope.cn/datasets/C-MTEB/DuRetrieval) | 一个大规模的中文网页搜索引擎段落检索基准 | 检索 | s2p | 4,000 |
| [CovidRetrieval](https://aclanthology.org/2022.emnlp-main.357.pdf) | [C-MTEB/CovidRetrieval](https://modelscope.cn/datasets/C-MTEB/CovidRetrieval) | COVID-19新闻文章 | 检索 | s2p | 949 |
| [CmedqaRetrieval](https://aclanthology.org/2022.emnlp-main.357.pdf) | [C-MTEB/CmedqaRetrieval](https://modelscope.cn/datasets/C-MTEB/CmedqaRetrieval) | 在线医疗咨询文本 | 检索 | s2p | 3,999 |
| [EcomRetrieval](https://arxiv.org/abs/2203.03367) | [C-MTEB/EcomRetrieval](https://modelscope.cn/datasets/C-MTEB/EcomRetrieval) | 从阿里巴巴电商领域搜索引擎系统收集的段落检索数据集 | 检索 | s2p | 1,000 |
| [MedicalRetrieval](https://arxiv.org/abs/2203.03367) | [C-MTEB/MedicalRetrieval](https://modelscope.cn/datasets/C-MTEB/MedicalRetrieval) | 从阿里巴巴医疗领域搜索引擎系统收集的段落检索数据集 | 检索 | s2p | 1,000 |
| [VideoRetrieval](https://arxiv.org/abs/2203.03367) | [C-MTEB/VideoRetrieval](https://modelscope.cn/datasets/C-MTEB/VideoRetrieval) | 从阿里巴巴视频领域搜索引擎系统收集的段落检索数据集 | 检索 | s2p | 1,000 |
| [T2Reranking](https://arxiv.org/abs/2304.03679) | [C-MTEB/T2Reranking](https://modelscope.cn/datasets/C-MTEB/T2Reranking) | T2Ranking：一个大规模的中文段落排序基准 | 重新排序 | s2p | 24,382 |
| [MMarcoReranking](https://github.com/unicamp-dl/mMARCO) | [C-MTEB/MMarco-reranking](https://modelscope.cn/datasets/C-MTEB/Mmarco-reranking) | mMARCO是MS MARCO段落排序数据集的多语言版本 | 重新排序 | s2p | 7,437 |
| [CMedQAv1](https://github.com/zhangsheng93/cMedQA) | [C-MTEB/CMedQAv1-reranking](https://modelscope.cn/datasets/C-MTEB/CMedQAv1-reranking) | 中文社区医疗问答 | 重新排序 | s2p | 2,000 |
| [CMedQAv2](https://github.com/zhangsheng93/cMedQA2) | [C-MTEB/CMedQAv2-reranking](https://modelscope.cn/datasets/C-MTEB/C-MTEB/CMedQAv2-reranking) | 中文社区医疗问答 | 重新排序 | s2p | 4,000 |
| [Ocnli](https://arxiv.org/abs/2010.05444) | [C-MTEB/OCNLI](https://modelscope.cn/datasets/C-MTEB/OCNLI) | 原始中文自然语言推理数据集 | 配对分类 | s2s | 3,000 |
| [Cmnli](https://modelscope.cn/datasets/clue/viewer/cmnli) | [C-MTEB/CMNLI](https://modelscope.cn/datasets/C-MTEB/CMNLI) | 中文多类别自然语言推理 | 配对分类 | s2s | 139,000 |
| [CLSClusteringS2S](https://arxiv.org/abs/2209.05034) | [C-MTEB/CLSClusteringS2S](https://modelscope.cn/datasets/C-MTEB/C-MTEB/CLSClusteringS2S) | 从CLS数据集中聚类标题。基于主要类别的13个集合的聚类。 | 聚类 | s2s | 10,000 |
| [CLSClusteringP2P](https://arxiv.org/abs/2209.05034) | [C-MTEB/CLSClusteringP2P](https://modelscope.cn/datasets/C-MTEB/CLSClusteringP2P) | 从CLS数据集中聚类标题+摘要。基于主要类别的13个集合的聚类。 | 聚类 | p2p | 10,000 |
| [ThuNewsClusteringS2S](http://thuctc.thunlp.org/) | [C-MTEB/ThuNewsClusteringS2S](https://modelscope.cn/datasets/C-MTEB/ThuNewsClusteringS2S) | 从THUCNews数据集中聚类标题 | 聚类 | s2s | 10,000 |
| [ThuNewsClusteringP2P](http://thuctc.thunlp.org/) | [C-MTEB/ThuNewsClusteringP2P](https://modelscope.cn/datasets/C-MTEB/ThuNewsClusteringP2P) | 从THUCNews数据集中聚类标题+摘要 | 聚类 | p2p | 10,000 |
| [ATEC](https://github.com/IceFlameWorm/NLP_Datasets/tree/master/ATEC) | [C-MTEB/ATEC](https://modelscope.cn/datasets/C-MTEB/ATEC) | ATEC NLP句子对相似性竞赛 | STS | s2s | 20,000 |
| [BQ](https://huggingface.co/datasets/shibing624/nli_zh) | [C-MTEB/BQ](https://modelscope.cn/datasets/C-MTEB/BQ) | 银行问题语义相似性 | STS | s2s | 10,000 |
| [LCQMC](https://huggingface.co/datasets/shibing624/nli_zh) | [C-MTEB/LCQMC](https://modelscope.cn/datasets/C-MTEB/LCQMC) | 大规模中文问题匹配语料库 | STS | s2s | 12,500 |
| [PAWSX](https://arxiv.org/pdf/1908.11828.pdf) | [C-MTEB/PAWSX](https://modelscope.cn/datasets/C-MTEB/PAWSX) | 翻译的PAWS评估对 | STS | s2s | 2,000 |
| [STSB](https://github.com/pluto-junzeng/CNSD) | [C-MTEB/STSB](https://modelscope.cn/datasets/C-MTEB/STSB) | 将STS-B翻译成中文 | STS | s2s | 1,360 |
| [AFQMC](https://github.com/CLUEbenchmark/CLUE) | [C-MTEB/AFQMC](https://modelscope.cn/datasets/C-MTEB/AFQMC) | 蚂蚁金服问答匹配语料库 | STS | s2s | 3,861 |
| [QBQTC](https://github.com/CLUEbenchmark/QBQTC) | [C-MTEB/QBQTC](https://modelscope.cn/datasets/C-MTEB/QBQTC) | QQ浏览器查询标题语料库 | STS | s2s | 5,000 |
| [TNews](https://github.com/CLUEbenchmark/CLUE) | [C-MTEB/TNews-classification](https://modelscope.cn/datasets/C-MTEB/TNews-classification) | 新闻短文本分类 | 分类 | s2s | 10,000 |
| [IFlyTek](https://github.com/CLUEbenchmark/CLUE) | [C-MTEB/IFlyTek-classification](https://modelscope.cn/datasets/C-MTEB/IFlyTek-classification) | 应用描述的长文本分类 | 分类 | s2s | 2,600 |
| [Waimai](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/waimai_10k/intro.ipynb) | [C-MTEB/waimai-classification](https://modelscope.cn/datasets/C-MTEB/waimai-classification) | 外卖平台用户评论的情感分析 | 分类 | s2s | 1,000 |
| [OnlineShopping](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/online_shopping_10_cats/intro.ipynb) | [C-MTEB/OnlineShopping-classification](https://modelscope.cn/datasets/C-MTEB/OnlineShopping-classification) | 在线购物网站用户评论的情感分析 | 分类 | s2s | 1,000 |
| [MultilingualSentiment](https://github.com/tyqiangz/multilingual-sentiment-datasets) | [C-MTEB/MultilingualSentiment-classification](https://modelscope.cn/datasets/C-MTEB/MultilingualSentiment-classification) | 一组按三类分组的多语言情感数据集--正面、中立、负面 | 分类 | s2s | 3,000 |
| [JDReview](https://huggingface.co/datasets/kuroneko5943/jd21) | [C-MTEB/JDReview-classification](https://modelscope.cn/datasets/C-MTEB/JDReview-classification) | iPhone的评论 | 分类 | s2s | 533 |

对于检索任务，从整个语料库中抽样100,000个候选项（包括真实值），以降低推理成本。

## 环境准备
安装依赖包
```bash
pip install mteb
```

## 配置评测参数
框架支持两种评测方式：单阶段评测 和 两阶段评测：
- 单阶段评测：直接使用模型预测，并计算指标。
- 两阶段评测：使用模型预测，再使用检索模型进行reranking，并计算指标。

### 单阶段评测
配置文件示例如下：
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
#### 参数说明

- `eval_backend`：默认值为 `RAGEval`，表示使用 RAGEval 评测后端。
- `eval_config`：字典，包含以下字段：
    - `tool`：评测工具，使用 `MTEB`。
    - `model`： 模型配置列表，**单阶段评测时只能放置一个模型**，包含以下字段：
        - `model_name_or_path`: `str` 模型名称或路径
        - `is_cross_encoder`: `bool` 模型是否为交叉编码器，默认为 False。  
        - `pooling_mode`: `Optional[str]` 池化模式，可选值为：“cls”、“lasttoken”、“max”、“mean”、“mean_sqrt_len_tokens”或“weightedmean”。  
        - `max_seq_length`: `int` 最大序列长度，默认为 512。  
        - `prompt`: `str` 用于基于大语言模型的提示。  
        - `model_kwargs`: `dict` 模型的关键字参数，默认值为 `{"torch_dtype": "auto"}`。  
        - `config_kwargs`: `Dict[str, Any]` 配置的关键字参数，默认为空字典。  
        - `encode_kwargs`: `dict` 编码的关键字参数，默认值为：  
            ```python  
            {  
                "show_progress_bar": True,  
                "batch_size": 32
            }  
            ```  
        - `hub`: `str` 模型来源，可以是 "modelscope" 或 "huggingface"。  
    - `eval`：字典，包含以下字段：
        - `tasks`: `List[str]` 任务名称  
        - `verbosity`: `int` 详细程度，范围为 0-3  
        - `output_folder`: `str` 输出文件夹，默认为 "outputs"  
        - `overwrite_results`: `bool` 是否覆盖结果，默认为 True  
        - `limits`: `Optional[int]` 限制样本数量，默认为 None  
        - `hub`: `str` 模型来源，可以是 "modelscope" 或 "huggingface"  
        - `top_k`: `int` 选取前 K 个结果，检索任务使用  

### 两阶段评测
配置文件示例如下，先进行检索，再进行reranking：
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
                "prompt": "为这个问题生成一个检索用的表示",
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
#### 参数说明

基本参数与单阶段相同，不同的地方在于model字段，这里需要传入两个模型，第一个模型用于检索，第二个模型用于reranking，reranking的模型需要使用cross encoder。

## 模型评测

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