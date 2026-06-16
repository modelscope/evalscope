(mteb)=

# MTEB 文本嵌入评测

本框架支持 [MTEB 2.x](https://github.com/embeddings-benchmark/mteb)（Massive Text Embedding Benchmark），用于衡量文本嵌入模型在检索、重排序、分类、聚类、语义文本相似度等多种任务上的性能。支持 35+ 中文数据集和 112+ 种语言。

```{warning}
**破坏性变更 (v1.9.0)**

如果你从旧版本升级，请注意以下变更：

1. **配置 schema**：`model` -> `models`，`eval.tasks` -> `eval.task_names`，`eval.topk` -> `eval.top_k`
2. **依赖升级**：要求 `mteb>=2.7.0`（原为 1.x）和 `ragas>=0.4.0`（原为 0.2.x）
3. **数据集迁移**：所有中文数据集从 `C-MTEB/` 迁移到 `mteb/` 组织下
4. **Custom Dataset 格式变更**：统一为 Retrieval 格式（JSONL），废弃 TSV qrels 和 Reranking/STS 自定义任务类型；qrels 从 `qrels/test.tsv` 改为平级 `qrels.jsonl`；列名统一: `_id`（corpus/queries），`query-id` + `corpus-id` + `score`（qrels）
5. **移除的导入路径**：`from evalscope.backend.rag_eval import EmbeddingModel` -> 使用 `load_model()`；`from evalscope.backend.rag_eval.cmteb import ...` -> 使用 `evalscope.backend.rag_eval.mteb`
```

## 环境准备

```bash
pip install evalscope[rag] -U
```

## 场景一：评测本地 Embedding 模型（Quick Start）

> 适用于：本地部署的 sentence-transformer 类模型

### 配置与运行

```python
from evalscope.run import run_task

task_cfg = {
    "eval_backend": "RAGEval",
    "eval_config": {
        "tool": "mteb",
        "models": [
            {
                "model_name_or_path": "AI-ModelScope/m3e-base",
                "max_seq_length": 512,
                "model_kwargs": {"torch_dtype": "auto"},
                "encode_kwargs": {"batch_size": 128},
            }
        ],
        "eval": {
            "task_names": ["TNews", "CLSClusteringS2S", "T2Reranking", "T2Retrieval", "ATEC"],
            "verbosity": 2,
            "overwrite_results": True,
            "top_k": 10,
            "limits": 500,
        },
    },
}

run_task(task_cfg=task_cfg)
```

### 结果解读

评测完成后，结果保存在 `outputs/<model_name>/<revision>/<task_name>.json`，核心字段示例：

```json
{
  "task_name": "TNews",
  "scores": {
    "validation": [
      {
        "main_score": 0.4744,
        "accuracy": 0.4744,
        "f1": 0.4456,
        "f1_weighted": 0.4754
      }
    ]
  }
}
```

其中 `main_score` 为该任务的主指标（分类任务为 accuracy，检索任务为 ndcg@10）。

### 该场景关键参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_name_or_path` | `str` | `""` | 模型名称或路径，支持从 ModelScope 自动下载 |
| `max_seq_length` | `int` | `512` | 最大序列长度 |
| `model_kwargs` | `dict` | `{}` | 模型加载参数，如 `{"torch_dtype": "auto"}` |
| `encode_kwargs` | `dict` | `{"batch_size": 32}` | 编码参数 |
| `task_names` | `List[str]` | `None` | 要评测的任务列表 |
| `limits` | `Optional[int]` | `None` | 限制样本数量（调试时建议设置） |

## 场景二：评测 API 模型（DashScope/OpenAI 等）

> 适用于：通过 API 调用的 Embedding/Reranker 服务

### 前置条件

确保模型服务已部署并可访问，获取 API 地址和密钥。

### 配置与运行

```python
import os
from evalscope.run import run_task
from evalscope import TaskConfig

task_cfg = TaskConfig(
    eval_backend='RAGEval',
    eval_config={
        'tool': 'MTEB',
        'models': [
            {
                'model_name': 'text-embedding-v3',
                'api_base': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'api_key': os.environ.get('DASHSCOPE_API_KEY', 'EMPTY'),
                'dimensions': 1024,
                'encode_kwargs': {
                    'batch_size': 10,
                },
            }
        ],
        'eval': {
            'task_names': ['T2Retrieval'],
            'verbosity': 2,
            'overwrite_results': True,
            'limits': 30,
        },
    },
)

run_task(task_cfg=task_cfg)
```

### 结果解读

```json
{
  "task_name": "T2Retrieval",
  "scores": {
    "dev": [
      {
        "main_score": 0.73143,
        "ndcg_at_10": 0.73143,
        "recall_at_10": 0.73318,
        "precision_at_1": 0.78989
      }
    ]
  }
}
```

### 该场景关键参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_name` | `str` | - | API 模型名称 |
| `api_base` | `str` | - | API 服务地址 |
| `api_key` | `str` | - | API 密钥 |
| `dimensions` | `Optional[int]` | `None` | 模型输出维度 |
| `is_cross_encoder` | `bool` | `False` | 是否为 reranker，API reranker 需设为 `True` |

## 场景三：两阶段评测（Retrieval + Reranking）

> 适用于：先用 Embedding 模型检索、再用 Reranker 模型精排的管线

### 何时选择两阶段

- **单阶段**：直接使用 embedding 模型做检索/分类/聚类/STS，适合大多数场景
- **两阶段**：需要评测 reranker 模型在真实检索管线中的效果时使用；先用 embedding 模型检索 top-K 候选，再用 reranker 对候选重排序

### 配置与运行

在 `models` 中传入两个模型：第一个为检索用的 embedding 模型，第二个为 reranking 模型（需设置 `is_cross_encoder: True`）。

```python
from evalscope.run import run_task

task_cfg = {
    "eval_backend": "RAGEval",
    "eval_config": {
        "tool": "MTEB",
        "models": [
            {
                "model_name_or_path": "AI-ModelScope/m3e-base",
                "is_cross_encoder": False,
                "max_seq_length": 512,
                "model_kwargs": {"torch_dtype": "auto"},
                "encode_kwargs": {"batch_size": 64},
            },
            {
                "model_name_or_path": "OpenBMB/MiniCPM-Reranker",
                "is_cross_encoder": True,
                "max_seq_length": 512,
                "prompt": "为这个问题生成一个检索用的表示",
                "model_kwargs": {"torch_dtype": "auto"},
                "encode_kwargs": {"batch_size": 32},
            },
        ],
        "eval": {
            "task_names": ["T2Retrieval"],
            "verbosity": 2,
            "overwrite_results": True,
            "top_k": 5,
            "limits": 100,
        },
    },
}

run_task(task_cfg=task_cfg)
```

### 结果解读

两阶段评测分别输出阶段一（检索）和阶段二（重排）的结果：

**阶段一**（`outputs/stage1/<embedding_model>/...`）：
```json
{
  "task_name": "T2Retrieval",
  "scores": {"dev": [{"main_score": 0.73143, "ndcg_at_10": 0.73143}]}
}
```

**阶段二**（`outputs/stage2/<reranker_model>/...`）：
```json
{
  "task_name": "T2Retrieval",
  "scores": {"dev": [{"main_score": 0.661, "ndcg_at_10": 0.661}]}
}
```

### 两阶段特有参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `is_cross_encoder` | `bool` | `False` | 第二个模型（reranker）需设置为 `True` |
| `top_k` | `int` | `10` | 阶段一检索的候选数量，传给阶段二重排 |
| `prompt` | `Optional[str]` | `None` | 检索任务的 query 前缀提示 |

## 场景四：自定义数据集评测

> 适用于：用私有数据集评测模型

### 数据准备

自定义数据集需要符合 Retrieval 格式（JSONL）。详细格式说明请参考：

```{seealso}
[自定义检索评测数据集](../../../advanced_guides/custom_dataset/embedding.md)
```

### 配置与运行

在 `eval` 中使用 `custom_tasks` 指定自定义数据集路径：

```python
from evalscope.run import run_task

task_cfg = {
    "eval_backend": "RAGEval",
    "eval_config": {
        "tool": "MTEB",
        "models": [
            {
                "model_name_or_path": "AI-ModelScope/m3e-base",
                "max_seq_length": 512,
                "model_kwargs": {"torch_dtype": "auto"},
                "encode_kwargs": {"batch_size": 128},
            }
        ],
        "eval": {
            "custom_tasks": ["path/to/your/custom_dataset"],
            "verbosity": 2,
            "overwrite_results": True,
        },
    },
}

run_task(task_cfg=task_cfg)
```

## 支持的数据集

<details><summary>点击展开完整数据集列表</summary>

| 名称 | Hub链接 | 描述 | 类型 | 类别 | 测试样本数量 |
|-----|-----|---------------------------|-----|-----|-----|
| [T2Retrieval](https://arxiv.org/abs/2304.03679) | [mteb/T2Retrieval](https://modelscope.cn/datasets/mteb/T2Retrieval) | T2Ranking：一个大规模的中文段落排序基准 | 检索 | s2p | 24,832 |
| [MMarcoRetrieval](https://github.com/unicamp-dl/mMARCO) | [mteb/MMarcoRetrieval](https://modelscope.cn/datasets/mteb/MMarcoRetrieval) | mMARCO是MS MARCO段落排序数据集的多语言版本 | 检索 | s2p | 7,437 |
| [DuRetrieval](https://aclanthology.org/2022.emnlp-main.357.pdf) | [mteb/DuRetrieval](https://modelscope.cn/datasets/mteb/DuRetrieval) | 一个大规模的中文网页搜索引擎段落检索基准 | 检索 | s2p | 4,000 |
| [CovidRetrieval](https://aclanthology.org/2022.emnlp-main.357.pdf) | [mteb/CovidRetrieval](https://modelscope.cn/datasets/mteb/CovidRetrieval) | COVID-19新闻文章 | 检索 | s2p | 949 |
| [CmedqaRetrieval](https://aclanthology.org/2022.emnlp-main.357.pdf) | [mteb/CmedqaRetrieval](https://modelscope.cn/datasets/mteb/CmedqaRetrieval) | 在线医疗咨询文本 | 检索 | s2p | 3,999 |
| [EcomRetrieval](https://arxiv.org/abs/2203.03367) | [mteb/EcomRetrieval](https://modelscope.cn/datasets/mteb/EcomRetrieval) | 从阿里巴巴电商领域搜索引擎系统收集的段落检索数据集 | 检索 | s2p | 1,000 |
| [MedicalRetrieval](https://arxiv.org/abs/2203.03367) | [mteb/MedicalRetrieval](https://modelscope.cn/datasets/mteb/MedicalRetrieval) | 从阿里巴巴医疗领域搜索引擎系统收集的段落检索数据集 | 检索 | s2p | 1,000 |
| [VideoRetrieval](https://arxiv.org/abs/2203.03367) | [mteb/VideoRetrieval](https://modelscope.cn/datasets/mteb/VideoRetrieval) | 从阿里巴巴视频领域搜索引擎系统收集的段落检索数据集 | 检索 | s2p | 1,000 |
| [T2Reranking](https://arxiv.org/abs/2304.03679) | [mteb/T2Reranking](https://modelscope.cn/datasets/mteb/T2Reranking) | T2Ranking：一个大规模的中文段落排序基准 | 重新排序 | s2p | 24,382 |
| [MMarcoReranking](https://github.com/unicamp-dl/mMARCO) | [mteb/MMarco-reranking](https://modelscope.cn/datasets/mteb/Mmarco-reranking) | mMARCO是MS MARCO段落排序数据集的多语言版本 | 重新排序 | s2p | 7,437 |
| [CMedQAv1](https://github.com/zhangsheng93/cMedQA) | [mteb/CMedQAv1-reranking](https://modelscope.cn/datasets/mteb/CMedQAv1-reranking) | 中文社区医疗问答 | 重新排序 | s2p | 2,000 |
| [CMedQAv2](https://github.com/zhangsheng93/cMedQA2) | [mteb/CMedQAv2-reranking](https://modelscope.cn/datasets/mteb/CMedQAv2-reranking) | 中文社区医疗问答 | 重新排序 | s2p | 4,000 |
| [Ocnli](https://arxiv.org/abs/2010.05444) | [mteb/OCNLI](https://modelscope.cn/datasets/mteb/OCNLI) | 原始中文自然语言推理数据集 | 配对分类 | s2s | 3,000 |
| [Cmnli](https://modelscope.cn/datasets/clue/viewer/cmnli) | [mteb/CMNLI](https://modelscope.cn/datasets/mteb/CMNLI) | 中文多类别自然语言推理 | 配对分类 | s2s | 139,000 |
| [CLSClusteringS2S](https://arxiv.org/abs/2209.05034) | [mteb/CLSClusteringS2S](https://modelscope.cn/datasets/mteb/CLSClusteringS2S) | 从CLS数据集中聚类标题。基于主要类别的13个集合的聚类。 | 聚类 | s2s | 10,000 |
| [CLSClusteringP2P](https://arxiv.org/abs/2209.05034) | [mteb/CLSClusteringP2P](https://modelscope.cn/datasets/mteb/CLSClusteringP2P) | 从CLS数据集中聚类标题+摘要。基于主要类别的13个集合的聚类。 | 聚类 | p2p | 10,000 |
| [ThuNewsClusteringS2S](http://thuctc.thunlp.org/) | [mteb/ThuNewsClusteringS2S](https://modelscope.cn/datasets/mteb/ThuNewsClusteringS2S) | 从THUCNews数据集中聚类标题 | 聚类 | s2s | 10,000 |
| [ThuNewsClusteringP2P](http://thuctc.thunlp.org/) | [mteb/ThuNewsClusteringP2P](https://modelscope.cn/datasets/mteb/ThuNewsClusteringP2P) | 从THUCNews数据集中聚类标题+摘要 | 聚类 | p2p | 10,000 |
| [ATEC](https://github.com/IceFlameWorm/NLP_Datasets/tree/master/ATEC) | [mteb/ATEC](https://modelscope.cn/datasets/mteb/ATEC) | ATEC NLP句子对相似性竞赛 | STS | s2s | 20,000 |
| [BQ](https://huggingface.co/datasets/shibing624/nli_zh) | [mteb/BQ](https://modelscope.cn/datasets/mteb/BQ) | 银行问题语义相似性 | STS | s2s | 10,000 |
| [LCQMC](https://huggingface.co/datasets/shibing624/nli_zh) | [mteb/LCQMC](https://modelscope.cn/datasets/mteb/LCQMC) | 大规模中文问题匹配语料库 | STS | s2s | 12,500 |
| [PAWSX](https://arxiv.org/pdf/1908.11828.pdf) | [mteb/PAWSX](https://modelscope.cn/datasets/mteb/PAWSX) | 翻译的PAWS评测对 | STS | s2s | 2,000 |
| [STSB](https://github.com/pluto-junzeng/CNSD) | [mteb/STSB](https://modelscope.cn/datasets/mteb/STSB) | 将STS-B翻译成中文 | STS | s2s | 1,360 |
| [AFQMC](https://github.com/CLUEbenchmark/CLUE) | [mteb/AFQMC](https://modelscope.cn/datasets/mteb/AFQMC) | 蚂蚁金服问答匹配语料库 | STS | s2s | 3,861 |
| [QBQTC](https://github.com/CLUEbenchmark/QBQTC) | [mteb/QBQTC](https://modelscope.cn/datasets/mteb/QBQTC) | QQ浏览器查询标题语料库 | STS | s2s | 5,000 |
| [TNews](https://github.com/CLUEbenchmark/CLUE) | [mteb/TNews-classification](https://modelscope.cn/datasets/mteb/TNews-classification) | 新闻短文本分类 | 分类 | s2s | 10,000 |
| [IFlyTek](https://github.com/CLUEbenchmark/CLUE) | [mteb/IFlyTek-classification](https://modelscope.cn/datasets/mteb/IFlyTek-classification) | 应用描述的长文本分类 | 分类 | s2s | 2,600 |
| [Waimai](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/waimai_10k/intro.ipynb) | [mteb/waimai-classification](https://modelscope.cn/datasets/mteb/waimai-classification) | 外卖平台用户评论的情感分析 | 分类 | s2s | 1,000 |
| [OnlineShopping](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/online_shopping_10_cats/intro.ipynb) | [mteb/OnlineShopping-classification](https://modelscope.cn/datasets/mteb/OnlineShopping-classification) | 在线购物网站用户评论的情感分析 | 分类 | s2s | 1,000 |
| [MultilingualSentiment](https://github.com/tyqiangz/multilingual-sentiment-datasets) | [mteb/MultilingualSentiment-classification](https://modelscope.cn/datasets/mteb/MultilingualSentiment-classification) | 一组按三类分组的多语言情感数据集--正面、中立、负面 | 分类 | s2s | 3,000 |
| [JDReview](https://huggingface.co/datasets/kuroneko5943/jd21) | [mteb/JDReview-classification](https://modelscope.cn/datasets/mteb/JDReview-classification) | iPhone的评论 | 分类 | s2s | 533 |

对于检索任务，从整个语料库中抽样100,000个候选项（包括真实值），以降低推理成本。

</details>

```{seealso}
- [MTEB支持的数据集](https://embeddings-benchmark.github.io/mteb/overview/)
```

## 完整参数参考

### 本地模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_name_or_path` | `str` | `""` | 模型名称或路径，支持从 ModelScope 自动下载 |
| `is_cross_encoder` | `bool` | `False` | 是否为交叉编码器，reranking 模型需设置为 `True` |
| `pooling_mode` | `Optional[str]` | `None` | 池化模式，可选: cls / lasttoken / max / mean / mean_sqrt_len_tokens / weightedmean。`bge` 系列模型请设置为 cls |
| `max_seq_length` | `int` | `512` | 最大序列长度 |
| `prompt` | `Optional[str]` | `None` | 检索任务的 query 前缀提示 |
| `prompts` | `Optional[Dict[str, str]]` | `None` | 按任务名设置不同 prompt，key 为任务名，value 为 prompt，仅在未设置 `prompt` 时生效 |
| `model_kwargs` | `dict` | `{}` | 模型加载参数，例如 `{"torch_dtype": "auto"}` |
| `encode_kwargs` | `dict` | `{"batch_size": 32}` | 编码参数 |
| `hub` | `str` | `"modelscope"` | 模型来源，可选 `modelscope` 或 `huggingface` |

### 远程 API 模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_name` | `str` | - | API 模型名称 |
| `is_cross_encoder` | `bool` | `False` | 是否为 reranker，API reranker 需设置为 `True` |
| `api_base` | `str` | - | API 服务地址；reranker 会调用 rerank 端点，地址未以 `/rerank` 或 `/reranks` 结尾时默认追加 `/rerank`（DashScope 域名自动追加 `/reranks`） |
| `api_key` | `str` | - | API 密钥 |
| `dimensions` | `Optional[int]` | `None` | 模型输出维度 |
| `max_seq_length` | `int` | `512` | 最大序列长度；超过此长度的文本将被自动截断后再发送到 API |
| `encode_kwargs` | `dict` | `{"batch_size": 10}` | 编码参数 |

### 评测配置参数（eval）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `task_names` | `Optional[List[str]]` | `None` | 任务名称列表，参见[支持的数据集](#支持的数据集) |
| `task_types` | `Optional[List[str]]` | `None` | 按任务类型筛选，如 `["Retrieval", "STS"]` |
| `languages` | `Optional[List[str]]` | `None` | 按语言筛选，如 `["cmn-Hans"]` |
| `custom_tasks` | `Optional[List]` | `None` | 自定义任务配置列表，参见[场景四](#场景四自定义数据集评测) |
| `top_k` | `int` | `10` | 检索任务选取前 K 个结果 |
| `overwrite_results` | `bool` | `True` | 是否覆盖已有结果 |
| `limits` | `Optional[int]` | `None` | 限制样本数量；检索任务同时限制 queries 和 corpus（仅保留被 qrels 引用的文档） |
| `output_folder` | `str` | `"outputs"` | 结果输出目录 |
| `hub` | `str` | `"modelscope"` | 数据集来源，可选 `modelscope` 或 `huggingface` |

### 外层配置

- `eval_backend`：固定为 `RAGEval`，表示使用 RAGEval 评测后端
- `eval_config.tool`：固定为 `MTEB`
- `eval_config.models`：模型配置列表。单阶段评测放一个模型；两阶段评测放两个模型（第一个检索，第二个重排）

## 常见问题

**模型加载失败 / OOM**

- 确保 GPU 显存足够，可通过 `encode_kwargs.batch_size` 减小批量大小
- 使用 `model_kwargs: {"torch_dtype": "float16"}` 降低显存占用
- 检查 `model_name_or_path` 路径是否正确

**数据集下载失败**

- 默认从 ModelScope 下载，确保网络可达 `modelscope.cn`
- 可设置 `hub: "huggingface"` 切换到 HuggingFace 源
- 本地已有数据集时，可通过 `model_name_or_path` 指定本地路径

**结果指标含义**

- `main_score`：该任务的主评测指标
- 检索任务：`ndcg_at_10`（归一化折损累计增益）
- 重排序任务：`map`（平均精度均值）
- 分类任务：`accuracy`
- STS 任务：`cosine_spearman`（余弦相似度的 Spearman 相关系数）
- 聚类任务：`v_measure`
