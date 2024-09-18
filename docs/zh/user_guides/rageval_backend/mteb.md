(mteb)=

# MTEB

本框架支持 [MTEB](https://github.com/embeddings-benchmark/mteb) 和 [CMTEB](https://github.com/embeddings-benchmark/mteb)，具体介绍如下：

- MTEB（Massive Text Embedding Benchmark）是一个大规模的基准测试，旨在衡量文本嵌入模型在多样化嵌入任务上的性能。MTEB 包括56个数据集，涵盖8个任务，并且支持超过112种不同的语言。这个基准测试的目标是帮助开发者找到适用于多种任务的最佳文本嵌入模型。

- C-MTEB（Chinese Massive Text Embedding Benchmark）是一个专门针对中文文本向量的评测基准，它基于MTEB构建，旨在评估中文文本向量模型的性能。C-MTEB收集了35个公共数据集，并分为6类评估任务，包括检索（retrieval）、重排序（re-ranking）、语义文本相似度（STS）、分类（classification）、对分类（pair classification）和聚类（clustering）。

## 环境准备
安装依赖包
```bash
pip install mteb
```

## 配置评估参数



## 模型评估