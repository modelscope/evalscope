# 支持的数据集

## 1. 原生支持的数据集

```{tip}
目前框架原生支持如下数据集，若您需要的数据集不在列表中，可以提交[issue](https://github.com/modelscope/evalscope/issues)，我们会尽快支持；也可以参考[基准评测添加指南](../advanced_guides/add_benchmark.md)，自行添加数据集并提交[PR](https://github.com/modelscope/evalscope/pulls)，欢迎贡献。

您也可以使用本框架支持的其他工具进行评测，如[OpenCompass](../user_guides/backend/opencompass_backend.md)进行语言模型评测；或使用[VLMEvalKit](../user_guides/backend/vlmevalkit_backend.md)进行多模态模型评测。
```


| 名称              | 数据集ID                                                                                           | 任务类别         | 备注                                                                                                                  |
|-------------------|----------------------------------------------------------------------------------------------------|------------------|-----------------------------------------------------------------------------------------------------------------------|
| `aime24`          | [HuggingFaceH4/aime_2024](https://modelscope.cn/datasets/HuggingFaceH4/aime_2024/summary)       | 数学竞赛         |                                                                                                                       |
| `aime25`          | [TIGER-Lab/AIME25](https://modelscope.cn/datasets/TIGER-Lab/AIME25/summary)       | 数学竞赛         |   Part1 |
| `alpaca_eval`<sup>3</sup>    | [AI-ModelScope/alpaca_eval](https://www.modelscope.cn/datasets/AI-ModelScope/alpaca_eval/dataPeview)          | 指令遵循         |    <details><summary>注意事项</summary>暂不支持`length-controlled winrate`；官方Judge模型为`gpt-4-1106-preview`，baseline模型为`gpt-4-turbo`</summary>               |
| `arc`             | [modelscope/ai2_arc](https://modelscope.cn/datasets/modelscope/ai2_arc/summary)                    | 考试         |                                                                                                                       |
| `arena_hard`<sup>3</sup>     | [AI-ModelScope/arena-hard-auto-v0.1](https://modelscope.cn/datasets/AI-ModelScope/arena-hard-auto-v0.1/summary)              | 综合推理         |  <details><summary>注意事项</summary>暂不支持`style-controled winrate`；官方Judge模型为`gpt-4-1106-preview`，baseline模型为`gpt-4-0314` </summary>                                                                               |
| `bbh`             | [modelscope/bbh](https://modelscope.cn/datasets/modelscope/bbh/summary)                            | 综合推理         |                                                                                                                       |
| `ceval`           | [modelscope/ceval-exam](https://modelscope.cn/datasets/modelscope/ceval-exam/summary)              | 中文-综合考试             |                                                                                                                       |
| `chinese_simpleqa`<sup>3</sup>             | [AI-ModelScope/Chinese-SimpleQA](https://modelscope.cn/datasets/AI-ModelScope/Chinese-SimpleQA/summary)                     | 中文知识问答        |                使用 `primary_category`字段作为子数据集           |
| `cmmlu`           | [modelscope/cmmlu](https://modelscope.cn/datasets/modelscope/cmmlu/summary)                        | 中文-综合考试   |                                                                                                                       |
| `competition_math`| [modelscope/competition_math](https://modelscope.cn/datasets/modelscope/competition_math/summary)   | 数学竞赛         |             使用`level`字段作为子数据集                                                                                                          |
| `gpqa`| [modelscope/gpqa](https://modelscope.cn/datasets/modelscope/gpqa/summary)   | 专家级考试         |                                                                                                                       |
| `gsm8k`           | [modelscope/gsm8k](https://modelscope.cn/datasets/modelscope/gsm8k/summary)                        | 数学问题         |                                                                                                                       |
| `hellaswag`       | [modelscope/hellaswag](https://modelscope.cn/datasets/modelscope/hellaswag/summary)                | 常识推理         |                                                                                                                       |
| `humaneval`<sup>2</sup>        | [modelscope/humaneval](https://modelscope.cn/datasets/modelscope/humaneval/summary)                | 代码生成         |  |
| `ifeval`<sup>4</sup>       | [modelscope/ifeval](https://modelscope.cn/datasets/opencompass/ifeval/summary)                | 指令遵循         |  |
| `iquiz`       | [modelscope/iquiz](https://modelscope.cn/datasets/AI-ModelScope/IQuiz/summary)                | 智商和情商         |  |
| `live_code_bench`<sup>2,4</sup>   | [AI-ModelScope/code_generation_lite](https://modelscope.cn/datasets/AI-ModelScope/code_generation_lite/summary) |  代码生成         |    <details><summary>说明</summary>   子数据集支持 `release_v1`,`release_v5`, `v1`, `v4_v5` 等版本标签；`datase-args`中支持设置`'extra_params': {'start_date': '2024-12-01','end_date': '2025-01-01'} `来筛选特定时间范围题目 </details>                              |
| `math_500`       | [AI-ModelScope/MATH-500](https://modelscope.cn/datasets/AI-ModelScope/MATH-500/summary)                | 数学竞赛         | 使用`level`字段作为子数据集                                                                                                          |
| `maritime_bench` | [HiDolphin/MaritimeBench](https://modelscope.cn/datasets/HiDolphin/MaritimeBench/summary)                | 航运知识         |                                                                                                                       |
| `mmlu`            | [modelscope/mmlu](https://modelscope.cn/datasets/modelscope/mmlu/summary)                          | 综合考试   |                                                                                                                       |
| `mmlu_pro`        | [modelscope/mmlu-pro](https://modelscope.cn/datasets/modelscope/mmlu-pro/summary)                    | 综合考试   |                   使用`category`字段作为子数据集                                                                                                    |
| `mmlu_redux`      | [AI-ModelScope/mmlu-redux-2.0](https://modelscope.cn/datasets/AI-ModelScope/mmlu-redux-2.0/summary)                | 综合考试   |                                                                                                                       |
| `musr`            | [AI-ModelScope/MuSR](https://www.modelscope.cn/datasets/AI-ModelScope/MuSR/summary)                          | 多步软推理         |                                                                                                                       |
| `process_bench`   | [Qwen/ProcessBench](https://www.modelscope.cn/datasets/Qwen/ProcessBench/summary)        | 数学过程推理         |                                                                                                                       |
| `race`            | [modelscope/race](https://modelscope.cn/datasets/modelscope/race/summary)                          | 阅读理解         |                                                                                                                       |
| `simple_qa`<sup>3</sup>          | [AI-ModelScope/SimpleQA](https://modelscope.cn/datasets/AI-ModelScope/SimpleQA/summary)               | 知识问答      |
| `super_gpqa`      | [m-a-p/SuperGPQA](https://www.modelscope.cn/datasets/m-a-p/SuperGPQA/dataPeview)                | 专家级考试         |       使用`field`字段作为子数据集                                                                                                              |
| `trivia_qa`       | [modelscope/trivia_qa](https://modelscope.cn/datasets/modelscope/trivia_qa/summary)                | 知识问答             |                                                                                                                       |
| `truthful_qa`<sup>1</sup>       | [modelscope/truthful_qa](https://modelscope.cn/datasets/modelscope/truthful_qa/summary)            | 安全性            |                                                                                                                       |

```{note}
**1.** 评测需要计算logits等，暂不支持API服务评测(`eval-type != server`)。

**2.** 因为涉及到代码运行的操作，建议在沙盒环境(docker)中运行，防止对本地环境造成影响。

**3.** 该数据集需要指定Judge Model进行评测，参考[Judge参数](./parameters.md#judge参数)。

**4.** 建议reasoning模型设置对应数据集的后处理，例如`{"filters": {"remove_until": "</think>"}}`，以获得更好的评测结果。
```


## 2. OpenCompass评测后端支持的数据集

参考[详细说明](https://github.com/open-compass/opencompass#-dataset-support)

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>语言</b>
      </td>
      <td>
        <b>知识</b>
      </td>
      <td>
        <b>推理</b>
      </td>
      <td>
        <b>考试</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
<details open>
<summary><b>字词释义</b></summary>

- WiC
- SummEdits

</details>

<details open>
<summary><b>成语习语</b></summary>

- CHID

</details>

<details open>
<summary><b>语义相似度</b></summary>

- AFQMC
- BUSTM

</details>

<details open>
<summary><b>指代消解</b></summary>

- CLUEWSC
- WSC
- WinoGrande

</details>

<details open>
<summary><b>翻译</b></summary>

- Flores
- IWSLT2017

</details>

<details open>
<summary><b>多语种问答</b></summary>

- TyDi-QA
- XCOPA

</details>

<details open>
<summary><b>多语种总结</b></summary>

- XLSum

</details>
      </td>
      <td>
<details open>
<summary><b>知识问答</b></summary>

- BoolQ
- CommonSenseQA
- NaturalQuestions
- TriviaQA

</details>
      </td>
      <td>
<details open>
<summary><b>文本蕴含</b></summary>

- CMNLI
- OCNLI
- OCNLI_FC
- AX-b
- AX-g
- CB
- RTE
- ANLI

</details>

<details open>
<summary><b>常识推理</b></summary>

- StoryCloze
- COPA
- ReCoRD
- HellaSwag
- PIQA
- SIQA

</details>

<details open>
<summary><b>数学推理</b></summary>

- MATH
- GSM8K

</details>

<details open>
<summary><b>定理应用</b></summary>

- TheoremQA
- StrategyQA
- SciBench

</details>

<details open>
<summary><b>综合推理</b></summary>

- BBH

</details>
      </td>
      <td>
<details open>
<summary><b>初中/高中/大学/职业考试</b></summary>

- C-Eval
- AGIEval
- MMLU
- GAOKAO-Bench
- CMMLU
- ARC
- Xiezhi

</details>

<details open>
<summary><b>医学考试</b></summary>

- CMB

</details>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>理解</b>
      </td>
      <td>
        <b>长文本</b>
      </td>
      <td>
        <b>安全</b>
      </td>
      <td>
        <b>代码</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
<details open>
<summary><b>阅读理解</b></summary>

- C3
- CMRC
- DRCD
- MultiRC
- RACE
- DROP
- OpenBookQA
- SQuAD2.0

</details>

<details open>
<summary><b>内容总结</b></summary>

- CSL
- LCSTS
- XSum
- SummScreen

</details>

<details open>
<summary><b>内容分析</b></summary>

- EPRSTMT
- LAMBADA
- TNEWS

</details>
      </td>
      <td>
<details open>
<summary><b>长文本理解</b></summary>

- LEval
- LongBench
- GovReports
- NarrativeQA
- Qasper

</details>
      </td>
      <td>
<details open>
<summary><b>安全</b></summary>

- CivilComments
- CrowsPairs
- CValues
- JigsawMultilingual
- TruthfulQA

</details>
<details open>
<summary><b>健壮性</b></summary>

- AdvGLUE

</details>
      </td>
      <td>
<details open>
<summary><b>代码</b></summary>

- HumanEval
- HumanEvalX
- MBPP
- APPs
- DS1000

</details>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>



## 3. VLMEvalKit评测后端支持的数据集
参考[详细说明](https://github.com/open-compass/VLMEvalKit/blob/main/docs/zh-CN/README_zh-CN.md#%E6%94%AF%E6%8C%81%E7%9A%84%E5%9B%BE%E6%96%87%E5%A4%9A%E6%A8%A1%E6%80%81%E8%AF%84%E6%B5%8B%E9%9B%86)

### 图文多模态评测集

使用的缩写：
- `MCQ`: 单项选择题; 
- `Y/N`: 正误判断题; 
- `MTT`: 多轮对话评测; 
- `MTI`: 多图输入评测

| 数据集                                                      | 名称                           | 任务                      |
|-------------------------------------------------------------|--------------------------------------------------------|--------------------------|
| [**MMBench Series**](https://github.com/open-compass/mmbench/): <br>MMBench, MMBench-CN, CCBench | MMBench\_DEV\_[EN/CN] <br>MMBench\_TEST\_[EN/CN]<br>MMBench\_DEV\_[EN/CN]\_V11<br>MMBench\_TEST\_[EN/CN]\_V11<br>CCBench | MCQ                      |
| [**MMStar**](https://github.com/MMStar-Benchmark/MMStar)   | MMStar                                                 | MCQ                      |
| [**MME**](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) | MME                                                    | Y/N                      |
| [**SEEDBench Series**](https://github.com/AILab-CVC/SEED-Bench) | SEEDBench_IMG <br>SEEDBench2 <br>SEEDBench2_Plus     | MCQ                      |
| [**MM-Vet**](https://github.com/yuweihao/MM-Vet)           | MMVet                                                  | VQA                      |
| [**MMMU**](https://mmmu-benchmark.github.io)               | MMMU\_[DEV_VAL/TEST]                                  | MCQ                      |
| [**MathVista**](https://mathvista.github.io)               | MathVista_MINI                                        | VQA                      |
| [**ScienceQA_IMG**](https://scienceqa.github.io)           | ScienceQA\_[VAL/TEST]                                 | MCQ                      |
| [**COCO Caption**](https://cocodataset.org)                | COCO_VAL                                              | Caption                  |
| [**HallusionBench**](https://github.com/tianyi-lab/HallusionBench) | HallusionBench                                        | Y/N                      |
| [**OCRVQA**](https://ocr-vqa.github.io)*                   | OCRVQA\_[TESTCORE/TEST]                              | VQA                      |
| [**TextVQA**](https://textvqa.org)*                        | TextVQA_VAL                                          | VQA                      |
| [**ChartQA**](https://github.com/vis-nlp/ChartQA)*         | ChartQA_TEST                                          | VQA                      |
| [**AI2D**](https://allenai.org/data/diagrams)              | AI2D\_[TEST/TEST_NO_MASK]                             | MCQ                      |
| [**LLaVABench**](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild) | LLaVABench                                            | VQA                      |
| [**DocVQA**](https://www.docvqa.org)+                       | DocVQA\_[VAL/TEST]                                   | VQA                      |
| [**InfoVQA**](https://www.docvqa.org/datasets/infographicvqa)+ | InfoVQA\_[VAL/TEST]                                  | VQA                      |
| [**OCRBench**](https://github.com/Yuliang-Liu/MultimodalOCR) | OCRBench                                              | VQA                      |
| [**RealWorldQA**](https://x.ai/blog/grok-1.5v)            | RealWorldQA                                           | MCQ                      |
| [**POPE**](https://github.com/AoiDragon/POPE)              | POPE                                                  | Y/N                      |
| [**Core-MM**](https://github.com/core-mm/core-mm)-          | CORE_MM (MTI)                                        | VQA                      |
| [**MMT-Bench**](https://mmt-bench.github.io)               | MMT-Bench\_[VAL/ALL]<br>MMT-Bench\_[VAL/ALL]\_MI     | MCQ (MTI)               |
| [**MLLMGuard**](https://github.com/Carol-gutianle/MLLMGuard) - | MLLMGuard_DS                                        | VQA                      |
| [**AesBench**](https://github.com/yipoh/AesBench)+        | AesBench\_[VAL/TEST]                                 | MCQ                      |
| [**VCR-wiki**](https://huggingface.co/vcr-org/) +         | VCR\_[EN/ZH]\_[EASY/HARD]_[ALL/500/100]              | VQA                      |
| [**MMLongBench-Doc**](https://mayubo2333.github.io/MMLongBench-Doc/)+ | MMLongBench_DOC                                   | VQA (MTI)                |
| [**BLINK**](https://zeyofu.github.io/blink/)               | BLINK                                                 | MCQ (MTI)               |
| [**MathVision**](https://mathvision-cuhk.github.io)+       | MathVision<br>MathVision_MINI                         | VQA                      |
| [**MT-VQA**](https://github.com/bytedance/MTVQA)+          | MTVQA_TEST                                           | VQA                      |
| [**MMDU**](https://liuziyu77.github.io/MMDU/)+             | MMDU                                                  | VQA (MTT, MTI)          |
| [**Q-Bench1**](https://github.com/Q-Future/Q-Bench)+       | Q-Bench1\_[VAL/TEST]                                 | MCQ                      |
| [**A-Bench**](https://github.com/Q-Future/A-Bench)+        | A-Bench\_[VAL/TEST]                                  | MCQ                      |
| [**DUDE**](https://arxiv.org/abs/2305.08455)+              | DUDE                                                  | VQA (MTI)               |
| [**SlideVQA**](https://arxiv.org/abs/2301.04883)+          | SLIDEVQA<br>SLIDEVQA_MINI                            | VQA (MTI)               |
| [**TaskMeAnything ImageQA Random**](https://huggingface.co/datasets/weikaih/TaskMeAnything-v1-imageqa-random)+ | TaskMeAnything_v1_imageqa_random                       | MCQ                      |
| [**MMMB and Multilingual MMBench**](https://sun-hailong.github.io/projects/Parrot/)+ | MMMB\_[ar/cn/en/pt/ru/tr]<br>MMBench_dev\_[ar/cn/en/pt/ru/tr]<br>MMMB<br>MTL_MMBench_DEV<br>PS: MMMB & MTL_MMBench_DEV <br>are **all-in-one** names for 6 langs | MCQ                      |
| [**A-OKVQA**](https://arxiv.org/abs/2206.01718)+           | A-OKVQA                                              | MCQ                      |
| [**MuirBench**](https://muirbench.github.io)               | MUIRBench                                             | MCQ                      |
| [**GMAI-MMBench**](https://huggingface.co/papers/2408.03361)+ | GMAI-MMBench\_VAL                                   | MCQ                      |
| [**TableVQABench**](https://arxiv.org/abs/2404.19205)+     | TableVQABench                                        | VQA                      |

```{note}
**\*** 只提供了部分模型上的[测试结果](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)，剩余模型无法在 zero-shot 设定下测试出合理的精度

**\+** 尚未提供这个评测集的测试结果

**\-** VLMEvalKit 仅支持这个评测集的推理，无法输出最终精度
```

### 视频多模态评测集

| 数据集                                              | 数据集名称                 | 任务  |
| ---------------------------------------------------- | -------------------------- | ----  |
| [**MMBench-Video**](https://mmbench-video.github.io) | MMBench-Video             | VQA  |
| [**MVBench**](https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/MVBENCH.md) | MVBench_MP4    | MCQ  |
| [**MLVU**](https://github.com/JUNJIE99/MLVU)        | MLVU                      | MCQ & VQA  |
| [**TempCompass**](https://arxiv.org/abs/2403.00476) | TempCompass                | MCQ & Y/N & Caption  |
| [**LongVideoBench**](https://longvideobench.github.io/) | LongVideoBench            | MCQ  |
| [**Video-MME**](https://video-mme.github.io/)      | Video-MME                 | MCQ  |

## 4. RAGEval评测后端支持的数据集

### CMTEB 评测数据集
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
| [PAWSX](https://arxiv.org/pdf/1908.11828.pdf) | [C-MTEB/PAWSX](https://modelscope.cn/datasets/C-MTEB/PAWSX) | 翻译的PAWS评测对 | STS | s2s | 2,000 |
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

### MTEB 评测数据集
```{seealso}
参考：[MTEB相关任务](https://github.com/embeddings-benchmark/mteb/blob/main/docs/tasks.md)
```

### CLIP-Benchmark

| 数据集名称                                                                                                     | 任务类型               | 备注 |
|--------------------------------------------------------------------------------------------------------------|--------------------|------|
| [muge](https://modelscope.cn/datasets/clip-benchmark/muge/)                                                   | zeroshot_retrieval |  中文多模态图文数据集    |
| [flickr30k](https://modelscope.cn/datasets/clip-benchmark/flickr30k/)                                         | zeroshot_retrieval |      |
| [flickr8k](https://modelscope.cn/datasets/clip-benchmark/flickr8k/)                                           | zeroshot_retrieval |      |
| [mscoco_captions](https://modelscope.cn/datasets/clip-benchmark/mscoco_captions/)                             | zeroshot_retrieval |      |
| [mscoco_captions2017](https://modelscope.cn/datasets/clip-benchmark/mscoco_captions2017/)                     | zeroshot_retrieval |      |
| [imagenet1k](https://modelscope.cn/datasets/clip-benchmark/imagenet1k/)                                       | zeroshot_classification |      |
| [imagenetv2](https://modelscope.cn/datasets/clip-benchmark/imagenetv2/)                                       | zeroshot_classification |      |
| [imagenet_sketch](https://modelscope.cn/datasets/clip-benchmark/imagenet_sketch/)                             | zeroshot_classification |      |
| [imagenet-a](https://modelscope.cn/datasets/clip-benchmark/imagenet-a/)                                       | zeroshot_classification |      |
| [imagenet-r](https://modelscope.cn/datasets/clip-benchmark/imagenet-r/)                                       | zeroshot_classification |      |
| [imagenet-o](https://modelscope.cn/datasets/clip-benchmark/imagenet-o/)                                       | zeroshot_classification |      |
| [objectnet](https://modelscope.cn/datasets/clip-benchmark/objectnet/)                                         | zeroshot_classification |      |
| [fer2013](https://modelscope.cn/datasets/clip-benchmark/fer2013/)                                             | zeroshot_classification |      |
| [voc2007](https://modelscope.cn/datasets/clip-benchmark/voc2007/)                                             | zeroshot_classification |      |
| [voc2007_multilabel](https://modelscope.cn/datasets/clip-benchmark/voc2007_multilabel/)                       | zeroshot_classification |      |
| [sun397](https://modelscope.cn/datasets/clip-benchmark/sun397/)                                               | zeroshot_classification |      |
| [cars](https://modelscope.cn/datasets/clip-benchmark/cars/)                                                   | zeroshot_classification |      |
| [fgvc_aircraft](https://modelscope.cn/datasets/clip-benchmark/fgvc_aircraft/)                                 | zeroshot_classification |      |
| [mnist](https://modelscope.cn/datasets/clip-benchmark/mnist/)                                                 | zeroshot_classification |      |
| [stl10](https://modelscope.cn/datasets/clip-benchmark/stl10/)                                                 | zeroshot_classification |      |
| [gtsrb](https://modelscope.cn/datasets/clip-benchmark/gtsrb/)                                                 | zeroshot_classification |      |
| [country211](https://modelscope.cn/datasets/clip-benchmark/country211/)                                       | zeroshot_classification |      |
| [renderedsst2](https://modelscope.cn/datasets/clip-benchmark/renderedsst2/)                                   | zeroshot_classification |      |
| [vtab_caltech101](https://modelscope.cn/datasets/clip-benchmark/vtab_caltech101/)                             | zeroshot_classification |      |
| [vtab_cifar10](https://modelscope.cn/datasets/clip-benchmark/vtab_cifar10/)                                   | zeroshot_classification |      |
| [vtab_cifar100](https://modelscope.cn/datasets/clip-benchmark/vtab_cifar100/)                                 | zeroshot_classification |      |
| [vtab_clevr_count_all](https://modelscope.cn/datasets/clip-benchmark/vtab_clevr_count_all/)                   | zeroshot_classification |      |
| [vtab_clevr_closest_object_distance](https://modelscope.cn/datasets/clip-benchmark/vtab_clevr_closest_object_distance/) | zeroshot_classification |      |
| [vtab_diabetic_retinopathy](https://modelscope.cn/datasets/clip-benchmark/vtab_diabetic_retinopathy/)         | zeroshot_classification |      |
| [vtab_dmlab](https://modelscope.cn/datasets/clip-benchmark/vtab_dmlab/)                                       | zeroshot_classification |      |
| [vtab_dsprites_label_orientation](https://modelscope.cn/datasets/clip-benchmark/vtab_dsprites_label_orientation/) | zeroshot_classification |      |
| [vtab_dsprites_label_x_position](https://modelscope.cn/datasets/clip-benchmark/vtab_dsprites_label_x_position/) | zeroshot_classification |      |
| [vtab_dsprites_label_y_position](https://modelscope.cn/datasets/clip-benchmark/vtab_dsprites_label_y_position/) | zeroshot_classification |      |
| [vtab_dtd](https://modelscope.cn/datasets/clip-benchmark/vtab_dtd/)                                           | zeroshot_classification |      |
| [vtab_eurosat](https://modelscope.cn/datasets/clip-benchmark/vtab_eurosat/)                                   | zeroshot_classification |      |
| [vtab_kitti_closest_vehicle_distance](https://modelscope.cn/datasets/clip-benchmark/vtab_kitti_closest_vehicle_distance/) | zeroshot_classification |      |
| [vtab_flowers](https://modelscope.cn/datasets/clip-benchmark/vtab_flowers/)                                   | zeroshot_classification |      |
| [vtab_pets](https://modelscope.cn/datasets/clip-benchmark/vtab_pets/)                                         | zeroshot_classification |      |
| [vtab_pcam](https://modelscope.cn/datasets/clip-benchmark/vtab_pcam/)                                         | zeroshot_classification |      |
| [vtab_resisc45](https://modelscope.cn/datasets/clip-benchmark/vtab_resisc45/)                                 | zeroshot_classification |      |
| [vtab_smallnorb_label_azimuth](https://modelscope.cn/datasets/clip-benchmark/vtab_smallnorb_label_azimuth/)   | zeroshot_classification |      |
| [vtab_smallnorb_label_elevation](https://modelscope.cn/datasets/clip-benchmark/vtab_smallnorb_label_elevation/) | zeroshot_classification |      |
| [vtab_svhn](https://modelscope.cn/datasets/clip-benchmark/vtab_svhn/)                                         | zeroshot_classification |      |