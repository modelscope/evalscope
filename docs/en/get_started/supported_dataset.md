# Supported Datasets

## 1. Native Supported Datasets
```{tip}
The framework currently supports the following datasets natively. If the dataset you need is not on the list, you may submit an [issue](https://github.com/modelscope/evalscope/issues), and we will support it as soon as possible. Alternatively, you can refer to the [Benchmark Addition Guide](../advanced_guides/add_benchmark.md) to add datasets by yourself and submit a [PR](https://github.com/modelscope/evalscope/pulls). Contributions are welcome.

You can also use other tools supported by this framework for evaluation, such as [OpenCompass](../user_guides/backend/opencompass_backend.md) for language model evaluation, or [VLMEvalKit](../user_guides/backend/vlmevalkit_backend.md) for multimodal model evaluation.
```

| Name              | Dataset ID                                                                                           | Task Category    | Remarks                                                                                                                  |
|-------------------|------------------------------------------------------------------------------------------------------|------------------|--------------------------------------------------------------------------------------------------------------------------|
| `aime24`          | [HuggingFaceH4/aime_2024](https://modelscope.cn/datasets/HuggingFaceH4/aime_2024/summary)       | Math Competition         |                                                                                                                       |
| `aime25`          | [TIGER-Lab/AIME25](https://modelscope.cn/datasets/TIGER-Lab/AIME25/summary)       | Math Competition         |   Part1 |
| `arc`             | [modelscope/ai2_arc](https://modelscope.cn/datasets/modelscope/ai2_arc/summary)                      | Exam             |                                                                                                                          |
| `bbh`             | [modelscope/bbh](https://modelscope.cn/datasets/modelscope/bbh/summary)                              | General Reasoning|                                                                                                                          |
| `ceval`           | [modelscope/ceval-exam](https://modelscope.cn/datasets/modelscope/ceval-exam/summary)                | Chinese Comprehensive Exam|                                                                                                                          |
| `cmmlu`           | [modelscope/cmmlu](https://modelscope.cn/datasets/modelscope/cmmlu/summary)                          | Chinese Comprehensive Exam|                                                                                                                          |
| `competition_math`| [modelscope/competition_math](https://modelscope.cn/datasets/modelscope/competition_math/summary)    | Math Competition |                                                                                                                          |
| `gpqa`| [modelscope/gpqa](https://modelscope.cn/datasets/modelscope/gpqa/summary)   | Expert level exam |                                                                                                                          |
| `gsm8k`           | [modelscope/gsm8k](https://modelscope.cn/datasets/modelscope/gsm8k/summary)                          | Math Problems    |                                                                                                                          |
| `hellaswag`      | [modelscope/hellaswag](https://modelscope.cn/datasets/modelscope/hellaswag/summary)                  | Commonsense Reasoning|                                                                                                                          |
| `humaneval`+      | [modelscope/humaneval](https://modelscope.cn/datasets/modelscope/humaneval/summary)                  | Code Generation  |  |
| `ifeval`       | [modelscope/ifeval](https://modelscope.cn/datasets/opencompass/ifeval/summary)                | Instruction Following         |  |
| `iquiz`       | [modelscope/iquiz](https://modelscope.cn/datasets/AI-ModelScope/IQuiz/summary)                | IQ and EQ         |  |
| `math_500`       | [AI-ModelScope/MATH-500](https://modelscope.cn/datasets/AI-ModelScope/MATH-500/summary)                | Math Competition         |  |
| `mmlu`            | [modelscope/mmlu](https://modelscope.cn/datasets/modelscope/mmlu/summary)                            | Comprehensive Exam|                                                                                                                          |
| `mmlu_pro`        | [modelscope/mmlu-pro](https://modelscope.cn/datasets/modelscope/mmlu-pro/summary)                    | Comprehensive Exam|                                                                                                                          |
| `musr`            | [AI-ModelScope/MuSR](https://www.modelscope.cn/datasets/AI-ModelScope/MuSR/summary)                          | Multistep Soft Reasoning         |                                                                                                                       |
| `process_bench`   | [Qwen/ProcessBench](https://www.modelscope.cn/datasets/Qwen/ProcessBench/summary)        |  Math Reasoning Process         |                                                                                                                       |
| `race`            | [modelscope/race](https://modelscope.cn/datasets/modelscope/race/summary)                            | Reading Comprehension|                                                                                                                          |
| `trivia_qa`      | [modelscope/trivia_qa](https://modelscope.cn/datasets/modelscope/trivia_qa/summary)                  | Knowledge Q&A    |                                                                                                                          |
| `truthful_qa`*     | [modelscope/truthful_qa](https://modelscope.cn/datasets/modelscope/truthful_qa/summary)              | Safety           |                                                                                                                          |

```{note}
**\*** Evaluation requires calculating logits, etc., and it does not support API service evaluation (`eval-type != server`) at present.

**+** Due to operations involving code execution, it is recommended to run in a sandbox environment (docker) to prevent impacts on the local environment.
```

## 2. OpenCompass Backend
Refer to the [detailed explanation](https://github.com/open-compass/opencompass#-dataset-support)

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Language</b>
      </td>
      <td>
        <b>Knowledge</b>
      </td>
      <td>
        <b>Reasoning</b>
      </td>
      <td>
        <b>Examination</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
<details open>
<summary><b>Word Definition</b></summary>

- WiC
- SummEdits

</details>

<details open>
<summary><b>Idiom Learning</b></summary>

- CHID

</details>

<details open>
<summary><b>Semantic Similarity</b></summary>

- AFQMC
- BUSTM

</details>

<details open>
<summary><b>Coreference Resolution</b></summary>

- CLUEWSC
- WSC
- WinoGrande

</details>

<details open>
<summary><b>Translation</b></summary>

- Flores
- IWSLT2017

</details>

<details open>
<summary><b>Multi-language Question Answering</b></summary>

- TyDi-QA
- XCOPA

</details>

<details open>
<summary><b>Multi-language Summary</b></summary>

- XLSum

</details>
      </td>
      <td>
<details open>
<summary><b>Knowledge Question Answering</b></summary>

- BoolQ
- CommonSenseQA
- NaturalQuestions
- TriviaQA

</details>
      </td>
      <td>
<details open>
<summary><b>Textual Entailment</b></summary>

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
<summary><b>Commonsense Reasoning</b></summary>

- StoryCloze
- COPA
- ReCoRD
- HellaSwag
- PIQA
- SIQA

</details>

<details open>
<summary><b>Mathematical Reasoning</b></summary>

- MATH
- GSM8K

</details>

<details open>
<summary><b>Theorem Application</b></summary>

- TheoremQA
- StrategyQA
- SciBench

</details>

<details open>
<summary><b>Comprehensive Reasoning</b></summary>

- BBH

</details>
      </td>
      <td>
<details open>
<summary><b>Junior High, High School, University, Professional Examinations</b></summary>

- C-Eval
- AGIEval
- MMLU
- GAOKAO-Bench
- CMMLU
- ARC
- Xiezhi

</details>

<details open>
<summary><b>Medical Examinations</b></summary>

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
        <b>Understanding</b>
      </td>
      <td>
        <b>Long Context</b>
      </td>
      <td>
        <b>Safety</b>
      </td>
      <td>
        <b>Code</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
<details open>
<summary><b>Reading Comprehension</b></summary>

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
<summary><b>Content Summary</b></summary>

- CSL
- LCSTS
- XSum
- SummScreen

</details>

<details open>
<summary><b>Content Analysis</b></summary>

- EPRSTMT
- LAMBADA
- TNEWS

</details>
      </td>
      <td>
<details open>
<summary><b>Long Context Understanding</b></summary>

- LEval
- LongBench
- GovReports
- NarrativeQA
- Qasper

</details>
      </td>
      <td>
<details open>
<summary><b>Safety</b></summary>

- CivilComments
- CrowsPairs
- CValues
- JigsawMultilingual
- TruthfulQA

</details>
<details open>
<summary><b>Robustness</b></summary>

- AdvGLUE

</details>
      </td>
      <td>
<details open>
<summary><b>Code</b></summary>

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


## 3. VLMEvalKit Backend
Refer to the [detailed explanation](https://github.com/open-compass/VLMEvalKit/tree/main#-datasets-models-and-evaluation-results)

### Image Understanding Dataset

Abbreviations used:
- `MCQ`: Multiple Choice Questions; 
- `Y/N`: Yes/No Questions; 
- `MTT`: Multiturn Dialogue Evaluation; 
- `MTI`: Multi-image Input Evaluation


| Dataset                                                      | Dataset Names                           | Task                      |
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
**\*** Partial model testing results are provided [here](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard), while remaining models cannot achieve reasonable accuracy under zero-shot conditions.

**\+** Testing results for this evaluation set have not yet been provided.

**\-** VLMEvalKit only supports inference for this evaluation set and cannot output final accuracy.
```

### Video Understanding Dataset

| Dataset                                              | Dataset Name                | Task                  |
| ---------------------------------------------------- | --------------------------- | --------------------- |
| [**MMBench-Video**](https://mmbench-video.github.io) | MMBench-Video              | VQA                   |
| [**MVBench**](https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/MVBENCH.md) | MVBench_MP4                | MCQ                   |
| [**MLVU**](https://github.com/JUNJIE99/MLVU)        | MLVU                       | MCQ & VQA             |
| [**TempCompass**](https://arxiv.org/abs/2403.00476) | TempCompass                 | MCQ & Y/N & Caption   |
| [**LongVideoBench**](https://longvideobench.github.io/) | LongVideoBench             | MCQ                   |
| [**Video-MME**](https://video-mme.github.io/)      | Video-MME                  | MCQ                   |

## 4. RAGEval Backend

### CMTEB Evaluation Dataset
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

### MTEB Evaluation Dataset
```{seealso}
See also: [MTEB Related Tasks](https://github.com/embeddings-benchmark/mteb/blob/main/docs/tasks.md)
```

### CLIP-Benchmark

| Dataset Name                                                                                                   | Task Type              | Notes                      |
|---------------------------------------------------------------------------------------------------------------|------------------------|----------------------------|
| [muge](https://modelscope.cn/datasets/clip-benchmark/muge/)                                                   | zeroshot_retrieval     | Chinese Multimodal Dataset |
| [flickr30k](https://modelscope.cn/datasets/clip-benchmark/flickr30k/)                                         | zeroshot_retrieval     |                            |
| [flickr8k](https://modelscope.cn/datasets/clip-benchmark/flickr8k/)                                           | zeroshot_retrieval     |                            |
| [mscoco_captions](https://modelscope.cn/datasets/clip-benchmark/mscoco_captions/)                             | zeroshot_retrieval     |                            |
| [mscoco_captions2017](https://modelscope.cn/datasets/clip-benchmark/mscoco_captions2017/)                     | zeroshot_retrieval     |                            |
| [imagenet1k](https://modelscope.cn/datasets/clip-benchmark/imagenet1k/)                                       | zeroshot_classification|                            |
| [imagenetv2](https://modelscope.cn/datasets/clip-benchmark/imagenetv2/)                                       | zeroshot_classification|                            |
| [imagenet_sketch](https://modelscope.cn/datasets/clip-benchmark/imagenet_sketch/)                             | zeroshot_classification|                            |
| [imagenet-a](https://modelscope.cn/datasets/clip-benchmark/imagenet-a/)                                       | zeroshot_classification|                            |
| [imagenet-r](https://modelscope.cn/datasets/clip-benchmark/imagenet-r/)                                       | zeroshot_classification|                            |
| [imagenet-o](https://modelscope.cn/datasets/clip-benchmark/imagenet-o/)                                       | zeroshot_classification|                            |
| [objectnet](https://modelscope.cn/datasets/clip-benchmark/objectnet/)                                         | zeroshot_classification|                            |
| [fer2013](https://modelscope.cn/datasets/clip-benchmark/fer2013/)                                             | zeroshot_classification|                            |
| [voc2007](https://modelscope.cn/datasets/clip-benchmark/voc2007/)                                             | zeroshot_classification|                            |
| [voc2007_multilabel](https://modelscope.cn/datasets/clip-benchmark/voc2007_multilabel/)                       | zeroshot_classification|                            |
| [sun397](https://modelscope.cn/datasets/clip-benchmark/sun397/)                                               | zeroshot_classification|                            |
| [cars](https://modelscope.cn/datasets/clip-benchmark/cars/)                                                   | zeroshot_classification|                            |
| [fgvc_aircraft](https://modelscope.cn/datasets/clip-benchmark/fgvc_aircraft/)                                 | zeroshot_classification|                            |
| [mnist](https://modelscope.cn/datasets/clip-benchmark/mnist/)                                                 | zeroshot_classification|                            |
| [stl10](https://modelscope.cn/datasets/clip-benchmark/stl10/)                                                 | zeroshot_classification|                            |
| [gtsrb](https://modelscope.cn/datasets/clip-benchmark/gtsrb/)                                                 | zeroshot_classification|                            |
| [country211](https://modelscope.cn/datasets/clip-benchmark/country211/)                                       | zeroshot_classification|                            |
| [renderedsst2](https://modelscope.cn/datasets/clip-benchmark/renderedsst2/)                                   | zeroshot_classification|                            |
| [vtab_caltech101](https://modelscope.cn/datasets/clip-benchmark/vtab_caltech101/)                             | zeroshot_classification|                            |
| [vtab_cifar10](https://modelscope.cn/datasets/clip-benchmark/vtab_cifar10/)                                   | zeroshot_classification|                            |
| [vtab_cifar100](https://modelscope.cn/datasets/clip-benchmark/vtab_cifar100/)                                 | zeroshot_classification|                            |
| [vtab_clevr_count_all](https://modelscope.cn/datasets/clip-benchmark/vtab_clevr_count_all/)                   | zeroshot_classification|                            |
| [vtab_clevr_closest_object_distance](https://modelscope.cn/datasets/clip-benchmark/vtab_clevr_closest_object_distance/) | zeroshot_classification|                            |
| [vtab_diabetic_retinopathy](https://modelscope.cn/datasets/clip-benchmark/vtab_diabetic_retinopathy/)         | zeroshot_classification|                            |
| [vtab_dmlab](https://modelscope.cn/datasets/clip-benchmark/vtab_dmlab/)                                       | zeroshot_classification|                            |
| [vtab_dsprites_label_orientation](https://modelscope.cn/datasets/clip-benchmark/vtab_dsprites_label_orientation/) | zeroshot_classification|                            |
| [vtab_dsprites_label_x_position](https://modelscope.cn/datasets/clip-benchmark/vtab_dsprites_label_x_position/) | zeroshot_classification|                            |
| [vtab_dsprites_label_y_position](https://modelscope.cn/datasets/clip-benchmark/vtab_dsprites_label_y_position/) | zeroshot_classification|                            |
| [vtab_dtd](https://modelscope.cn/datasets/clip-benchmark/vtab_dtd/)                                           | zeroshot_classification|                            |
| [vtab_eurosat](https://modelscope.cn/datasets/clip-benchmark/vtab_eurosat/)                                   | zeroshot_classification|                            |
| [vtab_kitti_closest_vehicle_distance](https://modelscope.cn/datasets/clip-benchmark/vtab_kitti_closest_vehicle_distance/) | zeroshot_classification|                            |
| [vtab_flowers](https://modelscope.cn/datasets/clip-benchmark/vtab_flowers/)                                   | zeroshot_classification|                            |
| [vtab_pets](https://modelscope.cn/datasets/clip-benchmark/vtab_pets/)                                         | zeroshot_classification|                            |
| [vtab_pcam](https://modelscope.cn/datasets/clip-benchmark/vtab_pcam/)                                         | zeroshot_classification|                            |
| [vtab_resisc45](https://modelscope.cn/datasets/clip-benchmark/vtab_resisc45/)                                 | zeroshot_classification|                            |
| [vtab_smallnorb_label_azimuth](https://modelscope.cn/datasets/clip-benchmark/vtab_smallnorb_label_azimuth/)   | zeroshot_classification|                            |
| [vtab_smallnorb_label_elevation](https://modelscope.cn/datasets/clip-benchmark/vtab_smallnorb_label_elevation/) | zeroshot_classification|                            |
| [vtab_svhn](https://modelscope.cn/datasets/clip-benchmark/vtab_svhn/)                                         | zeroshot_classification|                            |