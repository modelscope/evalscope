# Supported Datasets

## 1. Native Supported Datasets
```{tip}
The framework currently supports the following datasets. If the dataset you need is not in the list, please submit an issue, or use the [OpenCompass backend](../user_guides/opencompass_backend.md) for evaluation, or use the [VLMEvalKit backend](../user_guides/vlmevalkit_backend.md) for multi-modal model evaluation.
```

| Dataset Name       | Link                                                                                   | Status | Note |
|--------------------|----------------------------------------------------------------------------------------|--------|------|
| `mmlu`             | [mmlu](https://modelscope.cn/datasets/modelscope/mmlu/summary)                         | Active |      |
| `ceval`            | [ceval](https://modelscope.cn/datasets/modelscope/ceval-exam/summary)                  | Active |      |
| `gsm8k`            | [gsm8k](https://modelscope.cn/datasets/modelscope/gsm8k/summary)                       | Active |      |
| `arc`              | [arc](https://modelscope.cn/datasets/modelscope/ai2_arc/summary)                       | Active |      |
| `hellaswag`        | [hellaswag](https://modelscope.cn/datasets/modelscope/hellaswag/summary)               | Active |      |
| `truthful_qa`      | [truthful_qa](https://modelscope.cn/datasets/modelscope/truthful_qa/summary)           | Active |      |
| `competition_math` | [competition_math](https://modelscope.cn/datasets/modelscope/competition_math/summary) | Active |      |
| `humaneval`        | [humaneval](https://modelscope.cn/datasets/modelscope/humaneval/summary)               | Active |      |
| `bbh`              | [bbh](https://modelscope.cn/datasets/modelscope/bbh/summary)                           | Active |      |
| `race`             | [race](https://modelscope.cn/datasets/modelscope/race/summary)                         | Active |      |
| `trivia_qa`        | [trivia_qa](https://modelscope.cn/datasets/modelscope/trivia_qa/summary)               | To be integrated |      |

## 2. Datasets Supported by OpenCompass
Refer to the [detailed explanation](https://github.com/open-compass/opencompass#-dataset-support)

| **Language**       | **Knowledge**                      | **Inference**                         | **Exams**                                    | **Comprehension**       | **Long Text**                   | **Safety**                        | **Code**                                   |
|-------------------|-----------------------------------|---------------------------------------|---------------------------------------------|-------------------------|----------------------------------|----------------------------------|--------------------------------------------|
| **Word Definitions**   | **Knowledge Question Answering**  | **Textual Entailment**                  | **Middle School/High School/University/Professional Exams**                | **Reading Comprehension**   | **Long Text Understanding**      | **Safety**                       | **Code**                                   |
| - WiC              | - BoolQ                           | - CMNLI                               | - C-Eval                                   | - C3                     | - LEval                          | - CivilComments                  | - HumanEval                                |
| - SummEdits       | - CommonSenseQA                   | - OCNLI                               | - AGIEval                                  | - CMRC                   | - LongBench                      | - CrowsPairs                     | - HumanEvalX                               |
|                   | - NaturalQuestions                | - OCNLI_FC                            | - MMLU                                     | - DRCD                   | - GovReports                     | - CValues                        | - MBPP                                     |
|                   | - TriviaQA                       | - AX-b                                | - GAOKAO-Bench                             | - MultiRC                | - NarrativeQA                    | - JigsawMultilingual             | - APPs                                     |
|                   |                                   | - AX-g                                | - CMMLU                                    | - RACE                   | - Qasper                         | - TruthfulQA                     | - DS1000                                   |
|                   |                                   | - CB                                   | - ARC                                       | - DROP                   |                                  | **Robustness**                   |                                            |
|                   |                                   | - RTE                                  | - Xiezhi                                   | - OpenBookQA             |                                  | - AdvGLUE                       |                                            |
|                   |                                   | - ANLI                                 |                                             | - SQuAD2.0              |                                  |                                  |                                            |
| **Idioms and Expressions** |                              | **Common Sense Reasoning**             | **Medical Exams**                            | **Content Summary**      |                                  |                                  |                                            |
| - CHID            |                                   | - StoryCloze                          | - CMB                                       | - CSL                    |                                  |                                  |                                            |
|                   |                                   | - COPA                                 |                                             | - LCSTS                  |                                  |                                  |                                            |
|                   |                                   | - ReCoRD                               |                                             | - XSum                   |                                  |                                  |                                            |
|                   |                                   | - HellaSwag                            |                                             | - SummScreen             |                                  |                                  |                                            |
|                   |                                   | - PIQA                                 |                                             | **Content Analysis**     |                                  |                                  |                                            |
|                   |                                   | - SIQA                                 |                                             | - EPRSTMT                |                                  |                                  |                                            |
|                   |                                   |                                       |                                             | - LAMBADA                |                                  |                                  |                                            |
|                   |                                   |                                       |                                             | - TNEWS                  |                                  |                                  |                                            |
| **Semantic Similarity** |                               | **Mathematical Reasoning**             |                                             |                         |                                  |                                  |                                            |
| - AFQMC           |                                   | - MATH                                 |                                             |                         |                                  |                                  |                                            |
| - BUSTM           |                                   | - GSM8K                                |                                             |                         |                                  |                                  |                                            |
| **Coreference Resolution** |                           | **Theorem Application**                |                                             |                         |                                  |                                  |                                            |
| - CLUEWSC         |                                   | - TheoremQA                            |                                             |                         |                                  |                                  |                                            |
| - WSC             |                                   | - StrategyQA                           |                                             |                         |                                  |                                  |                                            |
| - WinoGrande      |                                   | - SciBench                             |                                             |                         |                                  |                                  |                                            |
| **Translation**    |                                   | **Integrated Reasoning**               |                                             |                         |                                  |                                  |                                            |
| - Flores          |                                   | - BBH                                  |                                             |                         |                                  |                                  |                                            |
| - IWSLT2017       |                                   |                                       |                                             |                         |                                  |                                  |                                            |
| **Multilingual QA** |                                  |                                       |                                             |                         |                                  |                                  |                                            |
| - TyDi-QA         |                                   |                                       |                                             |                         |                                  |                                  |                                            |
| - XCOPA           |                                   |                                       |                                             |                         |                                  |                                  |                                            |
| **Multilingual Summarization** |                       |                                       |                                             |                         |                                  |                                  |                                            |
| - XLSum           |                                   |                                       |                                             |                         |                                  |                                  |                                            |



## 3. Datasets Supported by VLMEvalKit
Refer to the [detailed explanation](https://github.com/open-compass/VLMEvalKit/tree/main#-datasets-models-and-evaluation-results)

### Image Understanding Dataset

Abbreviations used:
- `MCQ`: Multiple Choice Questions; 
- `Y/N`: Yes/No Questions; 
- `MTT`: Multiturn Dialogue Evaluation; 
- `MTI`: Multi-image Input Evaluation

| Dataset                                                      | Dataset Names (for run.py)                             | Task | Dataset | Dataset Names (for run.py) | Task |
| ------------------------------------------------------------ | ------------------------------------------------------ | --------- | --------- | --------- | --------- |
| [**MMBench Series**](https://github.com/open-compass/mmbench/): <br>MMBench, MMBench-CN, CCBench | MMBench\_DEV\_[EN/CN] <br>MMBench\_TEST\_[EN/CN]<br>MMBench\_DEV\_[EN/CN]\_V11<br>MMBench\_TEST\_[EN/CN]\_V11<br>CCBench | MCQ | [**MMStar**](https://github.com/MMStar-Benchmark/MMStar) | MMStar | MCQ |
| [**MME**](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) | MME | Y/N                                       | [**SEEDBench Series**](https://github.com/AILab-CVC/SEED-Bench) | SEEDBench_IMG <br>SEEDBench2 <br>SEEDBench2_Plus | MCQ                                                |
| [**MM-Vet**](https://github.com/yuweihao/MM-Vet)             | MMVet  | VQA                                              | [**MMMU**](https://mmmu-benchmark.github.io)  | MMMU_[DEV_VAL/TEST]                      | MCQ                                |
| [**MathVista**](https://mathvista.github.io)                 | MathVista_MINI | VQA                                         | [**ScienceQA_IMG**](https://scienceqa.github.io) | ScienceQA_[VAL/TEST]                     | MCQ                        |
| [**COCO Caption**](https://cocodataset.org)                  | COCO_VAL | Caption                                              | [**HallusionBench**](https://github.com/tianyi-lab/HallusionBench) | HallusionBench                                | Y/N                             |
| [**OCRVQA**](https://ocr-vqa.github.io)*                     | OCRVQA_[TESTCORE/TEST] | VQA                                 | [**TextVQA**](https://textvqa.org)* | TextVQA_VAL                      | VQA                              |
| [**ChartQA**](https://github.com/vis-nlp/ChartQA)*           | ChartQA_TEST | VQA                                          | [**AI2D**](https://allenai.org/data/diagrams) | AI2D_[TEST/TEST_NO_MASK]                                 | MCQ                         |
| [**LLaVABench**](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild) | LLaVABench | VQA                                            | [**DocVQA**](https://www.docvqa.org)+       | DocVQA_[VAL/TEST]                           | VQA                                         |
| [**InfoVQA**](https://www.docvqa.org/datasets/infographicvqa)+ | InfoVQA_[VAL/TEST] | VQA | [**OCRBench**](https://github.com/Yuliang-Liu/MultimodalOCR) | OCRBench | VQA |
| [**RealWorldQA**](https://x.ai/blog/grok-1.5v)            | RealWorldQA | MCQ                                          | [**POPE**](https://github.com/AoiDragon/POPE) | POPE                                           | Y/N                                            |
| [**Core-MM**](https://github.com/core-mm/core-mm)-          | CORE_MM (MTI) | VQA                                               | [**MMT-Bench**](https://mmt-bench.github.io)                 | MMT-Bench\_[VAL/ALL]<br>MMT-Bench\_[VAL/ALL]_MI | MCQ (MTI) |
| [**MLLMGuard**](https://github.com/Carol-gutianle/MLLMGuard) - | MLLMGuard_DS | VQA | [**AesBench**](https://github.com/yipoh/AesBench)+ | AesBench_[VAL/TEST] | MCQ |
| [**VCR-wiki**](https://huggingface.co/vcr-org/) + | VCR\_[EN/ZH]\_[EASY/HARD]_[ALL/500/100] | VQA | [**MMLongBench-Doc**](https://mayubo2333.github.io/MMLongBench-Doc/)+ | MMLongBench_DOC | VQA (MTI) |
| [**BLINK**](https://zeyofu.github.io/blink/) | BLINK | MCQ (MTI) | [**MathVision**](https://mathvision-cuhk.github.io)+ | MathVision<br>MathVision_MINI | VQA |
| [**MT-VQA**](https://github.com/bytedance/MTVQA)+ | MTVQA_TEST | VQA | [**MMDU**](https://liuziyu77.github.io/MMDU/)+ | MMDU | VQA (MTT, MTI) |
| [**Q-Bench1**](https://github.com/Q-Future/Q-Bench)+ | Q-Bench1_[VAL/TEST] | MCQ | [**A-Bench**](https://github.com/Q-Future/A-Bench)+ | A-Bench_[VAL/TEST] | MCQ |
| [**DUDE**](https://arxiv.org/abs/2305.08455)+ | DUDE | VQA (MTI) | [**SlideVQA**](https://arxiv.org/abs/2301.04883)+ | SLIDEVQA<br>SLIDEVQA_MINI | VQA (MTI) |
| [**TaskMeAnything ImageQA Random**](https://huggingface.co/datasets/weikaih/TaskMeAnything-v1-imageqa-random)+ | TaskMeAnything_v1_imageqa_random | MCQ  | [**MMMB and Multilingual MMBench**](https://sun-hailong.github.io/projects/Parrot/)+ | MMMB\_[ar/cn/en/pt/ru/tr]<br>MMBench_dev\_[ar/cn/en/pt/ru/tr]<br>MMMB<br>MTL_MMBench_DEV<br>PS: MMMB & MTL_MMBench_DEV <br>are **all-in-one** names for 6 langs | MCQ  |
| [**A-OKVQA**](https://arxiv.org/abs/2206.01718)+ | A-OKVQA | MCQ | [**MuirBench**](https://muirbench.github.io) | MUIRBench | MCQ |
| [**GMAI-MMBench**](https://huggingface.co/papers/2408.03361)+ | GMAI-MMBench_VAL | MCQ | [**TableVQABench**](https://arxiv.org/abs/2404.19205)+ | TableVQABench | VQA |

```{note}
**\*** Partial model testing results are provided [here](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard), while remaining models cannot achieve reasonable accuracy under zero-shot conditions.

**\+** Testing results for this evaluation set have not yet been provided.

**\-** VLMEvalKit only supports inference for this evaluation set and cannot output final accuracy.
```

### Video Understanding Dataset

| Dataset                                              | Dataset Name          | Task | Dataset                                       | Dataset Name | Task |
| ---------------------------------------------------- | --------------------- | ---- | --------------------------------------------- | ------------ | ---- |
| [**MMBench-Video**](https://mmbench-video.github.io) | MMBench-Video        | VQA  | [**Video-MME**](https://video-mme.github.io/) | Video-MME   | MCQ  |
| [**MVBench**](https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/MVBENCH.md)| MVBench/MVBench_MP4 | MCQ  |       |                |      |