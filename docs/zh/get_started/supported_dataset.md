# 支持的数据集

## 1. 原生支持的数据集

```{tip}
目前框架原生支持如下数据集，若您需要的数据集不在列表中，请提交issue；或者使用[OpenCompass backend](../user_guides/backend/opencompass_backend.md)进行语言模型评估；或使用[VLMEvalKit backend](../user_guides/backend/vlmevalkit_backend.md)进行多模态模型评估。
```

| 名称        | 链接                                                                                   | 状态 | 备注 |
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
| `trivia_qa`        | [trivia_qa](https://modelscope.cn/datasets/modelscope/trivia_qa/summary)               | To be intergrated |      |

## 2. OpenCompass支持的数据集

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



## 3. VLMEvalKit支持的数据集
参考[详细说明](https://github.com/open-compass/VLMEvalKit/blob/main/docs/zh-CN/README_zh-CN.md#%E6%94%AF%E6%8C%81%E7%9A%84%E5%9B%BE%E6%96%87%E5%A4%9A%E6%A8%A1%E6%80%81%E8%AF%84%E6%B5%8B%E9%9B%86)

### 支持的图文多模态评测集

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

### 支持的视频多模态评测集

| 数据集                                              | 数据集名称                 | 任务  |
| ---------------------------------------------------- | -------------------------- | ----  |
| [**MMBench-Video**](https://mmbench-video.github.io) | MMBench-Video             | VQA  |
| [**MVBench**](https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/MVBENCH.md) | MVBench/MVBench_MP4       | MCQ  |
| [**Video-MME**](https://video-mme.github.io/)      | Video-MME                 | MCQ  |
