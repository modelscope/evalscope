# VLMEvalKit

```{note}
更完整的说明和及时更新的数据集列表，请参考[详细说明](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb)
```

## 图文多模态评测集

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

## 视频多模态评测集

| 数据集                                              | 数据集名称                 | 任务  |
| ---------------------------------------------------- | -------------------------- | ----  |
| [**MMBench-Video**](https://mmbench-video.github.io) | MMBench-Video             | VQA  |
| [**MVBench**](https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/MVBENCH.md) | MVBench_MP4    | MCQ  |
| [**MLVU**](https://github.com/JUNJIE99/MLVU)        | MLVU                      | MCQ & VQA  |
| [**TempCompass**](https://arxiv.org/abs/2403.00476) | TempCompass                | MCQ & Y/N & Caption  |
| [**LongVideoBench**](https://longvideobench.github.io/) | LongVideoBench            | MCQ  |
| [**Video-MME**](https://video-mme.github.io/)      | Video-MME                 | MCQ  |