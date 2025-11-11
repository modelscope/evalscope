# Evaluating the Thinking Efficiency of Models

With the rapid development of large language models, their reasoning capabilities have significantly improved. In particular, long reasoning models such as OpenAI's o1, QwQ-32B, DeepSeek-R1-671B, and Kimi K1.5 have garnered attention for exhibiting human-like deep thinking abilities. These models are capable of continuous reasoning during the decoding stage through Inference-Time Scaling, allowing them to think and explore new ideas to arrive at the correct answer.

However, as research has progressed, researchers have identified two extreme issues during the reasoning process of these models: **Underthinking** and **Overthinking**:

- The phenomenon of **Underthinking** refers to the model frequently shifting its thought process during reasoning, often using phrases like "alternatively," "but wait," or "let me reconsider," and failing to focus on a correct thought process for deeper analysis, ultimately leading to incorrect answers.[^1] This phenomenon resembles "Attention Deficit Hyperactivity Disorder" in humans, adversely affecting the quality of the model's reasoning.

- The phenomenon of **Overthinking** manifests as the model generating unnecessarily long chains of thought, wasting a substantial amount of computational resources. For example, for a simple question like "2+3=?," some long reasoning models may consume over 900 tokens exploring various problem-solving strategies.[^2] While such chain-of-thought strategies are beneficial for complex problems, repeatedly validating existing answers and conducting overly broad explorations for simple problems is clearly a waste of computational resources.[^3]

Both phenomena highlight a key question: how can we improve the thinking efficiency of models while ensuring the quality of their answers? In other words, **we want models to arrive at correct answers with outputs as brief as possible**. In this best practice guide, we will evaluate the thinking efficiency of models such as DeepSeek-R1-Distill-Qwen-7B using the [MATH-500](https://www.modelscope.cn/datasets/AI-ModelScope/MATH-500) dataset, assessing model performance across six dimensions: the number of reasoning tokens, the number of first correct tokens, the number of reflection tokens, token efficiency, the number of sub-thought chains, and accuracy. Let’s get started.

## Installing Dependencies

First, install the [EvalScope](https://github.com/modelscope/evalscope) model evaluation framework:

```bash
pip install 'evalscope' -U
```

## Evaluating the Model

We will begin the formal evaluation process, which consists of two main steps:
1. **Model Reasoning Evaluation**: Use the EvalScope framework to have the model reason through the MATH-500 dataset. This dataset includes 500 math problems, each consisting of a mathematical expression and the corresponding answer, with difficulty levels ranging from 1 (easy) to 5 (complex). This step will yield the model's reasoning results for each problem, as well as the overall accuracy rate.
2. **Model Thinking Efficiency Evaluation**: Using the EvalThink component within the EvalScope framework to conduct an in-depth analysis of the model's outputs, further assessing thinking efficiency in terms of token efficiency, model thinking length, the number of sub-thought chains, and more.

### Model Reasoning

**Preparing to Evaluate the Model**

First, we need to access the model capabilities via an OpenAI API-compatible inference service for evaluation. It is worth noting that EvalScope also supports model inference evaluation using transformers; detailed information can be found in the EvalScope documentation.

In addition to deploying the model to a cloud service that supports the OpenAI API, you can also choose to launch the model locally using frameworks like vLLM or Ollama. These inference frameworks can efficiently support concurrent requests, speeding up the evaluation process. Particularly for R1-type models, which often produce long chains of thought, the output token count can frequently exceed 10,000. Deploying the model using an efficient inference framework can significantly enhance reasoning speed.

As an example, here is how to deploy the DeepSeek-R1-Distill-Qwen-7B model using vLLM:

```bash
VLLM_USE_MODELSCOPE=True CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B  --served-model-name DeepSeek-R1-Distill-Qwen-1.5B --trust_remote_code --port 8801
```

**Using EvalScope to Evaluate the Model**

Run the following command to have the model reason through the MATH-500 dataset and obtain the output results for each problem, as well as the overall accuracy:

```python
from evalscope import TaskConfig, run_task

task_config = TaskConfig(
    api_url='http://0.0.0.0:8801/v1/chat/completions',  # Inference service address
    model='DeepSeek-R1-Distill-Qwen-7B',  # Model name (must match the deployed model name)
    eval_type='openai_api',  # Evaluation type, 'openai_api' indicates evaluating the inference service
    datasets=['math_500'],  # Dataset name
    dataset_args={'math_500': {'few_shot_num': 0, 'subset_list': ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']}},  # Dataset parameters
    eval_batch_size=32,  # Number of concurrent requests
    generation_config={
        'max_tokens': 20000,  # Maximum number of tokens to generate; suggested to set a high value to avoid truncation
        'temperature': 0.6,  # Sampling temperature (recommended value from deepseek)
        'top_p': 0.95,  # Top-p sampling (recommended value from deepseek)
        'n': 1,  # Number of responses generated for each request
    },
)
run_task(task_config)
```

The output will look like this, showing the model's accuracy on problems at each difficulty level:

```text
+-----------------------------+-----------+---------------+----------+-------+---------+---------+
| Model                       | Dataset   | Metric        | Subset   |   Num |   Score | Cat.0   |
+=============================+===========+===============+==========+=======+=========+=========+
| DeepSeek-R1-Distill-Qwen-7B | math_500  | AveragePass@1 | Level 1  |    43 |  0.9535 | default |
+-----------------------------+-----------+---------------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-7B | math_500  | AveragePass@1 | Level 2  |    90 |  0.9667 | default |
+-----------------------------+-----------+---------------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-7B | math_500  | AveragePass@1 | Level 3  |   105 |  0.9587 | default |
+-----------------------------+-----------+---------------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-7B | math_500  | AveragePass@1 | Level 4  |   128 |  0.9115 | default |
+-----------------------------+-----------+---------------+----------+-------+---------+---------+
| DeepSeek-R1-Distill-Qwen-7B | math_500  | AveragePass@1 | Level 5  |   134 |  0.8557 | default |
+-----------------------------+-----------+---------------+----------+-------+---------+---------+
```

### Evaluating Model Thinking Efficiency

Once we have the model's reasoning results, we can begin evaluating its thinking efficiency. Before we start, we need to introduce several key metrics involved in the evaluation process: token efficiency, model thinking length, and the number of sub-thought chains.

- **Reasoning Tokens**: This refers to the number of tokens generated in the model's long chain of thought during reasoning. For O1/R1 type reasoning models, this metric represents the number of tokens before the `</think>` marker.
- **First Correct Tokens**: The number of tokens from the start of the model's reasoning process to the first position that can be recognized as the correct answer.
- **Reflection Tokens**: The number of tokens from the position of the first correct answer to the end of reasoning.
- **Num Thought**: This metric indicates the number of different thought paths generated by the model during reasoning. Specifically, it is calculated by counting the occurrences of generated marker words (e.g., "alternatively," "but wait," "let me reconsider"). This reflects the frequency at which the model switches its thought process during reasoning.
- **Token Efficiency**: This refers to the ratio of first correct tokens to the total number of reasoning tokens, calculated as follows:

    $$
    M_{token} = \frac{1}{N} \sum^{N}_{i=1}  \frac{\hat{T_i}}{T_i}
    $$

    Where $N$ is the number of problems, $\hat{T_i}$ is the number of tokens from the model's response to the first position recognized as the correct answer, and $T_i$ is the model's thinking length. If the model's answer is incorrect, then $\hat{T_i}$ is 0. A higher metric value indicates a higher proportion of effective thinking.

    In this evaluation framework, we refer to the construction method of [ProcessBench](https://github.com/QwenLM/ProcessBench) and use an additional model, `Qwen2.5-72B-Instruct`, to detect the earliest position of the correct answer during the reasoning process. To achieve this, we first decompose the model output into multiple steps, numbering each step, and then use the `Qwen2.5-72B-Instruct` model to verify these steps to identify the position of the first correct answer token. We have implemented three decomposition strategies:

    - **`separator`**: Decompose using the `\n\n` marker.
    - **`keywords`**: Decompose using marker words (e.g., `alternatively`, `but wait`, `let me reconsider`).
    - **`llm`**: Remove the `\n` markers from the response, use an LLM to rewrite the response, and insert `\n\n` markers for decomposition.

Using `Qwen2.5-72B-Instruct` as the judging model, simply run the following command to start the evaluation and obtain results:

```python
from evalscope.third_party.thinkbench import run_task

judge_config = dict(  # Evaluation service configuration
    api_key='EMPTY',
    base_url='http://0.0.0.0:8801/v1',
    model_name='Qwen2.5-72B-Instruct',
)

model_config = dict(
    report_path = './outputs/2025xxxx',  # Path to the model reasoning results from the previous step
    model_name = 'DeepSeek-R1-Distill-Qwen-7B',  # Model name
    tokenizer_path = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',  # Path to the model tokenizer for token count calculation
    dataset_name = 'math_500',  # Dataset name from the previous step
    subsets = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],  # Subsets from the previous step
    split_strategies='separator',  # Strategy for splitting reasoning steps, options are separator, keywords, llm
    judge_config=judge_config
)

max_tokens = 20000  # Filter outputs with token count less than max_tokens to improve evaluation efficiency
count = 200  # Filter count outputs for each subset to improve evaluation efficiency

# Evaluate model thinking efficiency
run_task(model_config, output_dir='outputs', max_tokens=max_tokens, count=count)
```

The output will look like this, showing the model's token efficiency, thinking length, and number of sub-thought chains for each difficulty level:

![DeepSeek-R1-Distill-Qwen-7B Thinking Efficiency](./images/DeepSeek-R1-Distill-Qwen-7B_math_500_metrics.png)
*Figure 1: DeepSeek-R1-Distill-Qwen-7B Thinking Efficiency*

Using the same method, we also evaluated four other reasoning models—QwQ-32B, QwQ-32B-Preview, DeepSeek-R1, DeepSeek-R1-Distill-Qwen-32B—and one non-reasoning model, Qwen2.5-Math-7B-Instruct (treating all tokens in the model output as part of the thought process), to observe the performance of different types of models. The specific results are summarized as follows:

![Comparison of Thinking Efficiency of 6 Models](./images/model_comparison_metrics_6models.png)
*Figure 2: Comparison of Thinking Efficiency of 6 Models*

Analyzing these line charts, we can draw some interesting conclusions:

- **Problem Difficulty vs. Model Performance**: As the problem difficulty increases, the accuracy of most models shows a downward trend, but QwQ-32B and DeepSeek-R1 perform exceptionally well, maintaining a high accuracy even on difficult problems, with QwQ-32B achieving the best performance at the highest difficulty level. Additionally, the output length for all models increases as problem difficulty rises, indicating that models require longer "thinking time" to solve more complex problems, consistent with the Inference-Time Scaling phenomenon.
- **Performance of O1/R1 Type Reasoning Models**:
    - For O1/R1 reasoning models, as the problem difficulty increases, the output length stabilizes while token efficiency also improves (DeepSeek-R1 increases from 36% to 54%, QwQ-32B from 31% to 49%). This indicates that reasoning-type models consume tokens in a more "worthwhile" manner for more complex problems. Conversely, for relatively simple problems, there may be more unnecessary token wastage: even for simple problems, there may be unnecessary repeated validations of answers. QwQ-32B produced a higher number of output tokens compared to other models, allowing it to maintain a high accuracy rate even for Level 5 difficult problems, but on the other hand, it may indicate an issue of over-analysis.
    - An interesting observation was noted: for problems at difficulty Level 4 and below, the number of sub-thought chains generated by the three models in the DeepSeek series remained relatively stable. However, at the challenging Level 5, there was a sudden significant increase in the number of sub-thought chains generated. This may be because Level 5 problems present a considerable challenge to these models, requiring multiple rounds of attempts and reasoning to arrive at solutions. In contrast, QwQ-32B and QwQ-32B-Preview exhibited a more uniform increase in the number of thought chains, which might reflect their differing strategies and capabilities in handling complex problems.
- **Performance of Non-Reasoning Models**: The accuracy of the non-reasoning model Qwen2.5-Math-7B-Instruct significantly decreased when dealing with high-difficulty math problems. Additionally, due to the lack of an in-depth thinking process, the output count of this model was only one-third that of reasoning models. It is evident from the graphs that specialized math models like Qwen2.5-Math-7B-Instruct outperform general reasoning models in terms of solving rates and resource consumption for ordinary problems, but as problem difficulty increases, the lack of a deep thinking process results in a more pronounced decline in model performance, exhibiting a clear "ceiling" effect.

## Tips

:::{card}

💡 During the writing of this best practice guide, I accumulated some insights to share:

1. **On the Definition of Thinking Efficiency Evaluation Metrics**:
    - This article draws on the definitions of "overthinking" and "underthinking" from the literature[^1] and [^2], simplifying the Outcome Efficiency metric and proposing the token efficiency metric. However, this metric primarily focuses on the number of generated tokens and does not capture all the details of the model's thought process.
    - The calculation of the number of sub-thought chains uses a heuristic approach, identifying common keywords predefined for this purpose. It should be noted that different models may require different sets of keywords to accurately capture their thinking processes.

2. **On the Applicability of the Metrics**:
    - Currently, these metrics are primarily applied to mathematical reasoning datasets, and thus may not fully reflect model performance in other application scenarios. For example, in open-ended question-answering or scenarios requiring creative responses, these metrics may be insufficient.

3. **On the Calculation of the Token Efficiency Metric**:
    - In the implementation process, we relied on an additional Judge model to assess the correctness of the model's reasoning steps. Referencing the work of ProcessBench[^4], this task is quite challenging for existing models and typically requires a strong model to make judgments.
    - If the Judge model makes incorrect judgments, it may affect the accuracy of the token efficiency metric, which means careful consideration is needed when selecting the Judge model.

:::

## Conclusion

This article evaluates the reasoning efficiency of several mainstream reasoning models, including QwQ-32B and DeepSeek-R1, based on the MATH-500 dataset. From the perspectives of "Token efficiency" and "Accuracy," we draw several noteworthy conclusions:
 - The model's ability to "think deeply" shows a clear correlation with performance, as more difficult problems require a "deeper" thought process.
 - Regarding reasoning efficiency evaluation, this article explores how to define quantifiable evaluation metrics based on related work on "overthinking" and "underthinking" and discusses the engineering implementation within the EvalScope framework.
 - Evaluating the reasoning efficiency of models provides crucial reference significance for GRPO and SFT training processes, helping to develop models that are "more efficient" and capable of "adaptive reasoning" based on problem difficulty.

## References

[^1]: Wang, Y. et al. Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs. Preprint at https://doi.org/10.48550/arXiv.2501.18585 (2025).
[^2]: Chen, X. et al. Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs. Preprint at https://doi.org/10.48550/arXiv.2412.21187 (2025).
[^3]: Think Less, Achieve More: Cut Reasoning Costs by 50% Without Sacrificing Accuracy. https://novasky-ai.github.io/posts/reduce-overthinking/.
[^4]: Zheng, C. et al. ProcessBench: Identifying Process Errors in Mathematical Reasoning. Preprint at https://doi.org/10.48550/arXiv.2412.06559(2024).
