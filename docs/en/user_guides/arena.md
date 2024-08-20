# Arena Mode

Arena mode allows multiple candidate models to be evaluated through pairwise comparisons. You can choose to use the AI Enhanced Auto-Reviewer (AAR) for automated evaluations or conduct manual assessments, ultimately producing an evaluation report. The framework supports the following three model evaluation processes:

## Pairwise Comparison of All Models (Pairwise Mode)

### 1. Environment Preparation
```text
a. Data preparation: The question data format can be referenced in: evalscope/registry/data/question.jsonl
b. If you need to use the automated evaluation process (AAR), you must configure the related environment variables. For example, if using the GPT-4 based auto-reviewer, you need to configure the following environment variable:
> export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
```

### 2. Configuration File
```text
The configuration file for the arena evaluation process can be found at: evalscope/registry/config/cfg_arena.yaml
Field descriptions:
    questions_file: Path to the question data
    answers_gen: Generation of candidate model predictions, supporting multiple models; you can control whether to enable a model through the enable parameter.
    reviews_gen: Generation of evaluation results, currently defaults to using GPT-4 as an Auto-reviewer; you can control whether to enable this step through the enable parameter.
    elo_rating: ELO rating algorithm, which can be controlled via the enable parameter; note that this step depends on the review_file being present.
```

### 3. Execution Script
```shell
# Usage:
cd evalscope
# Dry-run mode (model answers will be generated normally, but the expert model, such as GPT-4, will not be invoked; evaluation results will be generated randomly)
python evalscope/run_arena.py -c registry/config/cfg_arena.yaml --dry-run
# Execute the evaluation process
python evalscope/run_arena.py --c registry/config/cfg_arena.yaml
```

### 4. Result Visualization
```shell
# Usage:
streamlit run viz.py --review-file evalscope/registry/data/qa_browser/battle.jsonl --category-file evalscope/registry/data/qa_browser/category_mapping.yaml
```

## Single Model Scoring Mode (Single Mode)

In this mode, we score only a single model’s output without conducting pairwise comparisons.

### 1. Configuration File
```text
The configuration file for the evaluation process can be found at: evalscope/registry/config/cfg_single.yaml
Field descriptions:
    questions_file: Path to the question data
    answers_gen: Generation of candidate model predictions, supporting multiple models; you can control whether to enable a model through the enable parameter.
    reviews_gen: Generation of evaluation results, currently defaults to using GPT-4 as an Auto-reviewer; you can control whether to enable this step through the enable parameter.
    rating_gen: Rating algorithm, which can be controlled via the enable parameter; note that this step depends on the review_file being present.
```

### 2. Execution Script
```shell
# Example:
python evalscope/run_arena.py --c registry/config/cfg_single.yaml
```

## Baseline Model Comparison Mode (Pairwise-Baseline Mode)

In this mode, we select a baseline model, and other models are scored in comparison to the baseline. This mode allows for easy addition of new models to the leaderboard (you only need to score the new model against the baseline model).

### 1. Configuration File
```text
The configuration file for the evaluation process can be found at: evalscope/registry/config/cfg_pairwise_baseline.yaml
Field descriptions:
    questions_file: Path to the question data
    answers_gen: Generation of candidate model predictions, supporting multiple models; you can control whether to enable a model through the enable parameter.
    reviews_gen: Generation of evaluation results, currently defaults to using GPT-4 as an Auto-reviewer; you can control whether to enable this step through the enable parameter.
    rating_gen: Rating algorithm, which can be controlled via the enable parameter; note that this step depends on the review_file being present.
```

### 2. Execution Script
```shell
# Example:
python evalscope/run_arena.py --c registry/config/cfg_pairwise_baseline.yaml
```