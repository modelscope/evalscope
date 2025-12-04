# Building an Evaluation Index

Through **Collection**, you can define your own **Evaluation Index** in EvalScope:  
Combine multiple datasets according to business value proportions, run evaluation once, and get a comprehensive score that aligns with your real-world scenarios.

---

## When Do You Need a Custom Index?

Many teams only look at single benchmarks (e.g., MMLU scores) when selecting models, but often encounter several typical issues:

- **A single dataset doesn't equal real-world scenarios**  
  Your actual scenarios might be:
  - RAG Q&A (knowledge base retrieval + answering)
  - Vertical domain analysis like finance/legal
  - Code completion and code review
  These are often difficult to fully cover with any single public dataset.

- **Public leaderboards don't equal your best model**  
  Leaderboards focus on "general capabilities," while you care more about "business value":  
  - Which model is better suited for your RAG workflow?  
  - Which model has the lowest error rate in domains you care about?  
  - Which model offers the best balance between cost and performance?

- **You need an index that "reflects your own values"**  
  Just like:
  - [Artificial Analysis Intelligence Index](https://artificialanalysis.ai/methodology/intelligence-benchmarking#intelligence-index-evaluation-suite-summary) focuses on general intelligence capabilities
  - [Vals Index](https://www.vals.ai/benchmarks/vals_index) focuses on business value

  You can also define an **Index that serves only your business decisions**.

EvalScope's **Collection** is designed for this purpose:  
You can "package" multiple datasets, set weights based on business importance, and create an evaluation index that truly belongs to you.

---

## Core Mental Model: Collection = Weighted Multi-Dataset Evaluation

You can think of Collection as:

> A **configuration file** describing "what I care about and their respective weights"  
> → Sample a mixed dataset  
> → Run evaluation once with EvalScope  
> → Get a "comprehensive model score sheet" weighted by your values.

From a user's perspective, you only need to focus on three things:

1. Which **datasets** do I want to evaluate?
2. How much **weight** does each dataset have in my decision-making?
3. How do I want to **sample** from these datasets (how many samples, what distribution)?

---

## From 0 to 1: How to Build an Index Using Collection?

Below is a minimum viable workflow to help you get started quickly.

### 1. Define Schema: Declare "What I Care About"

In the **Schema**, you need to do two things:

- Select the **list of datasets** to include in the Index  
  For example:
  - `gsm8k`: Basic math reasoning
  - An internally constructed RAG Q&A dataset
  - A code completion dataset

- Set **weights** for each dataset  
  Weights can reflect your emphasis on different capability dimensions:
  - RAG scenarios: 50%
  - Financial Q&A: 30%
  - General reasoning: 20%

**For more details, see:** [Defining Your Schema](schema.md).

---

### 2. Sample Data: Extract Representative Samples from Multiple Datasets

Based on the Schema, EvalScope can generate a **mixed dataset JSONL** for you.  
You can control the sampling method through several common strategies:

- **Weight-based sampling**: Data volume roughly allocated by weight proportion;
- **Stratified sampling**: Different difficulty levels/subtasks are all covered;
- **Uniform sampling**: Fixed number of samples from each dataset for easy comparison.

The final result is a "pre-mixed" JSONL file, and subsequent evaluation only needs to run once on this JSONL.

**Sampling methods and command examples:** See [How to Sample](sample.md).

---

### 3. Unified Evaluation: Get Your Custom Index Score in One Run

After obtaining the mixed JSONL, you can:

- Call `evalscope eval` just like evaluating a regular dataset
- Get all at once:
  - Scores for each sub-dataset
  - **Total Index score** aggregated by weights
  - Corresponding detailed logs, predictions, reviews, and reports

After evaluation, you can use:

- Report summaries to compare different models' performance under the same Index;
- EvalScope's **visualization app (`evalscope app`)** to visually compare model strengths and weaknesses.

**Specific evaluation commands and sample outputs:** See [How to Evaluate](evaluate.md).


:::{toctree}
:maxdepth: 1

schema.md
sample.md
evaluate.md
:::