# Mixed Data Evaluation

This framework supports mixing multiple evaluation datasets for a unified evaluation, aiming to use less data to achieve a more comprehensive assessment of the model's capabilities.

The overall evaluation process is as follows:

1. Define a data mixing schema: Specify which datasets to use for evaluation and how the data should be grouped.
2. Sample data: The framework will sample from each specified dataset according to the schema.
3. Unified evaluation: The sampled data will be used in a unified evaluation process.

:::{toctree}
:maxdepth: 2

schema.md
sample.md
evaluate.md
:::