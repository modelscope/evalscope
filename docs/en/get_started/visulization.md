# Visualization

Visualization supports single model evaluation results and multi-model comparison, as well as visualization of mixed dataset evaluations.
## Install Dependencies

Install the dependencies required for visualization, including gradio, plotly, etc.
```bash
pip install 'evalscope[app]'
```

```{note}
Visualization requires `evalscope` version greater than or equal to `0.10.0`. If the version is less than `0.10.0`, please upgrade `evalscope` first.
```

## Start Visualization Service

Run the following command to start the visualization service.
```bash
evalscope app
```
You can access the visualization service in the browser if the following output appears.
```text
* Running on local URL:  http://127.0.0.1:7861

To create a public link, set `share=True` in `launch()`.
```

## Features Introduction

1. Configuration options on the left:
   - Root directory of evaluation reports
   - Selection of evaluation reports
   ![alt text](./images/setting.png)

2. Single Model Evaluation Results:
   - Evaluation Overview: Displays the composition of the evaluation dataset and the evaluation results
   ![alt text](./images/report_overview.png)
   - Detailed evaluation of a single dataset, including model prediction results
   ![alt text](./images/report_details.png)

3. Comparison of Multiple Model Evaluation Results:
   - Displayed using radar charts and comparison tables
   ![alt text](./images/model_compare.png)

4. Visualization of Mixed Dataset Evaluations:
   - Visual representation based on model capability dimensions
   ![alt text](./images/collection.png)