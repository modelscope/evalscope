# Visualization

Visualization supports single model evaluation results and multi-model comparison, as well as visualization of mixed dataset evaluations.

## Install Dependencies

Install the dependencies required for visualization, including gradio, plotly, etc.
```bash
pip install 'evalscope[app]' -U
```

```{note}
Visualization requires `evalscope>=0.10.0` output evaluation reports. If the version is less than `0.10.0`, please upgrade `evalscope` first.
```

## Start Visualization Service

Run the following command to start the visualization service.
```bash
evalscope app
```
The supported command-line arguments are as follows:

- `--outputs`: A string type used to specify the root directory of the evaluation report, with a default value of `./outputs`.
- `--lang`: A string type used to specify the interface language, with a default value of `zh`, supports `zh` and `en`.
- `--share`: A flag indicating whether to share the application, default value is `False`.
- `--server-name`: A string type with a default value of `0.0.0.0`, used to specify the server name.
- `--server-port`: An integer type with a default value of `7860`, used to specify the server port.
- `--debug`: A flag indicating whether to debug the application, default value is `False`.

You can access the visualization service in the browser if the following output appears.
```text
* Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
```

### Quick Start

Run the following commands to download the sample and quickly experience the visualization feature. The sample includes evaluation results of the Qwen2.5-0.5B and Qwen2.5-7B models on several datasets for some examples.

```bash
git clone https://github.com/modelscope/evalscope
evalscope app --outputs evalscope/examples/viz
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