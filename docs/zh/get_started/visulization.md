# 可视化

可视化功能支持单模型评测结果和多模型评测结果对比，支持数据集混合评测可视化。

## 安装依赖

安装可视化所需的依赖，包括gradio、plotly等。

```bash
pip install 'evalscope[app]' -U
```
```{note}
可视化功能需要`evalscope`版本大于等于`0.10.0`，若版本小于`0.10.0`，请先升级`evalscope`。
```

## 启动可视化服务

运行如下命令启动可视化服务。
```bash
evalscope app
```
输出如下内容即可在浏览器中访问可视化服务。
```text
* Running on local URL:  http://127.0.0.1:7861

To create a public link, set `share=True` in `launch()`.
```

## 功能介绍

1. 左侧配置选项：
   - 评测报告的根目录
   - 评测报告选择
   ![alt text](./images/setting.png)

2. 单模型评测结果：
   - 评测总览：展示评测数据集的组成和评测结果
   ![alt text](./images/report_overview.png)
   - 单个数据集评测详情，包含模型预测结果展示
   ![alt text](./images/report_details.png)

3. 多模型评测结果对比：
   - 使用雷达图和对比表格进行展示
   ![alt text](./images/model_compare.png)

4. 数据集混合评测可视化：
   - 按照模型能力维度进行可视化展示
   ![alt text](./images/collection.png)