"""
Visualization utilities for the Evalscope dashboard.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List

from evalscope.constants import DataCollection
from evalscope.report import Report, ReportKey, get_data_frame
from evalscope.utils.logger import get_logger
from ..constants import DEFAULT_BAR_WIDTH, PLOTLY_THEME

logger = get_logger()


def plot_single_report_scores(df: pd.DataFrame):
    if df is None:
        return None
    logger.debug(f'df: \n{df}')
    plot = px.bar(df, x=df[ReportKey.dataset_name], y=df[ReportKey.score], text=df[ReportKey.score])

    width = DEFAULT_BAR_WIDTH if len(df[ReportKey.dataset_name]) <= 5 else None
    plot.update_traces(width=width, texttemplate='%{text:.2f}', textposition='outside')
    plot.update_layout(uniformtext_minsize=12, uniformtext_mode='hide', yaxis=dict(range=[0, 1]), template=PLOTLY_THEME)
    return plot


def plot_single_report_sunburst(report_list: List[Report]):
    if report_list[0].name == DataCollection.NAME:
        df = get_data_frame(report_list=report_list)
        categories = sorted([i for i in df.columns if i.startswith(ReportKey.category_prefix)])
        path = categories + [ReportKey.subset_name]
    else:
        df = get_data_frame(report_list=report_list, flatten_metrics=False)
        categories = sorted([i for i in df.columns if i.startswith(ReportKey.category_prefix)])
        path = [ReportKey.dataset_name] + categories + [ReportKey.subset_name]
    logger.debug(f'df: \n{df}')
    df[categories] = df[categories].fillna('default')  # NOTE: fillna for empty categories

    plot = px.sunburst(
        df,
        path=path,
        values=ReportKey.num,
        color=ReportKey.score,
        color_continuous_scale='RdYlGn',  # see https://plotly.com/python/builtin-colorscales/
        color_continuous_midpoint=np.average(df[ReportKey.score], weights=df[ReportKey.num]),
        template=PLOTLY_THEME,
        maxdepth=4
    )
    plot.update_traces(insidetextorientation='radial')
    plot.update_layout(margin=dict(t=10, l=10, r=10, b=10), coloraxis=dict(cmin=0, cmax=1), height=600)
    return plot


def plot_single_dataset_scores(df: pd.DataFrame):
    # TODO: add metric radio and replace category name
    plot = px.bar(
        df,
        x=df[ReportKey.metric_name],
        y=df[ReportKey.score],
        color=df[ReportKey.subset_name],
        text=df[ReportKey.score],
        barmode='group'
    )

    width = 0.2 if len(df[ReportKey.subset_name]) <= 3 else None
    plot.update_traces(width=width, texttemplate='%{text:.2f}', textposition='outside')
    plot.update_layout(uniformtext_minsize=12, uniformtext_mode='hide', yaxis=dict(range=[0, 1]), template=PLOTLY_THEME)
    return plot


def plot_multi_report_radar(df: pd.DataFrame):
    fig = go.Figure()

    grouped = df.groupby(ReportKey.model_name)
    common_datasets = set.intersection(*[set(group[ReportKey.dataset_name]) for _, group in grouped])

    for model_name, group in grouped:
        common_group = group[group[ReportKey.dataset_name].isin(common_datasets)]
        fig.add_trace(
            go.Scatterpolar(
                r=common_group[ReportKey.score],
                theta=common_group[ReportKey.dataset_name],
                name=model_name,
                fill='toself'
            )
        )

    fig.update_layout(
        template=PLOTLY_THEME,
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        margin=dict(t=20, l=20, r=20, b=20)
    )
    return fig
