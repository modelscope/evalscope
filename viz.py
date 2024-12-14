# Copyright (c) Alibaba, Inc. and its affiliates.
# flake8: noqa

import argparse
import json
import os
import pandas as pd
import plotly.graph_objects as go
import re
import seaborn as sns
import streamlit as st
import yaml


def generate_color_palette(n):
    palette = sns.color_palette('hls', n)
    return palette.as_hex()


def read_yaml(yaml_file) -> dict:
    """
    Read yaml file to dict.
    """
    with open(yaml_file, 'r') as f:
        try:
            stream = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
            raise e

    return stream


def read_jsonl(input_file):
    all_data = []
    with open(input_file, 'r') as input_jsonl:
        for line in input_jsonl:
            data = json.loads(line)
            all_data.append(data)

    return all_data


def cat_view(df, category, models):
    cat_df = df[df['category'] == category] if category != 'all' else df

    model_scores = {}
    for model in models:
        model_a_scores = (cat_df[cat_df['model_a'] == model]['scores'].apply(lambda x: x[0]).sum())
        model_b_scores = (cat_df[cat_df['model_b'] == model]['scores'].apply(lambda x: x[1]).sum())

        # calculate count of occurrences for model_a and model_b
        model_a_counts = cat_df[cat_df['model_a'] == model].shape[0]
        model_b_counts = cat_df[cat_df['model_b'] == model].shape[0]

        # calculate average scores for each model
        model_scores[model] = (model_a_scores + model_b_scores) / (model_a_counts + model_b_counts)

    return dict(
        category=category,
        count=cat_df['question_id'].nunique(),
        **dict(model_scores),
    )


def get_color(value):
    good_thresholds = [0.2, 0.1, 0.05]
    bad_thresholds = [-0.2, -0.1, -0.05]
    good_colors = ['#32CD32', '#98FF98', '#D0F0C0']
    bad_colors = ['#FF6347', '#FA8072', '#ffcccb']
    color = ''
    for i in range(len(good_thresholds)):
        if value > good_thresholds[i]:
            color = good_colors[i]
            break
    for i in range(len(bad_thresholds)):
        if value < bad_thresholds[i]:
            color = bad_colors[i]
            break
    return 'background-color: %s' % color if color else ''


def get_category_map(category_file):
    if not category_file or not os.path.exists(category_file):
        return dict()

    category_mapping = read_yaml(category_file)
    return category_mapping


def get_category_group(category_map, cat):
    for key, value in category_map.items():
        if cat in value or '*' in value:
            return key
    return cat


def show_table_view(df):
    models = df['model_a'].unique().tolist()
    for model in df['model_b'].unique().tolist():
        if model not in models:
            models.append(model)
    catogories = df['category'].unique().tolist()

    # Calculate the average score for each category
    cat_data = [cat_view(df, category, models) for category in catogories]
    cat_df = pd.DataFrame(cat_data)
    cat_df.sort_values(by=models[0], ascending=False, inplace=True, ignore_index=True)

    # Add total data
    total_data = cat_view(df, 'all', models)
    cat_df.loc[cat_df.shape[0]] = [
        '总计',
        total_data['count'],
        *[total_data[model] for model in models],
    ]

    # Render as link for each category
    cat_df['category'] = cat_df['category'].apply(
        lambda x: '<a href="?category={}&model_a={}&model_b={}" target="_self">{}</a>'.format(
            x, models[0], models[1], x), )

    # Format the table
    if len(models) == 2:
        cat_df['score diff'] = cat_df.apply(lambda x: (x[models[1]] - x[models[0]]) / x[models[0]], axis=1)
        cat_df.rename(
            columns={
                'category': '类别',
                'count': 'Case数量',
                'score diff': '分差百分比',
            },
            inplace=True,
        )
        style_df = cat_df.style.format({
            **dict(zip(models, ['{:.2f}' for model in models])),
            '分差百分比': '{:.2%}',
        }).applymap(
            get_color, subset=['分差百分比'])
    else:
        cat_df.rename(
            columns={
                'category': '类别',
                'count': 'Case数量',
            },
            inplace=True,
        )

        def format(row):
            p = '{:.2%}'.format((row[model] - row[baseline_model]) / row[baseline_model])
            p = '+' + p if p[0] != '-' else p
            return f'{row[model]:.2f} ({p})'

        def color(value):
            match = re.search(r'\((.*?)\)', value)
            return get_color(float(match.group(1).strip('%')) / 100) if match else ''

        baseline_model = models[0]
        for model in models[1:]:
            cat_df[model] = cat_df.apply(
                format,
                axis=1,
            )
        cat_df[baseline_model] = cat_df[baseline_model].apply(lambda x: '{:.2f}'.format(x))
        style_df = cat_df.style.applymap(color, subset=models[1:])

    # df_html = style_df.to_html(escape=False, index=False) # TODO
    df_html = style_df.to_html()
    st.markdown(df_html, unsafe_allow_html=True)


def show_radar_chart(df):
    score_list = []
    for index, row in df.iterrows():
        score_list.append(dict(model=row['model_a'], category=row['category'], score=row['scores'][0]))
        score_list.append(dict(model=row['model_b'], category=row['category'], score=row['scores'][1]))

    score_df = pd.DataFrame(score_list)
    df_agg = score_df.groupby(['model', 'category'])['score'].mean().reset_index()

    pivot_df = df_agg.pivot(index='model', columns='category', values='score').fillna(0)

    categories = pivot_df.columns.tolist()

    fig = go.Figure()

    num_models = len(pivot_df.index.tolist())
    color_palette = generate_color_palette(num_models)
    color_dict = dict(zip(pivot_df.index.tolist(), color_palette))

    for model in pivot_df.index.tolist():
        model_values = pivot_df[pivot_df.index == model].values.tolist()[0]
        model_values.append(model_values[0])  # Make the data cyclic
        fig.add_trace(
            go.Scatterpolar(
                r=model_values,
                theta=categories + [categories[0]],  # Make the categories cyclic
                fill='none',
                name=model,
                line=dict(color=color_dict[model]),
            ))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), showlegend=True)
    st.plotly_chart(fig)


def show_single_result(df, category, model_a, model_b):
    categories = df['category'].unique().tolist()
    model_names = df['model_a'].unique().tolist()

    col1, col2 = st.columns([1, 3])
    with col1:
        category = st.selectbox(
            '选择类别',
            categories,
            index=categories.index(category) if category in categories else 0,
        )

    df = df[df['category'] == category]
    with col2:
        ques = st.selectbox('选择问题', df['question'].unique().tolist())

    col1, col2 = st.columns(2)
    with col1:
        model_a_options = [model_a]
        model_a = st.selectbox(
            '选择模型A',
            model_a_options,
            index=model_a_options.index(model_a),
        )

    with col2:
        model_b = st.selectbox('选择模型B', [m for m in model_names if m != model_a])

    with st.container():
        st.markdown(
            """
<div style="background-color:#DEEBF7;padding:10px;margin:10px 0;border-radius:8px">
<b>问题：</b>
{ques}
</div>""".format(ques=ques),
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.write(
                """
<div style="background-color:#E2F0D9;padding:10px;border-radius:8px">
<p><b>{model_a} 回答</b></p>
{output_a}
</div>""".format(
                    model_a=model_a,
                    output_a=df[(df['question'] == ques) & (df['model_a'] == model_a)].iloc[0]['output_a'],
                ),
                unsafe_allow_html=True,
            )
        with col2:
            st.write(
                """
<div style="background-color:#E2F0D9;padding:10px;border-radius:8px">
<p><b>{model_b} 回答</b></p>
{output_b}
</div>""".format(
                    model_b=model_b,
                    output_b=df[(df['question'] == ques) & (df['model_b'] == model_b)].iloc[0]['output_b'],
                ),
                unsafe_allow_html=True,
            )

    score_1 = df[(df['question'] == ques) & (df['model_a'] == model_a)].iloc[0]['scores']
    score_2 = df[(df['question'] == ques) & (df['model_b'] == model_a)].iloc[0]['scores']

    scores = [
        {
            'round': '第一轮',
            model_a: score_1[0],
            model_b: score_1[1]
        },
        {
            'round': '第二轮',
            model_a: score_2[1],
            model_b: score_2[0]
        },
    ]

    score_df = pd.DataFrame(scores)
    styled_df = score_df.style.highlight_max(
        axis=1,
        subset=[model_a, model_b],
        color='lightgreen',
    )
    styled_df.format({
        model_a: '{:.1f}',
        model_b: '{:.1f}',
    })
    # score_html = styled_df.to_html(index=False)       # TODO
    score_html = styled_df.to_html()
    st.markdown(
        """
<div style="background-color:#FBE5D6;padding:10px;margin:10px 0;border-radius:8px">
<b>GPT-4 评分：</b>
{score_html}
</div>""".format(score_html=score_html),
        unsafe_allow_html=True,
    )


def run_app(review_file, category_file):
    category_map = get_category_map(category_file)

    review_file = os.path.abspath(review_file)
    data = read_jsonl(review_file)
    df = pd.DataFrame(data)
    df = df[[
        'model_a',
        'model_b',
        'scores',
        'category',
        'question_id',
        'question',
        'output_a',
        'output_b',
    ]]
    df['category'] = df['category'].apply(lambda x: get_category_group(category_map, x))

    query_params = st.experimental_get_query_params()
    if 'category' in query_params:
        st.set_page_config(layout='wide')
        st.write("<a href='/' target='_self'>返回</a>", unsafe_allow_html=True)
        show_single_result(
            df,
            query_params['category'][0],
            query_params['model_a'][0],
            query_params['model_b'][0],
        )
        st.write("<a href='/' target='_self'>返回</a>", unsafe_allow_html=True)
    else:
        st.set_page_config(layout='centered')
        st.write('### 评测结果展示（Arena 模式）')
        st.write('#### 模型分类别得分')
        show_table_view(df)
        st.write('#### 模型得分雷达图')
        show_radar_chart(df)


def parse_args():
    parser = argparse.ArgumentParser(description='Run visualization on a evaluation.')

    parser.add_argument(
        '--review-file', type=str, default='evalscope/registry/data/qa_browser/battle.jsonl', required=True)
    parser.add_argument(
        '--category-file', type=str, default='evalscope/registry/data/qa_browser/category_mapping.yaml', required=True)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    print(args)
    run_app(args.review_file, args.category_file)


if __name__ == '__main__':

    print(
        '**Usage:\n streamlit run viz.py -- --review-file evalscope/registry/data/qa_browser/battle.jsonl --category-file evalscope/registry/data/qa_browser/category_mapping.yaml'
    )
    main()
