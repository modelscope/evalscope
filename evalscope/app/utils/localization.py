"""
Localization utilities for the Evalscope dashboard.
"""
from typing import Any, Dict


def get_sidebar_locale(lang: str) -> Dict[str, str]:
    locale_dict = {
        'settings': {
            'zh': '设置',
            'en': 'Settings'
        },
        'report_root_path': {
            'zh': '报告根路径',
            'en': 'Report Root Path'
        },
        'select_reports': {
            'zh': '请选择报告',
            'en': 'Select Reports'
        },
        'load_btn': {
            'zh': '加载并查看',
            'en': 'Load & View'
        },
        'note': {
            'zh': '请选择报告并点击`加载并查看`来查看数据',
            'en': 'Please select reports and click `Load & View` to view the data'
        },
        'warning': {
            'zh': '没有找到报告，请检查路径',
            'en': 'No reports found, please check the path'
        }
    }
    return {k: v[lang] for k, v in locale_dict.items()}


def get_visualization_locale(lang: str) -> Dict[str, str]:
    locale_dict = {
        'visualization': {
            'zh': '可视化',
            'en': 'Visualization'
        },
        'single_model': {
            'zh': '单模型',
            'en': 'Single Model'
        },
        'multi_model': {
            'zh': '多模型',
            'en': 'Multi Model'
        }
    }
    return {k: v[lang] for k, v in locale_dict.items()}


def get_single_model_locale(lang: str) -> Dict[str, str]:
    locale_dict = {
        'select_report': {
            'zh': '选择报告',
            'en': 'Select Report'
        },
        'task_config': {
            'zh': '任务配置',
            'en': 'Task Config'
        },
        'datasets_overview': {
            'zh': '数据集概览',
            'en': 'Datasets Overview'
        },
        'dataset_components': {
            'zh': '数据集组成',
            'en': 'Dataset Components'
        },
        'dataset_scores': {
            'zh': '数据集分数',
            'en': 'Dataset Scores'
        },
        'report_analysis': {
            'zh': '报告智能分析',
            'en': 'Report Intelligent Analysis'
        },
        'dataset_scores_table': {
            'zh': '数据集分数表',
            'en': 'Dataset Scores Table'
        },
        'dataset_details': {
            'zh': '数据集详情',
            'en': 'Dataset Details'
        },
        'select_dataset': {
            'zh': '选择数据集',
            'en': 'Select Dataset'
        },
        'model_prediction': {
            'zh': '模型预测',
            'en': 'Model Prediction'
        },
        'select_subset': {
            'zh': '选择子集',
            'en': 'Select Subset'
        },
        'answer_mode': {
            'zh': '答案模式',
            'en': 'Answer Mode'
        },
        'page': {
            'zh': '页码',
            'en': 'Page'
        },
        'score_threshold': {
            'zh': '分数阈值',
            'en': 'Score Threshold'
        },
    }
    return {k: v[lang] for k, v in locale_dict.items()}


def get_multi_model_locale(lang: str) -> Dict[str, str]:
    locale_dict = {
        'select_reports': {
            'zh': '请选择报告',
            'en': 'Select Reports'
        },
        'models_overview': {
            'zh': '模型概览',
            'en': 'Models Overview'
        },
        'model_radar': {
            'zh': '模型对比雷达',
            'en': 'Model Comparison Radar'
        },
        'model_scores': {
            'zh': '模型对比分数',
            'en': 'Model Comparison Scores'
        },
        'model_comparison_details': {
            'zh': '模型对比详情',
            'en': 'Model Comparison Details'
        },
        'select_model_a': {
            'zh': '选择模型A',
            'en': 'Select Model A'
        },
        'select_model_b': {
            'zh': '选择模型B',
            'en': 'Select Model B'
        },
        'select_dataset': {
            'zh': '选择数据集',
            'en': 'Select Dataset'
        },
        'model_predictions': {
            'zh': '模型预测',
            'en': 'Model Predictions'
        },
        'select_subset': {
            'zh': '选择子集',
            'en': 'Select Subset'
        },
        'answer_mode': {
            'zh': '答案模式',
            'en': 'Answer Mode'
        },
        'score_threshold': {
            'zh': '分数阈值',
            'en': 'Score Threshold'
        },
        'comparison_counts': {
            'zh': '对比统计',
            'en': 'Comparison Counts'
        },
        'page': {
            'zh': '页码',
            'en': 'Page'
        },
        'input': {
            'zh': '输入',
            'en': 'Input'
        },
        'gold_answer': {
            'zh': '标准答案',
            'en': 'Gold Answer'
        },
        'score': {
            'zh': '分数',
            'en': 'Score'
        },
        'normalized_score': {
            'zh': '归一化分数',
            'en': 'Normalized Score'
        },
        'prediction': {
            'zh': '预测',
            'en': 'Prediction'
        },
        'generated': {
            'zh': '生成结果',
            'en': 'Generated'
        }
    }
    return {k: v[lang] for k, v in locale_dict.items()}


def get_app_locale(lang: str) -> Dict[str, str]:
    locale_dict = {
        'title': {
            'zh': '📈 EvalScope 看板',
            'en': '📈 Evalscope Dashboard'
        },
        'star_beggar': {
            'zh':
            '喜欢<a href=\"https://github.com/modelscope/evalscope\" target=\"_blank\">EvalScope</a>就动动手指给我们加个star吧 🥺 ',
            'en':
            'If you like <a href=\"https://github.com/modelscope/evalscope\" target=\"_blank\">EvalScope</a>, '
            'please take a few seconds to star us 🥺 '
        },
        'note': {
            'zh': '请选择报告',
            'en': 'Please select reports'
        }
    }
    return {k: v[lang] for k, v in locale_dict.items()}
