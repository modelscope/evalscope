# flake8: noqa
import numpy as np
import os
from collections import defaultdict
from typing import Dict, List

from evalscope.utils import get_logger

logger = get_logger()


class End2EndEvaluator():

    def __init__(
        self,
        prediction: List,
        reference: List,
        metrics: Dict,
        match_method: str = 'quick_match',
        filter_types: dict = None
    ):

        self.match_method = match_method
        self.references = reference
        self.predictions = prediction
        self.dafault_metircs_dict = metrics

        filtered_gt_samples = []
        if filter_types:
            for gt_sample in self.references:
                select_flag = True
                for k, v in filter_types.items():
                    if gt_sample['page_info']['page_attribute'][k] != v:
                        select_flag = False
                if select_flag:
                    filtered_gt_samples.append(gt_sample)
        else:
            filtered_gt_samples = self.references  #[{},{},{}]
        self.references = filtered_gt_samples

    def score(self) -> dict:
        samples = self.get_matched_elements(self.references, self.predictions)
        metrics = self.process_generated_metric_results(samples)
        return metrics

    def get_page_elements(self, selected_annos):
        saved_element_dict = defaultdict(list)
        related_truncated = []
        truncated_all = {}
        for relation in selected_annos['extra']['relation']:  # Handle truncated text issues
            if relation['relation_type'] == 'truncated':
                truncated_all[relation['source_anno_id']] = ''
                truncated_all[relation['target_anno_id']] = ''
                exist_flag = False
                for merge_list in related_truncated:
                    if relation['source_anno_id'] in merge_list or relation[
                        'target_anno_id'] in merge_list:  # Consider cases where three text blocks may need to be merged
                        merge_list.append(relation['source_anno_id'])
                        merge_list.append(relation['target_anno_id'])
                        exist_flag = True
                if not exist_flag:
                    related_truncated.append([relation['source_anno_id'], relation['target_anno_id']])

        for item in selected_annos['layout_dets']:
            if item['anno_id'] not in truncated_all.keys():
                saved_element_dict[item['category_type']].append(item)
            else:
                truncated_all[item['anno_id']] = item

        for merge_list in related_truncated:
            text_block_list = [truncated_all[key] for key in merge_list]
            sorted_block = sorted(text_block_list, key=lambda x: x['order'])
            text = ''
            for block in sorted_block:
                text += block['text']
            merged_block = {
                'category_type': sorted_block[0]['category_type'],  # Directly use information from the first block
                'order': sorted_block[0]['order'],
                'anno_id': sorted_block[0]['anno_id'],
                'text': text,
                'merge_list': sorted_block
            }
            saved_element_dict[sorted_block[0]['category_type']].append(merged_block)

        return saved_element_dict

    def get_page_elements_list(self, gt_page_elements, category_list):
        element_list = []
        for category_type in category_list:
            if gt_page_elements.get(category_type):
                element_list.extend(gt_page_elements[category_type])
        return element_list

    def get_sorted_text_list(self, selected_annos):
        # txt_type: text, latex, html
        text_list = []
        for item in selected_annos:
            if item.get('order'):
                order = item['order']
            else:
                order = 0
            # 【txt_type,selecte_annos]
            text_list.append((order, item))
        sorted_text_list = sorted(text_list, key=lambda x: x[0])
        return [_[1] for _ in sorted_text_list]

    def filtered_out_ignore(self, items, ignore_category_list):
        filted_items = []
        for item in items:
            if item['gt_category_type'] not in ignore_category_list:
                filted_items.append(item)
        return filted_items

    def get_order_paired(self, order_match_s, img_name):
        matched = [(item['gt_position'], item['pred_position'])
                   for item in order_match_s
                   if (item['gt_position'] != [''] and item['pred_position'] != '')]
        gt_idx_all = [item['gt_position'] for item in order_match_s if (item['gt_position'] != [''])]
        read_order_pred = [i[0] for i in sorted(matched, key=lambda x: x[1])]
        read_order_gt = sum(gt_idx_all, [])  # Convert to one-dimensional list
        read_order_gt = [x for x in read_order_gt if x]
        gt = sorted(read_order_gt)
        pred = sum(read_order_pred, [])
        pred = [x for x in pred if x]
        if len(pred) > 0 or len(gt) > 0:
            import Levenshtein
            edit = Levenshtein.distance(gt, pred) / max(len(pred), len(gt))
            return {'gt': gt, 'pred': pred, 'img_id': img_name, 'edit': edit}
        else:
            return {}  # If both GT and pred are empty for the page, return empty

    def formula_format(self, formula_matches, img_name):
        # formated_list = []
        for i, item in enumerate(formula_matches):
            item['img_id'] = img_name + '_' + str(i)
        return formula_matches

    def get_matched_elements(self, references: list, predictions: list) -> dict:
        from .metrics import recogition_end2end_base_dataset, recogition_end2end_table_dataset

        plain_text_match = []
        display_formula_match = []
        html_table_match = []
        latex_table_match = []
        order_match = []

        for i, sample in enumerate(references):
            img_name = os.path.basename(sample['page_info']['image_path'])
            pred_content = predictions[i]
            result = self.process_get_matched_elements(sample, pred_content, img_name)
            [
                plain_text_match_clean, formated_display_formula, latex_table_match_s, html_table_match_s,
                order_match_single
            ] = result

            if order_match_single:
                order_match.append(order_match_single)
            if plain_text_match_clean:
                plain_text_match.extend(plain_text_match_clean)
            if formated_display_formula:
                display_formula_match.extend(formated_display_formula)
            if latex_table_match_s:
                latex_table_match.extend(latex_table_match_s)
            if html_table_match_s:
                html_table_match.extend(html_table_match_s)

        if len(latex_table_match) > len(html_table_match):
            table_match = latex_table_match
            table_format = 'latex'
        else:
            table_match = html_table_match
            table_format = 'html'

        matched_samples_all = {
            'text_block': recogition_end2end_base_dataset(plain_text_match),
            'display_formula': recogition_end2end_base_dataset(display_formula_match),
            'table': recogition_end2end_table_dataset(table_match, table_format),
            'reading_order': recogition_end2end_base_dataset(order_match)
        }

        return matched_samples_all

    def process_get_matched_elements(self, sample, pred_content, img_name):
        from .utils import match_gt2pred_no_split, match_gt2pred_quick, match_gt2pred_simple, md_tex_filter

        if self.match_method == 'simple_match':  # add match choice
            match_gt2pred = match_gt2pred_simple
        elif self.match_method == 'quick_match':
            match_gt2pred = match_gt2pred_quick
        elif self.match_method == 'no_split':
            match_gt2pred = match_gt2pred_no_split
        else:
            match_gt2pred = match_gt2pred_quick

        pred_dataset = md_tex_filter(pred_content)
        gt_page_elements = self.get_page_elements(sample)

        text_all = self.get_page_elements_list(
            gt_page_elements, [
                'text_block', 'title', 'code_txt', 'code_txt_caption', 'reference', 'equation_caption',
                'figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm',
                'code_algorithm_caption', 'header', 'footer', 'page_footnote', 'page_number'
            ]
        )

        display_formula_match_s = []
        plain_text_match_clean = []
        latex_table_match_s = []
        html_table_match_s = []
        order_match_single = []
        if text_all:
            gt_text_list = self.get_sorted_text_list(text_all)
            try:
                plain_text_match_s = match_gt2pred(gt_text_list, pred_dataset['text_all'], 'text', img_name)
            except Exception as e:
                logger.warning(f'Error for plain text match of {img_name}, match_gt2pred_simple will be used.')
                plain_text_match_s = match_gt2pred_simple(gt_text_list, pred_dataset['text_all'], 'text', img_name)
                logger.error(str(e))

            if not plain_text_match_s:
                logger.warning(f'No text match of {img_name}. The plain text match will be empty.')
            else:
                plain_text_match_clean = self.filtered_out_ignore(
                    plain_text_match_s, [
                        'figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm',
                        'code_algorithm_caption', 'header', 'footer', 'page_footnote', 'page_number', 'equation_caption'
                    ]
                )

        if gt_page_elements.get('equation_isolated'):
            gt_display_list = self.get_sorted_text_list(gt_page_elements['equation_isolated'])
            display_formula_match_s = match_gt2pred(
                gt_display_list, pred_dataset['equation_isolated'], 'formula', img_name
            )
            display_formula_match_s = [x for x in display_formula_match_s if x['gt_idx'] != ['']]
            if not display_formula_match_s:
                logger.warning(f'No display_formula_match of {img_name}. The display_formula_match will be empty.')

        if gt_page_elements.get('table'):
            gt_table_list = self.get_sorted_text_list(gt_page_elements['table'])
            if pred_dataset['latex_table']:
                latex_table_match_s = match_gt2pred_simple(
                    gt_table_list, pred_dataset['latex_table'], 'latex_table', img_name
                )
                latex_table_match_s = [x for x in latex_table_match_s if x['gt_idx'] != ['']]
            if pred_dataset['html_table']:
                html_table_match_s = match_gt2pred_simple(
                    gt_table_list, pred_dataset['html_table'], 'html_table', img_name
                )
                html_table_match_s = [x for x in html_table_match_s if x['gt_idx'] != ['']]
            else:
                html_table_match_s = match_gt2pred_simple(gt_table_list, [], 'html_table', img_name)
                html_table_match_s = [x for x in html_table_match_s if x['gt_idx'] != ['']]

        order_match_s = plain_text_match_clean
        if order_match_s:
            order_match_single = self.get_order_paired(order_match_s, img_name)

        return [
            plain_text_match_clean, display_formula_match_s, latex_table_match_s, html_table_match_s, order_match_single
        ]

    def process_generated_metric_results(self, samples, save_name: str = 'end2end_quick_match'):
        from .metrics import METRIC_REGISTRY, get_full_labels_results, get_page_split, show_result

        result_all = {}
        page_info = {}
        metircs_dict = self.dafault_metircs_dict
        pages = self.references  #gt_samples list

        for page in pages:
            img_path = os.path.basename(page['page_info']['image_path'])
            page_info[img_path] = page['page_info']['page_attribute']

        for element in metircs_dict.keys():

            result = {}
            group_info = metircs_dict[element].get('group', [])
            # samples = samples.get(element) ##
            cur_samples = samples[element]

            for metric in metircs_dict[element]['metric']:
                metric_val = METRIC_REGISTRY.get(metric)

                cur_samples, result_s = metric_val(cur_samples).evaluate(group_info, f'{save_name}_{element}')
                if result_s:
                    result.update(result_s)

            if result:
                logger.info(f'{element}')
                show_result(result)
            result_all[element] = {}

            group_result = get_full_labels_results(cur_samples)
            page_result = get_page_split(cur_samples, page_info)

            result_all[element] = {'all': result, 'group': group_result, 'page': page_result}

        save_dict = {}
        en_overall = []
        ch_overall = []
        for category_type, metric in [('text_block', 'Edit_dist'), ('display_formula', 'Edit_dist'),
                                      ('display_formula', 'CDM'), ('table', 'TEDS'), ('table', 'Edit_dist'),
                                      ('reading_order', 'Edit_dist')]:
            if metric == 'TEDS':
                if category_type in result_all and 'page' in result_all[category_type] and metric in result_all[
                    category_type]['page']:
                    save_dict[category_type + '_' + metric
                              + '_EN'] = result_all[category_type]['page'][metric]['language: english']
                    save_dict[category_type + '_' + metric
                              + '_CH'] = result_all[category_type]['page'][metric]['language: simplified_chinese']
                else:
                    save_dict[category_type + '_' + metric + '_EN'] = np.nan
                    save_dict[category_type + '_' + metric + '_CH'] = np.nan
            else:
                if category_type in result_all and 'page' in result_all[category_type] and metric in result_all[
                    category_type]['page']:
                    save_dict[category_type + '_' + metric
                              + '_EN'] = result_all[category_type]['page'][metric].get('language: english', np.nan)
                    save_dict[category_type + '_' + metric + '_CH'] = result_all[category_type]['page'][metric].get(
                        'language: simplified_chinese', np.nan
                    )
                else:
                    save_dict[category_type + '_' + metric + '_EN'] = np.nan
                    save_dict[category_type + '_' + metric + '_CH'] = np.nan

            if metric == 'Edit_dist':
                if category_type in result_all and 'page' in result_all[category_type] and metric in result_all[
                    category_type]['page']:
                    en_overall.append(result_all[category_type]['page'][metric].get('language: english', np.nan))
                    ch_overall.append(
                        result_all[category_type]['page'][metric].get('language: simplified_chinese', np.nan)
                    )
                else:
                    en_overall.append(np.nan)
                    ch_overall.append(np.nan)

        en_overall_filtered = [x for x in en_overall if not np.isnan(x)]
        ch_overall_filtered = [x for x in ch_overall if not np.isnan(x)]
        save_dict['overall_EN'] = sum(en_overall_filtered) / len(en_overall_filtered) if en_overall_filtered else np.nan
        save_dict['overall_CH'] = sum(ch_overall_filtered) / len(ch_overall_filtered) if ch_overall_filtered else np.nan

        return save_dict
