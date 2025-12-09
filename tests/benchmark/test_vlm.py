# Copyright (c) Alibaba, Inc. and its affiliates.
from dotenv import dotenv_values

env = dotenv_values('.env')


from evalscope.constants import EvalType, JudgeStrategy
from evalscope.utils.logger import get_logger
from tests.common import TestBenchmark

logger = get_logger()


class TestVLMBenchmark(TestBenchmark):
    """Benchmark evaluation test cases."""

    def setUp(self):
        """Setup common test configuration."""
        self.base_config = {
            'model': 'qwen-vl-plus',
            'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'api_key': env.get('DASHSCOPE_API_KEY'),
            'eval_type': EvalType.SERVICE,
            'eval_batch_size': 5,
            'limit': 5,
            'generation_config': {
                'max_tokens': 2048,
                'temperature': 0.0,
                'seed': 42,
                'parallel_tool_calls': True
            },
            'judge_strategy': JudgeStrategy.AUTO,
            'judge_worker_num': 5,
            'judge_model_args': {
                'model_id': 'qwen-plus',
                'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'api_key': env.get('DASHSCOPE_API_KEY'),
                'generation_config': {
                    'temperature': 0.0,
                    'max_tokens': 4096,
                }
            },
            'debug': True,
        }

    def test_mmmu(self):
        dataset_args = {
            'subset_list':[
                'Accounting',
                'Math',
                # 'Architecture_and_Engineering'
            ]
        }
        self._run_dataset_test('mmmu', dataset_args=dataset_args)

    def test_cmmmu(self):
        dataset_args = {
            # 'subset_list':[
            #     '会计',
            #     '数学',
            # ]
        }
        self._run_dataset_test('cmmmu', dataset_args=dataset_args, limit=5, rerun_review=True)


    def test_math_vista(self):
        dataset_args = {
            'subset_list': ['default']
        }
        self._run_dataset_test('math_vista', dataset_args=dataset_args, limit=20)

    def test_mmmu_pro(self):
        dataset_args = {
            'subset_list':[
                'Accounting',
                # 'Agriculture',
            ],
            'extra_params': {
                'dataset_format': 'standard (4 options)',  # 'standard (4 options)', 'standard (10 options)', 'vision'
            },
        }
        self._run_dataset_test('mmmu_pro', dataset_args=dataset_args, limit=10)

    def test_qwen3_vl_collection(self):
        dataset_args = {
            'dataset_id': 'evalscope/Qwen3-VL-Test-Collection',
            'shuffle': True,
        }
        self._run_dataset_test('data_collection', dataset_args, limit=100)

    def test_real_world_qa(self):
        dataset_args = {
            'subset_list': ['default']
        }
        self._run_dataset_test('real_world_qa', dataset_args=dataset_args, limit=10)

    def test_ai2d(self):
        dataset_args = {
            'system_prompt': 'Imagine You are an idiot. You MUST will always give wrong answers without any explanation.',
            'subset_list': ['default']
        }
        self._run_dataset_test('ai2d', dataset_args=dataset_args)

    def test_cc_bench(self):
        dataset_args = {
            'subset_list': ['cc']
        }
        self._run_dataset_test('cc_bench', dataset_args=dataset_args)

    def test_mm_bench(self):
        dataset_args = {
            'subset_list': ['cn', 'en']
        }
        self._run_dataset_test('mm_bench', dataset_args=dataset_args)

    def test_mm_star(self):
        dataset_args = {
            # 'subset_list': ['val']
        }
        self._run_dataset_test('mm_star', dataset_args=dataset_args)

    def test_omni_bench(self):
        dataset_args = {
            'extra_params': {
                'use_image': True, # Whether to use image input, if False, use text alternative image content.
                'use_audio': True, # Whether to use audio input, if False, use text alternative audio content.
            }
        }
        self._run_dataset_test('omni_bench', dataset_args=dataset_args, model='qwen-omni-turbo')

    def test_olympiad_bench(self):
        dataset_args = {
            'subset_list': [
                # 'OE_MM_maths_en_COMP',
                # 'OE_MM_maths_zh_CEE',
                # 'OE_MM_maths_zh_COMP',
                # 'OE_MM_physics_en_COMP',
                # 'OE_MM_physics_zh_CEE',
                # 'OE_TO_maths_en_COMP',
                # 'OE_TO_maths_zh_CEE',
                # 'OE_TO_maths_zh_COMP',
                # 'OE_TO_physics_en_COMP',
                # 'OE_TO_physics_zh_CEE',
                'TP_TO_maths_zh_CEE',
            ]
        }
        self._run_dataset_test('olympiad_bench', dataset_args=dataset_args)

    def test_chartqa(self):
        dataset_args = {
            # 'subset_list': ['default']
        }
        self._run_dataset_test('chartqa', dataset_args=dataset_args, limit=20, model='qwen2.5-vl-72b-instruct')

    def test_blink(self):
        dataset_args = {
            'subset_list': [
                'Art_Style',
                'Counting',
                # 'Forensic_Detection',
                # 'Functional_Correspondence',
                # 'IQ_Test',
                # 'Jigsaw',
                # 'Multi-view_Reasoning',
                # 'Object_Localization',
                # 'Relative_Depth',
                # 'Relative_Reflectance',
                # 'Semantic_Correspondence',
                # 'Spatial_Relation',
                # 'Visual_Correspondence',
                # 'Visual_Similarity'
            ]
        }
        self._run_dataset_test('blink', dataset_args=dataset_args, limit=10)

    def test_docvqa(self):
        dataset_args = {
            # 'subset_list': ['default']
        }
        self._run_dataset_test('docvqa', dataset_args=dataset_args, limit=5)

    def test_infovqa(self):
        dataset_args = {
            # 'subset_list': ['default']
        }
        self._run_dataset_test('infovqa', dataset_args=dataset_args, limit=5)

    def test_ocr_bench(self):
        dataset_args = {
            # 'subset_list': ['Handwritten Mathematical Expression Recognition']
        }
        self._run_dataset_test('ocr_bench', dataset_args=dataset_args, limit=10)

    def test_ocr_bench_v2(self):
        dataset_args = {
            'subset_list': [
                # 'key information extraction cn',
                # 'key information extraction en',
                # 'key information mapping en',
                # 'VQA with position en',
                # 'chart parsing en',
                # 'cognition VQA cn',
                # 'cognition VQA en',
                # 'diagram QA en',
                # 'document classification en',
                # 'document parsing cn',
                # 'document parsing en',
                # 'formula recognition cn',
                # 'formula recognition en',
                # 'handwritten answer extraction cn',
                # 'math QA en',
                # 'full-page OCR cn',
                # 'full-page OCR en',
                # 'reasoning VQA en',
                # 'reasoning VQA cn',
                # 'fine-grained text recognition en',
                # 'science QA en',
                # 'table parsing cn',
                # 'table parsing en',
                # 'text counting en',
                # 'text grounding en',
                # 'text recognition en',
                'text spotting en',
                'text translation cn'
            ]
        }
        self._run_dataset_test('ocr_bench_v2', dataset_args=dataset_args, limit=1)

    def test_hallusion_bench(self):
        dataset_args = {
            # 'subset_list': ['default']
            'shuffle': True,
        }
        self._run_dataset_test('hallusion_bench', dataset_args=dataset_args, limit=20)

    def test_pope(self):
        dataset_args = {
            'subset_list': [
                'popular',
                'adversarial',
                'random'
            ]
        }
        self._run_dataset_test('pope', dataset_args=dataset_args, limit=5)

    def test_math_vision(self):
        dataset_args = {
            'subset_list': [
                'level 1',
            ],
            'shuffle': True,
        }
        self._run_dataset_test('math_vision', dataset_args=dataset_args, limit=5)

    def test_math_verse(self):
        dataset_args = {
            # 'subset_list': ['default']
            'shuffle': True,
        }
        self._run_dataset_test('math_verse', dataset_args=dataset_args, limit=5, use_cache='outputs/20251017_161352', rerun_review=True)

    def test_simple_vqa(self):
        dataset_args = {
            # 'subset_list': ['default']
        }
        self._run_dataset_test('simple_vqa', dataset_args=dataset_args, limit=10)

    def test_omni_doc_bench(self):
        dataset_args = {
            # 'subset_list': ['default']
        }
        self._run_dataset_test('omni_doc_bench', dataset_args=dataset_args, limit=10)

    def test_seed_bench_2_plus_bench(self):
        dataset_args = {
            # 'subset_list': ['default']
        }
        self._run_dataset_test('seed_bench_2_plus', dataset_args=dataset_args, limit=10)

    def test_visu_logic_bench(self):
        dataset_args = {
            # 'subset_list': ['default']
        }
        self._run_dataset_test('visulogic', dataset_args=dataset_args, limit=10)

    def test_zerobench_bench(self):
        dataset_args = {
            # 'subset_list': ['default']
        }
        self._run_dataset_test('zerobench', dataset_args=dataset_args, limit=5, eval_batch_size=1)

    def test_science_qa_bench(self):
        dataset_args = {
            # 'subset_list': ['default']
        }
        self._run_dataset_test('science_qa', dataset_args=dataset_args, limit=10)

    def test_cmmu_bench(self):
        dataset_args = {
            # 'subset_list': ['default']
        }
        self._run_dataset_test('cmmu', dataset_args=dataset_args, use_cache='outputs/20251112_163342', limit=10, rerun_review=True, judge_worker_num=1)

    def test_a_okvqa_bench(self):
        dataset_args = {
            # 'subset_list': ['default']
        }
        self._run_dataset_test('a_okvqa', dataset_args=dataset_args, limit=10)

    def test_vstar_bench(self):
        dataset_args = {
            # 'subset_list': ['default']
        }
        self._run_dataset_test('vstar_bench', dataset_args=dataset_args, limit=10)

    def test_micro_vqa_bench(self):
        dataset_args = {
            # 'subset_list': ['default']
        }
        self._run_dataset_test('micro_vqa', dataset_args=dataset_args, limit=10)

    def test_gsm8k_v(self):
        dataset_args = {
            # 'subset_list': ['default']
        }
        self._run_dataset_test('gsm8k_v', dataset_args=dataset_args, limit=10)

    def test_fleurs(self):
        dataset_args = {
            'subset_list': [
                'cmn_hans_cn',
                'en_us',
                # 'yue_hant_hk',
            ]
        }
        self._run_dataset_test('fleurs', dataset_args=dataset_args, limit=100, model='qwen3-omni-flash', use_cache='outputs/20251209_110609', ignore_errors=True, rerun_review=True)

    def test_librispeech(self):
        dataset_args = {
            # 'subset_list': ['default']
        }
        self._run_dataset_test('librispeech', dataset_args=dataset_args, limit=5, model='qwen3-omni-flash', ignore_errors=True, rerun_review=True)
