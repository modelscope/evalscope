"""Tests for the unified ``--dataset-args`` framework.

Covers:
- Per-dataset ``args_schema`` selection via the dataset registry.
- ``extra='forbid'`` fail-fast rejection of unknown keys (at the plugin/schema
  layer, which owns the schema).
- Backward-compatible folding of the deprecated ``--multi-turn-args`` into
  the raw ``--dataset-args`` dict on ``Arguments`` (config layer, no plugin dep).
"""

import pytest

from evalscope.perf.arguments import Arguments
from evalscope.perf.multi_turn_args import MultiTurnArgs
from evalscope.perf.plugin.datasets.dataset_args import BaseDatasetArgs, MultiTurnDatasetArgs, TextDatasetArgs
from evalscope.perf.plugin.datasets.openqa import OpenqaDatasetPlugin
from evalscope.perf.plugin.registry import DatasetRegistry


def _args(**kwargs) -> Arguments:
    return Arguments(model='test-model', url='http://localhost:8080/v1/chat/completions', **kwargs)


def _resolve(args: Arguments):
    """Resolve raw ``dataset_args`` against the selected plugin's schema.

    Mirrors the plugin-layer resolution (``self.args_schema(**raw)``) so tests
    can validate the mapping without constructing a full plugin.
    """
    schema = DatasetRegistry.get_class(args.dataset).args_schema
    return schema(**(args.dataset_args or {}))


# ---------------------------------------------------------------------------
# Schema selection
# ---------------------------------------------------------------------------


class TestSchemaSelection:

    def test_text_datasets_use_text_schema(self):
        for name in ('openqa', 'longalpaca', 'line_by_line', 'share_gpt_zh', 'share_gpt_en'):
            assert DatasetRegistry.get_class(name).args_schema is TextDatasetArgs

    def test_multi_turn_datasets_use_multi_turn_schema(self):
        for name in ('swe_smith', 'random_multi_turn', 'share_gpt_zh_multi_turn', 'share_gpt_en_multi_turn'):
            assert DatasetRegistry.get_class(name).args_schema is MultiTurnDatasetArgs

    def test_default_schema_is_base(self):
        assert DatasetRegistry.get_class('random').args_schema is BaseDatasetArgs

    def test_multi_turn_schema_subclasses_multi_turn_args(self):
        assert issubclass(MultiTurnDatasetArgs, MultiTurnArgs)


# ---------------------------------------------------------------------------
# Schema resolution + fail-fast validation (plugin/schema layer)
# ---------------------------------------------------------------------------


class TestResolveDatasetArgs:

    def test_none_resolves_to_empty_schema(self):
        resolved = _resolve(_args(dataset='openqa'))
        assert isinstance(resolved, TextDatasetArgs)
        assert resolved.target_input_len is None
        assert resolved.input_len_mode == 'cap'

    def test_valid_keys_parsed(self):
        args = _args(dataset='openqa', dataset_args={'target_input_len': 256, 'input_len_mode': 'drop'})
        resolved = _resolve(args)
        assert resolved.target_input_len == 256
        assert resolved.input_len_mode == 'drop'

    def test_unknown_key_fails_fast(self):
        with pytest.raises(Exception):
            _resolve(_args(dataset='openqa', dataset_args={'not_a_real_key': 1}))

    def test_json_string_is_parsed(self):
        args = _args(dataset='openqa', dataset_args='{"target_input_len": 64}')
        assert _resolve(args).target_input_len == 64

    def test_invalid_input_len_mode_rejected(self):
        with pytest.raises(Exception):
            _resolve(_args(dataset='openqa', dataset_args={'target_input_len': 8, 'input_len_mode': 'nope'}))

    def test_non_positive_target_input_len_rejected(self):
        with pytest.raises(Exception):
            _resolve(_args(dataset='openqa', dataset_args={'target_input_len': 0}))

    def test_target_input_len_requires_tokenizer(self):
        # The tokenizer requirement is enforced at plugin construction (base layer).
        args = _args(dataset='openqa', dataset_args={'target_input_len': 128})
        with pytest.raises(ValueError, match='requires a tokenizer'):
            OpenqaDatasetPlugin(args)


# ---------------------------------------------------------------------------
# Backward-compat: --multi-turn-args folding (config layer, raw dict only)
# ---------------------------------------------------------------------------


class TestMultiTurnArgsFolding:

    def test_folds_into_dataset_args(self):
        args = _args(dataset='swe_smith', multi_turn_args='{"first_turn_length": 100}')
        assert args.dataset_args['first_turn_length'] == 100
        assert _resolve(args).first_turn_length == 100

    def test_only_user_set_keys_are_folded(self):
        args = _args(dataset='swe_smith', multi_turn_args='{"first_turn_length": 100}')
        # subsequent_turn_length was not set by the user -> not injected
        assert 'subsequent_turn_length' not in args.dataset_args

    def test_dataset_args_wins_on_conflict(self):
        args = _args(
            dataset='swe_smith',
            dataset_args={'first_turn_length': 999},
            multi_turn_args='{"first_turn_length": 100}',
        )
        assert _resolve(args).first_turn_length == 999

    def test_num_workers_promoted_to_top_level(self):
        args = _args(dataset='swe_smith', multi_turn_args='{"num_workers": 3}')
        assert args.num_workers == 3

    def test_num_workers_in_dataset_args_promoted_and_popped(self):
        # num_workers passed directly in --dataset-args must be promoted to top-level
        # and removed, so non-multi-turn schemas (extra='forbid') do not reject it.
        args = _args(dataset='openqa', dataset_args='{"num_workers": 4, "target_input_len": 100}')
        assert args.num_workers == 4
        assert 'num_workers' not in args.dataset_args
        # remaining keys still validate against the dataset schema
        assert _resolve(args).target_input_len == 100

    def test_no_multi_turn_args_leaves_dataset_args_none(self):
        args = _args(dataset='swe_smith')
        assert args.dataset_args is None
