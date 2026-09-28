"""Exercise vendored LAVIS configuration and actual processors without model downloads."""

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

OmegaConf = pytest.importorskip('omegaconf').OmegaConf
ROOT = Path(__file__).resolve().parents[2] / 'evalscope/metrics/vision/t2v_metrics/models/vqascore_models/lavis'


def load_lavis(monkeypatch: Any) -> Any:
    # Avoid the package's model registry imports: this test validates config and
    # processors, and does not initialize or download any model weights.
    for name,path in [('_math_lavis',ROOT),('_math_lavis.common',ROOT/'common'),
                      ('_math_lavis.processors',ROOT/'processors')]:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules,name,package)

    def load(name: Any,path: Any) -> Any:
        spec = importlib.util.spec_from_file_location(name,path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules,name,module)
        spec.loader.exec_module(module)
        return module

    base = load('_math_lavis.processors.base_processor',ROOT/'processors/base_processor.py').BaseProcessor
    sys.modules['_math_lavis.processors'].BaseProcessor = base
    registry = load('_math_lavis.common.registry',ROOT/'common/registry.py').registry
    config = load('_math_lavis.common.config',ROOT/'common/config.py').Config
    return registry, config


def test_lavis_configuration_merge_override_interpolation(tmp_path: Any,monkeypatch: Any) -> None:
    registry,Config = load_lavis(monkeypatch)
    class ConfigOnlyModel:
        @staticmethod
        def default_config_path(model_type: Any) -> Any:
            assert model_type == 'pretrain'
            return str(ROOT/'configs/models/blip2/blip2_pretrain.yaml')

    registry.mapping['model_name_mapping']['config_only'] = ConfigOnlyModel
    path = tmp_path/'user.yaml'
    OmegaConf.save(OmegaConf.create({'model':{'arch':'config_only','model_type':'pretrain','image_size':256},
                                   'run':{'seed':42,'seed_alias':'${run.seed}'},'datasets':{}}),path)
    results = []
    for options in [['model.image_size=336','run.seed=7'],['model.image_size','336','run.seed','7']]:
        config = Config(SimpleNamespace(cfg_path=str(path),options=options))
        assert config.model_cfg.image_size == 336
        assert config.run_cfg.seed == config.run_cfg.seed_alias == 7
        assert json.loads(config._convert_node_to_json(config.model_cfg))['image_size'] == 336
        results.append(config.to_dict())
    assert results[0] == results[1]
    for source in ROOT.rglob('*.yaml'):
        assert OmegaConf.to_container(OmegaConf.load(source),resolve=True) is not None


def test_lavis_processor_initialization(monkeypatch: Any) -> None:
    pytest.importorskip('torchvision')
    from PIL import Image

    registry,_ = load_lavis(monkeypatch)
    name = '_math_lavis.processors.blip_processors'
    spec = importlib.util.spec_from_file_location(name,ROOT/'processors/blip_processors.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,name,module)
    spec.loader.exec_module(module)
    caption = registry.get_processor_class('blip_caption').from_config(
        OmegaConf.create({'prompt':'Q: ','max_words':2}))
    assert caption('HELLO world again!') == 'Q: hello world'
    image = registry.get_processor_class('blip_image_eval').from_config(OmegaConf.create({'image_size':32}))
    assert tuple(image(Image.new('RGB',(64,64))).shape) == (3,32,32)
