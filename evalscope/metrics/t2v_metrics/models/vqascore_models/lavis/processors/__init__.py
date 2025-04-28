"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from ..common.registry import registry
from .base_processor import BaseProcessor
from .blip_processors import (Blip2ImageTrainProcessor, BlipCaptionProcessor, BlipImageEvalProcessor,
                              BlipImageTrainProcessor)

__all__ = [
    'BaseProcessor',
    # BLIP
    'BlipImageTrainProcessor',
    'Blip2ImageTrainProcessor',
    'BlipImageEvalProcessor',
    'BlipCaptionProcessor',
]


def load_processor(name, cfg=None):
    """
    Example

    >>> processor = load_processor("alpro_video_train", cfg=None)
    """
    processor = registry.get_processor_class(name).from_config(cfg)

    return processor
