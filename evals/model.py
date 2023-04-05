# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Optional
from dataclasses import dataclass


@dataclass
class ModelMeta:
    """
    Meta information for a model.
    """

    model_name: str
    model_id: str

    version: Optional[str] = None
    description: Optional[str] = None
    organization: Optional[str] = None

    def __post_init__(self):
        if self.model_name is None or self.model_id is None:
            raise ValueError(f"Must specify a model name and model id")

        if self.version is None:
            self.version = "0.0.0"
        if self.description is None:
            self.description = ""
        if self.organization is None:
            self.organization = ""
