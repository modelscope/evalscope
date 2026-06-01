from typing import Any, Dict, List, Optional

from evalscope.utils.argument_utils import BaseArgument


class ClipBenchmarkEvalConfig(BaseArgument):
    """Configuration for CLIP Benchmark evaluation.

    For CLIP model support, you can use the following fields in models:
        model_name: str
        revision: str = "master"
        hub: str = "modelscope"

    For API VLM model support, you can use the following fields (image caption only):
        model_name: str = "gpt-4o-mini"
        api_base: str = ""
        api_key: Optional[str] = None
        prompt: str = None
    """

    models: List[Dict[str, Any]] = []
    dataset_name: List[str] = []
    data_dir: Optional[str] = None
    split: str = 'test'
    task: Optional[str] = None
    batch_size: int = 128
    num_workers: int = 1
    verbose: bool = True
    output_dir: str = 'outputs'
    cache_dir: str = 'cache'
    skip_existing: bool = False
    limit: Optional[int] = None


# Backward-compatible alias
Arguments = ClipBenchmarkEvalConfig
