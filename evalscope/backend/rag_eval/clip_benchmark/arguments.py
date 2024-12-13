from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Arguments:
    # fmt: off
    """
    A dataclass to store and manage the arguments for the model configuration and data processing.
    """
    """
    For CLIP model support, you can use the following fields:
        model_name: str
        revision: str = "master"
        hub: str = "modelscope"

    For API VLM model support, you can use the following fields, (image caption only):
        model_name="gpt-4o-mini"
        api_base: str = "",
        api_key: Optional[str] = None
        prompt: str = None
    """
    models: List[Dict] = field(default_factory=dict)  # List of paths to the pre-trained models or model identifiers
    dataset_name: List[str] = field(default_factory=list)  # List of dataset names to be used
    data_dir: str = None  # Root directory where the datasets are stored
    split: str = 'test'  # Split of the dataset to be used (e.g., 'train', 'validation', 'test')
    task: str = None
    batch_size: int = 128  # Batch size for data loading
    num_workers: int = 1  # Number of workers for data loading
    verbose: bool = True  # Flag to enable verbose logging
    output_dir: str = 'outputs'  # Directory where the outputs (e.g., predictions, logs) will be saved
    cache_dir: str = 'cache'  # Directory where the dataset cache will be stored
    skip_existing: bool = False  # Flag to skip processing if outputs already exist
    limit: int = None  # Limit the number of samples to be processed
