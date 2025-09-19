import copy
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

from evalscope.constants import OutputType

if TYPE_CHECKING:
    from evalscope.api.benchmark import DataAdapter


@dataclass
class BenchmarkMeta:
    """Metadata for a benchmark, including dataset and model configurations."""

    name: str
    """ Unique name of the benchmark."""

    dataset_id: str
    """ Dataset id on modelscope or path to local dataset."""

    data_adapter: Optional[Type['DataAdapter']] = None
    """ Data adapter class for the benchmark."""

    output_types: List[str] = field(default_factory=lambda: [OutputType.GENERATION])
    """ List of output types supported by the benchmark."""

    subset_list: List[str] = field(default_factory=lambda: ['default'])
    """ List of subsets available for the benchmark."""

    default_subset: str = 'default'
    """ Default subset to use for the benchmark."""

    few_shot_num: int = 0
    """ Number of few-shot examples to use."""

    few_shot_random: bool = False
    """ Whether to use random few-shot examples."""

    train_split: Optional[str] = None
    """ Training split to use for the benchmark."""

    eval_split: Optional[str] = None
    """ Evaluation split to use for the benchmark."""

    prompt_template: Optional[str] = None
    """ Prompt template to use for the benchmark."""

    few_shot_prompt_template: Optional[str] = None
    """ Few-shot prompt template to use for the benchmark."""

    system_prompt: Optional[str] = None
    """ System prompt to use for the benchmark."""

    query_template: Optional[str] = None
    """ Query template to use for the benchmark."""

    pretty_name: Optional[str] = None
    """ Human-readable name for the benchmark."""

    description: Optional[str] = None
    """ Description of the benchmark."""

    tags: List[str] = field(default_factory=list)
    """ Tags associated with the benchmark."""

    filters: Optional[OrderedDict] = None
    """ Filters to apply to the dataset on model output."""

    metric_list: List[Union[str, Dict[str, Any]]] = field(default_factory=list)
    """ List of metrics to evaluate the benchmark."""

    aggregation: str = 'mean'
    """ Aggregation function for the metrics. Default is 'mean'. Can be 'mean', 'pass@<k>' or a custom function name."""

    shuffle: bool = False
    """Whether to shuffle the dataset before evaluation."""

    shuffle_choices: bool = False
    """Whether to shuffle the choices in multiple-choice datasets."""

    review_timeout: Optional[float] = None
    """ Timeout for review in seconds."""

    extra_params: Dict = field(default_factory=dict)
    """ Additional parameters for the benchmark."""

    def __post_init__(self):
        """Validate fields after initialization."""
        if self.few_shot_num < 0:
            raise ValueError('few_shot_num must be >= 0')

    def _update(self, args: dict):
        """Update instance with provided arguments, maintaining backward compatibility."""
        args = copy.deepcopy(args)

        if args.get('local_path'):
            self.dataset_id = args['local_path']
            del args['local_path']

        if args.get('filters'):
            if self.filters is None:
                self.filters = OrderedDict()
            new_filters = OrderedDict(args['filters'])
            # insert filters at the beginning
            self.filters = OrderedDict(list(new_filters.items()) + list(self.filters.items()))
            del args['filters']
        # Update fields with validation
        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)  # Validate few_shot_num if it's being updated
                if key == 'few_shot_num' and value < 0:
                    raise ValueError('few_shot_num must be >= 0')

    def to_dict(self) -> dict:
        """Convert to dictionary, maintaining backward compatibility."""
        return asdict(self)

    def to_string_dict(self) -> dict:
        """Convert to string dictionary, excluding data_adapter."""
        cur_dict = copy.deepcopy(asdict(self))
        if 'data_adapter' in cur_dict:
            del cur_dict['data_adapter']
        return cur_dict
