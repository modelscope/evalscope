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

    force_redownload: bool = False
    """Whether to force redownload the dataset from remote source."""

    review_timeout: Optional[float] = None
    """ Timeout for review in seconds."""

    extra_params: Dict = field(default_factory=dict)
    """Additional parameters for the benchmark.
        The structure is:
        {
          "param": {
             "type": "int",
             "description": "...",
             "value": 10,
             "choices": [5, 10]        # optional for enum type checks
          },
          ...
        }
    """

    sandbox_config: Optional[Dict[str, Any]] = field(default_factory=dict)
    """Configuration for sandboxed code execution environments. """

    def __post_init__(self):
        """Validate fields after initialization."""
        if self.few_shot_num < 0:
            raise ValueError('few_shot_num must be >= 0')

    def _is_spec_entry(self, entry: Any) -> bool:
        """Return True if entry is a spec dict (new structured form)."""
        return isinstance(entry, dict) and ('description' in entry) and ('type' in entry) and ('value' in entry)

    def _extract_value(self, entry: Any) -> Any:
        """Extract runtime value from a spec entry or return raw value."""
        if self._is_spec_entry(entry):
            value = entry.get('value')
            if entry.get('choices') and value not in entry['choices']:
                raise ValueError(f'Value {value} not in choices {entry["choices"]} in {self.name} extra_params.')
            return value
        return entry

    def get_extra_params(self) -> Dict:
        """Get plain extra param dict, only param name and value."""
        param_dict = {}
        for key, value in self.extra_params.items():
            param_dict[key] = self._extract_value(value)
        return param_dict

    def to_dict(self) -> dict:
        """Convert to dictionary, maintaining backward compatibility."""
        return asdict(self)

    def to_string_dict(self) -> dict:
        """Convert to string dictionary, excluding data_adapter."""
        cur_dict = copy.deepcopy(asdict(self))
        if 'data_adapter' in cur_dict:
            del cur_dict['data_adapter']

        cur_dict['extra_params'] = self.get_extra_params()
        return cur_dict

    def _update(self, args: dict):
        """Update instance with provided arguments, maintaining backward compatibility."""
        args = copy.deepcopy(args)

        if args.get('local_path'):
            self.dataset_id = args['local_path']
            del args['local_path']

        if args.get('filters'):
            self._update_filters(args['filters'])
            del args['filters']

        if args.get('extra_params'):
            self._update_extra_params(args['extra_params'])
            del args['extra_params']

        # Update fields with validation
        for key, value in args.items():
            if hasattr(self, key):
                setattr(self, key, value)  # Validate few_shot_num if it's being updated
                if key == 'few_shot_num' and value < 0:
                    raise ValueError('few_shot_num must be >= 0')

    def _update_filters(self, new_filters: dict):
        if self.filters is None:
            self.filters = OrderedDict()
        new_filters = OrderedDict(new_filters)
        # insert filters at the beginning
        self.filters = OrderedDict(list(new_filters.items()) + list(self.filters.items()))

    def _update_extra_params(self, new_params: dict):
        for key, value in new_params.items():
            if key in self.extra_params:
                # Update only the 'value' field if it's a spec entry
                if self._is_spec_entry(self.extra_params[key]):
                    self.extra_params[key]['value'] = value
                else:
                    self.extra_params[key] = value
            else:
                raise KeyError(f'Extra param {key} not found in benchmark {self.name} extra_params.')
