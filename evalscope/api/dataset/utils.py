import json
from tqdm import tqdm
from typing import Any, Callable, Dict, Iterable, List, Optional, Union, cast

from .dataset import Dataset, FieldSpec, Sample


def record_to_sample_fn(sample_fields: Union[FieldSpec, Callable, None] = None, ) -> Callable:
    if sample_fields is None:
        sample_fields = FieldSpec()

    if isinstance(sample_fields, FieldSpec):

        def record_to_sample(record: dict) -> Sample:
            # collect metadata if specified
            metadata: Optional[Dict[str, Any]] = None
            if sample_fields.metadata:
                if isinstance(sample_fields.metadata, list):
                    metadata = {}
                    for name in sample_fields.metadata:
                        metadata[name] = record.get(name)

            elif 'metadata' in record:
                metadata_field = record.get('metadata')
                if isinstance(metadata_field, str):
                    metadata = json.loads(metadata_field)
                elif isinstance(metadata_field, dict):
                    metadata = metadata_field
                else:
                    raise ValueError(f"Unexpected type for 'metadata' field: {type(metadata_field)}")

            # return sample
            return Sample(
                input=read_input(record.get(sample_fields.input)),
                target=read_target(record.get(sample_fields.target)),
                choices=read_choices(record.get(sample_fields.choices)),
                id=record.get(sample_fields.id, None),
                metadata=metadata,
                sandbox=read_sandbox(record.get(sample_fields.sandbox)),
                files=read_files(record.get(sample_fields.files)),
                setup=read_setup(record.get(sample_fields.setup)),
            )

        return record_to_sample

    else:
        return sample_fields


def data_to_samples(data: Iterable[dict], data_to_sample: Callable) -> List[Sample]:
    samples: List[Sample] = []
    for record in tqdm(data, desc='Processing records'):
        record_samples = as_sample_list(data_to_sample(record=record))
        samples.extend(record_samples)
    return samples


def as_sample_list(samples: Union[Sample, List[Sample]]) -> List[Sample]:
    if isinstance(samples, list):
        return samples
    else:
        return [samples]


def read_input(input_val: Optional[Any]) -> str:
    if not input_val:
        raise ValueError('No input in dataset')
    return str(input_val)


def read_target(obj: Optional[Any]) -> Union[str, List[str]]:
    if obj is not None:
        return [str(item) for item in obj] if isinstance(obj, list) else str(obj)
    else:
        return ''


def read_choices(obj: Optional[Any]) -> Optional[List[str]]:
    if obj is not None:
        if isinstance(obj, list):
            return [str(choice) for choice in obj]
        elif isinstance(obj, str):
            choices = obj.split(',')
            if len(choices) == 1:
                choices = obj.split()
            return [choice.strip() for choice in choices]
        else:
            return [str(obj)]
    else:
        return None


def read_setup(setup: Optional[Any]) -> Optional[str]:
    if setup is not None:
        return str(setup)
    else:
        return None


def read_sandbox(sandbox: Optional[Any]) -> Optional[str]:
    if sandbox is not None:
        if isinstance(sandbox, str):
            return sandbox
        elif isinstance(sandbox, dict):
            return json.dumps(sandbox)
        else:
            raise ValueError(f"Unexpected type for 'sandbox' field: {type(sandbox)}")
    else:
        return None


def read_files(files: Optional[Any]) -> Optional[Dict[str, str]]:
    if files is not None:
        if isinstance(files, str):
            files = json.loads(files)
        if isinstance(files, dict):
            if all(isinstance(v, str) for v in files.values()):
                return cast(Dict[str, str], files)

        # didn't find the right type
        raise ValueError(f"Unexpected type for 'files' field: {type(files)}")
    else:
        return None


def shuffle_choices_if_requested(dataset: Dataset, shuffle_choices: Optional[Union[bool, int]]) -> None:
    """
    Shuffle the choices in the dataset if requested.

    The `shuffle_choices` parameter passed to `json_dataset`, `csv_dataset`,
    and `hf_dataset` can be a boolean, an integer, or `None` (default).
    If it is a boolean, it will shuffle the choices if the value is `True`,
    and do nothing if it is `False`.
    If it is an integer, it will shuffle the choices using the integer as the seed.
    """
    # Note that `isinstance(x, int)` returns True if x is True or False,
    # so we need to check for both explicitly
    if shuffle_choices is True:
        dataset.shuffle_choices()
    elif shuffle_choices is False:
        pass
    elif isinstance(shuffle_choices, int):
        dataset.shuffle_choices(seed=shuffle_choices)
