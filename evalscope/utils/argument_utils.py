import json
from argparse import Namespace
from inspect import signature

from evalscope.utils.io_utils import json_to_dict, yaml_to_dict


class BaseArgument:
    """
    BaseArgument is a base class designed to facilitate the creation and manipulation
    of argument classes in the evalscope framework. It provides utility methods for
    instantiating objects from various data formats and converting objects back into
    dictionary representations.
    """

    @classmethod
    def from_dict(cls, d: dict):
        """Instantiate the class from a dictionary."""
        return cls(**d)

    @classmethod
    def from_json(cls, json_file: str):
        """Instantiate the class from a JSON file."""
        return cls.from_dict(json_to_dict(json_file))

    @classmethod
    def from_yaml(cls, yaml_file: str):
        """Instantiate the class from a YAML file."""
        return cls.from_dict(yaml_to_dict(yaml_file))

    @classmethod
    def from_args(cls, args: Namespace):
        """
        Instantiate the class from an argparse.Namespace object.
        Filters out None values and removes 'func' if present.
        """
        args_dict = {k: v for k, v in vars(args).items() if v is not None}

        if 'func' in args_dict:
            del args_dict['func']  # Note: compat CLI arguments

        return cls.from_dict(args_dict)

    def to_dict(self):
        """Convert the instance to a dictionary."""
        result = self.__dict__.copy()
        return result

    def __str__(self):
        """Return a JSON-formatted string representation of the instance."""
        return json.dumps(self.to_dict(), indent=4, default=str, ensure_ascii=False)


def parse_int_or_float(num):
    number = float(num)
    if number.is_integer():
        return int(number)
    return number


def get_supported_params(func):
    """Get the supported parameters of a function."""
    sig = signature(func)
    return set(sig.parameters.keys())
