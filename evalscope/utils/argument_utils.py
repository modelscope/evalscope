import json
from argparse import Namespace
from inspect import signature
from pydantic import BaseModel, ConfigDict, SecretStr
from typing import Optional, Union

from evalscope.utils.io_utils import json_to_dict, yaml_to_dict

SECRET_HEADER_KEYS = {'authorization', 'proxy-authorization', 'x-api-key', 'x-auth-token'}


class BaseArgument(BaseModel):
    """
    BaseArgument is a base class designed to facilitate the creation and manipulation
    of argument classes in the evalscope framework. It provides utility methods for
    instantiating objects from various data formats and converting objects back into
    dictionary representations.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=(),
        validate_default=True,
    )

    @classmethod
    def from_dict(cls, d: dict):
        """Instantiate the class from a dictionary."""
        return cls.model_validate(d)

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
        return self.model_dump()

    def __str__(self):
        """Return a JSON-formatted string representation of the instance."""
        return json.dumps(self.to_dict(), indent=4, default=str, ensure_ascii=False)


def parse_int_or_float(num):
    number = float(num)
    if number.is_integer():
        return int(number)
    return number


def get_secret_value(value: Optional[Union[str, SecretStr, dict, list]]) -> Optional[Union[str, dict, list]]:
    """Return the raw value for runtime use while keeping SecretStr masked for display."""
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    if isinstance(value, dict):
        return {key: get_secret_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [get_secret_value(item) for item in value]
    return value


def secretize_auth_headers(headers: Optional[dict]) -> Optional[dict]:
    """Wrap auth-style header values as SecretStr for display-safe serialization."""
    if not headers:
        return headers
    return {
        key: SecretStr(value) if str(key).lower() in SECRET_HEADER_KEYS and isinstance(value, str) else value
        for key, value in headers.items()
    }


def get_supported_params(func):
    """Get the supported parameters of a function."""
    sig = signature(func)
    return set(sig.parameters.keys())
