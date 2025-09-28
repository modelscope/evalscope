import types
import typing
from copy import deepcopy
from dataclasses import is_dataclass
from datetime import date, datetime, time
from enum import EnumMeta
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

JSONType = Literal['string', 'integer', 'number', 'boolean', 'array', 'object', 'null']
"""Valid types within JSON schema."""


class JSONSchema(BaseModel):
    """JSON Schema for type."""

    type: Optional[JSONType] = Field(default=None)
    """JSON type of tool parameter."""

    format: Optional[str] = Field(default=None)
    """Format of the parameter (e.g. date-time)."""

    description: Optional[str] = Field(default=None)
    """Parameter description."""

    default: Any = Field(default=None)
    """Default value for parameter."""

    enum: Optional[List[Any]] = Field(default=None)
    """Valid values for enum parameters."""

    items: Optional['JSONSchema'] = Field(default=None)
    """Valid type for array parameters."""

    properties: Optional[Dict[str, 'JSONSchema']] = Field(default=None)
    """Valid fields for object parametrs."""

    additionalProperties: Optional[Union['JSONSchema', bool]] = Field(default=None)
    """Are additional properties allowed?"""

    anyOf: Optional[List['JSONSchema']] = Field(default=None)
    """Valid types for union parameters."""

    required: Optional[List[str]] = Field(default=None)
    """Required fields for object parameters."""

    @model_validator(mode='before')
    def convert_type_before_validation(cls, values):
        values = deepcopy(values)

        def recursive_convert_type(obj):
            if isinstance(obj, dict):
                # Convert 'type' field if it's a string
                if 'type' in obj and isinstance(obj['type'], str):
                    try:
                        obj['type'] = python_type_to_json_type(obj['type'])
                    except ValueError:
                        # If conversion fails, leave it as is
                        pass
                # Recursively process nested structures
                for k, v in obj.items():
                    obj[k] = recursive_convert_type(v)
            elif isinstance(obj, list):
                return [recursive_convert_type(item) for item in obj]
            return obj

        return recursive_convert_type(values)


def json_schema(t: Type[Any]) -> JSONSchema:
    """Provide a JSON Schema for the specified type.

    Schemas can be automatically inferred for a wide variety of
    Python class types including Pydantic BaseModel, dataclasses,
    and typed dicts.

    Args:
        t: Python type

    Returns:
        JSON Schema for type.
    """
    origin = get_origin(t)
    args = get_args(t)

    if origin is None:
        if t is int:
            return JSONSchema(type='integer')
        elif t is float:
            return JSONSchema(type='number')
        elif t is str:
            return JSONSchema(type='string')
        elif t is bool:
            return JSONSchema(type='boolean')
        elif t is datetime:
            return JSONSchema(type='string', format='date-time')
        elif t is date:
            return JSONSchema(type='string', format='date')
        elif t is time:
            return JSONSchema(type='string', format='time')
        elif t is list or t is set:
            return JSONSchema(type='array', items=JSONSchema())
        elif t is dict:
            return JSONSchema(type='object', additionalProperties=JSONSchema())
        elif (is_dataclass(t) or is_typeddict(t) or (isinstance(t, type) and issubclass(t, BaseModel))):
            return cls_json_schema(t)
        elif isinstance(t, EnumMeta):
            return JSONSchema(enum=[item.value for item in t])
        elif t is type(None):
            return JSONSchema(type='null')
        else:
            return JSONSchema()
    elif (origin is list or origin is List or origin is tuple or origin is Tuple or origin is set or origin is Set):
        return JSONSchema(type='array', items=json_schema(args[0]) if args else JSONSchema())
    elif origin is dict or origin is Dict:
        return JSONSchema(
            type='object',
            additionalProperties=json_schema(args[1]) if len(args) > 1 else JSONSchema(),
        )
    elif origin is Union or origin is types.UnionType:
        return JSONSchema(anyOf=[json_schema(arg) for arg in args])
    elif origin is Optional:
        return JSONSchema(anyOf=[json_schema(arg) for arg in args] + [JSONSchema(type='null')])
    elif origin is typing.Literal:
        return JSONSchema(enum=list(args))

    return JSONSchema()  # Default case if we can't determine the type


def cls_json_schema(cls: Type[Any]) -> JSONSchema:
    properties: Dict[str, JSONSchema] = {}
    required: List[str] = []

    if is_dataclass(cls):
        fields = cls.__dataclass_fields__  # type: ignore
        for name, field in fields.items():
            properties[name] = json_schema(field.type)  # type: ignore
            if field.default == field.default_factory:
                required.append(name)
    elif isinstance(cls, type) and issubclass(cls, BaseModel):
        schema = cls.model_json_schema()
        schema = resolve_schema_references(schema)
        for name, prop in schema.get('properties', {}).items():
            properties[name] = JSONSchema(**prop)
        required = schema.get('required', [])
    elif is_typeddict(cls):
        annotations = get_type_hints(cls)
        for name, type_hint in annotations.items():
            properties[name] = json_schema(type_hint)
            if name in cls.__required_keys__:
                required.append(name)

    return JSONSchema(
        type='object',
        properties=properties,
        required=required if required else None,
        additionalProperties=False,
    )


def python_type_to_json_type(python_type: Optional[str]) -> JSONType:
    if python_type is not None and python_type in get_args(JSONType):
        return python_type
    if python_type == 'str':
        return 'string'
    elif python_type == 'int':
        return 'integer'
    elif python_type == 'float':
        return 'number'
    elif python_type == 'bool':
        return 'boolean'
    elif python_type == 'list':
        return 'array'
    elif python_type == 'dict':
        return 'object'
    elif python_type == 'None':
        return 'null'
    elif python_type is None:
        # treat 'unknown' as string as anything can be converted to string
        return 'string'
    else:
        raise ValueError(f'Unsupported type: {python_type} for Python to JSON conversion.')


def resolve_schema_references(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Resolves all $ref references in a JSON schema by inlining the definitions."""
    schema = deepcopy(schema)
    definitions = schema.pop('$defs', {})

    def _resolve_refs(obj: Any) -> Any:
        if isinstance(obj, dict):
            if '$ref' in obj and obj['$ref'].startswith('#/$defs/'):
                ref_key = obj['$ref'].split('/')[-1]
                if ref_key in definitions:
                    # Replace with a deep copy of the definition
                    resolved = deepcopy(definitions[ref_key])
                    # Process any nested references in the definition
                    resolved = _resolve_refs(resolved)

                    # Merge in the current object fields, which should take priority
                    # This means that if you have e.g.
                    # {"$ref": "#/$defs/SubType", "description": "subtype of type SubType"},
                    # and SubType resolves to
                    # {"description": "The SubType Class", "parameters": {"param1": {"type": "string"}}},
                    # the final result will be:
                    # {"description": "subtype of type SubType", "parameters": {"param1": {"type": "string"}}}
                    return resolved | {k: o for k, o in obj.items() if k != '$ref'}

            # Process all entries in the dictionary
            return {k: _resolve_refs(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_resolve_refs(item) for item in obj]
        else:
            return obj

    return cast(Dict[str, Any], _resolve_refs(schema))
