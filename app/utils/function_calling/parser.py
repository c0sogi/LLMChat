import json
from inspect import signature
from re import compile
from types import UnionType
from typing import (
    Annotated,
    Any,
    Callable,
    Iterable,
    Type,
    Union,
    get_args,
    get_origin,
)

from app.models.completion_models import (
    FunctionCallParsed,
    FunctionCallUnparsed,
)
from app.models.function_calling.base import (
    FunctionCall,
    FunctionCallParameter,
    JsonTypes,
)


def _get_type_and_optional(t) -> tuple[Type, bool]:
    """Returns the type and whether it's an Optional type."""
    # Optional[str] is equivalent to Union[str, None], so check if it's a Union type.
    if get_origin(t) in (UnionType, Union):
        args = get_args(t)
        # If there's a None type in the Union, it's an Optional type.
        optional = type(None) in args
        # Return the first argument that isn't None.
        first_arg = next(arg for arg in args if arg is not type(None))
        return first_arg, optional
    else:
        # If it's not a Union type, it's not an Optional type.
        return t, False


def parse_function_call_name_and_arguments(
    function_call_unparsed: FunctionCallUnparsed,
) -> FunctionCallParsed:
    """
    Parse function call response from API into a FunctionCallParsed object.
    This is a helper method to identify what function will be called and what
    arguments will be passed to it."""

    if "name" not in function_call_unparsed:
        raise ValueError("Function call name is required.")

    function_call_parsed: FunctionCallParsed = FunctionCallParsed(
        name=function_call_unparsed["name"]
    )
    try:
        function_call_parsed["arguments"] = json.loads(
            function_call_unparsed["arguments"]
        )
    except json.JSONDecodeError:
        pass
    return function_call_parsed


def parse_function_call_from_function(func: Callable) -> FunctionCall:
    """
    Parse a function into a FunctionCall object.
    FunctionCall objects are used to represent the specification of a function
    """
    json_types = get_args(JsonTypes)
    function_call_params: list[FunctionCallParameter] = []
    required: list[str] = []
    for name, param in signature(func).parameters.items():
        annotation = param.annotation
        description: str = ""
        enum: list[Any] = []

        if get_origin(annotation) is Annotated:
            # If the annotation is an Annotated type,
            # we need to parse the metadata
            _param_args = get_args(param.annotation)
            _param_type = _param_args[0]

            for metadata in _param_args[1:]:
                if isinstance(metadata, str):
                    # If the metadata is a string, it's the description
                    description += metadata
                elif isinstance(metadata, Iterable):
                    # If the metadata is an iterable, it's the enum
                    enum.extend(metadata)

        else:
            _param_type = annotation
        param_type, optional = _get_type_and_optional(_param_type)
        if not optional:
            required.append(name)
        if param_type not in json_types:
            continue
        function_call_params.append(
            FunctionCallParameter(
                name=name,
                type=param_type,
                description=description or None,
                enum=enum or None,
            )
        )
    line_break_pattern = compile(r"\n\s*")
    return FunctionCall(
        name=func.__name__,
        description=line_break_pattern.sub(" ", func.__doc__)
        if func.__doc__
        else None,
        parameters=function_call_params,
        required=required or None,
    )
