import json
from inspect import signature
from re import compile
from typing import (
    Annotated,
    Any,
    Callable,
    Iterable,
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
from app.utils.types import get_type_and_optional


def make_function_call_parsed_from_dict(
    unparsed_dict: dict[str, JsonTypes] | FunctionCallParsed | FunctionCallUnparsed,
) -> FunctionCallParsed:
    """
    Parse function call response from API into a FunctionCallParsed object.
    This is a helper method to identify what function will be called and what
    arguments will be passed to it."""

    if "name" not in unparsed_dict:
        raise ValueError("Function call name is required.")

    function_call_parsed: FunctionCallParsed = FunctionCallParsed(
        name=str(unparsed_dict["name"])
    )
    arguments = unparsed_dict.get("arguments", {})
    if isinstance(arguments, dict):
        function_call_parsed["arguments"] = arguments
        return function_call_parsed
    else:
        try:
            function_call_parsed["arguments"] = json.loads(str(arguments))
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
        param_type, optional = get_type_and_optional(_param_type)
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
        description=line_break_pattern.sub(" ", func.__doc__) if func.__doc__ else None,
        parameters=function_call_params,
        required=required or None,
    )
