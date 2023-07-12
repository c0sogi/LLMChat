from inspect import Parameter, signature
from types import NoneType, UnionType
from typing import (
    Any,
    Callable,
    Type,
    Union,
    get_args,
    get_origin,
)


def is_optional(t: Type) -> bool:
    """Returns whether the type is an Optional type.
    This is useful when Type can be Union and you want to know if it's an Optional type.
    """
    # Optional[str] is equivalent to Union[str, None], so check if it's a Union type.
    if get_origin(t) in (UnionType, Union):
        # If there's a None type in the Union, it's an Optional type.
        return type(None) in get_args(t)
    else:
        # If it's not a Union type, it's not an Optional type.
        return False


def get_type(t: Type) -> Type:
    """Returns the type of the type.
    If the type is not a Union type, it returns the type itself,
    but if the type is a Union type, it returns the first argument that isn't None.
    """
    #  Check if it's a Union type.
    if get_origin(t) in (UnionType, Union):
        args: tuple[Type, ...] = get_args(t)
        # Return the first argument that isn't None.
        first_arg = next(arg for arg in args if arg is not NoneType)
        return first_arg
    else:
        # If it's not a Union type, it returns the type itself.
        return t


def get_type_and_optional(t: Type) -> tuple[Type, bool]:
    """Returns the type and whether it's an Optional type.
    This is useful when Type can be Union and you want to know if it's an Optional type.
    """
    # Optional[str] is equivalent to Union[str, None], so check if it's a Union type.
    if get_origin(t) in (UnionType, Union):
        args: tuple[Type, ...] = get_args(t)
        # If there's a None type in the Union, it's an Optional type.
        optional = type(None) in args
        # Return the first argument that isn't None.
        first_arg = next(arg for arg in args if arg is not NoneType)
        return first_arg, optional
    else:
        # If it's not a Union type, it's not an Optional type.
        return t, False


def arguments_provider(
    func: Callable[..., Any],
    available_args: list[Any],
    available_annotated: dict[Any, Any],
    available_kwargs: dict[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    """Provides arguments to a function based on the function's signature."""

    args_to_pass: list[Any] = []
    kwargs_to_pass: dict[str, Any] = {}

    for param in signature(func).parameters.values():
        if param.kind == Parameter.VAR_POSITIONAL:
            # For *args, we just pass all available args and clear the list
            args_to_pass.extend(available_args)
            available_args.clear()

        elif param.kind == Parameter.VAR_KEYWORD:
            # For **kwargs, we just pass all available kwargs
            kwargs_to_pass.update(available_kwargs)

        elif param.kind == Parameter.KEYWORD_ONLY:
            # For (*, param),
            if param.annotation in available_annotated:
                # if param is available annotated, we pass the annotated value
                kwargs_to_pass[param.name] = available_annotated[
                    param.annotation
                ]
            elif param.name in available_kwargs:
                # if param is in available kwargs,
                if param.annotation is Parameter.empty:
                    # if param is not annotated, we pass the value
                    kwargs_to_pass[param.name] = available_kwargs[param.name]
                else:
                    # if param is annotated, we convert the value to annotated type
                    t = get_type(param.annotation)
                    kwargs_to_pass[param.name] = t(
                        available_kwargs[param.name]
                    )
            elif param.annotation is Parameter.empty:
                # if param is not annotated,
                if param.default is Parameter.empty:
                    # if there's no value to pass, we pass None
                    kwargs_to_pass[param.name] = None
                else:
                    # if there's no provided value to pass, but has default value,
                    # we pass the default value
                    kwargs_to_pass[param.name] = param.default
            else:
                # if param is annotated,
                if param.default is Parameter.empty:
                    # if there's no value to pass,
                    if is_optional(param.annotation):
                        # if the annotated type is optional, we pass None
                        kwargs_to_pass[param.name] = None
                    else:
                        raise ValueError(
                            f"Required argument {param.name} is missing"
                        )
                else:
                    # if there's no provided value to pass, but has default value,
                    # we try to pass the default value, and convert it to annotated type
                    t = get_type(param.annotation)
                    kwargs_to_pass[param.name] = t(param.default)

        elif param.kind == Parameter.POSITIONAL_OR_KEYWORD:
            # For (param),
            if param.annotation in available_annotated:
                # if param is available annotated, we pass the annotated value
                kwargs_to_pass[param.name] = available_annotated[
                    param.annotation
                ]
            elif param.name in available_kwargs:
                # if param is in available kwargs,
                if param.annotation is Parameter.empty:
                    # if param is not annotated, we pass the value
                    kwargs_to_pass[param.name] = available_kwargs[param.name]
                else:
                    # if param is annotated, we convert the value to annotated type
                    t = get_type(param.annotation)
                    kwargs_to_pass[param.name] = t(
                        available_kwargs[param.name]
                    )
            elif param.annotation is Parameter.empty:
                # if param is not annotated,
                if available_args:
                    # if there's available args, we pass the first arg
                    kwargs_to_pass[param.name] = available_args.pop(0)
                if param.default is Parameter.empty:
                    # if param is not available and has no default value,
                    # we pass None
                    kwargs_to_pass[param.name] = None
                else:
                    # if there's no available args, but has default value,
                    # we pass the default value
                    kwargs_to_pass[param.name] = param.default
            else:
                # if param is annotated,
                if available_args:
                    # if there's available args, we try to pass the first arg,
                    # and convert it to annotated type
                    t = get_type(param.annotation)
                    kwargs_to_pass[param.name] = t(available_args.pop(0))
                elif param.default is Parameter.empty:
                    # if there's no value to pass,
                    if is_optional(param.annotation):
                        # if the annotated type is optional, we pass None
                        kwargs_to_pass[param.name] = None
                    else:
                        raise ValueError(
                            f"Required argument {param.name} is missing"
                        )
                else:
                    # if there's no provided value to pass, but has default value,
                    # we try to pass the default value
                    kwargs_to_pass[param.name] = param.default

        elif param.kind == Parameter.POSITIONAL_ONLY:
            # For (/, param),
            if param.annotation in available_annotated:
                # if param is available annotated, we pass the annotated value
                kwargs_to_pass[param.name] = available_annotated[
                    param.annotation
                ]
            elif available_args:
                # if there's available args,
                if param.annotation is Parameter.empty:
                    # if param is not annotated, we pass the first arg
                    args_to_pass.append(available_args.pop(0))
                else:
                    # if param is annotated,
                    t = get_type(param.annotation)
                    if t is str:
                        # if the annotated type is str, we pass all available args
                        # as a single string, and clear the list
                        args_to_pass.append(" ".join(available_args))
                        available_args.clear()
                    else:
                        # if the annotated type is not str, we pass the first arg
                        # and convert it to annotated type
                        args_to_pass.append(t(available_args.pop(0)))
            elif param.annotation is Parameter.empty:
                # if there's no available args and param is not annotated,
                if param.default is Parameter.empty:
                    # if param has no default value, we pass None
                    # we pass None
                    args_to_pass.append(None)
                else:
                    # if param has default value, we pass the default value
                    args_to_pass.append(param.default)
            else:
                # if there's no available args and param is annotated,
                if param.default is Parameter.empty:
                    # if there's no value to pass and param has no default value,
                    if is_optional(param.annotation):
                        # if the annotated type is optional, we pass None
                        args_to_pass.append(None)
                    else:
                        raise ValueError(
                            f"Required argument {param.name} is missing"
                        )
                else:
                    # if param has default value, we try to pass the default value
                    args_to_pass.append(param.default)
    print(
        f"- DEBUG: Calling command: {func.__name__} with {len(args_to_pass)} args "
        f"and {[k for k in kwargs_to_pass.keys()]} kwargs",
        flush=True,
    )
    return args_to_pass, kwargs_to_pass
