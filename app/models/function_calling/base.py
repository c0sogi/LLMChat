"""Helper classes for wrapping functions in OpenAI's API"""

from dataclasses import dataclass
from sys import version_info
from types import NoneType
from typing import (
    Any,
    Generic,
    Literal,
    Optional,
    Type,
    TypeVar,
    TypedDict,
    Union,
)

# If python version >= 3.11, use the built-in NotRequired type.
# Otherwise, import it from typing_extensi
if version_info >= (3, 11):
    from typing import NotRequired  # type: ignore
else:
    from typing_extensions import NotRequired

# The types that can be used in JSON
JsonTypes = Union[int, float, str, bool, dict, list, None]

ParamType = TypeVar("ParamType", bound=JsonTypes)
ReturnType = TypeVar("ReturnType")


class ParameterProperty(TypedDict):
    type: str
    description: NotRequired[str]
    enum: NotRequired[list[JsonTypes]]


class ParameterDefinition(TypedDict):
    type: Literal["object"]
    properties: dict[str, ParameterProperty]
    required: NotRequired[list[str]]


class FunctionProperty(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: NotRequired[ParameterDefinition]


@dataclass
class FunctionCallParameter(Generic[ParamType]):
    """A class for wrapping function parameters in OpenAI's API"""

    name: str
    type: Type[ParamType]
    description: Optional[str] = None
    enum: Optional[list[ParamType]] = None

    def to_dict(self) -> dict[str, ParameterProperty]:
        """Returns a dictionary representation of the parameter"""
        parameter_property: ParameterProperty = {
            "type": self._get_json_type(self.type)
        }  # type: ignore
        if self.description:
            parameter_property["description"] = self.description
        if self.enum:
            parameter_property["enum"] = self.enum  # type: ignore
        return {self.name: parameter_property}

    @staticmethod
    def _get_json_type(python_type: Type[JsonTypes]) -> str:
        """Returns the JSON type for a given python type"""
        if python_type is int:
            return "integer"
        elif python_type is float:
            return "number"
        elif python_type is str:
            return "string"
        elif python_type is bool:
            return "boolean"
        elif python_type is dict:
            return "object"
        elif python_type is list:
            return "array"
        elif python_type is NoneType or python_type is None:
            return "null"
        else:
            raise ValueError(
                f"Invalid type {python_type} for JSON. "
                f"Permitted types are {JsonTypes}"
            )


@dataclass
class FunctionCall:
    """A class for wrapping functions in OpenAI's API"""

    name: str
    description: Optional[str] = None
    parameters: Optional[list[FunctionCallParameter[Any]]] = None
    required: Optional[list[str]] = None

    def to_dict(self) -> FunctionProperty:
        """Returns a dictionary representation of the function"""
        function_property: FunctionProperty = FunctionProperty(name=self.name)  # type: ignore
        if self.description:
            function_property["description"] = self.description
        if self.parameters:
            function_property["parameters"] = {
                "type": "object",
                "properties": {
                    param.name: param.to_dict()[param.name]
                    for param in self.parameters
                },
                "required": [
                    param.name
                    for param in self.parameters
                    if param.name in (self.required or [])
                ],
            }
        return function_property


if __name__ == "__main__":

    def test_callback(test: str) -> int:
        return len(test)

    param_1: FunctionCallParameter[str] = FunctionCallParameter(
        name="param_1",
        type=str,
        description="This is a test1",
        enum=["a", "b", "c", "d"],
    )  # `str` is the type of the parameter

    param_2: FunctionCallParameter[bool] = FunctionCallParameter(
        name="param_2",
        type=bool,
        description="This is a test2",
        enum=[True, False],
    )  # `bool` is the type of the parameter

    func: FunctionCall = FunctionCall(
        name="test_function",
        description="This is a test function",
        parameters=[param_1, param_2],
        required=["test"],
    )  # There's no type for the function
