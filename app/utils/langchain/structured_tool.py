"""Base implementation for tools or skills."""
from __future__ import annotations

from inspect import signature, iscoroutinefunction
from typing import Any, Awaitable, Callable, Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool, create_schema_from_function
from pydantic import (
    BaseModel,
    Field,
)


class StructuredTool(BaseTool):
    """Tool that can operate on any number of inputs."""

    description: str = ""
    args_schema: Type[BaseModel] = Field(..., description="The tool schema.")
    """The input arguments' schema."""
    func: Optional[Callable[..., Any]] = None
    """The function to run when the tool is called."""
    coroutine: Optional[Callable[..., Awaitable[Any]]] = None
    """The asynchronous version of the function."""

    @property
    def args(self) -> dict:
        """The tool's input arguments."""
        return self.args_schema.schema()["properties"]

    def _run(
        self,
        *args: Any,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Any:
        """Use the tool."""
        if self.func:
            new_argument_supported = signature(self.func).parameters.get("callbacks")
            return (
                self.func(
                    *args,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **kwargs,
                )
                if new_argument_supported
                else self.func(*args, **kwargs)
            )
        raise NotImplementedError("Tool does not support sync")

    async def _arun(
        self,
        *args: Any,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> str:
        """Use the tool asynchronously."""
        if self.coroutine:
            new_argument_supported = signature(self.coroutine).parameters.get(
                "callbacks"
            )
            return (
                await self.coroutine(
                    *args,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **kwargs,
                )
                if new_argument_supported
                else await self.coroutine(*args, **kwargs)
            )
        return self._run(*args, **kwargs)

    @classmethod
    def from_function(
        cls,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        return_direct: bool = False,
        args_schema: Optional[Type[BaseModel]] = None,
        infer_schema: bool = True,
        **kwargs: Any,
    ) -> StructuredTool:
        name = name or func.__name__
        description = description or func.__doc__
        assert (
            description is not None
        ), "Function must have a docstring if description not provided."

        # Description example:
        # search_api(query: str) - Searches the API for the query.
        description = f"{name}{signature(func)} - {description.strip()}"
        _args_schema = args_schema
        if _args_schema is None and infer_schema:
            _args_schema = create_schema_from_function(f"{name}Schema", func)
        is_coroutine_function = iscoroutinefunction(func)
        return cls(
            name=name,
            func=func if not is_coroutine_function else None,
            coroutine=func if is_coroutine_function else None,
            args_schema=_args_schema,
            description=description,
            return_direct=return_direct,
            **kwargs,
        )
