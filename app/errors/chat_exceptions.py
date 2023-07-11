from typing import Any


class ChatException(Exception):  # Base exception for chat
    def __init__(self, *, msg: str | None = None) -> None:
        self.msg = msg
        super().__init__()


class ChatConnectionException(ChatException):
    def __init__(self, *, msg: str | None = None) -> None:
        self.msg = msg
        super().__init__(msg=msg)


class ChatLengthException(ChatException):
    def __init__(self, *, msg: str | None = None) -> None:
        self.msg = msg
        super().__init__(msg=msg)


class ChatContentFilterException(ChatException):
    def __init__(self, *, msg: str | None = None) -> None:
        self.msg = msg
        super().__init__(msg=msg)


class ChatTooMuchTokenException(ChatException):
    def __init__(self, *, msg: str | None = None) -> None:
        self.msg = msg
        super().__init__(msg=msg)


class ChatTextGenerationException(ChatException):
    def __init__(self, *, msg: str | None = None) -> None:
        self.msg = msg
        super().__init__(msg=msg)


class ChatOtherException(ChatException):
    def __init__(self, *, msg: str | None = None) -> None:
        self.msg = msg
        super().__init__(msg=msg)


class ChatModelNotImplementedException(ChatException):
    def __init__(self, *, msg: str | None = None) -> None:
        self.msg = msg
        super().__init__(msg=msg)


class ChatBreakException(ChatException):
    def __init__(self, *, msg: str | None = None) -> None:
        self.msg = msg
        super().__init__(msg=msg)


class ChatContinueException(ChatException):
    def __init__(self, *, msg: str | None = None) -> None:
        self.msg = msg
        super().__init__(msg=msg)


class ChatInterruptedException(ChatException):
    def __init__(self, *, msg: str | None = None) -> None:
        self.msg = msg
        super().__init__(msg=msg)


class ChatFunctionCallException(ChatException):
    """Raised when function is called."""

    def __init__(self, *, func_name: str, func_kwargs: dict[str, Any]) -> None:
        self.func_name = func_name
        self.func_kwargs = func_kwargs
        super().__init__(msg=f"Function {func_name}({func_kwargs}) is called.")
