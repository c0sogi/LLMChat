from typing import Annotated, Callable, Optional

from app.common.constants import QueryTemplates
from app.models.chat_models import ChatRoles, ResponseType
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.tokens import make_formatted_query
from app.utils.function_calling.parser import parse_function_call_from_function
from app.utils.logger import ApiLogger

from .base import FunctionCall


class FunctionCallsMetaClass(type):
    """Metaclass for FunctionCalls class.
    This metaclass is used to parse all functions in the FunctionCalls class
    into FunctionCall objects."""

    function_calls: dict[str, FunctionCall] = {}

    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        for func in cls.__dict__.values():
            if callable(func) and not func.__name__.startswith("_"):
                print(
                    "- Parsing function for function calling:", func.__name__
                )
                function_call = parse_function_call_from_function(func)
                cls.function_calls[function_call.name] = function_call

    def get_function_call(cls, function: Callable) -> FunctionCall:
        if function.__name__ not in cls.function_calls:
            cls.function_calls[
                function.__name__
            ] = parse_function_call_from_function(function)
        return cls.function_calls[function.__name__]


class FunctionCalls(metaclass=FunctionCallsMetaClass):
    """Class that contains all functions that can be called by the user.
    This class is used to parse all functions in the FunctionCalls class into
    FunctionCall objects. FunctionCall objects are used to represent the
    specification of a function and will be used for `function_calling`."""

    @classmethod
    def _get_function(cls, function_name: str) -> Callable:
        print(f"- DEBUG: Getting function {function_name}")
        return getattr(cls, function_name)

    @staticmethod
    def control_browser(
        action: Annotated[
            str,
            (
                "Your action to take. Select `finish_browsing` if you can answer th"
                "e user's question or there's no relevant information. Select `clic"
                "k_link` only if you need to click on a link to gather more informa"
                "tion."
            ),
            ["finish_browsing", "click_link"],
        ],
        link_to_click: Annotated[
            str,
            "The link to click on if you selected `click_link` as your action.",
        ],
    ):
        """Control web browser to answer the user's question."""
        # Note: This function is not used in the demo.
        pass

    @staticmethod
    def control_web_page(
        action: Annotated[
            str,
            (
                "Whether to scroll down, go back, or pick the result. Select `scrol"
                "l_down` if you must scroll down to read more information. Select `g"
                "o_back` if you need to go back to the previous page to read other "
                "information. Select `pick` only if you can provide a satisfactory "
                "answer to the user from the given information."
            ),
            ["scroll_down", "go_back", "pick"],
        ],
        relevance_score: Annotated[
            int,
            (
                "A score that indicates how helpful the given context is in answeri"
                "ng the user's question. If the information is very relevant and su"
                "fficient to answer the user's question, give it a score of 10; if "
                "the information is very irrelevant, give it a score of 0."
            ),
            [score for score in range(11)],
        ],
    ):
        """Control the web page to read more information or stop reading. You
        have to evaluate the relevance of the information you read and de
        cide whether to scroll down, go back, or pick."""
        pass

    @staticmethod
    async def web_search(
        query_to_search: Annotated[
            str,
            "A generalized query to return sufficiently relevant results when s"
            "earching the web.",
        ],
        buffer: BufferedUserContext,
    ) -> tuple[None, ResponseType]:
        """Perform web search for a user's question."""

        from app.utils.chat.managers.message import MessageManager
        from app.utils.chat.messages.handler import MessageHandler
        from app.utils.function_calling.callbacks.full_browsing import (
            URL_PATTERN,
            full_web_browsing_callback,
        )

        user_query = (
            buffer.last_user_message.removeprefix("/browse ")
            if buffer.last_user_message
            else query_to_search
        )
        if not user_query:
            user_provided_links = []
        else:
            user_provided_links = URL_PATTERN.findall(user_query)

        browsing_result: Optional[str] = await full_web_browsing_callback(
            buffer=buffer,
            query_to_search=query_to_search,
            user_provided_links=user_provided_links,
            finish=True,
            wait_next_query=True,
        )
        if browsing_result:
            query_to_send: str = make_formatted_query(
                user_chat_context=buffer.current_user_chat_context,
                question=user_query,
                context=browsing_result,
                query_template=QueryTemplates.CONTEXT_QUESTION__WEB_BROWSING,
            )
        else:
            query_to_send: str = user_query

        ApiLogger("||browse||").info(query_to_send)
        await MessageHandler.user(
            msg=query_to_send,
            buffer=buffer,
            use_tight_token_limit=False,
        )
        try:
            await MessageHandler.ai(buffer=buffer)
        finally:
            if browsing_result is not None:
                await MessageManager.set_message_history_safely(
                    user_chat_context=buffer.current_user_chat_context,
                    role=ChatRoles.USER,
                    index=-1,
                    new_content=user_query,
                )
        return None, ResponseType.DO_NOTHING

    @staticmethod
    async def vectorstore_search(
        query_to_search: Annotated[
            str,
            "Hypothetical answer to facilitate searching in the Vector database.",
        ],
        **kwargs,
    ) -> Optional[str]:
        """Perform vector similarity-based search for user's question."""
        from app.utils.function_calling.callbacks.vectorstore_search import (
            vectorstore_search_callback,
        )

        buffer = kwargs.get("buffer")
        assert buffer is not None, "Buffer must be provided."
        return await vectorstore_search_callback(
            buffer=buffer,
            query=query_to_search,
            finish=True,
            wait_next_query=True,
        )
