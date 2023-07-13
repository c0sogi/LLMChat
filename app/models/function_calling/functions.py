from typing import Annotated, Callable, Optional

from app.common.lotties import Lotties
from app.utils.chat.buffer import BufferedUserContext
from app.utils.chat.managers.websocket import SendToWebsocket
from app.utils.chat.tokens import make_truncated_text
from app.utils.function_calling.parser import parse_function_call_from_function

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
        """Get the FunctionCall object for the given function."""
        if function.__name__ not in cls.function_calls:
            cls.function_calls[
                function.__name__
            ] = parse_function_call_from_function(function)
        return cls.function_calls[function.__name__]

    def get_function(cls, function_name: str) -> Callable:
        """Get the function for the given function name."""
        return getattr(cls, function_name)


class FunctionCalls(metaclass=FunctionCallsMetaClass):
    """Class that contains all functions that can be called by the user.
    This class is used to parse all functions in the FunctionCalls class into
    FunctionCall objects. FunctionCall objects are used to represent the
    specification of a function and will be used for `function_calling`."""

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
    ) -> Optional[str]:
        """Perform web search for a user's question."""

        from app.utils.function_calling.callbacks.full_browsing import (
            URL_PATTERN,
            full_web_browsing_callback,
        )

        # Get user's question, or use the last user message if the user did not
        # provide a question.
        # Also, get the links in the user's question if exist.
        # The links will be used to browse the web if AI chooses to do so.
        user_query = (
            buffer.current_user_message_histories[-1].content
            if buffer.current_user_message_histories
            else buffer.last_user_message.removeprefix("/browse")
            if buffer.last_user_message
            else query_to_search
        )
        user_provided_links = URL_PATTERN.findall(user_query)

        # Notify frontend that we are browsing the web.
        await SendToWebsocket.message(
            websocket=buffer.websocket,
            msg=Lotties.SEARCH_WEB.format(
                "### Browsing web\n---\n{query_to_search}".format(
                    query_to_search=query_to_search.replace("```", "'''")
                )
            ),
            chat_room_id=buffer.current_chat_room_id,
            finish=False,
        )

        # Return the browsing result.
        browsing_result: Optional[str] = await full_web_browsing_callback(
            buffer=buffer,
            query_to_search=query_to_search,
            user_provided_links=user_provided_links,
            finish=True,
            wait_next_query=True,
        )
        return (
            make_truncated_text(
                user_chat_context=buffer.current_user_chat_context,
                text=browsing_result,
                with_n_user_messages=1,
            )
            if browsing_result
            else None
        )

    @staticmethod
    async def vectorstore_search(
        query_to_search: Annotated[
            str,
            "Hypothetical answer to facilitate searching in the Vector database.",
        ],
        buffer: BufferedUserContext,
    ) -> Optional[str]:
        """Perform vector similarity-based search for user's question."""
        from app.utils.function_calling.callbacks.vectorstore_search import (
            vectorstore_search_callback,
        )

        # Notify frontend that we are browsing the web.
        await SendToWebsocket.message(
            websocket=buffer.websocket,
            msg=Lotties.SEARCH_DOC.format(
                "### Searching vectorstore\n---\n{query_to_search}".format(
                    query_to_search=query_to_search.replace("```", "'''")
                )
            ),
            chat_room_id=buffer.current_chat_room_id,
            finish=False,
        )

        # Return the browsing result.
        vectorstore_search_result: Optional[
            str
        ] = await vectorstore_search_callback(
            buffer=buffer,
            query_to_search=query_to_search,
            finish=True,
            wait_next_query=True,
        )
        return (
            make_truncated_text(
                user_chat_context=buffer.current_user_chat_context,
                text=vectorstore_search_result,
                with_n_user_messages=1,
            )
            if vectorstore_search_result
            else None
        )
