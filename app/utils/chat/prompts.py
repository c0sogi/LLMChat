# flake8: noqa

from itertools import zip_longest
from typing import TYPE_CHECKING, Union


from app.utils.logger import api_logger
from app.viewmodels.base_models import SendInitToWebsocket, SendToStream

if TYPE_CHECKING:
    from app.models.chat_models import UserChatContext

CONTEXT_QUESTION_TMPL_QUERY1 = (
    "Context information is below. \n"
    "---------------------\n"
    "{context}"
    "\n---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {question}\n"
)

CONTEXT_QUESTION_TMPL_QUERY2 = (
    "Context information is below. \n"
    "---------------------\n"
    "{context}"
    "\n---------------------\n"
    "answer the question: {question}\n"
)

USER_AI_TMPL_CHAT1 = (
    "The following is a friendly conversation between a {user} and an {ai}. "
    "The {ai} is talkative and provides lots of specific details from its context. "
    "If the {ai} does not know the answer to a question, it truthfully says it does not know.\n\n"
    "Current conversation:\n\n"
)

ROLE_CONTENT_TMPL_CHAT1 = "### {role}: {content}\n"


def message_history_organizer(
    user_chat_context: "UserChatContext",
    send_to_stream: bool = True,
    return_as_string: bool = False,
    chat_turn_prompt: str = ROLE_CONTENT_TMPL_CHAT1,
) -> Union[list[dict], str]:  # organize message history for openai api
    message_histories: list[dict[str, str]] = []
    if send_to_stream:
        for system_history in user_chat_context.system_message_histories:
            message_histories.append(SendToStream.from_orm(system_history).dict())  # append system message history
    for user_message_history, ai_message_history in zip_longest(
        user_chat_context.user_message_histories,
        user_chat_context.ai_message_histories,
    ):
        message_histories.append(
            SendToStream.from_orm(user_message_history).dict()
            if send_to_stream
            else SendInitToWebsocket.from_orm(user_message_history).dict()
        ) if user_message_history is not None else ...  # append user message history
        message_histories.append(
            SendToStream.from_orm(ai_message_history).dict()
            if send_to_stream
            else SendInitToWebsocket.from_orm(ai_message_history).dict()
        ) if ai_message_history is not None else ...  # append gpt message history
    if return_as_string:
        user_role: str = user_chat_context.user_chat_profile.user_role
        ai_role: str = user_chat_context.user_chat_profile.ai_role
        system_role: str = user_chat_context.user_chat_profile.system_role
        prefix: str = ""
        if hasattr(user_chat_context.llm_model.value, "description"):
            if user_chat_context.llm_model.value.description is not None:  # type: ignore
                prefix: str = user_chat_context.llm_model.value.description.format(  # type: ignore
                    user=user_role.upper(),
                    USER=user_role.upper(),
                    ai=ai_role.upper(),
                    AI=ai_role.upper(),
                    system=system_role.upper(),
                    SYSTEM=system_role.upper(),
                )

        for message_history in message_histories:
            if message_history["role"] == system_role:
                prefix += chat_turn_prompt.format(role=system_role.upper(), content=message_history["content"].strip())
            elif message_history["role"] == user_role:
                prefix += chat_turn_prompt.format(role=user_role.upper(), content=message_history["content"].strip())
            elif message_history["role"] == ai_role:
                prefix += chat_turn_prompt.format(role=ai_role.upper(), content=message_history["content"].strip())
            else:
                api_logger.error(f"Invalid message history: {message_history}")
                raise Exception("Invalid message history")
        prefix += chat_turn_prompt.format(role=ai_role.upper(), content="").strip() + " "
        return prefix
    else:
        return message_histories  # return message histories to be used in openai api
