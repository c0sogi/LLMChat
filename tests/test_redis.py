from pytest import CaptureFixture
from app.utils.chatgpt.chatgpt_context_manager import chatgpt_cache_manager
from app.viewmodels.gpt_models import MessageHistory, UserGptContext


def test_redis(capfd: CaptureFixture[str]):
    # set random context
    user_id: str = ""
    role: str = "user"
    message: str = "test message"
    default_context: UserGptContext = UserGptContext.construct_default(user_id=user_id)

    # reset context
    chatgpt_cache_manager.reset_context(user_id=user_id)

    # create context
    chatgpt_cache_manager.create_context(
        user_id="",
        user_gpt_context=default_context,
    )
    # read context
    context: UserGptContext = chatgpt_cache_manager.read_context(user_id="")
    assert context == default_context

    # add new message to redis
    new_message: MessageHistory = MessageHistory(
        role=role, content=message, is_user=True, tokens=len(context.tokenize(message))
    )
    chatgpt_cache_manager.append_message_history(user_id=user_id, role=role, message_history=new_message)

    # read message from redis
    message_histories: list[MessageHistory] = chatgpt_cache_manager.get_message_history(user_id=user_id, role=role)
    assert message_histories == [new_message]
