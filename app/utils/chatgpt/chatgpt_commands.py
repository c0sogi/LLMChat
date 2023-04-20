from app.utils.chatgpt.chatgpt_message_manager import MessageManager
from app.viewmodels.gpt_models import MessageHistory, UserGptContext
from app.utils.chatgpt.chatgpt_context_manager import chatgpt_cache_manager


class ChatGptCommands:  # commands for chat gpt
    @staticmethod
    def not_existing_callback(*args, user_gpt_context: UserGptContext) -> str:  # callback for not existing command
        return f"{user_gpt_context.user_gpt_profile.user_id}님, 죄송합니다. 현재 그런 명령어는 지원하지 않습니다."

    @staticmethod
    def clear(*args, user_gpt_context: UserGptContext) -> str:  # clear user and gpt message histories
        # user_id: str = user_gpt_context.user_gpt_profile.user_id
        n_user_tokens: int = user_gpt_context.user_message_tokens
        n_gpt_tokens: int = user_gpt_context.gpt_message_tokens
        n_system_tokens: int = user_gpt_context.system_message_tokens
        for role in ("user", "gpt", "system"):
            getattr(user_gpt_context, f"{role}_message_histories").clear()
            setattr(user_gpt_context, f"{role}_message_tokens", 0)
            chatgpt_cache_manager.delete_message_history(user_id=user_gpt_context.user_gpt_profile.user_id, role=role)
        response: str = f"총 {n_user_tokens}개의 사용자 토큰, {n_gpt_tokens}개의 GPT 토큰, {n_system_tokens}개의 시스템 토큰이 삭제되었습니다."
        return response  # return success message

    @staticmethod
    def test(*args, user_gpt_context: UserGptContext) -> str:  # test command showing user_gpt_context
        return str(user_gpt_context)

    @staticmethod
    def reset(*args, user_gpt_context: UserGptContext) -> str:  # reset user_gpt_context
        user_gpt_context.reset()
        if chatgpt_cache_manager.reset_context(user_id=user_gpt_context.user_gpt_profile.user_id):  # if reset success
            return "컨텍스트를 리셋했습니다."
        else:
            return "컨텍스트를 리셋하지 못했습니다."  # return fail message

    @staticmethod
    def system(*args, user_gpt_context: UserGptContext) -> str:  # add system message
        if len(args) < 1:  # if no args
            return "/system SYSTEM_MESSAGE와 같은 형식으로 입력해야 합니다."  # return fail message
        system_message: str = " ".join(args)
        MessageManager.add_message_history_safely(
            user_gpt_context=user_gpt_context,
            content=system_message,
            role=user_gpt_context.user_gpt_profile.system_role,
        )
        return f"시스템 메시지를 `{system_message}`로 추가하였습니다!"  # return success message

    @staticmethod
    def settemperature(*args, user_gpt_context: UserGptContext) -> str:  # set temperature of gpt
        if len(args) < 1:  # if no args
            return "/settemperature 0.5와 같은 형식으로 입력해야 합니다."  # return fail message
        try:
            now_temperature: float = float(args[0])  # try to convert args[0] to float
            assert 0 <= now_temperature <= 1  # assert temperature is between 0 and 1
        except ValueError:  # if args[0] is not float
            return "temperature는 float 타입이어야 합니다."  # return fail message
        except AssertionError:  # if temperature is not between 0 and 1
            return "temperature는 0 이상 1 이하여야 합니다."  # return fail message
        else:
            previous_temperature: str = user_gpt_context.user_gpt_profile.temperature
            user_gpt_context.user_gpt_profile.temperature = now_temperature
            chatgpt_cache_manager.update_profile_and_model(user_gpt_context)  # update user_gpt_context
            return f"temperature 값을 {previous_temperature}에서 {now_temperature}로 바꿨어요."  # return success message

    @classmethod
    def settemp(cls, *args, user_gpt_context: UserGptContext) -> str:  # alias for settemperature
        return cls.settemperature(*args, user_gpt_context=user_gpt_context)

    @staticmethod
    def settopp(*args, user_gpt_context: UserGptContext) -> str:  # set top_p of gpt
        if len(args) < 1:  # if no args
            return "/settopp 1.0와 같은 형식으로 입력해야 합니다."  # return fail message
        try:
            now_top_p: float = float(args[0])  # try to convert args[0] to float
            assert 0 <= now_top_p <= 1  # assert top_p is between 0 and 1
        except ValueError:  # if args[0] is not float
            return "top_p float 타입이어야 합니다."  # return fail message
        except AssertionError:  # if top_p is not between 0 and 1
            return "top_p는 0 이상 1 이하여야 합니다."  # return fail message
        else:
            previous_top_p: str = user_gpt_context.user_gpt_profile.top_p
            user_gpt_context.user_gpt_profile.top_p = now_top_p  # set top_p
            chatgpt_cache_manager.update_profile_and_model(user_gpt_context)  # update user_gpt_context
            return f"top_p 값을 {previous_top_p}에서 {now_top_p}로 바꿨어요."  # return success message

    @staticmethod
    def poplastmessage(
        *args, user_gpt_context: UserGptContext
    ) -> str:  # pop last message (user or system or gpt), format: /poplastmessage [user|system|gpt]
        if len(args) < 1:  # if no args
            return "/poplastmessage user|system|gpt와 같은 형식으로 입력해야 합니다."  # return fail message
        # if args contains arg equal to --silent, return no message
        is_silent: bool = "--silent" in args
        if args[0] not in (
            "user",
            "system",
            "gpt",
        ):  # if args[0] is not user or system or gpt
            return "user, system, gpt 중 하나를 입력해야 합니다." if not is_silent else ""  # return fail message
        last_message_history: MessageHistory | None = MessageManager.rpop_message_history_safely(
            user_gpt_context=user_gpt_context, role=args[0]
        )  # pop last message history
        if last_message_history is None:  # if last_message_history is None
            return f"{args[0]} 메시지가 없어서 삭제할 수 없습니다." if not is_silent else ""  # return fail message
        return (
            f"{args[0]}의 메시지인 `{last_message_history.content}`을 삭제하였습니다!" if not is_silent else ""
        )  # return success message

    @staticmethod
    def retry(*args, user_gpt_context: UserGptContext) -> None | str:
        # retry last gpt message, format: /retry
        if len(user_gpt_context.user_message_histories) < 1 or len(user_gpt_context.gpt_message_histories) < 1:
            return "메시지가 없어서 다시 할 수 없습니다."
        MessageManager.rpop_message_history_safely(user_gpt_context=user_gpt_context, role="gpt")
        return None

    @staticmethod
    def ping(*args, user_gpt_context: UserGptContext) -> str:
        return "pong"
