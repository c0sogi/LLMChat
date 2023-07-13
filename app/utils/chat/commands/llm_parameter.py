from app.models.chat_models import UserChatContext, command_response
from app.utils.chat.managers.cache import CacheManager


class LLMParameterCommands:
    @staticmethod
    @command_response.send_message_and_stop
    async def temp(
        temp_to_change: float, user_chat_context: UserChatContext
    ) -> str:  # set temperature of ai
        """Set temperature of ai\n
        /temp <temp_to_change>"""
        try:
            assert (
                0 <= temp_to_change <= 2
            )  # assert temperature is between 0 and 2
        except AssertionError:  # if temperature is not between 0 and 2
            return "Temperature must be between 0 and 2"
        else:
            previous_temperature: str = str(
                user_chat_context.user_chat_profile.temperature
            )
            user_chat_context.user_chat_profile.temperature = temp_to_change
            await CacheManager.update_profile_and_model(
                user_chat_context
            )  # update user_chat_context
            return f"I've changed temperature from {previous_temperature} to {temp_to_change}."  # return success msg

    @staticmethod
    @command_response.send_message_and_stop
    async def topp(
        top_p_to_change: float, user_chat_context: UserChatContext
    ) -> str:  # set top_p of ai
        """Set top_p of ai\n
        /topp <top_p_to_change>"""
        try:
            assert 0 <= top_p_to_change <= 1  # assert top_p is between 0 and 1
        except AssertionError:  # if top_p is not between 0 and 1
            return "Top_p must be between 0 and 1."  # return fail message
        else:
            previous_top_p: str = str(
                user_chat_context.user_chat_profile.top_p
            )
            user_chat_context.user_chat_profile.top_p = (
                top_p_to_change  # set top_p
            )
            await CacheManager.update_profile_and_model(
                user_chat_context
            )  # update user_chat_context
            return f"I've changed top_p from {previous_top_p} to {top_p_to_change}."  # return success message
