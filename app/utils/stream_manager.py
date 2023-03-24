from asyncio import sleep
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import AsyncGenerator
import json
import re
import httpx
from transformers import GPT2TokenizerFast
from fastapi import WebSocket
from app.errors.exceptions import Responses_500
from app.models import UserChatGptMessage
from app.utils.logger import logger


class GptException(Exception):  # Base exception for gpt
    def __init__(self, *, msg: str | None = None) -> None:
        self.msg = msg
        super().__init__()


class GptConnectionException(GptException):
    def __init__(self, *, msg: str | None = None) -> None:
        self.msg = msg
        super().__init__(msg=msg)


class GptLengthException(GptException):
    def __init__(self, *, msg: str | None = None) -> None:
        self.msg = msg
        super().__init__(msg=msg)


class GptContentFilterException(GptException):
    def __init__(self, *, msg: str | None = None) -> None:
        self.msg = msg
        super().__init__(msg=msg)


class GptTooMuchPromptTokenException(GptException):
    def __init__(self, *, msg: str | None = None) -> None:
        self.msg = msg
        super().__init__(msg=msg)


@dataclass
class GptModel:  # gpt model for openai api
    name: str = "gpt-3.5-turbo"  # model name for openai api
    api_url: str = "https://api.openai.com/v1/chat/completions"  # api url for openai
    max_total_tokens: int = 4096  # max total tokens for openai api
    max_tokens_per_request: int = 4096  # max tokens per request for openai api
    token_margin: int = 512  # token margin for openai api


@dataclass
class GptModels:  # gpt models for openai api
    gpt_3_5_turbo: GptModel = field(
        default_factory=lambda: GptModel(
            name="gpt-3.5-turbo",
            api_url="https://api.openai.com/v1/chat/completions",
            max_total_tokens=4096,
            max_tokens_per_request=4096,
            token_margin=512,
        )
    )
    gpt_4: GptModel = field(
        default_factory=lambda: GptModel(
            name="text-davinci-003",
            api_url="https://api.openai.com/v1/chat/completions",
            max_total_tokens=8192,
            max_tokens_per_request=4096,
            token_margin=512,
        )
    )


GPT_MODELS = GptModels()


@dataclass
class MessageHistory:  # message history for user and gpt
    role: str
    content: str
    tokens: int
    is_user: bool


@dataclass
class UserGptProfile:  # user gpt profile for user and gpt
    user_id: str
    user_role: str = "user"
    gpt_role: str = "assistant"
    temperature: float = 0.9
    top_p: float = 1.0
    presence_penalty: float = 0
    frequency_penalty: float = 0


@dataclass
class UserGptContext:  # user gpt context for user and gpt
    user_gpt_profile: UserGptProfile
    gpt_model: GptModel
    user_message_histories: list[MessageHistory] = field(default_factory=list)
    gpt_message_histories: list[MessageHistory] = field(default_factory=list)
    system_message_histories: list[MessageHistory] = field(default_factory=list)
    user_message_tokens: int = 0
    gpt_message_tokens: int = 0
    system_message_tokens: int = 0
    is_user_in_chat: bool = False

    @classmethod
    def make_user_gpt_context_default(
        cls, user_id: str, gpt_model_name: str | None = None
    ):
        return (
            cls(
                user_gpt_profile=UserGptProfile(user_id=user_id),
                gpt_model=getattr(GPT_MODELS, gpt_model_name),
            )
            if gpt_model_name is not None
            else cls(
                user_gpt_profile=UserGptProfile(user_id=user_id),
            )
        )

    async def add_user_message_history_safely(
        self,
        message_history: MessageHistory,
    ) -> None:
        self.user_message_histories.append(message_history)
        self.user_message_tokens += message_history.tokens
        await self.ensure_token_not_exceed()

    async def add_gpt_message_history_safely(
        self,
        message_history: MessageHistory,
    ) -> None:
        self.gpt_message_histories.append(message_history)
        self.gpt_message_tokens += message_history.tokens
        await self.ensure_token_not_exceed()

    async def ensure_token_not_exceed(
        self,
    ) -> None:
        exceeded_tokens: int = (
            self.gpt_message_tokens
            + self.user_message_tokens
            - self.gpt_model.max_total_tokens
            - self.gpt_model.token_margin
        )
        while exceeded_tokens > 0:
            emptied_user_tokens: int = self.user_message_histories.pop(0).tokens
            emptied_gpt_tokens: int = self.gpt_message_histories.pop(0).tokens
            self.user_message_tokens -= emptied_user_tokens
            self.gpt_message_tokens -= emptied_gpt_tokens
            exceeded_tokens -= emptied_user_tokens + emptied_gpt_tokens


async def message_history_organizer(
    user_gpt_context: UserGptContext,
) -> list[dict[str, str]]:  # organize message history for openai api
    message_histories: list[dict[str, str]] = []
    for system_history in user_gpt_context.system_message_histories:
        message_histories.append(
            {"role": system_history.role, "content": system_history.content}
        )  # append system message history
    for user_message_history, gpt_message_history in zip_longest(
        user_gpt_context.user_message_histories,
        user_gpt_context.gpt_message_histories,
    ):
        message_histories.append(
            {
                "role": user_message_history.role,
                "content": user_message_history.content,
            }
        ) if user_message_history is not None else ...  # append user message history
        message_histories.append(
            {
                "role": gpt_message_history.role,
                "content": gpt_message_history.content,
            }
        ) if gpt_message_history is not None else ...  # append gpt message history
    return message_histories  # return message histories to be used in openai api


class ChatGptStreamManager:
    api_url: str = "https://api.openai.com/v1/chat/completions"  # api url for openai
    wait_for_timeout: float = 5.0  # wait for this time before timeout
    wait_for_reconnect: float = 3.0  # wait for this time before reconnecting
    begin_message_with: str = "\n\n"  # begin stream with this
    end_message_with: str = "\n\n\n"  # end stream with this
    substitute_message_with: str = (
        "\n"  # substitute this for characters with special meaning
    )
    user_gpt_contexts: dict[str, UserGptContext] = {}
    api_regex_pattern: re.Pattern = re.compile(r"data:\s*({.+?})\n\n")
    tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")

    @classmethod
    async def stream_from_api(
        cls,
        api_key: str,  # api key for openai
        user_gpt_context: UserGptContext,  # gpt context for user
    ) -> AsyncGenerator:  # async generator for streaming
        async with httpx.AsyncClient(
            timeout=cls.wait_for_timeout
        ) as client:  # initialize client
            while True:  # stream until connection is closed
                gpt_content: str = ""  # initialize gpt_content
                gpt_tokens: int = 0  # initialize gpt_tokens
                try:
                    async with client.stream(
                        method="POST",
                        url=user_gpt_context.gpt_model.api_url,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {api_key}",
                        },  # set headers for openai api request
                        json={
                            "model": user_gpt_context.gpt_model.name,
                            "messages": await message_history_organizer(
                                user_gpt_context=user_gpt_context
                            ),
                            "temperature": user_gpt_context.user_gpt_profile.temperature,
                            "top_p": user_gpt_context.user_gpt_profile.top_p,
                            "n": 1,
                            "stream": True,
                            "presence_penalty": user_gpt_context.user_gpt_profile.presence_penalty,
                            "frequency_penalty": user_gpt_context.user_gpt_profile.frequency_penalty,
                        },  # set json for openai api request
                    ) as streaming_response:
                        if (
                            streaming_response.status_code != 200
                        ):  # if status code is not 200
                            error_msg: str = (
                                json.loads(await streaming_response.aread())
                                .get("error")
                                .get("message")
                            )
                            raise GptConnectionException(
                                msg=f"OpenAI 서버로부터 오류: {error_msg}"
                            )  # raise exception for connection error
                        async for raw_text in streaming_response.aiter_text():  # stream from api
                            if raw_text == "data: [DONE]":  # if stream is done
                                break
                            for text in cls.api_regex_pattern.findall(
                                raw_text
                            ):  # parse json from stream
                                try:
                                    json_data: dict = json.loads(text)["choices"][
                                        0
                                    ]  # data from api
                                except json.JSONDecodeError:  # if json is invalid
                                    continue
                                finish_reason: str | None = json_data.get(
                                    "finish_reason"
                                )  # reason for finishing stream
                                delta: str | None = json_data.get("delta").get(
                                    "content"
                                )  # generated text from api
                                if finish_reason == "length":
                                    raise GptLengthException(
                                        msg="Incomplete model output due to max_tokens parameter or token limit"
                                    )  # raise exception for token limit
                                elif finish_reason == "content_filter":
                                    raise GptContentFilterException(
                                        msg="Omitted content due to a flag from our content filters"
                                    )  # raise exception for openai content filter
                                elif delta is not None:
                                    gpt_tokens += 1
                                    gpt_content += delta
                                    if delta not in (
                                        cls.begin_message_with,
                                        cls.end_message_with,
                                    ):
                                        yield delta
                                    else:
                                        yield cls.substitute_message_with
                except GptException as gpt_exception:
                    user_gpt_context.user_message_tokens -= (
                        user_gpt_context.user_message_histories.pop().tokens
                    )
                    yield gpt_exception.msg
                    break
                except httpx.TimeoutException:
                    await sleep(cls.wait_for_reconnect)
                except Exception as exception:
                    user_gpt_context.user_message_tokens -= (
                        user_gpt_context.user_message_histories.pop().tokens
                    )
                    raise Responses_500.websocket_error(ex=exception)  # raise exception
                else:
                    await user_gpt_context.add_gpt_message_history_safely(
                        message_history=MessageHistory(
                            role=user_gpt_context.user_gpt_profile.gpt_role,
                            content=gpt_content,
                            tokens=gpt_tokens,
                            is_user=False,
                        ),
                    )
                    break

    @classmethod
    async def send_whole_message(
        cls,
        websocket: WebSocket,
        message: str,
    ) -> None:  # send whole message to websocket
        for msg in (cls.begin_message_with, message, cls.end_message_with):
            await websocket.send_text(msg)  # send whole message to websocket

    @classmethod
    async def send_whole_stream(
        cls,
        websocket: WebSocket,
        stream: AsyncGenerator,
    ) -> None:  # send whole stream to websocket
        await websocket.send_text(cls.begin_message_with)  # begin message
        async for one_token_text in stream:
            await websocket.send_text(one_token_text)  # send message to websocket
        await websocket.send_text(cls.end_message_with)  # end message

    @classmethod
    async def send_websocket(
        cls,
        websocket: WebSocket,
        user_gpt_context: UserGptContext,
        api_key: str,
    ) -> None:  # websocket for chat gpt
        while True:  # loop until connection is closed
            try:
                user_gpt_context.is_user_in_chat = True  # set user in chat
                user_chat_gpt_message: UserChatGptMessage = (
                    UserChatGptMessage.parse_raw(await websocket.receive_text())
                )  # receive user message from websocket
                user_message: str = user_chat_gpt_message.user_message  # user message
                if user_message.startswith("/"):  # if user message is command
                    user_command: list = user_message.split()
                    callback_name: str = user_command[0][1:]  # command name
                    callback_args: list = user_command[1:]  # command args
                    callback_response: str = (
                        getattr(ChatGptCommands, callback_name)(
                            *callback_args, user_gpt_context=user_gpt_context
                        )
                        if hasattr(ChatGptCommands, callback_name)
                        else ChatGptCommands.not_existing_callback(
                            user_gpt_context=user_gpt_context
                        )
                    )  # get callback response
                    await cls.send_whole_message(
                        websocket=websocket, message=callback_response
                    )  # send callback response to websocket
                    continue
                user_message_tokens: int = len(
                    cls.tokenizer(user_message)["input_ids"]
                )  # user message tokens
                max_tokens_per_request: int = (
                    user_gpt_context.gpt_model.max_tokens_per_request
                )  # max tokens per request
                if (
                    user_message_tokens > max_tokens_per_request
                ):  # if user message is too long
                    raise GptTooMuchPromptTokenException(
                        msg=f"너무 메시지가 길어요. 현재 토큰 개수는 {user_message_tokens}로, {max_tokens_per_request} 이하여야 합니다."
                    )  # raise exception for too much prompt token
                await user_gpt_context.add_user_message_history_safely(
                    message_history=MessageHistory(
                        role=user_gpt_context.user_gpt_profile.user_role,
                        content=user_message,
                        tokens=user_message_tokens,
                        is_user=True,
                    ),
                )  # add user message history safely
                await cls.send_whole_stream(
                    websocket=websocket,
                    stream=cls.stream_from_api(
                        user_gpt_context=user_gpt_context,
                        api_key=api_key,
                    ),
                )  # send whole stream to websocket
            except GptTooMuchPromptTokenException as too_much_token_exception:  # if user message is too long
                await cls.send_whole_message(
                    websocket=websocket, message=too_much_token_exception.msg
                )  # send too much token exception message to websocket
            except Exception as exception:  # if other exception is raised
                raise Responses_500.websocket_error(ex=exception)  # raise exception
            finally:
                user_gpt_context.is_user_in_chat = False  # set user not in chat


class ChatGptCommands:  # commands for chat gpt
    @staticmethod
    def not_existing_callback(
        *args, user_gpt_context: UserGptContext
    ) -> str:  # callback for not existing command
        return f"{user_gpt_context.user_gpt_profile.user_id}님, 죄송합니다. 현재 그런 명령어는 지원하지 않습니다."

    @staticmethod
    def clear(
        *args, user_gpt_context: UserGptContext
    ) -> str:  # clear user and gpt message histories
        user_id: str = user_gpt_context.user_gpt_profile.user_id
        n_user: int = len(user_gpt_context.user_message_histories)
        n_gpt: int = len(user_gpt_context.gpt_message_histories)
        response: str = f"좋아요. {user_id}님의 {n_user}개의 메시지와 저의 {n_gpt}개의 메시지를 삭제할게요."
        user_gpt_context.user_message_histories.clear()  # clear user message histories
        user_gpt_context.gpt_message_histories.clear()  # clear gpt message histories
        user_gpt_context.user_message_tokens = 0  # reset user message tokens
        user_gpt_context.gpt_message_tokens = 0  # reset gpt message tokens
        return response  # return success message

    @staticmethod
    def test(
        *args, user_gpt_context: UserGptContext
    ) -> str:  # test command showing user_gpt_context
        return f"args: {args} | user_gpt_context: {user_gpt_context}"

    @staticmethod
    def reset(*args, user_gpt_context: UserGptContext) -> str:  # reset user_gpt_context
        user_id: str = user_gpt_context.user_gpt_profile.user_id
        if (
            user_id in ChatGptStreamManager.user_gpt_contexts.keys()
        ):  # if user_id exists in user_gpt_contexts
            default_context: UserGptContext = (
                UserGptContext.make_user_gpt_context_default(user_id=user_id)
            )  # make default context
            for (
                key
            ) in (
                default_context.__annotations__.keys()
            ):  # reset user_gpt_context to default
                setattr(
                    ChatGptStreamManager.user_gpt_contexts[user_id],
                    key,
                    getattr(default_context, key),
                )
            return "정상적으로 컨텍스트가 리셋되었습니다."  # return success message
        else:
            return "컨텍스트를 리셋하지 못했습니다."  # return fail message

    @staticmethod
    def system(*args, user_gpt_context: UserGptContext) -> str:  # add system message
        if len(args) < 1:  # if no args
            return "/system SYSTEM_MESSAGE와 같은 형식으로 입력해야 합니다."  # return fail message
        system_message: str = " ".join(args)
        system_message_tokens: int = len(
            ChatGptStreamManager.tokenizer(system_message)["input_ids"]
        )  # get system message tokens
        if (
            system_message_tokens
            + user_gpt_context.system_message_tokens
            - user_gpt_context.gpt_model.token_margin
            > user_gpt_context.gpt_model.max_total_tokens
        ):  # if system message tokens are too much to add
            return "시스템 메시지가 너무 많아서 추가할 수 없습니다."  # return fail message
        user_gpt_context.system_message_tokens += (
            system_message_tokens  # add system message tokens
        )
        user_gpt_context.system_message_histories.append(
            MessageHistory(
                role="system", content=system_message, tokens=None, is_user=False
            )
        )  # add system message history
        return f"시스템 메시지를 `{system_message}`로 추가하였습니다!"  # return success message

    @staticmethod
    def settemperature(
        *args, user_gpt_context: UserGptContext
    ) -> str:  # set temperature of gpt
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
            return f"temperature 값을 {previous_temperature}에서 {now_temperature}로 바꿨어요."  # return success message

    @classmethod
    def settemp(
        cls, *args, user_gpt_context: UserGptContext
    ) -> str:  # alias for settemperature
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
            return (
                "user, system, gpt 중 하나를 입력해야 합니다." if not is_silent else ""
            )  # return fail message
        # find attributes of histories and tokens of user, system, gpt, using getattr and setattr
        message_histories: list[MessageHistory] = getattr(
            user_gpt_context, f"{args[0]}_message_histories"
        )
        message_tokens: int = getattr(user_gpt_context, f"{args[0]}_message_tokens")
        if len(message_histories) < 1:  # if no message histories
            return (
                f"{args[0]} 메시지가 없어서 삭제할 수 없습니다." if not is_silent else ""
            )  # return fail message
        last_message_history: MessageHistory = (
            message_histories.pop()
        )  # pop last message history
        setattr(
            user_gpt_context,
            f"{args[0]}_message_tokens",
            message_tokens - last_message_history.tokens,
        )  # pop last message tokens
        return (
            f"{args[0]} 메시지를 `{last_message_history.content}`로 삭제하였습니다!"
            if not is_silent
            else ""
        )  # return success message

    @staticmethod
    def retry(*args, user_gpt_context: UserGptContext) -> str:
        # retry last gpt message, format: /retry
        if (
            len(user_gpt_context.user_message_histories) < 1
            or len(user_gpt_context.gpt_message_histories) < 1
        ):
            return "메시지가 없어서 다시 할 수 없습니다."
        # pop the lastest user and gpt message histories and deduct tokens
        user_gpt_context.user_message_tokens -= (
            user_gpt_context.user_message_histories.pop().tokens
        )
        user_gpt_context.gpt_message_tokens -= (
            user_gpt_context.gpt_message_histories.pop().tokens
        )
        return "다시 말해 볼게요!"  # return success message
