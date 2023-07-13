from concurrent.futures import ProcessPoolExecutor

from app.common.app_settings_llama_cpp import (
    shutdown_llama_cpp_server,
    start_llama_cpp_server,
)
from app.common.config import config
from app.models.chat_models import command_response
from app.shared import Shared


class ServerCommands:
    @staticmethod
    @command_response.send_message_and_stop
    async def free() -> str:
        """Free the process pool executor\n
        /free"""
        shared = Shared()
        shared.process_pool_executor.shutdown(wait=True)
        shared.process_pool_executor = ProcessPoolExecutor()
        return "Process pool executor freed!"

    @staticmethod
    @command_response.send_message_and_stop
    async def shutdownllamacpp() -> str:
        """Shutdown the llama cpp server\n
        /shutdownllamacpp"""
        shutdown_llama_cpp_server(shared=Shared())
        return "Shutdown llama cpp server!"

    @staticmethod
    @command_response.send_message_and_stop
    async def startllamacpp() -> str:
        """Start the llama cpp server\n
        /startllamacpp"""
        start_llama_cpp_server(shared=Shared(), config=config)
        return "Started llama cpp server!"
