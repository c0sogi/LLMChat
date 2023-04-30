from typing import Any, Dict, List, Union

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

from app.utils.logger import api_logger


class StreamingCallbackHandler(StreamingStdOutCallbackHandler):
    """Callback handler for streaming. Only works with LLMs that support streaming."""

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Run when LLM starts running."""
        api_logger.critical(f"LLM started running: {serialized} {prompts}")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        api_logger.critical(f"LLM generated new token: {token}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        api_logger.critical(f"LLM ended running: {response}")

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Run when LLM errors."""
        api_logger.critical(f"LLM errored: {error}")

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain starts running."""
        api_logger.critical(f"Chain started running: {serialized} {inputs}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""
        api_logger.critical(f"Chain ended running: {outputs}")

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Run when chain errors."""
        api_logger.critical(f"Chain errored: {error}")

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """Run when tool starts running."""
        api_logger.critical(f"Tool started running: {serialized} {input_str}")

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        api_logger.critical(f"Agent action: {action}")

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""
        api_logger.critical(f"Tool ended running: {output}")

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Run when tool errors."""
        api_logger.critical(f"Tool errored: {error}")

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""
        api_logger.critical(f"Text: {text}")

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""
        api_logger.critical(f"Agent finish: {finish}")
