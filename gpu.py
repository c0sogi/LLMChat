from app.models.gpt_llms import LLMModels, LlamaCppModel
from app.models.gpt_models import UserGptContext  # noqa: F811
from app.dependencies import process_manager
from app.utils.chatgpt.chatgpt_llama_cpp import get_llama, llama_cpp_generation

print("Running in:", __name__, "\n\n\n")

if __name__ == "__main__":
    m_queue = process_manager.Queue()
    m_done = process_manager.Event()
    llama_cpp_model: LlamaCppModel = LLMModels.vicuna_uncensored.value
    llm = get_llama(llama_cpp_model)
    llm.echo = True
    llama_cpp_generation(
        llama_cpp_model=llama_cpp_model,
        prompt="Hello, how are you?",
        m_queue=m_queue,
        m_done=m_done,
        user_gpt_context=UserGptContext.construct_default(user_id="test_user_id", chat_room_id="test_chat_room_id"),
    )
